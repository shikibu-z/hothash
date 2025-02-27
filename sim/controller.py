import numpy as np
from math import ceil
from dataclasses import dataclass
from typing import Any, Dict, List
from abc import ABC, abstractmethod

from util import without
from policy import AbcPolicy
from workload import AbcWorkload
from entities import Node, Task, PerformaceDeltas, ConsistentHashing


@dataclass
class SimControllerConfig:
    compute_node_count: int = 20
    virtual_node_count: int = 1
    compute_cpu_per_node: int = 8

    chunk_size: int = 32
    compute_mem_per_node: int = 1024 * 4
    compute_mem_cache_per_node: int = 1024 * 4
    compute_disk_per_node: int = 1024 * 4

    capacity_per_cpu: float = 768
    storage_bandwidth: float = 32
    storage_latency: float = 0
    local_bandwidth: float = 0
    local_latency: float = 0
    granularity: float = 1000

    log_interval = 1


class AbcPerformanceModel(ABC):
    @abstractmethod
    def sample_deltas(
        self,
        timestep: int,
        nodes: List[Node],
        idle_tasks: List[Task],
        running_tasks: List[Task],
    ) -> List[PerformaceDeltas]:
        pass


class DefaultPerformanceModel(AbcPerformanceModel):
    """
    Default performance model simply adds some noise to bandwidth and latency as
    well as processing_capacity for each node. Magnitude of noise is constant
    over time.
    """

    def __init__(self, sim_config: SimControllerConfig):
        self.sim_config = sim_config

    def sample_deltas(self, timestep, nodes, idle_tasks, running_tasks):
        deltas = PerformaceDeltas(
            storage_bandwidth=np.random.normal(0, 0),
            storage_latency=0,
            processing_capacity=0,
        )

        return [deltas] * len(nodes)


class SimController:
    """
    Manages the progression of the simulation. Simulation proceeds through a
    sequence of discrete timesteps. The granularity (ms/step) can be specified
    in the config, which enables more fidelity in the modeling of latency.
    """

    def __init__(
        self,
        config: SimControllerConfig,
        workload: AbcWorkload,
        policy: AbcPolicy,
        performance_model: AbcPerformanceModel,
    ):
        self.config: SimControllerConfig = config
        self.workload: AbcWorkload = workload
        self.policy: AbcPolicy = policy
        self.performance_model: AbcPerformanceModel = performance_model
        self.cluster_logs: List[Any] = []
        self.node_logs: List[Any] = []
        self.task_logs = []

        self.nodes = []
        for i in range(config.compute_node_count):
            node = Node(
                i,
                storage_bandwidth=self.config.storage_bandwidth,
                storage_latency=self.config.storage_latency,
                local_bandwidth=self.config.local_bandwidth,
                local_latency=self.config.local_latency,
                memory=self.config.compute_mem_per_node,
                cache_memory=self.config.compute_mem_cache_per_node,
                chunk_size=self.config.chunk_size,
                cpus=self.config.compute_cpu_per_node,
                disk=self.config.compute_disk_per_node,
                processing_capacity=self.config.capacity_per_cpu,
                step_interval=self.config.granularity,
            )
            self.nodes.append(node)

        self.active_queries = []
        self.done_queries = []
        self.timestep = 0
        self.init_task_queue = []
        self.running_task_queue = []
        self.done_task_queue = []

        self.consistent_hash_ring = ConsistentHashing(
            self.nodes, self.config.virtual_node_count
        )

    def get_query_times(self):
        return [
            (q.finish_time - q.start_time) * self.config.granularity
            for q in self.done_queries
        ]

    def get_cache_hit(self):
        return [node.cache_hit_logs for node in self.nodes]

    def get_task_hist(self):
        return [len(node.task_history) for node in self.nodes]

    def step(self):
        # generate incoming workload
        new_queries = self.workload.get_work(self.timestep)

        # create tasks from ops
        for query in new_queries:
            query.set_start(self.timestep)
            for op in query.ops:
                # TODO should probably be multiple tasks for op based on fragments
                tasks = op.get_tasks()
                self.init_task_queue += tasks
                # experiment log
                for task in tasks:
                    self.task_logs.append(task.fragments[0].id)

        # register new queries
        self.active_queries += new_queries

        # apply policy to determine task assignments
        assignments = self.policy.manage_tasks(
            self.timestep,
            self.nodes,
            self.init_task_queue,
            self.running_task_queue,
            self.consistent_hash_ring,
        )

        # process assignments
        assigned = []
        for task, node in assignments.items():
            node.assign_task(task)
            task.node = node
            assigned.append(self.init_task_queue.index(task))
            self.running_task_queue.append(task)
        self.init_task_queue = without(self.init_task_queue, assigned)

        # handle caching
        cache_policy = self.policy.manage_cache(
            self.timestep, self.nodes, self.init_task_queue, self.running_task_queue
        )

        # simulate performance randomness
        perf_deltas = self.performance_model.sample_deltas(
            self.timestep, self.nodes, self.init_task_queue, self.running_task_queue
        )

        # simulate step
        for i, node in enumerate(self.nodes):
            node_deltas = perf_deltas[i]
            node.step(self.timestep, node_deltas, cache_policy=cache_policy)

        done_idx = []
        for i, task in enumerate(self.running_task_queue):
            if task.is_complete:
                self.done_task_queue.append(task)
                done_idx.append(i)
        self.running_task_queue = without(self.running_task_queue, done_idx)

        # remove processed results
        for node in self.nodes:
            remove_idx = []
            for i, (task, chunk) in enumerate(node.processed_queue):
                if task.is_complete:
                    remove_idx.append(i)
                    # don't change the same chunk twice
                    if chunk.availability(node) != ["mem"]:
                        continue

                    should_dump = True
                    for item in node.loaded_queue:
                        if chunk.id == item[1].id:
                            should_dump = False
                            break

                    if should_dump:
                        eviction = node.cache.put(chunk.id, chunk)
                        chunk.clear(node, "mem")
                        chunk.cache(node, "mem_cache")

                        # if cache eviction
                        if eviction != None:
                            eviction[1].clear(node, "mem")
                            eviction[1].clear(node, "mem_cache")
                            fragment_id = str(eviction[1].fragment.id)
                            node.signature[fragment_id] -= 1

                            if node.signature[fragment_id] < 1:
                                node.signature.pop(fragment_id)

                        node.resident_chunks.discard((chunk, "mem"))

            node.processed_queue = without(node.processed_queue, remove_idx)

        # Cleanup completed tasks
        for node in self.nodes:
            node_done_idx = []

            for i, task in enumerate(node.tasks):
                if task.is_complete:
                    node_done_idx.append(i)

            node.tasks = without(node.tasks, node_done_idx)

        # complete queries
        done_idx = []
        for i, query in enumerate(self.active_queries):
            if all([op.status == "complete" for op in query.ops]):
                self.done_queries.append(query)
                done_idx.append(i)
                query.set_finish(self.timestep)

        self.active_queries = without(self.active_queries, done_idx)

        # log state of sim
        if self.timestep % self.config.log_interval == 0:
            self.cluster_logs.append(
                {
                    "timestep": self.timestep,
                    "ms": self.timestep * self.config.granularity,
                    "sec": self.timestep * self.config.granularity / 1000,
                    "idle_tasks": len(self.init_task_queue),
                    "in_flight_tasks": len(self.running_task_queue),
                    "completed_tasks": len(self.done_task_queue),
                    "in_flight_queries": len(self.active_queries),
                    "completed_queries": len(self.done_queries),
                }
            )

            for j, node in enumerate(self.nodes):
                self.node_logs.append(
                    {
                        "node": node.id,
                        "timestep": self.timestep,
                        "ms": self.timestep * self.config.granularity,
                        "sec": self.timestep * self.config.granularity / 1000,
                        "request_queue_len": len(node.requested_queue),
                        "loaded_queue_len": len(node.loaded_queue),
                        "processed_queue_len": len(node.processed_queue),
                        "storage_bandwidth_usage": (
                            node.storage_bandwidth - perf_deltas[j].storage_bandwidth
                        )
                        - node.resources["storage_bandwidth"],
                        "local_bandwidth_usage": (
                            node.local_bandwidth - perf_deltas[j].local_bandwidth
                        )
                        - node.resources["local_bandwidth"],
                        "mem_usage": node.memory - node.resources["storage"]["mem"],
                        "mem_cache_usage": node.cache_memory
                        - node.resources["storage"]["mem_cache"],
                        "cpu_usage": (
                            node.processing_capacity
                            - perf_deltas[j].processing_capacity
                            - node.resources["processing_capacity"]
                        ),
                        "assigned_tasks": len(node.tasks),
                    }
                )

        self.timestep = self.timestep + 1
