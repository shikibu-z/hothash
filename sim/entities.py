import mmh3
import random
import numpy as np
from bisect import bisect
from math import floor, ceil
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

from util import with_idx, consolidate, without
from typing import Dict, List, Tuple, Optional, Callable, Union, Literal


class Task:
    """
    Tasks are scheduled to compute a portion of a query operation over one or
    more data fragments on a particular node

    A scan for example will have a single fragment referencing
    a column in remote storage.
    """

    def __init__(self, query_op, fragments, is_shuffle=False):
        self.fragments = fragments
        self.query_op: QueryOp = query_op
        self.progress: Dict[Fragment, int] = {fragment: 0 for fragment in fragments}
        self.totals: Dict[Fragment, int] = {
            fragment: len(fragment.chunks) for fragment in fragments
        }
        self.status: Literal["pending", "complete"] = "pending"
        self.is_shuffle: bool = is_shuffle
        if self.query_op.parent:
            output_fragments = self.query_op.parent.fragments
            self.output_maps = self.get_output_maps(self.fragments, output_fragments)

    def get_output_maps(self, fragments, output_fragments):
        mappings = {}

        for i, fragment in enumerate(fragments):
            out_fragment = output_fragments[i]
            mapping = consolidate(
                list(range(len(out_fragment.chunks))), len(fragment.chunks)
            )
            mappings[fragment] = mapping

        return mappings

    def get_next_output_chunks(self, chunk):
        if self.query_op.parent:
            output_fragments = self.query_op.parent.fragments
            fragment_idx = self.fragments.index(chunk.fragment)
            chunk_idx = chunk.fragment.chunks.index(chunk)
            out_idx = self.output_maps[chunk.fragment][chunk_idx]

            return with_idx(output_fragments[fragment_idx].chunks, out_idx)

        else:
            return []

    def inc_progress(self, fragment) -> None:
        """
        When a chunk has been processed on a node we increment the progress of
        this that chunk.
        """
        self.progress[fragment] += 1
        if sum(self.progress.values()) == sum(self.totals.values()):
            self.status = "complete"
            self.query_op.inc_progress(fragment)

    def reset_progress(self, fragment) -> None:
        """
        When a node fails we want to clear any intermediate progress that has been
        made that was lost when the failure occurred.
        """
        self.progress[fragment] = 0

    @property
    def is_complete(self):
        return self.status == "complete"

    @property
    def is_ready(self):
        """
        Determine if this task is ready to be scheduled.
        """
        if not self.query_op.dependencies:
            return True

        elif not self.is_shuffle:
            return True

        elif self.is_shuffle and all(
            [d.is_complete for d in self.query_op.dependencies]
        ):
            return True

        else:
            return False


@dataclass
class PerformaceDeltas:
    storage_latency: float = 0.0
    storage_bandwidth: float = 0.0
    local_latency: float = 0.0
    local_bandwidth: float = 0.0
    processing_capacity: float = 0.0


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        # capacity is the number of chunks
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return "ERR_CHUNK_ID"
        else:
            return self.cache.pop(key)

    def put(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)

        if len(self.cache) > self.capacity:
            return self.cache.popitem(last=False)

        return None


class TaskHist:
    """
    Track task history and frequency within a time window.
    """

    def __init__(self, time_window):
        # time_window: the window within which we track, int
        self.window = time_window
        self.history = defaultdict(list)

    def get_frequency(self, task, timestep):
        # task: the task to query, Task
        # timestep: the current timestep, int
        fid = str(task.fragments[0].id)
        self.history[fid].append(timestep)

        count = len(self.history[fid])
        total = sum(len(self.history[key]) for key in self.history.keys())

        return count / total

    def update(self, timestep):
        # timestep: the current timestep, int
        for key in list(self.history.keys()):
            remove = []
            for i, hist_time in enumerate(self.history[key]):
                if timestep - hist_time > self.window:
                    remove.append(i)
            self.history[key] = without(self.history[key], remove)

            if len(self.history[key]) == 0:
                self.history.pop(key)


class Node:
    def __init__(
        self,
        nid,
        storage_bandwidth=10.0,
        storage_latency=10.0,
        local_bandwidth=10.0,
        local_latency=10.0,
        memory=32,
        cache_memory=16,
        chunk_size=10,
        disk=512,
        processing_capacity=10.0,
        cpus=8,
        step_interval=10,
    ):
        self.id = nid
        self.tasks: List[Task] = []
        self.task_history = []
        self.resident_chunks = set()

        # cache_memory should devide by chunk size
        self.cache = LRUCache(ceil(cache_memory / chunk_size))
        self.signature = {}

        self.init_queue: List[Tuple[Task, Chunk]] = []
        self.requested_queue: List[Tuple[Task, Chunk, str]] = []
        self.loaded_queue: List[Tuple[Task, Chunk]] = []
        self.processed_queue: List[Tuple[Task, Chunk]] = []

        self.storage_bandwidth = storage_bandwidth
        self.storage_latency = storage_latency
        self.local_bandwidth = local_bandwidth
        self.local_latency = local_latency

        self.memory = memory
        self.cache_memory = cache_memory
        self.chunk_size = chunk_size

        self.disk = disk
        self.cpus = cpus
        self.processing_capacity = 1024 * 5
        self.step_interval = step_interval

        self.resources = {}
        self.cache_hit_logs = []

    def _reset_step_resources(self, deltas: PerformaceDeltas):
        self.resources["storage_bandwidth"] = max(
            self.storage_bandwidth - deltas.storage_bandwidth, 0
        )

        self.resources["local_bandwidth"] = max(
            self.local_bandwidth - deltas.local_bandwidth, 0
        )

        self.resources["storage_latency"] = max(
            self.storage_latency - deltas.storage_latency, 0
        )

        self.resources["local_latency"] = max(
            self.local_latency - deltas.local_latency, 0
        )

        self.resources["processing_capacity"] = max(
            self.processing_capacity - deltas.processing_capacity, 0
        )

        self.resources["storage"] = {
            "mem": self.memory - self._calc_mem_usage("mem"),
            "mem_cache": self.cache.capacity * self.chunk_size
            - self._calc_cache_usage(),
        }

    def assign_task(self, task):
        """
        Assign a task to be processed on this node. Task will be placed in node
        task_queue. Chunks for that task are added to the node processing pipeline,
        init_queue.
        """
        self.tasks.append(task)
        self.task_history.append(task)

        for fragment in task.fragments:
            for chunk in fragment.chunks:
                self.init_queue.append((task, chunk))

    def check_bandwidth_limits(self, size: float, loc: str) -> bool:
        available_bandwdith = (
            self.resources["storage_bandwidth"]
            if loc == "remote"
            else self.resources["local_bandwidth"]
        )

        return available_bandwdith >= size

    def check_storage_limits(self, size, target) -> bool:
        return self.resources["storage"][target] >= size

    def check_processing_limits(self, size) -> bool:
        return self.resources["processing_capacity"] >= size

    def dec_bandwidth(self, size, loc):
        if loc == "remote":
            self.resources["storage_bandwidth"] -= size

        elif loc == "local":
            self.resources["local_bandwidth"] -= size

    def dec_storage(self, size, target):
        if target == "mem":
            self.resources["storage"]["mem"] -= size

        elif target == "mem_cache":
            self.resources["storage"]["mem_cache"] -= size

    def dec_processing(self, size):
        self.resources["processing_capacity"] -= size

    def get_mem_usage(self, target):
        return self.resources["storage"][target]

    def _calc_mem_usage(self, target):
        return sum(
            (
                chunk.size
                for chunk, chunk_loc in self.resident_chunks
                if (chunk_loc == target)
            )
        )

    def _calc_cache_usage(self):
        return sum(value.size for key, value in self.cache.cache.items())

    def get_mem_available(self, target):
        return self.resources["storage"][target]

    def _drain_init_queue(self, timestep):
        """
        For each chunk in init_queue initiate request to load data. Think of
        this as Making a request to stream a fragment from some remote storage.
        """
        # request any new fragments
        stalled_chunks = []
        for task, chunk in self.init_queue:
            chunk.request_remote(self, timestep)
            self.requested_queue.append((task, chunk, "remote"))

        # We should have initiated a request for all chunks so we clear the init_queue
        self.init_queue = stalled_chunks

    def _drain_requested_queue(
        self, timestep: int, cache_policy: Optional["ICachePolicy"] = None
    ):
        """
        For each chunk in request_queue which has completed latency wait, load
        to this node if there is sufficient bandwidth and storage remaining on
        this timestep.
        """
        loaded = []

        # loaded queue should not exceed cache size, this is assured by the
        # check_storage_limits and early break in the loop
        for i, (task, chunk, loc) in enumerate(self.requested_queue):
            size = chunk.size
            cid = str(chunk.id)
            chunk_loc = chunk.availability(self)

            if len(chunk_loc) > 0:
                # if chunk is in cache
                if "mem_cache" in chunk_loc and "mem" not in chunk_loc:
                    if self.check_storage_limits(chunk.size, "mem"):
                        # read chunk from cache to mem
                        chunk_read = self.cache.get(cid)

                        # debug
                        if chunk_read == "ERR_CHUNK_ID":
                            print("[ERROR] Invalid chunk read!")
                            exit()
                        elif chunk_read.id != cid:
                            print("[ERROR] Chunk id doesn't match!")
                            exit()

                        # log chunk read from cache
                        chunk.clear(self, "mem_cache")
                        chunk.cache(self, "mem")

                        self.loaded_queue.append((task, chunk))
                        loaded.append(i)

                        self.resident_chunks.add((chunk, "mem"))
                        self.dec_storage(size, "mem")

                    else:
                        # we shouldn't read more mem can hold
                        break

                # if chunk is already in mem
                else:
                    self.loaded_queue.append((task, chunk))
                    loaded.append(i)

                self.cache_hit_logs.append(True)

            # chunk not in cache and mem, read remote storage
            else:
                storage_target = "mem"

                if (
                    # wait_is_complete
                    self.check_bandwidth_limits(size, loc)
                    and self.check_storage_limits(size, storage_target)
                ):
                    chunk.load(self, storage_target)
                    self.loaded_queue.append((task, chunk))
                    self.resident_chunks.add((chunk, storage_target))
                    loaded.append(i)

                    self.dec_bandwidth(size, loc)
                    self.dec_storage(size, storage_target)

                    # we will increase node chunk signature when read remote
                    # storage, i.e. new data to node
                    fid = str(chunk.fragment.id)
                    if fid in self.signature:
                        self.signature[fid] += 1
                    else:
                        self.signature[fid] = 1

                    self.cache_hit_logs.append(False)

                # stop for resource restriction or latency
                else:
                    break

        # remove any loaded chunks from the request queue
        self.requested_queue = without(self.requested_queue, loaded)

    def _drain_processing_queue(self):
        """
        For each chunk which has been loaded attempt to process. Once completed
        any processed chunks will have been removed from the loaded queue and
        added to the processed queue.
        """
        # process loaded chunks until cycles is exhausted
        processed = []

        for i, (task, chunk) in enumerate(self.loaded_queue):
            # we are able to process
            if self.check_processing_limits(chunk.size):
                # debug
                if chunk.process(self, task) == False:
                    print("[ERROR] Chunk is not in memory but processed!")
                    exit()

                self.processed_queue.append((task, chunk))
                processed.append(i)
                self.dec_processing(chunk.size)

            # early stop this step if processing limit is met
            else:
                break

        self.loaded_queue = without(self.loaded_queue, processed)

    def step(self, timestep, deltas, cache_policy: Optional["ICachePolicy"] = None):
        """
        Handle a single env step for this node. Three main phases:
            - requests new chunks
            - loads requested chunks
            - processes loaded chunks.
        """
        self._reset_step_resources(deltas)

        self._drain_init_queue(timestep)

        self._drain_requested_queue(timestep, cache_policy)

        self._drain_processing_queue()


class QueryOp:
    """
    Represents a single operation within a query plan. For example a table scan
    or a join of intermediate results. An op has dependencies which must be
    processed first. If is_shuffle=True, all dependencies must be complete
    before this op can begin.
    """

    def __init__(
        self,
        op_type,
        fragments,
        parent: Optional["QueryOp"] = None,
        deps: Optional[List["QueryOp"]] = None,
        is_shuffle=False,
    ):
        self.op_type = op_type
        self.fragments: List[Fragment] = fragments
        self.progress = 0
        self.parent: Optional[QueryOp] = parent
        self.dependencies: Union[List[QueryOp], None] = deps
        self.is_shuffle: bool = is_shuffle
        self.status: Literal["pending", "complete"] = "pending"
        self.tasks = []

    @property
    def is_complete(self):
        return self.status == "complete"

    def set_parent(self, parent: "QueryOp"):
        self.parent = parent

    def get_tasks(self):
        if self.op_type == "scan":
            self.tasks = [Task(self, [fragment]) for fragment in self.fragments]

        if self.op_type == "filter":
            self.tasks = [Task(self, [fragment]) for fragment in self.fragments]

        return self.tasks

    def inc_progress(self, fragment):
        self.progress += 1
        if self.progress == len(self.fragments):
            self.status = "complete"


class QueryPlan:
    def __init__(self, ops, output_op):
        self.ops: List[QueryOp] = ops
        self.output_op: QueryOp = output_op

    def get_leaves(self) -> List[QueryOp]:
        """
        Returns the operations a the leaves of the DAG where query processing
        can begin.
        """
        return [op for op in self.ops if op.dependencies == None]

    def set_start(self, timestep):
        self.start_time = timestep

    def set_finish(self, timestep):
        self.finish_time = timestep


class Chunk:
    """
    Represents the smallest unit of simulated data, think either a single
    column single record or a single colum small collection of records.

    All chunks exist in remote storage with associated bandwidth and latency
    peanalties.

    Chunks can be loaded to nodes with an associated bandwidth and latency
    cost. Chunks know the nodes they currently reside on, and what type of
    storage memory, disk, disk_cache, or mem_cache.

    Chunks can only be processed if they are in "mem" or "mem_cache" on a local
    node. Once processed if in "mem" they are automatically removed.

    Chunks can be in use by multiple tasks concurrently.

    To be loaded from remote to a local storage chunks must have been requested
    """

    def __init__(self, id, size, fragment: "Fragment", intermediate=False):
        """
        :param size: Size of chink in KB
        :param fragment: Reference to parent fragment of this chunk
        """
        self.intermediate = intermediate
        self.fragment = fragment

        self.id = str(self.fragment.id) + "_" + str(id)

        self.size = size
        self.cache_request: Dict["Node", int] = {}
        self.remote_request: Dict["Node", int] = {}
        self.local_request: Dict["Node", int] = {}

        self.residence: "Dict[str, set[Node]]" = {
            "disk": set(),
            "mem": set(),
            "disk_cache": set(),
            "mem_cache": set(),
        }

    def is_local(self) -> bool:
        return len(self.residence["disk"]) > 0 or len(self.residence["mem"]) > 0

    def is_cached(self) -> bool:
        return (
            len(self.residence["disk_cache"]) > 0
            or len(self.residence["mem_cache"]) > 0
        )

    def request_cached(self, node, timestep) -> None:
        """
        Register request for data to be loaded from the cache to the specified node
        at specified timestep
        """
        if self.is_cached():
            self.cache_request[node] = timestep

        else:
            raise Exception("requesting cached chunk, but chunk is not in cache")

    def request_remote(self, node, timestep) -> None:
        """
        Register request for data to be loaded from remote storage to the
        specified node at specified timestep
        """
        self.remote_request[node] = timestep

    def request_local(self, node, timestep) -> None:
        """
        Register request for data to be loaded from remote storage to the
        specified node at specified timestep
        """
        self.local_request[node] = timestep

    def request_complete(
        self, node, current_time, remote_latency, local_latency
    ) -> bool:
        return (
            (
                node in self.cache_request
                and self.cache_request[node] < current_time - local_latency
            )
            or (
                node in self.remote_request
                and self.remote_request[node] < current_time - remote_latency
            )
            or (
                node in self.local_request
                and self.local_request[node] < current_time - local_latency
            )
        )

    def load(self, node, level):
        self.residence[level].add(node)

    def cache(self, node, level):
        self.residence[level].add(node)

    def clear(self, node, level):
        if node in self.residence[level]:
            self.residence[level].remove(node)

    def availability(self, node):
        """
        Get a list of storage media where this chunk resides on the specified
        node.
        """
        return [k for k, v in self.residence.items() if node in v]

    def process(self, node, task) -> bool:
        """
        Process a chunk related to a specified task on the specified node.
        Chunk must already be resident in either memory, or mem_cache on
        specified node.
        """
        if node in self.residence["mem"]:
            task.inc_progress(self.fragment)
            return True

        else:
            return False


class Fragment:
    """
    A fragment is a contiguous portion of column data related to a particular
    task. Fragments maintain a list of chunks. Chunks are created on fragment
    initilization based on fragment size
    """

    chunks: List[Chunk]

    def __init__(self, id, size, chunk_size, intermediate=False):
        """
        :param size: Total size of this fragment in KB
        :param chunk_size: Size of each fragment chunk in KB
        """
        self.id: int = id
        self.intermediate = intermediate
        self.chunks: List[Chunk] = [
            Chunk(i, chunk_size, self, intermediate)
            for i in range(ceil(size / chunk_size))
        ]

    @property
    def size(self):
        return sum((c.size for c in self.chunks))


class Column:
    """
    A column is composed of a sequence of fragments which may not all be the
    same size.
    """

    fragments: List[Fragment]

    def __init__(self, name: str, fragments: List[Fragment]):
        """
        :param name: String identifier for column
        :param fragments: List of fragments
        """
        self.name: str = name
        self.fragments: List[Fragment] = fragments


class Table:
    """
    A table is composed of a sequence of columns each column can have different
    number of fragments, and a different size. Because we do not model at the
    record level there is no need to match record count between columns.
    """

    columns: List[Column]

    def __init__(self, name, columns: List[Column]):
        """
        :param name: String identifier for table
        :param columns: List of columns
        """
        self.name = name
        self.columns = columns


class ConsistentHashing:
    """
    The class contains implementations of consistent hashing, and our experiment
    algorithms.
    assign_task_ch: basic consistent hashing with virtual nodes
    assign_task_chbl: basic consistent hashing with bounded loads
    assign_task_rjch: the random jump consistent hashing
    assign_task_chrlu: the consistent hashing with random load updates
    assign_task_tjch: our solution, the traceable jump consistent hashing
    """

    def __init__(self, nodes: list[Node], vnode_number: int):
        self.nid2nodes = {}
        for node in nodes:
            self.nid2nodes[str(node.id)] = node

        self.node_position = {}
        for node in nodes:
            for vnode_id in range(vnode_number):
                nid = str(node.id) + "_" + str(vnode_id)
                hvalue = mmh3.hash(nid, 42, signed=False)
                self.node_position[nid] = hvalue

        # sort by node position (hvalues) for binary search
        self.node_position = {
            key: value
            for key, value in sorted(
                self.node_position.items(), key=lambda item: item[1]
            )
        }

        # these are used by ch-rlu
        self.iat_heap = {}
        self.last_access_times = {}

    def assign_task_ch(self, batch_tasks: list[Task]):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        assignment = {}
        nodes = list(self.node_position.values())

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            hvalue = mmh3.hash(fid, 42, signed=False)
            pos = bisect(nodes, hvalue)

            # make it a ring
            if pos == len(self.node_position):
                pos = 0

            node = self.nid2nodes[list(self.node_position)[pos].split("_")[0]]
            assignment[task] = node

        return assignment

    def assign_task_tjch(self, batch_tasks, epsilon):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        # compute the total workload for the batched tasks
        total_size = 0
        for task in batch_tasks:
            total_size += sum(task.totals.values())

        # determine the workload bound for each node
        workload_bound = 0
        if len(batch_tasks) >= len(self.nid2nodes):
            workload_bound = ceil((total_size / len(self.nid2nodes)) * (1 + epsilon))
        else:
            workload_bound = ceil((total_size / len(batch_tasks)) * (1 + epsilon))

        workload_stat = {}
        assignment = {}

        for nid in self.nid2nodes:
            workload_stat[nid] = 0
            assignment[nid] = []

        for task in batch_tasks:
            # generate a fixed-order permutation of nodes for each task
            fid = str(task.fragments[0].id)
            seed = mmh3.hash(fid, 42, signed=False)
            random.seed(seed)

            permutation = list(self.nid2nodes.values()).copy()
            random.shuffle(permutation)
            init_node = permutation[0]
            nid = str(init_node.id)

            if workload_stat[nid] + sum(task.totals.values()) <= workload_bound:
                workload_stat[nid] += sum(task.totals.values())
                assignment[nid].append(task)

            else:
                max_signature = tuple((0, None))
                next_choice = None

                for node in permutation[1:]:
                    nid = str(node.id)
                    if workload_stat[nid] + sum(task.totals.values()) > workload_bound:
                        continue

                    if next_choice == None:
                        next_choice = node

                    temp_signature = 0
                    for fragment in task.totals:
                        fid = str(fragment.id)
                        if fid in node.signature:
                            temp_signature += node.signature[fid]

                    if temp_signature > max_signature[0]:
                        max_signature = (temp_signature, node)

                # debug
                if next_choice == None and max_signature == (0, None):
                    print("[ERROR] Cannot find a new node in current epsilon!")
                    exit()

                if max_signature != (0, None):
                    nid = str(max_signature[1].id)
                    workload_stat[nid] += sum(task.totals.values())
                    assignment[nid].append(task)

                else:
                    nid = str(next_choice.id)
                    workload_stat[nid] += sum(task.totals.values())
                    assignment[nid].append(task)

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def assign_task_chrb(self, batch_tasks, alpha, timestep):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        assignment = {}
        for nid in self.nid2nodes:
            assignment[nid] = []
        # nodes = list(self.node_position.values())

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            frequency = self.task_hist.get_frequency(task, timestep)
            if frequency == 1:
                frequency = 0.99999
            r = min(
                max(int((frequency**alpha) * len(self.nid2nodes)), 1),
                len(self.nid2nodes),
            )
            seed = mmh3.hash(fid, 42, signed=False)

            # with virtual ring
            shuffler = random.Random(seed)
            permutation = list(self.nid2nodes.values()).copy()
            shuffler.shuffle(permutation)
            nid = str(permutation[random.randint(1, r)].id)
            assignment[nid].append(task)

            # # without virtual ring
            # pos = bisect(nodes, seed)
            # if pos == len(self.node_position):
            #     pos = 0
            # r = min(
            #     max(int((frequency**alpha) * len(self.node_position)), 1),
            #     len(self.node_position),
            # )
            # target_nodes = list(self.node_position)[pos : pos + r]
            # if len(target_nodes) < r:
            #     target_nodes += list(self.node_position)[: r - len(target_nodes)]
            # node_set = set()
            # for i in target_nodes:
            #     node_set.add(i.split("_")[0])
            # nid = random.choice(list(node_set))
            # assignment[nid].append(task)
            # # .split("_")[0]

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def assign_task_chbl(self, batch_tasks, epsilon):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        # compute the total workload for the batched tasks
        total_size = 0
        for task in batch_tasks:
            total_size += sum(task.totals.values())

        # determine the workload bound for each node
        workload_bound = 0
        if len(batch_tasks) >= len(self.nid2nodes):
            workload_bound = ceil((total_size / len(self.nid2nodes)) * (1 + epsilon))
        else:
            workload_bound = ceil((total_size / len(batch_tasks)) * (1 + epsilon))

        workload_stat = {}
        assignment = {}
        nodes = list(self.node_position.values())
        max_chain_len = len(nodes)

        for nid in self.nid2nodes:
            workload_stat[nid] = 0
            assignment[nid] = []

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            hvalue = mmh3.hash(fid, 42, signed=False)
            pos = bisect(nodes, hvalue)
            if pos == len(self.node_position):
                pos = 0

            nid = list(self.node_position)[pos].split("_")[0]

            if workload_stat[nid] + sum(task.totals.values()) <= workload_bound:
                workload_stat[nid] += sum(task.totals.values())
                assignment[nid].append(task)

            else:
                counter = 0
                nid = None
                assign_nid = None

                for counter in range(max_chain_len):
                    pos += 1
                    if pos == len(self.node_position):
                        pos = 0

                    nid = list(self.node_position)[pos].split("_")[0]

                    if workload_stat[nid] + sum(task.totals.values()) <= workload_bound:
                        workload_stat[nid] += sum(task.totals.values())
                        assignment[nid].append(task)
                        assign_nid = nid
                        break

                    else:
                        counter += 1

                if assign_nid == None:
                    nid = [
                        key
                        for key, value in sorted(
                            workload_stat.items(), key=lambda x: x[1], reverse=False
                        )
                    ][0]
                    workload_stat[nid] += sum(task.totals.values())
                    assignment[nid].append(task)

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def assign_task_rjch(self, batch_tasks, epsilon):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        # compute the total workload for the batched tasks
        total_size = 0
        for task in batch_tasks:
            total_size += sum(task.totals.values())

        # determine the workload bound for each node
        workload_bound = 0
        if len(batch_tasks) >= len(self.nid2nodes):
            workload_bound = ceil((total_size / len(self.nid2nodes)) * (1 + epsilon))
        else:
            workload_bound = ceil((total_size / len(batch_tasks)) * (1 + epsilon))

        workload_stat = {}
        assignment = {}
        nodes = list(self.node_position.values())

        for nid in self.nid2nodes:
            workload_stat[nid] = 0
            assignment[nid] = []

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            hvalue = mmh3.hash(fid, 42, signed=False)
            pos = bisect(nodes, hvalue)
            if pos == len(self.node_position):
                pos = 0

            nid = list(self.node_position)[pos].split("_")[0]

            if workload_stat[nid] + sum(task.totals.values()) <= workload_bound:
                workload_stat[nid] += sum(task.totals.values())
                assignment[nid].append(task)

            else:
                counter = 1
                while workload_stat[nid] + sum(task.totals.values()) > workload_bound:
                    fid = str(task.fragments[0].id) + "_" + str(counter)
                    hvalue = mmh3.hash(fid, 42, signed=False)
                    pos = bisect(nodes, hvalue)
                    if pos == len(self.node_position):
                        pos = 0

                    nid = list(self.node_position)[pos].split("_")[0]
                    counter += 1

                    if counter > len(nodes):
                        nid = [
                            key
                            for key, value in sorted(
                                workload_stat.items(), key=lambda x: x[1], reverse=False
                            )
                        ][0]
                        break

                workload_stat[nid] += sum(task.totals.values())
                assignment[nid].append(task)

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def assign_task_spore(self, batch_tasks):
        # no assignment if no tasks
        if len(batch_tasks) == 0:
            return {}

        stats = defaultdict(int)
        assignment = {}

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            stats[fid] += 1
        hotkeys = sorted(stats, key=stats.get, reverse=True)[:2]

        for nid in self.nid2nodes:
            assignment[nid] = []
        nodes = list(self.node_position.values())

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            hvalue = mmh3.hash(fid, 42, signed=False)
            pos = bisect(nodes, hvalue)
            if pos == len(self.node_position):
                pos = 0

            nid = list(self.node_position)[pos].split("_")[0]

            if fid not in hotkeys:
                assignment[nid].append(task)

            else:
                nodes_set = list()
                nodes_set.append(nid)
                counter = 1
                while counter < 3:
                    fid = str(task.fragments[0].id) + "_" + str(counter)
                    hvalue = mmh3.hash(fid, 42, signed=False)
                    pos = bisect(nodes, hvalue)
                    if pos == len(self.node_position):
                        pos = 0

                    nid = list(self.node_position)[pos].split("_")[0]
                    nodes_set.append(nid)
                    counter += 1

                nid = np.random.choice(nodes_set)
                assignment[nid].append(task)

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def assign_task_chrlu(self, batch_tasks, epsilon, current_time):
        max_chain_len = 3
        workload_bound = 1.2
        max_workload_bound = 6

        workload_stat = {}
        assignment = {}
        nodes = list(self.node_position.values())

        for nid in self.nid2nodes:
            workload_stat[nid] = (
                len(self.nid2nodes[nid].tasks) / self.nid2nodes[nid].cpus
            )
            assignment[nid] = []

        for task in batch_tasks:
            fid = str(task.fragments[0].id)
            hvalue = mmh3.hash(fid, 42, signed=False)
            pos = bisect(nodes, hvalue)
            if pos == len(self.node_position):
                pos = 0

            nid = list(self.node_position)[pos].split("_")[0]

            lmbd = 0
            random_load_update = 0
            avg_iat, is_popular = self.shards_popular(fid, hvalue, current_time)

            # TODO: avg_iat time will be buggy since we use batched query, we
            # should pass the ACTUAL time of the task to self.shards_popular()
            # instead of simulation time step
            if is_popular:
                if avg_iat == 0:
                    lmbd = max_workload_bound
                else:
                    lmbd = 1 / avg_iat

                random_load_update = np.random.normal(loc=lmbd, scale=0.1)

            if workload_stat[nid] + random_load_update < min(
                workload_bound, max_workload_bound
            ):
                workload_stat[nid] += 1 / self.nid2nodes[nid].cpus
                assignment[nid].append(task)

            else:
                counter = 0
                nid = None
                assign_nid = None

                for counter in range(max_chain_len):
                    pos += 1
                    if pos == len(self.node_position):
                        pos = 0

                    nid = list(self.node_position)[pos].split("_")[0]

                    if workload_stat[nid] + random_load_update <= min(
                        workload_bound, max_workload_bound
                    ):
                        workload_stat[nid] += 1 / self.nid2nodes[nid].cpus
                        assignment[nid].append(task)
                        assign_nid = nid
                        break

                    else:
                        counter += 1

                if assign_nid == None:
                    nid = [
                        key
                        for key, value in sorted(
                            workload_stat.items(), key=lambda x: x[1], reverse=False
                        )
                    ][0]
                    workload_stat[nid] += 1 / self.nid2nodes[nid].cpus
                    assignment[nid].append(task)

        result = {}
        for nid in assignment:
            if len(assignment[nid]) != 0:
                for task in assignment[nid]:
                    result[task] = self.nid2nodes[nid]

        return result

    def shards_popular(self, fid, hvalue, current_time):
        avg_iat = None
        is_popular = None

        if random.uniform(0, 1) <= 0.2:
            if fid in self.last_access_times:
                iat = (current_time - self.last_access_times[fid]) / 0.2
                self.last_access_times[fid] = current_time
                self.iat_heap[fid] = iat

            else:
                self.last_access_times[fid] = current_time
                self.iat_heap[fid] = current_time / 0.2

        pop_thresh = floor(len(self.iat_heap) * 0.2)

        if len(self.iat_heap) != 0:
            avg_iat = sum(self.iat_heap.values()) / len(self.iat_heap)
            pop_tasks = [
                key
                for key, value in sorted(
                    self.iat_heap.items(), key=lambda x: x[1], reverse=False
                )
            ][:pop_thresh]

            if fid in pop_tasks:
                is_popular = True

        return avg_iat, is_popular


ICachePolicy = Callable[[Node, Union[Chunk, Fragment]], bool]
