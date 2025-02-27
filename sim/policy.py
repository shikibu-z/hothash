from abc import ABC, abstractmethod
from typing import Callable, Union, Dict

from entities import Task, Chunk, Fragment, Node

ICachePolicy = Callable[[Node, Union[Chunk, Fragment]], bool]


class AbcPolicy(ABC):
    """
    Defines how the cluster should
        - schedule work
        - handle stragglers
        - manage cache
    """

    @abstractmethod
    def manage_tasks(
        self, timestep, nodes, idle_tasks, running_tasks
    ) -> Dict[Task, Node]:
        """
        Given:
            - a list of idle tasks which have not yet been started
            - running tasks which have been assigned to nodes previously
            - statistics for each node
            - current task assignments

        Generates a node assignment task -> node for some number of tasks from
        either the idle_task_queue or the running_task_queue.
        """
        pass

    @abstractmethod
    def manage_cache(self, timestep, nodes, idle_tasks, runnint_tasks) -> ICachePolicy:
        """
        Returns a function that is passed a node and a list of entities and
        returns a list of booleans indicating cache status
        """
        pass


class DefaultPolicy(AbcPolicy):
    """
    Basic policy which immediately assigns idle_tasks
    """

    def __init__(self, fragments):
        self.nodes = []
        self.cache_set = set(fragments)

    def manage_tasks(
        self, timestep, nodes, idle_tasks, running_tasks
    ) -> Dict[Task, Node]:
        idle_tasks = idle_tasks.copy()
        sorted_nodes = sorted(nodes, key=lambda n: len(n.tasks))
        assignments = {}

        for node in sorted_nodes:
            if len(node.tasks) < node.cpus and len(idle_tasks):
                task = idle_tasks.pop()

                if task.is_ready:
                    assignments[task] = node
        return assignments

    def should_cache(self, node: Node, entity: Union[Chunk, Fragment]) -> bool:
        if entity in self.cache_set and node in self.nodes:
            return True
        return False

    def manage_cache(self, timestep, nodes, idle_tasks, running_tasks) -> ICachePolicy:
        self.nodes = nodes
        return self.should_cache
