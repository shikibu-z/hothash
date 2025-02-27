from typing import List
from abc import ABC, abstractmethod

from entities import QueryPlan


class AbcWorkload(ABC):
    @abstractmethod
    def get_work(self, timestep) -> List[QueryPlan]:
        pass
