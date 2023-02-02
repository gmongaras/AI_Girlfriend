from abc import ABC, abstractmethod
from typing import List

from torch import Tensor
from torch.nn import Module

try:
    from tha2.nn.base.module_factory import ModuleFactory
except ModuleNotFoundError:
    from ....tha2.nn.base.module_factory import ModuleFactory


class BatchInputModule(Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_from_batch(self, batch: List[Tensor]):
        pass


class BatchInputModuleFactory(ModuleFactory):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self) -> BatchInputModule:
        pass
