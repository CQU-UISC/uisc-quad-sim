from abc import ABC, abstractmethod
from typing import Any


class Dynamics(ABC):
    @abstractmethod
    def initialize(self) -> Any:
        pass

    @abstractmethod
    def step(self, dt: float, state: Any, control_input: Any) -> Any:
        pass
