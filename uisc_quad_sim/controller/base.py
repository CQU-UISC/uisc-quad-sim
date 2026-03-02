import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


class ControlMode:
    CTBR = 0  # Collective Thrust Body Rate
    CTBM = 1  # Collective Thrust Body Moment
    SRT = 2  # Speed Rotor Thrust (Raw Motor Thrusts)


@dataclass
class ControlCommand:
    type: int
    u: np.ndarray  # Control inputs
    rotor_forces: Optional[np.ndarray] = None  # optional, motor thrusts


class BaseController(ABC):
    """
    Abstract base class for all controllers.
    """

    @abstractmethod
    def compute_control(self, state, state_sp) -> ControlCommand:
        pass
