from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Abstract base class for all controllers.
    """

    def __init__(self, dt: float):
        """
        Initialize the controller with a time step.

        Args:
            dt (float): Time step for the controller.
        """
        self.dt = dt

    @abstractmethod
    def compute_control(self, state: np.ndarray, *args) -> np.ndarray:
        """
        Compute the control input based on the current state.

        Args:
            state (np.ndarray): Current state of the system.

        Returns:
            np.ndarray: Control input.
        """
        pass