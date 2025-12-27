import numpy as np
import numba as nb
from dataclasses import dataclass
from .base import Dynamics
from ..integration import integrate_fn


@dataclass
class BatteryParams:
    # Physical parameters
    n_cells: int  # Number of battery cells in series
    capacity_mah: float  # Battery capacity in mAh

    # Electrical parameters
    r_internal: float  # Internal resistance in Ohms
    v_cuttoff: float  # Cutoff voltage per cell in Volts
    v_full: float  # Full charge voltage per cell in Volts

    i_avionics: float  # Avionics current draw in Amps
    soc_ocv_coeffs: np.ndarray  # Open circuit voltage polynomial coefficients


@dataclass
class BatteryState:
    x: np.ndarray  # state vector shape(1)
    v_ocv: float
    v_term: float
    shape: tuple = (1,)

    @property
    def soc(self) -> float:
        return self.x[0]

    @soc.setter
    def soc(self, value: float):
        self.x[0] = float(value)


@dataclass
class BatteryControl:
    u: np.ndarray  # control vector shape(4) [motor_currents(4)]

    @property
    def motor_currents(self) -> np.ndarray:
        return self.u


@nb.njit
def dynamics_fn(
    x: np.ndarray, u: np.ndarray, capacity_as: float, i_avionics: float
) -> np.ndarray:
    """Battery dynamics model"""
    # Open circuit voltage model (linear approximation)
    x_dot: np.ndarray = np.zeros_like(x)
    i_motor = np.sum(u)
    i_total = i_motor + i_avionics
    soc_dot = -i_total / capacity_as
    x_dot[0] = soc_dot
    return x_dot


@nb.njit
def step_fn(
    x: np.ndarray, u: np.ndarray, dt: float, capacity_as: float, i_avionics: float
) -> np.ndarray:
    dynamics_args = (capacity_as, i_avionics)
    x_new = integrate_fn(dynamics_fn, x, u, dt, dynamics_args)
    return x_new


class Battery(Dynamics):
    def __init__(self, params: BatteryParams):
        self._params = params
        self._capacity_as = params.capacity_mah / 1000.0 * 3600.0  # convert mAh to As

    def initialize(self) -> BatteryState:
        v_ocv_init = self._params.n_cells * self._params.v_full
        x0 = np.array(
            [
                1.0,  # soc
            ]
        )
        return BatteryState(x=x0, v_ocv=v_ocv_init, v_term=v_ocv_init)

    def _soc_ocv_curve(self, soc: float) -> float:
        """Open circuit voltage as a function of state of charge using polynomial fit"""
        coeffs = self._params.soc_ocv_coeffs
        v_ocv_per_cell = 0.0
        for i, c in enumerate(coeffs):
            v_ocv_per_cell += c * (soc**i)
        return v_ocv_per_cell

    def _v_ocv(self, battery_state: BatteryState) -> float:
        """Open circuit voltage"""
        soc = battery_state.soc
        return self._params.n_cells * self._soc_ocv_curve(soc)

    def _v_term(
        self, battery_state: BatteryState, battery_control: BatteryControl
    ) -> float:
        """Terminal voltage"""
        v_ocv = self._v_ocv(battery_state)
        i_motor = np.sum(battery_control.motor_currents)
        v_term = v_ocv - (i_motor + self._params.i_avionics) * self._params.r_internal
        return v_term

    def step(
        self, dt: float, state: BatteryState, control_input: BatteryControl
    ) -> BatteryState:
        battery_state = state
        battery_control = control_input
        current_soc = battery_state.x
        motor_currents = battery_control.motor_currents

        next_soc = step_fn(
            current_soc, motor_currents, dt, self._capacity_as, self._params.i_avionics
        )

        battery_state.x = np.clip(next_soc, 0.0, 1.0)
        battery_state.v_ocv = self._v_ocv(battery_state)
        battery_state.v_term = self._v_term(battery_state, battery_control)
        return battery_state
