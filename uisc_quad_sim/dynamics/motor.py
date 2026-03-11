import numba as nb
import numpy as np
from dataclasses import dataclass
from .base import Dynamics
from ..integration import integrate_fn


@dataclass
class MotorParams:
    using_battery: bool  # whether to use battery model
    pos_x: np.ndarray  # motor x positions in body frame: m shape(4)
    pos_y: np.ndarray  # motor y positions in body frame: m shape(4)
    # pos_z: np.ndarray  # motor z positions in body frame: m shape(4) --- IGNORE ---
    cw: np.ndarray  # motor rotation direction: 1 for cw, -1 for ccw
    tau_up: float  # motor up time constant: s
    tau_down: float  # motor down time constant: s
    k_v: float  # motor speed constant: rpm/V
    thrust_coeff: float  # motor thrust coeff: N/(rpm)^2
    torque_coeff: float  # motor torque coeff: Nm/(rpm)^2
    i_idle: float  # motor idle current: A
    v_idle: float  # motor idle voltage: V
    i_max: float  # motor max current: A
    v_max: float  # max voltage: V


@dataclass
class MotorState:
    x: np.ndarray  # state vector shape(3,4) [rpm(4), i(4), v(4)]
    shape: tuple = (3, 4)

    @property
    def rpm(self) -> np.ndarray:
        return self.x[0]

    @rpm.setter
    def rpm(self, value: np.ndarray):
        self.x[0] = value

    @property
    def i(self) -> np.ndarray:
        return self.x[1]

    @i.setter
    def i(self, value: np.ndarray):
        self.x[1] = value

    @property
    def v(self) -> np.ndarray:
        return self.x[2]

    @v.setter
    def v(self, value: np.ndarray):
        self.x[2] = value


@dataclass
class MotorControl:
    u: np.ndarray  # control vector shape(2,4)
    shape: tuple = (2, 4)

    @property
    def v_ocv(self) -> np.ndarray:
        return self.u[0]

    @property
    def esc_setpoints(self) -> np.ndarray:
        """Motor ESC setpoints: 0-1"""
        return self.u[1]


@nb.njit
def dynamics_fn(
    x: np.ndarray, u: np.ndarray, tau_inv_up: float, tau_inv_down: float
) -> np.ndarray:
    """
    Quadrotor motor dynamics
    Input:
        x: state vector shape(4) (motor speed: rpm)
        x_dot: state derivative shape(4)
        u: control vector shape(4)
        tau_inv_up: motor up time constant inverse
        tau_inv_down: motor down time constant inverse
    Output:
        None
    """
    x_dot: np.ndarray = np.zeros_like(x)
    tau_inv = np.where(u > x, tau_inv_up, tau_inv_down)
    x_dot[:] = (u - x) * tau_inv
    return x_dot


@nb.njit
def step_motor_fn(
    x_rpm: np.ndarray,
    u_target_rpm: np.ndarray,
    dt: float,
    tau_inv_up: float,
    tau_inv_down: float,
) -> np.ndarray:
    dynamics_args = (tau_inv_up, tau_inv_down)
    x_new = integrate_fn(dynamics_fn, x_rpm, u_target_rpm, dt, dynamics_args)
    return x_new


class Motors(Dynamics):
    def __init__(self, motor_params: MotorParams):
        self._params = motor_params
        self._tau_inv_up = 1.0 / motor_params.tau_up
        self._tau_inv_down = 1.0 / motor_params.tau_down
        self._min_rpm = self._params.v_idle * self._params.k_v
        self._max_rpm = self._params.v_max * self._params.k_v
        if self._params.using_battery:
            self._step_fn = self._step
        else:
            self._step_fn = self._step_simple

    def initialize(self) -> MotorState:
        x0 = np.zeros((3, 4))  # rpm(4), i(4), v(4)
        return MotorState(x=x0)

    def clamp_rpm(self, rpm: np.ndarray) -> np.ndarray:
        """Clamp motor RPM to [min_rpm, max_rpm]"""
        return np.clip(rpm, self.min_rpm(), self.max_rpm())

    def min_rpm(self) -> float:
        """Minimum motor RPM"""
        return self._min_rpm

    def max_rpm(self) -> float:
        """Maximum motor RPM"""
        return self._max_rpm

    def normalized_rpm(self, motor_state: MotorState) -> np.ndarray:
        """Normalized motor RPM in [0, 1]"""
        return (motor_state.rpm - self.min_rpm()) / (self.max_rpm() - self.min_rpm())

    def wrench(self, motor_state: MotorState) -> np.ndarray:
        """Motor wrench in body frame: [thrust_x, thrust_y, thrust_z, tau_x, tau_y, tau_z]"""
        thrust_single = self._params.thrust_coeff * motor_state.rpm**2
        thrust = np.array([0.0, 0.0, np.sum(thrust_single)])
        tau_z = np.sum(self._params.cw * self._params.torque_coeff * motor_state.rpm**2)
        tau_x = np.sum(self._params.pos_y * thrust_single)
        tau_y = np.sum(-self._params.pos_x * thrust_single)
        tau = np.array([tau_x, tau_y, tau_z])
        return np.concatenate((thrust, tau))

    def currents(self, motor_state: MotorState) -> np.ndarray:
        """Motor currents"""
        normalized_rpm = self.normalized_rpm(motor_state)
        # Simple linear model for motor current
        i = (
            self._params.i_idle
            + (self._params.i_max - self._params.i_idle) * normalized_rpm**2
        )
        return i

    def step(
        self, dt: float, state: MotorState, control_input: MotorControl
    ) -> MotorState:
        """Step motor state"""
        motor_state = state
        return self._step_fn(dt, motor_state, control_input)

    def _step(
        self, dt: float, motor_state: MotorState, control_input: MotorControl
    ) -> MotorState:
        """Step motor state with battery model"""
        current_rpm = motor_state.rpm
        motor_voltages = control_input.v_ocv * control_input.esc_setpoints
        # motor_voltages = np.clip(motor_voltages, 0.0, self._params.v_max)
        eta = 0.99  # efficiency factor
        target_rpm = (
            motor_voltages * self._params.k_v * eta
        )  # do a voltage compensation

        new_rpm = step_motor_fn(
            current_rpm, target_rpm, dt, self._tau_inv_up, self._tau_inv_down
        )

        motor_state.rpm = new_rpm
        motor_state.v = motor_voltages
        motor_state.i = self.currents(motor_state)
        return motor_state

    def _step_simple(
        self, dt: float, motor_state: MotorState, control_input: MotorControl
    ) -> MotorState:
        """Step motor state with simple model"""
        current_rpm = motor_state.rpm
        target_rpm = control_input.esc_setpoints * self.max_rpm()
        new_rpm = step_motor_fn(
            current_rpm, target_rpm, dt, self._tau_inv_up, self._tau_inv_down
        )
        motor_state.rpm = new_rpm
        motor_state.v[:] = 0.0
        motor_state.i[:] = 0.0
        return motor_state
