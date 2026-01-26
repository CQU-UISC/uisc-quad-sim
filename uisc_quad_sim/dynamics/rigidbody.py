import numba as nb
import numpy as np
from dataclasses import dataclass

from .base import Dynamics
from .disturbance import CompositeField, Disturbance, AirDrag
from ..integration import integrate_fn


@dataclass
class RigidbodyParams:
    mass: float  # mass: kg
    g: float  # gravity: m/s^2
    J: np.ndarray  # inertia matrix: kg*m^2 shape(3,3)
    J_inv: np.ndarray  # inverse inertia matrix: 1/(kg*m^2) shape(3,3)
    drag_coeff: np.ndarray  # drag coefficient shape(3)


@dataclass
class RigidbodyState:
    x: (
        np.ndarray
    )  # state vector shape(22) [pos(3), vel(3), quat(4), omega(3), lin_acc(3), ang_acc(3), blin_acc(3)]
    shape: tuple = (22,)

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:3]

    @pos.setter
    def pos(self, value: np.ndarray):
        self.x[0:3] = value

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6]

    @vel.setter
    def vel(self, value: np.ndarray):
        self.x[3:6] = value

    @property
    def quat(self) -> np.ndarray:
        return self.x[6:10]

    @quat.setter
    def quat(self, value: np.ndarray):
        v_norm = np.linalg.norm(value)
        if v_norm < 1e-6:
            raise ValueError("Quaternion norm is too small to normalize.")
        self.x[6:10] = value / v_norm

    @property
    def ang_vel(self) -> np.ndarray:
        return self.x[10:13]

    @ang_vel.setter
    def ang_vel(self, value: np.ndarray):
        self.x[10:13] = value

    @property
    def lin_acc(self) -> np.ndarray:
        return self.x[13:16]

    @lin_acc.setter
    def lin_acc(self, value: np.ndarray):
        self.x[13:16] = value

    @property
    def ang_acc(self) -> np.ndarray:
        return self.x[16:19]

    @ang_acc.setter
    def ang_acc(self, value: np.ndarray):
        self.x[16:19] = value

    @property
    def blin_acc(self) -> np.ndarray:
        return self.x[19:22]

    @blin_acc.setter
    def blin_acc(self, value: np.ndarray):
        self.x[19:22] = value


@dataclass
class RigidbodyControl:
    u: np.ndarray  # control vector shape(4) [thrust, tau_x, tau_y, tau_z]

    @property
    def thrust(self) -> float:
        return self.u[0]

    @property
    def tau(self) -> np.ndarray:
        return self.u[1:4]


@nb.njit
def dynamics_fn(
    x: np.ndarray,
    u: np.ndarray,
    mass: float,
    g: float,
    J: np.ndarray,
    J_inv: np.ndarray,
    ext_force: np.ndarray,
    ext_moment: np.ndarray,
) -> np.ndarray:
    # x layout: [pos(3), vel(3), quat(4), omega(3)] -> size 13

    # Unpack state
    vel = x[3:6]
    quat = x[6:10]  # w, x, y, z
    ang_vel = x[10:13]

    # Unpack control
    motor_thrust = u[0]
    body_tau = u[1:4]

    # Pre-computations
    mass_inv = 1.0 / mass
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    wx, wy, wz = ang_vel[0], ang_vel[1], ang_vel[2]

    # Initialize derivative vector (size 13)
    x_dot = np.zeros(13)

    # 1. Position derivative (Velocity)
    x_dot[0:3] = vel

    # 2. Velocity derivative (Linear Acceleration)
    # R_body_to_inertial * [0, 0, T] - [0, 0, g] + F_ext/m
    # Fz_body converted to inertial frame:
    # r13 = 2(qw*qy + qx*qz)
    # r23 = 2(qy*qz - qw*qx)
    # r33 = 1 - 2(qx^2 + qy^2)

    acc_x = (2 * (qw * qy + qx * qz)) * motor_thrust * mass_inv + ext_force[
        0
    ] * mass_inv
    acc_y = (2 * (qy * qz - qw * qx)) * motor_thrust * mass_inv + ext_force[
        1
    ] * mass_inv
    acc_z = (
        ((1.0 - 2.0 * (qx**2 + qy**2)) * motor_thrust * mass_inv)
        - g
        + ext_force[2] * mass_inv
    )

    x_dot[3] = acc_x
    x_dot[4] = acc_y
    x_dot[5] = acc_z

    # 3. Quaternion derivative (Kinematics)
    # q_dot = 0.5 * q * omega
    x_dot[6] = 0.5 * (-wx * qx - wy * qy - wz * qz)
    x_dot[7] = 0.5 * (wx * qw + wz * qy - wy * qz)
    x_dot[8] = 0.5 * (wy * qw - wz * qx + wx * qz)
    x_dot[9] = 0.5 * (wz * qw + wy * qx - wx * qy)

    # 4. Angular Velocity derivative (Angular Acceleration)
    # omega_dot = J_inv * (tau_total - omega x (J * omega))
    # cross_product: ang_vel x (J @ ang_vel)
    gyroscopic = np.cross(ang_vel, J @ ang_vel)
    torque_sum = ext_moment + body_tau - gyroscopic
    ang_acc = J_inv @ torque_sum

    x_dot[10:13] = ang_acc

    return x_dot


@nb.njit
def step_fn(
    x_ode: np.ndarray,
    u: np.ndarray,
    dt: float,
    mass: float,
    g: float,
    J: np.ndarray,
    J_inv: np.ndarray,
    ext_force: np.ndarray,
    ext_moment: np.ndarray,
) -> np.ndarray:
    dynamics_args = (mass, g, J, J_inv, ext_force, ext_moment)
    x_new = integrate_fn(dynamics_fn, x_ode, u, dt, dynamics_args)
    q = x_new[6:10]
    q_norm = np.sqrt(np.sum(q**2))
    if q_norm > 1e-6:
        x_new[6:10] = q / q_norm
    return x_new


@nb.njit(cache=True)
def compute_accelerations(
    x: np.ndarray,
    u: np.ndarray,
    mass: float,
    g: float,
    J: np.ndarray,
    J_inv: np.ndarray,
    ext_force: np.ndarray,
    ext_moment: np.ndarray,
) -> np.ndarray:
    x_dot = dynamics_fn(x, u, mass, g, J, J_inv, ext_force, ext_moment)
    res = np.empty(9)
    lin_acc = x_dot[3:6]  # vel_dot -> lin_acc
    ang_acc = x_dot[10:13]  # omega_dot -> ang_acc
    # Body frame linear acceleration
    qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    R_body_to_inertial = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    R_inertial_to_body = R_body_to_inertial.T
    blin_acc = R_inertial_to_body @ lin_acc
    res[0:3] = lin_acc
    res[3:6] = ang_acc
    res[6:9] = blin_acc
    return res


class Rigidbody(Dynamics):
    def __init__(self, params: RigidbodyParams):
        self._disturb = CompositeField(AirDrag(params.drag_coeff))
        self._params = params

    def initialize(self) -> RigidbodyState:
        x0 = np.zeros(22)
        x0[6] = 1.0  # initial quaternion w=1
        return RigidbodyState(x=x0)

    def add_disturbance(self, disturb: Disturbance):
        """
        Add disturbance to the quadrotor dynamics.
        :param disturb: Disturbance object to be added.
        """
        self._disturb.add(disturb)

    def reset_disturbance(self):
        """
        Reset the disturbance to an empty field.
        """
        self._disturb.clear()
        self._disturb.add(AirDrag(self._params.drag_coeff))

    def step(
        self, dt: float, state: RigidbodyState, control_input: RigidbodyControl
    ) -> RigidbodyState:
        rigid_state = state
        x_ode = rigid_state.x[0:13]
        u = control_input.u
        ext_force = self._disturb.force(rigid_state.x, u)
        ext_moment = self._disturb.moment(rigid_state.x, u)
        x_new_ode = step_fn(
            x_ode,
            u,
            dt,
            self._params.mass,
            self._params.g,
            self._params.J,
            self._params.J_inv,
            ext_force,
            ext_moment,
        )
        accs = compute_accelerations(
            x_ode,
            u,
            self._params.mass,
            self._params.g,
            self._params.J,
            self._params.J_inv,
            ext_force,
            ext_moment,
        )
        rigid_state.x[0:13] = x_new_ode
        rigid_state.lin_acc = accs[0:3]  # lin_acc
        rigid_state.ang_acc = accs[3:6]  # ang_acc
        rigid_state.blin_acc = accs[6:9]  # blin_acc
        return rigid_state
