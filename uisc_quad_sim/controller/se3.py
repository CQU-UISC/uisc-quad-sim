from typing import List
import numpy as np
from .base import BaseController, ControlCommand, ControlMode
from ..dynamics import RigidbodyState
from ..utils.quaternion import q_mult, q_inv, q_rot, mat_q, to_euler
from ..utils.lpf import LPF


class PX4PositionController(BaseController):

    def __init__(self, dt=0.01):
        super().__init__()
        self.err_v_integral = np.zeros(3)
        self.v_lpf = LPF(dim=3, f_cut=5)
        self.last_v = np.zeros(3)
        self.max_integral = 5.0
        self.p_kp = np.array([0.95, 0.95, 1.0])
        self.v_kp = np.array([1.8, 1.8, 4.0])
        self.v_kd = np.array([0.2, 0.2, 0.0])
        self.v_ki = np.array([0.4, 0.4, 2.0])
        self.kw = 5
        self.tau = 0.1
        self.dt = dt

    def compute_control(
        self, state: RigidbodyState, state_sp: RigidbodyState
    ) -> ControlCommand:
        p = state.pos
        v = state.vel
        q = state.quat
        w = state.ang_vel

        p_d = state_sp.pos
        yaw_d = to_euler(state_sp.quat)[2]

        e_p = p - p_d
        v_d = -self.kw * e_p
        e_v = v - v_d

        v_filt = self.v_lpf.update(v, self.dt)
        err_v_diff = (v_filt - self.last_v) / self.dt

        self.err_v_integral += e_v * self.dt
        err_v_integral = np.clip(
            self.err_v_integral, -self.max_integral, self.max_integral
        )

        self.last_v = v_filt
        self.err_v_integral = err_v_integral

        a_d = (
            -self.v_kp * e_v
            - self.v_kd * err_v_diff
            - self.v_ki * err_v_integral
            + 9.81 * np.array([0, 0, 1])
        )

        # x corss y = z
        # z cross x = y
        # y cross z = x
        z_d = a_d / np.linalg.norm(a_d)  # desired z-axis
        x_d = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])  # desired x-axis
        y_d = np.cross(z_d, x_d)  # desired y-axis
        n_yd = np.linalg.norm(y_d)
        if n_yd < 1e-6:
            y_d = np.array([0, 1, 0])
        else:
            y_d = y_d / n_yd
        x_d = np.cross(y_d, z_d)

        R_d = np.array([x_d, y_d, z_d]).T
        q_d = mat_q(R_d)

        e_q = q_mult(q_inv(q), q_d)
        e_q_s = e_q[0]
        e_q_v = e_q[1:]

        w_d = 2 / self.tau * e_q_v * np.sign(e_q_s)  # desired angular velocity
        # CTBR
        # thrust
        z = q_rot(q, np.array([0, 0, 1]))
        u1 = np.dot(a_d, z)  # projection of a_d on z-axis
        cmd = ControlCommand(ControlMode.CTBR, np.concatenate([[u1], w_d]))
        return cmd


class SE3Controller(BaseController):

    def __init__(self, dt=0.01):
        super().__init__()
        self.err_p_integral = np.zeros(3)
        self.max_integral = 2.0

        self.p_kp = np.array([10.0, 10.0, 15.0])
        self.p_ki = np.array([2.0, 2.0, 4.0])
        self.v_kp = np.array([4.0, 4.0, 6.0])
        self.kw = 5
        self.tau = 0.1
        self.dt = dt

    def compute_control(
        self, state: RigidbodyState, state_sp: RigidbodyState
    ) -> ControlCommand:
        p = state.pos
        v = state.vel
        q = state.quat
        w = state.ang_vel
        acc = state.lin_acc

        p_d = state_sp.pos
        v_d = state_sp.vel
        a_d = state_sp.lin_acc
        yaw_d = to_euler(state_sp.quat)[2]

        e_p = p - p_d
        e_v = v - v_d

        self.err_p_integral += e_p * self.dt
        err_p_integral = np.clip(
            self.err_p_integral, -self.max_integral, self.max_integral
        )

        self.err_p_integral = err_p_integral

        a_d = (
            -self.p_kp * e_p
            - self.p_ki * err_p_integral
            - self.v_kp * e_v
            + 9.81 * np.array([0, 0, 1])
            + a_d
        )

        # x corss y = z
        # z cross x = y
        # y cross z = x
        z_d = a_d / np.linalg.norm(a_d)  # desired z-axis
        x_d = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])  # desired x-axis
        y_d = np.cross(z_d, x_d)  # desired y-axis
        n_yd = np.linalg.norm(y_d)
        if n_yd < 1e-6:
            y_d = np.array([0, 1, 0])
        else:
            y_d = y_d / n_yd
        x_d = np.cross(y_d, z_d)

        R_d = np.array([x_d, y_d, z_d]).T
        q_d = mat_q(R_d)

        e_q = q_mult(q_inv(q), q_d)
        e_q_s = e_q[0]
        e_q_v = e_q[1:]

        w_d = 2 / self.tau * e_q_v * np.sign(e_q_s)  # desired angular velocity
        # CTBR
        # thrust
        z = q_rot(q, np.array([0, 0, 1]))
        u1 = np.dot(a_d, z)  # projection of a_d on z-axis
        cmd = ControlCommand(ControlMode.CTBR, np.concatenate([[u1], w_d]))
        return cmd
