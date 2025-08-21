from .base import BaseController
import numpy as np


class LowlevelSimpleController(BaseController):
    def __init__(self, J: np.ndarray):
        super().__init__()
        self.Kp = np.diag([20, 20, 41])
        self.J = J

    def compute_control(self, state, *args):
        """
            Input:
                u: control inputs [bodyrates] unit: [rad/s] shape:[3,N]
            Output:
                torque: [3,N] unit: [Nm] shape:[3,N]

            # Control law:
            w_dot = J_inv @ (tau - w x J @ w)\\
            if we want w_err have asymptotic stability=>\\
            w_err = w_d - w\\
            v(t) = 0.5*w_err.T@w_err\\
            v_dot(t) = w_err.T @ dot_w_err = w_err @ (-dot_w) = -w_err @  J_inv @ (tau - w x J @ w)\\
            if let tau = J @ Kp * w_err +  w x J @ w\\
            v_dot = -w_err @  J_inv @ (J @ Kp * w_err) <= 0
        """
        J = self.J
        Kp = np.diag([20, 20, 41])
        w = state[10:13]
        w_d = args[0]
        w_err = w_d - w  # [3,N]
        # J_inv@(ext_moment + tau - np.cross(w.T,np.dot(J,w).T).T)
        #
        tau = J @ Kp @ w_err + np.cross(w.T, np.dot(J, w).T).T
        return tau
