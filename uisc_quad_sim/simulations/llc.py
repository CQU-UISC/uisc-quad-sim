import numpy as np
from dataclasses import dataclass


@dataclass
class LLCParams:
    inertia: np.ndarray  # 1-d array of shape (3,)
    kp: np.ndarray  # 1-d array of shape (3,)


class LowLevelSimpleController:
    def __init__(self, params: LLCParams):
        super().__init__()
        self.Kp = np.diag(params.kp)
        self.J = np.diag(params.inertia)

    def compute_control(self, state, *args):
        """
            Input:
                u: control inputs [bodyrates] unit: [rad/s] shape:[3]
            Output:
                torque: [3] unit: [Nm] shape:[3]

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
        Kp = self.Kp
        w = state[10:13]
        w_d = args[0]
        w_err = w_d - w  # [3]
        tau = J @ (Kp @ w_err) + np.cross(w, J @ w)
        return tau
