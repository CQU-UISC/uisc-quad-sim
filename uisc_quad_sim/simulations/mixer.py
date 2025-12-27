import numpy as np
from dataclasses import dataclass


@dataclass
class MixerParams:
    pos_x: np.ndarray  # motor x positions
    pos_y: np.ndarray  # motor y positions
    cw: np.ndarray  # motor spin directions
    thrust_map: (
        np.ndarray
    )  # thrust map coefficients: [a,b,c] for thrust = a*cmd^2 + b*cmd + c
    k_tau_z: float  # torque to thrust ratio around z axis, unit: Nm/N


class Mixer:
    def __init__(self, params: MixerParams):
        self.params = params
        self.B = self._mixing_matrix()
        self.B_inv = np.linalg.pinv(self.B)

    def mix(self, thrust: float, tau: np.ndarray) -> np.ndarray:
        """
        Mix desired thrust and torques into motor commands.
        Args:
            thrust: float - desired total thrust unit: N
            tau: np.ndarray - desired torques [3] unit: Nm
        Returns:
            motor_cmds: np.ndarray - motor commands in [0,1], unitless shape:[4]
        """
        n_motors = len(self.params.pos_x)
        motor_thrusts = np.zeros(n_motors)
        wrench = np.array([thrust, tau[0], tau[1], tau[2]])  # [4]
        motor_thrusts = self.B_inv @ wrench  # [n_motors]
        motor_cmds = np.zeros(n_motors)
        for i in range(n_motors):
            motor_cmds[i] = self._thrust_to_cmd(motor_thrusts[i])
        return motor_cmds

    def _mixing_matrix(self) -> np.ndarray:
        """
        Compute the mixing matrix B that maps motor thrusts to body wrench.
        Returns:
            B: np.ndarray - mixing matrix shape:[4, n_motors]
        """
        n_motors = len(self.params.pos_x)
        B = np.zeros((4, n_motors))
        B[0, :] = 1.0  # Total thrust
        B[1, :] = self.params.pos_y  # Tau_x
        B[2, :] = -self.params.pos_x  # Tau_y
        B[3, :] = self.params.cw * self.params.k_tau_z  # Tau_z
        return B

    def _thrust_to_cmd(self, thrust: float) -> float:
        """
        Convert desired thrust to motor command using inverse of thrust map.
        Args:
            thrust: float - desired thrust in N
        Returns:
            cmd: float - motor command in [0,1]
        """
        # Inverse of thrust map
        a, b, c = self.params.thrust_map
        discrim = b**2 - 4 * a * (c - thrust)
        if discrim < 0:
            return 0.0
        cmd = (-b + np.sqrt(discrim)) / (2 * a)
        cmd = np.clip(cmd, 0.0, 1.0)
        return cmd

    def _cmd_to_thrust(self, cmd: float) -> float:
        """
        Convert motor command to thrust using thrust map.
        Args:
            cmd: float - motor command in [0,1]
        Returns:
            thrust: float - thrust in N
        """
        a, b, c = self.params.thrust_map
        thrust = a * cmd**2 + b * cmd + c
        return thrust
