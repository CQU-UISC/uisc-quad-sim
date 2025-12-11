from dataclasses import dataclass
import numpy as np
import yaml
from loguru import logger

import os
from .base import Sim
from ..quadrotor.quadrotor import Quadrotor
from ..dynamics import QuadrotorDynamics, MotorDynamics, disturbance
from ..dynamics.disturbance import Disturbance
from ..controller.low_level_simple import LowlevelSimpleController


class ControlMode:
    CTBR = 0  # Collective Thrust Body Rate
    CTBM = 1  # Collective Thrust Body Moment
    SRT = 2  # Speed Rotor Thrust


@dataclass
class ControlCommand:
    type: ControlMode
    u: np.ndarray  # Control inputs, shape [4, N] for CTBR and CTBM, [4, N] for SRT


class QuadSimParams:

    def __init__(
        self, dt: float, quad: Quadrotor, g: float, nums: int, noise_std: np.ndarray
    ) -> None:
        self.dt = dt
        self.quad: Quadrotor = quad
        self.g = g
        self.nums = nums
        self.noise_std = noise_std
        self.disturb = disturbance.EmptyField()

    def __str__(self) -> str:
        return f"""Quadrotor Simulator Params:
    dt:{self.dt}
    quadrotor:{self.quad}
    g:{self.g}
    nums:{self.nums}
    noise_std:{self.noise_std}
    disturbance:{self.disturb}"""

    @staticmethod
    def load(file_path):
        cfg = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
        assert "dt" in cfg, "dt should be provided"
        assert "quadrotor" in cfg, "quadrotor config should be provided"
        assert "g" in cfg, "gravity should be provided"
        assert "nums" in cfg, "nums should be provided"
        assert "noise_std" in cfg, "noise_std should be provided"
        dt: float = cfg["dt"]
        low_level_dt: float = cfg["low_level_dt"] if "low_level_dt" in cfg else 0.001
        quad = Quadrotor.load(
            os.path.join(os.path.dirname(file_path), cfg["quadrotor"])
        )
        g: float = cfg["g"]
        nums: int = cfg["nums"]
        noise_std = np.array(cfg["noise_std"])
        assert noise_std.shape[0] == 13 + 4, "noise_std should be 17x1"

        return QuadSimParams(dt, quad, g, nums, noise_std)


"""
Quadrotor simulator
"""


class VecQuadSim(Sim):
    def __init__(self, sim: QuadSimParams) -> None:
        self._low_level_dt = (
            0.001  # angle velocity control run at 1000Hz current not use config
        )
        self._sim_cfg = sim
        self._quad = sim.quad
        self._low_level_steps = np.ceil(self._sim_cfg.dt / self._low_level_dt).astype(
            int
        )
        assert np.isclose(
            self._low_level_steps * self._low_level_dt, self._sim_cfg.dt
        ), "low level dt should be multiple of high level dt"
        self._rigid_dynamics = QuadrotorDynamics(
            self._quad._mass,
            self._sim_cfg.g,
            self._quad._J,
            self._quad._J_inv,
            self._quad._drag_coeff,
            self._sim_cfg.disturb,
        )
        self._low_level_ctrl = LowlevelSimpleController(self._quad._J)
        self._motor_dynamics = MotorDynamics(self._quad._tau_inv)
        super().__init__(self._low_level_dt)
        # 0  1  2  3  4  5  6  7  8  9  10 11 12
        # px py pz vx vy vz qw qx qy qz wx wy wz
        logger.info(
            "Quadrotor simulator initialized with params:\n{}".format(self._sim_cfg)
        )
        self.reset()

    @property
    def sim_dt(self):
        return self._sim_cfg.dt

    @property
    def quadrotor(self):
        return self._quad

    @property
    def disturbance(self):
        return self._rigid_dynamics.disturb

    def add_disturbance(self, disturb: Disturbance):
        """
        Add disturbance for the simulation
        Input:
            disturb: disturbance object
        """
        self._rigid_dynamics.add_disturbance(disturb)
        logger.info("Add disturbance:{}".format(disturb))

    def reset_disturbance(self):
        """
        Reset the disturbance to an empty field.
        """
        self._rigid_dynamics.reset_disturbance()
        logger.info("Reset disturbance to empty field")

    @property
    def motor_speed(self):
        return self._motors_omega

    @property
    def state(self):
        return self._x

    def _motor_commands(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate motor commands from control inputs
        Input:
            u: control inputs [thrust,torques] unit: [N,Nm] shape:[4,N]
        Output:
            omega: motor commands [4,N] unit: [0,1]
        """

        """Uncomment this block to use close form solution"""
        motor_thrust = self._quad.allocatioMatrixInv @ u
        cliped_motor_thrust = self._quad.clipMotorThrust(motor_thrust)
        motor_cmd = self._quad.thrustMapInv(cliped_motor_thrust)

        """Uncomment this block to use more accurate motor thrust allocation (current not support vectorized input)"""
        # assert u.shape[1] == 1, "Currently only support single control input"
        # m_sp = u[:,0]
        # m_sp = np.matrix([m_sp[1], m_sp[2], m_sp[3], m_sp[0]]).T
        # P = self._quad.Bm_inv
        # (motor_thrust, motor_thrust_new) = normal_mode(
        #     m_sp,
        #     P/ self._quad.thrust_map[1],
        #     self._quad.motor_min,
        #     self._quad.motor_max
        # )
        # logger.debug("raw motor omega:{}, new motor omega:{}".format(motor_thrust.T,motor_thrust_new.T))
        # motor_thrust = motor_thrust_new
        # motor_cmd = np.array(motor_thrust)
        logger.debug(
            "compute motor commands, thrust_torque:{}N, motor thrust:{}N, motor commands:{}".format(
                u.T, motor_thrust.T, motor_cmd.T
            )
        )
        return motor_cmd

    def _step_motor(self, motor_cmd: np.ndarray) -> np.ndarray:
        """
        Step the simulation by one time step
        Input:
            motor_cmd: control inputs [motor omega] unit: [0,1] shape:[4,N]
        Output:
            thrust_torque: [thrust,torques] unit: [N,Nm] shape:[4,N]
        """
        new_omega = self._run(
            self._motor_dynamics.dxdt,
            self._motors_omega,
            motor_cmd,
        )

        logger.debug(
            "motor omega:{}, new motor omega:{}".format(
                self._motors_omega.T, new_omega.T
            )
        )
        self._motors_omega = self._quad.clipMotorSpeed(new_omega)  # 4xN
        thrust = self._quad.thrustMap(self._motors_omega)  # 4xN
        logger.debug("motor thrust:{}N".format(thrust.T))
        thrust_torque = self._quad._B @ thrust  # 4xN
        return thrust_torque

    def _step_rigid(self, u: np.ndarray) -> np.ndarray:
        """
        Step the simulation by one time step
        u: control inputs [bodyrates] unit: [rad/s] shape:[3,N]
        """
        self._x = self._run(
            self._rigid_dynamics.dxdt,
            self._x,
            u,
        )
        return self._x

    def step(self, cmd: ControlCommand):
        for _ in range(self._low_level_steps):
            # step the simulation by one time step
            # u: control inputs [thrust,torques] unit: [m/s^2,Nm] shape:[4,N]
            # or
            # u: control inputs [thrust,bodyrates] unit: [m/s^2,rad/s] shape:[4,N]
            # u = u[:,None]
            self._x = self._step(cmd)
        return self._x

    def _log(self, state, control_setpoint):
        """
        Log the simulation
        """
        if not hasattr(self, "_state_log"):
            self._state_log = []
        if not hasattr(self, "_setpoint_log"):
            self._setpoint_log = []
        self._state_log.append(state)
        self._setpoint_log.append(control_setpoint)
        return

    def _step(self, cmd: ControlCommand) -> np.ndarray:
        """
        Step the simulation by one time step using control inputs
        Input:
            u: control inputs [thrust,torques] unit: [m/s^2,Nm] shape:[4,N]
            or
            u: control inputs [thrust,bodyrates] unit: [m/s^2,rad/s] shape:[4,N]
        Output:
            state: state after stepping the simulation

        # Simulation steps:
        - If control mode is bodyrates, calculate moment from control inputs. u = [collective_thrust,moment], 4xN
        - Calculate desired motor commands from control inputs. u = [motor omega], 4xN
        - Step the motor dynamics by one time step, get new motor omega. 4xN
        - Calculate thrust and torques from motor omega. 4xN [collective_thrust,torques]
        - Step the rigid body dynamics by one time step, get new state. 13xN
        """
        super()._step_t()
        u_sp = (
            cmd.u + np.random.randn(4, 1) * self._sim_cfg.noise_std[13 : 13 + 4, None]
        )
        logger.debug("control inputs:{}".format(cmd.u.T))
        mode = cmd.type
        if mode == ControlMode.CTBR:
            # CTBR
            tbm = np.zeros((4, self._sim_cfg.nums))
            tbm[1:4] = self._low_level_ctrl.compute_control(self._x, u_sp[1:4])
            tbm[0] = (
                self._quad.clipCollectiveThrust(u_sp[0]) * self._quad._mass
            )  # clip thrust=>[N]
            motor_cmd = self._motor_commands(tbm)  # 4xN omega in [0,1]
        elif mode == ControlMode.CTBM:
            # CTBM
            tbm = np.zeros((4, self._sim_cfg.nums))
            tbm[1:4] = u_sp[1:4]  # body torque
            tbm[0] = (
                self._quad.clipCollectiveThrust(u_sp[0]) * self._quad._mass
            )  # clip thrust=>[N]
            motor_cmd = self._motor_commands(tbm)  # 4xN omega in [0,1]
        elif mode == ControlMode.SRT:
            # SRT
            cliped_motor_thrust = self._quad.clipMotorThrust(u_sp)
            motor_cmd = self._quad.thrustMapInv(cliped_motor_thrust)
        # step dynamics
        thrust_torque = self._step_motor(
            motor_cmd=motor_cmd
        )  # 4xN, [thrust(m/s^2),torques]
        self._x = self._step_rigid(thrust_torque)
        self._x[6:10, :] /= np.linalg.norm(self._x[6:10, :])  # norm quaternion
        self._log(self._x, u_sp)
        return self._x

    def reset_pos(self, p: np.ndarray):
        """
        Reset the position of the quadrotor
        Input:
            p: position [x,y,z] unit: [m] shape:[3]
        """
        self._x[0:3, :] = p[:, None]
        return

    def set_seed(self, seed: int):
        return np.random.seed(seed)

    def reset(self, mean: np.ndarray = None, std: np.ndarray = None):
        """
        Reset the simulation
        Input:
            rand: if True, reset the state with random values
            mean: mean of the state 13x1
            std: standard deviation of the state 13x1
            13 * N
            4 * N
        """
        super().reset()
        rand = mean is not None and std is not None
        if rand:
            self._x = (
                np.random.randn(13, self._sim_cfg.nums) * std[:, None] + mean[:, None]
            )
            # norm quaternion
            self._x[6:10] /= np.linalg.norm(self._x[6:10], axis=0)
        else:
            self._x = np.zeros((13, self._sim_cfg.nums))
            self._motors_omega = np.zeros((4, self._sim_cfg.nums))
            self._x[6] = 1

    def estimate(self, gt: bool = False) -> np.ndarray:
        """
        Return Unbiased estimate of the state
        """
        state = self._x
        # motor_thrust = self._quad.thrustMap(self._motors_omega)
        # state = np.concatenate((self._x,motor_thrust),axis=0)
        if gt:
            return state
        noise = np.zeros_like(state)
        noise[:13, :] = np.random.randn(13, 1) * self._sim_cfg.noise_std[:13, None]
        est = state + noise
        return est

    def get_control_input(self, control_mode: ControlMode) -> np.ndarray:
        control_input = np.zeros((4, self._sim_cfg.nums))
        if control_mode == ControlMode.CTBR:
            thrust = self._quad.thrustMap(self._motors_omega)
            real_ct = np.sum(thrust) / self._quad.mass
            real_angvel = self._x[10:13, :]
            control_input[0, :] = real_ct
            control_input[1:4, :] = self._low_level_ctrl.compute_control(
                self._x, real_angvel
            )
        elif control_mode == ControlMode.CTBM:
            thrust = self._quad.thrustMap(self._motors_omega)
            real_wrench = self._quad._B @ thrust
            control_input[0, :] = real_wrench[0, :] / self._quad.mass
            control_input[1:4, :] = real_wrench[1:4, :]
        elif control_mode == ControlMode.SRT:
            thrust = self._quad.thrustMap(self._motors_omega)
            control_input = thrust
        else:
            raise ValueError("Invalid control mode")
        return control_input


class QuadSim(VecQuadSim):
    def step(self, cmd: ControlCommand):
        cmd.u = cmd.u[:, None]
        return super().step(cmd)[:, 0]

    def estimate(self, gt=False):
        return super().estimate(gt)[:, 0]

    @property
    def motor_speed(self):
        return super().motor_speed[:, 0]

    @property
    def state(self):
        return super().state[:, 0]

    def get_control_input(self, control_mode):
        return super().get_control_input(control_mode)[:, 0]
