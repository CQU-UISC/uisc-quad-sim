import numpy as np
import yaml

from .llc import LowLevelSimpleController, LLCParams
from .mixer import Mixer, MixerParams
from ..controller import ControlCommand, ControlMode
from ..dynamics import Rigidbody, RigidbodyParams, RigidbodyState, RigidbodyControl
from ..dynamics import Motors, MotorParams, MotorState, MotorControl
from ..dynamics import Battery, BatteryParams, BatteryState, BatteryControl
from ..dynamics.disturbance import Disturbance


class QuadParams:
    """
    Parses and holds configuration parameters for the Quadrotor subsystems.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.name: str = cfg.get("name", "quad")
        self.low_level_dt: float = cfg.get("low_level_dt", 0.001)
        self.high_level_dt: float = cfg.get("high_level_dt", 0.01)

        # 1. Battery Params
        b_cfg = cfg["battery"]
        self.battery = BatteryParams(
            n_cells=b_cfg["n_cells"],
            capacity_mah=b_cfg["capacity_mah"],
            r_internal=b_cfg["r_internal"],
            r_polarization=b_cfg["r_polarization"],
            c_polarization=b_cfg["c_polarization"],
            v_cutoff=b_cfg["v_cutoff"],
            v_full=b_cfg["v_full"],
            i_avionics=b_cfg["i_avionics"],
            soc_ocv_coeffs=np.array(b_cfg["soc_ocv_coeffs"]),
        )

        # 2. Motor Params
        m_cfg = cfg["motors"]
        self.motor = MotorParams(
            using_battery=m_cfg["using_battery"],
            pos_x=np.array(m_cfg["pos_x"]),
            pos_y=np.array(m_cfg["pos_y"]),
            cw=np.array(m_cfg["cw"]),
            tau_up=m_cfg["tau_up"],
            tau_down=m_cfg["tau_down"],
            k_v=m_cfg["k_v"],
            thrust_coeff=m_cfg["thrust_coeff"],
            torque_coeff=m_cfg["torque_coeff"],
            i_idle=m_cfg["i_idle"],
            v_idle=m_cfg["v_idle"],
            i_max=m_cfg["i_max"],
            v_max=m_cfg["v_max"],
        )

        # 3. Rigid Body Params
        r_cfg = cfg["rigid_body"]
        inertia = np.diag(r_cfg["inertia"])
        inertia_inv = np.linalg.inv(inertia)
        self.rigid = RigidbodyParams(
            mass=r_cfg["mass"],
            g=r_cfg["g"],
            J=inertia,
            J_inv=inertia_inv,
            drag_coeff=np.array(r_cfg["drag_coeff"]),
        )

        # 4. Low Level Controller & Mixer Params
        ll_cfg = cfg["low_level"]
        mix_cfg = ll_cfg["mixer"]
        ctrl_cfg = ll_cfg["low_level_ctrl"]

        self.mixer = MixerParams(
            pos_x=np.array(mix_cfg["pos_x"]),
            pos_y=np.array(mix_cfg["pos_y"]),
            cw=np.array(mix_cfg["cw"]),
            thrust_map=np.array(mix_cfg["thrust_map"]),
            k_tau_z=mix_cfg["k_tau_z"],
        )

        self.llc = LLCParams(
            inertia=np.array(ctrl_cfg["inertia"]), kp=np.array(ctrl_cfg["kp"])
        )


class QuadSim:
    """
    Quadrotor Simulator aggregating Rigidbody, Motors, and Battery dynamics.
    """

    def __init__(self, params: QuadParams):
        self._params = params
        self._dt = params.low_level_dt
        self._steps_per_ctrl = int(round(params.high_level_dt / params.low_level_dt))
        self.t = 0.0

        # Initialize Subsystems
        self._rb_dyn = Rigidbody(params.rigid)
        self._motor_dyn = Motors(params.motor)
        self._batt_dyn = Battery(params.battery)
        self._mixer = Mixer(params.mixer)
        self._llc = LowLevelSimpleController(params.llc)

        # Initialize States
        self.reset()

    def reset(self):
        """Reset the simulation to initial conditions."""
        self.t = 0.0
        self._rb_state = self._rb_dyn.initialize()
        self._motor_state = self._motor_dyn.initialize()
        self._batt_state = self._batt_dyn.initialize()

        # Reset disturbance to default (usually just AirDrag)
        self._rb_dyn.reset_disturbance()

    def add_disturbance(self, disturb: Disturbance):
        """Add a specific disturbance to the rigid body dynamics."""
        self._rb_dyn.add_disturbance(disturb)

    # --- Accessors ---
    @property
    def params(self) -> QuadParams:
        return self._params

    @property
    def rb_dyn(self) -> Rigidbody:
        return self._rb_dyn

    @property
    def motor_dyn(self) -> Motors:
        return self._motor_dyn

    @property
    def batt_dyn(self) -> Battery:
        return self._batt_dyn

    @property
    def rb_state(self) -> RigidbodyState:
        return self._rb_state

    @property
    def motor_state(self) -> MotorState:
        return self._motor_state

    @property
    def batt_state(self) -> BatteryState:
        return self._batt_state

    @property
    def time(self) -> float:
        return self.t

    # --- Core Logic ---

    def _get_static_motor_setpoints(self, cmd: ControlCommand) -> np.ndarray:
        """
        Calculate motor setpoints for modes that do NOT require high-frequency feedback (CTBM, SRT).
        For CTBR, this returns zeros (handled inside the loop).
        """
        esc_setpoints = np.zeros(4)

        if cmd.type == ControlMode.CTBM:
            # Collective Thrust + Body Moment -> Mixer (Open Loop Torque)
            target_thrust = cmd.u[0] * self._params.rigid.mass
            target_torque = cmd.u[1:4]
            esc_setpoints = self._mixer.mix(target_thrust, target_torque)

        elif cmd.type == ControlMode.SRT:
            # Speed Rotor Thrust (Desire Thrust per motor) -> Inverse Thrust Map
            desired_motor_thrusts = cmd.u
            for i in range(4):
                esc_setpoints[i] = self._mixer._thrust_to_cmd(desired_motor_thrusts[i])

        return esc_setpoints

    def step(self, cmd: ControlCommand) -> RigidbodyState:
        is_rate_control = cmd.type == ControlMode.CTBR

        esc_setpoints = np.zeros(4)
        target_thrust = 0.0
        target_body_rate = np.zeros(3)

        # ZOH (Zero-Order Hold) for high-level commands
        if not is_rate_control:
            esc_setpoints = self._get_static_motor_setpoints(cmd)
        else:
            target_thrust = cmd.u[0] * self._params.rigid.mass
            target_body_rate = cmd.u[1:4]

        for _ in range(self._steps_per_ctrl):
            if is_rate_control:
                torque = self._llc.compute_control(self._rb_state.x, target_body_rate)
                esc_setpoints = self._mixer.mix(target_thrust, torque)
            # print("target_thrust:", target_thrust, " torque:", torque, " esc_setpoints:", esc_setpoints)
            self._low_level_step_logic(esc_setpoints)
            self.t += self._dt

        return self._rb_state

    def _low_level_step_logic(self, esc_setpoints: np.ndarray):
        """
        Execute one low-level integration step for all subsystems.
        """

        # 1. Prepare Motor Control
        # Motor voltage depends on Battery open-circuit voltage
        v_ocv = self._batt_state.v_ocv

        # Combine voltage and setpoints into control matrix
        # u shape: (2, 4) -> [ [v, v, v, v], [cmd, cmd, cmd, cmd] ]
        u_motor = np.vstack([np.full(4, v_ocv), esc_setpoints])
        motor_ctrl = MotorControl(u=u_motor)

        # 2. Step Motors
        # Updates RPM, Current, Voltage in motor state
        self._motor_dyn.step(self._dt, self._motor_state, motor_ctrl)

        # 3. Prepare Battery Control
        # Battery load depends on Motor currents
        motor_currents = self._motor_state.i
        batt_ctrl = BatteryControl(u=motor_currents)

        # 4. Step Battery
        # Updates SOC, OCV, Terminal Voltage
        self._batt_dyn.step(self._dt, self._batt_state, batt_ctrl)

        # 5. Prepare Rigid Body Control
        # Calculate forces/torques generated by motors
        wrench = self._motor_dyn.wrench(self._motor_state)
        # wrench format: [thrust_x, thrust_y, thrust_z, tau_x, tau_y, tau_z]

        # Extract total Z thrust and body torques
        total_thrust = wrench[2]
        total_torque = wrench[3:6]

        rb_ctrl = RigidbodyControl(u=np.concatenate(([total_thrust], total_torque)))

        # 6. Step Rigid Body
        self._rb_dyn.step(self._dt, self._rb_state, rb_ctrl)
