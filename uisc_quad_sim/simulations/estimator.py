import numpy as np
from dataclasses import dataclass, field
from uisc_quad_sim.dynamics import RigidbodyState, BatteryState, MotorState
from uisc_quad_sim.utils.quaternion import q_inv, q_rot


@dataclass
class EstimatorParams:
    ground_truth: bool = (
        True  # If True, the estimator will return the true state without noise
    )
    position_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for position measurements (m)
    velocity_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for velocity measurements (m/s)
    attitude_std: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )  # Standard deviation for attitude measurements (unitless quaternion)
    ang_velocity_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for angular velocity measurements (rad/s)
    lin_acc_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for linear acceleration measurements (m/s^2)
    ang_acc_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for angular acceleration measurements (rad/s^2)
    imu_acc_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for IMU acceleration measurements (m/s^2)
    imu_acc_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Initial bias for IMU acceleration (m/s^2)
    imu_gyro_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Standard deviation for IMU gyroscope measurements (rad/s)
    imu_gyro_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Initial bias for IMU gyroscope (rad/s)


class Estimator:
    def __init__(self, params: EstimatorParams):
        self._params = params
        self._initialized = False

    def initialize(
        self,
        rigid_state: RigidbodyState,
        motor_state: MotorState,
        batt_state: BatteryState,
    ):
        """
        Initialize the estimator state based on initial measurements.
        """
        self._rigid_state_gt = rigid_state
        self._motor_state_gt = motor_state
        self._batt_state_gt = batt_state
        self._g_vec = np.array([0, 0, -9.812])  # Gravity vector in inertial frame
        self._initialized = True

    def rigid(self, ground_truth: bool = False) -> RigidbodyState:
        """
        Return the current estimated rigid body state.
        """
        rigid_state_est = RigidbodyState(self._rigid_state_gt.x.copy())
        if self._params.ground_truth or ground_truth:
            return rigid_state_est
        # Add measurement noise
        rigid_state_est.pos += np.random.normal(0, self._params.position_std)
        rigid_state_est.vel += np.random.normal(0, self._params.velocity_std)
        rigid_state_est.quat += np.random.normal(0, self._params.attitude_std)
        rigid_state_est.quat /= np.linalg.norm(
            rigid_state_est.quat
        )  # Normalize quaternion
        rigid_state_est.ang_vel += np.random.normal(0, self._params.ang_velocity_std)
        rigid_state_est.lin_acc += np.random.normal(0, self._params.lin_acc_std)
        rigid_state_est.ang_acc += np.random.normal(0, self._params.ang_acc_std)
        return rigid_state_est

    def motor(self) -> MotorState:
        """
        Return the current estimated motor state.
        """
        motor_state_est = MotorState(self._motor_state_gt.x.copy())
        return motor_state_est

    def battery(self) -> BatteryState:
        """
        Return the current estimated battery state.
        """
        batt_state_est = BatteryState(
            self._batt_state_gt.x.copy(),
            v_ocv=self._batt_state_gt.v_ocv,
            v_term=self._batt_state_gt.v_term,
        )
        return batt_state_est

    def imu_acc(self, ground_truth: bool = False) -> np.ndarray:
        """
        Return the current estimated IMU acceleration measurement.
        """
        blin_acc = (
            self._rigid_state_gt.blin_acc.copy()
        )  # Get the true body-frame linear acceleration
        body_g_vec = q_rot(
            q_inv(self._rigid_state_gt.quat), self._g_vec
        )  # Rotate gravity vector into body frame
        imu_acc_est = (
            blin_acc - body_g_vec
        )  # Subtract gravity to get linear acceleration
        if self._params.ground_truth or ground_truth:
            return imu_acc_est
        # Add bias and noise
        imu_acc_est += self._params.imu_acc_bias
        imu_acc_est += np.random.normal(0, self._params.imu_acc_std)
        return imu_acc_est

    def imu_gyro(self, ground_truth: bool = False) -> np.ndarray:
        """
        Return the current estimated IMU gyroscope measurement.
        """
        ang_vel = self._rigid_state_gt.ang_vel.copy()  # Get the true angular velocity
        imu_gyro_est = ang_vel  # IMU gyro measures angular velocity directly
        if self._params.ground_truth or ground_truth:
            return imu_gyro_est
        # Add bias and noise
        imu_gyro_est += self._params.imu_gyro_bias
        imu_gyro_est += np.random.normal(0, self._params.imu_gyro_std)
        return imu_gyro_est
