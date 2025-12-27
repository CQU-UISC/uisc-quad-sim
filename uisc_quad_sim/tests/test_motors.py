import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
from uisc_quad_sim.dynamics.motor import Motors, MotorParams, MotorState, MotorControl

# Create output directory for plots
OUTPUT_DIR = ".tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestMotorDynamics(unittest.TestCase):
    def setUp(self):
        # Simulate typical brushless motor parameters
        self.params = MotorParams(
            using_battery=True,
            pos_x=np.array([0.1, 0.1, -0.1, -0.1]),
            pos_y=np.array([-0.1, 0.1, 0.1, -0.1]),
            cw=np.array([1, -1, 1, -1]),
            tau_up=0.1,  # Time constant up
            tau_down=0.2,  # Time constant down (slower)
            k_v=1000.0,  # 1000 RPM/V
            thrust_coeff=1e-6,  # 1e-6 N / (rpm^2)
            torque_coeff=1e-7,
            i_idle=0.5,
            v_idle=0.0,
            i_max=20.0,
            v_max=12.0,
        )
        self.motors = Motors(self.params)
        self.dt = 0.01

    def test_step_response_with_plot(self):
        """Test motor step response and visualize"""
        state = self.motors.initialize()

        # Apply full throttle (12V)
        control = MotorControl(
            u=np.array(
                [
                    [12.0, 12.0, 12.0, 12.0],  # v_term
                    [1.0, 1.0, 1.0, 1.0],  # throttle 100%
                ]
            )
        )

        target_rpm = 12.0 * 1000.0  # 12000 RPM

        # Logging
        time_log = []
        rpm_log = []

        # Simulate 0.5 seconds (5 tau_up)
        steps = int(0.5 / self.dt)

        for i in range(steps):
            state = self.motors.step(self.dt, state, control)
            time_log.append(i * self.dt)
            rpm_log.append(state.rpm[0])  # Log Motor 1

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time_log, rpm_log, label="Actual RPM")
        plt.axhline(y=target_rpm, color="r", linestyle="--", label="Target RPM")
        plt.title(f"Motor Step Response (tau_up={self.params.tau_up}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (RPM)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "motor_step_response.png"))
        plt.close()

        # Check basic physics
        self.assertTrue(np.all(state.rpm > 0))
        # After 5 time constants, should be > 99%
        self.assertGreater(state.rpm[0], target_rpm * 0.99)

    def test_tau_asymmetry_with_plot(self):
        """Test asymmetry between acceleration and deceleration"""
        state = self.motors.initialize()

        time_log = []
        rpm_log = []

        # 1. Acceleration phase (0 to 0.5s)
        control_up = MotorControl(u=np.array([[10, 10, 10, 10], [1, 1, 1, 1]]))
        steps = int(0.5 / 0.01)
        for i in range(steps):
            state = self.motors.step(0.01, state, control_up)
            time_log.append(len(time_log) * 0.01)
            rpm_log.append(state.rpm[0])

        # 2. Deceleration phase (0.5s to 1.5s)
        control_down = MotorControl(u=np.array([[10, 10, 10, 10], [0, 0, 0, 0]]))
        steps = int(1.0 / 0.01)
        for i in range(steps):
            state = self.motors.step(0.01, state, control_down)
            time_log.append(len(time_log) * 0.01)
            rpm_log.append(state.rpm[0])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time_log, rpm_log)
        plt.title("Motor Acceleration vs Deceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (RPM)")
        plt.axvline(x=0.5, color="k", linestyle="--", label="Input Change")
        plt.text(0.1, 5000, f"Tau Up: {self.params.tau_up}s")
        plt.text(0.8, 5000, f"Tau Down: {self.params.tau_down}s")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "motor_asymmetry.png"))
        plt.close()

    def test_static_characteristics_with_plot(self):
        """
        Test and plot static characteristics:
        1. Thrust vs RPM
        2. Current vs RPM
        3. Check Normalization logic
        """
        # Create a sweep of RPMs from 0 to Max
        max_rpm_val = self.motors.max_rpm()
        min_rpm_val = self.motors.min_rpm()
        rpms = np.linspace(0, max_rpm_val, 100)

        thrust_log = []
        current_log = []
        norm_rpm_log = []

        state = self.motors.initialize()

        for r in rpms:
            # Manually set RPM to probe the static maps
            state.rpm[:] = r

            # 1. Calculate Normalization
            norm = self.motors.normalized_rpm(state)[0]  # Check motor 0
            norm_rpm_log.append(norm)

            # 2. Calculate Thrust (Single Motor)
            # wrench returns [F_x, F_y, F_z, T_x, T_y, T_z]
            # We want single motor thrust to verify coeff
            # But wrench sums them up.
            # So we check the logic: Total Thrust Z = 4 * coeff * rpm^2
            w = self.motors.wrench(state)
            thrust_single = w[2] / 4.0  # Average/Single thrust
            thrust_log.append(thrust_single)

            # 3. Calculate Current
            i = self.motors.currents(state)[0]
            current_log.append(i)

        # --- Assertions ---
        # Check Normalization at boundaries
        # At 0 RPM (if v_idle=0), norm should be 0
        self.assertAlmostEqual(norm_rpm_log[0], 0.0)
        # At Max RPM, norm should be 1.0
        self.assertAlmostEqual(norm_rpm_log[-1], 1.0)

        # Check Current at Max RPM
        self.assertAlmostEqual(current_log[-1], self.params.i_max)

        # Check Thrust at Max RPM: T = coeff * rpm^2
        expected_max_thrust = self.params.thrust_coeff * (max_rpm_val**2)
        self.assertAlmostEqual(thrust_log[-1], expected_max_thrust)

        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:blue"
        ax1.set_xlabel("Motor Speed (RPM)")
        ax1.set_ylabel("Thrust per Motor (N)", color=color)
        ax1.plot(rpms, thrust_log, color=color, linewidth=2, label="Thrust")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True)

        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Current (A)", color=color)
        ax2.plot(
            rpms, current_log, color=color, linewidth=2, linestyle="--", label="Current"
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Motor Static Characteristics")
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "motor_static_curves.png"))
        plt.close()

    def test_wrench_generation(self):
        """Test force and torque generation"""
        state = self.motors.initialize()
        state.rpm = np.array([1000.0, 1000.0, 1000.0, 1000.0])

        w = self.motors.wrench(state)
        # w shape should be (6,)
        self.assertEqual(w.shape, (6,))

        # Thrust Z should be positive (assuming thrust up)
        thrust_z = w[2]
        expected_thrust = 4 * (1e-6 * 1000.0**2)
        self.assertAlmostEqual(thrust_z, expected_thrust)


if __name__ == "__main__":
    unittest.main()
