import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
from uisc_quad_sim.dynamics.rigidbody import (
    Rigidbody,
    RigidbodyParams,
    RigidbodyState,
    RigidbodyControl,
)

# Create output directory for plots
OUTPUT_DIR = ".tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestRigidbodyDynamics(unittest.TestCase):
    def setUp(self):
        self.params = RigidbodyParams(
            mass=1.0,
            g=9.81,
            J=np.eye(3),  # Identity inertia matrix for easy calculation
            J_inv=np.eye(3),
            drag_coeff=np.zeros(3),  # Ignore drag
        )
        self.body = Rigidbody(self.params)
        # Ensure default drag is cleared to avoid interference with pure dynamics tests
        self.body._disturb.clear()
        self.dt = 0.01

    def test_free_fall_with_plot(self):
        """Test free fall (zero thrust) and plot velocity"""
        state = self.body.initialize()
        control = RigidbodyControl(u=np.array([0.0, 0.0, 0.0, 0.0]))  # 0 Thrust

        time_log = []
        pos_z_log = []
        vel_z_log = []

        steps = 100  # 1 second
        for i in range(steps):
            state = self.body.step(self.dt, state, control)
            time_log.append(i * self.dt)
            pos_z_log.append(
                state.pos[2]
            )  # NED frame, Down is positive usually, or Up positive depends on convention
            vel_z_log.append(state.vel[2])

        # Verify acceleration
        # If NED: acc_z = g = 9.81. If ENU: acc_z = -9.81
        # Code logic: acc_z = ... - g. With 0 thrust, acc_z = -9.81.
        # So velocity should decrease (become more negative).
        self.assertAlmostEqual(state.lin_acc[2], -9.81)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(time_log, pos_z_log, "b-")
        ax1.set_ylabel("Position Z (m)")
        ax1.set_title("Free Fall Simulation")
        ax1.grid(True)

        ax2.plot(time_log, vel_z_log, "r-")
        ax2.set_ylabel("Velocity Z (m/s)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)

        plt.savefig(os.path.join(OUTPUT_DIR, "rigid_free_fall.png"))
        plt.close()

    def test_hover_acceleration(self):
        """Test hover equilibrium"""
        state = self.body.initialize()
        # Need to cancel gravity: Thrust = mass * g = 9.81 N
        control = RigidbodyControl(u=np.array([9.81, 0.0, 0.0, 0.0]))

        state = self.body.step(self.dt, state, control)

        # Z-axis acceleration should be close to 0
        self.assertAlmostEqual(state.lin_acc[2], 0.0, places=5)
        # Velocity should remain 0
        self.assertAlmostEqual(state.vel[2], 0.0, places=5)

    def test_pure_torque_with_plot(self):
        """Test pure torque rotation and plot angular velocity"""
        state = self.body.initialize()
        # Apply X-axis torque 1.0 Nm
        control = RigidbodyControl(u=np.array([0.0, 1.0, 0.0, 0.0]))

        time_log = []
        omega_x_log = []

        steps = 100
        for i in range(steps):
            state = self.body.step(self.dt, state, control)
            time_log.append(i * self.dt)
            omega_x_log.append(state.ang_vel[0])

        # J = eye(3), tau = 1 => alpha = 1
        # omega should increase linearly
        self.assertAlmostEqual(state.ang_acc[0], 1.0, places=5)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time_log, omega_x_log)
        plt.title("Angular Velocity under Constant Torque (Tau_x=1Nm)")
        plt.xlabel("Time (s)")
        plt.ylabel("Omega X (rad/s)")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "rigid_torque_response.png"))
        plt.close()

    def test_quaternion_normalization(self):
        """Test quaternion normalization"""
        state = self.body.initialize()

        # Manually break quaternion normalization
        state.x[6:10] = np.array([10.0, 0.0, 0.0, 0.0])

        control = RigidbodyControl(u=np.zeros(4))
        state = self.body.step(self.dt, state, control)

        q = state.quat
        norm = np.sqrt(np.sum(q**2))
        self.assertAlmostEqual(norm, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
