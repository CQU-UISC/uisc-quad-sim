import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
from uisc_quad_sim.dynamics.battery import (
    Battery,
    BatteryParams,
    BatteryState,
    BatteryControl,
)

# Create output directory for plots
OUTPUT_DIR = ".tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestBattery(unittest.TestCase):
    def setUp(self):
        # Create standard parameters for testing
        self.params = BatteryParams(
            n_cells=6,
            capacity_mah=1000.0,  # 1Ah for easy calculation
            i_avionics=1.0,  # 1A static current
            r_internal=0.15,  # Internal resistance 150mOhm,
            v_cuttoff=2.5,
            v_full=4.2,
            soc_ocv_coeffs=np.array(
                [
                    2.59015588,
                    7.73913782,
                    -30.47793475,
                    62.98798581,
                    -60.57264964,
                    21.87660915,
                ]
            ),
        )
        self.dyn = Battery(self.params)
        self.dt = 0.1

    def test_initialization(self):
        """Test initialization state"""
        state = self.dyn.initialize()
        self.assertAlmostEqual(state.soc, 1.0)
        # 6S * 4.2V = 25.2V
        self.assertAlmostEqual(state.v_ocv, 6 * 4.2)
        self.assertAlmostEqual(state.v_term, 6 * 4.2)  # No load initially

    def test_step_response(self):
        """Test step response of the battery dynamics, untill SOC drops to 0"""
        state = self.dyn.initialize()

        # Apply 9A motor current + 1A avionics = 10A total
        control = BatteryControl(u=np.array([2.25, 2.25, 2.25, 2.25]))

        steps = 6000  # Simulate for 6000 steps (10 minutes)
        ocv_log = []
        term_log = []
        soc_log = []
        for _ in range(steps):
            state = self.dyn.step(self.dt, state, control)
            ocv_log.append(state.v_ocv)
            term_log.append(state.v_term)
            soc_log.append(state.soc)
            if state.soc <= 0.0:
                break
        self.assertLessEqual(state.soc, 0.0)
        plt.figure(figsize=(4, 9))
        time_axis = np.arange(len(soc_log)) * self.dt
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, soc_log, label="SOC")
        plt.ylabel("State of Charge (SOC)")
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, ocv_log, label="Open Circuit Voltage", color="g")
        plt.ylabel("V_ocv (V)")
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, term_log, label="Terminal Voltage", color="r")
        plt.ylabel("V_term (V)")
        plt.xlabel("Time (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "battery_step_response.png"))
        plt.close()

    def test_drain_logic_with_plot(self):
        """Test power drain logic and plot the results"""
        state = self.dyn.initialize()

        # Apply 9A motor current + 1A avionics = 10A total
        control = BatteryControl(u=np.array([2.25, 2.25, 2.25, 2.25]))

        steps = 100

        time_log = []
        soc_log = []
        voltage_log = []

        for i in range(steps):
            state = self.dyn.step(self.dt, state, control)
            time_log.append(i * self.dt)
            soc_log.append(state.soc)
            voltage_log.append(state.v_term)

        expected_drop = (10.0 / 3600.0) * (steps * self.dt)
        expected_soc = 1.0 - expected_drop

        self.assertAlmostEqual(state.soc, expected_soc, places=5)

    def test_voltage_sag_high_load(self):
        """
        Test Voltage Sag effect under high current load.
        Scenario: Hover (Low) -> Punch Out (Max 40A) -> Hover (Low)
        """
        state = self.dyn.initialize()

        # Simulation parameters
        t_total = 10.0
        steps = int(t_total / self.dt)

        # Logs
        time_log = []
        v_ocv_log = []
        v_term_log = []
        current_log = []

        # Define Loads
        # 1. Hover: ~2A per motor -> 8A total
        u_hover = np.array([2.0, 2.0, 2.0, 2.0])
        # 2. Punch Out: 10A per motor -> 40A total (Requested by user)
        u_max = np.array([10.0, 10.0, 10.0, 10.0])

        for i in range(steps):
            t = i * self.dt

            # Create a pulse profile
            # 0-2s: Hover
            # 2-4s: Max Load
            # 4-10s: Hover
            if 2.0 <= t < 4.0:
                u = u_max
            else:
                u = u_hover

            control = BatteryControl(u=u)
            state = self.dyn.step(self.dt, state, control)

            time_log.append(t)
            v_ocv_log.append(state.v_ocv)
            v_term_log.append(state.v_term)
            current_log.append(np.sum(u) + self.params.i_avionics)

        # --- Assertions ---
        # Find indices for the high load period
        high_load_indices = [i for i, t in enumerate(time_log) if 2.5 < t < 3.5]
        low_load_indices = [i for i, t in enumerate(time_log) if 0.5 < t < 1.5]

        # Calculate average Voltage Sag during high load
        # Sag = V_ocv - V_term = I * R
        # At 40.5A, Sag should be ~ 40.5 * 0.15 = 6.075 V
        idx = high_load_indices[0]
        sag = v_ocv_log[idx] - v_term_log[idx]
        current_at_sag = current_log[idx]
        expected_sag = current_at_sag * self.params.r_internal

        self.assertAlmostEqual(sag, expected_sag, places=5)

        # Verify that V_term recovered after load was removed (voltage bounce back)
        # Compare V_term at t=3.5s (load) vs t=4.5s (no load)
        idx_load = int(3.5 / self.dt)
        idx_recover = int(4.5 / self.dt)
        self.assertLess(v_term_log[idx_load], v_term_log[idx_recover])

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Plot 1: Voltages
        ax1.plot(time_log, v_ocv_log, "g--", label="OCV (No Load Voltage)", alpha=0.7)
        ax1.plot(
            time_log, v_term_log, "r-", label="Terminal Voltage (Actual)", linewidth=2
        )
        ax1.set_ylabel("Voltage (V)")
        ax1.set_title("Battery Voltage Sag Test")
        ax1.legend()
        ax1.grid(True)

        # Annotate the sag
        sag_time = 3.0
        sag_idx = int(sag_time / self.dt)
        ax1.annotate(
            "Voltage Sag",
            xy=(sag_time, v_term_log[sag_idx]),
            xytext=(sag_time, v_term_log[sag_idx] - 2),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        # Plot 2: Current
        ax2.plot(time_log, current_log, "b-", label="Total Current")
        ax2.set_ylabel("Current (A)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)
        ax2.legend()

        plt.savefig(os.path.join(OUTPUT_DIR, "battery_voltage_sag.png"))
        plt.close()

    def test_soc_clipping(self):
        """Test that SOC does not drop below 0"""
        state = self.dyn.initialize()
        state.soc = 0.001  # Almost empty

        control = BatteryControl(u=np.array([100.0, 0, 0, 0]))  # Huge current

        # Run for long enough
        for _ in range(100):
            state = self.dyn.step(1.0, state, control)

        self.assertEqual(state.soc, 0.0)

    def test_soc_ocv_curve(self):
        """Test the SOC to OCV mapping function"""
        state = self.dyn.initialize()

        soc_values = [1.0, 0.8, 0.5, 0.2, 0.0]
        expected_ocv_per_cell = [
            sum(c * (soc**i) for i, c in enumerate(self.params.soc_ocv_coeffs))
            for soc in soc_values
        ]
        expected_ocv = [self.params.n_cells * v for v in expected_ocv_per_cell]

        for soc, expected_v in zip(soc_values, expected_ocv):
            state.soc = soc
            v_ocv = self.dyn._v_ocv(state)
            self.assertAlmostEqual(v_ocv, expected_v, places=5)
        plt.figure(figsize=(8, 5))
        soc_range = np.linspace(0, 1, 100)
        ocv_values = [
            self.dyn._v_ocv(BatteryState(x=np.array([soc]), v_ocv=0, v_term=0))
            for soc in soc_range
        ]
        plt.plot(soc_range, ocv_values, label="SOC-OCV Curve")
        plt.xlabel("State of Charge (SOC)")
        plt.ylabel("Open Circuit Voltage (V)")
        plt.title("Battery SOC to OCV Characteristic")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "battery_soc_ocv_curve.png"))
        plt.close()


if __name__ == "__main__":
    unittest.main()
