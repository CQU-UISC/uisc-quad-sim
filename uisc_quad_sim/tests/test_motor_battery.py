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
from uisc_quad_sim.dynamics.motor import Motors, MotorParams, MotorState, MotorControl

OUTPUT_DIR = ".tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestBatteryMotorCoupling(unittest.TestCase):
    def setUp(self):
        # 1. Setup Battery (High Internal Resistance to exaggerate sag)
        self.bat_params = BatteryParams(
            n_cells=3,
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
        self.battery = Battery(self.bat_params)

        # 2. Setup Motors
        self.motor_params = MotorParams(
            using_battery=True,
            pos_x=np.zeros(4),
            pos_y=np.zeros(4),
            cw=np.ones(4),  # Geometry doesn't matter here
            tau_up=0.05,
            tau_down=0.1,
            k_v=1000.0,  # 1000 RPM/V
            thrust_coeff=1e-6,
            torque_coeff=1e-7,
            i_idle=0.5,
            v_idle=0.0,
            i_max=15.0,  # Max current per motor
            v_max=12.6,  # Max voltage rating
        )
        self.motors = Motors(self.motor_params)
        self.dt = 0.01

    def test_full_throttle_sag(self):
        """
        Scenario:
        1. Idle for 1s.
        2. Full Throttle (100%) for 2s.
        3. Observe Voltage Sag limiting the RPM.
        """
        # Initialize States
        bat_state = self.battery.initialize()
        mot_state = self.motors.initialize()

        # Logging
        times, v_terms, currents, rpms = [], [], [], []

        steps = int(3.0 / self.dt)  # 3 Seconds total

        for i in range(steps):
            t = i * self.dt

            # --- Control Logic ---
            # 0-1s: Idle (0% throttle)
            # 1-3s: Max (100% throttle)
            throttle = 1.0 if t > 1.0 else 0.0
            esc_setpoints = np.full(4, throttle)

            # --- Simulation Step (Coupling Logic) ---

            # 1. Get current draw from motors (based on PREVIOUS state)
            #    Motors calculate current based on their RPM
            mot_currents = self.motors.currents(mot_state)

            # 2. Step Battery
            #    Input: Total motor currents
            #    Output: Updated v_term (Terminal Voltage)
            bat_ctrl = BatteryControl(u=mot_currents)
            bat_state = self.battery.step(self.dt, bat_state, bat_ctrl)

            # 3. Step Motors
            #    Input: v_term from Battery! This is the coupling.
            #           The motor tries to reach: RPM = v_term * throttle * Kv
            mot_ctrl = MotorControl(
                u=np.array(
                    [
                        np.full(4, bat_state.v_term),  # Voltage provided by battery
                        esc_setpoints,  # Throttle command
                    ]
                )
            )
            mot_state = self.motors.step(self.dt, mot_state, mot_ctrl)

            # --- Logging ---
            times.append(t)
            v_terms.append(bat_state.v_term)
            currents.append(np.sum(mot_currents))
            rpms.append(mot_state.rpm[0])  # Log Motor 1

        # --- Verification ---
        # Theoretical Max RPM without sag: 12.6V * 1000Kv = 12600
        # Actual Voltage under load ~ 12.6 - (15A*4 * 0.15R) = 12.6 - 9V = 3.6V (Wow, intense sag!)
        # Let's check the data at t=2.0s
        idx = int(2.0 / self.dt)
        actual_v = v_terms[idx]
        actual_rpm = rpms[idx]

        print(f"\n[Coupling Test] Time: 2.0s")
        print(f"  Voltage under load: {actual_v:.2f} V")
        print(f"  Motor RPM: {actual_rpm:.1f}")
        expected_rpm = actual_v * 1000.0
        print(f"  Expected RPM based on Battery Voltage: {expected_rpm:.1f}")
        # Verify the coupling: RPM should roughly equal V_term * Kv (ignoring lag)
        # Allow 5% margin for tau lag and numerical integration
        self.assertTrue(
            abs(actual_rpm - expected_rpm) / expected_rpm < 0.05,
            "Motor RPM did not track Battery Voltage!",
        )

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Voltage and Current
        ax1.set_title("Battery-Motor Coupling: Voltage Sag Effect")
        ax1.plot(times, v_terms, "r-", label="Battery Voltage (V)")
        ax1.set_ylabel("Voltage (V)", color="r")
        ax1.tick_params(axis="y", labelcolor="r")
        ax1.grid(True)

        ax1_b = ax1.twinx()
        ax1_b.plot(times, currents, "b--", label="Total Current (A)")
        ax1_b.set_ylabel("Current (A)", color="b")
        ax1_b.tick_params(axis="y", labelcolor="b")

        # Plot RPM
        ax2.plot(times, rpms, "k-", label="Motor RPM")
        ax2.set_ylabel("Speed (RPM)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)

        # Draw theoretical max line
        ax2.axhline(
            12.6 * 1000, color="gray", linestyle="--", label="Theoretical Max (No Load)"
        )
        ax2.legend()

        plt.savefig(os.path.join(OUTPUT_DIR, "coupling_sag_rpm.png"))
        plt.close()


if __name__ == "__main__":
    unittest.main()
