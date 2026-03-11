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
    def setUp(self) -> None:
        return super().setUp()


if __name__ == "__main__":
    unittest.main()
