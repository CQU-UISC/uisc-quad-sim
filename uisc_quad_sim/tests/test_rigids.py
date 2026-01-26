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

OUTPUT_DIR = ".tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestRigidbodyDynamics(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()


if __name__ == "__main__":
    unittest.main()
