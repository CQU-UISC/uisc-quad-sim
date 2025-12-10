import os
import numpy as np

import rclpy
from rclpy.node import Node
from px4ctrl_msgs.msg import Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.quadrotor import Quadrotor
from uisc_quad_sim.controller.se3 import SE3Controller


def example_ref(t):
    """Generates a circular reference trajectory."""
    p = np.array([np.sin(t), np.cos(t), 1.0])
    v = np.array([np.cos(t), -np.sin(t), 0.0])
    yaw = 0.0
    return np.concatenate([p, v, [yaw]])


class SE3Node(Node):
    def __init__(self):
        super().__init__("se3_node")

        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "../configs/xi35.yaml"
            )
            self.quad = Quadrotor.load(config_path)
        except Exception as e:
            self.get_logger().fatal(f"Failed to load quadrotor config: {e}")
            return

        self.ctrl = SE3Controller(SE3Controller.M_PV)

        # --- Service Client Setup ---
        self.sim_step_client = self.create_client(StepSim, "/px4ctrl/step_sim")
        while not self.sim_step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("sim_step service not available, waiting again...")

        self.get_logger().info("Service is available.")

        # --- State and Command Initialization ---
        self.t = 0.0
        self.dt = 0.01  # Timestep for the simulation
        self.state = None
        self.command = Command()
        self.command.type = Command.THRUST_BODYRATE

    def run(self):
        """
        Executes the sequential control loop.
        In each step, it calls the service, waits for the result,
        and then computes the next command.
        """
        request = StepSim.Request()
        request.command_valid = False

        while rclpy.ok():
            future = self.sim_step_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                result: StepSim.Response = future.result()
                self.state = result.state
            else:
                self.get_logger().error(
                    "Exception while calling service. Shutting down."
                )
                break

            self.t += self.dt
            ref = example_ref(self.t)

            if self.state is None:
                self.get_logger().warn("State is None, skipping control calculation.")
                continue

            current_state_array = np.array(self.state.x)

            command_u = self.ctrl.compute_control(
                current_state_array,
                ref,
            )

            request.command.u = command_u.tolist()
            request.command.type = self.command.type
            request.command_valid = True

        self.get_logger().info("Control loop finished.")


if __name__ == "__main__":
    rclpy.init()
    node = SE3Node()
    try:
        if rclpy.ok():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
