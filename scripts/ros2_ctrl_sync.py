import numpy as np

import rclpy
from rclpy.node import Node
from px4ctrl_msgs.msg import Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.controller.se3 import SE3Controller
from uisc_quad_sim.dynamics.rigidbody import RigidbodyState
from uisc_quad_sim.utils.quaternion import from_euler


def example_ref(t, r, omega) -> RigidbodyState:
    x = np.zeros(RigidbodyState.shape)
    rg_state = RigidbodyState(x=x)
    rg_state.pos = np.array([r * np.sin(omega * t), r * np.cos(omega * t), 1])
    rg_state.vel = np.array(
        [r * omega * np.cos(omega * t), -r * omega * np.sin(omega * t), 0]
    )
    rg_state.lin_acc = np.array(
        [-r * omega**2 * np.sin(omega * t), -r * omega**2 * np.cos(omega * t), 0]
    )
    # let head to the center(0,0)
    yaw = np.arctan2(-rg_state.pos[1], -rg_state.pos[0])
    rg_state.quat = from_euler(0, 0, yaw)
    return rg_state


class SE3Node(Node):
    def __init__(self):
        super().__init__("se3_node")
        self.ctrl = SE3Controller()
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
            ref_state = example_ref(self.t)

            if self.state is None:
                self.get_logger().warn("State is None, skipping control calculation.")
                continue

            current_state_array = np.array(self.state.x)
            rb_state = RigidbodyState(x=np.zeros(RigidbodyState.shape))
            rb_state.x[:13] = current_state_array
            command_u = self.ctrl.compute_control(
                rb_state,
                ref_state,
            )

            request.command.u = command_u.u.tolist()
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
