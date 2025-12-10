import os
import sys
import loguru
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from px4ctrl_msgs.msg import State, Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.simulations import QuadSim, QuadSimParams
from uisc_quad_sim.simulations.quadrotor_sim import ControlCommand, ControlMode
from uisc_quad_sim.visualize.vis import DroneVisualizer


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node", namespace="px4ctrl")
        self.sim_step_service = self.create_service(
            StepSim, "step_sim", self.sim_step_callback
        )
        self.sim_params = QuadSimParams.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../configs/ctbr.yaml"
            )
        )
        self.quad_sim = QuadSim(self.sim_params)
        self.quad_vis = DroneVisualizer()
        self.type_map = {
            Command.THRUST_BODYRATE: ControlMode.CTBR,
            Command.THRUST_TORQUE: ControlMode.CTBM,
            Command.ROTORS_FORCE: ControlMode.SRT,
        }

        self.state_publisher_ = self.create_publisher(State, "~/state", 10)
        self.publish_timer_ = self.create_timer(0.01, self.publish_state_callback)

    def publish_state_callback(self):
        msg = State()
        sim_time_sec = self.quad_sim.t
        sec = int(sim_time_sec)
        nanosec = int((sim_time_sec - sec) * 1e9)
        msg.header.stamp = Time(seconds=sec, nanoseconds=nanosec).to_msg()
        msg.header.frame_id = "world"
        msg.x = self.quad_sim.estimate().tolist()
        msg.rotors = self.quad_sim.motor_speed.tolist()
        self.state_publisher_.publish(msg)

    def sim_step_callback(self, request: StepSim.Request, response: StepSim.Response):
        command = request.command
        response.state.header.stamp = self.get_clock().now().to_msg()
        if not request.command_valid:
            self.get_logger().info("Received empty command, returning current state")
            response.state.x = self.quad_sim.estimate().tolist()
            return response
        if command.type not in self.type_map:
            self.get_logger().error(f"Unsupported command type: {command.type}")
            return response
        self.get_logger().info(f"Received command: {command.type}, u: {command.u}")
        cmd = ControlCommand(type=self.type_map[command.type], u=np.array(command.u))
        x_est = self.quad_sim.step(cmd)
        x_gt = self.quad_sim.estimate(gt=True)
        self.quad_vis.log_state(self.quad_sim.t, x_gt, self.quad_sim.motor_speed, cmd.u)
        response.state.x = x_est.tolist()
        return response


if __name__ == "__main__":
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level="INFO")
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
