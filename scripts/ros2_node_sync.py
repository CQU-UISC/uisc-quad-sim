import os
import sys
import loguru
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from px4ctrl_msgs.msg import State, Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.simulations import QuadSim, QuadParams
from uisc_quad_sim.simulations.quadrotor import ControlCommand, ControlMode
from uisc_quad_sim.visualize.vis import DroneVisualizer


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node", namespace="px4ctrl")
        self.sim_step_service = self.create_service(
            StepSim, "step_sim", self.sim_step_callback
        )
        self.sim_params = QuadParams(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../configs/uisc_xi35.yaml"
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
        msg.x = self.quad_sim.rb_state.x[:13].tolist()
        msg.rotors = self.quad_sim.motor_state.rpm.tolist()
        self.state_publisher_.publish(msg)

    def sim_step_callback(self, request: StepSim.Request, response: StepSim.Response):
        command = request.command
        response.state.header.stamp = self.get_clock().now().to_msg()
        if not request.command_valid:
            self.get_logger().info("Received empty command, returning current state")
            response.state.x = self.quad_sim.rb_state.x[:13].tolist()
            return response
        if command.type not in self.type_map:
            self.get_logger().error(f"Unsupported command type: {command.type}")
            return response
        self.get_logger().info(f"Received command: {command.type}, u: {command.u}")
        cmd = ControlCommand(type=self.type_map[command.type], u=np.array(command.u))
        rb_state = self.quad_sim.step(cmd)
        self.quad_vis.step(self.quad_sim.t)
        self.quad_vis.log_rigidbody_states(rb_state)
        self.quad_vis.log_battery_states(self.quad_sim.batt_state)
        self.quad_vis.log_motor_states(self.quad_sim.motor_state)
        self.quad_vis.log_controls(command.u)
        self.get_logger().info(f"Post-step state estimate: {rb_state.x[:13]}")
        response.state.x = rb_state.x[:13].tolist()
        return response


if __name__ == "__main__":
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level="INFO")
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
