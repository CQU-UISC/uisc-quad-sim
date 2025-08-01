import os
import sys
import loguru
import numpy as np
import rclpy
from rclpy.node import Node
from px4ctrl_msgs.msg import State,Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.simulations import QuadSim,QuadSimParams
from uisc_quad_sim.visualize.vis import DroneVisualizer

class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')
        self.sim_step_client = self.create_service(StepSim, '~/step_sim', self.sim_step_callback)
        self.sim_params = QuadSimParams.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/ctbm.yaml")
        )
        self.quad_sim = QuadSim(self.sim_params)
        self.quad_vis = DroneVisualizer()
        self.type_map = {
            Command.THRUST_BODYRATE: self.sim_params.CTBR,
            Command.THRUST_TORQUE: self.sim_params.CTBM,
            Command.ROTORS_FORCE: self.sim_params.SRT
        }

    def sim_step_callback(self, request: StepSim.Request, response: StepSim.Response):
        command = request.command
        response.state.header.stamp = self.get_clock().now().to_msg()
        if not request.command_valid:
            self.get_logger().info("Received empty command, returning current state")
            response.state.state = self.quad_sim.estimate()
            return response
        if command.type not in self.type_map:
            self.get_logger().error(f"Unsupported command type: {command.type}")
            return response
        if self.type_map[command.type] != QuadSimParams.control_modes[self.sim_params.mode]:
            self.get_logger().error(message=f"Command type {command.type} does not match simulation mode {self.sim_params.mode}")
            return response
        u = np.array(command.u)
        x_est = self.quad_sim.step(u)
        x_gt = self.quad_sim.estimate(gt=True)
        self.quad_vis.log_state(
            self.quad_sim.t,
            x_gt,
            self.quad_sim.motor_speed,
            u
        )
        response.state.state = x_est
        return response
    
if __name__ == '__main__':
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level='INFO')
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()