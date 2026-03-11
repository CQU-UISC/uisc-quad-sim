import os
import sys
import loguru
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from px4ctrl_msgs.msg import State, Command
from px4ctrl_msgs.srv import StepSim
from uisc_quad_sim.simulations import QuadSim, QuadParams
from uisc_quad_sim.simulations.quadrotor import ControlCommand, ControlMode
from uisc_quad_sim.visualize.vis import DroneVisualizer


class SimNode(Node):
    def __init__(self):
        super().__init__("sim", namespace="px4ctrl")
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
        self.odom_publisher_ = self.create_publisher(Odometry, "~/odom", 10)
        self.publish_timer_ = self.create_timer(0.01, self.timer_callback)
        self.sensor_sub = self.create_subscription(
            Image, "~/depth_image", self.depth_image_callback, 10
        )

        # jit-warmup
        self.quad_sim.warm_up()
        self.quad_sim.reset()
        self.get_logger().info("Simulation node initialized.")

    def timer_callback(self):
        self.publish_state_callback()
        self.publish_odom_callback()

    def publish_odom_callback(self):
        msg = Odometry()
        sim_time_sec = self.quad_sim.t
        sec = int(sim_time_sec)
        nanosec = int((sim_time_sec - sec) * 1e9)
        msg.header.stamp = Time(seconds=sec, nanoseconds=nanosec).to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = self.quad_sim.rb_state.pos[0]
        msg.pose.pose.position.y = self.quad_sim.rb_state.pos[1]
        msg.pose.pose.position.z = self.quad_sim.rb_state.pos[2]
        msg.pose.pose.orientation.x = self.quad_sim.rb_state.quat[1]
        msg.pose.pose.orientation.y = self.quad_sim.rb_state.quat[2]
        msg.pose.pose.orientation.z = self.quad_sim.rb_state.quat[3]
        msg.pose.pose.orientation.w = self.quad_sim.rb_state.quat[0]
        msg.twist.twist.linear.x = self.quad_sim.rb_state.vel[0]
        msg.twist.twist.linear.y = self.quad_sim.rb_state.vel[1]
        msg.twist.twist.linear.z = self.quad_sim.rb_state.vel[2]
        # Note: Angular velocity in body frame
        msg.twist.twist.angular.x = self.quad_sim.rb_state.ang_vel[0]
        msg.twist.twist.angular.y = self.quad_sim.rb_state.ang_vel[1]
        msg.twist.twist.angular.z = self.quad_sim.rb_state.ang_vel[2]
        self.odom_publisher_.publish(msg)

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

    def depth_image_callback(self, msg: Image):
        height = msg.height
        width = msg.width
        depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape((height, width))
        self.quad_vis.log_depth_image(depth_image)
        return

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
        self.get_logger().debug(f"Received command: {command.type}, u: {command.u}")
        cmd = ControlCommand(type=self.type_map[command.type], u=np.array(command.u))
        rb_state = self.quad_sim.step(cmd)
        self.quad_vis.step(self.quad_sim.t)
        self.quad_vis.log_rigidbody_states(rb_state)
        self.quad_vis.log_battery_states(self.quad_sim.batt_state)
        self.quad_vis.log_motor_states(self.quad_sim.motor_state)
        self.quad_vis.log_controls(command.u)
        self.get_logger().debug(f"Post-step state estimate: {rb_state.x[:13]}")
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
