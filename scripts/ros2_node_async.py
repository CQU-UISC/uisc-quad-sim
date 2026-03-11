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
from uisc_quad_sim.simulations import QuadSim, QuadParams
from uisc_quad_sim.simulations.quadrotor import ControlCommand, ControlMode
from uisc_quad_sim.visualize.vis import DroneVisualizer


class SimNode(Node):
    def __init__(self):
        super().__init__("sim", namespace="px4ctrl")

        # Simulation Parameters
        self.sim_params = QuadParams(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../configs/uisc_xi35.yaml"
            )
        )
        self.quad_sim = QuadSim(self.sim_params)
        self.quad_vis = DroneVisualizer()

        # Control mapping
        self.type_map = {
            Command.THRUST_BODYRATE: ControlMode.CTBR,
            Command.THRUST_TORQUE: ControlMode.CTBM,
            Command.ROTORS_FORCE: ControlMode.SRT,
        }

        # Communication
        self.state_publisher_ = self.create_publisher(State, "~/state", 10)
        self.odom_publisher_ = self.create_publisher(Odometry, "~/odom", 10)
        self.cmd_subscription_ = self.create_subscription(
            Command, "~/cmd", self.cmd_callback, 10
        )
        self.sensor_sub = self.create_subscription(
            Image, "~/depth_image", self.depth_image_callback, 10
        )

        # Simulation Loop (100Hz)
        self.sim_timer = self.create_timer(0.01, self.sim_loop_callback)

        # Control State
        self.last_cmd_time = self.get_clock().now()
        self.current_cmd = self.get_hover_command()  # Default to hover
        self.timeout_duration = 0.02  # 20ms (50Hz)

        # jit-warmup
        self.quad_sim.warm_up()
        self.quad_sim.reset()
        self.get_logger().info("Simulation node initialized (Asynchronous Mode).")

    def get_hover_command(self):
        # Using CTBR mode (Collective Thrust + Body Rates)
        u = np.array([9.81, 0.0, 0.0, 0.0])
        return ControlCommand(type=ControlMode.CTBR, u=u)

    def cmd_callback(self, msg: Command):
        self.last_cmd_time = self.get_clock().now()

        if msg.type not in self.type_map:
            self.get_logger().error(f"Unsupported command type: {msg.type}")
            return

        self.current_cmd = ControlCommand(
            type=self.type_map[msg.type], u=np.array(msg.u)
        )

    def sim_loop_callback(self):
        # Check for command timeout
        now = self.get_clock().now()
        dt_last_cmd = (now - self.last_cmd_time).nanoseconds * 1e-9

        if dt_last_cmd > self.timeout_duration:
            # Timeout: Enter Hover mode
            self.current_cmd = self.get_hover_command()

        # Step Simulation
        rb_state = self.quad_sim.step(self.current_cmd)

        # Visualize
        self.quad_vis.step(self.quad_sim.t)
        self.quad_vis.log_rigidbody_states(rb_state)
        self.quad_vis.log_battery_states(self.quad_sim.batt_state)
        self.quad_vis.log_motor_states(self.quad_sim.motor_state)
        self.quad_vis.log_controls(self.current_cmd.u)

        # Publish Data
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


if __name__ == "__main__":
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level="INFO")
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
