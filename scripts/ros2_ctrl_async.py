import sys
import termios
import threading
import tty
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4ctrl_msgs.msg import Command, State
from uisc_quad_sim.controller.se3 import SE3Controller
from uisc_quad_sim.dynamics.rigidbody import RigidbodyState
from uisc_quad_sim.utils.quaternion import from_euler
from pynput import keyboard


class KeyboardInput:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        with self.lock:
            self.pressed_keys.add(k)

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        with self.lock:
            if k in self.pressed_keys:
                self.pressed_keys.remove(k)

    def is_pressed(self, key_char):
        with self.lock:
            return key_char in self.pressed_keys

    def stop(self):
        self.listener.stop()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class SE3Node(Node):
    def __init__(self):
        super().__init__("se3_node")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.ctrl = SE3Controller()
        self.pub_cmd = self.create_publisher(Command, "/px4ctrl/sim/cmd", 10)
        self.sub_state = self.create_subscription(
            State, "/px4ctrl/sim/state", self.state_callback, qos_profile
        )

        self.keyboard = KeyboardInput()

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0
        self.current_state = None
        self.target_state = RigidbodyState(x=np.zeros(RigidbodyState.shape))

        self.v_body_filtered = np.zeros(3)
        self.vel_smoothing = 0.9

        self.step_vel_xy = 1.5
        self.step_vel_z = 1.5
        self.step_yaw = 2.0
        self.dt = 0.02

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            "Controller started. Use WS(x) AD(y) RF(z) QE(yaw) to control."
        )

    def state_callback(self, msg: State):
        rb = RigidbodyState(x=np.zeros(RigidbodyState.shape))
        rb.x[:13] = np.array(msg.x)
        self.current_state = rb

    def control_loop(self):
        if self.current_state is None:
            return

        v_raw = np.zeros(3)
        yaw_rate = 0.0

        if self.keyboard.is_pressed("w"):
            v_raw[0] = self.step_vel_xy
        if self.keyboard.is_pressed("s"):
            v_raw[0] = -self.step_vel_xy
        if self.keyboard.is_pressed("a"):
            v_raw[1] = self.step_vel_xy
        if self.keyboard.is_pressed("d"):
            v_raw[1] = -self.step_vel_xy
        if self.keyboard.is_pressed("r"):
            v_raw[2] = self.step_vel_z
        if self.keyboard.is_pressed("f"):
            v_raw[2] = -self.step_vel_z
        if self.keyboard.is_pressed("q"):
            yaw_rate = self.step_yaw
        if self.keyboard.is_pressed("e"):
            yaw_rate = -self.step_yaw

        self.v_body_filtered = (
            self.vel_smoothing * self.v_body_filtered + (1 - self.vel_smoothing) * v_raw
        )

        self.target_yaw += yaw_rate * self.dt

        cy = np.cos(self.target_yaw)
        sy = np.sin(self.target_yaw)
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        v_world = R_z @ self.v_body_filtered

        self.target_pos += v_world * self.dt

        self.target_state.pos = self.target_pos
        self.target_state.vel = v_world
        self.target_state.quat = from_euler(0, 0, self.target_yaw)

        command_u = self.ctrl.compute_control(self.current_state, self.target_state)

        cmd_msg = Command()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.type = Command.THRUST_BODYRATE
        cmd_msg.u = command_u.u.tolist()
        self.pub_cmd.publish(cmd_msg)

    def destroy_node(self):
        self.keyboard.stop()
        super().destroy_node()


if __name__ == "__main__":
    rclpy.init()
    node = SE3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
