import os
from typing import Callable, Literal, Optional, List
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.spatial.transform import Rotation
from uuid import uuid4
from ..dynamics import MotorState, RigidbodyState, BatteryState


class DroneVisualizer:
    def __init__(self):
        """Initialize the drone visualizer"""
        base_dir = os.path.dirname(__file__)
        self._drone_asset_path = os.path.join(
            base_dir, "../../assets/hummingbird_s.glb"
        )
        self._gate_assets_path = {
            "l": os.path.join(base_dir, "../../assets/gate_l.glb"),  # size 2x2m
            "m": os.path.join(base_dir, "../../assets/gate_m.glb"),  # size 1x1m
            "s": os.path.join(base_dir, "../../assets/gate_s.glb"),  # size 0.5x0.5m
        }
        self.reset()

    def reset(self):
        """Reset the visualizer"""
        rr.init("uisc_quad_sim.render", recording_id=uuid4(), spawn=True)
        self.step(0)  # Ensure time starts at 0
        self._setup_blueprint()
        self.set_quad_transform(
            translation=np.zeros(3),
            rotation_xyzw=np.array([0, 0, 0, 1]),
        )
        self.set_camera_transform(
            translation=np.zeros(3),
            rotation_xyzw=Rotation.from_euler(
                "xyz", [np.pi / 2, np.pi, np.pi / 2]
            ).as_quat(),
        )
        self.set_quad_mesh_transform(
            translation=np.zeros(3),
            rotation_xyzw=Rotation.from_euler("xyz", [0, 0, np.pi / 4]).as_quat(),
        )
        rr.log(
            "world/drone/baselink/axis",
            rr.TransformAxes3D(axis_length=0.4),
            static=True,
        )
        if os.path.exists(self._drone_asset_path):
            rr.log(
                "world/drone/baselink/mesh/stl",
                rr.Asset3D(path=self._drone_asset_path),
                static=True,
            )
        else:
            raise FileNotFoundError(
                f"Drone asset not found at {self._drone_asset_path}"
            )
        self._gate_assets = {}
        for size, path in self._gate_assets_path.items():
            if os.path.exists(path):
                self._gate_assets[size] = rr.Asset3D(path=path)
            else:
                raise FileNotFoundError(f"Gate asset not found at {path}")

    def _setup_blueprint(self):
        """Configure the visualization view layout using Rerun 0.22+ API"""
        rigid_views = [
            ("Position", "/rigid/position"),
            ("Velocity", "/rigid/velocity"),
            ("Angular Velocity", "/rigid/angular_velocity"),
            ("Linear Acc", "/rigid/lin_acc"),
            ("Forces", "/rigid/forces"),
            ("Torques", "/rigid/torques"),
            ("Orientation (Euler)", "/rigid/euler"),
        ]

        quad_views = [
            ("Controls", "/controls/"),
            ("Motor RPM", "/motors/rpm"),
            ("Battery SoC", "/battery/soc"),
            ("Voltage", "/voltage/"),
            ("Current", "/current/"),
        ]

        rigid = [
            rrb.TimeSeriesView(origin=path, name=name) for name, path in rigid_views
        ]

        quad = [rrb.TimeSeriesView(origin=path, name=name) for name, path in quad_views]

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    name="3D View",
                ),
                rrb.Vertical(
                    rrb.Tabs(*rigid),
                    rrb.Tabs(*quad),
                    rrb.Tabs(
                        rrb.Spatial2DView(
                            origin="/world/drone/baselink/camera", name="2D View"
                        ),
                        rrb.TextLogView(origin="/logs", name="Logs"),
                        rrb.TimeSeriesView(origin="/metrics", name="Metrics"),
                    ),
                ),
                column_shares=[2, 1],
            ),
            rrb.SelectionPanel(state="hidden"),
            rrb.TimePanel(state="collapsed", timeline="sec"),
        )
        rr.send_blueprint(blueprint)

    def step(self, time_s: float):
        """Advance the visualizer by one step"""
        rr.set_time("sec", duration=time_s)

    def log_motor_states(self, motor_states: MotorState):
        """Log motor states efficiently"""
        size_of_motor = len(motor_states.rpm)
        for i in range(size_of_motor):
            rpm = motor_states.rpm[i]
            c = motor_states.i[i]
            rr.log(f"motors/rpm/motor_{i+1}", rr.Scalars(rpm))
            rr.log(f"current/motor_{i+1}", rr.Scalars(c))
        rr.log("current/motor_total", rr.Scalars(np.sum(motor_states.i)))

    def log_battery_states(self, battery_states: BatteryState):
        rr.log("battery/soc", rr.Scalars(battery_states.soc))
        rr.log("voltage/ocv", rr.Scalars(battery_states.v_ocv))
        rr.log("voltage/term", rr.Scalars(battery_states.v_term))
        rr.log("voltage/v_polarization", rr.Scalars(battery_states.v_polarization))

    def log_rigidbody_states(self, rb_states: RigidbodyState):
        """Log rigidbody states"""
        x = rb_states.x
        position = x[:3]
        velocity = x[3:6]
        quat_wxyz = x[6:10]
        angular_vel = x[10:13]
        lin_acc = x[13:16]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self.set_quad_transform(position, quat_xyzw)
        if hasattr(self, "_cached_traj_pos"):
            self._cached_traj_pos = np.vstack([self._cached_traj_pos, position])
            rr.log(
                f"world/drone/position_trace",
                rr.LineStrips3D.from_fields(strips=[self._cached_traj_pos]),
            )
        else:
            rr.log(
                f"world/drone/position_trace",
                rr.LineStrips3D(strips=[position], colors=[[0, 255, 0]], radii=0.01),
            )
            self._cached_traj_pos = position.copy()

        self._log_vec3(f"rigid/position", position)
        self._log_vec3(f"rigid/velocity", velocity)
        self._log_vec3(f"rigid/angular_velocity", angular_vel)
        self._log_vec3(f"rigid/lin_acc", lin_acc)

        euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
        self._log_vec3(f"rigid/euler", euler, labels=["roll", "pitch", "yaw"])

    def log_disturbance(self, force: np.ndarray, torque: np.ndarray, prefix="gt"):
        self._log_vec3(
            f"rigid/forces",
            force,
            labels=[f"{prefix}_fx", f"{prefix}_fy", f"{prefix}_fz"],
        )
        self._log_vec3(
            f"rigid/torques",
            torque,
            labels=[f"{prefix}_tx", f"{prefix}_ty", f"{prefix}_tz"],
        )

    def log_controls(self, controls: dict):
        """Log control commands"""
        for key, value in controls.items():
            rr.log(f"controls/{key}", rr.Scalars(value))

    def _log_vec3(self, root_path: str, vector: np.ndarray, labels=["x", "y", "z"]):
        """Helper to log vector components efficiently"""
        for i, label in enumerate(labels):
            rr.log(f"{root_path}/{label}", rr.Scalars(vector[i]))

    def log_ref_traj(self, traj_pos: np.ndarray):
        if hasattr(self, "_cached_ref_traj_pos"):
            self._cached_ref_traj_pos = np.vstack([self._cached_ref_traj_pos, traj_pos])
            rr.log(
                f"world/drone/ref_traj",
                rr.LineStrips3D.from_fields(strips=[self._cached_ref_traj_pos]),
            )
        else:
            rr.log(
                f"world/drone/ref_traj",
                rr.LineStrips3D(strips=[traj_pos], colors=[[0, 0, 255]], radii=0.01),
            )
            self._cached_ref_traj_pos = traj_pos.copy()

    def log_mpc_ref_traj(self, traj: "List[RigidbodyState]"):
        traj_pos = np.array([state.pos for state in traj])
        if hasattr(self, "_mpc_ref_traj_received"):
            rr.log(
                f"world/drone/mpc_ref_traj",
                rr.LineStrips3D.from_fields(strips=[traj_pos]),
            )
        else:
            rr.log(
                f"world/drone/mpc_ref_traj",
                rr.LineStrips3D(strips=[traj_pos], colors=[[0, 255, 255]], radii=0.02),
            )
            self._mpc_ref_traj_received = True

    def log_mpc_traj(self, traj: np.ndarray):
        if hasattr(self, "_mpc_traj_received"):
            rr.log(
                f"world/drone/mpc_traj",
                rr.LineStrips3D.from_fields(strips=[traj]),
            )
        else:
            rr.log(
                f"world/drone/mpc_traj",
                rr.LineStrips3D(strips=[traj], colors=[[255, 0, 0]], radii=0.02),
            )
            self._mpc_traj_received = True

    def set_quad_transform(self, translation: np.ndarray, rotation_xyzw: np.ndarray):
        rr.log(
            "world/drone/baselink",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
            ),
        )

    def set_quad_mesh_transform(
        self, translation: np.ndarray, rotation_xyzw: np.ndarray
    ):
        rr.log(
            "world/drone/baselink/mesh",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
            ),
            static=True,
        )

    def set_camera_transform(self, translation: np.ndarray, rotation_xyzw: np.ndarray):
        rr.log(
            "world/drone/baselink/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
            ),
            static=True,
        )

    def set_gate_transform(
        self,
        gate_id: str,
        translation: np.ndarray,
        rotation_wxyz: np.ndarray,
        gate_size: Literal["l", "m", "s"] = "s",
    ):
        offset = (
            -1.0
        )  # m, adjust the gate mesh to be centered at the middle of the gate instead of the bottom
        rr.log(
            f"world/gate_{gate_id}",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(
                    xyzw=[
                        rotation_wxyz[1],
                        rotation_wxyz[2],
                        rotation_wxyz[3],
                        rotation_wxyz[0],
                    ]
                ),
            ),
        )
        rr.log(
            f"world/gate_{gate_id}/axis",
            rr.TransformAxes3D(axis_length=0.5),
            static=True,
        )
        rr.log(
            f"world/gate_{gate_id}/mesh",
            rr.Transform3D(translation=[0, 0, offset]),
            static=True,
        )
        rr.log(
            f"world/gate_{gate_id}/mesh/stl", self._gate_assets[gate_size], static=True
        )

    def log_depth_image(self, image: np.ndarray):
        rr.log(
            "world/drone/baselink/camera/",
            rr.Pinhole(
                width=160,
                height=90,
                focal_length=[80, 80],
                principal_point=[80, 45],
            ),
            static=True,
        )
        rr.log("world/drone/baselink/camera/image", rr.DepthImage(image))

    def log_metrics(self, metrics: dict):
        for key, value in metrics.items():
            rr.log(f"metrics/{key}", rr.Scalars(value))

    def log_text(self, text: str, level="INFO"):
        rr.log("logs", rr.TextLog(text, level=level))

    def log_custom_msg(self, custom_callback: Callable):
        custom_callback(rr)
