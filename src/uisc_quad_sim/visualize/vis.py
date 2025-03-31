import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation

class DroneVisualizer:
    def __init__(self):
        """Initialize the drone visualizer"""
        rr.init("uisc_quad_sim.render", spawn=True)
        self._setup_visualization()


    def _setup_visualization(self):
        """Configure the visualization view layout"""
        blueprint = rr.blueprint.Blueprint(
            rr.blueprint.Horizontal(
                rr.blueprint.Vertical(
                    rr.blueprint.Spatial3DView(origin="/world",
                                               contents=[
                                                   "/world/drone/**",
                                               ],
                                                name="3D View"),
                    # rr.blueprint.TextDocumentView(origin="/logs", name="Logs"),
                    row_shares=[3],
                ),
                rr.blueprint.Vertical(
                    rr.blueprint.TimeSeriesView(origin="/state/position", name="Position"),
                    rr.blueprint.TimeSeriesView(origin="/state/velocity", name="Velocity"),
                    rr.blueprint.TimeSeriesView(origin="/state/angular_velocity", name="Angular Velocity"),
                    rr.blueprint.TimeSeriesView(origin="/state/euler", name="Orientation"),
                    rr.blueprint.TimeSeriesView(origin="/motors", name="Motors RPM"),
                    rr.blueprint.TimeSeriesView(origin="/controls", name="Control Commands"),
                    row_shares=[1, 1, 1, 1, 1, 1],
                ),
                column_shares=[6, 3],
            )
        )
        rr.send_blueprint(blueprint)
        self._drone_path = []

    def log_state(
        self,
        timestep: float,
        position: np.ndarray,
        orientation_quat: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        motor_rpms: list[float],
        control_commands: list[float],
    ):
        """Log drone state information
        
        Args:
            timestep: Simulation timestep
            position: 3D position (x, y, z)
            orientation_quat: Orientation quaternion (w, x, y, z)
            velocity: 3D velocity (vx, vy, vz)
            angular_velocity: 3D angular velocity (wx, wy, wz)
            motor_rpms: Four motor RPM values [rpm1, rpm2, rpm3, rpm4]
            control_commands: Four control commands [cmd1, cmd2, cmd3, cmd4]
        """
        rr.set_time_seconds("sim_time", timestep)
        
        # Log 3D pose and position
        self._log_3d_pose(position, orientation_quat)
        
        # Log 2D time series data
        self._log_position(position)
        self._log_velocity(velocity)
        self._log_angular_velocity(angular_velocity)
        self._log_orientation(orientation_quat)
        self._log_motors(motor_rpms)
        self._log_controls(control_commands)
        self._drone_path.append(position)
        self._log_path(self._drone_path)

    def _log_3d_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """Log 3D pose visualization"""
        rr.log("world/drone/odom",
            rr.Transform3D(
                translation=rr.components.Vector3D(
                    xyz=position
                ),
                rotation=rr.Quaternion(xyzw=[
                    quaternion[1],  # x
                    quaternion[2],  # y
                    quaternion[3],  # z
                    quaternion[0],  # w
                ]),
                axis_length=0.6
            )
        )

    def _log_path(self, path):
        """Log 3D path"""
        rr.log(
            "world/drone/path",
            rr.LineStrips3D(
                strips=path,
                colors=np.tile([0, 255, 0], (len(path), 1))
            )
        )

    def _log_position(self, position: np.ndarray):
        """Log position components"""
        components = ["x", "y", "z"]
        for i, comp in enumerate(components):
            rr.log(
                f"state/position/{comp}",
                rr.Scalar(position[i])
            )

    def _log_velocity(self, velocity: np.ndarray):
        """Log velocity components"""
        components = ["x", "y", "z"]
        for i, comp in enumerate(components):
            rr.log(f"state/velocity/{comp}", rr.Scalar(velocity[i]))

    def _log_angular_velocity(self, angular_velocity: np.ndarray):
        """Log angular velocity components"""
        components = ["x", "y", "z"]
        for i, comp in enumerate(components):
            rr.log(f"state/angular_velocity/{comp}", rr.Scalar(angular_velocity[i]))

    def _log_orientation(self, quaternion: np.ndarray):
        """Log Euler angle orientation"""
        euler = Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
        components = ["roll", "pitch", "yaw"]
        for i, comp in enumerate(components):
            rr.log(f"state/euler/{comp}", rr.Scalar(euler[i]))

    def _log_motors(self, rpms: list[float]):
        """Log motor RPM values"""
        for i in range(4):
            rr.log(
                f"motors/motor_{i+1}",
                rr.Scalar(rpms[i])
            )

    def _log_controls(self, commands: list[float]):
        """Log control commands"""
        for i in range(4):
            rr.log(
                f"controls/control_{i+1}",
                rr.Scalar(commands[i])
            )