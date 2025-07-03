from typing import Callable
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation
from uuid import uuid4
class DroneVisualizer:
    def __init__(self,env_num=1):
        """Initialize the drone visualizer"""
        self.env_num = env_num
        self.reset()
    
    def reset(self):
        """Reset the visualizer"""
        rr.init("uisc_quad_sim.render",recording_id=uuid4(),spawn=True)
        if self.env_num==1:
            self._setup_single_visualization()
        else:
            self._setup_vectorized_visualization(self.env_num)
    
    def _setup_vectorized_visualization(self, nums=1):
        """Configure the visualization view layout"""
        blueprint = rr.blueprint.Blueprint(
            rr.blueprint.Vertical(
                rr.blueprint.Spatial3DView(origin="/world",
                                            contents=[
                                                "/world/drone/**",
                                            ],
                                            name="3D View"),
                # rr.blueprint.TextDocumentView(origin="/logs", name="Logs"),
                row_shares=[3],
            ),
        )
        rr.send_blueprint(blueprint)
        self._drone_paths = [[] for _ in range(nums)]

    def _setup_single_visualization(self):
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

    # vectorized visualization
    def log_states(
        self,
        timestep: float,
        states: np.ndarray):
        """
            Log drone state information:
            Args:
                timestep: Simulation timestep
                states: State information for all drones has shape (13,N)
        """
        positions = states[:3, :].T
        orientations = states[6:10, :].T
        for i in range(states.shape[1]):
            self._log_3d_pose(positions[i], orientations[i], i)
            self._drone_paths[i].append(positions[i])
            self._log_path(self._drone_paths[i], i)
        rr.set_time_seconds("sim_time", timestep)

    # single visualization
    def log_state(
        self,
        timestep: float,
        state: np.ndarray,
        motor_rpms: list[float],
        control_commands: list[float],
    ):
        """Log drone state information
        
        Args:
            timestep: Simulation timestep
            state: State information [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            motor_rpms: Four motor RPM values [rpm1, rpm2, rpm3, rpm4]
            control_commands: Four control commands [cmd1, cmd2, cmd3, cmd4]
        """
        rr.set_time_seconds("sim_time", timestep)
        position = state[:3]
        velocity = state[3:6]
        orientation_quat = state[6:10]
        angular_velocity = state[10:]
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

    def log_mpc_traj(self, traj, quad_id=0):
        """Log 3D traj"""
        rr.log(
            f"world/drone/{quad_id}/mpc_traj",
            rr.LineStrips3D(
                strips=traj,
                colors=np.tile([255, 0, 0], (len(traj), 1))
            )
        )
    

    def log_traj_ref(self, traj, quad_id=0):
        """Log 3D traj"""
        rr.log(
            f"world/drone/{quad_id}/traj",
            rr.LineStrips3D(
                strips=traj,
                colors=np.tile([125, 125, 0], (len(traj), 1))
            )
        )

    def log_custom_msg(self,custom_callback:Callable):
        custom_callback(
            rr
        )

    def _log_3d_pose(self, position: np.ndarray, quaternion: np.ndarray,quad_id=0):
        """Log 3D pose visualization"""
        rr.log(f"world/drone/{quad_id}/odom",
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

    def _log_path(self, path, quad_id=0):
        """Log 3D path"""
        rr.log(
            f"world/drone/{quad_id}/path",
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
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
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