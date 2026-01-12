import os, sys, time
import numpy as np
from tqdm import tqdm
from loguru import logger
from uisc_quad_sim.simulations import QuadSim, QuadSimParams
from uisc_quad_sim.simulations.quadrotor_sim import ControlCommand, ControlMode
from uisc_quad_sim.visualize.vis import DroneVisualizer
from uisc_quad_sim.controller.se3 import SE3Controller

logger.remove()
logger.add(sys.stdout, level="INFO")

# Input:
# x: state estimation 13x1
# x_d: desired state 7x1[px, py, pz, vx, vy, vz, yaw]
# Output: u: control input


def example_ref(t, r=5.0, omega=1.0):
    p = np.array([r * np.sin(omega * t), r * np.cos(omega * t), 1])
    v = np.array([r * omega * np.cos(omega * t), -r * omega * np.sin(omega * t), 0])
    yaw = t
    return np.concatenate([p, v, [yaw]])


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    quad_params = QuadSimParams.load(os.path.join(dir_path, "../configs/ctbr.yaml"))
    quad_sim = QuadSim(quad_params)
    quad_vis = DroneVisualizer()
    t_end = 15
    logger.info("Start simulation")
    s_t = time.perf_counter()
    quad_sim.reset_pos(example_ref(0)[:3])
    ctrl = SE3Controller(SE3Controller.M_PV)
    with tqdm(total=t_end // quad_params.dt) as pbar:
        while quad_sim.t < t_end:
            x = quad_sim.estimate()
            x_gt = quad_sim.estimate(gt=True)
            x_ref = example_ref(quad_sim.t)
            ctbr = ctrl.compute_control(x, x_ref)
            thrust = quad_sim._quad.thrustMap(quad_sim._motors_omega)  # 4xN
            real_thrust_acc = np.sum(thrust) / quad_sim._quad.mass
            real_angvel = x_gt[10:13]
            control_state = np.array(
                [real_thrust_acc, real_angvel[0], real_angvel[1], real_angvel[2]]
            )
            quad_vis.log_state(
                quad_sim.t, x_gt, quad_sim.motor_speed, ctbr, control_state
            )
            cmd = ControlCommand(ControlMode.CTBR, ctbr)
            quad_sim.step(cmd)
            pbar.update(1)
    e_t = time.perf_counter()
    logger.info(f"Simulation finished in {e_t-s_t:.2f}s")
    return


if __name__ == "__main__":
    main()
