import os, sys, time, argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from uisc_quad_sim.dynamics.rigidbody import RigidbodyState
from uisc_quad_sim.simulations import QuadSim, QuadParams
from uisc_quad_sim.visualize.vis import DroneVisualizer
from uisc_quad_sim.controller.se3 import SE3Controller
from uisc_quad_sim.utils.quaternion import from_euler

logger.remove()
logger.add(sys.stdout, level="INFO")


def example_ref(t, r, omega) -> RigidbodyState:
    x = np.zeros(RigidbodyState.shape)
    rg_state = RigidbodyState(x=x)
    rg_state.pos = np.array([r * np.sin(omega * t), r * np.cos(omega * t), 1])
    rg_state.vel = np.array(
        [r * omega * np.cos(omega * t), -r * omega * np.sin(omega * t), 0]
    )
    rg_state.lin_acc = np.array(
        [-r * omega**2 * np.sin(omega * t), -r * omega**2 * np.cos(omega * t), 0]
    )
    # let head to the center(0,0)
    yaw = np.arctan2(-rg_state.pos[1], -rg_state.pos[0])
    rg_state.quat = from_euler(0, 0, yaw)
    return rg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="uisc_xi35.yaml")
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=1.0,
        help="Radius of the circular trajectory",
    )
    parser.add_argument(
        "-o",
        "--omega",
        type=float,
        default=1.0,
        help="Angular speed of the circular trajectory",
    )
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    quad_params = QuadParams(os.path.join(dir_path, f"../configs/{args.config}"))
    quad_sim = QuadSim(quad_params)
    quad_vis = DroneVisualizer()
    t_end = 15
    logger.info("Start simulation")
    s_t = time.perf_counter()
    quad_sim.reset()
    quad_sim.rb_state.x[2] = 1.0  # initial height 1m
    quad_sim.batt_state.soc = 1.0  # initial SOC 100%
    ctrl = SE3Controller()
    with tqdm(total=t_end // quad_params.high_level_dt) as pbar:
        while quad_sim.t < t_end:
            state = quad_sim.rb_state
            ref_state = example_ref(quad_sim.t, r=args.radius, omega=args.omega)
            cmd = ctrl.compute_control(state, ref_state)
            quad_sim.step(cmd)

            real_thrust_acc = quad_sim.rb_state.blin_acc[2] + quad_params.rigid.g
            real_angvel = quad_sim.rb_state.ang_vel
            ctbr_response = np.array(
                [real_thrust_acc, real_angvel[0], real_angvel[1], real_angvel[2]]
            )
            quad_vis.step(quad_sim.t)
            quad_vis.log_battery_states(quad_sim.batt_state)
            quad_vis.log_rigidbody_states(quad_sim.rb_state)
            quad_vis.log_motor_states(quad_sim.motor_state)
            quad_vis.log_controls(cmd.u, ctbr_response)
            quad_vis.log_ref_traj(ref_state.pos)
            pbar.update(1)
    e_t = time.perf_counter()
    logger.info(f"Simulation finished in {e_t-s_t:.2f}s")
    return


if __name__ == "__main__":
    main()
