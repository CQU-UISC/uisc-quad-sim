import os,sys,time
import numpy as np
from tqdm import tqdm
from loguru import logger
from uisc_quad_sim.simulations import QuadSim,QuadSimParams
from uisc_quad_sim.visualize.vis import DroneVisualizer
from uisc_quad_sim.controller.se3 import SE3Controller

logger.remove()
logger.add(sys.stdout, level="INFO")

#Input: 
# x: state estimation 13x1
# x_d: desired state 7x1[px, py, pz, vx, vy, vz, yaw]
#Output: u: control input

def example_ref(t):
    p = np.array(
        [
            np.sin(t),
            np.cos(t),
            1
        ]
    )
    v = np.array(
        [
            np.cos(t),
            -np.sin(t),
            0
        ]
    )
    yaw = 0
    return np.concatenate([p, v, [yaw]])

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    quad_params = QuadSimParams.loadFromFile(
        os.path.join(dir_path, "../../../configs/ctbr.yaml")
    )
    quad_sim = QuadSim(quad_params)
    quad_vis = DroneVisualizer()
    t_end = 15
    logger.info("Start simulation")
    s_t = time.perf_counter()
    quad_sim.step(np.zeros(4))
    quad_sim.reset_pos(
        example_ref(0)[:3]
    )
    ctrl = SE3Controller(quad_params.dt, SE3Controller.M_V)
    with tqdm(total=t_end//quad_params.dt) as pbar:
        while quad_sim.t < t_end:
            x = quad_sim.estimate()
            x_gt = quad_sim.estimate(gt=True)
            x_ref = example_ref(quad_sim.t)
            ctbr = ctrl.compute_control(x, x_ref)
            quad_vis.log_state(
                quad_sim.t,
                x_gt,
                quad_sim.motor_speed,
                ctbr
            )
            quad_sim.step(ctbr)
            pbar.update(1)
    e_t = time.perf_counter()
    logger.info(f"Simulation finished in {e_t-s_t:.2f}s")
    return

if __name__ == "__main__":
    main()