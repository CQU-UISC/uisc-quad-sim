import os,sys,time
import numpy as np
from tqdm import tqdm
from loguru import logger
from ..utils.quaternion import q_mult, q_inv, q_rot, mat_q
from ..simulations import QuadSim,QuadSimParams
from ..visualize.vis import DroneVisualizer

logger.remove()
logger.add(sys.stdout, level="INFO")

#Input: 
# x: state estimation 13x1
# x_d: desired state 7x1[px, py, pz, vx, vy, vz, yaw]
#Output: u: control input
err_v_integral = np.zeros(3)

def so3_ctrl(x, x_d):
    global err_v_integral
    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    p_d = x_d[0:3]
    v_d = x_d[3:6]
    yaw_d = x_d[6]
    kp = 10
    kv = 5
    ki = 0
    kw = 5
    tau = 0.1
    # kr = 5
    # kw = 5

    e_p = p - p_d
    e_v = v - v_d
    err_v_integral += e_v
    max_integral = 10
    err_v_integral = np.clip(err_v_integral, -max_integral, max_integral)
    a_d = -kp * e_p - kv * e_v + 9.81*np.array([0, 0, 1]) - ki * err_v_integral

    # x corss y = z
    # z cross x = y
    # y cross z = x
    z_d = a_d / np.linalg.norm(a_d) #desired z-axis
    x_d = np.array([np.cos(yaw_d), np.sin(yaw_d), 0]) #desired x-axis
    y_d = np.cross(z_d, x_d) #desired y-axis
    
    R_d = np.array([x_d, y_d, z_d]).T
    q_d = mat_q(R_d)

    e_q = q_mult(q_d, q_inv(q))
    e_q_s = e_q[0]
    e_q_v = e_q[1:]

    w_d =2/tau*e_q_v*np.sign(e_q_s) #desired angular velocity
    # CTBR
    #thrust
    z = q_rot(q, np.array([0, 0, 1]))
    u1 = np.dot(a_d, z)#projection of a_d on z-axis
    return np.concatenate([[u1],w_d])
    # CTBM
    e_w = w - w_d#angular velocity error
    #omega
    u2 = - (kw * e_w) 
    #torque
    # u2 = -kw * J@e_w + np.cross(w, J @ w)
    return np.concatenate([[u1], u2])

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
    t_end = 20
    logger.info("Start simulation")
    s_t = time.perf_counter()
    quad_sim.step(np.zeros(4))
    quad_sim.reset_pos(
        example_ref(0)[:3]
    )
    with tqdm(total=t_end//quad_sim.dt) as pbar:
        while quad_sim.t < t_end:
            x = quad_sim.estimate()
            x_gt = quad_sim.estimate(gt=True)
            x_ref = example_ref(quad_sim.t)
            ctbr = so3_ctrl(x, x_ref)
            quad_vis.log_state(
                quad_sim.t,
                x_gt[:3],
                x_gt[6:10],
                x_gt[3:6],
                x_gt[10:13],
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