from .base import BaseController
import numpy as np
from uisc_quad_sim.utils.quaternion import q_mult, q_inv, q_rot, mat_q

class SE3Controller(BaseController):
    C_POS = 0b0001
    C_VEL = 0b0010

    M_PV = 0b0011
    M_V = 0b0010

    def __init__(self, mode:int):
        super().__init__()
        self.err_v_integral = np.zeros(3)
        self.max_integral = 10
        self.kp = 12
        self.kv = 5
        self.ki = 0
        self.kw = 5
        self.tau = 0.1

        # mode=>
        if mode&self.C_POS==0:
            self.kp = 0
        
        if mode&self.C_VEL==0:
            self.kv = 0 


    def compute_control(self, state, *args):
        '''
            Input:
                state: state vector shape(13,N)
                args: desired state 7x1[px, py, pz, vx, vy, vz, yaw]
            Output:
                control input: [thrust, angular velocity] shape(4,N)
                or
                control input: [thrust, torque] shape(4,N)
        '''
        p = state[0:3]
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]
        x_d = args[0]

        p_d = x_d[0:3]
        v_d = x_d[3:6]
        yaw_d = x_d[6]

        # kr = 5
        # kw = 5

        e_p = p - p_d
        e_v = v - v_d
        self.err_v_integral += e_v
        self.err_v_integral = np.clip(self.err_v_integral, -self.max_integral, self.max_integral)
        a_d = -self.kp * e_p - self.kv * e_v + 9.81*np.array([0, 0, 1]) - self.ki * self.err_v_integral

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

        w_d =2/self.tau*e_q_v*np.sign(e_q_s) #desired angular velocity
        # CTBR
        #thrust
        z = q_rot(q, np.array([0, 0, 1]))
        u1 = np.dot(a_d, z)#projection of a_d on z-axis
        return np.concatenate([[u1],w_d])
