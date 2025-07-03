import numba as nb
import numpy as np
from .disturbance import Disturbance

'''
quadrotor dynamics
'''
@nb.njit()
def quadrotor_dynamics(x:np.ndarray,
                       x_dot:np.ndarray,
                       u:np.ndarray,
                       mass:float,
                       g:float,
                       J:np.ndarray,
                       J_inv:np.ndarray,
                       ext_force:np.ndarray,
                       ext_moment:np.ndarray,
                       drag_coeff:np.ndarray)->None:
    '''
        quadrotor's dynamics
        Input:
            x: state vector shape(13,N)
            x_dot: state derivative shape(13,N)
            u: control vector shape(4,N)
            mass: mass scalar
            g: gravity scalar
            J: inertia matrix shape(3,3)
            J_inv: inverse inertia matrix shape(3,3)
            ext_force: external force shape(3,N) inertial frame
            ext_moment: external ext_moment shape(3,N) body frame
            drag_coeff: drag coefficient shape(3) [N/(m/s)]
        Output:
            None
    '''
    v = x[3:6]
    q = x[6:10] #w,x,y,z
    w = x[10:13]
    mass_inv = 1/mass
    colective_thrust = u[0]*mass_inv
    tau = u[1:4]

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    wx = w[0]
    wy = w[1]
    wz = w[2]

    drag_acc = -drag_coeff[:,None] * v * mass_inv
    # print("ExtForce:",ext_force)
    x_dot[0:3] = v
    v_dot = x_dot[3:6]
    v_dot[0] =  2*(qw*qy+qx*qz)*colective_thrust + ext_force[0]*mass_inv + drag_acc[0]
    v_dot[1] =  2*(qy*qz-qw*qx)*colective_thrust + ext_force[1]*mass_inv + drag_acc[1]
    v_dot[2] = (1-2*(qx**2+qy**2))*colective_thrust - g + ext_force[2]*mass_inv + drag_acc[2]
    q_dot = x_dot[6:10]
    q_dot[0] =  0.5*(-wx*qx - wy*qy - wz*qz)
    q_dot[1] =  0.5*(wx*qw + wz*qy - wy*qz)
    q_dot[2] =  0.5*(wy*qw - wz*qx + wx*qz)
    q_dot[3] =  0.5*(wz*qw + wy*qx - wx*qy)

    #For single rigid body: J_inv*(tau - w x Jw)
    # x_dot[:,10:13] = np.dot(tau - np.cross(w,np.dot(w,J.T)), J_inv.T)
    x_dot[10:13] = J_inv@(ext_moment + tau - np.cross(w.T,np.dot(J,w).T).T)
    return

class QuadrotorDynamics:
    def __init__(self,
                 mass:float,
                 g:float,
                 J:np.ndarray,
                 J_inv:np.ndarray,
                 drag_coeff:np.ndarray,
                 disturb:Disturbance):
        self.mass = mass
        self.g = g
        self.J = J
        self.J_inv = J_inv
        self.disturb = disturb
        self.drag_coeff = drag_coeff

    def dxdt(self,x,u):
        x_dot = np.zeros_like(x)
        quadrotor_dynamics(x,
                           x_dot,
                           u,
                           self.mass,
                           self.g,
                           self.J,
                           self.J_inv,
                        self.disturb.force(x,u),
                        self.disturb.moment(x,u),
                        self.drag_coeff)
        return x_dot