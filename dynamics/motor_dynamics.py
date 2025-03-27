import numba as nb
import numpy as np
from .base import Dynamics
'''
Quadrotor motor dynamics dynamics
'''
@nb.njit()
def motor_dynamics(x,x_dot, u, tau_inv):
    '''
        Quadrotor motor dynamics
        Input:
            x: state vector shape(4,N) (normalized motor speed 0-1)
            x_dot: state derivative shape(4,N)
            u: control vector shape(4,N)
            tau_inv: inverse time constant scalar
        Output:
            None
    '''
    x_dot[:] = (u - x)*tau_inv
    return

class MotorDynamics(Dynamics):
    def __init__(self, tau_inv):
        self.tau_inv = tau_inv

    def dxdt(self, x, u):
        x_dot = np.zeros_like(x)
        motor_dynamics(x, x_dot, u, self.tau_inv)
        return x_dot