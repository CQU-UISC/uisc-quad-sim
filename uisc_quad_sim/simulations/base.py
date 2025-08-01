from ..integration import rk4,forward_eulr
from ..dynamics import Dynamics

class Sim(object):
    def __init__(self,dt:float) -> None:
        self.__dt = dt
        self.__t = 0

    @property
    def dt(self):
        return self.__dt
    
    @property
    def t(self):
        return self.__t
    
    def step(u):
        raise NotImplementedError
    
    def _step_t(self):
        '''
            Step the simulation by one time step
        '''
        self.__t += self.__dt
    
    def set_seed(self,seed:int):
        raise NotImplementedError

    def reset(self):
        self.__t = 0
        
    def _run(self,dynamics:Dynamics,x0,u):
        '''
            Run one simulation step
            Args:
                dynamics:Dynamics - dynamics of the system
                x0:np.ndarray - initial state
                u:np.ndarray - control input
            Returns:
                x:np.ndarray - final state
        '''
        return rk4(dynamics,x0,u,self.dt)

    def state(self):
        raise NotImplementedError