from ..integration import rk4
from ..dynamics import Dynamics

class Sim(object):
    def __init__(self,dt:float) -> None:
        self.__dt = dt
        self.__t = 0
        self.__min_dt = 0.005
        self.__step_size = int(dt//self.__min_dt)
        if self.__step_size < 1:
            self.__step_size = 1

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
        x = x0
        for _ in range(self.__step_size): 
            x = rk4(dynamics,x,u,self.__min_dt)
        return x
    
    def state(self):
        raise NotImplementedError