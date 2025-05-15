import numpy as np

class LPF:
    def __init__(self,dim,f_cut):
        self.lpf_prev = np.zeros(dim)
        self.f_cut = f_cut
    
    def update(self,lpf_now,dt)->np.ndarray:
        if(dt<=0):
            return lpf_now
        rc = 2*np.pi * self.f_cut * dt
        alpha = rc/(rc+1)
        lpf = alpha*lpf_now + (1-alpha)*self.lpf_prev
        self.lpf_prev = lpf
        return lpf