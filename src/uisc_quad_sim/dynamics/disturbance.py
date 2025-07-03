#raw author: https://github.com/KevinHuang8/DATT/blob/main/python_utils/wind.py
import numpy as np
import numba as nb

@nb.jit
def velocity(pos,w_pos,w_dir,radius,coneslope,vmax,decay_lat,decay_long):
        """
        The fan velocity is radially symmetric around the fan,
        has a maximum of vmax and decays exponentially in two directions:
          perpendicular to the fan direction and
          parallel to the fan direction

        decay_lat controls how fast it decays perpendicularly
          if set to 0, results in an infinitely large wall of wind
        decay_long for parallel
          if set to 0, results in an infinitely long cone of wind

        radius denotes a disk perpendicular to the direction over which there is no decay

        Ideally decay_lat is large and > decay_long, to get a sharp cutoff at the fan edge (edge of disk)

        dispangle controls at what angle the radius grows, i.e. simulates wind field dispersion
        if set to 0, wind beam will not expand

        Ideally, this can be made more physically correct by solving some boundary value diff eq.
        """
        dist = pos - w_pos  # N*3
        r_para = np.dot(dist, w_dir)  # N

        # if r_para < 0:
        # return 0
        r_para = np.maximum(0, r_para)

        r_perp = np.linalg.norm(np.cross(dist, w_dir))

        r_eff = radius + coneslope * r_para
        r_perp = np.maximum(0, r_perp - r_eff)

        return (
            vmax
            * np.exp(-decay_lat * r_perp)
            * np.exp(-decay_long * r_para)
        )  

class Disturbance:
    name = 'D_BASE'

    def force(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def moment(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def __str__(self):
        return 'Base Disturbance'

    def field(self,box=(5,5,3),n=(12,12,6)) -> np.ndarray:
        # return force field
        nx, ny, nz = n
        x_f,y_f, z_f = box
        x, y, z = np.meshgrid(
            np.linspace(-x_f, x_f, nx),
            np.linspace(-y_f, y_f, ny),
            np.linspace(0, z_f, nz),
            indexing='ij'
        )

        output = np.zeros((nx, ny, nz, 3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = np.array([(x[i, j, k], y[i, j, k], z[i, j, k])])
                    output[i, j, k] = self.force(pos.T, np.zeros((3, 1))).T
        return (x, y, z, output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2])

class EmptyField(Disturbance):
    name = 'D_EMPTY'
    def force(self, x, u):
        N = x.shape[1]
        return np.zeros((3, N))

    def moment(self, x, u):
        N = x.shape[1]
        return np.zeros((3, N))

    def __str__(self):
        return 'Empty Field'

class ConstField(Disturbance):
    name = 'D_CONST'
    def __init__(self,force, moment):
        # force and torque has shape (3,)
        self.force_val = force
        self.moment_val = moment
    
    def force(self, x, u):
        return np.tile(self.force_val, (x.shape[1],1)).T
    
    def moment(self, x, u):
        return np.tile(self.moment_val, (x.shape[1],1)).T
    
    def __str__(self):
        return f'Const Field: {self.force_val}, {self.moment_val}'

class TimeVarField(Disturbance):
    name = 'D_TVAR'
    def __init__(self,force, moment, dt):
        # force and torque has shape (3,)
        self.force_val = force
        self.moment_val = moment
        self.tf = 0
        self.tm = 0
        self.dt = dt

    def force(self, x, u):
        force_val = np.sin(self.tf)*self.force_val
        self.tf += self.dt
        return np.tile(force_val, (x.shape[1],1)).T
    
    def moment(self, x, u):
        moment_val = np.cos(self.tm)*self.moment_val
        self.tm += self.dt
        return np.tile(moment_val, (x.shape[1],1)).T
    
    def __str__(self):
        return f'TimeVar Field: {self.force_val}, {self.moment_val}'

class WindField(Disturbance):
    name = 'D_WIND'
    def __init__(
        self,
        pos,
        to,
        vmax=10.0,
        radius=0.2,
        noisevar=0.5,
        decay_lat=4,
        decay_long=0.6,
        dispangle = 15
    ):
        self.vmax = vmax
        self.noisevar = noisevar
        self.pos = pos
        self.dir = (to-pos) / np.linalg.norm(to-pos)
        self.vmax = vmax
        self.radius = radius
        self.decay_lat = decay_lat
        self.decay_long = decay_long
        self.coneslope = np.tan(np.radians(dispangle))

    def force(self, x, u):
        # shape of x is [13,N]
        pos = x[0:3].T  # N*3
        windvel = velocity(pos,self.pos,self.dir,self.radius,self.coneslope,self.vmax,self.decay_lat,self.decay_long)[:, None] * np.tile(self.dir, (pos.shape[0], 1))
        noise_scale = np.linalg.norm(windvel) / self.vmax
        f = 0.5 * windvel + self.noisevar * noise_scale * np.random.normal(size=3)
        # return should be 3*N
        return f.T 

    def moment(self, x, u):
        return np.zeros((3, x.shape[1]))
    
    def __str__(self):
        return f'''Wind Field: 
    pos:{self.pos}
    dir:{self.dir}
    vmax:{self.vmax}
    radius:{self.radius}
    noisevar:{self.noisevar}
    decay_lat:{self.decay_lat}
    decay_long:{self.decay_long}
    coneslope:{self.coneslope}'''


class CompositeFiled(Disturbance):
    name = 'D_COMPOSITE'
    def __init__(self, *disturbances: Disturbance):
        self.disturbances = []
        for d in disturbances:
            if not isinstance(d, Disturbance):
                raise TypeError(f'Expected Disturbance, got {type(d)}')
            self.disturbances.append(d)

    def add(self, disturbance: Disturbance):
        if not isinstance(disturbance, Disturbance):
            raise TypeError(f'Expected Disturbance, got {type(disturbance)}')
        self.disturbances.append(disturbance)
    
    def clear(self):
        self.disturbances = []

    def __len__(self):
        return len(self.disturbances)

    def force(self, x, u):
        f = np.zeros((3, x.shape[1]))
        for d in self.disturbances:
            f += d.force(x, u)
        return f

    def moment(self, x, u):
        m = np.zeros((3, x.shape[1]))
        for d in self.disturbances:
            m += d.moment(x, u)
        return m
    
    def __str__(self):
        return 'Composite Disturbance: ' + ', '.join([str(d) for d in self.disturbances])