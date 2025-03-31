#thanks to https://github.com/uzh-rpg
import numpy as np
import yaml
from loguru import logger

class Quadrotor:
    '''
        Quadrotor configuration
    '''
    def __init__(self,
                name:str,
                mass:float,#kg
                inertia:np.ndarray,#kg*m^2 inertia matrix \in R^3
                arm_length:float,#m
                kappa:float,#Nm/N momentum constant
                drag_coeff:np.ndarray,#drag coefficient shape(3) [N/(m/s)]
                motor_min:float,#min motor speed(always 0)
                motor_max:float,#max motor speed(always 1)
                motor_time_constant:float,#s
                thrust_map:np.ndarray,#thrust map[c2,c1,c0]
                thrust_max:float,#N
                thrust_min:float,#N 
                omega_min:np.ndarray,#rad/s
                omega_max:np.ndarray#rad/s
                ):
        assert mass > 0, "mass should be positive"
        assert np.all(inertia > 0), "inertia should be positive"
        assert len(inertia) == 3, "inertia should be 3x1"
        assert arm_length > 0, "arm_length should be positive"
        assert kappa > 0, "kappa should be positive"
        assert len(thrust_map) == 3, "thrust_map should be 3x1"
        assert len(drag_coeff) == 3, "drag_coeff should be 3x1"
        assert len(omega_min) == 3, "omega_min should be 3x1"
        assert len(omega_max) == 3, "omega_max should be 3x1"
        self.name = name
        self.mass = mass
        self.J = np.diag(inertia)
        self.J_inv = np.linalg.inv(self.J)
        self.arm_length = arm_length
        self.kappa = kappa
        self.motor_min = motor_min
        self.motor_max = motor_max
        self.tau = motor_time_constant
        self.tau_inv = 1/motor_time_constant
        self.thrust_map = thrust_map
        self.thrust_max = thrust_max
        self.thrust_min = thrust_min
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.tbm = self.arm_length*(0.5)**0.5*np.array([[1,-1,-1,1],[-1,-1,1,1],[0,0,0,0]])
        self.tbm[2,:] = self.kappa*np.array([1,-1,1,-1])
        # Init variables
        self._min_collective_thrust = None
        self._max_collective_thrust = None
        self.B = None
        self.B = self.allocatioMatrix() #allocation matrix B@thrust[4] = u[thrust,torques]
        # B@[t1,t2,t3,t4] 
        self.B_inv = np.linalg.pinv(self.B)
        self.drag_coeff = drag_coeff
        arm_l_xy = self.arm_length*(0.5)**0.5
        self.max_torque = np.array([2*arm_l_xy*self.thrust_max,
                                    2*arm_l_xy*self.thrust_max,
                                    2*self.kappa*self.thrust_max])
        self.min_torque = -self.max_torque
    def __str__(self):
        return f'''Quadrotor {self.name}
    Mass: {self.mass} kg
    Inertia: {np.diag(self.J)}
    Arm Length: {self.arm_length} m
    Kappa: {self.kappa} Nm/N
    Drag Coefficient: {self.drag_coeff} N/(m/s)
    Motor Speed Range: [{self.motor_min},{self.motor_max}]
    Motor Time Constant: {self.tau} s
    Thrust Map: {self.thrust_map}
    Thrust Range: [{self.thrust_min},{self.thrust_max}]'''
    
    @staticmethod
    def loadFromFile(file_path):
        '''
            Load quadrotor parameters from file
        '''
        cfg = yaml.load(open(file_path,'r'),Loader=yaml.FullLoader)

        # check if all parameters are provided
        assert 'name' in cfg, "name should be provided"
        assert 'mass' in cfg, "mass should be provided"
        assert 'inertia' in cfg, "inertia should be provided"
        assert 'motors' in cfg, "motors should be provided"
        assert 'arm_length' in cfg['motors'], "arm_length should be provided"
        assert 'kappa' in cfg['motors'], "kappa should be provided"
        assert 'drag_coeff' in cfg, "drag_coeff should be provided"
        assert 'motor_min' in cfg['motors'], "motor_min should be provided"
        assert 'motor_max' in cfg['motors'], "motor_max should be provided"
        assert 'motor_time_constant' in cfg['motors'], "motor_time_constant should be provided"
        assert 'thrust_map' in cfg['motors'], "thrust_map should be provided"
        assert 'thrust_min' in cfg['motors'], "thrust_min should be provided"
        assert 'thrust_max' in cfg['motors'], "thrust_max should be provided"
        name:str = cfg['name']
        mass :float= cfg['mass']
        inertia = np.array(cfg['inertia'])
        arm_length:float = cfg['motors']['arm_length']
        kappa:float = cfg['motors']['kappa']
        drag_coeff = np.array(cfg['drag_coeff'])
        motor_min:float = cfg['motors']['motor_min']
        motor_max :float= cfg['motors']['motor_max']
        motor_time_constant:float = cfg['motors']['motor_time_constant']
        thrust_map = np.array(cfg['motors']['thrust_map'])
        thrust_max:float = cfg['motors']['thrust_max']
        thrust_min:float = cfg['motors']['thrust_min']
        omega_min:np.ndarray = np.array( cfg['limits']['omega_min'])
        omega_max:np.ndarray = np.array( cfg['limits']['omega_max'])
        return Quadrotor(    name,
                        mass,
                        inertia,
                        arm_length,
                        kappa,
                        drag_coeff,
                        motor_min,
                        motor_max,
                        motor_time_constant,
                        thrust_map,
                        thrust_max,
                        thrust_min,
                        omega_min,
                        omega_max)
    
    def clipMotorSpeed(self,omega):
        '''
            Clip motor speed to the range [motor_min,motor_max]
            Input:
                omega: motor speed \in R has shape (4,N)
        '''
        return np.clip(omega,self.motor_min,self.motor_max)
        
    def thrustMap(self,omega):
        '''
            Calculate thrust from motor speed(range from 0 to 1)
            Input:
                omega: motor speed \in R has shape (4,N)
            Output:
                thrust: thrust \in R^3 (unit:N) has shape (4,N)
        '''
        # element wise multiply
        omega2 = omega**2
        thrust = np.zeros_like(omega)
        thrust[:] = self.thrust_map[0]*omega2 + self.thrust_map[1]*omega + self.thrust_map[2]
        return thrust
    
    def thrustMapInv(self,thrust):
        '''
            Calculate motor speed from thrust
            Input:
                thrust: thrust \in R^3 (unit:N) has shape (4,N)
            Output:
                omega: motor speed \in R has shape (4,N)
        '''
        # ax^2 + bx + c = 0
        # thrust = c2*omega^2 + c1*omega + c0
        # c2*omega^2 + c1*omega + c0 - thrust = 0
        # a = c2, b = c1, c = c0 - thrust
        # ==> x = (-b + (b^2 - 4ac)^0.5)/(2a)
        c2 = self.thrust_map[0] 
        c1 = self.thrust_map[1]
        c0 = self.thrust_map[2]
        scale = 1/(2*c2)
        offset = -c1*scale
        root = np.sqrt(c1**2 - 4*c2*(c0 - thrust))
        return offset + root*scale
    
    def allocatioMatrix(self):
        '''
            Calculate the allocation matrix
            Output:
                B: allocation matrix \in R^{4,4}
        '''
        if self.B is not None:
            return self.B
        self.B = np.ones((4,4))
        self.B[1:4,:] = self.tbm
        return self.B

    def clipThrust(self,thrust):
        '''
            Clip thrust to the range [thrust_min,thrust_max]
            Input:
                thrust: thrust \in R (unit:N) has shape (4,N)
        '''
        return np.clip(thrust,self.thrust_min,self.thrust_max)
    
    def minCollectiveThrust(self):
        '''
            Calculate minimum collective thrust
        '''
        if self._min_collective_thrust is not None:
            return self._min_collective_thrust
        self._min_collective_thrust = self.thrust_min*4/self.mass
    
    def maxCollectiveThrust(self):
        '''
            Calculate maximum collective thrust
        '''
        if self._max_collective_thrust is not None:
            return self._max_collective_thrust
        self._max_collective_thrust = self.thrust_max*4/self.mass
        return self.thrust_max*4/self.mass
    
    def clipCollectiveThrust(self,thrust):
        '''
            Clip collective thrust to the range [minCollectiveThrust,maxCollectiveThrust]
            Input:
                thrust: thrust \in R (unit:N) has shape (1,N)
        '''
        return np.clip(thrust,self.minCollectiveThrust(),self.maxCollectiveThrust())
    
    def clipMotorThrust(self,thrust):
        '''
            Clip motor thrust to the range [thrust_min,thrust_max]
            Input:
                thrust: thrust \in R (unit:N) has shape (4,N)
        '''
        return np.clip(thrust,self.thrust_min,self.thrust_max)