#thanks to https://github.com/uzh-rpg
import numpy as np
import yaml


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
        self._name = name
        self._mass = mass
        self._J = np.diag(inertia)
        self._J_inv = np.linalg.inv(self._J)
        self._arm_length: float = arm_length
        self._kappa = kappa
        self._motor_min = motor_min
        self._motor_max = motor_max
        self._tau = motor_time_constant
        self._tau_inv = 1/motor_time_constant
        self._thrust_map = thrust_map
        self._thrust_max = thrust_max
        self._thrust_min = thrust_min
        self._bodyrates_omega_min = omega_min
        self._bodyrates_omega_max = omega_max
        self._tbm = self._arm_length*(0.5)**0.5*np.array([[1,-1,-1,1],[-1,-1,1,1],[0,0,0,0]])
        self._tbm[2,:] = self._kappa*np.array([1,-1,1,-1])
        # Init variables
        self._min_collective_thrust = None
        self._max_collective_thrust = None
        self._B = None
        self._B = self.allocatioMatrix #allocation matrix B@thrust[4] = u[thrust,torques]
        # B@[t1,t2,t3,t4] 
        self._B_inv = np.linalg.pinv(self._B)
        self._drag_coeff = drag_coeff
        arm_l_xy = self._arm_length*(0.5)**0.5
        self._max_torque = np.array([2*arm_l_xy*self._thrust_max,
                                    2*arm_l_xy*self._thrust_max,
                                    2*self._kappa*self._thrust_max])
        self._min_torque = -self._max_torque

    def __str__(self):
        return f'''Quadrotor {self._name}
    Mass: {self._mass} kg
    Inertia: {np.diag(self._J)}
    Arm Length: {self._arm_length} m
    Kappa: {self._kappa} Nm/N
    Drag Coefficient: {self._drag_coeff} N/(m/s)
    Motor Speed Range: [{self._motor_min},{self._motor_max}]
    Motor Time Constant: {self._tau} s
    Thrust Map: {self._thrust_map}
    Thrust Range: [{self._thrust_min},{self._thrust_max}]'''
    
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
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def inertia(self):
        return self._J
    
    @property
    def arm_length(self):
        return self._arm_length
    
    @property
    def inertia_inv(self):
        return self._J_inv
    
    @property
    def allocatioMatrix(self):
        '''
            Calculate the allocation matrix
            Output:
                B: allocation matrix \in R^{4,4}
        '''
        if self._B is not None:
            return self._B
        self._B = np.ones((4,4))
        self._B[1:4,:] = self._tbm
        return self._B

    def clipMotorSpeed(self,omega):
        '''
            Clip motor speed to the range [motor_min,motor_max]
            Input:
                omega: motor speed \in R has shape (4,N)
        '''
        return np.clip(omega,self._motor_min,self._motor_max)
        
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
        thrust[:] = self._thrust_map[0]*omega2 + self._thrust_map[1]*omega + self._thrust_map[2]
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
        c2 = self._thrust_map[0] 
        c1 = self._thrust_map[1]
        c0 = self._thrust_map[2]
        scale = 1/(2*c2)
        offset = -c1*scale
        root = np.sqrt(c1**2 - 4*c2*(c0 - thrust))
        return offset + root*scale


    def clipThrust(self,thrust):
        '''
            Clip thrust to the range [thrust_min,thrust_max]
            Input:
                thrust: thrust \in R (unit:N) has shape (4,N)
        '''
        return np.clip(thrust,self._thrust_min,self._thrust_max)
    
    @property
    def minThrust(self):
        '''
            minimum thrust
            return: min_thrust \in R
        '''
        return self._thrust_min
    
    @property
    def maxThrust(self):
        '''
            maximum thrust
            return: max_thrust \in R
        '''
        return self._thrust_max

    @property
    def minCollectiveThrust(self):
        '''
            Calculate minimum collective thrust
            return: min_collective_thrust \in R
        '''
        if self._min_collective_thrust is not None:
            return self._min_collective_thrust
        self._min_collective_thrust = self._thrust_min*4/self._mass
    
    @property
    def maxCollectiveThrust(self):
        '''
            Calculate maximum collective thrust
            return: max_collective_thrust \in R
        '''
        if self._max_collective_thrust is not None:
            return self._max_collective_thrust
        self._max_collective_thrust = self._thrust_max*4/self._mass
        return self._thrust_max*4/self._mass
    
    @property
    def minBodyrates(self):
        '''
            minimum body rates
            return: omega_min \in R^3
        '''
        return self._bodyrates_omega_min
    
    @property
    def maxBodyrates(self):
        '''
            maximum body rates
            return: omega_max \in R^3
        '''
        return self._bodyrates_omega_max

    @property
    def minTorques(self):
        '''
            minimum torques
            return: min_torque \in R^3
        '''
        return self._min_torque
    
    @property
    def maxTorques(self):
        '''
            maximum torques
            return: max_torque \in R^3
        '''
        return self._max_torque
    

    def clipCollectiveThrust(self,thrust):
        '''
            Clip collective thrust to the range [minCollectiveThrust,maxCollectiveThrust]
            Input:
                thrust: thrust \in R (unit:N) has shape (1,N)
        '''
        return np.clip(thrust,self.minCollectiveThrust,self.maxCollectiveThrust)
    
    def clipMotorThrust(self,thrust):
        '''
            Clip motor thrust to the range [thrust_min,thrust_max]
            Input:
                thrust: thrust \in R (unit:N) has shape (4,N)
        '''
        return np.clip(thrust,self._thrust_min,self._thrust_max)