import numpy as np
import yaml
from loguru import logger

import os
from .base import Sim
from ..quadrotor.quadrotor import Quadrotor
from ..dynamics import QuadrotorDynamics,MotorDynamics,disturbance
from ..dynamics.disturbance import Disturbance
from ..controller.low_level_simple import LowlevelSimpleController
class QuadSimParams:
    CTBR = "CTBR"
    CTBM = "CTBM"
    SRT = "SRT"
    control_modes = [CTBR,CTBM,SRT]
    
    def __init__(self,
                 dt:float,
                 low_level_dt:float,
                 disturb:Disturbance,
                 quad:Quadrotor,
                 g:float,
                 nums:int,
                 mode:int,
                 noise_std:np.ndarray) -> None:
        self.dt = dt
        self.low_level_dt = 0.001
        self.disturb = disturb
        self.quad: Quadrotor = quad
        self.g = g
        self.nums = nums
        self.mode = mode
        self.noise_std =  noise_std
    
    def __str__(self) -> str:
        return f'''Quadrotor Simulator Params:
    dt:{self.dt}
    low_level_dt:{self.low_level_dt}
    quadrotor:{self.quad}
    g:{self.g}
    nums:{self.nums}
    control_mode:{QuadSimParams.control_modes[self.mode]}
    noise_std:{self.noise_std}
    disturbance:{self.disturb}'''

    @staticmethod
    def loadFromFile(file_path):
        cfg = yaml.load(open(file_path,'r'),Loader=yaml.FullLoader)
        assert 'dt' in cfg, "dt should be provided"
        assert 'quadrotor' in cfg, "quadrotor config should be provided"
        assert 'g' in cfg, "gravity should be provided"
        assert 'nums' in cfg, "nums should be provided"
        assert 'control_mode' in cfg, "control_mode should be provided"
        assert 'noise_std' in cfg, "noise_std should be provided"
        dt:float = cfg['dt']
        low_level_dt:float = cfg['low_level_dt'] if 'low_level_dt' in cfg else 0.001
        quad = Quadrotor.loadFromFile(os.path.join(os.path.dirname(file_path),cfg['quadrotor']))
        g:float = cfg['g']
        nums:int = cfg['nums']
        noise_std = np.array(cfg['noise_std'])
        assert noise_std.shape[0] == 13, "noise_std should be 13x1"
        
        # control mode: bodyrates or torques
        mode = 0
        if cfg['control_mode'] ==  QuadSimParams.control_modes[0]:
            mode = 0
        elif cfg['control_mode'] == QuadSimParams.control_modes[1]:
            mode = 1
        elif cfg['control_mode'] == QuadSimParams.control_modes[2]:
            mode = 2
        else:
            raise ValueError("control_mode should be one of {}".format(QuadSimParams.control_modes))
        
        # Now hard code the disturbance
        disturb = disturbance.EmptyField()
        if cfg['disturbance']['type'] == 'const':
            assert 'force' in cfg['disturbance'], "force should be provided"
            assert 'moment' in cfg['disturbance'], "moment should be provided"
            # size of force and moment should be 3
            assert len(cfg['disturbance']['force']) == 3, "force should be 3x1"
            assert len(cfg['disturbance']['moment']) == 3, "moment should be 3x1"
            force = np.array(cfg['disturbance']['force'])
            moment = np.array(cfg['disturbance']['moment'])
            disturb = disturbance.ConstField(force,moment)
        elif cfg['disturbance']['type'] == 'empty':
            disturb = disturbance.EmptyField()
        elif cfg['disturbance']['type'] == 'wind':
            assert 'pos' in cfg['disturbance'], "pos should be provided"
            assert 'to' in cfg['disturbance'], "to should be provided"
            assert 'vmax' in cfg['disturbance'], "vmax should be provided"
            assert 'radius' in cfg['disturbance'], "radius should be provided"
            assert 'noisevar' in cfg['disturbance'], "noisevar should be provided"
            # pos and to should be 3x1
            assert len(cfg['disturbance']['pos']) == 3, "pos should be 3x1"
            assert len(cfg['disturbance']['to']) == 3, "to should be 3x1"
            pos = np.array(cfg['disturbance']['pos'])
            to = np.array(cfg['disturbance']['to'])
            disturb = disturbance.WindField(pos,to,cfg['disturbance']['vmax'],cfg['disturbance']['radius'],cfg['disturbance']['noisevar'])
        elif cfg['disturbance']['type'] == 'timevar':
            assert 'force' in cfg['disturbance'], "force should be provided"
            assert 'moment' in cfg['disturbance'], "moment should be provided"
            # size of force and moment should be 3
            assert len(cfg['disturbance']['force']) == 3, "force should be 3x1"
            assert len(cfg['disturbance']['moment']) == 3, "moment should be 3x1"
            force = np.array(cfg['disturbance']['force'])
            moment = np.array(cfg['disturbance']['moment'])
            disturb = disturbance.TimeVarField(force,moment,low_level_dt)
        return QuadSimParams(dt,low_level_dt,disturb,quad,g,nums,mode,noise_std)

'''
Quadrotor simulator
'''
class VecQuadSim(Sim):
    def __init__(self, sim:QuadSimParams) -> None:
        self._low_level_dt = 0.001 #angle velocity control run at 1000Hz
        self._sim_cfg = sim
        self._quad = sim.quad
        self._low_level_steps = np.ceil(self._sim_cfg.dt/self._low_level_dt).astype(int)
        assert np.isclose(self._low_level_steps*self._low_level_dt,self._sim_cfg.dt), "low level dt should be multiple of high level dt"
        self._rigid_dynamics=QuadrotorDynamics(self._quad._mass,
                                                self._sim_cfg.g,
                                                self._quad._J,
                                                self._quad._J_inv,
                                                self._quad._drag_coeff,
                                                self._sim_cfg.disturb)
        self._low_level_ctrl = LowlevelSimpleController(self._low_level_dt,self._quad._J)
        self._motor_dynamics=MotorDynamics(self._quad._tau_inv)
        super().__init__(self._low_level_dt)
        logger.info("Control Mode:{}".format(QuadSimParams.control_modes[self._sim_cfg.mode]))
        #0  1  2  3  4  5  6  7  8  9  10 11 12
        #px py pz vx vy vz qw qx qy qz wx wy wz
        self.mode = QuadSimParams.control_modes[self._sim_cfg.mode]
        self.reset()

    @property
    def sim_dt(self):
        return self._sim_cfg.dt

    @property
    def quadrotor(self):
        return self._quad
    
    @property
    def disturbance(self):
        return self._rigid_dynamics.disturb
    
    @property
    def motor_speed(self):
        return self._motors_omega
    
    @property
    def state(self):
        return self._x
    

    def _motor_commands(self,u:np.ndarray)->np.ndarray:
        '''
            Calculate motor commands from control inputs
            Input:
                u: control inputs [thrust,torques] unit: [N,Nm] shape:[4,N]
            Output:
                omega: motor commands [4,N] unit: [0,1]
        '''
        
        '''Uncomment this block to use close form solution'''
        motor_thrust = self._quad.allocatioMatrixInv @ u
        cliped_motor_thrust = self._quad.clipMotorThrust(motor_thrust)
        motor_cmd = self._quad.thrustMapInv(
            cliped_motor_thrust
        )

        '''Uncomment this block to use more accurate motor thrust allocation (current not support vectorized input)'''
        # assert u.shape[1] == 1, "Currently only support single control input"
        # m_sp = u[:,0]
        # m_sp = np.matrix([m_sp[1], m_sp[2], m_sp[3], m_sp[0]]).T
        # P = self._quad.Bm_inv
        # (motor_thrust, motor_thrust_new) = normal_mode(
        #     m_sp,
        #     P/ self._quad.thrust_map[1],
        #     self._quad.motor_min,
        #     self._quad.motor_max
        # )
        # logger.debug("raw motor omega:{}, new motor omega:{}".format(motor_thrust.T,motor_thrust_new.T))
        # motor_thrust = motor_thrust_new
        # motor_cmd = np.array(motor_thrust)
        logger.debug("compute motor commands, thrust_torque:{}N, motor thrust:{}N, motor commands:{}".format(u.T,motor_thrust.T,motor_cmd.T))
        return motor_cmd 
    
    def _step_motor(self,motor_cmd:np.ndarray)->np.ndarray:
        '''
            Step the simulation by one time step
            Input:
                motor_cmd: control inputs [motor omega] unit: [0,1] shape:[4,N]
            Output:
                thrust_torque: [thrust,torques] unit: [N,Nm] shape:[4,N]
        '''
        new_omega = self._run(
            self._motor_dynamics.dxdt,
            self._motors_omega,
            motor_cmd,
        )

        logger.debug("motor omega:{}, new motor omega:{}".format(self._motors_omega.T,new_omega.T))
        self._motors_omega = self._quad.clipMotorSpeed(new_omega)#4xN
        thrust = self._quad.thrustMap(self._motors_omega) #4xN
        logger.debug("motor thrust:{}N".format(thrust.T))
        thrust_torque = self._quad._B @ thrust #4xN
        return thrust_torque
        
    
    def _step_rigid(self,u:np.ndarray)->np.ndarray:
        '''
            Step the simulation by one time step
            u: control inputs [bodyrates] unit: [rad/s] shape:[3,N]
        '''
        self._x = self._run(
            self._rigid_dynamics.dxdt,
            self._x,
            u,
        )
        return self._x
    

    def step(self,u):
        for _ in range(self._low_level_steps):
            # step the simulation by one time step
            # u: control inputs [thrust,torques] unit: [m/s^2,Nm] shape:[4,N]
            # or
            # u: control inputs [thrust,bodyrates] unit: [m/s^2,rad/s] shape:[4,N]
            # u = u[:,None]
            self._x = self._step(u)
        return self._x
    
    def _log(self,state,control_setpoint):
        '''
            Log the simulation
        '''
        if not hasattr(self,'_state_log'):
            self._state_log = []
        if not hasattr(self,'_setpoint_log'):
            self._setpoint_log = []
        self._state_log.append(state)
        self._setpoint_log.append(control_setpoint)
        return

    def _step(self,u)->np.ndarray:
        '''
            Step the simulation by one time step using control inputs
            Input:
                u: control inputs [thrust,torques] unit: [m/s^2,Nm] shape:[4,N]
                or
                u: control inputs [thrust,bodyrates] unit: [m/s^2,rad/s] shape:[4,N]
            Output:
                state: state after stepping the simulation
                
            # Simulation steps:
            - If control mode is bodyrates, calculate moment from control inputs. u = [collective_thrust,moment], 4xN
            - Calculate desired motor commands from control inputs. u = [motor omega], 4xN
            - Step the motor dynamics by one time step, get new motor omega. 4xN
            - Calculate thrust and torques from motor omega. 4xN [collective_thrust,torques]
            - Step the rigid body dynamics by one time step, get new state. 13xN
        '''
        super()._step_t()
        u_sp = u
        logger.debug("control inputs:{}".format(u.T))
        if self.mode==QuadSimParams.CTBR:
            # CTBR
            tbm = np.zeros((4,self._sim_cfg.nums))
            tbm[1:4] = self._low_level_ctrl.compute_control(self._x, u_sp[1:4])
            tbm[0] = self._quad.clipCollectiveThrust(u[0])*self._quad._mass #clip thrust=>[N]
            motor_cmd = self._motor_commands(tbm)#4xN omega in [0,1]
        elif self.mode==QuadSimParams.CTBM:
            # CTBM
            tbm = np.zeros((4,self._sim_cfg.nums))
            tbm[1:4] = u[1:4]#body torque
            tbm[0] = self._quad.clipCollectiveThrust(u[0])*self._quad._mass #clip thrust=>[N]
            motor_cmd = self._motor_commands(tbm)#4xN omega in [0,1]
        elif self.mode==QuadSimParams.SRT:
            # SRT
            cliped_motor_thrust = self._quad.clipMotorThrust(u)
            motor_cmd = self._quad.thrustMapInv(cliped_motor_thrust)
        # step dynamics
        thrust_torque = self._step_motor(motor_cmd=motor_cmd)#4xN, [thrust(m/s^2),torques]
        self._x = self._step_rigid(thrust_torque)
        self._x[6:10,:] /= np.linalg.norm(self._x[6:10,:]) #norm quaternion
        self._log(self._x,u_sp)
        return self._x
   
    def reset_pos(self,p:np.ndarray):
        '''
            Reset the position of the quadrotor
            Input:
                p: position [x,y,z] unit: [m] shape:[3]
        '''
        self._x[0:3,:] = p[:,None]
        return

    def set_seed(self, seed:int):
        return np.random.seed(seed)

    def reset(self,mean:np.ndarray=None,std:np.ndarray=None):
        '''
            Reset the simulation
            Input:
                rand: if True, reset the state with random values
                mean: mean of the state 13x1
                std: standard deviation of the state 13x1
                13 * N
                4 * N
        '''
        super().reset()
        rand = mean is not None and std is not None
        if rand:
            self._x = np.random.randn(13,self._sim_cfg.nums) * std[:,None] + mean[:,None]
            # norm quaternion
            self._x[6:10] /= np.linalg.norm(self._x[6:10],axis=0)
        else:
            self._x = np.zeros((13,self._sim_cfg.nums))
            self._motors_omega = np.zeros((4,self._sim_cfg.nums))
            self._x[6] = 1
    
    def estimate(self,gt:bool=False)->np.ndarray:
        '''
            Return Unbiased estimate of the state
        '''
        state = self._x
        # motor_thrust = self._quad.thrustMap(self._motors_omega)
        # state = np.concatenate((self._x,motor_thrust),axis=0)
        if gt:
            return state
        noise = np.zeros_like(state)
        noise[:13,:] = np.random.randn(13,1) * self._sim_cfg.noise_std[:,None]
        est = state + noise
        return est
    
class QuadSim(VecQuadSim):
    def step(self,u):
        u = u[:,None]
        return super().step(u)[:,0]
    
    def estimate(self, gt = False):
        return super().estimate(gt)[:,0]
    
        
    @property
    def motor_speed(self):
        return super().motor_speed[:,0]
    
    @property
    def state(self):
        return super().state[:,0]