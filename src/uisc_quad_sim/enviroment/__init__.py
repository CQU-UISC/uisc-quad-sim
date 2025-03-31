# grm wrapper for uisc quadrotor simulator
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from uisc_quad_sim.quadrotor.quadrotor import Quadrotor
from ..simulations import VecQuadSim,QuadSimParams

class BaseEnv(gym.Env):
    """基础环境类，处理动作归一化和基本配置"""
    metadata = {'render.modes': ['human']}

    def __init__(self, sim_params:QuadSimParams, num_envs=1):
        super().__init__()
        
        # 初始化仿真环境
        self.sim = VecQuadSim(sim_params)
        self.num_envs = num_envs
        
        # 从四旋翼参数获取动作范围
        self.quad_params: Quadrotor = sim_params.quad
        self.control_mode = self.sim.mode
        
        # 定义动作和观测空间
        self._define_spaces()
        
        # 初始化动作缩放参数
        self._setup_action_scaling()

    def _define_spaces(self):
        """定义Gym空间"""
        # 观测空间：位置(3) + 速度(3) + 四元数(4) + 角速度(3) = 13维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # 动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _setup_action_scaling(self):
        """设置动作缩放参数"""
        # 根据控制模式配置缩放参数
        if self.control_mode == QuadSimParams.CTBR:
            self.action_scale = {
                'thrust': (0.0, self.quad_params.max_collective_thrust),
                'rates': (-self.quad_params.max_angular_rates, 
                          self.quad_params.max_angular_rates)
            }
        elif self.control_mode == QuadSimParams.CTBM:
            self.action_scale = {
                'thrust': (0.0, self.quad_params.max_collective_thrust),
                'torques': (-self.quad_params.max_torques, 
                           self.quad_params.max_torques)
            }
        elif self.control_mode == QuadSimParams.SRT:
            self.action_scale = {
                'motor_thrust': (self.quad_params.motor_min, 
                                self.quad_params.motor_max)
            }

    def _convert_action(self, action):
        """将归一化动作(-1,1)转换为实际物理量"""
        action = np.clip(action, -1.0, 1.0)
        
        if self.control_mode == QuadSimParams.CTBR:
            # 转换推力(0~max)和角速度(-max~max)
            scaled_action = np.zeros_like(action)
            scaled_action[0] = (action[0] + 1) * 0.5 * self.action_scale['thrust'][1]
            scaled_action[1:] = action[1:] * self.action_scale['rates'][1]
            
        elif self.control_mode == QuadSimParams.CTBM:
            # 转换推力(0~max)和力矩(-max~max)
            scaled_action = np.zeros_like(action)
            scaled_action[0] = (action[0] + 1) * 0.5 * self.action_scale['thrust'][1]
            scaled_action[1:] = action[1:] * self.action_scale['torques'][1]
            
        elif self.control_mode == QuadSimParams.SRT:
            # 转换单个转子推力
            scaled_action = (action + 1) * 0.5 * (
                self.action_scale['motor_thrust'][1] - self.action_scale['motor_thrust'][0]
            ) + self.action_scale['motor_thrust'][0]
        
        return scaled_action

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        pass  # 可添加可视化逻辑

    def close(self):
        pass
