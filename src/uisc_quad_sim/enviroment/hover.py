from . import BaseEnv
import numpy as np

class HoverEnv(BaseEnv):
    """悬停任务环境"""
    def __init__(self, sim_params, num_envs=1):
        super().__init__(sim_params, num_envs)
        
        # 悬停目标设置
        self.target_pos = np.zeros(3)  # 目标位置（原点）
        self.target_yaw = 0.0          # 目标偏航角
        
        # 奖励参数
        self.pos_coeff = 1.0
        self.vel_coeff = 0.1
        self.ang_vel_coeff = 0.05
        self.action_coeff = 0.01
        
        # 终止条件
        self.max_pos_error = 2.0      # 最大允许位置误差
        self.max_episode_steps = 500  # 最大步数
        self.current_step = 0

    def reset(self):
        """重置环境状态"""
        self.sim.reset()
        
        # 随机化初始位置和姿态
        init_pos = np.random.uniform(
            low=[-0.5, -0.5, 0.5], 
            high=[0.5, 0.5, 1.5], 
            size=(3, self.num_envs)
        )
        self.sim.reset_pos(init_pos)
        
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        """获取带噪声的观测"""
        return self.sim.estimate(gt=False).T  # 转换为(num_envs, 13)

    def _calculate_reward(self, states, actions):
        """计算奖励函数"""
        # 位置误差
        pos_error = np.linalg.norm(states[:, :3] - self.target_pos, axis=1)
        
        # 速度惩罚
        vel_penalty = np.linalg.norm(states[:, 3:6], axis=1)
        
        # 姿态惩罚（四元数转换为欧拉角，仅考虑偏航）
        yaw_error = np.abs(self._quat_to_yaw(states[:, 6:10]) - self.target_yaw)
        
        # 角速度惩罚
        ang_vel_penalty = np.linalg.norm(states[:, 10:13], axis=1)
        
        # 动作平滑惩罚
        action_penalty = np.linalg.norm(actions, axis=1)
        
        # 组合奖励
        reward = (
            - self.pos_coeff * pos_error
            - self.vel_coeff * vel_penalty
            - self.ang_vel_coeff * ang_vel_penalty
            - self.action_coeff * action_penalty
        )
        return reward

    def _quat_to_yaw(self, quaternions):
        """从四元数提取偏航角"""
        w, x, y, z = quaternions.T
        return np.arct2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    def _check_done(self, states):
        """检查终止条件"""
        pos = states[:, :3]
        pos_error = np.linalg.norm(pos - self.target_pos, axis=1)
        
        # 位置超出阈值
        out_of_bounds = pos_error > self.max_pos_error
        
        # 达到最大步数
        max_steps = self.current_step >= self.max_episode_steps
        
        return np.logical_or(out_of_bounds, max_steps)

    def step(self, action):
        """执行环境步进"""
        # 转换动作
        scaled_action = self._convert_action(action)
        
        # 执行仿真步
        self.sim.step(scaled_action.T)  # 转置为(4, num_envs)
        self.current_step += 1
        
        # 获取状态
        states = self.sim.state.T      # (num_envs, 13)
        obs = self._get_obs()
        
        # 计算奖励
        rewards = self._calculate_reward(states, action)
        
        # 检查终止条件
        dones = self._check_done(states)
        
        # 附加信息
        infos = [{} for _ in range(self.num_envs)]
        
        return obs, rewards, dones, infos

# 使用示例
if __name__ == "__main__":
    # 加载配置文件
    pass
    # sim_params = QuadSimParams.loadFromFile("your_config.yaml")
    
    # # 创建矢量化环境
    # env = HoverEnv(sim_params, num_envs=4)
    
    # # 典型使用方式
    # obs = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     next_obs, reward, done, info = env.step(action)
        
    # env.close()