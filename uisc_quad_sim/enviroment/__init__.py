# grm wrapper for uisc quadrotor simulator
from gymnasium import spaces
from rsl_rl.env import VecEnv
import numpy as np

from ..simulations import VecQuadSim, QuadSimParams
from ..visualize import DroneVisualizer


class BaseEnv(VecEnv):
    "Base environment class, handling action normalization and basic configuration"

    def __init__(self, paras: QuadSimParams):
        self._quadsim_params = paras
        self.sim = VecQuadSim(paras)
        self.quad = self.sim.quadrotor
        self._num_envs = paras.nums
        self._observation_space = spaces.Box(
            np.ones(self._get_obs_dim()) * -np.inf,
            np.ones(self._get_obs_dim()) * np.inf,
            dtype=np.float32,
        )
        self._action_space = spaces.Box(
            low=np.ones(self._get_act_dim()) * -1.0,
            high=np.ones(self._get_act_dim()) * 1.0,
            dtype=np.float32,
        )
        self.render_mode = ["human"] * self._num_envs
        self.__init_render = False

    def _get_act_dim(self) -> int:
        raise NotImplementedError

    def _get_obs_dim(self) -> int:
        raise NotImplementedError

    def _scale_ctbr(self, actions):
        ct = (actions[:, 0] + 1) / 2.0 * self.quad.maxCollectiveThrust  # linear mapping
        br = actions[:, 1:] * self.quad.maxBodyrates  # ignore minBodyrates
        return np.concatenate([ct[:, None], br], axis=1)

    def _scale_ctbm(self, actions):
        raise NotImplementedError  # not planned

    def _scale_srt(self, actions):
        srt = (actions + 1) / 2.0 * self.quad.maxThrust
        return srt

    def _get_obs(self):
        raise NotImplementedError

    def _get_reward(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def step(self, actions):
        if self.sim.mode == QuadSimParams.CTBR:
            actions = self._scale_ctbr(actions)
        elif self.sim.mode == QuadSimParams.CTBM:
            actions = self._scale_ctbm(actions)
        elif self.sim.mode == QuadSimParams.SRT:
            actions = self._scale_srt(actions)
        else:
            raise ValueError("Unknown control mode")
        # 4*N -> N*4
        actions = actions.T
        return self._step(actions)

    def _step(self, actions):
        self.sim.step(actions)
        obs = self._get_obs()
        reward, done = self._get_reward()
        info = self._get_info()
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
        return self._get_obs()

    def render(self, mode=None):
        if not self.__init_render:
            self.vis = DroneVisualizer(self._num_envs)
            self.__init_render = True
        if mode == "human":
            self.vis.log_states(self.sim.t, self.sim.estimate(gt=True))
