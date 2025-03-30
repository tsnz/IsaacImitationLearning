from typing import List

import gymnasium as gym
import torch


class IsaacLabLowdimWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obs_keys: List[str],
    ):
        self.env = env
        self.obs_keys = obs_keys

    def reset(self, *args, **kwargs):
        raw_obs, _ = self.env.reset(*args, **kwargs)
        raw_obs = raw_obs["policy"]
        obs = torch.cat([raw_obs[key] for key in self.obs_keys], dim=2)
        return obs

    def reset_to(self, *args, **kwargs):
        raw_obs, _ = self.env.reset_to(*args, **kwargs)
        raw_obs = raw_obs["policy"]
        obs = torch.cat([raw_obs[key] for key in self.obs_keys], dim=2)
        return obs

    def step(self, action):
        raw_obs, rewards, terminated, timeout, _ = self.env.step(action)
        raw_obs = raw_obs["policy"]
        obs = torch.cat([raw_obs[key] for key in self.obs_keys], dim=2)
        return obs, rewards, torch.logical_or(terminated, timeout)
