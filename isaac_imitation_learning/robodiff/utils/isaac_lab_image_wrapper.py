import gymnasium as gym
import torch


class IsaacLabImageWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        shape_meta: dict,
    ):
        self.env = env
        self.shape_meta = shape_meta

    def process_obs(self, raw_obs):
        obs = {}
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            val = raw_obs[key]
            type = attr.get("type", "low_dim")
            if type == "rgb" or type == "depth":
                val = torch.moveaxis(val, -1, -3)
            if type == "rgb":
                val = val / 255.0
            obs[key] = val

        return obs

    def reset(self, *args, **kwargs):
        raw_obs, _ = self.env.reset(*args, **kwargs)
        raw_obs = raw_obs["policy"]
        obs = self.process_obs(raw_obs)
        return obs

    def reset_to(self, *args, **kwargs):
        raw_obs, _ = self.env.reset_to(*args, **kwargs)
        raw_obs = raw_obs["policy"]
        obs = self.process_obs(raw_obs)
        return obs

    def step(self, action):
        raw_obs, rewards, terminated, timeout, _ = self.env.step(action)
        raw_obs = raw_obs["policy"]
        obs = self.process_obs(raw_obs)
        return obs, rewards, torch.logical_or(terminated, timeout)
