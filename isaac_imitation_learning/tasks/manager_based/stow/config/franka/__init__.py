import gymnasium as gym

from .teddy import franka_stow_teddy_env_cfg

gym.register(
    id="IIL-Stow-Teddy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_stow_teddy_env_cfg.FrankaStowTeddyEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Stow-Teddy-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_stow_teddy_env_cfg.FrankaStowTeddyLowDimEnvCfg},
    disable_env_checker=True,
)
