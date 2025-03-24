import gymnasium as gym

from .cube import franka_lift_cube_env_cfg
from .teddy import franka_lift_teddy_env_cfg

gym.register(
    id="IIL-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_cube_env_cfg.FrankaLiftCubeEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-Cube-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_cube_env_cfg.FrankaLiftCubeLowDimEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-Teddy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_teddy_env_cfg.FrankaLiftTeddyEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-Teddy-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_teddy_env_cfg.FrankaLiftTeddyLowDimEnvCfg},
    disable_env_checker=True,
)
