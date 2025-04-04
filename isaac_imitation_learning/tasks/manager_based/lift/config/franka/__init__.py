import gymnasium as gym

from .banana import franka_lift_banana_env_cfg
from .cube import franka_lift_cube_env_cfg
from .foam_brick import franka_lift_foam_brick_env_cfg
from .mustard_bottle import franka_lift_mustard_bottle_env_cfg
from .teddy import franka_lift_teddy_env_cfg
from .tennis_ball import franka_lift_tennis_ball_env_cfg

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

gym.register(
    id="IIL-Lift-Banana-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_banana_env_cfg.FrankaLiftBananaEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-Banana-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_banana_env_cfg.FrankaLiftBananaLowDimEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-MustardBottle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_mustard_bottle_env_cfg.FrankaLiftMustardBottleEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-MustardBottle-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_mustard_bottle_env_cfg.FrankaLiftMustardBottleLowDimEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-TennisBall-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_tennis_ball_env_cfg.FrankaLiftTennisBallEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-TennisBall-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_tennis_ball_env_cfg.FrankaLiftTennisBallLowDimEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-FoamBrick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_foam_brick_env_cfg.FrankaLiftFoamBrickEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="IIL-Lift-FoamBrick-v0-LowDim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": franka_lift_foam_brick_env_cfg.FrankaLiftFoamBrickLowDimEnvCfg},
    disable_env_checker=True,
)
