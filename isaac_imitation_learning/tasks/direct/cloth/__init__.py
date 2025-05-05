import gymnasium as gym

from .fold_cloth_direct_env import FoldClothEnv
from .fold_cloth_direct_env_cfg import FoldClothEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="IIL-Fold-Cloth-Direct-v0",
    entry_point=FoldClothEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FoldClothEnvCfg,
    },
)
