import gymnasium as gym

from .factory_env_cfg import FactoryTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="IIL-Factory-PegInsert-Direct-v0",
    entry_point="tasks.direct.factory.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
    },
)
