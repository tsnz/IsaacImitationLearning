"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)  # noqa: SIM115
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)  # noqa: SIM115


import pathlib
from typing import TYPE_CHECKING

import hydra
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.joinpath("cfg")))
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        # shut down IsaacLab if it is running
        try:
            from omni.kit.app import get_app

            get_app().shutdown()

        except ModuleNotFoundError:
            # module not found means IsaacLab is not running, nothing to shut down
            pass
