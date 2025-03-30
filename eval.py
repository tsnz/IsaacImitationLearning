"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import json
import os
import pathlib
from typing import TYPE_CHECKING

import click
import dill
import hydra
import torch

import wandb

if TYPE_CHECKING:
    from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-s", "--show", is_flag=True, show_default=True, default=False)
def main(checkpoint, output_dir, device, show):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)  # noqa: SIM115

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    if show:
        # turn of headless mode for visualzation
        cfg.task.env_runner.isaac_args.headless = False

    # disable dummy rollout
    cfg.task.env_runner.dummy_rollout = False

    # run eval
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)  # noqa: SIM115


if __name__ == "__main__":
    try:
        main()
    finally:
        # shut down IsaacLab if it is running
        try:
            from omni.kit.app import get_app

            get_app().shutdown()

        except ModuleNotFoundError:
            # module not found means IsaacLab is not running, nothing to shut down
            pass
