# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Overwrite model task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=300, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=102, help="Random seed.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=1,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 1.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from typing import TYPE_CHECKING

import dill
import gymnasium as gym
import hydra
import numpy as np
import robomimic.utils.torch_utils as TorchUtils
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from isaaclab_tasks.utils import parse_env_cfg

import isaac_imitation_learning.tasks  # noqa: F401

if TYPE_CHECKING:
    from diffusion_policy.workspace.base_workspace import BaseWorkspace


def rollout(policy, env, horizon, abs_action, num_success_steps):
    obs = env.reset()

    obs_buffer = {}
    for term in obs:
        obs_buffer[term] = torch.stack([obs[term], obs[term]], dim=1)

    step = 0
    success_counter = 0
    done = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    success_steps = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)

    while step < horizon and not torch.all(done):
        # device transfer
        obs_dict = dict_apply(obs_buffer, lambda x: x.to(device=policy.device))

        # run policy
        with torch.no_grad():
            actions = policy.predict_action(obs_dict)

        # device transfer
        actions = dict_apply(actions, lambda x: x.detach().to(env.unwrapped.device))
        actions = actions["action"]

        if not torch.all(torch.isfinite(actions)):
            print(actions)
            raise RuntimeError("Nan or Inf action")

        # step env
        if abs_action:
            actions = undo_transform_action(actions)

        # swap axis (N_envs, Ta, Da) -> (Ta, N_envs, Da)
        actions = torch.moveaxis(actions, 0, 1)

        for act in actions:
            # only execute action if env is not done
            act = act * torch.logical_not(done)[:, None]
            obs, rewards, terminations = env.step(act)
            done = torch.logical_or(done, terminations)

            for term in obs:
                obs_buffer[term][:, 1] = obs_buffer[term][:, 0]
                obs_buffer[term][:, 0] = obs[term]

            success_state = env.unwrapped.get_success_state()
            success_steps = (success_steps + 1) * success_state
            success = success_steps >= num_success_steps
            done = torch.logical_or(done, success)
            success_counter += success.sum()

            step += 1

    return success_counter


def undo_transform_action(self, action):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = self.rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""

    # Load policy
    payload = torch.load(open(args_cli.checkpoint, "rb"), pickle_module=dill)  # noqa: SIM115
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # parse configuration
    # TODO: Check if task ID exists
    task = args_cli.task if args_cli.task is not None else cfg.task.task_id
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)

    # env_cfg.viewer.eye = (1.3, 0, 0.4)
    # env_cfg.viewer.resolution = (1920, 1080)

    # Set seed
    torch.manual_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed

    env_cfg.disable_success_reset = True

    # Create environment
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(env, video_folder="/home/timo/", video_length=200, disable_logger=True, fps=30)

    env_wrapper = hydra.utils.instantiate(cfg.task.env_runner.env_wrapper)
    env = env_wrapper(env=env)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Run policy

    success = 0
    for trial in range(args_cli.num_rollouts):
        success += rollout(policy, env, args_cli.horizon, cfg.task.abs_action, args_cli.num_success_steps)

    env.close()

    print(
        f"Success: {success} / {args_cli.num_rollouts * args_cli.num_envs} ({success / (args_cli.num_rollouts * args_cli.num_envs)})"
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
