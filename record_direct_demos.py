# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="IIL-Lift-Cube-v0-LowDim", help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="gamepad",
    help="Device for interacting with environment.",
)
parser.add_argument("--sensitivity", type=float, default=1, help="Sensitivity factor.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./data/dataset.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--simpub",
    action="store_true",
    default=False,
    help="Enable SimPub.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed for simulation.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f"{os.environ['ISAACLAB_PATH']}/apps/isaaclab.python.xr.openxr.kit"

if "simpub" in args_cli.teleop_device.lower():
    args_cli.simpub = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import time

import gymnasium as gym
import omni.log  # noqa: F401
import torch
from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaac_imitation_learning.tasks  # noqa: F401

# only import simpub if needed, so user is not forced to install it
if args_cli.simpub:
    from simpub.sim.isaacsim_publisher import IsaacSimPublisher

    from isaac_imitation_learning.devices.SimPub.se3_SimPubHandTracking import Se3SimPubHandTrackingRel


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def add_to_episode(key: str, value: torch.Tensor | dict, episode: EpisodeData):
    """Adds the given key-value pair to the episodes for the given environment ids.

    Args:
        key: The key of the given value to be added to the episodes. The key can contain nested keys
            separated by '/'. For example, "obs/joint_pos" would add the given value under ['obs']['policy']
            in the underlying dictionary in the episode data.
        value: The value to be added to the episodes. The value can be a tensor or a nested dictionary of tensors.
            The shape of a tensor in the value is (env_ids, ...).
        env_ids: The environment ids. Defaults to None, in which case all environments are considered.
    """

    # resolve environment ids
    if key is None:
        return

    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            add_to_episode(f"{key}/{sub_key}", sub_value, episode)
        return

    episode.add(key, value[0])


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
    gripper_vel[:] = -1 if gripper_command else 1
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    rate_limiter = (
        None
        if args_cli.teleop_device.lower() == "handtracking" or args_cli.step_hz == 0
        else RateLimiter(args_cli.step_hz)
    )

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=1)
    assert isinstance(env_cfg, DirectRLEnvCfg), "For recording manager env demos use record_demos.py"
    env_cfg.env_name = args_cli.task

    env_cfg.seed = args_cli.seed

    # disable success checking function to invoke in the main loop
    env_cfg.disable_success_reset = True
    env_cfg.disable_timeout_reset = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # add teleoperation key for reset current recording instance
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # enable simpub if selected
    if args_cli.simpub and env.unwrapped.sim is not None and env.unwrapped.sim.stage is not None:
        print("parsing usd stage...")
        IsaacSimPublisher(host="127.0.0.1", stage=env.unwrapped.sim.stage)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
        teleop_interface.add_callback("R", reset_recording_instance)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "gamepad":
        from carb.input import GamepadInput

        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity,
            rot_sensitivity=0.1 * args_cli.sensitivity,
            dead_zone=0.2,
        )
        teleop_interface.add_callback(GamepadInput.B, reset_recording_instance)
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", reset_recording_instance)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)
    elif args_cli.teleop_device.lower() == "simpub":
        teleop_interface = Se3SimPubHandTrackingRel(
            pos_sensitivity=(args_cli.sensitivity / env_cfg.decimation),
            rot_sensitivity=(args_cli.sensitivity / env_cfg.decimation),
        )
        teleop_interface.add_callback("Y", reset_recording_instance)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking', 'gamepad', 'simpub'."
        )

    print(teleop_interface)

    # setup recording
    output_file = HDF5DatasetFileHandler()
    output_file.create(args_cli.dataset_file, env_cfg.env_name)

    # reset before starting
    obs = env.reset()[0]["policy"]
    episode_data = EpisodeData()
    initial_state = env.get_env_state(0)
    add_to_episode("initial_state", initial_state, episode_data)

    env.reset_to(episode_data.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)
            add_to_episode("obs", obs, episode_data)
            add_to_episode("actions", actions, episode_data)

            # perform action on environment
            # TODO: Support for envs not using full range of SE3 input
            obs = env.step(actions)[0]["policy"]

            success = env.get_success_state()
            if bool(success):
                success_step_count += 1
                if success_step_count >= args_cli.num_success_steps:
                    episode_data.seed = args_cli.seed
                    episode_data.success = True
                    output_file.write_episode(episode_data)
                    current_recorded_demo_count += 1
                    should_reset_recording_instance = True
            else:
                success_step_count = 0

            if should_reset_recording_instance:
                obs = env.reset()[0]["policy"]
                teleop_interface.reset()
                episode_data = EpisodeData()
                should_reset_recording_instance = False
                initial_state = env.get_env_state(0)
                add_to_episode("initial_state", initial_state, episode_data)

            if args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()
    output_file.flush()
    output_file.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
