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
parser.add_argument("--sensitivity", type=float, default=0.5, help="Sensitivity factor.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/dataset.hdf5",
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
import omni.log
import torch
from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from simpub.sim.isaacsim_publisher import IsaacSimPublisher

import tasks  # noqa: F401
from devices.SimPub.se3_SimPubHandTracking import Se3SimPubHandTrackingRel


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
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # reset teleop if env is reset
    def teleop_reset(env, env_ids):
        teleop_interface.reset()

    env_cfg.events.reset_teleop = EventTerm(func=teleop_reset, mode="reset")

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
        IsaacSimPublisher(host="192.168.0.103", stage=env.unwrapped.sim.stage)

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
        # TODO: Consider decimation for sensitivity
        teleop_interface = Se3SimPubHandTrackingRel()
        teleop_interface.add_callback("Y", reset_recording_instance)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking', 'gamepad', 'simpub'."
        )

    print(teleop_interface)

    # reset before starting
    env.reset()

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

            # perform action on environment
            # TODO: Support for envs not using full range of SE3 input
            env.step(actions)

            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0],
                            torch.tensor([[True]], dtype=torch.bool, device=env.device),
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                env.recorder_manager.reset()
                env.reset()
                should_reset_recording_instance = False
                success_step_count = 0

            # print out the current demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
