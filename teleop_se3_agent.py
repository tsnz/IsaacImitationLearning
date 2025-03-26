# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="IIL-Lift-Cube-v0", help="Name of the task.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="gamepad",
    help="Device for interacting with environment.",
)
parser.add_argument("--sensitivity", type=float, default=0.5, help="Sensitivity factor.")
parser.add_argument(
    "--simpub",
    action="store_true",
    default=False,
    help="Enable SimPub.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f"{os.environ['ISAACLAB_PATH']}/apps/isaaclab.python.xr.openxr.kit"

if "simpub" in args_cli.teleop_device.lower():
    args_cli.simpub = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import omni.log  # noqa: F401
import torch
from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils import parse_env_cfg
from simpub.sim.isaacsim_publisher import IsaacSimPublisher

import tasks  # noqa: F401
from devices.SimPub.se3_SimPubHandTracking import Se3SimPubHandTrackingRel


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)


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


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    rate_limiter = (
        None
        if args_cli.teleop_device.lower() == "handtracking" or args_cli.step_hz == 0
        else RateLimiter(args_cli.step_hz)
    )

    # parse configuration
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # modify configuration

    # make lowdim to improve performance
    # env_cfg.make_lowdim_only()

    # remove timeout
    env_cfg.terminations.time_out = None

    # reset teleop if env is reset
    def teleop_reset(env, env_ids):
        teleop_interface.reset()

    env_cfg.events.reset_teleop = EventTerm(func=teleop_reset, mode="reset")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # add teleoperation key for env reset
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
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.05 * args_cli.sensitivity,
        )
        teleop_interface.add_callback("R", reset_recording_instance)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.05 * args_cli.sensitivity,
        )
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
        teleop_interface = Se3SimPubHandTrackingRel()
        teleop_interface.add_callback("Y", reset_recording_instance)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
        )

    print(teleop_interface)

    # reset environment
    env.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()

            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.device).repeat(env.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)

            # apply actions
            env.step(actions)

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
