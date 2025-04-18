# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import Tuple

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="data/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./data/dataset_obs_gen.hdf5",
    help="File path to export rerecorded obs to.",
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--skip_first_n_steps",
    type=int,
    default=-1,
    help=(
        "Skip recording the first n steps. Intended as a workaround for incorrect camera sensor data for a few frames "
        "after a rest"
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import omni.log  # noqa: F401
import torch
from isaaclab.devices import Se3Keyboard
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaac_imitation_learning.tasks  # noqa: F401

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> Tuple[bool, str]:
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type]:
            for state_name in runtime_state[asset_type][asset_name]:
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    episode_names = list(dataset_file_handler.get_episode_names())

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    # get seed assuming the seeds for all episodes are the same
    # for more info see https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html
    episode_data = dataset_file_handler.load_episode(episode_names[0], "cpu")
    if episode_data.seed is not None:
        env_cfg.seed = int(episode_data.seed)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to skip demos if not successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states:
        state_validation_enabled = True

    # reset before starting
    env.reset()
    # workaround for blank frame at start issue
    # https://isaac-sim.github.io/IsaacLab/main/source/refs/issues.html#blank-initial-frames-from-the-camera
    sim = SimulationContext.instance()
    # note: the number of steps might vary depending on how complicated the scene is.
    for _ in range(12):
        sim.render()

    # simulate environment -- run everything in inference mode
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # env_episode_data_map = {index: EpisodeData() for index in range(1)}
            env_episode_data_map = EpisodeData()
            first_loop = True
            has_next_action = True
            step_idx = 0
            while has_next_action:
                # initialize actions with zeros so those without next action will not move
                actions = torch.zeros(env.action_space.shape)
                has_next_action = False

                # hacky workaround for incorrect image sensor output for a few frames after reset
                # https://isaac-sim.github.io/IsaacLab/main/source/refs/issues.html#stale-values-after-resetting-the-environment
                if args_cli.skip_first_n_steps == step_idx:
                    env.recorder_manager.reset()

                env_next_action = env_episode_data_map.get_next_action()
                if env_next_action is None:
                    # store replay obs
                    if not first_loop:
                        store_obs = True
                        if success_term is not None:
                            # check success term
                            store_obs = bool(success_term.func(env, **success_term.params)[0])
                            if not store_obs:
                                print(f"{replayed_episode_count:4}: success term not true, obs not saved")

                        if store_obs:
                            print(f"{replayed_episode_count:4}: Obs saved")
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.get_episode(0).seed = env_cfg.seed
                            env.recorder_manager.set_success_to_episodes(
                                [0],
                                torch.tensor([[True]], dtype=torch.bool, device=env.device),
                            )
                            env.recorder_manager.export_episodes([0])

                    env.recorder_manager.reset()

                    next_episode_index = None
                    while episode_indices_to_replay:
                        next_episode_index = episode_indices_to_replay.pop(0)
                        if next_episode_index < episode_count:
                            break
                        next_episode_index = None

                    if next_episode_index is not None:
                        replayed_episode_count += 1
                        print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{0}")
                        episode_data = dataset_file_handler.load_episode(episode_names[next_episode_index], env.device)
                        env_episode_data_map = episode_data
                        # Set initial state for the new episode
                        initial_state = episode_data.get_initial_state()
                        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
                        # Get the first action for the new episode
                        env_next_action = env_episode_data_map.get_next_action()
                        has_next_action = True
                        step_idx = 0
                    else:
                        continue
                else:
                    has_next_action = True
                actions[0] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                env.step(actions)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map.get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map.next_state_index - 1:4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)

                step_idx += 1
            break
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
