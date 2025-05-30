import collections
import math
import os
from enum import Enum

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.real_world.video_recorder import VideoRecorder
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import wandb
from isaac_imitation_learning.robodiff.utils import VideoRecordingWrapper


class InitMode(Enum):
    TRAIN = 1
    TEST = 2


class IsaacLabLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        env_wrapper,
        dataset_path,
        isaac_args,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        render_video=False,
        render_hw=(256, 256),
        render_camera_pos=(1.2, 0.0, 0.6),
        # w.r.t. parent frame rotate Z (up) 180 deg and y -35 deg to look down
        render_camera_rot=(0, -0.3007058, 0, 0.953717),
        fps=30,
        crf=22,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        dummy_rollout=False,
        persistent_env=True,
    ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if dummy_rollout:
            self.dummy_rollout = dummy_rollout
            return

        self.app_launcher = self._startup_sim(isaac_args)
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # prepare rollout
        sim_device = isaac_args.device

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        env_cfg = parse_env_cfg(env_meta["env_name"], device=sim_device, num_envs=n_envs)
        env_cfg.seed = test_seed

        # Disable all recorders
        env_cfg.recorders = {}

        # use history_length instead of a multi step wrapper
        env_cfg.observations.policy.history_length = env_n_obs_steps
        env_cfg.observations.policy.flatten_history_dim = False

        # Create camera for video if needed
        if render_video:
            env_cfg = self.add_cam_to_env(env_cfg, render_hw, render_camera_pos, render_camera_rot)

        # extract success checking function to invoke during rollout
        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None

        env = None
        if persistent_env:
            env = gym.make(env_meta["env_name"], cfg=env_cfg).unwrapped
            if render_video:
                env = self.add_video_wrapper(env, fps, crf, output_dir)
            env = env_wrapper(env=env)

        else:
            from isaacsim.core.utils.stage import close_stage

            close_stage()
            # needed for env creation
            self.crf = crf
            self.env_cfg = env_cfg
            self.env_wrapper = env_wrapper
            self.fps = fps
            self.output_dir = output_dir

        self.abs_action = abs_action
        self.dataset_path = dataset_path
        self.dummy_rollout = dummy_rollout
        self.env = env
        self.env_meta = env_meta
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.n_latency_steps = n_latency_steps
        self.n_obs_steps = n_obs_steps
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.persistent_env = persistent_env
        self.render_video = render_video
        self.rotation_transformer = rotation_transformer
        self.sim_device = sim_device
        self.success_term = success_term
        self.tqdm_interval_sec = tqdm_interval_sec
        self.train_start_idx = train_start_idx

    def run(self, policy: BaseLowdimPolicy):
        if self.dummy_rollout:
            log_data = dict()
            return log_data

        env = self.env
        success_term = self.success_term
        if not self.persistent_env:
            from isaacsim.core.utils.stage import close_stage, create_new_stage

            create_new_stage()
            env = gym.make(self.env_meta["env_name"], cfg=self.env_cfg).unwrapped
            if self.render_video:
                env = self.add_video_wrapper(env, self.fps, self.crf, self.output_dir)
            env = self.env_wrapper(env=env)

        # plan for rollout
        n_inits = self.n_train + self.n_test
        n_chunks = math.ceil(n_inits / self.n_envs)

        run_info = {}
        run_info["init_mode"] = [InitMode.TRAIN] * self.n_train + [InitMode.TEST] * self.n_test
        run_info["videos"] = {}
        run_info["task_successful"] = [None] * n_inits
        video_recorder_mask = [False] * n_inits
        video_recorder_mask[: self.n_train_vis] = [True] * self.n_train_vis
        video_recorder_mask[self.n_train : self.n_train + self.n_test_vis] = [True] * self.n_test_vis

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.n_envs
            end = min(n_inits, start + self.n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            # prepare env
            obs = self._prep_slice_env(env, start, this_n_active_envs, run_info, this_global_slice)

            # start rollout
            if self.render_video:
                env.env.set_mask(video_recorder_mask[this_global_slice])

            slice_desctiption = f"Eval {self.env_meta['env_name']} {chunk_idx + 1}/{n_chunks}"
            success = self._run_slice_rollout(env, policy, obs, slice_desctiption, this_local_slice, success_term)

            # collect data for this round
            if self.render_video:
                env_videos = env.env.render()
                env_videos = {x + start: y for x, y in env_videos.items()}
                run_info["videos"].update(env_videos)
            run_info["task_successful"][this_global_slice] = success.cpu().numpy()

        if not self.persistent_env:
            # close env and stage after all runs are finished
            env.close()
            close_stage()

        # log individual run metrics
        log_data = dict()
        success = np.stack(run_info["task_successful"]) > 0
        rollout_results = collections.defaultdict(list)

        for i in range(n_inits):
            prefix = "train/" if i < self.n_train else "test/"
            success_state = success[i]
            rollout_results[prefix].append(success_state)

            # check for video
            idx_vid_path = run_info["videos"].get(i)
            if idx_vid_path is not None:
                idx = i if i < self.n_train else i - self.n_train
                sim_video = wandb.Video(idx_vid_path, f"Success: {success_state}")
                log_data[prefix + f"sim_video_{idx}"] = sim_video

        # log aggregate metrics
        for prefix, value in rollout_results.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def _run_slice_rollout(self, env, policy, init_obs, description, local_slice, success_term):
        policy.reset()
        step = 0
        dones = torch.zeros(self.n_envs, device=self.sim_device)
        success_state = torch.zeros(self.n_envs, device=self.sim_device)

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=description,
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        obs = init_obs
        # only check active envs for termination
        while not torch.all(dones[local_slice]) and step < self.max_steps:
            # create obs dict
            obs_dict = {
                # handle n_latency_steps by discarding the last n_latency_steps
                "obs": obs[:, : self.n_obs_steps]
            }

            # device transfer
            obs_dict = dict_apply(obs_dict, lambda x: x.to(device=policy.device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            action_dict = dict_apply(action_dict, lambda x: x.detach().to(self.sim_device))

            # handle latency_steps, we discard the first n_latency_steps actions
            # to simulate latency
            actions = action_dict["action"][:, self.n_latency_steps :]
            if not torch.all(torch.isfinite(actions)):
                print(actions)
                raise RuntimeError("Nan or Inf action")

            # step env
            if self.abs_action:
                actions = self.undo_transform_action(actions)

            # swap axis (N_envs, Ta, Da) -> (Ta, N_envs, Da)
            actions = torch.moveaxis(actions, 0, 1)

            for act in actions:
                # only execute action if env is not done
                act = act * torch.logical_not(dones)[:, None]
                obs, rewards, terminations = env.step(act)
                dones = torch.logical_or(dones, terminations)
                step += 1

                # check for success
                if success_term is not None:
                    success = success_term.func(env.unwrapped, **success_term.params)
                    # reset successful envs
                    success_ids = success.nonzero().flatten()
                    if success_ids.size(dim=0) > 0:
                        # reset unwrapped env so a potential video recorder wont get reset
                        # TODO: Find better solution, maybe change video wrapper behaviour
                        env.unwrapped.reset(env_ids=success_ids)
                        success_state = torch.logical_or(success, success_state)
                        dones = torch.logical_or(dones, success_state)

            # update pbar
            pbar.update(actions.shape[0])

        pbar.close()
        return success_state[local_slice]

    def _prep_slice_env(self, env, start, n_active_envs, run_info, global_slice):
        obs = None
        local_idx = 0

        # train
        n_train_inits = len([x for x in run_info["init_mode"][global_slice] if x == InitMode.TRAIN])
        if n_train_inits > 0:
            dataset_file_handler = HDF5DatasetFileHandler()
            dataset_file_handler.open(self.dataset_path)
            episode_names = list(dataset_file_handler.get_episode_names())
            for idx in range(n_train_inits):
                # prep env
                train_idx = start + idx
                episode_data = dataset_file_handler.load_episode(
                    episode_names[train_idx + self.train_start_idx], self.sim_device
                )
                initial_state = episode_data.get_initial_state()
                obs = env.reset_to(
                    initial_state,
                    torch.tensor([local_idx], device=self.sim_device),
                    is_relative=True,
                )
                local_idx += 1
                train_idx += 1
            dataset_file_handler.close()

        # test
        n_test_inits = len([x for x in run_info["init_mode"][global_slice] if x == InitMode.TEST])
        if n_test_inits > 0:
            obs = env.reset(
                env_ids=torch.tensor(
                    # reset all remaining envs, even it not relevant
                    list(range(local_idx, n_active_envs)),
                    device=self.sim_device,
                )
            )

        return obs

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
        uaction = torch.cat([pos, rot, gripper], dim=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def add_video_wrapper(self, env, fps, crf, output_dir):
        video_recoder = VideoRecorder.create_h264(
            fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf, thread_type="FRAME", thread_count=1
        )
        env = VideoRecordingWrapper(env, output_dir, video_recoder)
        return env

    def add_cam_to_env(self, env_cfg, render_hw, camera_pos, camera_rot):
        # create render camera
        from isaaclab.envs.mdp.observations import image
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg

        height, width = render_hw
        env_cfg.scene.render_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/render_camera",
            update_period=0.1,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="world"),
        )

        # add render camera to obs
        env_cfg.observations.policy.render_camera = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("render_camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )

        return env_cfg

    def _startup_sim(self, isaac_args):
        # start up sim
        import argparse

        from isaaclab.app import AppLauncher

        # set isaac params form hydra
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        # parse only the known arguments
        args_cli, _ = parser.parse_known_args()
        # launch the simulator
        for key, val in isaac_args.items():
            setattr(args_cli, key, val)
        app_launcher = AppLauncher(args_cli)

        import isaac_imitation_learning.tasks  # noqa:F401

        return app_launcher
