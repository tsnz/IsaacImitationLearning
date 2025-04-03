import collections
import math

import gymnasium as gym
import numpy as np
import omegaconf
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.video_recorder import VideoRecorder

import wandb
from isaac_imitation_learning.robodiff.utils import VideoRecordingWrapper


class IsaacLabMultiEnvImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        env_wrapper,
        isaac_args,
        task_ids,
        n_test=22,
        n_test_vis=6,
        test_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_video=False,
        render_obs_key="agentview_image",
        fps=30,
        crf=22,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        dummy_rollout=False,
    ):
        super().__init__(output_dir)

        if dummy_rollout:
            self.dummy_rollout = dummy_rollout
            return

        self.app_launcher = self._startup_sim(isaac_args)

        # prepare rollout
        sim_device = isaac_args.device

        # make sure needed settings are present
        n_tasks = len(task_ids)
        n_test = n_test if type(n_test) is omegaconf.ListConfig else [n_test] * n_tasks
        n_test_vis = n_test_vis if type(n_test_vis) is omegaconf.ListConfig else [n_test_vis] * n_tasks

        if n_envs is None:  # noqa: SIM108
            n_envs = n_test
        else:
            n_envs = n_envs if type(n_envs) is omegaconf.ListConfig else [n_envs] * n_tasks

        assert n_tasks == len(n_test) == len(n_test_vis) == len(n_envs), (
            "Env specific settings do not match number of envs"
        )

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        # needed for env creation
        self.crf = crf
        self.env_wrapper = env_wrapper
        self.fps = fps
        self.output_dir = output_dir
        self.render_obs_key = render_obs_key
        self.test_seed = test_seed

        self.abs_action = abs_action
        self.dummy_rollout = dummy_rollout
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.n_obs_steps = n_obs_steps
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.render_video = render_video
        self.rotation_transformer = rotation_transformer
        self.sim_device = sim_device
        self.task_ids = task_ids
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        if self.dummy_rollout:
            log_data = dict()
            return log_data

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from isaacsim.core.utils.stage import close_stage, create_new_stage

        run_info = {}
        run_info["videos"] = {}
        run_info["task_successful"] = [None] * np.sum(self.n_test)

        # run rollout for each task
        for run_idx, task_id in enumerate(self.task_ids):
            # get env settings
            n_envs = self.n_envs[run_idx]
            n_test = self.n_test[run_idx]
            n_test_vis = self.n_test_vis[run_idx]

            env_cfg = parse_env_cfg(task_id, device=self.sim_device, num_envs=n_envs)
            env_cfg.seed = self.test_seed

            # Disable all recorders and terminations
            env_cfg.recorders = {}

            # use history_length instead of a multi step wrapper
            env_cfg.observations.policy.history_length = self.n_obs_steps
            env_cfg.observations.policy.flatten_history_dim = False

            video_recorder_mask = [False] * n_test
            video_recorder_mask[:n_test_vis] = [True] * n_test_vis

            # extract success checking function to invoke during rollout
            success_term = None
            if hasattr(env_cfg.terminations, "success"):
                success_term = env_cfg.terminations.success
                env_cfg.terminations.success = None

            create_new_stage()
            env = gym.make(task_id, cfg=env_cfg).unwrapped
            if self.render_video:
                env = self.add_video_wrapper(env, self.render_obs_key, self.fps, self.crf, self.output_dir)
            env = self.env_wrapper(env=env)

            # plan for rollout
            n_inits = n_test
            n_chunks = math.ceil(n_inits / n_envs)

            for chunk_idx in range(n_chunks):
                task_start = chunk_idx * n_envs
                task_end = min(n_inits, task_start + n_envs)
                this_task_slice = slice(task_start, task_end)
                this_n_active_envs = task_end - task_start
                this_local_slice = slice(0, this_n_active_envs)
                start = task_start + np.sum(self.n_test[:run_idx], dtype=int)
                end = task_end + np.sum(self.n_test[:run_idx], dtype=int)
                this_global_slice = slice(start, end)

                # prepare env
                obs = env.reset()

                # start rollout
                if self.render_video:
                    env.env.set_mask(video_recorder_mask[this_task_slice])

                slice_desctiption = f"Eval {task_id} {chunk_idx + 1}/{n_chunks}"
                success = self._run_slice_rollout(env, policy, obs, slice_desctiption, this_local_slice, success_term)

                # collect data for this round
                run_info["task_successful"][this_global_slice] = success.cpu().numpy()
                if self.render_video:
                    env_videos = env.env.render()
                    env_videos = {x + start: y for x, y in env_videos.items()}
                    run_info["videos"].update(env_videos)

            # clear out video buffer
            env.reset()
            env.close()
            close_stage()

        # log individual run metrics
        log_data = dict()
        success = np.stack(run_info["task_successful"]) > 0
        rollout_results = collections.defaultdict(list)

        idx = 0
        for i in range(len(self.task_ids)):
            for j in range(self.n_test[i]):
                prefix = self.task_ids[i] + "/"
                success_state = success[idx]
                rollout_results[prefix].append(success_state)

                # visualize sim
                idx_vid_path = run_info["videos"].get(idx)
                if idx_vid_path is not None:
                    sim_video = wandb.Video(idx_vid_path, f"Finished: {success_state}")
                    log_data[prefix + f"sim_video_{j}"] = sim_video

                idx += 1

        # log aggregate task metrics
        for prefix, value in rollout_results.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        # log aggregate metrics
        log_data["mean_score"] = success.mean()

        return log_data

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

    def add_video_wrapper(self, env, camera_name, fps, crf, output_dir):
        video_recoder = VideoRecorder.create_h264(
            fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf, thread_type="FRAME", thread_count=1
        )
        env = VideoRecordingWrapper(env, output_dir, video_recoder, camera_name=camera_name)
        return env

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

        from isaacsim.core.utils.stage import close_stage

        import isaac_imitation_learning.tasks  # noqa:F401

        close_stage()

        return app_launcher

    def _run_slice_rollout(self, env, policy, init_obs, description, local_slice, success_term):
        policy.reset()
        step = 0
        n_envs = env.unwrapped.num_envs
        dones = torch.zeros(n_envs, device=self.sim_device)
        success_state = torch.zeros(n_envs, device=self.sim_device)

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=description,
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        obs = init_obs
        # only check active envs for termination
        while not torch.all(dones[local_slice]) and step < self.max_steps:
            # device transfer
            obs_dict = dict_apply(obs, lambda x: x.to(device=policy.device))

            # run policy
            with torch.no_grad():
                actions = policy.predict_action(obs_dict)

            # device transfer
            actions = dict_apply(actions, lambda x: x.detach().to(self.sim_device))

            actions = actions["action"]
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
