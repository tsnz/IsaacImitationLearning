import pathlib
from copy import deepcopy

import gymnasium as gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder

import wandb.sdk.data_types.video as wv


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        output_dir,
        video_recoder: VideoRecorder,
        mode="rgb_array",
        mask=None,
        steps_per_render=1,
        camera_name="render_camera",
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.n_envs = env.num_envs
        self.output_dir = output_dir
        self.camera_name = camera_name
        self.active_recorders = {}
        self.video_recorders = [None] * self.n_envs
        for idx in range(self.n_envs):
            self.video_recorders[idx] = deepcopy(video_recoder)

        if mask is None:
            mask = [False] * self.n_envs
        self.mask = mask
        self._generate_file_paths(mask)

        self.step_count = 0

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.step_count = 1
        self._stop_recorders()
        return result

    def reset_to(self, *args, **kwargs):
        result = self.env.reset_to(*args, **kwargs)
        self.step_count = 1
        self._stop_recorders()
        return result

    def _stop_recorders(self):
        for rec in self.video_recorders:
            rec.stop()

    def _generate_file_paths(self, mask):
        # filter file paths for set vars
        assert len(mask) == self.n_envs
        self.active_recorders = {}
        active_recorders = [idx for idx, active in enumerate(mask) if active]
        for idx in active_recorders:
            filename = pathlib.Path(self.output_dir).joinpath("media", wv.util.generate_id() + ".mp4")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            self.active_recorders[idx] = filename

    def set_mask(self, mask):
        assert len(mask) == self.n_envs
        self.mask = mask
        self._stop_recorders()
        self._generate_file_paths(mask)

    def step(self, action):
        result = self.env.step(action)
        self.step_count += 1

        if (self.step_count % self.steps_per_render) == 0:
            for idx, path in self.active_recorders.items():
                if not self.video_recorders[idx].is_ready():
                    self.video_recorders[idx].start(path)

                frame = result[0]["policy"][self.camera_name][idx][0].detach().cpu().numpy()
                # convert float to int if needed
                assert frame.dtype == np.uint8
                self.video_recorders[idx].write_frame(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        self._stop_recorders()
        return self.active_recorders
