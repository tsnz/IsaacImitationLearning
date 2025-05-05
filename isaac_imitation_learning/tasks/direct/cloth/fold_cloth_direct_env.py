from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionManager
from isaacsim.core.prims import ClothPrim
from pxr import UsdUtils
from usdrt import Usd as RtUsd
from usdrt import UsdGeom as RtGeom
from usdrt import Vt as RtVt

from . import mdp
from .fold_cloth_direct_env_cfg import FoldClothEnvCfg


class FoldClothEnv(DirectRLEnv):
    cfg: FoldClothEnvCfg

    def __init__(self, cfg: FoldClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # apparently action managers also work with non manager based environments
        self.action_manager = ActionManager(self.cfg.actions, self)

    def _setup_scene(self):
        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)

        # store default cloth particle pos for reset
        stage_id = UsdUtils.StageCache.Get().Insert(self.unwrapped.sim.stage)
        self.rt_stage = RtUsd.Stage.Attach(stage_id.ToLongInt())
        rt_cloth_prim = self.rt_stage.GetPrimAtPath("/World/envs/env_0/Object/Xform/Cloth")
        self.default_cloth_pos = np.array(rt_cloth_prim.GetAttribute(RtGeom.Tokens.points).Get())

    def _get_observations(self) -> dict:
        obs = {}

        robot0_eef_pos = mdp.ee_frame_pos(self)
        robot0_eef_quat = mdp.ee_frame_quat(self)
        robot0_gripper_qpos = mdp.gripper_pos(self)
        robot0_eef_image = mdp.image(self, SceneEntityCfg("eef_camera"), "rgb", False, False)
        agentview_image = mdp.image(self, SceneEntityCfg("agentview_camera"), "rgb", False, False)

        obs["robot0_eef_pos"] = robot0_eef_pos
        obs["robot0_eef_quat"] = robot0_eef_quat
        obs["robot0_gripper_qpos"] = robot0_gripper_qpos
        obs["robot0_eef_image"] = robot0_eef_image
        obs["agentview_image"] = agentview_image

        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # only return success / time out reset signals, if enabled
        success = (
            torch.zeros(size=[self.num_envs], device=self.device, dtype=bool)
            if self.cfg.disable_success_reset
            else self.get_success_state()
        )
        time_out = (
            torch.zeros(size=[self.num_envs], device=self.device, dtype=bool)
            if self.cfg.disable_timeout_reset
            else (self.episode_length_buf >= self.max_episode_length - 1)
        )

        return (success, time_out)

    def get_success_state(self) -> torch.Tensor:
        cloth_prims = ClothPrim("/World/envs/env_.*/Object/Xform/Cloth")

        success_state = torch.zeros(size=[self.num_envs], device=self.device, dtype=bool)
        for i, prim_path in enumerate(cloth_prims.prim_paths):
            rt_cloth_prim = self.rt_stage.GetPrimAtPath(prim_path)

            points = rt_cloth_prim.GetAttribute(RtGeom.Tokens.points).Get()

            # looking at robot, cloth in front
            BACK_IDX = 2  # noqa: N806
            LEFT_IDX = 265  # noqa: N806
            RIGHT_IDX = 341  # noqa: N806
            FRONT_IDX = 424  # noqa: N806

            # check distance of front to back and left to right corners
            # converting points to np array can cause issues
            # https://forums.developer.nvidia.com/t/with-fabric-enabled-particle-cloth-position-reading-causes-reset/315290/6
            distance = min(
                (points[FRONT_IDX] - points[BACK_IDX]).GetLength(), (points[LEFT_IDX] - points[RIGHT_IDX]).GetLength()
            )

            success_state[i] = distance < self.cfg.success_distance_threshold

        return success_state

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(size=[self.num_envs], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.action_manager.process_action(actions)

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        super()._reset_idx(env_ids)

        # reset cloth to random position, y is up
        pose_range = {"x": (-0.10, 0.25), "z": (-0.10, 0.25)}
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list)
        # store for env state
        self.cloth_rand = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device="cpu")

        cloth_prims = ClothPrim("/World/envs/env_.*/Object/Xform/Cloth")
        for idx, env_id in enumerate(env_ids):
            prim_path = cloth_prims.prim_paths[env_id]

            # use non fabric memory because reasons
            cloth_prim = self.scene.stage.GetPrimAtPath(prim_path)
            cloth_prim.GetAttribute(RtGeom.Tokens.points).Set(
                RtVt.Vec3fArray(self.default_cloth_pos + np.array(self.cloth_rand[idx, :3].cpu()))
            )
            cloth_prim.GetAttribute(RtGeom.Tokens.velocities).Set(RtVt.Vec3fArray(self.default_cloth_pos * 0))

    def get_env_state(self, env_id):
        # function to recursively read env state dict
        def extract_env_ids_values(value):
            nonlocal env_id
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            # return with leading dim, needed for recording
            return torch.unsqueeze(value[env_id], dim=0)

        initial_state = extract_env_ids_values(self.scene.get_state(is_relative=True))

        initial_state["cloth_rand"] = self.cloth_rand[env_id].unsqueeze(dim=0)
        return initial_state

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> None:
        """Resets specified environments to known states.

        Note that this is different from reset() function as it resets the environments to specific states

        Args:
            state: The state to reset the specified environments to.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins. Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # set the seed
        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # set cloth pos
        cloth_prims = ClothPrim("/World/envs/env_.*/Object/Xform/Cloth")
        for env_id in env_ids:
            prim_path = cloth_prims.prim_paths[env_id]

            # use non fabric memory because reasons
            cloth_prim = self.scene.stage.GetPrimAtPath(prim_path)
            cloth_prim.GetAttribute(RtGeom.Tokens.points).Set(
                RtVt.Vec3fArray(self.default_cloth_pos + np.array(state["cloth_rand"][0, :3].cpu()))
            )
            cloth_prim.GetAttribute(RtGeom.Tokens.velocities).Set(RtVt.Vec3fArray(self.default_cloth_pos * 0))

        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        return self._get_observations()
