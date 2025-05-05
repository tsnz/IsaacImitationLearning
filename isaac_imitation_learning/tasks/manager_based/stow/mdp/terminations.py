# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def object_in_bin(
    env: ManagerBasedRLEnv,
    finish_range: dict[str, tuple[float, float]],
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position."""
    # extract the used quantities (to enable type-hinting)
    bin: RigidObject | DeformableObject = env.scene[bin_cfg.name]
    object: RigidObject | DeformableObject = env.scene[object_cfg.name]

    object_pos_in_bin_frame, _ = subtract_frame_transforms(
        bin.data.root_state_w[:, :3], bin.data.root_state_w[:, 3:7], object.data.root_pos_w[:, :3]
    )

    in_x_range = torch.logical_and(
        object_pos_in_bin_frame[:, 0] > finish_range["x"][0], object_pos_in_bin_frame[:, 0] < finish_range["x"][1]
    )
    in_y_range = torch.logical_and(
        object_pos_in_bin_frame[:, 1] > finish_range["y"][0], object_pos_in_bin_frame[:, 1] < finish_range["y"][1]
    )
    in_z_range = torch.logical_and(
        object_pos_in_bin_frame[:, 2] > finish_range["z"][0], object_pos_in_bin_frame[:, 2] < finish_range["z"][1]
    )

    success = torch.logical_and(in_x_range, in_y_range)
    success = torch.logical_and(success, in_z_range)

    return success
