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

# from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

from isaac_imitation_learning.tasks.mdp import object_position_in_robot_root_frame


def object_reached_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.35,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold height for the object to reach the goal position. Defaults to 0.35.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    obj_pos = object_position_in_robot_root_frame(env=env, robot_cfg=robot_cfg, object_cfg=object_cfg)
    # rewarded if the object is lifted above the threshold
    return obj_pos[:, 2] > threshold
