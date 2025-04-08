import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaac_imitation_learning.tasks.assets import IIL_ASSET_PATH

from .... import mdp
from ..franka_lift_env_cam_setup_cfg import FrankaLiftEnvCamCfg
from ..franka_lift_env_cfg import FrankaLiftEnvCfg


@configclass
class FrankaLiftCubeLowDimEnvCfg(FrankaLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set default pose for franka
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0444,
            "panda_joint2": -0.1894,
            "panda_joint3": -0.1107,
            "panda_joint4": -2.5148,
            "panda_joint5": 0.0044,
            "panda_joint6": 2.3775,
            "panda_joint7": 0.6952,
            "panda_finger_joint.*": 0.04,
        }

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{IIL_ASSET_PATH}/dex_cube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Add cube reset event
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class FrankaLiftCubeEnvCfg(FrankaLiftCubeLowDimEnvCfg, FrankaLiftEnvCamCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # create cameras
        self.setup_cameras()


@configclass
class FrankaLiftRotatedCubeLowDimEnvCfg(FrankaLiftCubeLowDimEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (0, np.pi / 2)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class FrankaLiftRotatedCubeEnvCfg(FrankaLiftRotatedCubeLowDimEnvCfg, FrankaLiftEnvCamCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.setup_cameras()
