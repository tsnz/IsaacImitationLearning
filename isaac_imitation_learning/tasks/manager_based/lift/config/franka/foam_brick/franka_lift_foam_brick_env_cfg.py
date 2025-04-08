import numpy as np
from isaaclab.assets import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaac_imitation_learning.tasks.assets import IIL_ASSET_PATH

from .... import mdp
from ..franka_lift_env_cam_setup_cfg import FrankaLiftEnvCamCfg
from ..franka_lift_env_cfg import FrankaLiftEnvCfg


@configclass
class FrankaLiftFoamBrickLowDimEnvCfg(FrankaLiftEnvCfg):
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

        # Make the end effector less stiff
        self.scene.robot.actuators["panda_hand"].effort_limit = 50.0
        self.scene.robot.actuators["panda_hand"].stiffness = 200.0
        self.scene.robot.actuators["panda_hand"].damping = 10.0

        # Set mustard bottle as object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.1), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{IIL_ASSET_PATH}/foam_brick/foam_brick.usd",
                deformable_props=DeformableBodyPropertiesCfg(solver_position_iteration_count=150),
            ),
        )

        # Add mustard bottle reset event
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Disable replicate physics as it doesn't work for deformable objects
        # FIXME: This should be fixed by the PhysX replication system.
        self.scene.replicate_physics = False


@configclass
class FrankaLiftFoamBrickEnvCfg(FrankaLiftFoamBrickLowDimEnvCfg, FrankaLiftEnvCamCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # create cameras
        self.setup_cameras()


@configclass
class FrankaLiftRotatedFoamBrickLowDimEnvCfg(FrankaLiftFoamBrickLowDimEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.reset_object_rotation = EventTerm(
            func=mdp.randomize_nodal_rotation,
            mode="reset",
            params={
                "rotation_range": {"yaw": (0, np.pi)},
                "object_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class FrankaLiftRotatedFoamBrickCfg(FrankaLiftRotatedFoamBrickLowDimEnvCfg, FrankaLiftEnvCamCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.setup_cameras()
