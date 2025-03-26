import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.rotations import euler_angles_to_quat

from .... import mdp
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
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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
class FrankaLiftCubeEnvCfg(FrankaLiftCubeLowDimEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # create cameras
        self.scene.agentview_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/agentview_camera",
            height=128,
            width=128,
            data_types=["distance_to_camera", "rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
            ),
            # rotate 180 deg on Y axis to look back at robot and 35 deg down
            offset=CameraCfg.OffsetCfg(pos=(1.2, 0.0, 0.6), rot=(0, -0.3007058, 0, 0.953717), convention="world"),
        )

        self.scene.eef_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eef_camera",
            height=128,
            width=128,
            data_types=["distance_to_camera", "rgb"],
            # spawn=sim_utils.PinholeCameraCfg(
            spawn=sim_utils.FisheyeCameraCfg(
                projection_type="fisheyeKannalaBrandtK3",
                clipping_range=(0.001, 1.0e5),
                horizontal_aperture=10.0,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.058),
                rot=euler_angles_to_quat(np.asarray([0, -90, 180]), True),
                convention="world",
            ),
        )

        # add cameras to obs
        self.observations.policy.agentview_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("agentview_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": True,
            },
        )
        self.observations.policy.agentview_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("agentview_camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        self.observations.policy.robot0_eef_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eef_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": True,
            },
        )
        self.observations.policy.robot0_eef_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eef_camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )

        self.rerender_on_reset = True
