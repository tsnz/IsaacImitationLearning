import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from tasks.lift import mdp
from utils.math import pi, quat_from_euler_xyz

from ..franka_lift_env_cfg import FrankaLiftEnvCfg


@configclass
class FrankaLiftTeddyLowDimEnvCfg(FrankaLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set default pose for franka
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.287,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }

        # Set teddy as object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.02), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
                deformable_props=DeformableBodyPropertiesCfg(solver_position_iteration_count=32),
                # mass_props=MassPropertiesCfg(mass=0.2),
            ),
        )

        # Make the end effector less stiff to not hurt the poor teddy bear
        self.scene.robot.actuators["panda_hand"].effort_limit = 400.0
        self.scene.robot.actuators["panda_hand"].stiffness = 5e3
        self.scene.robot.actuators["panda_hand"].damping = 10.0

        # Add teddy reset event
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class FrankaLiftTeddyEnvCfg(FrankaLiftTeddyLowDimEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # create cameras
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/front_camera",
            update_period=0.1,
            height=96,
            width=96,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            # rotate 180 on Y axis to look back at robot
            offset=CameraCfg.OffsetCfg(pos=(1.7, 0.0, 0.250), rot=(0, 0, 0, 1), convention="world"),
        )

        self.scene.eef_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eef_camera",
            update_period=0.1,
            height=96,
            width=96,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=quat_from_euler_xyz(0, -pi / 2, pi),
                convention="world",
            ),
        )

        # add cameras to obs
        self.observations.policy.depth_front = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        self.observations.policy.depth_eef = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eef_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
