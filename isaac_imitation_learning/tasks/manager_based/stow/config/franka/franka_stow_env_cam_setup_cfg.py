import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from ... import mdp


@configclass
class FrankaStowEnvCamCfg:
    def setup_cameras(self):
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
            # w.r.t. parent frame rotate Z (up) 180 deg and y -35 deg to look down at table
            offset=CameraCfg.OffsetCfg(pos=(1.7, 0.0, 0.75), rot=(0, -0.2164396, 0, 0.976296), convention="world"),
        )

        self.scene.eef_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eef_camera",
            height=128,
            width=128,
            data_types=["distance_to_camera", "rgb"],
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
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("agentview_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": True,
                "max_depth": 2,
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
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("eef_camera"),
                "data_type": "distance_to_camera",
                "convert_perspective_to_orthogonal": False,
                "normalize": True,
                "max_depth": 1,
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
        # needed so image sensors run at full sim rate
        self.sim.render_interval = 1
