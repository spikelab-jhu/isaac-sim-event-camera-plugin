from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.terrains import TerrainImporterCfg

from dvs_gen.assets.objects import YCB_OBJECTS
from dvs_gen.data import DATA_DIR

@configclass
class MySceneCfg(InteractiveSceneCfg):

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    # )

    
    # cam0: CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam0",
    #     update_period=0.001,
    #     height=480, width=640,
    #     data_types=["rgb"], 
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         clipping_range=(0.1, 1.0e5),
    #         projection_type="fisheyeRadTanThinPrism",
    #     ),
    # )

    # cam1: CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam1",
    #     update_period=0.001,
    #     height=480, width=640,
    #     data_types=["rgb"], 
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         clipping_range=(0.1, 1.0e5),
    #         projection_type="fisheyeRadTanThinPrism",
    #     ),
    # )

    cam0: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0",
        update_period=0.001,
        height=480, width=640,
        data_types=["rgb", "motion_vectors"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.01, 1.0e5),
            # Pinhole is the default, so no projection_type is needed
        ),
    )
    cam1: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam1",
        update_period=0.001,
        height=480, width=640,
        data_types=["rgb", "motion_vectors"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.01, 1.0e5),
            # Pinhole is the default, so no projection_type is needed
        ),
    )

    # Aux camera for rendering GT at lower freq
    cam0_anno: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0_anno",
        update_period=0.1,           # 10 Hz
        height=480, width=640,
        data_types=["depth", "motion_vectors"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.01, 1.0e5),
        ),
    )

    cam1_anno: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam1_anno",
        update_period=0.1,           # 10 Hz
        height=480, width=640,
        data_types=["depth", "motion_vectors"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.01, 1.0e5),
        ),
    )


    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/BackgroundDomeLight",  # We will target this path in our randomizer
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            # Point this to any default HDR in your folder so it doesn't crash on startup
            texture_file=str(DATA_DIR / "resources" / "dome_texture" / "1.jpg"),
        ),
    )

        # DroppedObject = RigidObjectCfg(
        # prim_path="{ENV_REGEX_NS}/DroppedObject",
        # spawn = sim_utils.CylinderCfg(
        #         radius   = 0.05,
        #         height   = 0.2,
        #         rigid_props = sim_utils.RigidBodyPropertiesCfg(
        #             rigid_body_enabled         = True,
        #             kinematic_enabled          = False,    # ✅ dynamic body
        #             disable_gravity            = False,
        #             enable_gyroscopic_forces   = True,
        #             max_depenetration_velocity = 1.0,
        #         ),
        #         mass_props = sim_utils.MassPropertiesCfg(
        #             mass = 10.0,                           # ✅ heavier — won't move when hit
        #         ),
        #         collision_props = sim_utils.CollisionPropertiesCfg(
        #             collision_enabled = True,              # ✅ explicit
        #             contact_offset    = 0.02,              # ✅ contact detection margin
        #             rest_offset       = 0.001,             # ✅ must be < contact_offset
        #         ),
        #         visual_material = sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(0.8, 0.1, 0.1)
        #         ),
        #     ),
        #     init_state = RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.05)),)
    DroppedObject: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DroppedObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=YCB_OBJECTS['mustard_bottle'], 
            # You no longer need to manually inject rigid_props or collision_props here!
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4), # Just set the mass
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.05)),)