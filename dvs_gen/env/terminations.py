"""
terminations.py
===============
Termination conditions for the DVS environments:

* timeout        — episode time limit (mdp.time_out)
* object_dropped — tracked object fell below a minimum height
                   (mdp.root_height_below_minimum)

A commented-out out-of-FOV check (projecting the object into cam0's
image plane) is kept below as possible future work.
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as TermTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)

# project modules
import isaaclab.envs.mdp as mdp


@configclass
class MyTerminationCfg:
    timeout: TermTerm = TermTerm(func=mdp.time_out, time_out=True)

    object_dropped: TermTerm = TermTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.0,
            "asset_cfg": SceneEntityCfg("DroppedObject")
        }
    )


    # object_out_of_frame: TermTerm = TermTerm(
    #     func=object_out_of_frame,
    #     params={
    #         "calib_path":      camera_config['calib_path'],
    #         "env_prim_prefix": "/World/envs/env_0",
    #         "margin_px":       5.0,    # shrink valid zone by 10 px on each edge
    #         "min_depth_m":     0.05,    # terminate if object closer than 5 cm
    #         "max_depth_m":     20.0,    # terminate if object further than 20 m
    #     },
    # )