from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field, MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
import isaaclab.envs.mdp as mdp


def obs_camera_pose(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the pose (position + quaternion) of a camera sensor in the world frame.
    Returns tensor of shape (num_envs, 7).
    """
    sensor = env.scene[sensor_cfg.name]

    return torch.cat([sensor.data.pos_w, sensor.data.quat_w_ros], dim=-1)

def obs_object_pose(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the root pose (position + quaternion) of a rigid object in the world frame.
    Returns tensor of shape (num_envs, 7).
    """
    asset = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_pos_w, asset.data.root_quat_w], dim=-1)

def obs_object_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the root linear and angular velocity of a rigid object in the world frame.
    Returns tensor of shape (num_envs, 6).
    """
    asset = env.scene[asset_cfg.name]
    speed = torch.cat([asset.data.root_lin_vel_w, asset.data.root_ang_vel_w], dim=-1)
    
    return speed

@configclass
class MyObsCfg:
    @configclass
    class SimOnlyGroup(ObsGroup):

        cam0_pose: ObsTerm = ObsTerm(
            func=obs_camera_pose,
            params={"sensor_cfg": SceneEntityCfg("cam0")}
        )
        
        cam1_pose: ObsTerm = ObsTerm(
            func=obs_camera_pose,
            params={"sensor_cfg": SceneEntityCfg("cam1")}
        )

        dropped_object_pose: ObsTerm = ObsTerm(
            func=obs_object_pose,
            params={"asset_cfg": SceneEntityCfg("DroppedObject")}
        )

        dropped_object_vel: ObsTerm = ObsTerm(
            func=obs_object_velocity,
            params={"asset_cfg": SceneEntityCfg("DroppedObject")}
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: SimOnlyGroup = SimOnlyGroup()