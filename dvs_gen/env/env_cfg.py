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

from .events import MyEventCfg
from .terminations import MyTerminationCfg
from .scene import MySceneCfg
from .observation import MyObsCfg


# ── minimal no-op action ──────────────────────────────────────
@configclass
class MyActionCfg:
    # empty = no actuated joints, no action tensor needed
    pass


# ── minimal no-op reward ──────────────────────────────────────
@configclass
class MyRewardCfg:
    # empty = reward is always 0.0
    pass

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):          # swap for ManagerBasedRLEnvCfg if needed

    # Re-render the RTX cameras twice after each reset so cameras reflect the
    # post-reset scene (new background + object pose) before recording.
    num_rerenders_on_reset = 2

    # ── simulation ────────────────────────────────────────────
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.001,
        render_interval=1,
    )

    # ── scene ─────────────────────────────────────────────────
    scene: MySceneCfg = MySceneCfg()

    terminations: MyTerminationCfg        = MyTerminationCfg()   # ← new

    # ── events (includes background randomization) ────────────
    events: MyEventCfg = MyEventCfg()
    observations: MyObsCfg        = MyObsCfg()
    actions:      MyActionCfg     = MyActionCfg()
    rewards:      MyRewardCfg     = MyRewardCfg()
    # ── episode & action ──────────────────────────────────────
    episode_length_s: float = 3.0
    decimation:       int   = 1

    # ── DVS-specific ──────────────────────────────────────────
    dvs_threshold:    float = 0.15
    dvs_pixel_size_um: float = 15.0

    def __post_init__(self):
        self.sim.dt             = 0.001              # 1k Hz physics
        self.sim.render_interval = self.decimation   # 1k Hz policy
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        self.sim.light = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 1.0),
        )
        # Propagate num_envs into scene
        self.scene.num_envs = 1280
        self.scene.env_spacing = 4.0