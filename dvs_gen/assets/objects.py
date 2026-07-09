from __future__ import annotations
import torch
import numpy as np

import omni.usd
from pxr import UsdGeom, Gf

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
from isaaclab.envs import ManagerBasedRLEnv

# Bundled YCB USD objects live in the package data dir (CWD-independent).
from dvs_gen.data import DATA_DIR
LOCAL_YCB_DIR = DATA_DIR / "ycb_objects"
YCB_OBJECTS: dict[str, str] = {
    name: str(LOCAL_YCB_DIR / f"{name}.usd")
    for name in (
        "banana", "bleach_cleanser", "bowl", "cracker_box", "extra_large_clamp",
        "foam_brick", "gelatin_box", "large_clamp", "large_marker", "master_chef_can",
        "mug", "mustard_bottle", "pitcher_base", "potted_meat_can", "power_drill",
        "pudding_box", "scissors", "sugar_box", "tomato_soup_can", "tuna_fish_can",
        "wood_block",
    )
}



def _random_pose(
    xy_range:    tuple[float,float],   # (−r, +r) uniform in X and Y
    height_range:tuple[float,float],   # (min_m, max_m) drop height
    rng: np.random.Generator,
) -> tuple[tuple[float,float,float], tuple[float,float,float,float]]:
    """Return a random (position, quaternion_wxyz)."""
    x   = float(rng.uniform(*xy_range))
    y   = float(rng.uniform(*xy_range))
    z   = float(rng.uniform(*height_range))

    # Random rotation — uniform quaternion via Shoemake's method
    u1, u2, u3 = rng.uniform(0, 1, 3)
    qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qx = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qy = np.sqrt(u1)      * np.sin(2 * np.pi * u3)
    qz = np.sqrt(u1)      * np.cos(2 * np.pi * u3)

    return (x, y, z), (float(qw), float(qx), float(qy), float(qz))



def reset_obstacles_random_sparse(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    # Spawn arena (relative to env origin)
    xy_range:      tuple[float, float] = (-0.5, 0.5),
    height_range:  tuple[float, float] = (1.0,  3.0),
    angvel_range:  tuple[float, float] = (-2.0,  2.0),  # rad/s per axis
    seed:          int | None          = None,
) -> None:
    num_envs_reset = len(env_ids)
    device = env.device
    asset  = env.scene["DroppedObject"]
    rng    = np.random.default_rng(seed)

    pos, quat = _random_pose(xy_range, height_range, rng)

    # --- random angular velocity (wx, wy, wz) ---
    ang_vel = _random_angular_velocity(angvel_range, rng)  # (3,)

    # Build tensors with correct batch dim
    pos_t    = torch.tensor([pos],     dtype=torch.float32)          # (1, 3)
    quat_t   = torch.tensor([quat],    dtype=torch.float32)          # (1, 4)
    lin_vel  = torch.zeros((1, 3),     dtype=torch.float32)          # (1, 3)
    ang_vel_t = torch.tensor([ang_vel], dtype=torch.float32)         # (1, 3)
    vel_t    = torch.cat([lin_vel, ang_vel_t], dim=-1)               # (1, 6)

    asset.write_root_pose_to_sim(
        torch.cat([pos_t, quat_t], dim=-1).to(device), env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel_t.to(device), env_ids=env_ids)


def _random_angular_velocity(
    angvel_range: tuple[float, float],
    rng: np.random.Generator,
    *,
    mode: str = "uniform",          # "uniform" | "spherical"
    max_speed: float | None = None, # optional L2 cap (rad/s)
) -> np.ndarray:
    """
    Sample a random 3-DOF angular velocity.

    Modes
    -----
    uniform   : each axis sampled independently from [low, high]
    spherical : random direction, magnitude uniform in [0, high]
    """
    low, high = angvel_range

    if mode == "uniform":
        w = rng.uniform(low, high, size=(3,))

    elif mode == "spherical":
        # random unit vector via Gaussian trick (avoids pole bias)
        direction = rng.standard_normal(size=(3,))
        direction /= np.linalg.norm(direction) + 1e-9
        magnitude  = rng.uniform(0.0, high)
        w          = direction * magnitude

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'uniform' or 'spherical'.")

    # Optional speed cap
    if max_speed is not None:
        speed = np.linalg.norm(w)
        if speed > max_speed:
            w = w * (max_speed / speed)

    return w.astype(np.float32)