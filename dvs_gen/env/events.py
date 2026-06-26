"""
env_cfg.py — Isaac Lab RL environment config with background randomization
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field, MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Sdf
import yaml

# The package name `sim_utils` no longer collides with cv2's bundled `utils`,
# so the background randomizer imports normally (no importlib workaround needed).
from dvs_gen.sim_utils import background_randomizer as bg_rand
from dvs_gen.assets import objects as obj_drop
from dvs_gen.assets import camera as dvs_cam
from dvs_gen.data import DATA_DIR
import omni.usd

# Bundled dome textures (CWD-independent).
_DOME_TEXTURE_DIR = str(DATA_DIR / "resources" / "dome_texture")

# ══════════════════════════════════════════════════════════════
# 1.  Custom event functions
#     These are plain functions; EventManager calls them with
#     (env, env_ids, **kwargs) automatically.
# ══════════════════════════════════════════════════════════════

def initialize_stereo_cameras(
        env,
        env_ids:         torch.Tensor,
        yaml_path: str,
        pixel_size_um : float,
        T_w_c0_np: np.ndarray,
        margin=(0, 0, 0, 0),     # (left, right, top, bottom) over-render in pixels
        ):
    """
    Reads Kalibr YAML, injects USD intrinsics/distortions, and sets batched global poses for event camera and annotation cams.
    
    Args:
        env: The Isaac Lab environment instance.
        yaml_path: Path to the Kalibr calibration YAML.
        T_w_c0_np: 4x4 numpy array representing the global pose of cam0.
    """
    device = env.device
    num_envs = env.num_envs

    
    stage = omni.usd.get_context().get_stage()

    # 1. Load Calibration Data
    with open(yaml_path, 'r') as f:
        calib = yaml.safe_load(f)

    cam0_calib = calib['cam0']
    cam1_calib = calib['cam1']
    
    # 2. Compute Global Poses
    # T_c1_c0 is provided in the YAML
    T_c1_c0 = np.array(cam1_calib['T_cn_cnm1']) 
    T_c0_c1 = np.linalg.inv(T_c1_c0)
    
    # Calculate global pose of cam1
    T_w_c1_np = T_w_c0_np @ T_c0_c1

    # Extract Translations (in meters)
    pos0 = T_w_c0_np[:3, 3]
    pos1 = T_w_c1_np[:3, 3]

    # Extract Rotations and convert to Quaternions [w, x, y, z] for Isaac Lab
    rot0_scipy = Rotation.from_matrix(T_w_c0_np[:3, :3]).as_quat() # returns [x,y,z,w]
    quat0 = np.array([rot0_scipy[3], rot0_scipy[0], rot0_scipy[1], rot0_scipy[2]]) 
    
    rot1_scipy = Rotation.from_matrix(T_w_c1_np[:3, :3]).as_quat() 
    quat1 = np.array([rot1_scipy[3], rot1_scipy[0], rot1_scipy[1], rot1_scipy[2]])

    # 3. Create Batched Tensors
    pos0_tensor = torch.tensor([pos0] * num_envs, dtype=torch.float32, device=device)
    quat0_tensor = torch.tensor([quat0] * num_envs, dtype=torch.float32, device=device)
    
    pos1_tensor = torch.tensor([pos1] * num_envs, dtype=torch.float32, device=device)
    quat1_tensor = torch.tensor([quat1] * num_envs, dtype=torch.float32, device=device)

    # 4. Apply Poses via Isaac Lab Sensor API
    print('camera_pose_readout ',  quat0_tensor)
    env.scene["cam0"].set_world_poses(pos0_tensor, quat0_tensor, convention = 'ros')
    env.scene["cam1"].set_world_poses(pos1_tensor, quat1_tensor, convention = 'ros')

    try:                                    # anno cams are optional (may be removed for benchmarks)
        env.scene["cam0_anno"].set_world_poses(pos0_tensor, quat0_tensor, convention = 'ros')
        env.scene["cam1_anno"].set_world_poses(pos1_tensor, quat1_tensor, convention = 'ros')
    except KeyError:
        pass


    K_cam0 = get_intrinsic_pinhole(cam0_calib)
    K_cam1 = get_intrinsic_pinhole(cam1_calib)
    # Over-render margin: shift the principal point by (left, top) so the central
    # crop reproduces the original calibrated view exactly (cx,cy stay valid for
    # the cropped image). Focal length is unchanged -> the extra pixels just widen
    # the FOV around the same view.
    m_left, _m_right, m_top, _m_bottom = margin
    if m_left or m_top:
        for K in (K_cam0, K_cam1):
            K[0, 2] += m_left   # cx += left margin
            K[1, 2] += m_top    # cy += top  margin
    K_batched_0 = K_cam0.unsqueeze(0).repeat(num_envs, 1, 1)
    K_batched_1 = K_cam1.unsqueeze(0).repeat(num_envs, 1, 1)
    # 4. Apply to the camera

    env.scene["cam1"].set_intrinsic_matrices(K_batched_1)
    env.scene["cam0"].set_intrinsic_matrices(K_batched_0)

    try:
        env.scene["cam1_anno"].set_intrinsic_matrices(K_batched_1)
        env.scene["cam0_anno"].set_intrinsic_matrices(K_batched_0)
    except KeyError:
        pass

    # 5. Inject Distortion and Intrinsics directly into the USD Prims
    # for i in range(num_envs):
    #     cam0_path = f"/World/envs/env_{i}/cam0"
    #     cam1_path = f"/World/envs/env_{i}/cam1"
        
    #     _apply_calibration_to_prim(stage, cam0_path, cam0_calib, pixel_size_um)
    #     _apply_calibration_to_prim(stage, cam1_path, cam1_calib, pixel_size_um)

    print(f"[Stereo Calibration] Applied to {num_envs} environments.")



def get_intrinsic_pinhole(calib_dict):
    intrinsics = calib_dict['intrinsics']
    # Handle both omni (5 params) and pinhole (4 params)
    if len(intrinsics) == 5:
        xi, fx, fy, cx, cy = intrinsics
    else:
        xi = None
        fx, fy, cx, cy = intrinsics

    return torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0,0,1]
        ]
    )

def _apply_calibration_to_prim(stage, prim_path: str, calib_dict: dict, pixel_size_um: float = 3.0):
    cam_prim = UsdGeom.Camera.Get(stage, prim_path)
    prim = cam_prim.GetPrim()
    
    intrinsics = calib_dict['intrinsics']
    
    # Handle both omni (5 params) and pinhole (4 params)
    if len(intrinsics) == 5:
        xi, fx, fy, cx, cy = intrinsics
    else:
        xi = None
        fx, fy, cx, cy = intrinsics

    width = calib_dict['resolution'][0]
    height = calib_dict['resolution'][1]

    # ---------------------------------------------------------
    # 1. Focal Length and Apertures (unchanged)
    # ---------------------------------------------------------
    pixel_size_mm = pixel_size_um / 1000.0
    horizontal_aperture_mm = width * pixel_size_mm
    focal_length_mm = fx * pixel_size_mm
    vertical_aperture_mm = (focal_length_mm / fy) * height

    prim.GetAttribute("horizontalAperture").Set(horizontal_aperture_mm)
    prim.GetAttribute("verticalAperture").Set(vertical_aperture_mm)
    prim.GetAttribute("focalLength").Set(focal_length_mm)

    # ---------------------------------------------------------
    # 2. Optical Center (unchanged)
    # ---------------------------------------------------------
    h_offset = (cx - width / 2.0) / width
    v_offset = (cy - height / 2.0) / height
    prim.GetAttribute("horizontalApertureOffset").Set(h_offset * horizontal_aperture_mm)
    prim.GetAttribute("verticalApertureOffset").Set(v_offset * vertical_aperture_mm)

    if xi is not None:
        prim.CreateAttribute(
            "cameraFisheyeRadTanThinPrism:xi",
            Sdf.ValueTypeNames.Float
        ).Set(float(xi))

    dist = calib_dict['distortion_coeffs']
    prim.CreateAttribute("cameraFisheyeRadTanThinPrism:k1", Sdf.ValueTypeNames.Float).Set(dist[0])
    prim.CreateAttribute("cameraFisheyeRadTanThinPrism:k2", Sdf.ValueTypeNames.Float).Set(dist[1])
    prim.CreateAttribute("cameraFisheyeRadTanThinPrism:p1", Sdf.ValueTypeNames.Float).Set(dist[2])
    prim.CreateAttribute("cameraFisheyeRadTanThinPrism:p2", Sdf.ValueTypeNames.Float).Set(dist[3])


def event_randomize_background(
    env,
    env_ids: torch.Tensor,
    texture_folder: str          = _DOME_TEXTURE_DIR,
    use_dome:       bool         = True,
    use_backdrop:   bool         = True,
    backdrop_dist_m:     float   = 3.0,
    backdrop_halfsize_m: float   = 5.0,
) -> None:
    """
    EventManager-compatible wrapper.
    On the very first call it initialises the Replicator graph;
    every subsequent call just randomizes.
    """
    bg_rand.setup_background_randomizer(
        texture_folder       = texture_folder,
        use_dome             = use_dome,
        use_backdrop         = use_backdrop,
        backdrop_dist_m      = backdrop_dist_m,
        backdrop_halfsize_m  = backdrop_halfsize_m,
    )
    bg_rand.randomize_background()

# ══════════════════════════════════════════════════════════════
# 2.  EventCfg  — drop this into your env config
# ══════════════════════════════════════════════════════════════
@configclass
class MyEventCfg:

    # setup_dvs_camera: EventTerm = EventTerm(
    #     func=event_setup_dvs_camera,
    #     mode="startup",
    #     params={
    #         "calib_path":      dvs_cam.camera_config["calib_path"],
    #         "pixel_size_um":   dvs_cam.camera_config["pixel_size_um"],
    #         "init_pose":       dvs_cam.camera_config["init_pose"].tolist(),
    #         "visualize": True
    #     },
    # )

    

    # # ── physics / robot resets (your existing events) ─────────
    # reset_robot_joints: EventTerm = EventTerm(
    #     func=mdp.reset_joints_by_offset,          # your existing reset
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0,  0.0),
    #     },
    # )

    # ── background randomization ───────────────────────────────
    randomize_background: EventTerm = EventTerm(
        func=event_randomize_background,
        mode="reset",
        params={
            "texture_folder":       _DOME_TEXTURE_DIR,
            "use_dome":             True,
            "use_backdrop":         True,
            "backdrop_dist_m":      3.0,
            "backdrop_halfsize_m":                5.0,
        },
    )

    # randomize_dvs_threshold: EventTerm = EventTerm(
    #     func=event_randomize_dvs_threshold,
    #     mode="reset",                   # new threshold every episode
    #     params={
    #         "low":  0.10,               # very sensitive
    #         "high": 0.30,               # very coarse
    #     },
    # )

    reinit_dvs: EventTerm = EventTerm(
        func = initialize_stereo_cameras,
        mode = "reset",
        params={
            "yaml_path":      dvs_cam.camera_config["calib_path"],
            "pixel_size_um":   dvs_cam.camera_config["pixel_size_um"],
            "T_w_c0_np":       dvs_cam.camera_config["init_pose"],
            # "visualize": True
        }
    )

    # drops from a new random height every episode reset
    drop_object: EventTerm = EventTerm(
        func=obj_drop.reset_obstacles_random_sparse,
        mode="reset",
        params={
            "xy_range":     (-0.4, 0.4),  # metres left/right of scene centre
            "height_range": (3.0,  3.5),  # metres above ground
        },
    )

    # # Optional: randomize more often than resets
    # randomize_background_mid_episode: EventTerm = EventTerm(
    #     func=event_randomize_background,
    #     mode="interval",
    #     interval_range_s=(5.0, 15.0),     # every 5-15 sim-seconds
    #     params={
    #         "texture_folder":       "resources/dome_texture",
    #         "use_dome":             False,  # dome only changes on reset
    #         "use_backdrop":         True,   # texture swaps more frequently
    #         "backdrop_dist_m":      3.0,
    #         "backdrop_halfsize_m":  5.0,
    #     },
    # )


