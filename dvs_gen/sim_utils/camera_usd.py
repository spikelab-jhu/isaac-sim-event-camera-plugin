"""
camera_usd.py
=============
USD / Omniverse helpers for placing DVS cameras on the live stage and for
parsing stereo calibration. These run *inside* Isaac Sim (they import ``omni.*``
and ``pxr``), unlike the pure event-generation core in :mod:`dvs_gen.dvs`.

``load_calibration`` and ``_parse_matrix_text`` are themselves omni-free, but
they live here alongside the camera-pose helpers that consume them.
"""
import numpy as np
import yaml

# USD / Isaac stage helpers — used to correctly place cameras
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, Gf, Sdf


def load_calibration(yaml_path: str) -> dict:
    """
    Parse a Kalibr-style stereo YAML file and return a normalised dict.

    Supported keys per camera block
    --------------------------------
    intrinsics        : [xi, fx, fy, cx, cy]   (omni model)
                        [fx, fy, cx, cy]        (pinhole model — xi set to 0)
    distortion_coeffs : list of floats
    distortion_model  : str
    camera_model      : str
    resolution        : [W, H]
    T_cn_cnm1         : 4x4 list (cam1 in cam0 frame), absent on cam0
    """
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    result = {}
    for cam_key, cam_data in raw.items():
        if not cam_key.startswith("cam"):
            continue
        intrinsics = cam_data.get("intrinsics", [])
        # Normalise to always store [xi, fx, fy, cx, cy]
        if len(intrinsics) == 4:
            intrinsics = [0.0] + list(intrinsics)   # pinhole: xi = 0
        result[cam_key] = {
            "camera_model":      cam_data.get("camera_model", "pinhole"),
            "distortion_model":  cam_data.get("distortion_model", "radtan"),
            "distortion_coeffs": cam_data.get("distortion_coeffs", []),
            "intrinsics":        intrinsics,          # [xi, fx, fy, cx, cy]
            "resolution":        cam_data.get("resolution", [640, 480]),
            "rostopic":          cam_data.get("rostopic", ""),
            "T_cn_cnm1":         cam_data.get("T_cn_cnm1", None),
        }
    return result


# ──────────────────────────────────────────────────────────────
# USD camera creation  (correct approach for Isaac Lab)
# ──────────────────────────────────────────────────────────────
def _rotation_from_mat4(mat4: np.ndarray) -> Gf.Vec3d:
    """
    Extract ZYX Euler angles (degrees) from the upper-left 3x3 of a 4x4 matrix.
    Returns (rx, ry, rz) suitable for UsdGeom xformOp:rotateXYZ.
    """
    R  = mat4[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy >= 1e-6:
        rx = np.degrees(np.arctan2( R[2, 1],  R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0],  sy))
        rz = np.degrees(np.arctan2( R[1, 0],  R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2],  R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0],  sy))
        rz = 0.0
    return Gf.Vec3d(rx, ry, rz)


def _create_usd_camera(prim_path: str,
                        translate_cm: tuple,
                        euler_xyz_deg: Gf.Vec3d,
                        focal_length_mm: float = 24.0) -> str:
    """
    Create (or replace) a UsdGeom.Camera prim on the live stage.
    Returns the prim_path so Replicator can reference it by string.
    """

    translate_m = [v * 0.01 for v in translate_cm]
    stage = omni.usd.get_context().get_stage()

    # Remove stale prim so we can re-spawn cleanly
    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        stage.RemovePrim(prim_path)

    # Ensure parent scope exists
    parent_path = str(Sdf.Path(prim_path).GetParentPath())
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Scope.Define(stage, parent_path)

    cam_prim = UsdGeom.Camera.Define(stage, prim_path)

    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()

    xform.AddTranslateOp().Set(Gf.Vec3d(*translate_m))
    xform.AddRotateXYZOp().Set(euler_xyz_deg)

    cam_prim.GetFocalLengthAttr().Set(focal_length_mm)

    print(f"[dvs_gen] USD camera → '{prim_path}'  "
          f"pos={tuple(round(v,2) for v in translate_m)}  "
          f"rot_deg={tuple(round(v,3) for v in euler_xyz_deg)}")
    return prim_path


def _spawn_camera_with_writer(prim_path: str,
                               translate_cm: tuple,
                               euler_xyz_deg: Gf.Vec3d,
                               resolution: tuple,
                               writer_name: str,
                               focal_length_mm: float = 24.0):
    """Create a USD camera prim, attach a render product and a DVS writer."""
    _create_usd_camera(prim_path, translate_cm, euler_xyz_deg, focal_length_mm)

    # Replicator references the prim by its USD path string
    render_product = rep.create.render_product(prim_path, resolution)

    writer = rep.writers.get(writer_name)
    writer.initialize()
    writer.attach(render_product)
    return render_product


# ──────────────────────────────────────────────────────────────
# Pose matrix helpers
# ──────────────────────────────────────────────────────────────
def _parse_matrix_text(text: str) -> "np.ndarray | None":
    """
    Accept a 4x4 matrix as free-form text (whitespace / comma / semicolon
    delimiters). Returns (4,4) float64 ndarray or None on failure.
    """
    import re
    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        return None
    if len(vals) == 16:
        return np.array(vals, dtype=np.float64).reshape(4, 4)
    return None


def _mat4_to_translate_and_euler(mat4: np.ndarray) -> "tuple[tuple, Gf.Vec3d]":
    """
    Decompose a 4x4 transform (translation assumed in metres → converted to cm)
    into (translate_cm_tuple, euler_xyz_deg_GfVec3d).
    """
    tx_cm = mat4[0, 3] * 100.0
    ty_cm = mat4[1, 3] * 100.0
    tz_cm = mat4[2, 3] * 100.0
    euler = _rotation_from_mat4(mat4)
    return (tx_cm, ty_cm, tz_cm), euler
