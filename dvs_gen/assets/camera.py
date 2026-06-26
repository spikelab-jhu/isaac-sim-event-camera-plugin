"""
camera.py
=========
Stereo DVS rig configuration: where the calibration lives, the sensor pixel
size, and the world pose of cam0. ``cam1``'s pose is derived from cam0 + the
``T_cn_cnm1`` block of the calibration YAML at reset time (see
:func:`dvs_gen.env.events.initialize_stereo_cameras`).

The bundled default calibration is :data:`dvs_gen.data.DATA_DIR`/``stereo.yaml``.
Point ``camera_config["calib_path"]`` at your own Kalibr-style YAML to use a
different rig.
"""
import numpy as np

from dvs_gen.data import DATA_DIR

camera_config = {
    # Kalibr-style stereo calibration (intrinsics, distortion, T_cn_cnm1).
    "calib_path": str(DATA_DIR / "stereo.yaml"),
    # Physical sensor pixel pitch (micrometres) — used to derive focal length.
    "pixel_size_um": 15.0,
    # World pose of cam0 (4x4, metres). cam1 is placed relative to this.
    "init_pose": np.array([
        [1, 0,  0,  0.0],
        [0, 0,  1, -0.7],
        [0, -1, 0,  1.8],
        [0, 0,  0,  1.0],
    ], dtype=np.float64),
}
