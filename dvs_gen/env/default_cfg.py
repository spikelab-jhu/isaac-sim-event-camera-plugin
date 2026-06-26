"""
default_cfg.py
==============
``DVSEnvCfg`` — a clean, minimal, ready-to-run Isaac Lab environment for DVS data
generation. One stereo :class:`~dvs_gen.sensors.DVSCameraCfg` pair watches a
single YCB object dropped from a random pose against a randomized dome
background. This is the config used by ``examples/quickstart.py`` and the
default for ``scripts/simulate_warp.py``.

It is intentionally small and well-commented so you can copy it and change one
thing at a time. For the full research configuration (auxiliary annotation
cameras, etc.) see :class:`dvs_gen.env.env_cfg.MyEnvCfg`.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg

from dvs_gen.sensors import DVSCameraCfg
from dvs_gen.assets.objects import YCB_OBJECTS
from dvs_gen.data import DATA_DIR
from .events import MyEventCfg
from .observation import MyObsCfg
from .terminations import MyTerminationCfg
from .env_cfg import MyActionCfg, MyRewardCfg


# ── scene: stereo DVS pair + one dropped object + dome ─────────
@configclass
class DVSSceneCfg(InteractiveSceneCfg):

    # Stereo DVS cameras. DVSCameraCfg auto-requests rgb + motion_vectors +
    # depth (for occlusion-aware warp). Poses/intrinsics are applied at reset
    # from the calibration YAML (see env.events.initialize_stereo_cameras).
    cam0: DVSCameraCfg = DVSCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0",
        update_period=0.001,
        height=480, width=640,
        threshold=0.15,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 1.0e5)),
    )
    cam1: DVSCameraCfg = DVSCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam1",
        update_period=0.001,
        height=480, width=640,
        threshold=0.15,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 1.0e5)),
    )

    # Textured dome background (randomized each reset by the event manager).
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/BackgroundDomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=str(DATA_DIR / "resources" / "dome_texture" / "1.jpg"),
        ),
    )

    # The object whose motion generates the events.
    DroppedObject: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DroppedObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=YCB_OBJECTS["mustard_bottle"],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.05)),
    )


# ── environment ───────────────────────────────────────────────
@configclass
class DVSEnvCfg(ManagerBasedRLEnvCfg):

    # Re-render the RTX cameras twice after each reset so the post-reset scene
    # (new randomized background + new object pose) is reflected before recording
    # — avoids a stale "old background" first frame and its spurious event burst.
    # (`rerender_on_reset` bool is deprecated; use num_rerenders_on_reset.)
    num_rerenders_on_reset = 2

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.001, render_interval=1)
    scene: DVSSceneCfg = DVSSceneCfg(num_envs=1, env_spacing=4.0)

    events: MyEventCfg = MyEventCfg()
    observations: MyObsCfg = MyObsCfg()
    terminations: MyTerminationCfg = MyTerminationCfg()
    actions: MyActionCfg = MyActionCfg()
    rewards: MyRewardCfg = MyRewardCfg()

    episode_length_s: float = 3.0
    decimation: int = 1

    # DVS knobs (kept here for discoverability; the processor reads threshold).
    dvs_threshold: float = 0.15
    dvs_pixel_size_um: float = 15.0

    def __post_init__(self):
        self.sim.dt = 0.001                      # 1 kHz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        self.sim.light = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        self.scene.env_spacing = 4.0
