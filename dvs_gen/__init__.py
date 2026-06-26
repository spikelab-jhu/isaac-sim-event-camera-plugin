"""
dvs_gen — DVS (event camera) data generation in Isaac Lab, accelerated by
motion-vector frame interpolation.

Public surface
--------------
Pure core (importable without Isaac Sim)::

    from dvs_gen import GeneralDVSRecorder, BatchedMultiCamProcessor, bidir_warp_gap

Isaac-side helpers (require the Isaac Lab Python environment); imported lazily so
that the line above keeps working outside Isaac::

    from dvs_gen import DVSCamera, DVSCameraCfg, DVSEnvCfg
"""
# Pure (Omniverse-free) — safe to import anywhere.
from .dvs import GeneralDVSRecorder, BatchedMultiCamProcessor
from .warp import bidir_warp_gap

__all__ = [
    "GeneralDVSRecorder",
    "BatchedMultiCamProcessor",
    "bidir_warp_gap",
    "DVSCamera",
    "DVSCameraCfg",
    "DVSEnvCfg",
]

# Isaac-dependent symbols are resolved lazily (PEP 562) so that importing the
# pure core does not require isaaclab / omni to be installed.
_LAZY = {
    "DVSCamera":    ("dvs_gen.sensors", "DVSCamera"),
    "DVSCameraCfg": ("dvs_gen.sensors", "DVSCameraCfg"),
    "DVSEnvCfg":    ("dvs_gen.env",     "DVSEnvCfg"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module 'dvs_gen' has no attribute {name!r}")
