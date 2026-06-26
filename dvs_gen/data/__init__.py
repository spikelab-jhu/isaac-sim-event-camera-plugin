"""Bundled package data (YCB USD objects, dome textures, stereo calibration).

Use :data:`DATA_DIR` to build absolute, CWD-independent paths::

    from dvs_gen.data import DATA_DIR
    usd = str(DATA_DIR / "ycb_objects" / "mustard_bottle.usd")
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent

__all__ = ["DATA_DIR"]
