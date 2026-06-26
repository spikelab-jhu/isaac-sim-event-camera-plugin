"""Isaac Sim / USD-side utilities.

Submodules here import ``omni.*`` / ``pxr`` and must run inside Isaac Sim:
  * :mod:`dvs_gen.sim_utils.camera_usd`            — USD camera placement + calibration
  * :mod:`dvs_gen.sim_utils.background_randomizer` — dome / backdrop randomization

They are intentionally NOT imported here so that ``import dvs_gen.sim_utils``
does not pull in Omniverse; import the submodule you need explicitly.
"""
