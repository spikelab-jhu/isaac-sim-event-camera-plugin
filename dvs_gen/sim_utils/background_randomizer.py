"""
background_randomizer.py
Encapsulates all Replicator logic for background / dome randomization.
Designed to be called from Isaac Lab EventManager custom functions.
"""

import glob
import os
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf
import random
from pxr import UsdLux, Gf


# ──────────────────────────────────────────────────────────────
# Internal state — one-time setup guard
# ──────────────────────────────────────────────────────────────
_initialized: bool = False
_dome_node          = None
_backdrop_node      = None
_texture_list: list[str] = []


def _collect_textures(folder: str, exts=("*.png","*.jpg","*.jpeg","*.hdr","*.exr")) -> list[str]:
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    return sorted(paths)


def _make_backdrop_plane(dist_m: float, half_size_m: float) -> str:
    """Create a quad mesh behind the scene and return its USD path."""
    prim_path = "/World/Background/Backdrop"
    stage     = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    parent = "/World/Background"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Scope.Define(stage, parent)

    s    = half_size_m * 100.0          # metres → cm (Isaac default units)
    dist = dist_m      * 100.0

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([(-s,-s,0),(s,-s,0),(s,s,0),(-s,s,0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateNormalsAttr([(0,0,1)]*4)

    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -dist))

    return prim_path


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def setup_background_randomizer(
    texture_folder: str,
    use_dome:       bool  = True,
    use_backdrop:   bool  = True,
    backdrop_dist_m:      float = 3.0,
    backdrop_halfsize_m:  float = 5.0,
):
    global _initialized, _dome_path, _backdrop_path, _texture_list, _hdr_list

    if _initialized:
        return

    _texture_list = _collect_textures(texture_folder)
    if not _texture_list:
        print(f"[bg_randomizer] WARNING: no textures found in '{texture_folder}'")
        return

    _hdr_list = [t for t in _texture_list if t.endswith((".hdr", ".exr"))] or _texture_list

    stage = omni.usd.get_context().get_stage()

    if use_dome:
        # 1. Create a standard Dome Light via USD API
        _dome_path = "/World/BackgroundDomeLight"
        dome_light = UsdLux.DomeLight.Define(stage, _dome_path)
        
        # Set initial texture
        dome_light.CreateTextureFileAttr(_hdr_list[0])
        
        # Add a rotation operation so we can spin it later
        dome_light.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

    if use_backdrop:
        # 2. Get the path from your custom function
        _backdrop_path = _make_backdrop_plane(backdrop_dist_m, backdrop_halfsize_m)

    _initialized = True
    print(f"[bg_randomizer] Ready (Direct USD Mode) — {len(_texture_list)} textures")


def randomize_background():
    """
    Call this from your EventManager. It applies instantly without OmniGraph.
    """
    if not _initialized or not _texture_list:
        return
        
    stage = omni.usd.get_context().get_stage()

    if _dome_path is not None:
        dome_prim = stage.GetPrimAtPath(_dome_path)
        if dome_prim.IsValid():
            # Randomize Dome Texture
            new_hdr = random.choice(_hdr_list)
            dome_prim.GetAttribute("inputs:texture:file").Set(new_hdr)
            
            # Randomize Dome Rotation (0 to 360 degrees on Y axis)
            dome_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0, random.uniform(0.0, 360.0), 0))

    if _backdrop_path is not None:
        # NOTE: You need to target the exact shader prim attached to your backdrop.
        # This assumes _make_backdrop_plane creates a Material/Shader path.
        # Adjust the string below to point to wherever 'inputs:diffuse_texture' lives.
        
        shader_path = f"{_backdrop_path}/Looks/Material/Shader" 
        shader_prim = stage.GetPrimAtPath(shader_path)
        
        if shader_prim.IsValid():
            new_tex = random.choice(_texture_list)
            shader_prim.GetAttribute("inputs:diffuse_texture").Set(new_tex)