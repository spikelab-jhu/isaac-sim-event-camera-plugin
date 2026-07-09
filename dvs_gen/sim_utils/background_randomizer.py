"""
background_randomizer.py
Direct-USD background / dome randomization (no Replicator / OmniGraph).
Designed to be called from Isaac Lab EventManager custom functions.
"""

import glob
import os
import random

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


# ──────────────────────────────────────────────────────────────
# Internal state — one-time setup guard
# ──────────────────────────────────────────────────────────────
_initialized: bool = False
_dome_path: str | None = None
_backdrop_path: str | None = None
_texture_list: list[str] = []
_hdr_list: list[str] = []


def _collect_textures(folder: str, exts=("*.png","*.jpg","*.jpeg","*.hdr","*.exr")) -> list[str]:
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    return sorted(paths)


def _make_backdrop_plane(dist_m: float, half_size_m: float) -> str:
    """Create a textured quad standing behind the scene and return its USD path.

    The quad is authored in metres (Isaac Lab stages are meter/Z-up), rotated
    upright to face the origin, and gets a UsdPreviewSurface material whose
    texture shader lives at ``<prim>/Looks/Material/Texture`` so
    :func:`randomize_background` can swap the image file.
    """
    prim_path = "/World/Background/Backdrop"
    stage     = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    parent = "/World/Background"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Scope.Define(stage, parent)

    s, dist = half_size_m, dist_m                    # metres — the stage unit

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([(-s,-s,0),(s,-s,0),(s,s,0),(-s,s,0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateNormalsAttr([(0,0,1)]*4)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.varying).Set([(0,0),(1,0),(1,1),(0,1)])

    # Stand the quad upright (+90 deg about X turns its +Z normal toward -Y,
    # i.e. facing the origin/camera in the default look-+Y setup) and place it
    # dist metres behind the scene along +Y.
    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, dist, 0))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, 0))

    # UsdPreviewSurface material with a swappable UsdUVTexture.
    material = UsdShade.Material.Define(stage, f"{prim_path}/Looks/Material")
    surface = UsdShade.Shader.Define(stage, f"{prim_path}/Looks/Material/Shader")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, f"{prim_path}/Looks/Material/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{prim_path}/Looks/Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    return prim_path


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def setup_background_randomizer(
    texture_folder: str,
    use_dome:       bool  = True,
    use_backdrop:   bool  = False,
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
        # 2. Textured quad standing behind the scene
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

            # Randomize dome azimuth: stages are Z-up, so spin about Z
            # (a Y rotation would tilt the horizon sideways).
            dome_prim.GetAttribute("xformOp:rotateXYZ").Set(
                Gf.Vec3d(0, 0, random.uniform(0.0, 360.0)))

    if _backdrop_path is not None:
        # The texture shader created by _make_backdrop_plane.
        tex_path = f"{_backdrop_path}/Looks/Material/Texture"
        tex_prim = stage.GetPrimAtPath(tex_path)

        if tex_prim.IsValid():
            new_tex = random.choice(_texture_list)
            tex_prim.GetAttribute("inputs:file").Set(new_tex)
