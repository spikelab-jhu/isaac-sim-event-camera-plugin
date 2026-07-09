import os
import argparse

# 1. Initialize Isaac Sim before importing any pxr or omni modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

from dvs_gen.data import DATA_DIR

parser = argparse.ArgumentParser(
    description="Download YCB objects from Isaac Nucleus, add colliders + rigid-body "
                "dynamics, and save them as local USD files.")
parser.add_argument("--out", type=str, default=str(DATA_DIR / "ycb_objects"),
                    help="output directory for the processed .usd files "
                         "(default: the bundled dvs_gen data dir)")
args = parser.parse_args()

def add_colliders(root_prim):
    """Iterate descendant prims (including root) and add colliders to mesh or primitive types"""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            # Use convexHull so the physics engine can calculate bounces
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

def has_colliders(root_prim):
    """Check if prim (or its descendants) has colliders"""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False

def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    """Enables rigid body dynamics (physics simulation) on the prim"""
    if has_colliders(prim):
        # Physics
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        
        # PhysX
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
        if angular_damping is not None:
            physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
    else:
        print(f"Prim '{prim.GetPath()}' has no colliders. Skipping rigid body dynamics properties.")

def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
    """Add dynamics properties to the root and colliders to the meshes"""
    add_colliders(prim)
    add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)


def main():
    ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"
    result, entries = omni.client.list(ISAAC_NUCLEUS_DIR)

    ISAAC_OBJECTS = {}

    if result != omni.client.Result.OK:
        print("Error: Failed to list the Isaac assets S3 bucket. Check your network connection.")
        simulation_app.close()
        return
    for entry in entries:
        filename = entry.relative_path
        if filename.endswith(".usd"):
            # Strip the 4-char numeric prefix (e.g. "003_") and the ".usd" suffix
            # so the dictionary key matches the YCB_OBJECTS names
            obj_name = filename[4:-4]
            full_url = f"{ISAAC_NUCLEUS_DIR}/{filename}"
            ISAAC_OBJECTS[obj_name] = full_url
            # print(obj_name)
    print(ISAAC_OBJECTS)
    output_dir = os.path.abspath(args.out)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processing objects to: {output_dir}\n")

    context = omni.usd.get_context()

    for name, url in ISAAC_OBJECTS.items():
        print(f"Processing '{name}'...")
        local_filepath = os.path.join(output_dir, f"{name}.usd")

        # 1. Create a new empty stage
        context.new_stage()
        stage = context.get_stage()

        # 2. Create a root Xform primitive
        root_path = f"/{name}"
        root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        stage.SetDefaultPrim(root_prim)

        # 3. Add the remote Nucleus object as a reference
        # This "downloads" the visual geometry into the stage without breaking textures
        root_prim.GetReferences().AddReference(url)

        # Tick the simulation app once to ensure the reference fully loads into memory
        simulation_app.update()

        # 4. Recursively apply the physics to the root and all child meshes
        add_colliders_and_rigid_body_dynamics(root_prim)

        # 5. Save the local stage
        stage.GetRootLayer().Export(local_filepath)
        print(f" -> Saved physics-enabled object to: {local_filepath}\n")

    print("All objects processed successfully!")
    simulation_app.close()

if __name__ == "__main__":
    main()