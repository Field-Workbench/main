"""Field Workbench Blender surface-region exporter.

Paste this file into Blender's Scripting workspace and Run Script. It opens a
file picker and writes a matched ``.obj`` + ``.regions.json`` pair from the
active mesh object. Blender material assignments become named Workbench
surface regions.

The original Blender object is not modified. A temporary evaluated mesh is
triangulated for export, so modifiers visible in the viewport are included.
Object transforms are intentionally not baked: exported coordinates are in the
object's local mesh frame. Apply transforms / establish the desired origin in
Blender before export when that frame matters.
"""

import hashlib
import json
import os
import struct

import bmesh
import bpy
from bpy.props import StringProperty
from bpy.types import Operator

SCHEMA = "fieldworkbench.surface_regions"
SCHEMA_VERSION = 1


def _safe_output_paths(filepath: str) -> tuple[str, str]:
    path = os.path.abspath(bpy.path.abspath(filepath))
    root, ext = os.path.splitext(path)
    if ext.lower() == ".obj":
        obj_path = path
    else:
        obj_path = root + ".obj" if ext else path + ".obj"
    root, _ = os.path.splitext(obj_path)
    return obj_path, root + ".regions.json"


def _evaluated_triangulated_mesh(obj: bpy.types.Object) -> bpy.types.Mesh:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    if mesh is None:
        raise RuntimeError("Blender could not evaluate the selected mesh object.")

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if bm.faces:
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()
    return mesh


def _material_name(mesh: bpy.types.Mesh, material_index: int) -> str | None:
    if material_index < 0 or material_index >= len(mesh.materials):
        return None
    material = mesh.materials[material_index]
    if material is None:
        return None
    name = str(material.name).strip()
    return name or None


def _ordered_geometry_sha256(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    for vertex in mesh.vertices:
        digest.update(struct.pack("<3d", *(float(value) for value in vertex.co)))
    for polygon in mesh.polygons:
        digest.update(struct.pack("<3q", *(int(value) for value in polygon.vertices)))
    return digest.hexdigest()


def _write_obj(mesh: bpy.types.Mesh, path: str) -> None:
    # Workbench's OBJ reader preserves this explicit vertex and face order.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Field Workbench matched surface-region mesh\n")
        for vertex in mesh.vertices:
            x, y, z = (float(value) for value in vertex.co)
            handle.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for polygon in mesh.polygons:
            if len(polygon.vertices) != 3:
                raise RuntimeError("Internal triangulation failed; a non-triangle remains.")
            a, b, c = (int(value) + 1 for value in polygon.vertices)
            handle.write(f"f {a} {b} {c}\n")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_active_mesh(filepath: str) -> tuple[str, str, dict]:
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select one mesh object and make it active before running the exporter.")

    obj_path, region_path = _safe_output_paths(filepath)
    os.makedirs(os.path.dirname(obj_path) or ".", exist_ok=True)

    mesh = _evaluated_triangulated_mesh(obj)
    try:
        if not mesh.vertices or not mesh.polygons:
            raise RuntimeError("The evaluated object contains no mesh surface.")

        regions: dict[str, list[int]] = {}
        unassigned = 0
        for polygon in mesh.polygons:
            name = _material_name(mesh, int(polygon.material_index))
            if name is None:
                unassigned += 1
                continue
            regions.setdefault(name, []).append(int(polygon.index))

        if not regions:
            raise RuntimeError(
                "No marked regions were found. Assign at least one material to mesh faces first."
            )

        _write_obj(mesh, obj_path)
        mesh_sha256 = _sha256_file(obj_path)
        source_blend = os.path.basename(bpy.data.filepath) if bpy.data.filepath else ""
        payload = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "asset": os.path.splitext(os.path.basename(obj_path))[0],
            "mesh_file": os.path.basename(obj_path),
            "mesh_sha256": mesh_sha256,
            "authoring": {
                "source_blend": source_blend,
                "source_object": str(obj.name),
                "method": "Blender material-slot face assignments",
                "exporter": "Field Workbench Blender surface-region exporter",
                "blender_unit_system": str(bpy.context.scene.unit_settings.system),
                "blender_unit_scale_length": float(bpy.context.scene.unit_settings.scale_length),
                "blender_length_unit": str(bpy.context.scene.unit_settings.length_unit),
            },
            "topology": {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.polygons),
                "all_triangles": True,
                "ordered_geometry_sha256": _ordered_geometry_sha256(mesh),
            },
            "regions": {
                name: {
                    "display_name": name,
                    "face_count": len(indices),
                    "face_indices": indices,
                }
                for name, indices in regions.items()
            },
        }
        with open(region_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        summary = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.polygons),
            "regions": len(regions),
            "assigned_faces": sum(len(values) for values in regions.values()),
            "unassigned_faces": unassigned,
        }
        return obj_path, region_path, summary
    except Exception:
        # Never leave half of an apparently matched pair behind after a failed run.
        for path in (obj_path, region_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise
    finally:
        bpy.data.meshes.remove(mesh)


class FIELDWORKBENCH_OT_export_surface_regions(Operator):
    bl_idname = "fieldworkbench.export_surface_regions"
    bl_label = "Export Field Workbench Region Mesh"
    bl_description = "Write a matched OBJ and Field Workbench surface-region sidecar"

    filepath: StringProperty(
        name="OBJ filename",
        subtype="FILE_PATH",
        default="surface_regions.obj",
    )
    filter_glob: StringProperty(default="*.obj", options={"HIDDEN"})

    def execute(self, context):
        try:
            obj_path, region_path, summary = export_active_mesh(self.filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        message = (
            f"Exported {summary['regions']} region(s), {summary['assigned_faces']}/"
            f"{summary['faces']} faces mapped: {os.path.basename(obj_path)} + "
            f"{os.path.basename(region_path)}"
        )
        if summary["unassigned_faces"]:
            message += f" ({summary['unassigned_faces']} unassigned faces left unmarked)"
        self.report({"INFO"}, message)
        print("Field Workbench surface-region export complete")
        print(" OBJ:", obj_path)
        print(" Regions:", region_path)
        print(" Summary:", summary)
        return {"FINISHED"}

    def invoke(self, context, event):
        obj = context.active_object
        if obj is not None and obj.type == "MESH":
            stem = bpy.path.clean_name(obj.name) or "surface_regions"
            self.filepath = stem + ".obj"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def _run():
    try:
        bpy.utils.unregister_class(FIELDWORKBENCH_OT_export_surface_regions)
    except Exception:
        pass
    bpy.utils.register_class(FIELDWORKBENCH_OT_export_surface_regions)
    bpy.ops.fieldworkbench.export_surface_regions("INVOKE_DEFAULT")


if __name__ == "__main__":
    _run()
