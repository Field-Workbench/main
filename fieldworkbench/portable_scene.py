"""Portable physical-scene export and translator helpers.

The native ``.magpy.json`` document is an editable Workbench project.  This
module deliberately builds a smaller, explicit interchange representation from
that project before any target-format translator runs.  Translators therefore
do not need to understand Workbench UI state, optimizer history, snapshots, or
other editor-only metadata.
"""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from . import __version__
from .getdp_export import (
    GETDP_MAX_COIL_SOURCES,
    GETDP_PROJECT_SUFFIX,
    GETDP_PROJECT_VERSION,
    write_getdp_project,
)
from .studio_adapter import (
    BACKGROUND_FIELD_TYPE,
    MAGNETOMETER_PROBE_TYPE,
    StudioAdapter,
    StudioOperationError,
)

PORTABLE_SCENE_FORMAT = "field_workbench.portable_scene"
PORTABLE_SCENE_VERSION = 1
PORTABLE_PACKAGE_SUFFIX = ".fwpkg"
OPENSCAD_SUFFIX = ".scad"
FEMM_SUFFIX = ".lua"
COMSOL_SUFFIX = ".java"

TRANSLATOR_PORTABLE = "portable"
TRANSLATOR_OPENSCAD = "openscad"
TRANSLATOR_FEMM = "femm"
TRANSLATOR_GETDP = "getdp"
TRANSLATOR_COMSOL = "comsol"

_TRANSLATOR_LABELS = {
    TRANSLATOR_OPENSCAD: "OpenSCAD model (.scad)",
    TRANSLATOR_FEMM: "FEMM axisymmetric builder (.lua)",
    TRANSLATOR_GETDP: "Gmsh/GetDP 3D magnetostatic project (.getdp.zip)",
    TRANSLATOR_COMSOL: "COMSOL 6.4 Java builder (.java)",
    TRANSLATOR_PORTABLE: "Portable scene package (.fwpkg)",
}


def translator_label(translator: str) -> str:
    return _TRANSLATOR_LABELS.get(str(translator), str(translator))


def translator_suffix(translator: str) -> str:
    if translator == TRANSLATOR_OPENSCAD:
        return OPENSCAD_SUFFIX
    if translator == TRANSLATOR_FEMM:
        return FEMM_SUFFIX
    if translator == TRANSLATOR_GETDP:
        return GETDP_PROJECT_SUFFIX
    if translator == TRANSLATOR_COMSOL:
        return COMSOL_SUFFIX
    if translator == TRANSLATOR_PORTABLE:
        return PORTABLE_PACKAGE_SUFFIX
    raise StudioOperationError(f"Unknown scene-export translator: {translator!r}.")


def translator_filter(translator: str) -> str:
    if translator == TRANSLATOR_OPENSCAD:
        return "OpenSCAD model (*.scad)"
    if translator == TRANSLATOR_FEMM:
        return "FEMM Lua builder (*.lua)"
    if translator == TRANSLATOR_GETDP:
        return "Gmsh/GetDP project (*.getdp.zip)"
    if translator == TRANSLATOR_COMSOL:
        return "COMSOL Java builder (*.java)"
    if translator == TRANSLATOR_PORTABLE:
        return "Field Workbench portable package (*.fwpkg)"
    return "All files (*)"


def translator_description(translator: str) -> str:
    if translator == TRANSLATOR_OPENSCAD:
        return (
            "Creates an OpenSCAD model from the portable physical scene. Coils use the saved "
            "winding/bobbin/flange construction when available; otherwise a thin reference "
            "centreline is emitted. Primitive magnets and passive geometry are also translated."
        )
    if translator == TRANSLATOR_FEMM:
        return (
            "Creates a FEMM 4.2 Lua builder for magnetostatic 2D axisymmetric analysis. "
            "Select any circular coil or coaxial set: Workbench automatically maps the selected "
            "common coil axis onto FEMM's Z axis, so the source scene does not need to be rotated "
            "or centred. Passive scene geometry can also be selected as reference-only cross-section "
            "outlines using a literal source XY/XZ/YZ orientation; its offset is automatically locked "
            "to the selected coils' common axis, and either source-side half can be retained as FEMM "
            "+r. Non-coaxial circular selections are rejected."
        )
    if translator == TRANSLATOR_GETDP:
        return (
            "Creates a runnable 3D magnetostatic Gmsh/GetDP project for up to "
            f"{GETDP_MAX_COIL_SOURCES} circular, Square/rectangular, or racetrack coils, with optional "
            "Linear axis sensor samples. Includes editable .geo/.pro files, Linux/Windows run scripts, "
            "reference samples, provenance, and configurable source, mesh, and air-boundary settings."
        )
    if translator == TRANSLATOR_COMSOL:
        return (
            "Creates an experimental COMSOL Multiphysics 6.4 Java API builder using the 3D "
            "Magnetic Fields interface. Coil centrelines become ideal Edge Current loops with "
            "effective current equal to turns × drive current."
        )
    if translator == TRANSLATOR_PORTABLE:
        return (
            "Writes the translator-neutral Field Workbench portable scene package. The package "
            "contains an explicit scene.json plus a manifest and is intended as the stable input "
            "for future translators."
        )
    return ""


def _param_map(adapter: StudioAdapter, object_id: str) -> dict[str, Any]:
    return {
        str(item.get("name", "")): item.get("value")
        for item in adapter.get_params(object_id)
        if str(item.get("name", ""))
    }


def _finite_vector(value: Any, length: int, default: list[float]) -> list[float]:
    try:
        values = [float(component) for component in value]
    except (TypeError, ValueError):
        return list(default)
    if len(values) != length or not all(math.isfinite(component) for component in values):
        return list(default)
    return values




def _square_polyline_geometry(vertices: Any) -> dict[str, Any] | None:
    """Recognize the user-facing Square coil's closed local-XY rectangle.

    Field Workbench stores Square coils as ``current.Polyline`` objects.  Keep
    arbitrary polylines outside the GetDP scope, but recognize the four-corner
    axis-aligned rectangle produced by the Square-coil UI (including rectangular
    X/Y edits made by the optimizer or Scene Recipe).
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4:
        return None
    if not np.all(np.isfinite(points)):
        return None
    scale = max(1.0, float(np.max(np.linalg.norm(points, axis=1))))
    tolerance = max(1e-12, scale * 1e-8)
    if float(np.ptp(points[:, 2])) > tolerance:
        return None
    if float(np.linalg.norm(points[0] - points[-1])) <= tolerance:
        points = points[:-1]
    cleaned: list[np.ndarray] = []
    for point in points:
        if not cleaned or float(np.linalg.norm(point - cleaned[-1])) > tolerance:
            cleaned.append(point.copy())
    if len(cleaned) != 4:
        return None
    loop = np.asarray(cleaned, dtype=float)
    xmin, ymin = np.min(loop[:, :2], axis=0)
    xmax, ymax = np.max(loop[:, :2], axis=0)
    length = float(xmax - xmin)
    width = float(ymax - ymin)
    if length <= tolerance or width <= tolerance:
        return None
    corner_keys: set[tuple[int, int]] = set()
    for point in loop:
        if abs(float(point[0]) - xmin) <= tolerance:
            ix = 0
        elif abs(float(point[0]) - xmax) <= tolerance:
            ix = 1
        else:
            return None
        if abs(float(point[1]) - ymin) <= tolerance:
            iy = 0
        elif abs(float(point[1]) - ymax) <= tolerance:
            iy = 1
        else:
            return None
        corner_keys.add((ix, iy))
    if len(corner_keys) != 4:
        return None
    for start, stop in zip(loop, np.roll(loop, -1, axis=0)):
        delta = np.abs(stop[:2] - start[:2])
        if not ((delta[0] <= tolerance) ^ (delta[1] <= tolerance)):
            return None
    return {
        "shape": "square",
        "centerline_vertices_mm": np.round(loop * 1000.0, 9).tolist(),
        "length_mm": 1000.0 * length,
        "width_mm": 1000.0 * width,
        "center_xy_mm": [500.0 * float(xmin + xmax), 500.0 * float(ymin + ymax)],
    }

def _world_transform(adapter: StudioAdapter, object_id: str) -> dict[str, Any]:
    raw = adapter.get_transform(object_id)
    position = np.asarray(raw.get("position", [0.0, 0.0, 0.0]), dtype=float)
    if position.ndim > 1:
        position = position[-1]
    if position.shape != (3,):
        position = np.zeros(3, dtype=float)
    euler = np.asarray(raw.get("euler", [0.0, 0.0, 0.0]), dtype=float)
    if euler.ndim > 1:
        euler = euler[-1]
    if euler.shape != (3,):
        euler = np.zeros(3, dtype=float)
    rotation = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position * 1000.0
    return {
        "position_mm": np.round(position * 1000.0, 9).tolist(),
        "rotation_euler_xyz_deg": np.round(euler, 9).tolist(),
        # The exact affine matrix is authoritative for translators; it avoids
        # relying on target-format Euler rotation-order conventions.
        "matrix4x4_mm": np.round(matrix, 12).tolist(),
    }


def _group_path(adapter: StudioAdapter, object_id: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        parent_id = str(adapter.get_object(object_id).get("parent") or "")
    except KeyError:
        return result
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        try:
            parent = adapter.get_object(parent_id)
        except KeyError:
            break
        if str(parent.get("type", "")) == "Collection":
            result.append(
                {
                    "id": parent_id,
                    "label": str(parent.get("label") or parent_id),
                }
            )
        parent_id = str(parent.get("parent") or "")
    result.reverse()
    return result


def export_capability(adapter: StudioAdapter, object_id: str, translator: str) -> tuple[bool, str]:
    """Return ``(supported, explanation)`` for one target translator."""
    try:
        obj = adapter.get_object(object_id)
    except KeyError:
        return False, "Object no longer exists"
    if obj.get("derived"):
        return False, "Derived display object"
    native_type = str(obj.get("type", ""))
    if native_type == "Collection":
        return False, "Container; select its descendants"

    if translator == TRANSLATOR_PORTABLE:
        if native_type in {
            "current.Circle",
            "current.Polyline",
            "magnet.Cuboid",
            "magnet.Cylinder",
            "magnet.Sphere",
            "misc.Dipole",
            "Sensor",
            "workbench.Mesh",
            BACKGROUND_FIELD_TYPE,
        }:
            return True, "Portable scene v1"
        return False, "Not represented by portable scene v1"

    if translator == TRANSLATOR_OPENSCAD:
        if native_type in {"current.Circle", "current.Polyline"}:
            return True, "Coil geometry"
        if native_type in {"magnet.Cuboid", "magnet.Cylinder", "magnet.Sphere"}:
            return True, "Primitive magnet geometry"
        if native_type == "workbench.Mesh":
            try:
                mesh_kind = str(adapter.geometry_properties(object_id).get("kind", ""))
            except Exception:  # noqa: BLE001 - capability reporting must remain non-fatal
                mesh_kind = ""
            if mesh_kind in {"box", "cylinder", "sphere", "triangles"}:
                return True, "Passive geometry"
            return False, "Passive geometry kind is not supported by OpenSCAD v1"
        if native_type in {"Sensor", BACKGROUND_FIELD_TYPE, "misc.Dipole"}:
            return False, "No finite CAD solid in OpenSCAD v1"
        return False, "Not supported by OpenSCAD v1"

    if translator == TRANSLATOR_FEMM:
        if native_type == "workbench.Mesh":
            try:
                mesh_kind = str(adapter.geometry_properties(object_id).get("kind", ""))
            except Exception:  # noqa: BLE001 - capability reporting must remain non-fatal
                mesh_kind = ""
            if mesh_kind in {"box", "cylinder", "sphere", "triangles"}:
                return True, "Reference outline only; no FEMM material or magnetic physics"
            return False, "Passive geometry kind cannot be sliced for FEMM reference outlines"
        if native_type != "current.Circle":
            if native_type in {"current.Polyline", "magnet.Cuboid", "magnet.Cylinder", "magnet.Sphere"}:
                return False, "FEMM v1 is restricted to circular coils"
            return False, "Not represented by the FEMM axisymmetric translator"
        try:
            transform = _world_transform(adapter, object_id)
            matrix = np.asarray(transform.get("matrix4x4_mm", np.eye(4)), dtype=float)
            normal = np.asarray(matrix[:3, 2], dtype=float)
            norm = float(np.linalg.norm(normal))
        except Exception:  # noqa: BLE001 - capability reporting must remain non-fatal
            return False, "Could not resolve coil transform"
        if matrix.shape != (4, 4) or not math.isfinite(norm) or norm <= 1e-12:
            return False, "Could not resolve coil axis"
        return True, "Circular coil; selected FEMM coils must share one common axis"

    if translator == TRANSLATOR_GETDP:
        if native_type == "Sensor":
            return True, "Post-processing sample points; no FEM material"
        if native_type not in {"current.Circle", "current.Polyline"}:
            return False, f"Not represented by GetDP project v{GETDP_PROJECT_VERSION}"
        construction_warning = ""
        try:
            physical = adapter.coil_physical_properties(object_id)
            shape = str(physical.get("coil_shape", "generic"))
            square_geometry = None
            if native_type == "current.Polyline" and shape != "racetrack":
                square_geometry = _square_polyline_geometry(_param_map(adapter, object_id).get("vertices", []))
                if square_geometry is None:
                    return False, f"GetDP project v{GETDP_PROJECT_VERSION} supports Square/rectangular coils and parametric racetracks, not arbitrary Polyline coils"
            if bool(physical.get("physical_geometry_enabled", False)):
                assembly = physical.get("calculations", {}).get("assembly")
                if not isinstance(assembly, dict) or not assembly.get("valid", False):
                    details = [
                        str(value)
                        for value in (assembly or {}).get("errors", [])
                        if str(value)
                    ]
                    suffix = f": {'; '.join(details)}" if details else ""
                    construction_warning = (
                        f"; saved physical winding pack is invalid{suffix} — "
                        "choose numerical-source mode for GetDP export"
                    )
        except Exception as error:  # noqa: BLE001 - capability reporting must remain non-fatal
            return False, f"Could not resolve winding data: {error}"
        if native_type == "current.Circle":
            return True, "3D homogenized circular stranded-current source" + construction_warning
        if shape == "racetrack":
            return True, "3D homogenized parametric racetrack stranded-current source" + construction_warning
        return True, "3D homogenized Square/rectangular stranded-current source" + construction_warning

    if translator == TRANSLATOR_COMSOL:
        if native_type in {"current.Circle", "current.Polyline"}:
            return True, "3D ideal Edge Current coil"
        if native_type in {"magnet.Cuboid", "magnet.Cylinder", "magnet.Sphere"}:
            return False, "Permanent-magnet physics is not in COMSOL translator v1"
        if native_type == "workbench.Mesh":
            return False, "Passive geometry is not included in COMSOL magnetic translator v1"
        if native_type in {"Sensor", BACKGROUND_FIELD_TYPE, "misc.Dipole"}:
            return False, "Not included in COMSOL magnetic translator v1"
        return False, "Not supported by COMSOL magnetic translator v1"

    return False, "Unknown translator"


def export_catalog(adapter: StudioAdapter, translator: str) -> list[dict[str, Any]]:
    """Return scene-hierarchy records suitable for the export selection dialog."""
    records: list[dict[str, Any]] = []
    for obj in adapter.list_objects():
        object_id = str(obj.get("id", ""))
        if not object_id or obj.get("derived"):
            continue
        supported, status = export_capability(adapter, object_id, translator)
        native_type = str(obj.get("type", ""))
        records.append(
            {
                "id": object_id,
                "label": str(obj.get("label") or object_id),
                "type": native_type,
                "type_label": adapter.object_type_label(object_id),
                "parent": str(obj.get("parent") or ""),
                "container": native_type == "Collection",
                "supported": supported,
                "status": status,
                "visible": bool(obj.get("visible", True)),
            }
        )
    return records


def getdp_selection_status(
    adapter: StudioAdapter,
    object_ids: list[str],
    use_physical_construction: bool = True,
) -> tuple[bool, str]:
    """Validate the runnable multi-source GetDP project scope."""
    selected = [str(value) for value in object_ids if str(value)]
    if not selected:
        return False, "Select one or more circular/Square/racetrack coils; Linear axis sensors are optional."

    unsupported: list[str] = []
    coil_ids: list[str] = []
    sensor_ids: list[str] = []
    for object_id in selected:
        supported, explanation = export_capability(adapter, object_id, TRANSLATOR_GETDP)
        if not supported:
            try:
                label = str(adapter.get_object(object_id).get("label") or object_id)
            except KeyError:
                label = object_id
            unsupported.append(f"{label}: {explanation}")
            continue
        native_type = str(adapter.get_object(object_id).get("type", ""))
        if native_type in {"current.Circle", "current.Polyline"}:
            if use_physical_construction:
                try:
                    physical = adapter.coil_physical_properties(object_id)
                    if bool(physical.get("physical_geometry_enabled", False)):
                        assembly = physical.get("calculations", {}).get("assembly")
                        if not isinstance(assembly, dict) or not assembly.get("valid", False):
                            details = [
                                str(value)
                                for value in (assembly or {}).get("errors", [])
                                if str(value)
                            ]
                            suffix = f": {'; '.join(details)}" if details else ""
                            label = str(adapter.get_object(object_id).get("label") or object_id)
                            unsupported.append(
                                f"{label}: enabled physical winding pack is invalid{suffix}; "
                                "disable physical construction in GetDP options to use a numerical surrogate"
                            )
                            continue
                except Exception as error:  # noqa: BLE001 - validation boundary
                    unsupported.append(f"{object_id}: could not resolve winding data: {error}")
                    continue
            coil_ids.append(object_id)
        elif native_type == "Sensor":
            sensor_ids.append(object_id)

    if unsupported:
        return False, "Unsupported selection — " + "; ".join(unsupported)
    if not coil_ids:
        return False, f"GetDP project v{GETDP_PROJECT_VERSION} needs at least one selected circular/Square/racetrack coil."
    if len(coil_ids) > GETDP_MAX_COIL_SOURCES:
        return (
            False,
            f"GetDP project v{GETDP_PROJECT_VERSION} supports at most {GETDP_MAX_COIL_SOURCES} circular/Square/racetrack coils in one solve.",
        )

    source_descriptions: list[str] = []
    for coil_id in coil_ids:
        if not use_physical_construction:
            source_descriptions.append("forced numerical winding pack")
        else:
            physical = adapter.coil_physical_properties(coil_id)
            construction = physical.get("calculations", {}).get("assembly")
            if bool(physical.get("physical_geometry_enabled", False)) and isinstance(
                construction, dict
            ):
                source_descriptions.append("saved physical winding pack")
            else:
                source_descriptions.append(
                    "numerical winding pack derived from turns, finished-wire diameter, and packing factor"
                )
    if len(set(source_descriptions)) == 1:
        source_description = source_descriptions[0]
    else:
        source_description = "a mix of saved physical and derived numerical winding packs"

    sample_count = 0
    for sensor_id in sensor_ids:
        transform = adapter.get_transform(sensor_id)
        path = np.asarray(transform.get("path", []), dtype=float)
        if path.ndim == 1 and path.shape == (3,):
            sample_count += 1
        elif path.ndim == 2 and path.shape[1:] == (3,):
            sample_count += len(path)
    sample_note = (
        f"{sample_count:,} selected sensor sample point{'s' if sample_count != 1 else ''}"
        if sample_count
        else "five generated verification points"
    )
    source_count = len(coil_ids)
    source_word = "source" if source_count == 1 else "sources"
    source_shapes: list[str] = []
    for coil_id in coil_ids:
        native_type = str(adapter.get_object(coil_id).get("type", ""))
        if native_type == "current.Circle":
            source_shapes.append("circular")
        elif str(adapter.coil_physical_properties(coil_id).get("coil_shape", "generic")) == "racetrack":
            source_shapes.append("racetrack")
        else:
            source_shapes.append("square")
    unique_shapes = list(dict.fromkeys(source_shapes))
    display_shape = {"circular": "circular", "racetrack": "racetrack", "square": "Square/rectangular"}
    if len(unique_shapes) == 1:
        shape_description = f"{display_shape[unique_shapes[0]]} "
    else:
        mixed_names = ["Square" if shape == "square" else shape for shape in unique_shapes]
        shape_description = f"mixed {'/'.join(mixed_names)} "
    return (
        True,
        f"Ready: {source_count} 3D {shape_description}{source_word} using {source_description}; {sample_note}. "
        "The archive will mesh, solve, and post-process without editing.",
    )

def _construction_record(adapter: StudioAdapter, object_id: str) -> dict[str, Any] | None:
    """Return normalized physical assembly components in local millimetres."""
    # Construction components are already the common geometry used by Workbench
    # rendering and DRC. Reusing them keeps CAD export aligned with what the user
    # sees rather than re-deriving bobbin dimensions in each translator.
    assembly, components = adapter._coil_assembly_components(object_id)  # noqa: SLF001
    if not assembly:
        return None
    result: dict[str, Any] = {
        "enabled": bool(adapter.coil_specs.get(object_id, {}).get("physical_geometry_enabled", False)),
        "supported": bool(assembly.get("supported", False)),
        "valid": bool(assembly.get("valid", False)),
        "shape": str(assembly.get("shape", "")),
        "errors": [str(value) for value in assembly.get("errors", [])],
        "warnings": [str(value) for value in assembly.get("warnings", [])],
        "components": [],
    }
    for component in components:
        kind = str(component.get("solid_kind", ""))
        common = {
            "kind": kind,
            "name": str(component.get("name", "component")),
            "centre_z_mm": 1000.0 * float(component.get("centre_z_m", 0.0)),
            "half_thickness_mm": 1000.0 * float(component.get("half_thickness_m", 0.0)),
        }
        if kind == "annular":
            common.update(
                {
                    "inner_radius_mm": 1000.0 * float(component.get("inner_radius_m", 0.0)),
                    "outer_radius_mm": 1000.0 * float(component.get("outer_radius_m", 0.0)),
                }
            )
        elif kind == "path_band":
            path = np.asarray(component.get("path_xy_m", []), dtype=float).reshape(-1, 2)
            common.update(
                {
                    "path_xy_mm": np.round(path * 1000.0, 9).tolist(),
                    "inner_offset_mm": 1000.0 * float(component.get("inner_offset_m", 0.0)),
                    "outer_offset_mm": 1000.0 * float(component.get("outer_offset_m", 0.0)),
                }
            )
        result["components"].append(common)
    return result


def _coil_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    params = _param_map(adapter, object_id)
    spec = adapter.coil_physical_properties(object_id)
    native_type = str(obj.get("type", ""))
    shape = str(spec.get("coil_shape", "generic"))
    if native_type == "current.Circle":
        geometry: dict[str, Any] = {
            "shape": "circle",
            "mean_diameter_mm": 1000.0 * float(params.get("diameter", 0.0)),
        }
    else:
        vertices = np.asarray(params.get("vertices", []), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            vertices = np.empty((0, 3), dtype=float)
        if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        geometry = {
            "shape": "racetrack" if shape == "racetrack" else "polyline",
            "centerline_vertices_mm": np.round(vertices * 1000.0, 9).tolist(),
        }
        if shape == "racetrack":
            geometry.update(
                {
                    "end_diameter_mm": float(spec.get("racetrack_end_diameter_mm", 0.0)),
                    "straight_length_mm": float(spec.get("racetrack_straight_length_mm", 0.0)),
                }
            )
        else:
            square_geometry = _square_polyline_geometry(params.get("vertices", []))
            if square_geometry is not None:
                geometry = square_geometry

    calculations = spec.get("calculations", {}) if isinstance(spec, dict) else {}
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "coil",
        "native_type": native_type,
        "group_path": _group_path(adapter, object_id),
        "transform": _world_transform(adapter, object_id),
        "geometry": geometry,
        "excitation": {
            "enabled": bool(spec.get("enabled", True)),
            "turns": int(spec.get("turns", 1)),
            "drive_current_a": float(spec.get("drive_current_a", 0.0)),
            "current_mode": str(spec.get("current_mode", "rms_dc")),
            "field_scale_factor": float(spec.get("field_scale_factor", 1.0)),
        },
        "conductor": {
            "awg": int(spec.get("awg", 30)),
            "resistance_20_ohm_per_km": float(spec.get("resistance_20_ohm_per_km", 0.0)),
            "lead_length_mm": 1000.0 * float(spec.get("lead_length_m", 0.0)),
            "enamel_radial_mm": float(spec.get("enamel_radial_mm", 0.0)),
            "packing_factor": float(spec.get("packing_factor", 0.75)),
            "temperature_c": float(spec.get("temperature_c", 20.0)),
        },
        "winding": {
            "axial_width_mm": float(spec.get("winding_axial_width_mm", 0.0)),
            "radial_build_mode": str(spec.get("winding_radial_build_mode", "auto")),
            "radial_build_mm": float(spec.get("winding_radial_build_mm", 0.0)),
            "outer_wire_diameter_mm": float(calculations.get("outer_diameter_mm", 1.0) or 1.0),
        },
        "construction": _construction_record(adapter, object_id),
    }


def _magnet_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    params = _param_map(adapter, object_id)
    native_type = str(obj.get("type", ""))
    subtype = native_type.split(".", 1)[-1].lower()
    geometry: dict[str, Any] = {"shape": subtype}
    if native_type == "magnet.Cuboid":
        geometry["dimensions_mm"] = [1000.0 * value for value in _finite_vector(params.get("dimension"), 3, [0.0, 0.0, 0.0])]
    elif native_type == "magnet.Cylinder":
        dimensions = _finite_vector(params.get("dimension"), 2, [0.0, 0.0])
        geometry.update({"diameter_mm": 1000.0 * dimensions[0], "height_mm": 1000.0 * dimensions[1]})
    elif native_type == "magnet.Sphere":
        geometry["diameter_mm"] = 1000.0 * float(params.get("diameter", 0.0))
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "magnet",
        "native_type": native_type,
        "group_path": _group_path(adapter, object_id),
        "transform": _world_transform(adapter, object_id),
        "geometry": geometry,
        "polarization": _finite_vector(params.get("polarization"), 3, [0.0, 0.0, 0.0]),
    }


def _geometry_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    mesh = adapter._mesh(object_id)  # noqa: SLF001
    geometry = mesh.get("mesh", {})
    kind = str(geometry.get("kind", ""))
    record: dict[str, Any] = {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "geometry",
        "native_type": "workbench.Mesh",
        "group_path": _group_path(adapter, object_id),
        "role": str(mesh.get("role", "visual")),
        "transform": _world_transform(adapter, object_id),
        "geometry": {"shape": kind},
    }
    if mesh.get("builtin_template"):
        record["builtin_template"] = str(mesh.get("builtin_template"))
    if isinstance(mesh.get("well_plate"), dict):
        record["well_plate"] = copy.deepcopy(mesh["well_plate"])
        if mesh.get("role") == "measurement":
            try:
                record["measurement"] = adapter.measurement_properties(object_id)
            except StudioOperationError:
                pass
    if kind == "box":
        record["geometry"]["dimensions_mm"] = (
            np.asarray(geometry.get("dimension", [0.0, 0.0, 0.0]), dtype=float) * 1000.0
        ).tolist()
    elif kind == "sphere":
        record["geometry"]["diameter_mm"] = 1000.0 * float(geometry.get("diameter", 0.0))
    elif kind == "cylinder":
        record["geometry"]["diameter_mm"] = 1000.0 * float(geometry.get("diameter", 0.0))
        record["geometry"]["height_mm"] = 1000.0 * float(geometry.get("height", 0.0))
    elif kind == "triangles":
        # Bake scale/pose into world-space mesh coordinates. This is verbose but
        # completely self-contained and avoids broken source-file paths.
        vertices, faces = adapter._mesh_world_vertices_faces(mesh)  # noqa: SLF001
        record["transform"] = {
            "position_mm": [0.0, 0.0, 0.0],
            "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
            "matrix4x4_mm": np.eye(4, dtype=float).tolist(),
        }
        record["geometry"].update(
            {
                "coordinate_space": "world_mm",
                "vertices_mm": np.round(vertices * 1000.0, 9).tolist(),
                "faces": np.asarray(faces, dtype=int).tolist(),
            }
        )
    return record


def _sensor_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    transform = adapter.get_transform(object_id)
    path = np.asarray(transform.get("path", []), dtype=float)
    if path.ndim == 1 and path.shape == (3,):
        path = path.reshape(1, 3)
    if path.ndim != 2 or path.shape[1:] != (3,):
        path = np.empty((0, 3), dtype=float)
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "sensor",
        "native_type": "Sensor",
        "group_path": _group_path(adapter, object_id),
        "transform": _world_transform(adapter, object_id),
        "sample_path_mm": np.round(path * 1000.0, 9).tolist(),
    }


def _background_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    props = adapter.background_field_properties(object_id)
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "background_field",
        "native_type": BACKGROUND_FIELD_TYPE,
        "group_path": _group_path(adapter, object_id),
        "vector_uT": _finite_vector(props.get("vector_uT"), 3, [0.0, 0.0, 0.0]),
        "mode": str(props.get("mode", "manual")),
    }


def _magnetometer_probe_record(
    adapter: StudioAdapter, object_id: str, obj: dict[str, Any]
) -> dict[str, Any]:
    properties = adapter.magnetometer_probe_properties(object_id)
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "magnetometer_probe",
        "native_type": MAGNETOMETER_PROBE_TYPE,
        "group_path": _group_path(adapter, object_id),
        "transform": _world_transform(adapter, object_id),
        "definition": copy.deepcopy(properties.get("definition", {})),
        "channel_geometry": adapter.magnetometer_probe_sample_geometry(object_id),
    }


def _dipole_record(adapter: StudioAdapter, object_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    params = _param_map(adapter, object_id)
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "kind": "dipole",
        "native_type": "misc.Dipole",
        "group_path": _group_path(adapter, object_id),
        "transform": _world_transform(adapter, object_id),
        "moment": _finite_vector(params.get("moment"), 3, [0.0, 0.0, 0.0]),
    }


def build_portable_scene(adapter: StudioAdapter, object_ids: list[str]) -> dict[str, Any]:
    """Normalize selected Workbench objects into portable scene schema v1."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for value in object_ids:
        object_id = str(value)
        if object_id and object_id not in seen:
            seen.add(object_id)
            unique_ids.append(object_id)
    if not unique_ids:
        raise StudioOperationError("Select at least one exportable scene object.")

    records: list[dict[str, Any]] = []
    for object_id in unique_ids:
        obj = adapter.get_object(object_id)
        native_type = str(obj.get("type", ""))
        if obj.get("derived") or native_type == "Collection":
            continue
        if native_type in {"current.Circle", "current.Polyline"}:
            records.append(_coil_record(adapter, object_id, obj))
        elif native_type in {"magnet.Cuboid", "magnet.Cylinder", "magnet.Sphere"}:
            records.append(_magnet_record(adapter, object_id, obj))
        elif native_type == "workbench.Mesh":
            records.append(_geometry_record(adapter, object_id, obj))
        elif native_type == "Sensor":
            records.append(_sensor_record(adapter, object_id, obj))
        elif native_type == BACKGROUND_FIELD_TYPE:
            records.append(_background_record(adapter, object_id, obj))
        elif native_type == MAGNETOMETER_PROBE_TYPE:
            records.append(_magnetometer_probe_record(adapter, object_id, obj))
        elif native_type == "misc.Dipole":
            records.append(_dipole_record(adapter, object_id, obj))

    if not records:
        raise StudioOperationError("None of the selected objects are represented by portable scene v1.")
    return {
        "format": PORTABLE_SCENE_FORMAT,
        "version": PORTABLE_SCENE_VERSION,
        "generator": {
            "application": "Field Workbench",
            "version": __version__,
        },
        "units": {
            "length": "mm",
            "current": "A",
            "magnetic_flux_density": "uT",
            "angle": "deg",
        },
        "objects": records,
    }


def _scad_number(value: Any) -> str:
    number = float(value)
    if abs(number) < 5e-13:
        number = 0.0
    return f"{number:.9g}"


def _scad_vector(values: Any) -> str:
    return "[" + ", ".join(_scad_number(value) for value in values) + "]"


def _scad_matrix(matrix: Any) -> str:
    rows = ["[" + ", ".join(_scad_number(value) for value in row) + "]" for row in matrix]
    return "[" + ", ".join(rows) + "]"


def _scad_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _scad_polygon(points: list[list[float]]) -> str:
    return "[" + ", ".join(_scad_vector(point[:2]) for point in points) + "]"


def _scad_component(component: dict[str, Any], indent: str = "") -> list[str]:
    lines: list[str] = []
    kind = str(component.get("kind", ""))
    z = float(component.get("centre_z_mm", 0.0))
    half = max(0.0, float(component.get("half_thickness_mm", 0.0)))
    height = 2.0 * half
    if height <= 0.0:
        return lines
    if kind == "annular":
        inner = max(0.0, float(component.get("inner_radius_mm", 0.0)))
        outer = max(inner, float(component.get("outer_radius_mm", 0.0)))
        if outer <= inner:
            return lines
        lines.append(f"{indent}translate([0, 0, {_scad_number(z)}]) difference() {{")
        lines.append(f"{indent}  cylinder(h={_scad_number(height)}, r={_scad_number(outer)}, center=true, $fn=128);")
        if inner > 0.0:
            lines.append(f"{indent}  cylinder(h={_scad_number(height + 0.02)}, r={_scad_number(inner)}, center=true, $fn=128);")
        lines.append(f"{indent}}}")
        return lines
    if kind == "path_band":
        points = component.get("path_xy_mm", [])
        if not isinstance(points, list) or len(points) < 3:
            return lines
        inner = float(component.get("inner_offset_mm", 0.0))
        outer = float(component.get("outer_offset_mm", 0.0))
        if outer <= inner:
            return lines
        polygon = _scad_polygon(points)
        lines.append(f"{indent}translate([0, 0, {_scad_number(z)}]) linear_extrude(height={_scad_number(height)}, center=true, convexity=10) difference() {{")
        lines.append(f"{indent}  offset(delta={_scad_number(outer)}) polygon(points={polygon});")
        lines.append(f"{indent}  offset(delta={_scad_number(inner)}) polygon(points={polygon});")
        lines.append(f"{indent}}}")
    return lines


def _scad_reference_coil(obj: dict[str, Any], indent: str = "") -> list[str]:
    geometry = obj.get("geometry", {})
    winding = obj.get("winding", {})
    thickness = max(0.35, min(3.0, float(winding.get("outer_wire_diameter_mm", 1.0) or 1.0)))
    axial = thickness
    shape = str(geometry.get("shape", ""))
    if shape == "circle":
        diameter = max(0.0, float(geometry.get("mean_diameter_mm", 0.0)))
        radius = diameter / 2.0
        inner = max(0.0, radius - thickness / 2.0)
        outer = radius + thickness / 2.0
        return [
            f"{indent}// No physical assembly enabled: thin centreline reference only.",
            f"{indent}difference() {{",
            f"{indent}  cylinder(h={_scad_number(axial)}, r={_scad_number(outer)}, center=true, $fn=128);",
            f"{indent}  cylinder(h={_scad_number(axial + 0.02)}, r={_scad_number(inner)}, center=true, $fn=128);",
            f"{indent}}}",
        ]
    points = geometry.get("centerline_vertices_mm", [])
    if isinstance(points, list) and len(points) >= 3:
        polygon = _scad_polygon(points)
        return [
            f"{indent}// No physical assembly enabled: thin centreline reference only.",
            f"{indent}linear_extrude(height={_scad_number(axial)}, center=true, convexity=10) difference() {{",
            f"{indent}  offset(delta={_scad_number(thickness / 2.0)}) polygon(points={polygon});",
            f"{indent}  offset(delta={_scad_number(-thickness / 2.0)}) polygon(points={polygon});",
            f"{indent}}}",
        ]
    return [f"{indent}// Coil centreline geometry unavailable."]


def _scad_object_body(obj: dict[str, Any], include_construction: bool) -> list[str]:
    kind = str(obj.get("kind", ""))
    if kind == "coil":
        construction = obj.get("construction")
        components = construction.get("components", []) if isinstance(construction, dict) else []
        if include_construction and construction and construction.get("enabled") and construction.get("valid") and components:
            lines = ["union() {"]
            for component in components:
                name = str(component.get("name", "component"))
                lines.append(f"  // {name}")
                lines.extend(_scad_component(component, "  "))
            lines.append("}")
            return lines
        return _scad_reference_coil(obj)

    geometry = obj.get("geometry", {})
    shape = str(geometry.get("shape", ""))
    if kind == "magnet":
        if shape == "cuboid":
            dims = geometry.get("dimensions_mm", [0.0, 0.0, 0.0])
            return [f"cube({_scad_vector(dims)}, center=true);"]
        if shape == "cylinder":
            return [f"cylinder(h={_scad_number(geometry.get('height_mm', 0.0))}, d={_scad_number(geometry.get('diameter_mm', 0.0))}, center=true, $fn=96);"]
        if shape == "sphere":
            return [f"sphere(d={_scad_number(geometry.get('diameter_mm', 0.0))}, $fn=96);"]

    if kind == "geometry":
        if shape == "box":
            return [f"cube({_scad_vector(geometry.get('dimensions_mm', [0.0, 0.0, 0.0]))}, center=true);"]
        if shape == "cylinder":
            return [f"cylinder(h={_scad_number(geometry.get('height_mm', 0.0))}, d={_scad_number(geometry.get('diameter_mm', 0.0))}, center=true, $fn=96);"]
        if shape == "sphere":
            return [f"sphere(d={_scad_number(geometry.get('diameter_mm', 0.0))}, $fn=96);"]
        if shape == "triangles":
            vertices = geometry.get("vertices_mm", [])
            faces = geometry.get("faces", [])
            points = "[" + ", ".join(_scad_vector(point) for point in vertices) + "]"
            face_text = "[" + ", ".join("[" + ", ".join(str(int(index)) for index in face) + "]" for face in faces) + "]"
            return [f"polyhedron(points={points}, faces={face_text}, convexity=10);"]
    return ["// Unsupported portable object body."]


def render_openscad(scene: dict[str, Any], *, include_construction: bool = True) -> str:
    """Translate a normalized portable scene document into OpenSCAD source."""
    if scene.get("format") != PORTABLE_SCENE_FORMAT:
        raise StudioOperationError("OpenSCAD translator expected a Field Workbench portable scene.")
    lines = [
        "// Field Workbench OpenSCAD export",
        f"// Portable scene schema v{PORTABLE_SCENE_VERSION}; generated by Field Workbench {__version__}",
        "// Units: millimetres. Electromagnetic excitation is retained in comments/portable metadata,",
        "// but OpenSCAD represents geometry only.",
        "",
    ]
    for obj in scene.get("objects", []):
        if str(obj.get("kind", "")) not in {"coil", "magnet", "geometry"}:
            continue
        label = str(obj.get("label") or obj.get("id") or "Object")
        lines.append(f"// --- {label} [{obj.get('id', '')}] ---")
        if obj.get("kind") == "coil":
            excitation = obj.get("excitation", {})
            lines.append(
                "// Coil: turns={turns}; drive_current_a={current}; current_mode={mode}; field_scale_factor={scale}".format(
                    turns=excitation.get("turns", ""),
                    current=_scad_number(excitation.get("drive_current_a", 0.0)),
                    mode=excitation.get("current_mode", ""),
                    scale=_scad_number(excitation.get("field_scale_factor", 1.0)),
                )
            )
        matrix = obj.get("transform", {}).get("matrix4x4_mm", np.eye(4).tolist())
        lines.append(f"multmatrix(m={_scad_matrix(matrix)}) {{")
        for body_line in _scad_object_body(obj, include_construction):
            lines.append("  " + body_line)
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"




def _world_points_for_coil(obj: dict[str, Any], *, circle_segments: int = 96) -> np.ndarray:
    """Return a closed coil centreline in world millimetres."""
    geometry = obj.get("geometry", {})
    shape = str(geometry.get("shape", ""))
    if shape == "circle":
        radius = 0.5 * float(geometry.get("mean_diameter_mm", 0.0))
        if radius <= 0.0:
            return np.empty((0, 3), dtype=float)
        count = max(24, int(circle_segments))
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        local = np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.zeros(count, dtype=float)]
        )
    else:
        local = np.asarray(geometry.get("centerline_vertices_mm", []), dtype=float)
        if local.ndim != 2 or local.shape[1:] != (3,) or len(local) < 2:
            return np.empty((0, 3), dtype=float)
        if len(local) > 1 and np.allclose(local[0], local[-1]):
            local = local[:-1]
    matrix = np.asarray(
        obj.get("transform", {}).get("matrix4x4_mm", np.eye(4).tolist()), dtype=float
    )
    if matrix.shape != (4, 4):
        matrix = np.eye(4, dtype=float)
    homogeneous = np.column_stack([local, np.ones(len(local), dtype=float)])
    world = (matrix @ homogeneous.T).T[:, :3]
    if len(world) and not np.allclose(world[0], world[-1]):
        world = np.vstack([world, world[0]])
    return world



FEMM_AXIS_PARALLEL_TOL = 1e-7
FEMM_COAXIAL_OFFSET_TOL_MM = 1e-4
FEMM_REFERENCE_GAP_MM = 0.25
FEMM_REFERENCE_AXIS_CLEARANCE_MM = 0.25
FEMM_REFERENCE_JOIN_TOL_MM = 1e-4
FEMM_REFERENCE_SIMPLIFY_TOL_MM = 0.02
FEMM_REFERENCE_MIN_SEGMENT_MM = 0.005
FEMM_REFERENCE_SLICE_PLANES = {"auto", "xy", "xz", "yz"}
FEMM_REFERENCE_HALVES = {"positive", "negative"}


def _femm_alignment(scene: dict[str, Any]) -> dict[str, Any]:
    """Validate circular coils and map their common 3D axis onto FEMM Z."""
    coils = [obj for obj in scene.get("objects", []) if str(obj.get("kind", "")) == "coil"]
    if not coils:
        raise StudioOperationError("FEMM export requires at least one circular coil.")

    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for obj in coils:
        geometry = obj.get("geometry", {})
        if str(geometry.get("shape", "")) != "circle":
            raise StudioOperationError("FEMM v1 supports circular coils only.")
        matrix = np.asarray(
            obj.get("transform", {}).get("matrix4x4_mm", np.eye(4).tolist()), dtype=float
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise StudioOperationError(f"{obj.get('label', 'Coil')} has an invalid transform.")
        center = np.asarray(matrix[:3, 3], dtype=float)
        normal = np.asarray(matrix[:3, 2], dtype=float)
        magnitude = float(np.linalg.norm(normal))
        if not math.isfinite(magnitude) or magnitude <= 1e-12:
            raise StudioOperationError(f"{obj.get('label', 'Coil')} has an invalid coil axis.")
        centers.append(center)
        normals.append(normal / magnitude)

    axis = normals[0].copy()
    direction_signs: list[float] = []
    for obj, normal in zip(coils, normals, strict=True):
        dot = float(np.dot(normal, axis))
        cross = float(np.linalg.norm(np.cross(normal, axis)))
        if cross > FEMM_AXIS_PARALLEL_TOL or abs(abs(dot) - 1.0) > FEMM_AXIS_PARALLEL_TOL:
            raise StudioOperationError(
                "FEMM axisymmetric export requires all selected circular coils to be parallel "
                f"or anti-parallel. {obj.get('label', 'A selected coil')} does not share the "
                "common coil axis."
            )
        direction_signs.append(1.0 if dot >= 0.0 else -1.0)

    origin = np.mean(np.vstack(centers), axis=0)
    axial_mm: list[float] = []
    radial_offsets_mm: list[float] = []
    for center in centers:
        delta = center - origin
        axial = float(np.dot(delta, axis))
        radial = delta - axial * axis
        radial_offset = float(np.linalg.norm(radial))
        axial_mm.append(axial)
        radial_offsets_mm.append(radial_offset)

    max_offset = max(radial_offsets_mm, default=0.0)
    if max_offset > FEMM_COAXIAL_OFFSET_TOL_MM:
        worst_index = int(np.argmax(radial_offsets_mm))
        worst = coils[worst_index]
        raise StudioOperationError(
            "FEMM axisymmetric export requires all selected circular coils to be coaxial. "
            f"{worst.get('label', 'A selected coil')} is {max_offset:.6g} mm off the common "
            "axis after removing axial separation."
        )

    return {
        "coils": coils,
        "axis_world": axis,
        "origin_world_mm": origin,
        "axial_positions_mm": axial_mm,
        "direction_signs": direction_signs,
        "max_radial_offset_mm": max_offset,
    }


def _femm_reference_slice_basis(
    alignment: dict[str, Any],
    reference_slice_plane: str,
) -> dict[str, Any]:
    """Return an exact source-world principal plane for FEMM reference guides.

    Passive reference geometry is deliberately *not* part of the axisymmetric
    physics model.  The requested XY/XZ/YZ section therefore follows the same
    literal world-plane semantics as the Workbench 2-D viewer rather than
    silently rotating the requested slice into a meridional plane.

    Within that exact source slice we use only a rigid 2-D rotation when
    mapping the guide into FEMM's r-z display: the projection of the common
    coil axis becomes the vertical direction when it is well-defined.  A
    rigid rotation preserves the source cross-section's shape.
    """
    plane = str(reference_slice_plane or "auto").strip().lower()
    if plane not in FEMM_REFERENCE_SLICE_PLANES:
        raise StudioOperationError("FEMM reference slice plane must be Auto, XY, XZ, or YZ.")

    axis = np.asarray(alignment["axis_world"], dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 1e-12:
        raise StudioOperationError("FEMM reference slice could not resolve the common coil axis.")
    axis = axis / axis_norm

    definitions = {
        "xy": (
            np.asarray([1.0, 0.0, 0.0], dtype=float),
            np.asarray([0.0, 1.0, 0.0], dtype=float),
            np.asarray([0.0, 0.0, 1.0], dtype=float),
        ),
        "xz": (
            np.asarray([1.0, 0.0, 0.0], dtype=float),
            np.asarray([0.0, 0.0, 1.0], dtype=float),
            np.asarray([0.0, 1.0, 0.0], dtype=float),
        ),
        "yz": (
            np.asarray([0.0, 1.0, 0.0], dtype=float),
            np.asarray([0.0, 0.0, 1.0], dtype=float),
            np.asarray([1.0, 0.0, 0.0], dtype=float),
        ),
    }
    requested_plane = plane
    if plane == "auto":
        # Choose the principal plane that most nearly contains the common coil
        # axis, but keep that chosen world plane exact.  XZ wins exact ties to
        # preserve the conventional default for a global-Z coil pair.
        plane = min(
            ("xz", "yz", "xy"),
            key=lambda candidate: abs(float(np.dot(definitions[candidate][2], axis))),
        )

    u_world, v_world, normal_world = definitions[plane]
    axis_uv = np.asarray(
        [float(np.dot(axis, u_world)), float(np.dot(axis, v_world))],
        dtype=float,
    )
    projected_norm = float(np.linalg.norm(axis_uv))
    if projected_norm > 1e-9:
        axial_uv = axis_uv / projected_norm
        # A 90-degree in-plane rotation gives the FEMM +r display direction.
        radial_uv = np.asarray([axial_uv[1], -axial_uv[0]], dtype=float)
        # Keep the radial orientation deterministic and visually close to the
        # source plane's first coordinate (X for XY/XZ, Y for YZ).
        if radial_uv[0] < -1e-12 or (abs(float(radial_uv[0])) <= 1e-12 and radial_uv[1] < 0.0):
            radial_uv *= -1.0
    else:
        # A slice normal to the coil axis has no physically meaningful axial
        # direction in its plane.  It is a visual-only reference, so retain the
        # source viewer orientation exactly: first coordinate -> r, second -> z.
        radial_uv = np.asarray([1.0, 0.0], dtype=float)
        axial_uv = np.asarray([0.0, 1.0], dtype=float)

    radial_world = radial_uv[0] * u_world + radial_uv[1] * v_world
    display_axial_world = axial_uv[0] * u_world + axial_uv[1] * v_world
    return {
        "plane": plane,
        "requested_plane": requested_plane,
        "axis_world": axis,
        "u_world": u_world,
        "v_world": v_world,
        "normal_world": normal_world,
        "radial_uv": radial_uv,
        "axial_uv": axial_uv,
        "radial_world": radial_world,
        "display_axial_world": display_axial_world,
        "axis_projection_norm": projected_norm,
        "axis_parallel_to_slice": abs(float(np.dot(axis, normal_world))) <= 1e-8,
        "exact_world_plane": True,
        "auto_selected": requested_plane == "auto",
    }



def _femm_reference_slice_offset_from_axis(
    alignment: dict[str, Any],
    basis: dict[str, Any],
) -> float:
    """Return the principal-plane offset that passes through the common coil axis."""
    origin = np.asarray(alignment["origin_world_mm"], dtype=float)
    normal = np.asarray(basis["normal_world"], dtype=float)
    return float(np.dot(origin, normal))


def _femm_reference_half_basis(
    basis: dict[str, Any],
    reference_half: str,
) -> dict[str, Any]:
    """Orient FEMM +r toward the requested source-side half of the slice."""
    half = str(reference_half or "positive").strip().lower()
    if half not in FEMM_REFERENCE_HALVES:
        raise StudioOperationError("FEMM reference half must be Positive or Opposite.")
    result = dict(basis)
    sign = 1.0 if half == "positive" else -1.0
    result["radial_uv"] = np.asarray(result["radial_uv"], dtype=float) * sign
    result["radial_world"] = np.asarray(result["radial_world"], dtype=float) * sign
    result["reference_half"] = half
    return result


def _femm_direction_label(vector: np.ndarray) -> str:
    """Compact source-world label for the half retained as FEMM +r."""
    value = np.asarray(vector, dtype=float).reshape(3)
    axes = ("X", "Y", "Z")
    best = int(np.argmax(np.abs(value)))
    if abs(abs(float(value[best])) - 1.0) <= 1e-6 and all(
        abs(float(value[index])) <= 1e-6 for index in range(3) if index != best
    ):
        return ("+" if value[best] >= 0.0 else "−") + axes[best]
    return "[{}]".format(
        ", ".join(f"{component:+.3g}" for component in value)
    )

def _portable_geometry_world_mesh(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate one portable passive-geometry record in world millimetres."""
    if str(obj.get("kind", "")) != "geometry":
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    geometry = obj.get("geometry", {})
    shape = str(geometry.get("shape", ""))

    vertices: np.ndarray
    faces: np.ndarray
    if shape == "triangles":
        vertices = np.asarray(geometry.get("vertices_mm", []), dtype=float)
        faces = np.asarray(geometry.get("faces", []), dtype=int)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        # Imported triangle meshes are baked to world coordinates by the
        # portable-scene builder, but honour the transform if a future schema
        # provides a non-identity one.
    elif shape == "box":
        dims = np.asarray(geometry.get("dimensions_mm", [0.0, 0.0, 0.0]), dtype=float)
        if dims.shape != (3,) or np.any(dims <= 0.0):
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        x, y, z = 0.5 * dims
        vertices = np.asarray(
            [
                [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
            ],
            dtype=float,
        )
        faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            ],
            dtype=int,
        )
    elif shape == "cylinder":
        radius = 0.5 * float(geometry.get("diameter_mm", 0.0))
        height = float(geometry.get("height_mm", 0.0))
        if radius <= 0.0 or height <= 0.0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        count = 64
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        bottom = np.column_stack([ring, np.full(count, -height / 2.0)])
        top = np.column_stack([ring, np.full(count, height / 2.0)])
        vertices = np.vstack([bottom, top, [[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]])
        bottom_center = 2 * count
        top_center = bottom_center + 1
        face_rows: list[list[int]] = []
        for index in range(count):
            nxt = (index + 1) % count
            face_rows.extend(
                [
                    [index, nxt, count + nxt],
                    [index, count + nxt, count + index],
                    [bottom_center, nxt, index],
                    [top_center, count + index, count + nxt],
                ]
            )
        faces = np.asarray(face_rows, dtype=int)
    elif shape == "sphere":
        radius = 0.5 * float(geometry.get("diameter_mm", 0.0))
        if radius <= 0.0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        longitude_count = 64
        latitude_count = 24
        rows: list[list[float]] = [[0.0, 0.0, -radius]]
        for lat_index in range(1, latitude_count):
            phi = -0.5 * math.pi + math.pi * lat_index / latitude_count
            ring_radius = radius * math.cos(phi)
            z = radius * math.sin(phi)
            for lon_index in range(longitude_count):
                theta = 2.0 * math.pi * lon_index / longitude_count
                rows.append([ring_radius * math.cos(theta), ring_radius * math.sin(theta), z])
        rows.append([0.0, 0.0, radius])
        vertices = np.asarray(rows, dtype=float)
        bottom = 0
        top = len(vertices) - 1
        face_rows = []
        first_ring = 1
        for lon_index in range(longitude_count):
            nxt = (lon_index + 1) % longitude_count
            face_rows.append([bottom, first_ring + lon_index, first_ring + nxt])
        for lat_index in range(latitude_count - 2):
            row0 = first_ring + lat_index * longitude_count
            row1 = row0 + longitude_count
            for lon_index in range(longitude_count):
                nxt = (lon_index + 1) % longitude_count
                face_rows.extend(
                    [
                        [row0 + lon_index, row1 + lon_index, row1 + nxt],
                        [row0 + lon_index, row1 + nxt, row0 + nxt],
                    ]
                )
        last_ring = first_ring + (latitude_count - 2) * longitude_count
        for lon_index in range(longitude_count):
            nxt = (lon_index + 1) % longitude_count
            face_rows.append([top, last_ring + nxt, last_ring + lon_index])
        faces = np.asarray(face_rows, dtype=int)
    else:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    matrix = np.asarray(
        obj.get("transform", {}).get("matrix4x4_mm", np.eye(4).tolist()), dtype=float
    )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        matrix = np.eye(4, dtype=float)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices), dtype=float)])
    world = (matrix @ homogeneous.T).T[:, :3]
    return world, faces


def _clip_femm_reference_segment_positive_r(
    first_mm: np.ndarray,
    second_mm: np.ndarray,
    *,
    min_r_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a reference segment to FEMM's retained radial half-plane.

    Passive guide geometry deliberately uses a small positive ``min_r_mm``
    clearance.  A merely open half-contour whose two ends land on FEMM's
    symmetry axis otherwise becomes a *closed* FEMM region together with the
    r=0 axis and therefore demands an unwanted block label.
    """
    first = np.asarray(first_mm, dtype=float).reshape(2).copy()
    second = np.asarray(second_mm, dtype=float).reshape(2).copy()
    minimum_r = max(0.0, float(min_r_mm))
    tolerance = 1e-9
    if first[0] < minimum_r - tolerance and second[0] < minimum_r - tolerance:
        return None
    if first[0] < minimum_r or second[0] < minimum_r:
        denominator = second[0] - first[0]
        if abs(float(denominator)) <= 1e-15:
            return None
        parameter = (minimum_r - first[0]) / denominator
        if parameter < -tolerance or parameter > 1.0 + tolerance:
            return None
        crossing = first + parameter * (second - first)
        crossing[0] = minimum_r
        if first[0] < minimum_r:
            first = crossing
        else:
            second = crossing
    first[0] = max(minimum_r, float(first[0]))
    second[0] = max(minimum_r, float(second[0]))
    if float(np.linalg.norm(second - first)) <= 1e-8:
        return None
    return first, second



def _femm_reference_point_key(point_mm: np.ndarray, tolerance_mm: float) -> tuple[int, int]:
    """Stable quantized key used to join triangle-slice endpoints."""
    point = np.asarray(point_mm, dtype=float).reshape(2)
    tolerance = max(float(tolerance_mm), 1e-12)
    return (
        int(round(float(point[0]) / tolerance)),
        int(round(float(point[1]) / tolerance)),
    )


def _femm_reference_rdp(points_mm: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for a single open guide polyline."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) <= 2 or float(tolerance_mm) <= 0.0:
        return points.copy()
    start = points[0]
    end = points[-1]
    baseline = end - start
    baseline_length = float(np.linalg.norm(baseline))
    if baseline_length <= 1e-15:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        rel = points - start
        distances = np.abs(baseline[0] * rel[:, 1] - baseline[1] * rel[:, 0]) / baseline_length
    split = int(np.argmax(distances))
    if float(distances[split]) <= float(tolerance_mm):
        return np.vstack([start, end])
    left = _femm_reference_rdp(points[: split + 1], tolerance_mm)
    right = _femm_reference_rdp(points[split:], tolerance_mm)
    return np.vstack([left[:-1], right])


def _femm_reference_stitch_segments(
    segments_mm: list[tuple[np.ndarray, np.ndarray]],
    *,
    join_tolerance_mm: float = FEMM_REFERENCE_JOIN_TOL_MM,
) -> list[dict[str, Any]]:
    """Join raw triangle intersections into unique open/closed contour chains.

    The 2-D viewer can cheaply draw one line per intersected STL triangle. FEMM's
    geometry editor is much less tolerant of that representation: duplicate
    edges, coincident endpoints, and tiny tessellation slivers can make geometry
    insertion pathologically slow.  Build a small undirected graph of the
    intersection instead and traverse each unique edge exactly once.
    """
    if not segments_mm:
        return []

    points: list[np.ndarray] = []
    key_to_index: dict[tuple[int, int], int] = {}

    def point_index(point_mm: np.ndarray) -> int:
        point = np.asarray(point_mm, dtype=float).reshape(2)
        key = _femm_reference_point_key(point, join_tolerance_mm)
        existing = key_to_index.get(key)
        if existing is not None:
            return existing
        index = len(points)
        points.append(point.copy())
        key_to_index[key] = index
        return index

    edges: set[tuple[int, int]] = set()
    for first, second in segments_mm:
        first_index = point_index(first)
        second_index = point_index(second)
        if first_index == second_index:
            continue
        edge = (
            (first_index, second_index)
            if first_index < second_index
            else (second_index, first_index)
        )
        edges.add(edge)
    if not edges:
        return []

    adjacency: dict[int, set[int]] = defaultdict(set)
    for first_index, second_index in edges:
        adjacency[first_index].add(second_index)
        adjacency[second_index].add(first_index)

    unused = set(edges)
    chains: list[dict[str, Any]] = []

    def edge_key(first_index: int, second_index: int) -> tuple[int, int]:
        return (
            (first_index, second_index)
            if first_index < second_index
            else (second_index, first_index)
        )

    def walk(first_index: int, second_index: int) -> list[int]:
        line = [first_index, second_index]
        unused.discard(edge_key(first_index, second_index))
        previous = first_index
        current = second_index
        while len(adjacency[current]) == 2:
            candidates = [
                value
                for value in adjacency[current]
                if value != previous and edge_key(current, value) in unused
            ]
            if not candidates:
                break
            nxt = candidates[0]
            unused.discard(edge_key(current, nxt))
            line.append(nxt)
            previous, current = current, nxt
        return line

    # Consume open chains and non-manifold branch arms first.  Degree-two
    # leftovers are closed loops and are handled below.
    starts = [index for index, neighbours in adjacency.items() if len(neighbours) != 2]
    for start in starts:
        for neighbour in list(adjacency[start]):
            if edge_key(start, neighbour) not in unused:
                continue
            indices = walk(start, neighbour)
            chains.append(
                {
                    "points_mm": np.asarray([points[index] for index in indices], dtype=float),
                    "closed": False,
                }
            )

    while unused:
        first_index, second_index = next(iter(unused))
        indices = [first_index, second_index]
        unused.discard(edge_key(first_index, second_index))
        previous = first_index
        current = second_index
        while True:
            candidates = [
                value
                for value in adjacency[current]
                if value != previous and edge_key(current, value) in unused
            ]
            if not candidates:
                break
            nxt = candidates[0]
            unused.discard(edge_key(current, nxt))
            indices.append(nxt)
            previous, current = current, nxt
            if current == indices[0]:
                break
        closed = len(indices) >= 3 and indices[-1] == indices[0]
        chains.append(
            {
                "points_mm": np.asarray([points[index] for index in indices], dtype=float),
                "closed": closed,
            }
        )

    return chains


def _femm_reference_gap_segment(
    first_mm: np.ndarray,
    second_mm: np.ndarray,
    *,
    gap_mm: float = FEMM_REFERENCE_GAP_MM,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut a centred FEMM-safe break into one guide segment."""
    first = np.asarray(first_mm, dtype=float).reshape(2).copy()
    second = np.asarray(second_mm, dtype=float).reshape(2).copy()
    vector = second - first
    length = float(np.linalg.norm(vector))
    if length < FEMM_REFERENCE_MIN_SEGMENT_MM:
        return []

    requested_gap = max(float(gap_mm), FEMM_REFERENCE_MIN_SEGMENT_MM)
    # Keep a useful stub on both sides when possible.  For a short edge it is
    # safer to omit the whole edge than to leave a microscopic FEMM closure.
    available = length - 2.0 * FEMM_REFERENCE_MIN_SEGMENT_MM
    if available <= FEMM_REFERENCE_MIN_SEGMENT_MM:
        return []
    actual_gap = min(requested_gap, available)
    direction = vector / length
    midpoint = 0.5 * (first + second)
    left = midpoint - 0.5 * actual_gap * direction
    right = midpoint + 0.5 * actual_gap * direction
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    if float(np.linalg.norm(left - first)) >= FEMM_REFERENCE_MIN_SEGMENT_MM:
        pieces.append((first, left))
    if float(np.linalg.norm(second - right)) >= FEMM_REFERENCE_MIN_SEGMENT_MM:
        pieces.append((right, second))
    return pieces


def _femm_reference_choose_break_edge(points_mm: np.ndarray) -> int:
    """Choose a stable, inconspicuous edge on which to open a closed guide."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) < 2:
        return 0
    next_points = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(next_points - points, axis=1)
    # Prefer a long edge away from the symmetry axis.  This gives FEMM a
    # robust numerical gap while keeping the visual interruption unobtrusive.
    mid_r = 0.5 * (points[:, 0] + next_points[:, 0])
    axis_margin = np.maximum(0.0, mid_r - FEMM_REFERENCE_AXIS_CLEARANCE_MM)
    score = lengths * (1.0 + np.minimum(axis_margin, 10.0) / 100.0)
    return int(np.argmax(score))



def _femm_reference_simplify_closed(
    points_mm: np.ndarray,
    tolerance_mm: float,
) -> np.ndarray:
    """Simplify a closed contour without inventing a short pre-simplification gap."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) <= 3 or float(tolerance_mm) <= 0.0:
        return points.copy()

    # Split the ring at an approximate diameter, simplify both open arcs, then
    # reassemble the ring.  This avoids the degenerate same-start/end baseline
    # that ordinary RDP sees on a closed polygon.
    first_index = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    second_index = int(np.argmax(np.linalg.norm(points - points[first_index], axis=1)))
    if first_index == second_index:
        return points.copy()

    def arc(start: int, stop: int) -> np.ndarray:
        rows = [points[start].copy()]
        index = start
        while index != stop:
            index = (index + 1) % len(points)
            rows.append(points[index].copy())
        return np.asarray(rows, dtype=float)

    first_arc = _femm_reference_rdp(arc(first_index, second_index), tolerance_mm)
    second_arc = _femm_reference_rdp(arc(second_index, first_index), tolerance_mm)
    combined = np.vstack([first_arc[:-1], second_arc[:-1]])
    return combined if len(combined) >= 3 else points.copy()

def _femm_reference_open_closed_chain(points_mm: np.ndarray) -> np.ndarray:
    """Return one closed contour as an explicitly open polyline with a real gap."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) < 3:
        return points.copy()
    gap_index = _femm_reference_choose_break_edge(points)
    next_index = (gap_index + 1) % len(points)
    first_gap = points[gap_index]
    second_gap = points[next_index]
    vector = second_gap - first_gap
    length = float(np.linalg.norm(vector))
    if length <= 2.0 * FEMM_REFERENCE_MIN_SEGMENT_MM:
        # Removing the complete short edge still opens the loop robustly.
        right = second_gap.copy()
        left = first_gap.copy()
    else:
        actual_gap = min(
            FEMM_REFERENCE_GAP_MM,
            max(FEMM_REFERENCE_MIN_SEGMENT_MM, length - 2.0 * FEMM_REFERENCE_MIN_SEGMENT_MM),
        )
        direction = vector / length
        midpoint = 0.5 * (first_gap + second_gap)
        left = midpoint - 0.5 * actual_gap * direction
        right = midpoint + 0.5 * actual_gap * direction

    ordered: list[np.ndarray] = [right.copy()]
    index = next_index
    while True:
        ordered.append(points[index].copy())
        if index == gap_index:
            break
        index = (index + 1) % len(points)
    ordered.append(left.copy())
    return np.asarray(ordered, dtype=float)


def _femm_reference_cycle_edge_indices(
    segments_mm: list[tuple[np.ndarray, np.ndarray]],
) -> list[int]:
    """Return guide edges that would close cycles in the endpoint graph.

    This is a final translator-side sanity check, independent of how the raw
    triangle slice was stitched.  Sorting short edges first tends to reserve a
    longer, cleaner edge for the deliberate break when a cycle survives.
    """
    if not segments_mm:
        return []

    parent: dict[tuple[int, int], tuple[int, int]] = {}
    rank: dict[tuple[int, int], int] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        rank.setdefault(node, 0)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: tuple[int, int], second: tuple[int, int]) -> bool:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return False
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1
        return True

    indexed = list(enumerate(segments_mm))
    indexed.sort(key=lambda item: float(np.linalg.norm(item[1][1] - item[1][0])))
    cycle_indices: list[int] = []
    for index, (first, second) in indexed:
        first_key = _femm_reference_point_key(first, FEMM_REFERENCE_JOIN_TOL_MM)
        second_key = _femm_reference_point_key(second, FEMM_REFERENCE_JOIN_TOL_MM)
        if first_key == second_key or not union(first_key, second_key):
            cycle_indices.append(index)
    return cycle_indices


def _femm_reference_sanity_open_records(
    records: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], dict[str, int]]:
    """Guarantee that combined emitted reference guides contain no closed cycles."""
    working = [
        (int(owner), np.asarray(first, dtype=float).copy(), np.asarray(second, dtype=float).copy())
        for owner, first, second in records
    ]
    breaks = 0
    duplicate_count = 0

    # Remove exact/reversed duplicates across different passive objects too.
    unique: list[tuple[int, np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for owner, first, second in working:
        first_key = _femm_reference_point_key(first, FEMM_REFERENCE_JOIN_TOL_MM)
        second_key = _femm_reference_point_key(second, FEMM_REFERENCE_JOIN_TOL_MM)
        edge_key = tuple(sorted((first_key, second_key)))
        if edge_key in seen:
            duplicate_count += 1
            continue
        seen.add(edge_key)
        unique.append((owner, first, second))
    working = unique

    segments = [(first, second) for _, first, second in working]
    cycle_indices = set(_femm_reference_cycle_edge_indices(segments))
    if cycle_indices:
        repaired: list[tuple[int, np.ndarray, np.ndarray]] = []
        for index, (owner, first, second) in enumerate(working):
            if index in cycle_indices:
                for gap_first, gap_second in _femm_reference_gap_segment(first, second):
                    repaired.append((owner, gap_first, gap_second))
                breaks += 1
            else:
                repaired.append((owner, first, second))
        working = repaired

    residual = _femm_reference_cycle_edge_indices(
        [(first, second) for _, first, second in working]
    )
    if residual:
        raise StudioOperationError(
            "FEMM reference-outline sanity check could not open all closed guide contours."
        )

    axis_contacts = sum(
        1
        for _, first, second in working
        for point in (first, second)
        if float(point[0]) < FEMM_REFERENCE_AXIS_CLEARANCE_MM - FEMM_REFERENCE_JOIN_TOL_MM
    )
    if axis_contacts:
        raise StudioOperationError(
            "FEMM reference-outline sanity check found guide geometry touching the r=0 axis."
        )

    return working, {
        "sanity_break_count": breaks,
        "global_duplicate_count": duplicate_count,
        "residual_cycle_count": len(residual),
        "axis_contact_count": axis_contacts,
    }


def _femm_reference_sanity_open_segments(
    segments_mm: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Convenience wrapper for sanity-checking an ungrouped guide set."""
    records = [(0, first, second) for first, second in segments_mm]
    sane, stats = _femm_reference_sanity_open_records(records)
    return [(first, second) for _, first, second in sane], stats


def _femm_reference_clean_segments(
    segments_mm: list[tuple[np.ndarray, np.ndarray]],
    *,
    simplify_tolerance_mm: float = FEMM_REFERENCE_SIMPLIFY_TOL_MM,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Convert raw STL triangle cuts into FEMM-safe visual guide segments."""
    raw_count = len(segments_mm)
    chains = _femm_reference_stitch_segments(segments_mm)
    cleaned: list[tuple[np.ndarray, np.ndarray]] = []
    candidate_edge_count = 0
    contour_break_count = 0

    for chain in chains:
        points = np.asarray(chain["points_mm"], dtype=float)
        if len(points) < 2:
            continue
        closes_geometrically = (
            len(points) >= 3
            and float(np.linalg.norm(points[0] - points[-1])) <= FEMM_REFERENCE_JOIN_TOL_MM
        )
        closed = bool(chain.get("closed", False)) or closes_geometrically
        if closes_geometrically:
            points = points[:-1]
        if len(points) < 2:
            continue

        if closed and len(points) >= 3:
            points = _femm_reference_simplify_closed(points, simplify_tolerance_mm)
            points = _femm_reference_open_closed_chain(points)
            contour_break_count += 1
            simplified = points
        else:
            simplified = _femm_reference_rdp(points, simplify_tolerance_mm)
        if len(simplified) < 2:
            continue
        for first, second in zip(simplified[:-1], simplified[1:], strict=True):
            candidate_edge_count += 1
            if float(np.linalg.norm(second - first)) >= FEMM_REFERENCE_MIN_SEGMENT_MM:
                cleaned.append((first.copy(), second.copy()))

    return cleaned, {
        "raw_segment_count": raw_count,
        "chain_count": len(chains),
        "clean_segment_count": len(cleaned),
        "candidate_segment_count": candidate_edge_count,
        "contour_break_count": contour_break_count,
    }


def _femm_reference_sections(
    scene: dict[str, Any],
    alignment: dict[str, Any],
    reference_slice_plane: str,
    reference_slice_offset_mm: float | None = None,
    reference_half: str = "positive",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Slice selected passive geometry into non-material FEMM guide segments.

    The world-plane intersection is the same literal XY/XZ/YZ section used by
    the 2-D viewer.  The resulting 2-D coordinates are then only rotated and
    translated into FEMM's r-z display, so the source outline cannot be
    stretched or sheared by the axisymmetric normalization.
    """
    basis = _femm_reference_slice_basis(alignment, reference_slice_plane)
    basis = _femm_reference_half_basis(basis, reference_half)
    plane = str(basis["plane"])
    normal = np.asarray(basis["normal_world"], dtype=float)
    u_world = np.asarray(basis["u_world"], dtype=float)
    v_world = np.asarray(basis["v_world"], dtype=float)
    radial_uv = np.asarray(basis["radial_uv"], dtype=float)
    axial_uv = np.asarray(basis["axial_uv"], dtype=float)
    origin = np.asarray(alignment["origin_world_mm"], dtype=float)
    if reference_slice_offset_mm is None:
        offset_mm = _femm_reference_slice_offset_from_axis(alignment, basis)
    else:
        offset_mm = float(reference_slice_offset_mm)
        if not math.isfinite(offset_mm):
            raise StudioOperationError("FEMM reference slice offset must be finite.")

    # Project the common-axis origin onto the chosen source slice.  Using this
    # as the 2-D display anchor keeps the reference sketch near the normalized
    # coil model without altering its shape.
    anchor_world = origin + (offset_mm - float(np.dot(origin, normal))) * normal
    anchor_uv = np.asarray(
        [float(np.dot(anchor_world, u_world)), float(np.dot(anchor_world, v_world))],
        dtype=float,
    )
    results: list[dict[str, Any]] = []

    for obj in scene.get("objects", []):
        if str(obj.get("kind", "")) != "geometry":
            continue
        world_vertices, faces = _portable_geometry_world_mesh(obj)
        if not len(world_vertices) or not len(faces):
            continue
        extent_mm = max(1.0, float(np.max(np.abs(world_vertices))))
        source_segments_m = StudioAdapter._mesh_plane_segments_2d(  # noqa: SLF001
            world_vertices / 1000.0,
            faces,
            plane=plane,
            offset_m=offset_mm / 1000.0,
            half_span_m=(extent_mm * 4.0 + abs(offset_mm) + 10.0) / 1000.0,
        )
        raw_segments_mm: list[tuple[np.ndarray, np.ndarray]] = []
        for first_m, second_m in source_segments_m:
            first_uv = np.asarray(first_m, dtype=float) * 1000.0 - anchor_uv
            second_uv = np.asarray(second_m, dtype=float) * 1000.0 - anchor_uv
            first_rz = np.asarray(
                [float(np.dot(first_uv, radial_uv)), float(np.dot(first_uv, axial_uv))],
                dtype=float,
            )
            second_rz = np.asarray(
                [float(np.dot(second_uv, radial_uv)), float(np.dot(second_uv, axial_uv))],
                dtype=float,
            )
            clipped = _clip_femm_reference_segment_positive_r(
                first_rz,
                second_rz,
                min_r_mm=FEMM_REFERENCE_AXIS_CLEARANCE_MM,
            )
            if clipped is None:
                continue
            first, second = clipped
            if float(np.linalg.norm(second - first)) > 1e-8:
                raw_segments_mm.append((first, second))

        # FEMM is a CAD/FEA preprocessor, not a triangle-mesh line renderer.
        # Feeding it one independent line for every intersected STL triangle
        # can create thousands of coincident endpoints, duplicate edges, and
        # microscopic slivers.  The Workbench preview tolerates that raw form;
        # FEMM can become non-responsive while trying to reconcile it.  Clean
        # the triangle cut into unique contour chains before writing Lua.
        segments_mm, cleanup = _femm_reference_clean_segments(raw_segments_mm)
        if segments_mm:
            results.append(
                {
                    "id": str(obj.get("id", "")),
                    "label": str(obj.get("label") or obj.get("id") or "Reference object"),
                    "segments_mm": segments_mm,
                    **cleanup,
                }
            )

    # Final combined-geometry sanity pass.  This catches a loop assembled from
    # multiple passive objects even when each individual source contour was
    # open.  Section ownership is preserved when a final gap is inserted.
    sanity = {
        "sanity_break_count": 0,
        "global_duplicate_count": 0,
        "residual_cycle_count": 0,
        "axis_contact_count": 0,
    }
    if results:
        records: list[tuple[int, np.ndarray, np.ndarray]] = []
        for section_index, section in enumerate(results):
            for first, second in section["segments_mm"]:
                records.append((section_index, first, second))
        sane_records, sanity = _femm_reference_sanity_open_records(records)
        reassigned: list[list[tuple[np.ndarray, np.ndarray]]] = [list() for _ in results]
        for owner, first, second in sane_records:
            reassigned[owner].append((first, second))
        for section_index, section in enumerate(results):
            section["segments_mm"] = reassigned[section_index]
            section["clean_segment_count"] = len(reassigned[section_index])


    basis = dict(basis)
    basis["offset_mm"] = offset_mm
    basis["anchor_world_mm"] = anchor_world
    basis["reference_sanity"] = sanity
    return basis, results

def femm_selection_status(
    adapter: StudioAdapter,
    object_ids: list[str],
    reference_slice_plane: str = "auto",
    reference_half: str = "positive",
) -> tuple[bool, str]:
    """Validate the jointly selected FEMM coil set for the export dialog."""
    selected = [str(value) for value in object_ids if str(value)]
    if not selected:
        return False, "Select one or more circular coils."
    try:
        scene = build_portable_scene(adapter, selected)
        alignment = _femm_alignment(scene)
    except (StudioOperationError, KeyError, ValueError, TypeError) as error:
        return False, str(error)
    count = len(alignment["coils"])
    reference_count = sum(
        1 for obj in scene.get("objects", []) if str(obj.get("kind", "")) == "geometry"
    )
    basis = None
    if reference_count:
        try:
            basis = _femm_reference_slice_basis(alignment, reference_slice_plane)
            basis = _femm_reference_half_basis(basis, reference_half)
        except (StudioOperationError, KeyError, ValueError, TypeError) as error:
            return False, str(error)
    axis = np.asarray(alignment["axis_world"], dtype=float)
    origin = np.asarray(alignment["origin_world_mm"], dtype=float)
    if basis is not None:
        plane = str(basis["plane"]).upper()
        offset_mm = _femm_reference_slice_offset_from_axis(alignment, basis)
        plane_note = "exact source plane"
        if basis.get("auto_selected"):
            plane_note = f"auto-selected {plane_note}"
        if not basis.get("axis_parallel_to_slice", False):
            mapping_note = (
                " This orientation intersects the common axis at its reference origin but does "
                "not contain the full coil-axis direction; its outline is a shape-preserving "
                "visual reference only."
            )
        else:
            mapping_note = " The slice contains the common coil axis."
        half_label = _femm_direction_label(np.asarray(basis["radial_world"], dtype=float))
        reference_note = (
            f" {reference_count} selected passive object{'s' if reference_count != 1 else ''} will be "
            f"sliced on the {plane} {plane_note}; its offset is automatically locked to the common "
            f"axis at {offset_mm:g} mm. The {half_label} source side is retained as FEMM +r and "
            "added as reference-only guide outlines." + mapping_note
        )
    else:
        reference_note = " No passive reference geometry is selected."
    return (
        True,
        "FEMM will automatically map the selected "
        f"{count} coil{'s' if count != 1 else ''} from source axis "
        f"[{axis[0]:.4g}, {axis[1]:.4g}, {axis[2]:.4g}] through "
        f"[{origin[0]:.4g}, {origin[1]:.4g}, {origin[2]:.4g}] mm onto canonical Z."
        + reference_note
    )

def _femm_coil_section(obj: dict[str, Any]) -> dict[str, float]:
    """Approximate one circular winding as the r-z rectangle expected by FEMM."""
    geometry = obj.get("geometry", {})
    mean_radius = max(0.0, 0.5 * float(geometry.get("mean_diameter_mm", 0.0)))
    construction = obj.get("construction")
    if isinstance(construction, dict):
        for component in construction.get("components", []):
            if (
                str(component.get("kind", "")) == "annular"
                and str(component.get("name", "")).strip().lower() == "winding pack"
            ):
                inner = max(0.0, float(component.get("inner_radius_mm", mean_radius)))
                outer = max(inner, float(component.get("outer_radius_mm", mean_radius)))
                half = max(0.0, float(component.get("half_thickness_mm", 0.0)))
                if outer > inner and half > 0.0:
                    return {
                        "inner_radius_mm": inner,
                        "outer_radius_mm": outer,
                        "axial_width_mm": 2.0 * half,
                    }
    winding = obj.get("winding", {})
    wire = max(0.05, float(winding.get("outer_wire_diameter_mm", 1.0) or 1.0))
    radial = float(winding.get("radial_build_mm", 0.0) or 0.0)
    axial = float(winding.get("axial_width_mm", 0.0) or 0.0)
    if radial <= 0.0:
        radial = wire
    if axial <= 0.0:
        axial = wire
    inner = max(1e-6, mean_radius - radial / 2.0)
    outer = max(inner + 1e-6, mean_radius + radial / 2.0)
    return {
        "inner_radius_mm": inner,
        "outer_radius_mm": outer,
        "axial_width_mm": max(axial, 1e-6),
    }


def _lua_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_femm(
    scene: dict[str, Any],
    *,
    model_stem: str = "field_workbench_export",
    apply_field_scale: bool = False,
    reference_slice_plane: str = "auto",
    reference_slice_offset_mm: float | None = None,
    reference_half: str = "positive",
) -> str:
    """Translate a coaxial circular-coil selection into a FEMM axisymmetric builder."""
    if scene.get("format") != PORTABLE_SCENE_FORMAT:
        raise StudioOperationError("FEMM translator expected a Field Workbench portable scene.")

    alignment = _femm_alignment(scene)
    coils = list(alignment["coils"])
    axis = np.asarray(alignment["axis_world"], dtype=float)
    origin = np.asarray(alignment["origin_world_mm"], dtype=float)
    axial_positions = list(alignment["axial_positions_mm"])
    direction_signs = list(alignment["direction_signs"])
    has_reference_geometry = any(
        str(obj.get("kind", "")) == "geometry" for obj in scene.get("objects", [])
    )
    slice_basis, reference_sections = _femm_reference_sections(
        scene,
        alignment,
        reference_slice_plane if has_reference_geometry else "auto",
        reference_slice_offset_mm if has_reference_geometry else None,
        reference_half if has_reference_geometry else "positive",
    )

    rows: list[dict[str, Any]] = []
    for index, (obj, z_mm, direction_sign) in enumerate(
        zip(coils, axial_positions, direction_signs, strict=True),
        start=1,
    ):
        section = _femm_coil_section(obj)
        excitation = obj.get("excitation", {})
        current = float(excitation.get("drive_current_a", 0.0))
        scale = float(excitation.get("field_scale_factor", 1.0))
        if apply_field_scale:
            current *= scale
        current *= float(direction_sign)

        matrix = np.asarray(
            obj.get("transform", {}).get("matrix4x4_mm", np.eye(4).tolist()), dtype=float
        )
        source_position = np.asarray(matrix[:3, 3], dtype=float)
        rows.append(
            {
                "index": index,
                "label": str(obj.get("label") or obj.get("id") or f"Coil {index}"),
                "inner": float(section["inner_radius_mm"]),
                "outer": float(section["outer_radius_mm"]),
                "z": float(z_mm),
                "width": float(section["axial_width_mm"]),
                "turns": int(excitation.get("turns", 1)),
                "current": current,
                "raw_current": float(excitation.get("drive_current_a", 0.0)),
                "field_scale": scale,
                "current_mode": str(excitation.get("current_mode", "rms_dc")),
                "direction_sign": float(direction_sign),
                "source_position_mm": source_position.tolist(),
            }
        )

    z_min = min(row["z"] - row["width"] / 2.0 for row in rows)
    z_max = max(row["z"] + row["width"] / 2.0 for row in rows)
    reference_points = [
        point
        for section in reference_sections
        for segment in section["segments_mm"]
        for point in segment
    ]
    if reference_points:
        z_min = min(z_min, min(float(point[1]) for point in reference_points))
        z_max = max(z_max, max(float(point[1]) for point in reference_points))
    z_mid = 0.5 * (z_min + z_max)
    radial_max = max(row["outer"] for row in rows)
    if reference_points:
        radial_max = max(radial_max, max(float(point[0]) for point in reference_points))
    extent = max(radial_max, z_max - z_mid, z_mid - z_min, 1.0)
    boundary_radius = max(50.0, 4.0 * extent)
    air_r = 0.5 * boundary_radius
    safe_stem = Path(str(model_stem)).stem or "field_workbench_export"

    axis_text = ", ".join(_scad_number(value) for value in axis)
    origin_text = ", ".join(_scad_number(value) for value in origin)
    radial_text = ", ".join(_scad_number(value) for value in slice_basis["radial_world"])
    normal_text = ", ".join(_scad_number(value) for value in slice_basis["normal_world"])
    slice_plane = str(slice_basis["plane"]).upper()
    requested_slice_plane = str(reference_slice_plane or "auto").upper()
    reference_half_name = str(slice_basis.get("reference_half", "positive"))
    half_label = _femm_direction_label(np.asarray(slice_basis["radial_world"], dtype=float))
    slice_mode = "exact source plane"
    if slice_basis.get("auto_selected"):
        slice_mode = f"auto-selected {slice_mode}"
    slice_offset_mm = float(slice_basis.get("offset_mm", 0.0))
    lines = [
        "-- Field Workbench FEMM 4.2 axisymmetric export",
        f"-- Portable scene schema v{PORTABLE_SCENE_VERSION}; generated by Field Workbench {__version__}",
        "-- Units: millimetres; magnetostatic (0 Hz).",
        "-- IMPORTANT: FEMM is 2D/axisymmetric. Workbench automatically rigidly normalizes",
        "-- the selected coaxial circular-coil assembly onto FEMM's canonical Z axis.",
        f"-- Source common-axis direction (world): [{axis_text}]",
        f"-- Source common-axis origin (world mm): [{origin_text}]",
        (
            f"-- Reference slice request: {requested_slice_plane}; using {slice_plane} ({slice_mode})."
            if has_reference_geometry
            else f"-- Reference slice request: {requested_slice_plane}; unused because no passive reference geometry was selected."
        ),
        f"-- FEMM +r direction in source world coordinates: [{radial_text}]",
        f"-- Reference slice plane normal in source world coordinates: [{normal_text}]",
        f"-- Reference slice offset in source-world plane coordinates: {_scad_number(slice_offset_mm)} mm (automatically locked to common axis)",
        f"-- Reference half: {reference_half_name}; retained source side for FEMM +r: {half_label}",
        "-- Passive reference outlines use the literal source-world slice; only a rigid 2-D",
        "-- rotation/translation is used for display in FEMM, so their cross-section shape is preserved.",
        "-- Only relative axial separation, coil radius/build, turns, current and polarity",
        "-- enter the FEMM model; the Workbench scene itself is not rotated or modified.",
        "-- Each winding is represented by an axisymmetric rectangular bulk region with",
        "-- the saved turn count and drive current. Current sign preserves each coil's",
        "-- orientation relative to the selected common source axis.",
        "-- Selected passive geometry may be added as open, reference-only guide segments.",
        "-- The triangle-mesh slice is deduplicated/stitched before export and closed contours",
        f"-- receive a deliberate {_scad_number(FEMM_REFERENCE_GAP_MM)} mm break. Guides are also held",
        f"-- {_scad_number(FEMM_REFERENCE_AXIS_CLEARANCE_MM)} mm off r=0 so an open half-contour cannot close against FEMM's symmetry axis.",
        "-- A final combined-geometry sanity pass removes duplicates and breaks any remaining",
        "-- endpoint-graph cycles. Guides are added only after the FEMM physics geometry and",
        "-- open-boundary construction are complete, so they cannot interfere with setup.",
        "-- The main Air label is also attached as FEMM's default block label. This is a final",
        "-- FEMM-native safety net for regions created only after guide segments intersect one",
        "-- another or existing model boundaries between endpoints; such regions remain Air.",
        "-- Guides carry no material, circuit, or boundary property, but remain FEMM input segments",
        "-- and can therefore influence triangulation if you choose to solve with them present.",
        "-- Empirical Field correction is {}applied to excitation.".format("" if apply_field_scale else "NOT "),
        "-- Run this file from FEMM: File -> Open Lua Script.",
        "",
        "newdocument(0)",
        'mi_probdef(0, "millimeters", "axi", 1e-8, 0, 30)',
        'mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)',
        'mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)',
        "",
    ]
    for row in rows:
        circuit = f"FW_coil_{row['index']}"
        z0 = row["z"] - row["width"] / 2.0
        z1 = row["z"] + row["width"] / 2.0
        mesh = max(0.02, min(row["outer"] - row["inner"], row["width"]) / 4.0)
        source_position = ", ".join(_scad_number(value) for value in row["source_position_mm"])
        lines.extend(
            [
                f"-- {row['label']}",
                f"-- source centre world mm=[{source_position}]; normalized FEMM z={_scad_number(row['z'])} mm",
                "-- raw current={} A; turns={}; mode={}; field_scale_factor={}; axis_direction_sign={}".format(
                    _scad_number(row["raw_current"]),
                    row["turns"],
                    row["current_mode"],
                    _scad_number(row["field_scale"]),
                    _scad_number(row["direction_sign"]),
                ),
                f"mi_addcircprop({_lua_string(circuit)}, {_scad_number(row['current'])}, 1)",
                "mi_drawrectangle({}, {}, {}, {})".format(
                    _scad_number(row["inner"]),
                    _scad_number(z0),
                    _scad_number(row["outer"]),
                    _scad_number(z1),
                ),
                "mi_addblocklabel({}, {})".format(
                    _scad_number(0.5 * (row["inner"] + row["outer"])),
                    _scad_number(row["z"]),
                ),
                "mi_selectlabel({}, {})".format(
                    _scad_number(0.5 * (row["inner"] + row["outer"])),
                    _scad_number(row["z"]),
                ),
                "mi_setblockprop(\"Copper\", 0, {}, {}, 0, {}, {})".format(
                    _scad_number(mesh),
                    _lua_string(circuit),
                    row["index"],
                    row["turns"],
                ),
                "mi_clearselected()",
                "",
            ]
        )
    lines.extend(
        [
            "-- Improvised asymptotic boundary condition for open space.",
            "mi_makeABC(7, {}, 0, {}, 0)".format(
                _scad_number(boundary_radius), _scad_number(z_mid)
            ),
            "mi_addblocklabel({}, {})".format(_scad_number(air_r), _scad_number(z_mid)),
            "mi_selectlabel({}, {})".format(_scad_number(air_r), _scad_number(z_mid)),
            "mi_setblockprop(\"Air\", 0, {}, \"<None>\", 0, 0, 0)".format(
                _scad_number(boundary_radius / 20.0)
            ),
            "mi_attachdefault()",
            "mi_clearselected()",
            "",
        ]
    )

    # Reference geometry is deliberately appended after the real FEMM model is
    # complete.  In particular, mi_makeABC never has to inspect imported STL
    # tessellation and the guide drawing never participates in block labeling.
    if reference_sections:
        reference_count = sum(len(section["segments_mm"]) for section in reference_sections)
        raw_reference_count = sum(int(section.get("raw_segment_count", 0)) for section in reference_sections)
        reference_sanity = dict(slice_basis.get("reference_sanity", {}))
        lines.extend(
            [
                "-- Reference-only scene cross-sections (added after physics setup).",
                f"-- Requested source plane: {requested_slice_plane}; using {slice_plane} at auto axis offset {_scad_number(slice_offset_mm)} mm; retained half {half_label}.",
                f"-- FEMM-safe contour cleanup: {raw_reference_count} raw triangle cuts -> {reference_count} emitted guide segments.",
                "-- Reference opening: gap={} mm; r=0 clearance={} mm; final sanity breaks={}; removed duplicate edges={}; residual cycles={}.".format(
                    _scad_number(FEMM_REFERENCE_GAP_MM),
                    _scad_number(FEMM_REFERENCE_AXIS_CLEARANCE_MM),
                    int(reference_sanity.get("sanity_break_count", 0)),
                    int(reference_sanity.get("global_duplicate_count", 0)),
                    int(reference_sanity.get("residual_cycle_count", 0)),
                ),
                "-- No FEMM selection/grouping commands are used for guides; this intentionally avoids",
                "-- the geometry-selection path that can become non-responsive on dense imported meshes.",
                "",
            ]
        )
        for section in reference_sections:
            lines.append(
                "-- Reference outline: {} [{}]; raw={} chains={} emitted={}".format(
                    section["label"],
                    section["id"],
                    int(section.get("raw_segment_count", len(section["segments_mm"]))),
                    int(section.get("chain_count", 0)),
                    len(section["segments_mm"]),
                )
            )
            for first, second in section["segments_mm"]:
                lines.append(
                    "mi_drawline({}, {}, {}, {})".format(
                        _scad_number(first[0]),
                        _scad_number(first[1]),
                        _scad_number(second[0]),
                        _scad_number(second[1]),
                    )
                )
            lines.append("")

    lines.extend(
        [
            "mi_zoomnatural()",
            f"mi_saveas({_lua_string(safe_stem + '.fem')})",
            "",
            "-- Uncomment these lines if you want the builder to solve immediately:",
            "-- mi_analyze()",
            "-- mi_loadsolution()",
            "",
        ]
    )
    return "\n".join(lines)



def _java_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        number = 0.0
    if abs(number) < 5e-13:
        number = 0.0
    return f"{number:.12g}"


def _java_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _java_double_array(values: Any) -> str:
    return "new double[]{" + ", ".join(_java_number(value) for value in values) + "}"


def _java_point_matrix(points: np.ndarray) -> str:
    if points.ndim != 2 or points.shape[1:] != (3,):
        return "new double[][]{{},{},{}}"
    # BezierPolygon expects one row per coordinate dimension.
    rows = []
    for axis in range(3):
        rows.append("{" + ", ".join(_java_number(value) for value in points[:, axis]) + "}")
    return "new double[][]{" + ", ".join(rows) + "}"


def render_comsol_java(
    scene: dict[str, Any],
    *,
    apply_field_scale: bool = False,
) -> str:
    """Create an experimental COMSOL 6.4 Java builder using ideal edge currents."""
    if scene.get("format") != PORTABLE_SCENE_FORMAT:
        raise StudioOperationError("COMSOL translator expected a Field Workbench portable scene.")
    coils = [obj for obj in scene.get("objects", []) if str(obj.get("kind", "")) == "coil"]
    if not coils:
        raise StudioOperationError("COMSOL export requires at least one coil.")

    entries: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    for index, obj in enumerate(coils, start=1):
        points = _world_points_for_coil(obj, circle_segments=96)
        if len(points) < 4:
            raise StudioOperationError(
                f"{obj.get('label', 'Coil')} does not have a usable closed centreline for COMSOL export."
            )
        # BezierPolygon type=closed adds the final closure segment itself; avoid a duplicate point.
        if np.allclose(points[0], points[-1]):
            points = points[:-1]
        excitation = obj.get("excitation", {})
        drive = float(excitation.get("drive_current_a", 0.0))
        scale = float(excitation.get("field_scale_factor", 1.0))
        effective = drive * int(excitation.get("turns", 1))
        if apply_field_scale:
            effective *= scale
        entries.append(
            {
                "index": index,
                "tag": f"coil{index}",
                "selection": f"geom1_coil{index}_edg",
                "label": str(obj.get("label") or obj.get("id") or f"Coil {index}"),
                "points": points,
                "effective_current": effective,
                "drive_current": drive,
                "turns": int(excitation.get("turns", 1)),
                "field_scale": scale,
                "current_mode": str(excitation.get("current_mode", "rms_dc")),
            }
        )
        all_points.append(points)

    cloud = np.vstack(all_points)
    mins = np.min(cloud, axis=0)
    maxs = np.max(cloud, axis=0)
    center = 0.5 * (mins + maxs)
    half_diag = 0.5 * float(np.linalg.norm(maxs - mins))
    max_extent = float(np.max(np.linalg.norm(cloud - center, axis=1)))
    radius = max(50.0, 4.0 * max(max_extent, half_diag, 1.0))

    lines = [
        "// Field Workbench COMSOL Multiphysics 6.4 Java API export",
        f"// Portable scene schema v{PORTABLE_SCENE_VERSION}; generated by Field Workbench {__version__}",
        "// Requires COMSOL Multiphysics with the AC/DC Module.",
        "// This first-pass translator uses 3D ideal Edge Current loops, not volumetric winding packs.",
        "// Effective edge current = turns * drive current{}.".format(
            " * empirical field scale" if apply_field_scale else " (empirical field scale is intentionally not applied)"
        ),
        "// The surrounding sphere is a finite free-space truncation. Inspect/refine its size and mesh",
        "// before treating the result as a validated COMSOL model.",
        "",
        "import com.comsol.model.*;",
        "import com.comsol.model.util.*;",
        "",
        "class FieldWorkbenchModel {",
        "  public static Model run() {",
        '    Model model = ModelUtil.create("Model");',
        '    model.label("Field Workbench magnetic export");',
        '    model.component().create("comp1");',
        '    model.component("comp1").geom().create("geom1", 3);',
        '    model.component("comp1").geom("geom1").lengthUnit("mm");',
        "",
        "    // Free-space solution domain.",
        '    model.component("comp1").geom("geom1").create("air", "Sphere");',
        "    model.component(\"comp1\").geom(\"geom1\").feature(\"air\").set(\"r\", {});".format(
            _java_number(radius)
        ),
        "    model.component(\"comp1\").geom(\"geom1\").feature(\"air\").set(\"pos\", {});".format(
            _java_double_array(center)
        ),
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"    // {entry['label']}",
                "    // drive_current={} A; turns={}; mode={}; field_scale_factor={}".format(
                    _java_number(entry["drive_current"]),
                    entry["turns"],
                    entry["current_mode"],
                    _java_number(entry["field_scale"]),
                ),
                f'    model.component("comp1").geom("geom1").create("{entry["tag"]}", "BezierPolygon");',
                f'    model.component("comp1").geom("geom1").feature("{entry["tag"]}").set("type", "closed");',
                f'    model.component("comp1").geom("geom1").feature("{entry["tag"]}").set("degree", 1);',
                f'    model.component("comp1").geom("geom1").feature("{entry["tag"]}").set("p", {_java_point_matrix(entry["points"])});',
                f'    model.component("comp1").geom("geom1").feature("{entry["tag"]}").set("selresult", "on");',
                f'    model.component("comp1").geom("geom1").feature("{entry["tag"]}").set("selresultshow", "edg");',
                "",
            ]
        )
    lines.extend(
        [
            '    model.component("comp1").geom("geom1").run();',
            "",
            "    // Magnetic Fields interface. COMSOL 6.2+ creates Free Space as the default air-domain feature.",
            '    model.component("comp1").physics().create("mf", "InductionCurrents", "geom1");',
        ]
    )
    for entry in entries:
        param = f"Ieff{entry['index']}"
        feature = f"ec{entry['index']}"
        lines.extend(
            [
                f'    model.param().set("{param}", "{_java_number(entry["effective_current"])}[A]", {_java_string(entry["label"] + " effective ampere-turn current")});',
                f'    model.component("comp1").physics("mf").create("{feature}", "EdgeCurrent", 1);',
                f'    model.component("comp1").physics("mf").feature("{feature}").selection().named("{entry["selection"]}");',
                f'    model.component("comp1").physics("mf").feature("{feature}").set("I0", "{param}");',
                "",
            ]
        )
    lines.extend(
        [
            '    model.component("comp1").mesh().create("mesh1");',
            '    model.component("comp1").mesh("mesh1").autoMeshSize(5);',
            '    model.study().create("std1");',
            '    model.study("std1").create("stat", "Stationary");',
            "",
            "    // Deliberately do not solve automatically. Open the generated model in COMSOL, inspect",
            "    // coil edge directions, free-space extent, and mesh, then compute Study 1.",
            "    return model;",
            "  }",
            "",
            "  public static void main(String[] args) {",
            "    run();",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def write_portable_package(scene: dict[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    manifest = {
        "format": "field_workbench.portable_package",
        "version": 1,
        "scene": "scene.json",
        "scene_format": scene.get("format"),
        "scene_version": scene.get("version"),
        "generator": scene.get("generator", {}),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        archive.writestr("scene.json", json.dumps(scene, indent=2, ensure_ascii=False) + "\n")
    return path


def write_export(
    adapter: StudioAdapter,
    object_ids: list[str],
    translator: str,
    filename: str | Path,
    *,
    include_construction: bool = True,
    apply_field_scale: bool = False,
    getdp_use_physical_construction: bool = True,
    getdp_mesh_preset: str = "standard",
    getdp_source_elements: float | None = None,
    getdp_near_radius_divisions: float | None = None,
    getdp_far_radius_divisions: float | None = None,
    femm_reference_slice_plane: str = "auto",
    femm_reference_slice_offset_mm: float | None = None,
    femm_reference_half: str = "positive",
) -> Path:
    """Build the canonical portable scene, then run the requested translator."""
    allowed = [
        object_id
        for object_id in object_ids
        if export_capability(adapter, object_id, translator)[0]
    ]
    if not allowed:
        raise StudioOperationError("Select at least one object supported by the chosen translator.")
    scene = build_portable_scene(adapter, allowed)
    path = Path(filename)
    if translator == TRANSLATOR_PORTABLE:
        return write_portable_package(scene, path)
    if translator == TRANSLATOR_OPENSCAD:
        path.write_text(
            render_openscad(scene, include_construction=include_construction),
            encoding="utf-8",
        )
        return path
    if translator == TRANSLATOR_FEMM:
        path.write_text(
            render_femm(
                scene,
                model_stem=path.stem,
                apply_field_scale=apply_field_scale,
                reference_slice_plane=femm_reference_slice_plane,
                reference_slice_offset_mm=femm_reference_slice_offset_mm,
                reference_half=femm_reference_half,
            ),
            encoding="utf-8",
        )
        return path
    if translator == TRANSLATOR_GETDP:
        valid, status = getdp_selection_status(
            adapter,
            allowed,
            use_physical_construction=getdp_use_physical_construction,
        )
        if not valid:
            raise StudioOperationError(status)
        return write_getdp_project(
            scene,
            path,
            apply_field_scale=apply_field_scale,
            use_physical_construction=getdp_use_physical_construction,
            mesh_preset=getdp_mesh_preset,
            source_elements=getdp_source_elements,
            near_radius_divisions=getdp_near_radius_divisions,
            far_radius_divisions=getdp_far_radius_divisions,
        )
    if translator == TRANSLATOR_COMSOL:
        path.write_text(
            render_comsol_java(scene, apply_field_scale=apply_field_scale),
            encoding="utf-8",
        )
        return path
    raise StudioOperationError(f"Unknown scene-export translator: {translator!r}.")
