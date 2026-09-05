"""Generate self-contained Gmsh/GetDP magnetostatic projects.

The exporter consumes the normalized ``field_workbench.portable_scene`` schema
rather than editor state. Version 10 supports general multi-source selections of
circular, Square/rectangular, and/or parametric racetrack coils. Each Workbench source becomes a
homogenized stranded-current winding region whose cross-section carries exactly
the selected signed ampere-turns.

Generated projects use conservative syntax and ASCII MSH 2.2 so they remain
compatible with the user's Gmsh 4.8.4 and GetDP 3.2.0 installations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

from .studio_adapter import StudioOperationError


GETDP_PROJECT_FORMAT = "field_workbench.getdp_project"
GETDP_PROJECT_VERSION = 10
GETDP_PROJECT_SUFFIX = ".getdp.zip"
GETDP_MIN_GMSH_VERSION = "4.8.4"
GETDP_MIN_SOLVER_VERSION = "3.2.0"
GETDP_MAX_SAMPLE_POINTS = 5000
GETDP_MAX_COIL_SOURCES = 64

# User-facing mesh presets tune only local discretization.  Open-space treatment
# is no longer a mesh-quality knob: every preset uses the same automatically sized
# ordinary-air sphere plus a finite spherical shell that GetDP maps to infinity
# with VolSphShell.  Gmsh characteristic lengths are targets rather than guaranteed
# element counts, hence the deliberately approximate "divisions" language in the
# UI and project metadata.
GETDP_MESH_PRESETS: dict[str, dict[str, float | str]] = {
    "fast": {
        "label": "Fast",
        "source_elements": 1.5,
        "near_radius_divisions": 10.0,
        "far_radius_divisions": 6.0,
    },
    "standard": {
        "label": "Standard",
        # Keep the normal engineering preset practical for 16 GB-class machines
        # with the default direct MUMPS solve.  Sparse direct factor storage grows
        # much faster than raw mesh/DOF count, so modest local coarsening here has
        # a disproportionately useful effect on peak memory.
        "source_elements": 1.8,
        "near_radius_divisions": 14.0,
        "far_radius_divisions": 8.0,
    },
    "validation": {
        "label": "Validation",
        "source_elements": 3.0,
        "near_radius_divisions": 32.0,
        "far_radius_divisions": 12.0,
    },
}


@dataclass(frozen=True)
class GetDPMeshSettings:
    """Scale-independent controls used to derive a concrete Gmsh mesh plan."""

    preset: str
    source_elements: float
    near_radius_divisions: float
    far_radius_divisions: float


@dataclass(frozen=True)
class _MeshPlan:
    air_centre_m: np.ndarray
    inner_air_radius_m: float
    shell_outer_radius_m: float
    shell_mesh_m: float
    source_mesh_m: float
    near_mesh_m: float
    far_mesh_m: float
    source_grading_enabled: bool
    source_grading_air_min_m: float
    source_grading_start_m: float
    source_grading_end_m: float
    local_min_m: np.ndarray
    local_max_m: np.ndarray

_MU0 = 4.0e-7 * math.pi
_AIR_PHYSICAL_TAG = 1000
_INFINITE_AIR_PHYSICAL_TAG = 1001
_COIL_PHYSICAL_TAG_BASE = 1101
_OUTER_BOUNDARY_PHYSICAL_TAG = 2000
_AIR_ENTITY_TAG = 1
_INFINITE_SHELL_OUTER_ENTITY_TAG = 2
# Circular primitives live above the racetrack OCC block range (100..6499 for
# the supported 64 sources), preventing mixed multi-coil scenes from reusing
# an entity tag when source indices grow.
_COIL_ENTITY_TAG_BASE = 10000
_COIL_ENTITY_STRIDE = 10
_SQUARE_ENTITY_TAG_BASE = 7000
_SQUARE_ENTITY_STRIDE = 20


@dataclass(frozen=True)
class _CoilSource:
    object_id: str
    label: str
    shape: str
    magnetic_centre_m: np.ndarray
    source_centre_m: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    normal: np.ndarray
    axial_thickness_m: float
    radial_build_m: float
    cross_section_area_m2: float
    characteristic_radius_m: float
    planar_extent_m: float
    turns: int
    drive_current_a: float
    field_scale_factor: float
    ampere_turns_a: float
    current_density_a_m2: float
    current_mode: str
    enabled: bool
    source_geometry_kind: str


@dataclass(frozen=True)
class _CircularSource(_CoilSource):
    mean_radius_m: float
    inner_radius_m: float
    outer_radius_m: float


@dataclass(frozen=True)
class _RacetrackSource(_CoilSource):
    end_radius_m: float
    half_straight_m: float
    inner_offset_m: float
    outer_offset_m: float
    inner_end_radius_m: float
    outer_end_radius_m: float
    centerline_vertices_local_m: np.ndarray


@dataclass(frozen=True)
class _SquareSource(_CoilSource):
    half_length_m: float
    half_width_m: float
    inner_offset_m: float
    outer_offset_m: float
    inner_half_length_m: float
    inner_half_width_m: float
    outer_half_length_m: float
    outer_half_width_m: float
    path_orientation_sign: float
    centerline_vertices_local_m: np.ndarray


_Source = _CircularSource | _RacetrackSource | _SquareSource


@dataclass(frozen=True)
class _SamplePoint:
    label: str
    point_m: np.ndarray
    reference_b_ut: np.ndarray | None
    reference_kind: str


def getdp_mesh_settings(
    preset: str = "standard",
    *,
    source_elements: float | None = None,
    near_radius_divisions: float | None = None,
    far_radius_divisions: float | None = None,
) -> GetDPMeshSettings:
    """Return validated mesh controls for a preset or custom configuration."""
    key = str(preset or "standard").strip().casefold()
    if key == "custom":
        base = GETDP_MESH_PRESETS["standard"]
    elif key in GETDP_MESH_PRESETS:
        base = GETDP_MESH_PRESETS[key]
    else:
        raise StudioOperationError(f"Unknown GetDP mesh preset: {preset!r}.")

    def value(name: str, override: float | None, minimum: float, maximum: float) -> float:
        result = float(base[name] if override is None else override)
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise StudioOperationError(
                f"GetDP {name.replace('_', ' ')} must be between {minimum:g} and {maximum:g}."
            )
        return result

    return GetDPMeshSettings(
        preset=key,
        source_elements=value("source_elements", source_elements, 1.0, 12.0),
        near_radius_divisions=value(
            "near_radius_divisions", near_radius_divisions, 4.0, 128.0
        ),
        far_radius_divisions=value(
            "far_radius_divisions", far_radius_divisions, 4.0, 128.0
        ),
    )


def _number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise StudioOperationError("GetDP export encountered a non-finite number.")
    if abs(number) < 5e-16:
        number = 0.0
    return f"{number:.12g}"


def _project_stem(value: str) -> str:
    stem = str(value).strip()
    if stem.lower().endswith(GETDP_PROJECT_SUFFIX):
        stem = stem[: -len(GETDP_PROJECT_SUFFIX)]
    else:
        stem = Path(stem).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return stem[:96] or "field_workbench_model"


def _normalised_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise StudioOperationError(f"GetDP export could not resolve {label}.")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-12:
        raise StudioOperationError(f"GetDP export could not resolve {label}.")
    return vector / magnitude


def _coil_transform(coil: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = coil.get("transform", {})
    matrix = np.asarray(transform.get("matrix4x4_mm", []), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise StudioOperationError(
            f"{coil.get('label', 'Coil')} does not have a usable world transform."
        )
    centre_m = matrix[:3, 3] / 1000.0
    basis_x = _normalised_vector(matrix[:3, 0], "the coil local X axis")
    raw_y = np.asarray(matrix[:3, 1], dtype=float)
    raw_y = raw_y - float(np.dot(raw_y, basis_x)) * basis_x
    basis_y = _normalised_vector(raw_y, "the coil local Y axis")
    normal = _normalised_vector(np.cross(basis_x, basis_y), "the coil normal")
    saved_normal = _normalised_vector(matrix[:3, 2], "the saved coil normal")
    if float(np.dot(normal, saved_normal)) < 0.0:
        basis_y = -basis_y
        normal = -normal
    return centre_m, basis_x, basis_y, normal


def _saved_winding_pack(coil: dict[str, Any]) -> dict[str, float] | None:
    construction = coil.get("construction")
    if not isinstance(construction, dict) or not construction.get("enabled", False):
        return None
    if not construction.get("supported", False) or not construction.get("valid", False):
        details = [str(value) for value in construction.get("errors", []) if str(value)]
        suffix = f" ({'; '.join(details)})" if details else ""
        raise StudioOperationError(
            f"{coil.get('label', 'Coil')} has enabled but invalid physical construction{suffix}."
        )
    for component in construction.get("components", []):
        if (
            isinstance(component, dict)
            and str(component.get("kind", "")) == "annular"
            and str(component.get("name", "")).strip().casefold() == "winding pack"
        ):
            return {
                "inner_radius_mm": float(component.get("inner_radius_mm", 0.0)),
                "outer_radius_mm": float(component.get("outer_radius_mm", 0.0)),
                "axial_thickness_mm": 2.0 * float(component.get("half_thickness_mm", 0.0)),
                "centre_z_mm": float(component.get("centre_z_mm", 0.0)),
            }
    raise StudioOperationError(
        f"{coil.get('label', 'Coil')} has enabled physical construction but no "
        "valid winding-pack solid."
    )


def _estimated_numerical_pack_area_mm2(coil: dict[str, Any]) -> float:
    excitation = coil.get("excitation", {})
    winding = coil.get("winding", {})
    conductor = coil.get("conductor", {})
    turns = max(1, int(excitation.get("turns", 1)))
    wire_diameter_mm = float(winding.get("outer_wire_diameter_mm", 0.0))
    packing_factor = float(conductor.get("packing_factor", 0.75))
    if not math.isfinite(wire_diameter_mm) or wire_diameter_mm <= 0.0:
        raise StudioOperationError(
            f"{coil.get('label', 'Coil')} needs a positive finished-wire diameter "
            "for its numerical winding pack."
        )
    if not math.isfinite(packing_factor) or not 0.0 < packing_factor <= 1.0:
        raise StudioOperationError(
            f"{coil.get('label', 'Coil')} needs a packing factor between zero and one."
        )
    return (
        turns * math.pi * (wire_diameter_mm / 2.0) ** 2 / packing_factor
    )


def _fallback_winding_pack(coil: dict[str, Any], mean_radius_mm: float) -> dict[str, float]:
    estimated_area_mm2 = _estimated_numerical_pack_area_mm2(coil)
    # A half-millimetre floor avoids pathological one-turn mesh scales.  The
    # current density is always normalized against the actual surrogate area,
    # so this affects only the finite source approximation, never total NI.
    nominal_side_mm = max(0.5, math.sqrt(max(estimated_area_mm2, 1e-12)))
    radial_build_mm = min(nominal_side_mm, max(0.05, mean_radius_mm))
    axial_thickness_mm = max(0.5, estimated_area_mm2 / radial_build_mm)
    inner_radius_mm = mean_radius_mm - radial_build_mm / 2.0
    outer_radius_mm = mean_radius_mm + radial_build_mm / 2.0
    return {
        "inner_radius_mm": inner_radius_mm,
        "outer_radius_mm": outer_radius_mm,
        "axial_thickness_mm": axial_thickness_mm,
        "centre_z_mm": 0.0,
    }


def _saved_path_winding_pack(coil: dict[str, Any]) -> dict[str, Any] | None:
    construction = coil.get("construction")
    if not isinstance(construction, dict) or not construction.get("enabled", False):
        return None
    if not construction.get("supported", False) or not construction.get("valid", False):
        details = [str(value) for value in construction.get("errors", []) if str(value)]
        suffix = f" ({'; '.join(details)})" if details else ""
        raise StudioOperationError(
            f"{coil.get('label', 'Coil')} has enabled but invalid physical construction{suffix}."
        )
    for component in construction.get("components", []):
        if (
            isinstance(component, dict)
            and str(component.get("kind", "")) == "path_band"
            and str(component.get("name", "")).strip().casefold() == "winding pack"
        ):
            path_xy_mm = np.asarray(component.get("path_xy_mm", []), dtype=float)
            if path_xy_mm.ndim != 2 or path_xy_mm.shape[1:] != (2,):
                raise StudioOperationError(
                    f"{coil.get('label', 'Coil')} has an invalid saved winding-pack path."
                )
            return {
                "path_xy_mm": path_xy_mm,
                "inner_offset_mm": float(component.get("inner_offset_mm", 0.0)),
                "outer_offset_mm": float(component.get("outer_offset_mm", 0.0)),
                "axial_thickness_mm": 2.0 * float(component.get("half_thickness_mm", 0.0)),
                "centre_z_mm": float(component.get("centre_z_mm", 0.0)),
            }
    raise StudioOperationError(
        f"{coil.get('label', 'Coil')} has enabled physical construction but no "
        "valid winding-pack path band."
    )


def _fallback_racetrack_winding_pack(
    coil: dict[str, Any], *, end_radius_mm: float
) -> dict[str, float]:
    estimated_area_mm2 = _estimated_numerical_pack_area_mm2(coil)
    nominal_side_mm = max(0.5, math.sqrt(max(estimated_area_mm2, 1e-12)))
    # Keep the inner parallel racetrack well-defined. As with the circular
    # fallback, the exact surrogate dimensions do not alter total NI.
    radial_build_mm = min(nominal_side_mm, max(0.05, end_radius_mm))
    axial_thickness_mm = max(0.5, estimated_area_mm2 / radial_build_mm)
    return {
        "inner_offset_mm": -radial_build_mm / 2.0,
        "outer_offset_mm": radial_build_mm / 2.0,
        "axial_thickness_mm": axial_thickness_mm,
        "centre_z_mm": 0.0,
    }


def _fallback_square_winding_pack(
    coil: dict[str, Any], *, minimum_half_span_mm: float
) -> dict[str, float]:
    estimated_area_mm2 = _estimated_numerical_pack_area_mm2(coil)
    nominal_side_mm = max(0.5, math.sqrt(max(estimated_area_mm2, 1e-12)))
    radial_build_mm = min(nominal_side_mm, max(0.05, minimum_half_span_mm))
    axial_thickness_mm = max(0.5, estimated_area_mm2 / radial_build_mm)
    return {
        "inner_offset_mm": -radial_build_mm / 2.0,
        "outer_offset_mm": radial_build_mm / 2.0,
        "axial_thickness_mm": axial_thickness_mm,
        "centre_z_mm": 0.0,
    }


def _circular_source(
    coil: dict[str, Any], *, apply_field_scale: bool, use_physical_construction: bool
) -> _CircularSource:
    if str(coil.get("kind", "")) != "coil" or str(
        coil.get("geometry", {}).get("shape", "")
    ) != "circle":
        raise StudioOperationError("GetDP circular-source translation requires a circular coil.")

    label = str(coil.get("label") or coil.get("id") or "Coil")
    diameter_mm = float(coil.get("geometry", {}).get("mean_diameter_mm", 0.0))
    if not math.isfinite(diameter_mm) or diameter_mm <= 0.0:
        raise StudioOperationError(f"{label} needs a positive mean diameter for GetDP export.")
    mean_radius_mm = diameter_mm / 2.0
    magnetic_centre_m, basis_x, basis_y, normal = _coil_transform(coil)

    pack = _saved_winding_pack(coil) if use_physical_construction else None
    if pack is None:
        pack = _fallback_winding_pack(coil, mean_radius_mm)
        source_kind = (
            "derived numerical winding pack"
            if use_physical_construction
            else "forced numerical winding pack"
        )
    else:
        source_kind = "saved physical winding pack"

    inner_radius_m = float(pack["inner_radius_mm"]) / 1000.0
    outer_radius_m = float(pack["outer_radius_mm"]) / 1000.0
    axial_thickness_m = float(pack["axial_thickness_mm"]) / 1000.0
    centre_z_m = float(pack["centre_z_mm"]) / 1000.0
    if (
        not all(
            math.isfinite(value)
            for value in (inner_radius_m, outer_radius_m, axial_thickness_m, centre_z_m)
        )
        or inner_radius_m <= 0.0
        or outer_radius_m <= inner_radius_m
        or axial_thickness_m <= 0.0
    ):
        raise StudioOperationError(f"{label} does not produce a valid annular winding region.")
    source_centre_m = magnetic_centre_m + centre_z_m * normal
    cross_section_area_m2 = (outer_radius_m - inner_radius_m) * axial_thickness_m

    excitation = coil.get("excitation", {})
    turns = int(excitation.get("turns", 1))
    drive_current_a = float(excitation.get("drive_current_a", 0.0))
    field_scale_factor = float(excitation.get("field_scale_factor", 1.0))
    enabled = bool(excitation.get("enabled", True))
    if turns < 1 or not math.isfinite(drive_current_a):
        raise StudioOperationError(f"{label} has invalid turns or drive current.")
    if not math.isfinite(field_scale_factor) or field_scale_factor <= 0.0:
        raise StudioOperationError(f"{label} has an invalid empirical Field correction.")
    applied_scale = field_scale_factor if apply_field_scale else 1.0
    ampere_turns_a = turns * drive_current_a * applied_scale if enabled else 0.0
    current_density_a_m2 = ampere_turns_a / cross_section_area_m2

    return _CircularSource(
        object_id=str(coil.get("id") or "coil"),
        label=label,
        shape="circle",
        magnetic_centre_m=magnetic_centre_m,
        source_centre_m=source_centre_m,
        basis_x=basis_x,
        basis_y=basis_y,
        normal=normal,
        axial_thickness_m=axial_thickness_m,
        radial_build_m=outer_radius_m - inner_radius_m,
        cross_section_area_m2=cross_section_area_m2,
        characteristic_radius_m=mean_radius_mm / 1000.0,
        planar_extent_m=outer_radius_m,
        turns=turns,
        drive_current_a=drive_current_a,
        field_scale_factor=field_scale_factor,
        ampere_turns_a=ampere_turns_a,
        current_density_a_m2=current_density_a_m2,
        current_mode=str(excitation.get("current_mode", "rms_dc")),
        enabled=enabled,
        source_geometry_kind=source_kind,
        mean_radius_m=mean_radius_mm / 1000.0,
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer_radius_m,
    )


def _racetrack_source(
    coil: dict[str, Any], *, apply_field_scale: bool, use_physical_construction: bool
) -> _RacetrackSource:
    geometry = coil.get("geometry", {})
    if str(coil.get("kind", "")) != "coil" or str(geometry.get("shape", "")) != "racetrack":
        raise StudioOperationError("GetDP racetrack translation requires a parametric racetrack coil.")

    label = str(coil.get("label") or coil.get("id") or "Coil")
    end_diameter_mm = float(geometry.get("end_diameter_mm", 0.0))
    straight_length_mm = float(geometry.get("straight_length_mm", 0.0))
    if not math.isfinite(end_diameter_mm) or end_diameter_mm <= 0.0:
        raise StudioOperationError(f"{label} needs a positive racetrack end diameter for GetDP export.")
    if not math.isfinite(straight_length_mm) or straight_length_mm < 0.0:
        raise StudioOperationError(f"{label} needs a non-negative racetrack straight length for GetDP export.")
    end_radius_mm = end_diameter_mm / 2.0
    half_straight_m = straight_length_mm / 2000.0
    magnetic_centre_m, basis_x, basis_y, normal = _coil_transform(coil)

    vertices_mm = np.asarray(geometry.get("centerline_vertices_mm", []), dtype=float)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1:] != (3,) or len(vertices_mm) < 8:
        raise StudioOperationError(f"{label} does not contain a usable racetrack centreline.")
    if not np.all(np.isfinite(vertices_mm)):
        raise StudioOperationError(f"{label} contains non-finite racetrack centreline coordinates.")
    centerline_vertices_local_m = vertices_mm / 1000.0

    pack = _saved_path_winding_pack(coil) if use_physical_construction else None
    if pack is None:
        pack = _fallback_racetrack_winding_pack(coil, end_radius_mm=end_radius_mm)
        source_kind = (
            "derived numerical racetrack winding pack"
            if use_physical_construction
            else "forced numerical racetrack winding pack"
        )
    else:
        source_kind = "saved physical racetrack winding pack"

    inner_offset_m = float(pack["inner_offset_mm"]) / 1000.0
    outer_offset_m = float(pack["outer_offset_mm"]) / 1000.0
    axial_thickness_m = float(pack["axial_thickness_mm"]) / 1000.0
    centre_z_m = float(pack["centre_z_mm"]) / 1000.0
    end_radius_m = end_radius_mm / 1000.0
    inner_end_radius_m = end_radius_m + inner_offset_m
    outer_end_radius_m = end_radius_m + outer_offset_m
    radial_build_m = outer_offset_m - inner_offset_m
    if (
        not all(
            math.isfinite(value)
            for value in (
                inner_offset_m,
                outer_offset_m,
                axial_thickness_m,
                centre_z_m,
                inner_end_radius_m,
                outer_end_radius_m,
            )
        )
        or inner_end_radius_m <= 0.0
        or outer_end_radius_m <= inner_end_radius_m
        or radial_build_m <= 0.0
        or axial_thickness_m <= 0.0
    ):
        raise StudioOperationError(f"{label} does not produce a valid racetrack winding region.")

    # The portable construction path is retained for provenance/DRC alignment,
    # while the exact parametric end radius/straight length define the smooth
    # OCC capsule used by Gmsh. The two representations share the same magnetic
    # centreline in Field Workbench.
    if isinstance(pack.get("path_xy_mm"), np.ndarray):
        saved_path = np.asarray(pack["path_xy_mm"], dtype=float)
        if len(saved_path) >= 8:
            saved_radius = 0.5 * float(np.ptp(saved_path[:, 1]))
            saved_straight = max(0.0, float(np.ptp(saved_path[:, 0])) - 2.0 * saved_radius)
            tolerance_mm = max(1e-5, 1e-4 * max(end_diameter_mm, straight_length_mm, 1.0))
            if abs(saved_radius - end_radius_mm) > tolerance_mm or abs(saved_straight - straight_length_mm) > 5.0 * tolerance_mm:
                raise StudioOperationError(
                    f"{label}'s saved physical winding path no longer matches its racetrack dimensions."
                )

    source_centre_m = magnetic_centre_m + centre_z_m * normal
    cross_section_area_m2 = radial_build_m * axial_thickness_m
    excitation = coil.get("excitation", {})
    turns = int(excitation.get("turns", 1))
    drive_current_a = float(excitation.get("drive_current_a", 0.0))
    field_scale_factor = float(excitation.get("field_scale_factor", 1.0))
    enabled = bool(excitation.get("enabled", True))
    if turns < 1 or not math.isfinite(drive_current_a):
        raise StudioOperationError(f"{label} has invalid turns or drive current.")
    if not math.isfinite(field_scale_factor) or field_scale_factor <= 0.0:
        raise StudioOperationError(f"{label} has an invalid empirical Field correction.")
    applied_scale = field_scale_factor if apply_field_scale else 1.0
    ampere_turns_a = turns * drive_current_a * applied_scale if enabled else 0.0
    current_density_a_m2 = ampere_turns_a / cross_section_area_m2
    characteristic_radius_m = half_straight_m + end_radius_m
    planar_extent_m = half_straight_m + outer_end_radius_m

    return _RacetrackSource(
        object_id=str(coil.get("id") or "coil"),
        label=label,
        shape="racetrack",
        magnetic_centre_m=magnetic_centre_m,
        source_centre_m=source_centre_m,
        basis_x=basis_x,
        basis_y=basis_y,
        normal=normal,
        axial_thickness_m=axial_thickness_m,
        radial_build_m=radial_build_m,
        cross_section_area_m2=cross_section_area_m2,
        characteristic_radius_m=characteristic_radius_m,
        planar_extent_m=planar_extent_m,
        turns=turns,
        drive_current_a=drive_current_a,
        field_scale_factor=field_scale_factor,
        ampere_turns_a=ampere_turns_a,
        current_density_a_m2=current_density_a_m2,
        current_mode=str(excitation.get("current_mode", "rms_dc")),
        enabled=enabled,
        source_geometry_kind=source_kind,
        end_radius_m=end_radius_m,
        half_straight_m=half_straight_m,
        inner_offset_m=inner_offset_m,
        outer_offset_m=outer_offset_m,
        inner_end_radius_m=inner_end_radius_m,
        outer_end_radius_m=outer_end_radius_m,
        centerline_vertices_local_m=centerline_vertices_local_m,
    )


def _square_source(
    coil: dict[str, Any], *, apply_field_scale: bool, use_physical_construction: bool
) -> _SquareSource:
    geometry = coil.get("geometry", {})
    if str(coil.get("kind", "")) != "coil" or str(geometry.get("shape", "")) != "square":
        raise StudioOperationError("GetDP Square translation requires a Square/rectangular coil.")

    label = str(coil.get("label") or coil.get("id") or "Coil")
    vertices_mm = np.asarray(geometry.get("centerline_vertices_mm", []), dtype=float)
    if vertices_mm.ndim != 2 or vertices_mm.shape != (4, 3) or not np.all(np.isfinite(vertices_mm)):
        raise StudioOperationError(f"{label} does not contain a usable four-corner Square centreline.")
    if float(np.ptp(vertices_mm[:, 2])) > 1e-6:
        raise StudioOperationError(f"{label} Square centreline is not planar in local XY.")
    xmin, ymin = np.min(vertices_mm[:, :2], axis=0)
    xmax, ymax = np.max(vertices_mm[:, :2], axis=0)
    length_mm = float(xmax - xmin)
    width_mm = float(ymax - ymin)
    if not all(math.isfinite(v) and v > 0.0 for v in (length_mm, width_mm)):
        raise StudioOperationError(f"{label} needs positive Square/rectangular X and Y spans.")
    centre_xy_mm = np.asarray([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=float)
    centred_vertices_m = vertices_mm.copy()
    centred_vertices_m[:, 0] -= centre_xy_mm[0]
    centred_vertices_m[:, 1] -= centre_xy_mm[1]
    centred_vertices_m /= 1000.0
    area_twice = float(
        np.sum(centred_vertices_m[:, 0] * np.roll(centred_vertices_m[:, 1], -1))
        - np.sum(centred_vertices_m[:, 1] * np.roll(centred_vertices_m[:, 0], -1))
    )
    if abs(area_twice) <= 1e-18:
        raise StudioOperationError(f"{label} Square centreline encloses no usable area.")
    path_orientation_sign = 1.0 if area_twice > 0.0 else -1.0

    object_origin_m, basis_x, basis_y, normal = _coil_transform(coil)
    magnetic_centre_m = (
        object_origin_m
        + (centre_xy_mm[0] / 1000.0) * basis_x
        + (centre_xy_mm[1] / 1000.0) * basis_y
    )
    half_length_m = length_mm / 2000.0
    half_width_m = width_mm / 2000.0

    pack = _saved_path_winding_pack(coil) if use_physical_construction else None
    if pack is None:
        pack = _fallback_square_winding_pack(
            coil, minimum_half_span_mm=min(length_mm, width_mm) / 2.0
        )
        source_kind = (
            "derived numerical Square winding pack"
            if use_physical_construction
            else "forced numerical Square winding pack"
        )
    else:
        source_kind = "saved physical Square winding pack"

    inner_offset_m = float(pack["inner_offset_mm"]) / 1000.0
    outer_offset_m = float(pack["outer_offset_mm"]) / 1000.0
    axial_thickness_m = float(pack["axial_thickness_mm"]) / 1000.0
    centre_z_m = float(pack["centre_z_mm"]) / 1000.0
    radial_build_m = outer_offset_m - inner_offset_m
    inner_half_length_m = half_length_m + inner_offset_m
    inner_half_width_m = half_width_m + inner_offset_m
    outer_half_length_m = half_length_m + outer_offset_m
    outer_half_width_m = half_width_m + outer_offset_m
    values = (
        inner_offset_m, outer_offset_m, axial_thickness_m, centre_z_m, radial_build_m,
        inner_half_length_m, inner_half_width_m, outer_half_length_m, outer_half_width_m,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or radial_build_m <= 0.0
        or axial_thickness_m <= 0.0
        or inner_half_length_m <= 0.0
        or inner_half_width_m <= 0.0
        or outer_half_length_m <= inner_half_length_m
        or outer_half_width_m <= inner_half_width_m
    ):
        raise StudioOperationError(f"{label} does not produce a valid Square winding region.")

    if isinstance(pack.get("path_xy_mm"), np.ndarray):
        saved_path = np.asarray(pack["path_xy_mm"], dtype=float)
        if saved_path.ndim == 2 and saved_path.shape[1:] == (2,) and len(saved_path) >= 4:
            saved_length = float(np.ptp(saved_path[:, 0]))
            saved_width = float(np.ptp(saved_path[:, 1]))
            tolerance_mm = max(1e-5, 1e-4 * max(length_mm, width_mm, 1.0))
            if abs(saved_length - length_mm) > tolerance_mm or abs(saved_width - width_mm) > tolerance_mm:
                raise StudioOperationError(
                    f"{label}'s saved physical winding path no longer matches its Square dimensions."
                )

    source_centre_m = magnetic_centre_m + centre_z_m * normal
    cross_section_area_m2 = radial_build_m * axial_thickness_m
    excitation = coil.get("excitation", {})
    turns = int(excitation.get("turns", 1))
    drive_current_a = float(excitation.get("drive_current_a", 0.0))
    field_scale_factor = float(excitation.get("field_scale_factor", 1.0))
    enabled = bool(excitation.get("enabled", True))
    if turns < 1 or not math.isfinite(drive_current_a):
        raise StudioOperationError(f"{label} has invalid turns or drive current.")
    if not math.isfinite(field_scale_factor) or field_scale_factor <= 0.0:
        raise StudioOperationError(f"{label} has an invalid empirical Field correction.")
    applied_scale = field_scale_factor if apply_field_scale else 1.0
    ampere_turns_a = turns * drive_current_a * applied_scale if enabled else 0.0
    current_density_a_m2 = ampere_turns_a / cross_section_area_m2

    return _SquareSource(
        object_id=str(coil.get("id") or "coil"),
        label=label,
        shape="square",
        magnetic_centre_m=magnetic_centre_m,
        source_centre_m=source_centre_m,
        basis_x=basis_x,
        basis_y=basis_y,
        normal=normal,
        axial_thickness_m=axial_thickness_m,
        radial_build_m=radial_build_m,
        cross_section_area_m2=cross_section_area_m2,
        characteristic_radius_m=max(half_length_m, half_width_m),
        planar_extent_m=math.hypot(outer_half_length_m, outer_half_width_m),
        turns=turns,
        drive_current_a=drive_current_a,
        field_scale_factor=field_scale_factor,
        ampere_turns_a=ampere_turns_a,
        current_density_a_m2=current_density_a_m2,
        current_mode=str(excitation.get("current_mode", "rms_dc")),
        enabled=enabled,
        source_geometry_kind=source_kind,
        half_length_m=half_length_m,
        half_width_m=half_width_m,
        inner_offset_m=inner_offset_m,
        outer_offset_m=outer_offset_m,
        inner_half_length_m=inner_half_length_m,
        inner_half_width_m=inner_half_width_m,
        outer_half_length_m=outer_half_length_m,
        outer_half_width_m=outer_half_width_m,
        path_orientation_sign=path_orientation_sign,
        centerline_vertices_local_m=centred_vertices_m,
    )


def _coil_source(
    coil: dict[str, Any], *, apply_field_scale: bool, use_physical_construction: bool
) -> _Source:
    shape = str(coil.get("geometry", {}).get("shape", ""))
    if shape == "circle":
        return _circular_source(
            coil,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
        )
    if shape == "racetrack":
        return _racetrack_source(
            coil,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
        )
    if shape == "square":
        return _square_source(
            coil,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
        )
    raise StudioOperationError(
        f"GetDP project v{GETDP_PROJECT_VERSION} supports circular, Square/rectangular, and parametric racetrack coils only."
    )


def _coil_physical_tag(index: int) -> int:
    return _COIL_PHYSICAL_TAG_BASE + int(index)


def _coil_entity_tags(index: int) -> tuple[int, int]:
    base = _COIL_ENTITY_TAG_BASE + int(index) * _COIL_ENTITY_STRIDE
    return base, base + 1


def _polyline_centerline_reference_ut(
    source: _RacetrackSource | _SquareSource, point_m: np.ndarray
) -> np.ndarray:
    """Return the Field Workbench centreline Biot-Savart field for one Polyline family.

    Racetracks and Square/rectangular coils are both stored by Workbench as
    closed Polyline centrelines. Evaluating the same saved vertices here gives
    the acceptance table a direct Workbench reference without involving the FEM
    winding volume.
    """
    local = np.asarray(source.centerline_vertices_local_m, dtype=float)
    world = (
        source.magnetic_centre_m
        + local[:, 0, None] * source.basis_x
        + local[:, 1, None] * source.basis_y
        + local[:, 2, None] * source.normal
    )
    if len(world) < 2:
        return np.zeros(3, dtype=float)
    result_t = np.zeros(3, dtype=float)
    observation = np.asarray(point_m, dtype=float)
    for index, start in enumerate(world):
        stop = world[(index + 1) % len(world)]
        segment = stop - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-15:
            continue
        tangent = segment / length
        from_start = observation - start
        along = float(np.dot(from_start, tangent))
        perpendicular = from_start - along * tangent
        rho2 = float(np.dot(perpendicular, perpendicular))
        if rho2 <= 1e-24:
            # A verification sample placed directly on an ideal filament is
            # singular; leave it unreferenced rather than emitting nonsense.
            return np.full(3, np.nan, dtype=float)
        x0 = -along
        x1 = length - along
        integral = (
            x1 / (rho2 * math.sqrt(rho2 + x1 * x1))
            - x0 / (rho2 * math.sqrt(rho2 + x0 * x0))
        )
        result_t += (
            _MU0
            * source.ampere_turns_a
            / (4.0 * math.pi)
            * np.cross(tangent, perpendicular)
            * integral
        )
    return 1e6 * result_t


def _reference_field(
    sources: list[_Source], point_m: np.ndarray
) -> tuple[np.ndarray | None, str]:
    """Return an ideal-centreline reference where every source is evaluable."""
    result_ut = np.zeros(3, dtype=float)
    kinds: list[str] = []
    for source in sources:
        if isinstance(source, _CircularSource):
            delta = np.asarray(point_m, dtype=float) - source.magnetic_centre_m
            axial_offset_m = float(np.dot(delta, source.normal))
            radial = delta - axial_offset_m * source.normal
            tolerance_m = max(1e-10, source.mean_radius_m * 1e-8)
            if float(np.linalg.norm(radial)) > tolerance_m:
                return None, ""
            magnitude_t = (
                _MU0
                * source.ampere_turns_a
                * source.mean_radius_m**2
                / (2.0 * (source.mean_radius_m**2 + axial_offset_m**2) ** 1.5)
            )
            result_ut += 1e6 * magnitude_t * source.normal
            kinds.append("ideal circular-loop axis equation")
            continue
        if isinstance(source, _RacetrackSource):
            field_ut = _polyline_centerline_reference_ut(source, point_m)
            if not np.all(np.isfinite(field_ut)):
                return None, ""
            result_ut += field_ut
            kinds.append("FWB segmented-racetrack centreline Biot-Savart")
            continue
        if isinstance(source, _SquareSource):
            field_ut = _polyline_centerline_reference_ut(source, point_m)
            if not np.all(np.isfinite(field_ut)):
                return None, ""
            result_ut += field_ut
            kinds.append("FWB Square/rectangular centreline Biot-Savart")
            continue
        return None, ""
    if len(kinds) == 1:
        return result_ut, kinds[0]
    if len(set(kinds)) == 1:
        if kinds[0] == "FWB segmented-racetrack centreline Biot-Savart":
            return result_ut, "sum of FWB segmented-racetrack centreline Biot-Savart references"
        return result_ut, f"sum of {kinds[0]}s"
    return result_ut, "sum of ideal/centreline source references"


def _generated_verification_axis(sources: list[_Source]) -> np.ndarray:
    """Return a stable display/sample axis for an arbitrary source system."""
    first = np.asarray(sources[0].normal, dtype=float)
    aligned_normals: list[np.ndarray] = []
    for source in sources:
        normal = np.asarray(source.normal, dtype=float)
        if float(np.dot(normal, first)) < 0.0:
            normal = -normal
        aligned_normals.append(normal)

    # Preserve the intuitive coil axis for coaxial/parallel systems such as
    # Helmholtz and Maxwell arrangements. Sign-opposed source normals are
    # aligned before the comparison so current reversal cannot cancel the axis.
    if all(abs(float(np.dot(normal, first))) >= 1.0 - 1e-8 for normal in aligned_normals):
        return _normalised_vector(first, "the generated verification axis")

    # For a genuinely non-parallel system, prefer the dominant geometric spread
    # of source centres when one exists. This gives arbitrary user scenes a
    # deterministic and inspectable default without pretending it is a common
    # magnetic axis.
    centres = np.vstack([source.magnetic_centre_m for source in sources])
    centred = centres - np.mean(centres, axis=0)
    if len(sources) >= 2 and float(np.linalg.norm(centred)) > 1e-12:
        _, singular_values, vh = np.linalg.svd(centred, full_matrices=False)
        if len(singular_values) and float(singular_values[0]) > 1e-12:
            axis = np.asarray(vh[0], dtype=float)
            if float(np.dot(axis, first)) < 0.0:
                axis = -axis
            return _normalised_vector(axis, "the generated verification axis")

    mean_normal = np.sum(np.vstack(aligned_normals), axis=0)
    if float(np.linalg.norm(mean_normal)) > 1e-12:
        return _normalised_vector(mean_normal, "the generated verification axis")
    return np.asarray([0.0, 0.0, 1.0], dtype=float)


def _default_sample_points(
    sources: list[_Source],
) -> list[tuple[str, np.ndarray]]:
    if len(sources) == 1:
        source = sources[0]
        scale_label = "R" if isinstance(source, _CircularSource) else "scale"
        return [
            (
                f"axis {multiplier:+g} {scale_label}",
                source.magnetic_centre_m
                + multiplier * source.characteristic_radius_m * source.normal,
            )
            for multiplier in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]

    centre_m = np.mean([source.magnetic_centre_m for source in sources], axis=0)
    axis = _generated_verification_axis(sources)
    radius_m = max(source.characteristic_radius_m for source in sources)
    scale_label = "Rmax" if all(isinstance(source, _CircularSource) for source in sources) else "scale"
    label_prefix = "pair axis" if len(sources) == 2 else "system axis"
    return [
        (
            f"{label_prefix} {multiplier:+g} {scale_label}",
            centre_m + multiplier * radius_m * axis,
        )
        for multiplier in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]


def _sample_points(
    scene: dict[str, Any], sources: list[_Source]
) -> list[_SamplePoint]:
    raw_points: list[tuple[str, np.ndarray]] = []
    for obj in scene.get("objects", []):
        if str(obj.get("kind", "")) != "sensor":
            continue
        sensor_label = str(obj.get("label") or obj.get("id") or "Sensor")
        points_mm = np.asarray(obj.get("sample_path_mm", []), dtype=float)
        if points_mm.ndim != 2 or points_mm.shape[1:] != (3,):
            continue
        for index, point_mm in enumerate(points_mm):
            if np.all(np.isfinite(point_mm)):
                raw_points.append((f"{sensor_label} {index + 1}", point_mm / 1000.0))

    if not raw_points:
        raw_points.extend(_default_sample_points(sources))

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int, int]] = set()
    for label, point_m in raw_points:
        key = tuple(int(round(float(value) * 1e12)) for value in point_m)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, point_m))
    if len(unique) > GETDP_MAX_SAMPLE_POINTS:
        raise StudioOperationError(
            f"GetDP export supports at most {GETDP_MAX_SAMPLE_POINTS:,} explicit sample points; "
            f"the selection contains {len(unique):,}."
        )

    samples: list[_SamplePoint] = []
    for label, point_m in unique:
        reference, reference_kind = _reference_field(sources, point_m)
        samples.append(
            _SamplePoint(
                label=label,
                point_m=np.asarray(point_m, dtype=float),
                reference_b_ut=reference,
                reference_kind=reference_kind,
            )
        )
    return samples


def _air_domain(
    sources: list[_Source],
    samples: list[_SamplePoint],
) -> tuple[np.ndarray, float, float]:
    """Return the automatic ordinary-air and infinite-shell radii.

    The ordinary finite region only needs to enclose the physical sources and
    requested samples with useful clearance.  A surrounding finite shell is then
    mapped by GetDP's ``VolSphShell`` Jacobian to the complementary unbounded
    exterior.  Consequently mesh quality no longer changes the physical boundary
    placement.
    """
    centre_m = np.mean([source.magnetic_centre_m for source in sources], axis=0)
    characteristic_radius_m = max(source.characteristic_radius_m for source in sources)
    source_extent_m = max(
        float(np.linalg.norm(source.source_centre_m - centre_m))
        + math.hypot(source.planar_extent_m, source.axial_thickness_m / 2.0)
        for source in sources
    )
    sample_extent_m = max(
        (float(np.linalg.norm(sample.point_m - centre_m)) for sample in samples),
        default=0.0,
    )

    # Keep the transformed shell comfortably outside every finite source/sample
    # while avoiding the old multi-radius empty-air sphere.  The half-R clearance
    # mirrors the scale used in the official GetDP transformation examples, but
    # is applied to our true radial source extent instead of an axis-aligned box.
    clearance_m = 0.5 * characteristic_radius_m
    inner_radius_m = max(
        source_extent_m + clearance_m,
        sample_extent_m + clearance_m,
        1.5 * characteristic_radius_m,
    )
    # The finite shell is only the coordinate-transform carrier; it does not need
    # to represent additional physical free space.  Scale its thickness with both
    # source size and the chosen inner radius so widely separated source systems
    # remain well behaved.
    shell_thickness_m = max(
        0.5 * characteristic_radius_m,
        0.25 * inner_radius_m,
    )
    outer_radius_m = inner_radius_m + shell_thickness_m
    if (
        not math.isfinite(inner_radius_m)
        or not math.isfinite(outer_radius_m)
        or inner_radius_m <= source_extent_m
        or outer_radius_m <= inner_radius_m
    ):
        raise StudioOperationError(
            "GetDP export could not construct the automatic infinite-shell domain."
        )
    return centre_m, inner_radius_m, outer_radius_m


def _build_mesh_plan(
    sources: list[_Source],
    samples: list[_SamplePoint],
    settings: GetDPMeshSettings,
) -> _MeshPlan:
    """Resolve scale-independent controls into concrete SI mesh lengths."""
    air_centre_m, inner_air_radius_m, shell_outer_radius_m = _air_domain(
        sources, samples
    )
    source_mesh_sizes = [
        max(
            1e-5,
            min(source.radial_build_m, source.axial_thickness_m)
            / settings.source_elements,
        )
        for source in sources
    ]
    source_mesh_m = min(source_mesh_sizes)
    characteristic_radius_m = max(source.characteristic_radius_m for source in sources)
    far_mesh_m = max(
        4.0 * source_mesh_m,
        inner_air_radius_m / settings.far_radius_divisions,
    )
    # Keep the source mesh local, but resolve the source/sample neighbourhood
    # more tightly than ordinary far air. The 3x source-size floor prevents the
    # background field from needlessly propagating the very finest winding scale.
    near_mesh_m = min(
        far_mesh_m,
        max(
            3.0 * source_mesh_m,
            characteristic_radius_m / settings.near_radius_divisions,
        ),
    )
    # Very fine winding packs can otherwise create an abrupt source-to-air size jump:
    # e.g. a ~0.1 mm winding target directly adjoining millimetre-scale near air.
    # The first Validation grading implementation used lc_source itself as the
    # Distance/Threshold SizeMin on every winding surface.  That was too literal: a
    # thin homogenized winding then forced sub-millimetre triangles over its entire
    # air interface (the Maxwell fixture jumped from ~106k to >2M pre-volume nodes).
    # Keep the fine source target for the winding geometry itself, but give the AIR
    # side a separately derived minimum size and grow that smoothly into lc_near.
    # Fast/Standard retain their accepted behaviour; this remains an internal quality
    # safeguard rather than another user-facing Gmsh knob.
    source_grading_enabled = (
        settings.preset == "validation"
        or (settings.preset == "custom" and settings.source_elements >= 3.0)
    ) and near_mesh_m > 1.25 * source_mesh_m
    source_grading_air_min_m = min(
        near_mesh_m,
        max(4.0 * source_mesh_m, near_mesh_m / 4.0),
    )
    # Hold roughly one air-interface element at the derived minimum before allowing
    # growth.  Then transition over several local element widths, capped so a small
    # winding does not drag its fine scale throughout the whole near-field box.
    source_grading_start_m = source_grading_air_min_m
    grading_span_m = max(
        6.0 * source_grading_air_min_m,
        4.0 * max(0.0, near_mesh_m - source_grading_air_min_m),
    )
    grading_span_m = min(grading_span_m, 0.20 * characteristic_radius_m)
    source_grading_end_m = source_grading_start_m + grading_span_m

    # Give the coordinate-transform shell at least four nominal radial layers.
    # This is automatic and deliberately independent of the UI quality preset: the
    # unbounded-boundary model is part of the physics, not an expert tuning knob.
    shell_thickness_m = shell_outer_radius_m - inner_air_radius_m
    shell_mesh_m = min(far_mesh_m, shell_thickness_m / 4.0)

    extent_mins: list[np.ndarray] = []
    extent_maxs: list[np.ndarray] = []
    for source in sources:
        half_axial_m = source.axial_thickness_m / 2.0
        if isinstance(source, _CircularSource):
            radial_extent_m = source.outer_radius_m * np.sqrt(
                source.basis_x**2 + source.basis_y**2
            )
        elif isinstance(source, _RacetrackSource):
            radial_extent_m = (
                source.half_straight_m * np.abs(source.basis_x)
                + source.outer_end_radius_m
                * np.sqrt(source.basis_x**2 + source.basis_y**2)
            )
        else:
            radial_extent_m = (
                source.outer_half_length_m * np.abs(source.basis_x)
                + source.outer_half_width_m * np.abs(source.basis_y)
            )
        axial_extent_m = half_axial_m * np.abs(source.normal)
        extent_mins.append(source.source_centre_m - radial_extent_m - axial_extent_m)
        extent_maxs.append(source.source_centre_m + radial_extent_m + axial_extent_m)
    source_extent_min_m = np.min(np.vstack(extent_mins), axis=0)
    source_extent_max_m = np.max(np.vstack(extent_maxs), axis=0)
    sample_coordinates = np.vstack([sample.point_m for sample in samples])
    local_padding_m = max(
        characteristic_radius_m / 4.0,
        *(2.0 * source.radial_build_m for source in sources),
        *(2.0 * source.axial_thickness_m for source in sources),
    )
    local_min_m = (
        np.minimum(source_extent_min_m, sample_coordinates.min(axis=0))
        - local_padding_m
    )
    local_max_m = (
        np.maximum(source_extent_max_m, sample_coordinates.max(axis=0))
        + local_padding_m
    )
    return _MeshPlan(
        air_centre_m=air_centre_m,
        inner_air_radius_m=inner_air_radius_m,
        shell_outer_radius_m=shell_outer_radius_m,
        shell_mesh_m=shell_mesh_m,
        source_mesh_m=source_mesh_m,
        near_mesh_m=near_mesh_m,
        far_mesh_m=far_mesh_m,
        source_grading_enabled=source_grading_enabled,
        source_grading_air_min_m=source_grading_air_min_m,
        source_grading_start_m=source_grading_start_m,
        source_grading_end_m=source_grading_end_m,
        local_min_m=local_min_m,
        local_max_m=local_max_m,
    )


def _source_probe_point(source: _Source) -> np.ndarray:
    """Return a point strictly inside a winding volume for Gmsh ``Closest``."""
    if isinstance(source, _CircularSource):
        return source.source_centre_m + source.mean_radius_m * source.basis_x
    if isinstance(source, _RacetrackSource):
        return source.source_centre_m + source.end_radius_m * source.basis_y
    return source.source_centre_m + source.half_width_m * source.basis_y


def _rotation_axis_angle(source: _Source) -> tuple[np.ndarray, float]:
    """Return the local-XYZ to source-world rotation as axis/angle."""
    matrix = np.column_stack([source.basis_x, source.basis_y, source.normal])
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0]), 0.0
    if abs(math.pi - angle) <= 1e-8:
        # Stable 180-degree extraction from the symmetric part of R.
        diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) / 2.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] <= 1e-10:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            if largest == 0:
                axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif largest == 1:
                axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
            axis = _normalised_vector(axis, "the path-coil export rotation axis")
        return axis, angle
    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(angle))
    return _normalised_vector(axis, "the path-coil export rotation axis"), angle


def _append_racetrack_geo(
    lines: list[str], source_index: int, source: _RacetrackSource
) -> str:
    """Append one exact OCC capsule-band prism and return its volume-array name."""
    display_index = source_index + 1
    base = 100 + 100 * source_index
    half = source.half_straight_m
    thickness = source.axial_thickness_m
    z0 = -thickness / 2.0
    outer = source.outer_end_radius_m
    inner = source.inner_end_radius_m
    array_name = f"coil_{display_index}_volume"

    lines.append(
        f"// Racetrack {display_index}: exact capsule band in local XY, then rigidly placed in world coordinates."
    )
    if half <= 1e-15:
        outer_tag = base + 1
        inner_tag = base + 2
        lines.extend(
            [
                f"Cylinder({outer_tag}) = {{0, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(outer)}}};",
                f"Cylinder({inner_tag}) = {{0, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(inner)}}};",
                (
                    f"{array_name}[] = BooleanDifference"
                    f"{{ Volume{{{outer_tag}}}; Delete; }}"
                    f"{{ Volume{{{inner_tag}}}; Delete; }};"
                ),
            ]
        )
    else:
        straight = 2.0 * half
        outer_rect, outer_left, outer_right = base + 1, base + 2, base + 3
        inner_rect, inner_left, inner_right = base + 4, base + 5, base + 6
        lines.extend(
            [
                f"Box({outer_rect}) = {{{_number(-half)}, {_number(-outer)}, {_number(z0)}, {_number(straight)}, {_number(2.0 * outer)}, {_number(thickness)}}};",
                f"Cylinder({outer_left}) = {{{_number(-half)}, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(outer)}}};",
                f"Cylinder({outer_right}) = {{{_number(half)}, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(outer)}}};",
                f"outer_{display_index}_a[] = BooleanUnion{{ Volume{{{outer_rect}}}; Delete; }}{{ Volume{{{outer_left}}}; Delete; }};",
                f"outer_{display_index}[] = BooleanUnion{{ Volume{{outer_{display_index}_a[]}}; Delete; }}{{ Volume{{{outer_right}}}; Delete; }};",
                f"Box({inner_rect}) = {{{_number(-half)}, {_number(-inner)}, {_number(z0)}, {_number(straight)}, {_number(2.0 * inner)}, {_number(thickness)}}};",
                f"Cylinder({inner_left}) = {{{_number(-half)}, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(inner)}}};",
                f"Cylinder({inner_right}) = {{{_number(half)}, 0, {_number(z0)}, 0, 0, {_number(thickness)}, {_number(inner)}}};",
                f"inner_{display_index}_a[] = BooleanUnion{{ Volume{{{inner_rect}}}; Delete; }}{{ Volume{{{inner_left}}}; Delete; }};",
                f"inner_{display_index}[] = BooleanUnion{{ Volume{{inner_{display_index}_a[]}}; Delete; }}{{ Volume{{{inner_right}}}; Delete; }};",
                (
                    f"{array_name}[] = BooleanDifference"
                    f"{{ Volume{{outer_{display_index}[]}}; Delete; }}"
                    f"{{ Volume{{inner_{display_index}[]}}; Delete; }};"
                ),
            ]
        )

    axis, angle = _rotation_axis_angle(source)
    if abs(angle) > 1e-12:
        lines.append(
            f"Rotate {{{{{_number(axis[0])}, {_number(axis[1])}, {_number(axis[2])}}}, "
            f"{{0, 0, 0}}, {_number(angle)}}} {{ Volume{{{array_name}[]}}; }}"
        )
    centre = source.source_centre_m
    if float(np.linalg.norm(centre)) > 1e-15:
        lines.append(
            f"Translate {{{_number(centre[0])}, {_number(centre[1])}, {_number(centre[2])}}} "
            f"{{ Volume{{{array_name}[]}}; }}"
        )
    lines.append("")
    return array_name


def _append_square_geo(
    lines: list[str], source_index: int, source: _SquareSource
) -> str:
    """Append one exact rectangular path-band prism and return its volume-array name."""
    display_index = source_index + 1
    base = _SQUARE_ENTITY_TAG_BASE + _SQUARE_ENTITY_STRIDE * source_index
    outer_tag = base + 1
    inner_tag = base + 2
    thickness = source.axial_thickness_m
    z0 = -thickness / 2.0
    array_name = f"coil_{display_index}_volume"
    lines.extend(
        [
            f"// Square/rectangular source {display_index}: exact mitered path band in local XY, then rigidly placed in world coordinates.",
            f"Box({outer_tag}) = {{{_number(-source.outer_half_length_m)}, {_number(-source.outer_half_width_m)}, {_number(z0)}, {_number(2.0 * source.outer_half_length_m)}, {_number(2.0 * source.outer_half_width_m)}, {_number(thickness)}}};",
            f"Box({inner_tag}) = {{{_number(-source.inner_half_length_m)}, {_number(-source.inner_half_width_m)}, {_number(z0)}, {_number(2.0 * source.inner_half_length_m)}, {_number(2.0 * source.inner_half_width_m)}, {_number(thickness)}}};",
            (
                f"{array_name}[] = BooleanDifference"
                f"{{ Volume{{{outer_tag}}}; Delete; }}"
                f"{{ Volume{{{inner_tag}}}; Delete; }};"
            ),
        ]
    )
    axis, angle = _rotation_axis_angle(source)
    if abs(angle) > 1e-12:
        lines.append(
            f"Rotate {{{{{_number(axis[0])}, {_number(axis[1])}, {_number(axis[2])}}}, "
            f"{{0, 0, 0}}, {_number(angle)}}} {{ Volume{{{array_name}[]}}; }}"
        )
    centre = source.source_centre_m
    if float(np.linalg.norm(centre)) > 1e-15:
        lines.append(
            f"Translate {{{_number(centre[0])}, {_number(centre[1])}, {_number(centre[2])}}} "
            f"{{ Volume{{{array_name}[]}}; }}"
        )
    lines.append("")
    return array_name


def _use_hxt_mesher(
    mesh_settings: GetDPMeshSettings,
    mesh_plan: _MeshPlan,
) -> bool:
    """Return whether the generated project should use Gmsh HXT for 3-D meshing.

    Fast and Standard deliberately keep the established single-threaded Delaunay
    path. Validation uses HXT so strongly graded, high-resolution source meshes do
    not spend millions of classic Delaunay cavity-repair iterations. A Custom
    configuration follows the same rule when it activates Validation-style source
    grading.
    """
    return mesh_settings.preset == "validation" or (
        mesh_settings.preset == "custom" and mesh_plan.source_grading_enabled
    )


def _render_geo(
    sources: list[_Source],
    samples: list[_SamplePoint],
    mesh_plan: _MeshPlan,
    mesh_settings: GetDPMeshSettings,
) -> str:
    air_centre_m = mesh_plan.air_centre_m
    inner_air_radius_m = mesh_plan.inner_air_radius_m
    shell_outer_radius_m = mesh_plan.shell_outer_radius_m
    shell_mesh_m = mesh_plan.shell_mesh_m
    source_mesh_m = mesh_plan.source_mesh_m
    near_mesh_m = mesh_plan.near_mesh_m
    far_mesh_m = mesh_plan.far_mesh_m
    source_grading_enabled = mesh_plan.source_grading_enabled
    source_grading_air_min_m = mesh_plan.source_grading_air_min_m
    source_grading_start_m = mesh_plan.source_grading_start_m
    source_grading_end_m = mesh_plan.source_grading_end_m
    local_min_m = mesh_plan.local_min_m
    local_max_m = mesh_plan.local_max_m
    use_hxt = _use_hxt_mesher(mesh_settings, mesh_plan)
    mesher_label = "HXT" if use_hxt else "Delaunay"

    lines = [
        "// Field Workbench Gmsh/GetDP magnetostatic project",
        f"// Stranded-current source count: {len(sources)}",
        f"// Mesh preset: {mesh_settings.preset}",
        "// Open boundary: automatic ordinary-air sphere + VolSphShell infinite exterior.",
        (
            "// Derived mesh lengths: source="
            f"{_number(1000.0 * source_mesh_m)} mm, near={_number(1000.0 * near_mesh_m)} mm, "
            f"far={_number(1000.0 * far_mesh_m)} mm, shell={_number(1000.0 * shell_mesh_m)} mm"
        ),
        (
            "// Automatic radii: ordinary air="
            f"{_number(1000.0 * inner_air_radius_m)} mm; shell outer="
            f"{_number(1000.0 * shell_outer_radius_m)} mm"
        ),
    ]
    for index, source in enumerate(sources, start=1):
        label = source.label.replace("\n", " ").replace("\r", " ")
        lines.append(f"// Source {index}: {label}; {source.source_geometry_kind}")
    lines.extend(
        [
            f"// Compatibility floor: Gmsh {GETDP_MIN_GMSH_VERSION};",
            "// mesh format intentionally fixed to ASCII MSH 2.2.",
            'SetFactory("OpenCASCADE");',
            "Geometry.OCCBooleanPreserveNumbering = 1;",
            "General.Terminal = 1;",
            *(
                [
                    "// Validation-scale meshing: HXT is Gmsh's parallel 3D Delaunay implementation.",
                    "// General.NumThreads = 0 asks Gmsh to use the system thread count.",
                    "General.NumThreads = 0;",
                    "Mesh.MaxNumThreads3D = 0;",
                ]
                if use_hxt
                else []
            ),
            "Mesh.MshFileVersion = 2.2;",
            "Mesh.Binary = 0;",
            "Mesh.SaveAll = 0;",
            f"// 3D mesher: {mesher_label}",
            f"Mesh.Algorithm3D = {10 if use_hxt else 1};",
            "Mesh.Optimize = 1;",
            "",
            f"lc_source = {_number(source_mesh_m)};",
            f"lc_near = {_number(near_mesh_m)};",
            f"lc_far = {_number(far_mesh_m)};",
            f"lc_shell = {_number(shell_mesh_m)};",
            f"source_grade_air_min = {_number(source_grading_air_min_m)};",
            f"source_grade_start = {_number(source_grading_start_m)};",
            f"source_grade_end = {_number(source_grading_end_m)};",
            "Mesh.CharacteristicLengthMin = lc_source;",
            "Mesh.CharacteristicLengthMax = lc_far;",
            "Mesh.CharacteristicLengthFromPoints = 1;",
            "Mesh.CharacteristicLengthFromCurvature = 0;",
            "Mesh.CharacteristicLengthExtendFromBoundary = 0;",
            "",
            "// Automatic finite geometry for an unbounded magnetic exterior.",
            "// The inner sphere is ordinary free space.  The annulus between these",
            "// two spheres is mapped to infinity by GetDP's VolSphShell Jacobian.",
            (
                f"Sphere({_AIR_ENTITY_TAG}) = "
                f"{{{_number(air_centre_m[0])}, {_number(air_centre_m[1])}, "
                f"{_number(air_centre_m[2])}, {_number(inner_air_radius_m)}}};"
            ),
            (
                f"Sphere({_INFINITE_SHELL_OUTER_ENTITY_TAG}) = "
                f"{{{_number(air_centre_m[0])}, {_number(air_centre_m[1])}, "
                f"{_number(air_centre_m[2])}, {_number(shell_outer_radius_m)}}};"
            ),
            "",
            "// Homogenized stranded-current winding regions.",
        ]
    )

    # Create each source first.  We deliberately re-identify the final conformal
    # volumes geometrically after Coherence, following the official GetDP/Gmsh
    # infinite-shell examples, rather than relying on OCC tags surviving a global
    # BooleanFragments operation.
    for source_index, source in enumerate(sources):
        display_index = source_index + 1
        if isinstance(source, _RacetrackSource):
            _append_racetrack_geo(lines, source_index, source)
            continue
        if isinstance(source, _SquareSource):
            _append_square_geo(lines, source_index, source)
            continue

        outer_tag, bore_tag = _coil_entity_tags(source_index)
        half_axial_m = source.axial_thickness_m / 2.0
        start_m = source.source_centre_m - half_axial_m * source.normal
        direction_m = source.axial_thickness_m * source.normal
        array_name = f"coil_{display_index}_volume"
        lines.extend(
            [
                (
                    f"Cylinder({outer_tag}) = "
                    f"{{{_number(start_m[0])}, {_number(start_m[1])}, {_number(start_m[2])}, "
                    f"{_number(direction_m[0])}, {_number(direction_m[1])}, {_number(direction_m[2])}, "
                    f"{_number(source.outer_radius_m)}}};"
                ),
                (
                    f"Cylinder({bore_tag}) = "
                    f"{{{_number(start_m[0])}, {_number(start_m[1])}, {_number(start_m[2])}, "
                    f"{_number(direction_m[0])}, {_number(direction_m[1])}, {_number(direction_m[2])}, "
                    f"{_number(source.inner_radius_m)}}};"
                ),
                (
                    f"{array_name}[] = BooleanDifference"
                    f"{{ Volume{{{outer_tag}}}; Delete; }}"
                    f"{{ Volume{{{bore_tag}}}; Delete; }};"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "// Fragment every overlapping OCC volume at once. This makes the winding/air",
            "// and ordinary-air/infinite-shell interfaces conformal with no duplicate faces.",
            "Coherence;",
            "",
        ]
    )

    shell_probe = air_centre_m.copy()
    shell_probe[0] += 0.5 * (inner_air_radius_m + shell_outer_radius_m)
    lines.append(
        "airinf() = Closest "
        f"{{{_number(shell_probe[0])}, {_number(shell_probe[1])}, {_number(shell_probe[2])}}} "
        "{ Volume{:}; };"
    )
    coil_region_names: list[str] = []
    for source_index, source in enumerate(sources):
        display_index = source_index + 1
        probe = _source_probe_point(source)
        region_name = f"coil_{display_index}_region"
        coil_region_names.append(region_name)
        lines.append(
            f"{region_name}() = Closest "
            f"{{{_number(probe[0])}, {_number(probe[1])}, {_number(probe[2])}}} "
            "{ Volume{:}; };"
        )

    coil_region_refs = ", ".join(f"{name}(0)" for name in coil_region_names)
    lines.extend(
        [
            f"coil_volumes() = {{{coil_region_refs}}};",
            "coil_surfaces_signed() = CombinedBoundary{ Volume{coil_volumes()}; };",
            "coil_surfaces() = Abs[coil_surfaces_signed()];",
            f"excluded_volumes() = {{airinf(0), {coil_region_refs}}};",
            "air_volume() = Volume{:};",
            "air_volume() -= excluded_volumes();",
        ]
    )
    for source_index in range(len(sources)):
        display_index = source_index + 1
        lines.append(
            f'Physical Volume("Coil_{display_index}", {_coil_physical_tag(source_index)}) = '
            f"{{coil_{display_index}_region(0)}};"
        )
    lines.extend(
        [
            f'Physical Volume("Air", {_AIR_PHYSICAL_TAG}) = {{air_volume()}};',
            f'Physical Volume("AirInf", {_INFINITE_AIR_PHYSICAL_TAG}) = {{airinf(0)}};',
            "outer_boundary() = CombinedBoundary{ Volume{:}; };",
            (
                f'Physical Surface("OuterBoundary", {_OUTER_BOUNDARY_PHYSICAL_TAG}) = '
                "{outer_boundary()};"
            ),
            "",
            "MeshSize { PointsOf{ Volume{coil_volumes()}; } } = lc_source;",
            "MeshSize { PointsOf{ Volume{airinf(0)}; } } = lc_shell;",
            "MeshSize { PointsOf{ Surface{outer_boundary()}; } } = lc_shell;",
            "",
            "// Keep the field-sampling neighbourhood resolved without refining all ordinary air.",
            "Field[1] = Box;",
            "Field[1].VIn = lc_near;",
            "Field[1].VOut = lc_far;",
            f"Field[1].XMin = {_number(local_min_m[0])};",
            f"Field[1].XMax = {_number(local_max_m[0])};",
            f"Field[1].YMin = {_number(local_min_m[1])};",
            f"Field[1].YMax = {_number(local_max_m[1])};",
            f"Field[1].ZMin = {_number(local_min_m[2])};",
            f"Field[1].ZMax = {_number(local_max_m[2])};",
        ]
    )
    if source_grading_enabled:
        lines.extend(
            [
                "",
                "// Validation safeguard: retain the fine winding-geometry target, but",
                "// grade surrounding air from a separate interface size into lc_near.",
                "Field[2] = Distance;",
                "Field[2].SurfacesList = {coil_surfaces()};",
                "Field[2].Sampling = 100;",
                "Field[3] = Threshold;",
                "Field[3].InField = 2;",
                "Field[3].SizeMin = source_grade_air_min;",
                "Field[3].SizeMax = lc_near;",
                "Field[3].DistMin = source_grade_start;",
                "Field[3].DistMax = source_grade_end;",
                "Field[3].StopAtDistMax = 1;",
                "Field[4] = Min;",
                "Field[4].FieldsList = {1, 3};",
                "Background Field = 4;",
                "",
            ]
        )
    else:
        lines.extend(["Background Field = 1;", ""])
    return "\n".join(lines)


def _render_pro(
    sources: list[_Source], samples: list[_SamplePoint], mesh_plan: _MeshPlan
) -> str:
    coordinate_functions = ("X[]", "Y[]", "Z[]")

    def shifted_coordinate(name: str, offset: float) -> str:
        if abs(float(offset)) < 5e-16:
            return name
        operator = "-" if offset > 0.0 else "+"
        return f"({name} {operator} {_number(abs(offset))})"

    coil_names = [f"Coil_{index}" for index in range(1, len(sources) + 1)]
    lines = [
        "// Field Workbench GetDP 3-D magnetostatic project",
        f"// Compatibility floor: GetDP {GETDP_MIN_SOLVER_VERSION}",
        "// SI units throughout: metre, ampere, tesla. Display/sample B output is microtesla.",
        "",
        'ResultDir = "results/";',
        "",
        f"InfiniteShellRInt = {_number(mesh_plan.inner_air_radius_m)};",
        f"InfiniteShellRExt = {_number(mesh_plan.shell_outer_radius_m)};",
        f"InfiniteShellCenterX = {_number(mesh_plan.air_centre_m[0])};",
        f"InfiniteShellCenterY = {_number(mesh_plan.air_centre_m[1])};",
        f"InfiniteShellCenterZ = {_number(mesh_plan.air_centre_m[2])};",
        "",
        "Group {",
        f"  Air = Region[{_AIR_PHYSICAL_TAG}];",
        f"  AirInf = Region[{_INFINITE_AIR_PHYSICAL_TAG}];",
    ]
    for source_index, coil_name in enumerate(coil_names):
        lines.append(f"  {coil_name} = Region[{_coil_physical_tag(source_index)}];")
    lines.extend(
        [
            f"  OuterBoundary = Region[{_OUTER_BOUNDARY_PHYSICAL_TAG}];",
            f"  DomainWithSourceCurrentDensity = Region[{{{', '.join(coil_names)}}}];",
            "  Domain = Region[{Air, AirInf, DomainWithSourceCurrentDensity}];",
            "}",
            "",
            "Function {",
            "  mu0 = 4.e-7 * Pi;",
            "  nu[Domain] = 1. / mu0;",
            "",
        ]
    )

    for source_index, source in enumerate(sources, start=1):
        centre = source.source_centre_m
        bx = source.basis_x
        by = source.basis_y
        u_expression = " + ".join(
            f"{shifted_coordinate(coordinate_functions[index], centre[index])} * {_number(bx[index])}"
            for index in range(3)
        )
        v_expression = " + ".join(
            f"{shifted_coordinate(coordinate_functions[index], centre[index])} * {_number(by[index])}"
            for index in range(3)
        )
        lines.extend(
            [
                f"  // Source {source_index}: {source.label.replace(chr(10), ' ').replace(chr(13), ' ')}",
                f"  // Shape: {source.shape}; {source.turns} turns * {_number(source.drive_current_a)} A",
                "  // Effective signed ampere-turns in this export: "
                f"{_number(source.ampere_turns_a)} A.turn",
                "  // Homogenized winding cross-section: "
                f"{_number(source.cross_section_area_m2)} m^2",
                f"  Coil{source_index}U[] = {u_expression};",
                f"  Coil{source_index}V[] = {v_expression};",
            ]
        )
        if isinstance(source, _CircularSource):
            tangent_components = [
                (
                    f"(-Coil{source_index}V[] * {_number(bx[index])} + "
                    f"Coil{source_index}U[] * {_number(by[index])}) / Coil{source_index}R[]"
                )
                for index in range(3)
            ]
            lines.extend(
                [
                    f"  Coil{source_index}R[] = Sqrt[Coil{source_index}U[]^2 + Coil{source_index}V[]^2];",
                    f"  SourceCurrentDensity[Coil_{source_index}] =",
                    f"    {_number(source.current_density_a_m2)} * Vector[",
                    f"      {tangent_components[0]},",
                    f"      {tangent_components[1]},",
                    f"      {tangent_components[2]}",
                    "    ];",
                    "",
                ]
            )
            continue

        if isinstance(source, _SquareSource):
            # The winding band is bounded by two miter-offset rectangles.  Let
            # d=max(|u|-halfX, |v|-halfY); its level sets are those same nested
            # rectangles.  A 90-degree rotation of grad(d) is therefore tangent
            # to every inner/outer boundary.  The diagonal selector gives the
            # expected constant direction on each side while preserving current
            # continuity through each miter plane.
            lines.extend(
                [
                    f"  Coil{source_index}DX[] = Fabs[Coil{source_index}U[]] - {_number(source.half_length_m)};",
                    f"  Coil{source_index}DY[] = Fabs[Coil{source_index}V[]] - {_number(source.half_width_m)};",
                ]
            )
            tangent_components = [
                (
                    f"((Coil{source_index}DX[] >= Coil{source_index}DY[]) ? "
                    f"(Sign[Coil{source_index}U[]] * {_number(by[index])}) : "
                    f"(-Sign[Coil{source_index}V[]] * {_number(bx[index])}))"
                )
                for index in range(3)
            ]
            signed_density = source.current_density_a_m2 * source.path_orientation_sign
            lines.extend(
                [
                    f"  SourceCurrentDensity[Coil_{source_index}] =",
                    f"    {_number(signed_density)} * Vector[",
                    f"      {tangent_components[0]},",
                    f"      {tangent_components[1]},",
                    f"      {tangent_components[2]}",
                    "    ];",
                    "",
                ]
            )
            continue

        # A racetrack is the offset of a finite line segment by a radius. The
        # closest point Q on that centre segment gives a single smooth tangent
        # definition for both straight sides and semicircular ends:
        #   q = clamp(u, -halfStraight, +halfStraight)
        #   tangent_local = (-v, u-q) / hypot(u-q, v)
        # This has unit magnitude everywhere inside the winding band and is
        # divergence-free apart from the harmless piecewise derivative at the
        # straight/end join.
        half = source.half_straight_m
        lines.extend(
            [
                f"  Coil{source_index}Q[] = Max[{_number(-half)}, Min[{_number(half)}, Coil{source_index}U[]]];",
                f"  Coil{source_index}DU[] = Coil{source_index}U[] - Coil{source_index}Q[];",
                f"  Coil{source_index}R[] = Sqrt[Coil{source_index}DU[]^2 + Coil{source_index}V[]^2];",
            ]
        )
        tangent_components = [
            (
                f"(-Coil{source_index}V[] * {_number(bx[index])} + "
                f"Coil{source_index}DU[] * {_number(by[index])}) / Coil{source_index}R[]"
            )
            for index in range(3)
        ]
        lines.extend(
            [
                f"  SourceCurrentDensity[Coil_{source_index}] =",
                f"    {_number(source.current_density_a_m2)} * Vector[",
                f"      {tangent_components[0]},",
                f"      {tangent_components[1]},",
                f"      {tangent_components[2]}",
                "    ];",
                "",
            ]
        )

    lines.extend(
        [
            "}",
            "",
            "Constraint {",
            "  { Name MagneticVectorPotential; Type Assign;",
            "    Case {",
            "      { Region OuterBoundary; Value 0.; }",
            "    }",
            "  }",
            "  { Name GaugeCondition; Type Assign;",
            "    Case {",
            "      { Region Domain; SubRegion OuterBoundary; Value 0.; }",
            "    }",
            "  }",
            "}",
            "",
            "FunctionSpace {",
            "  { Name Hcurl_a_Gauge; Type Form1;",
            "    BasisFunction {",
            "      { Name se; NameOfCoef ae; Function BF_Edge;",
            "        Support Domain; Entity EdgesOf[All]; }",
            "    }",
            "    Constraint {",
            "      { NameOfCoef ae; EntityType EdgesOf;",
            "        NameOfConstraint MagneticVectorPotential; }",
            "      { NameOfCoef ae; EntityType EdgesOfTreeIn; EntitySubType StartingOn;",
            "        NameOfConstraint GaugeCondition; }",
            "    }",
            "  }",
            "}",
            "",
            "Jacobian {",
            "  { Name Vol;",
            "    Case {",
            "      { Region AirInf;",
            "        Jacobian VolSphShell {InfiniteShellRInt, InfiniteShellRExt,",
            "                              InfiniteShellCenterX, InfiniteShellCenterY, InfiniteShellCenterZ}; }",
            "      { Region All; Jacobian Vol; }",
            "    }",
            "  }",
            "}",
            "",
            "Integration {",
            "  { Name Int;",
            "    Case {",
            "      { Type Gauss;",
            "        Case {",
            "          { GeoElement Tetrahedron; NumberOfPoints 4; }",
            "          { GeoElement Hexahedron; NumberOfPoints 6; }",
            "          { GeoElement Prism; NumberOfPoints 9; }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "",
            "Formulation {",
            "  { Name Magnetostatics_a_3D; Type FemEquation;",
            "    Quantity {",
            "      { Name a; Type Local; NameOfSpace Hcurl_a_Gauge; }",
            "    }",
            "    Equation {",
            "      Galerkin { [ nu[] * Dof{d a}, {d a} ];",
            "                 In Domain; Jacobian Vol; Integration Int; }",
            "      Galerkin { [ -SourceCurrentDensity[], {a} ];",
            "                 In DomainWithSourceCurrentDensity; Jacobian Vol; Integration Int; }",
            "    }",
            "  }",
            "}",
            "",
            "Resolution {",
            "  { Name Magnetostatics;",
            "    System {",
            "      { Name Sys_Mag; NameOfFormulation Magnetostatics_a_3D; }",
            "    }",
            "    Operation {",
            "      Generate[Sys_Mag]; Solve[Sys_Mag]; SaveSolution[Sys_Mag];",
            "    }",
            "  }",
            "}",
            "",
            "PostProcessing {",
            "  { Name MagnetostaticFields; NameOfFormulation Magnetostatics_a_3D;",
            "    PostQuantity {",
            "      { Name b_uT; Value {",
            "          Term { [ 1.e6 * {d a} ]; In Domain; Jacobian Vol; }",
            "      } }",
            "      { Name js_A_m2; Value {",
            "          Term { [ SourceCurrentDensity[] ];",
            "                 In DomainWithSourceCurrentDensity; Jacobian Vol; }",
            "      } }",
            "    }",
            "  }",
            "}",
            "",
            "PostOperation {",
            "  { Name ExportFields; NameOfPostProcessing MagnetostaticFields;",
            "    Operation {",
            "      CreateDir[ResultDir];",
            '      Print[ b_uT, OnElementsOf Domain, File StrCat[ResultDir, "B_uT.pos"] ];',
            "      Print[ js_A_m2, OnElementsOf DomainWithSourceCurrentDensity,",
            '             File StrCat[ResultDir, "J_A_per_m2.pos"] ];',
        ]
    )
    for index, sample in enumerate(samples):
        operator = ">" if index == 0 else ">>"
        point = sample.point_m
        lines.append(
            "      Print[ b_uT, OnPoint "
            f"{{{_number(point[0])}, {_number(point[1])}, {_number(point[2])}}}, "
            "Smoothing, Format Table, File "
            f'{operator} StrCat[ResultDir, "samples.txt"] ];'
        )
    lines.extend(
        [
            "    }",
            "  }",
            "}",
            "",
            "DefineConstant[",
            '  R_ = {"Magnetostatics", Name "GetDP/1ResolutionChoices", Visible 0},',
            '  C_ = {"-solve Magnetostatics -ksp_error_if_not_converged -pos ExportFields",',
            '        Name "GetDP/9ComputeCommand", Visible 0},',
            '  P_ = {"ExportFields", Name "GetDP/2PostOperationChoices", Visible 0}',
            "];",
            "",
        ]
    )
    return "\n".join(lines)


def _render_samples_csv(samples: list[_SamplePoint]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "index",
            "label",
            "x_m",
            "y_m",
            "z_m",
            "reference_Bx_uT",
            "reference_By_uT",
            "reference_Bz_uT",
            "reference_kind",
        ]
    )
    for index, sample in enumerate(samples, start=1):
        reference_values = (
            ["", "", ""]
            if sample.reference_b_ut is None
            else [_number(value) for value in sample.reference_b_ut]
        )
        writer.writerow(
            [
                index,
                sample.label,
                *[_number(value) for value in sample.point_m],
                *reference_values,
                sample.reference_kind,
            ]
        )
    return stream.getvalue()


def _render_readme(
    sources: list[_Source],
    stem: str,
    samples: list[_SamplePoint],
    *,
    apply_field_scale: bool,
    use_physical_construction: bool,
    mesh_settings: GetDPMeshSettings,
    mesh_plan: _MeshPlan,
) -> str:
    source_blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        field_scale_note = (
            f"APPLIED ({source.field_scale_factor:.9g})"
            if apply_field_scale
            else f"not applied (saved value {source.field_scale_factor:.9g})"
        )
        source_blocks.append(
            "\n".join(
                [
                    f"Source {index}: {source.label} ({source.object_id})",
                    f"  Shape: {source.shape}",
                    f"  Coil representation: {source.source_geometry_kind}",
                    f"  Turns: {source.turns}",
                    f"  Saved signed drive current: {source.drive_current_a:.12g} A",
                    f"  Current convention metadata: {source.current_mode}",
                    f"  Exported signed ampere-turns: {source.ampere_turns_a:.12g} A.turn",
                    f"  Empirical Field correction: {field_scale_note}",
                    f"  Homogenized source cross-section: {source.cross_section_area_m2:.12g} m^2",
                    f"  Uniform tangential source-current density: {source.current_density_a_m2:.12g} A/m^2",
                ]
            )
        )
    source_text = "\n\n".join(source_blocks)
    turns_note = ", ".join(str(source.turns) for source in sources)
    return f"""Field Workbench runnable Gmsh/GetDP project
================================================

Compatibility target
--------------------
Gmsh {GETDP_MIN_GMSH_VERSION} or later
GetDP {GETDP_MIN_SOLVER_VERSION} or later

Model
-----
Physics: linear 3-D magnetostatics in free space
Formulation: magnetic vector potential A with an H(curl) edge space and tree gauge
Stranded-current source count: {len(sources)}
Source-geometry policy: {'use saved physical winding construction when available' if use_physical_construction else 'force numerical winding surrogates'}

Mesh / open-boundary choices
----------------------------
Preset: {mesh_settings.preset}
Approx. winding elements across the smallest pack dimension: {mesh_settings.source_elements:.6g}
Near-region divisions per largest source characteristic radius: {mesh_settings.near_radius_divisions:.6g}
Far-air divisions per automatic ordinary-air radius: {mesh_settings.far_radius_divisions:.6g}
Open boundary: automatic spherical infinite-element shell (GetDP VolSphShell)
Derived source characteristic length: {1000.0 * mesh_plan.source_mesh_m:.6g} mm
Derived near/sample characteristic length: {1000.0 * mesh_plan.near_mesh_m:.6g} mm
Derived far-air characteristic length: {1000.0 * mesh_plan.far_mesh_m:.6g} mm
Derived infinite-shell characteristic length: {1000.0 * mesh_plan.shell_mesh_m:.6g} mm
Automatic source-to-air grading: {'enabled, air-interface minimum ' + format(1000.0 * mesh_plan.source_grading_air_min_m, '.6g') + ' mm through ' + format(1000.0 * mesh_plan.source_grading_end_m, '.6g') + ' mm' if mesh_plan.source_grading_enabled else 'not required for this preset'}
3-D mesher: {'HXT (system thread count)' if _use_hxt_mesher(mesh_settings, mesh_plan) else 'Delaunay (established Fast/Standard path)'}
Automatic ordinary-air radius: {1000.0 * mesh_plan.inner_air_radius_m:.6g} mm
Automatic infinite-shell outer radius: {1000.0 * mesh_plan.shell_outer_radius_m:.6g} mm

{source_text}

Each winding region is a stranded-current homogenization. Integrating J across
its radial/axial cross-section gives that coil's exported NI; the turns
({turns_note}) are not separately meshed wires. Circular sources use annular
volumes with azimuthal J. Square/rectangular sources use exact mitered rectangular
path-band volumes with a piecewise tangent that conserves current through each
corner. Parametric racetracks use one continuous capsule-band volume with a
unit-magnitude path tangent through the straight sides and semicircular ends. The finite source cross-sections will differ slightly from
Field Workbench's ideal centrelines.

Verification samples
--------------------
results/samples.txt uses GetDP's nodal Smoothing option before interpolation at
the requested points. This is intentional: the lowest-order H(curl) solution has
an element-local curl, so raw OnPoint B values can show avoidable tetrahedron-to-
tetrahedron noise. samples.csv retains the ideal circular-loop axis reference
where applicable and FWB centreline Biot-Savart references for Square/rectangular
and parametric racetrack sources.

Run on Linux
------------
    sh run.sh

Run on Windows Command Prompt
-----------------------------
    run.bat

Equivalent commands
-------------------
    gmsh "{stem}.geo" -3 {'-nt 0 ' if _use_hxt_mesher(mesh_settings, mesh_plan) else ''}-format msh2 -o "{stem}.msh"
    getdp "{stem}.pro" -msh "{stem}.msh" -solve Magnetostatics -ksp_error_if_not_converged
    getdp "{stem}.pro" -msh "{stem}.msh" -pos ExportFields

Outputs
-------
{stem}.msh                 ASCII Gmsh MSH 2.2 mesh generated locally
{stem}.res                 GetDP solution
results/B_uT.pos           magnetic flux density in microtesla
results/J_A_per_m2.pos     source current density in A/m^2
results/samples.txt        smoothed GetDP table at {len(samples)} requested sample points
samples.csv                requested coordinates and available centreline references

Open the mesh and result maps together in Gmsh after the run:
    gmsh "{stem}.msh" "results/B_uT.pos" "results/J_A_per_m2.pos"

Scope and assumptions
---------------------
* Version {GETDP_PROJECT_VERSION} accepts up to 64 circular, Square/rectangular, and/or parametric
  racetrack coils plus optional Linear axis sensors.
* Air and every homogenized winding use permeability mu0.
* Conductivity, eddy currents, nonlinear materials, and time/frequency-domain
  physics remain outside this magnetostatic exporter.
* FWB automatically encloses every source and requested sample in an ordinary-air sphere,
  then surrounds it with a finite spherical shell. GetDP applies VolSphShell to that shell,
  mapping its outer A=0 boundary to infinity instead of approximating free space with a very
  large finite sphere. The derived radii and mesh lengths remain inspectable in {stem}.geo and
  {stem}.pro.
* Validation-scale winding meshes automatically use a Gmsh Distance/Threshold field around
  winding surfaces. The winding keeps its fine source target, while the surrounding air starts
  at a separately derived interface size and grows smoothly into the near-air size. Validation
  also selects Gmsh HXT for 3-D tetrahedralization and lets HXT use the system thread count; HXT
  is Gmsh's parallel Delaunay implementation and handles strongly graded large meshes more
  efficiently. Fast and Standard keep their established Delaunay mesh behaviour.
* Positive current follows each saved Field Workbench loop orientation; negative
  current reverses that source's J and B.

Provenance
----------
scene.json is the normalized portable scene used by the translator.
project.json records the GetDP-project schema and export choices.
No Gmsh or GetDP solver binaries are included.
"""


def _render_run_sh(stem: str, *, use_hxt: bool = False) -> str:
    return f"""#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

command -v gmsh >/dev/null 2>&1 || {{ echo "gmsh was not found on PATH" >&2; exit 127; }}
command -v getdp >/dev/null 2>&1 || {{ echo "getdp was not found on PATH" >&2; exit 127; }}

echo "Meshing {stem}.geo..."
gmsh "{stem}.geo" -3 {'-nt 0 ' if use_hxt else ''}-format msh2 -o "{stem}.msh"
echo "Solving magnetostatic project..."
getdp "{stem}.pro" -msh "{stem}.msh" -solve Magnetostatics -ksp_error_if_not_converged
echo "Writing field maps and samples..."
getdp "{stem}.pro" -msh "{stem}.msh" -pos ExportFields
echo "Complete. Open {stem}.msh and results/B_uT.pos in Gmsh."
"""


def _render_run_bat(stem: str, *, use_hxt: bool = False) -> str:
    text = f"""@echo off
setlocal
cd /d "%~dp0"

where gmsh >nul 2>nul
if errorlevel 1 (
  echo gmsh was not found on PATH
  exit /b 127
)
where getdp >nul 2>nul
if errorlevel 1 (
  echo getdp was not found on PATH
  exit /b 127
)

echo Meshing {stem}.geo...
gmsh "{stem}.geo" -3 {'-nt 0 ' if use_hxt else ''}-format msh2 -o "{stem}.msh"
if errorlevel 1 exit /b %errorlevel%
echo Solving magnetostatic project...
getdp "{stem}.pro" -msh "{stem}.msh" -solve Magnetostatics -ksp_error_if_not_converged
if errorlevel 1 exit /b %errorlevel%
echo Writing field maps and samples...
getdp "{stem}.pro" -msh "{stem}.msh" -pos ExportFields
if errorlevel 1 exit /b %errorlevel%
echo Complete. Open {stem}.msh and results\\B_uT.pos in Gmsh.
"""
    return text.replace("\n", "\r\n")


def _validated_scene_coils(scene: dict[str, Any]) -> list[dict[str, Any]]:
    if scene.get("format") != "field_workbench.portable_scene":
        raise StudioOperationError(
            "GetDP translator expected a Field Workbench portable scene."
        )
    coils = [
        obj
        for obj in scene.get("objects", [])
        if str(obj.get("kind", "")) == "coil"
    ]
    unsupported_coils = [
        obj
        for obj in coils
        if str(obj.get("geometry", {}).get("shape", "")) not in {"circle", "racetrack", "square"}
    ]
    if unsupported_coils:
        raise StudioOperationError(
            f"GetDP project v{GETDP_PROJECT_VERSION} supports circular, Square/rectangular, and parametric racetrack coils only."
        )
    if not 1 <= len(coils) <= GETDP_MAX_COIL_SOURCES:
        raise StudioOperationError(
            f"GetDP project v{GETDP_PROJECT_VERSION} requires between 1 and "
            f"{GETDP_MAX_COIL_SOURCES} selected circular/Square/racetrack coils."
        )
    return coils


def getdp_export_plan(
    scene: dict[str, Any],
    *,
    apply_field_scale: bool = False,
    use_physical_construction: bool = True,
    mesh_preset: str = "standard",
    source_elements: float | None = None,
    near_radius_divisions: float | None = None,
    far_radius_divisions: float | None = None,
) -> dict[str, Any]:
    """Return the concrete source/mesh choices that a GetDP export will use.

    This intentionally contains only JSON-friendly values so the Export dialog
    can show a live plan without reproducing exporter geometry logic.
    """
    coils = _validated_scene_coils(scene)
    settings = getdp_mesh_settings(
        mesh_preset,
        source_elements=source_elements,
        near_radius_divisions=near_radius_divisions,
        far_radius_divisions=far_radius_divisions,
    )
    sources = [
        _coil_source(
            coil,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
        )
        for coil in coils
    ]
    samples = _sample_points(scene, sources)
    mesh_plan = _build_mesh_plan(sources, samples, settings)
    physical_count = sum(
        "saved physical" in source.source_geometry_kind for source in sources
    )
    return {
        "mesh_preset": settings.preset,
        "source_elements": settings.source_elements,
        "near_radius_divisions": settings.near_radius_divisions,
        "far_radius_divisions": settings.far_radius_divisions,
        "source_mesh_mm": 1000.0 * mesh_plan.source_mesh_m,
        "near_mesh_mm": 1000.0 * mesh_plan.near_mesh_m,
        "far_mesh_mm": 1000.0 * mesh_plan.far_mesh_m,
        "shell_mesh_mm": 1000.0 * mesh_plan.shell_mesh_m,
        "source_grading_enabled": mesh_plan.source_grading_enabled,
        "source_grading_air_min_mm": 1000.0 * mesh_plan.source_grading_air_min_m,
        "source_grading_start_mm": 1000.0 * mesh_plan.source_grading_start_m,
        "source_grading_end_mm": 1000.0 * mesh_plan.source_grading_end_m,
        "inner_air_radius_mm": 1000.0 * mesh_plan.inner_air_radius_m,
        "shell_outer_radius_mm": 1000.0 * mesh_plan.shell_outer_radius_m,
        "shell_thickness_mm": 1000.0 * (mesh_plan.shell_outer_radius_m - mesh_plan.inner_air_radius_m),
        "open_boundary": "VolSphShell",
        "gmsh_mesher": "HXT" if _use_hxt_mesher(settings, mesh_plan) else "Delaunay",
        "gmsh_thread_policy": "system" if _use_hxt_mesher(settings, mesh_plan) else "default",
        "sample_count": len(samples),
        "source_geometry": [source.source_geometry_kind for source in sources],
        "physical_source_count": physical_count,
        "numerical_source_count": len(sources) - physical_count,
    }


def render_getdp_project(
    scene: dict[str, Any],
    *,
    project_stem: str = "field_workbench_model",
    apply_field_scale: bool = False,
    use_physical_construction: bool = True,
    mesh_preset: str = "standard",
    source_elements: float | None = None,
    near_radius_divisions: float | None = None,
    far_radius_divisions: float | None = None,
) -> dict[str, str]:
    """Return all text files for a runnable multi-source GetDP project."""
    coils = _validated_scene_coils(scene)
    mesh_settings = getdp_mesh_settings(
        mesh_preset,
        source_elements=source_elements,
        near_radius_divisions=near_radius_divisions,
        far_radius_divisions=far_radius_divisions,
    )

    sources = [
        _coil_source(
            coil,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
        )
        for coil in coils
    ]
    samples = _sample_points(scene, sources)
    mesh_plan = _build_mesh_plan(sources, samples, mesh_settings)
    stem = _project_stem(project_stem)
    project_manifest = {
        "format": GETDP_PROJECT_FORMAT,
        "version": GETDP_PROJECT_VERSION,
        "generator": scene.get("generator", {}),
        "compatibility": {
            "gmsh_minimum": GETDP_MIN_GMSH_VERSION,
            "getdp_minimum": GETDP_MIN_SOLVER_VERSION,
            "mesh_format": "2.2 ASCII",
        },
        "model": {
            "physics": "linear_3d_magnetostatics",
            "coil_count": len(sources),
            "coil_shapes": [source.shape for source in sources],
            "sample_count": len(samples),
            "apply_field_scale": bool(apply_field_scale),
            "use_physical_construction": bool(use_physical_construction),
            "source_geometry": [source.source_geometry_kind for source in sources],
            "sample_postprocessing": "nodal_smoothing",
            "mesh": {
                "preset": mesh_settings.preset,
                "gmsh_mesher": "HXT" if _use_hxt_mesher(mesh_settings, mesh_plan) else "Delaunay",
                "gmsh_thread_policy": "system" if _use_hxt_mesher(mesh_settings, mesh_plan) else "default",
                "controls": {
                    "source_elements": mesh_settings.source_elements,
                    "near_radius_divisions": mesh_settings.near_radius_divisions,
                    "far_radius_divisions": mesh_settings.far_radius_divisions,
                },
                "open_boundary": {
                    "type": "spherical_infinite_shell",
                    "getdp_jacobian": "VolSphShell",
                    "automatic": True,
                },
                "derived": {
                    "source_mesh_mm": 1000.0 * mesh_plan.source_mesh_m,
                    "near_mesh_mm": 1000.0 * mesh_plan.near_mesh_m,
                    "far_mesh_mm": 1000.0 * mesh_plan.far_mesh_m,
                    "shell_mesh_mm": 1000.0 * mesh_plan.shell_mesh_m,
                    "source_grading_enabled": mesh_plan.source_grading_enabled,
                    "source_grading_air_min_mm": 1000.0 * mesh_plan.source_grading_air_min_m,
                    "source_grading_start_mm": 1000.0 * mesh_plan.source_grading_start_m,
                    "source_grading_end_mm": 1000.0 * mesh_plan.source_grading_end_m,
                    "inner_air_radius_mm": 1000.0 * mesh_plan.inner_air_radius_m,
                    "shell_outer_radius_mm": 1000.0 * mesh_plan.shell_outer_radius_m,
                },
            },
        },
        "entry_points": {
            "geometry": f"{stem}.geo",
            "problem": f"{stem}.pro",
            "linux": "run.sh",
            "windows": "run.bat",
        },
    }
    return {
        f"{stem}.geo": _render_geo(sources, samples, mesh_plan, mesh_settings),
        f"{stem}.pro": _render_pro(sources, samples, mesh_plan),
        "run.sh": _render_run_sh(stem, use_hxt=_use_hxt_mesher(mesh_settings, mesh_plan)),
        "run.bat": _render_run_bat(stem, use_hxt=_use_hxt_mesher(mesh_settings, mesh_plan)),
        "README.txt": _render_readme(
            sources,
            stem,
            samples,
            apply_field_scale=apply_field_scale,
            use_physical_construction=use_physical_construction,
            mesh_settings=mesh_settings,
            mesh_plan=mesh_plan,
        ),
        "samples.csv": _render_samples_csv(samples),
        "scene.json": json.dumps(scene, indent=2, ensure_ascii=False) + "\n",
        "project.json": json.dumps(project_manifest, indent=2, ensure_ascii=False) + "\n",
    }


def _zip_text(
    archive: zipfile.ZipFile,
    name: str,
    content: str,
    *,
    executable: bool = False,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (0o100000 | mode) << 16
    archive.writestr(info, content.encode("utf-8"))


def write_getdp_project(
    scene: dict[str, Any],
    filename: str | Path,
    *,
    apply_field_scale: bool = False,
    use_physical_construction: bool = True,
    mesh_preset: str = "standard",
    source_elements: float | None = None,
    near_radius_divisions: float | None = None,
    far_radius_divisions: float | None = None,
) -> Path:
    """Write a deterministic ``.getdp.zip`` project archive."""
    path = Path(filename)
    files = render_getdp_project(
        scene,
        project_stem=path.name,
        apply_field_scale=apply_field_scale,
        use_physical_construction=use_physical_construction,
        mesh_preset=mesh_preset,
        source_elements=source_elements,
        near_radius_divisions=near_radius_divisions,
        far_radius_divisions=far_radius_divisions,
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            _zip_text(archive, name, content, executable=name == "run.sh")
    return path
