"""Reusable magnetometer-probe definitions for Field Workbench.

Probe definitions are deliberately plain JSON-compatible dictionaries.  A scene
stores a validated copy of the selected definition, so it remains reproducible
even when the application library changes later.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


PROBE_DEFINITION_SCHEMA = "field-workbench-magnetometer-probe"
PROBE_DEFINITION_VERSION = 1


class ProbeDefinitionError(ValueError):
    """Raised when a reusable probe definition is malformed."""


SENSYS_FGM3D_STANDARD: dict[str, Any] = {
    "schema": PROBE_DEFINITION_SCHEMA,
    "version": PROBE_DEFINITION_VERSION,
    "id": "sensys-fgm3d-standard-v3",
    "manufacturer": "SENSYS GmbH",
    "model": "FGM3D",
    "display_name": "SENSYS FGM3D",
    "description": (
        "Standard rectangular three-axis fluxgate magnetometer. The three "
        "single-axis sensing elements are separated along the housing length."
    ),
    "origin_description": (
        "Geometric centre of the documented 26 x 26 x 149 mm housing envelope, "
        "so the scene position can be indexed directly from the probe body."
    ),
    "coordinate_system": (
        "Local +Z follows the arrow along the housing toward the marked reference "
        "edge; local +X follows the sticker arrow and local +Y points inward through "
        "the sticker face, as shown by the manufacturer axis symbols."
    ),
    "verification_status": "manufacturer-derived",
    "housing": {
        "shape": "box",
        # The first draft intentionally uses the documented overall rectangular
        # envelope. Small connector/end-cap details are not represented.
        "dimensions_mm": [26.0, 26.0, 149.0],
        "center_mm": [0.0, 0.0, 0.0],
        "sticker_face": "-Y",
        "sticker_label": "STICKER SIDE",
        "cable_face": "-Z",
        "colour": "#475569",
        "opacity": 0.58,
        "geometry_note": (
            "Documented 26 x 26 x 149 mm overall envelope; connector and fastener "
            "details are simplified."
        ),
    },
    "channels": [
        {
            "name": "X",
            "output": "Bx",
            "position_mm": [0.0, 0.0, 20.0],
            "sensitive_axis": [1.0, 0.0, 0.0],
            "colour": "#dc2626",
            "reference_from_edge_mm": 54.5,
        },
        {
            "name": "Y",
            "output": "By",
            "position_mm": [0.0, 0.0, 60.0],
            "sensitive_axis": [0.0, 1.0, 0.0],
            "colour": "#16a34a",
            "reference_from_edge_mm": 14.5,
        },
        {
            "name": "Z",
            "output": "Bz",
            "position_mm": [0.0, 0.0, 40.0],
            "sensitive_axis": [0.0, 0.0, 1.0],
            "colour": "#2563eb",
            "reference_from_edge_mm": 34.5,
        },
    ],
    "sources": [
        {
            "title": "SENSYS FGM3D product page",
            "url": (
                "https://sensysmagnetometer.com/products/"
                "fgm3d-fluxgate-magnetometer-sensys/"
            ),
            "kind": "manufacturer-product-page",
        },
        {
            "title": "SENSYS FGM3D Product Information, version 1.03",
            "url": "https://my.hidrive.com/lnk/uHHoPFSB",
            "kind": "manufacturer-drawing",
            "note": (
                "Housing Figure 1 and parameter matrix: 26 x 26 x 149 mm; "
                "single-axis reference distances 14.5/34.5/54.5 mm."
            ),
        },
    ],
}


MEDA_FM300_FVM400: dict[str, Any] = {
    "schema": PROBE_DEFINITION_SCHEMA,
    "version": PROBE_DEFINITION_VERSION,
    "id": "meda-fm300-fvm400-shared-probe-v1",
    "manufacturer": "MEDA, Inc.",
    "model": "FM300 / FVM400",
    "display_name": "MEDA FM300 / FVM400 probe",
    "description": (
        "Rectangular three-axis fluxgate probe used by the FM300 and its enhanced "
        "FVM400 successor. The Y and Z windings share one ring-core centre while "
        "the X winding uses a second centre. Geometry and axes follow MEDA's FVM400 "
        "manual; the FM300/FVM400 physical-probe equivalence is owner-confirmed for "
        "this first-draft entry."
    ),
    "origin_description": (
        "Geometric centre of the documented 4.00 x 1.00 x 1.00 inch probe body, "
        "so the scene position can be indexed directly from the housing."
    ),
    "coordinate_system": (
        "Manufacturer base-referenced right-handed frame: local +X runs lengthwise "
        "from the cable/base end toward the probe tip, local +Y runs across the "
        "housing width as shown in Figure 3, and local +Z points down through the "
        "housing toward and out of the base surface. The top/label face is -Z."
        " Channel reference distances are measured inward from the +X tip."
    ),
    "verification_status": "manufacturer-derived",
    "housing": {
        "shape": "box",
        "dimensions_mm": [101.6, 25.4, 25.4],
        "center_mm": [0.0, 0.0, 0.0],
        "sticker_face": "-Z",
        "sticker_label": "TOP / LABEL SIDE",
        "cable_face": "-X",
        "colour": "#334155",
        "opacity": 0.58,
        "geometry_note": (
            "Figure 5 documents a 4.00 x 1.00 x 1.00 inch rectangular envelope, "
            "represented here as 101.6 x 25.4 x 25.4 mm. The General Specifications "
            "table prints 100.6 mm beside 4 inches; this model follows the explicit "
            "4.00 inch drawing dimension. Connector and case details are simplified."
        ),
    },
    "channels": [
        {
            "name": "X",
            "output": "Bx",
            # Figure 5: 1.54 inches inward from the +X tip and 0.49 inches
            # below the -Z top face, expressed about the body-centred origin.
            "position_mm": [11.684, 0.0, -0.254],
            "sensitive_axis": [1.0, 0.0, 0.0],
            "colour": "#dc2626",
            "reference_from_edge_mm": 39.116,
        },
        {
            "name": "Y",
            "output": "By",
            # The Y and Z windings share the ring-core centre 0.68 inches
            # inward from the +X tip.
            "position_mm": [33.528, 0.0, -0.254],
            "sensitive_axis": [0.0, 1.0, 0.0],
            "colour": "#16a34a",
            "reference_from_edge_mm": 17.272,
        },
        {
            "name": "Z",
            "output": "Bz",
            "position_mm": [33.528, 0.0, -0.254],
            "sensitive_axis": [0.0, 0.0, 1.0],
            "colour": "#2563eb",
            "reference_from_edge_mm": 17.272,
        },
    ],
    "sources": [
        {
            "title": "MEDA FVM400 product page",
            "url": "https://www.meda.com/index.php/products?name=FVM400",
            "kind": "manufacturer-product-page",
            "note": "Describes the fluxgate sensors in a 1 inch square by 4 inch long probe.",
        },
        {
            "title": "MEDA FVM400 Instruction Manual, Revision A (January 2018)",
            "url": (
                "https://www.meda.com/pdf/"
                "FVM400%20Instruction%20Manual%20Rev%20A.pdf"
            ),
            "kind": "manufacturer-manual",
            "note": (
                "Figure 3 defines the base-referenced axes; Figure 5 documents the "
                "housing and sensor-centre offsets."
            ),
        },
        {
            "title": "MEDA company history",
            "url": "https://www.meda.com/index.php/pages/about",
            "kind": "manufacturer-history",
            "note": "Identifies the FVM400 as the enhanced replacement for the FM300.",
        },
    ],
}


BUILTIN_PROBE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    SENSYS_FGM3D_STANDARD,
    MEDA_FM300_FVM400,
)


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ProbeDefinitionError(f"{label} must contain {length} numbers.") from error
    if len(result) != length or not all(math.isfinite(component) for component in result):
        raise ProbeDefinitionError(f"{label} must contain {length} finite numbers.")
    return result


def _colour(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    if len(text) != 7 or not text.startswith("#"):
        raise ProbeDefinitionError("Probe colours must use #RRGGBB notation.")
    try:
        int(text[1:], 16)
    except ValueError as error:
        raise ProbeDefinitionError("Probe colours must use #RRGGBB notation.") from error
    return text


def validate_probe_definition(value: Any) -> dict[str, Any]:
    """Return a canonical, detached, JSON-compatible probe definition."""
    if not isinstance(value, dict):
        raise ProbeDefinitionError("A probe definition must be a JSON object.")
    schema = str(value.get("schema", PROBE_DEFINITION_SCHEMA)).strip()
    if schema != PROBE_DEFINITION_SCHEMA:
        raise ProbeDefinitionError(f"Unsupported probe-definition schema: {schema}")
    try:
        version = int(value.get("version", PROBE_DEFINITION_VERSION))
    except (TypeError, ValueError) as error:
        raise ProbeDefinitionError("Probe-definition version must be an integer.") from error
    if version != PROBE_DEFINITION_VERSION:
        raise ProbeDefinitionError(
            f"Unsupported probe-definition version {version}; expected {PROBE_DEFINITION_VERSION}."
        )

    manufacturer = str(value.get("manufacturer", "")).strip()
    model = str(value.get("model", "")).strip()
    if not manufacturer:
        raise ProbeDefinitionError("Probe manufacturer is required.")
    if not model:
        raise ProbeDefinitionError("Probe model is required.")
    definition_id = str(value.get("id", "")).strip()
    if not definition_id:
        definition_id = f"custom-{manufacturer}-{model}".lower()
        definition_id = "-".join(part for part in definition_id.replace("_", "-").split() if part)

    housing_value = value.get("housing", {})
    if not isinstance(housing_value, dict):
        raise ProbeDefinitionError("Probe housing must be an object.")
    shape = str(housing_value.get("shape", "box")).strip().lower()
    if shape != "box":
        raise ProbeDefinitionError("Probe definition version 1 supports box housings.")
    dimensions_mm = _finite_vector(
        housing_value.get("dimensions_mm", [10.0, 10.0, 50.0]), 3, "Housing dimensions"
    )
    if any(component <= 0.0 for component in dimensions_mm):
        raise ProbeDefinitionError("Housing dimensions must be positive.")
    center_mm = _finite_vector(
        housing_value.get("center_mm", [0.0, 0.0, 0.0]), 3, "Housing centre"
    )
    sticker_face = str(housing_value.get("sticker_face", "")).strip().upper()
    if sticker_face not in {"", "+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise ProbeDefinitionError(
            "Sticker face must be None, +X, -X, +Y, -Y, +Z, or -Z."
        )
    sticker_label = str(housing_value.get("sticker_label", "STICKER SIDE")).strip()
    cable_face = str(housing_value.get("cable_face", "")).strip().upper()
    if cable_face not in {"", "+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise ProbeDefinitionError(
            "Cable connection face must be None, +X, -X, +Y, -Y, +Z, or -Z."
        )
    try:
        opacity = float(housing_value.get("opacity", 0.58))
    except (TypeError, ValueError) as error:
        raise ProbeDefinitionError("Housing opacity must be numeric.") from error
    if not math.isfinite(opacity) or not 0.05 <= opacity <= 1.0:
        raise ProbeDefinitionError("Housing opacity must be between 0.05 and 1.0.")

    raw_channels = value.get("channels", [])
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ProbeDefinitionError("A probe must define at least one measurement channel.")
    channels: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    defaults = ("#dc2626", "#16a34a", "#2563eb", "#9333ea", "#0891b2")
    for index, raw in enumerate(raw_channels):
        if not isinstance(raw, dict):
            raise ProbeDefinitionError(f"Probe channel {index + 1} must be an object.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ProbeDefinitionError(f"Probe channel {index + 1} requires a name.")
        if name.casefold() in seen_names:
            raise ProbeDefinitionError(f"Probe channel names must be unique: {name}")
        seen_names.add(name.casefold())
        position_mm = _finite_vector(raw.get("position_mm", [0.0, 0.0, 0.0]), 3, f"{name} position")
        axis = _finite_vector(raw.get("sensitive_axis", [0.0, 0.0, 1.0]), 3, f"{name} sensitive axis")
        magnitude = math.sqrt(sum(component * component for component in axis))
        if magnitude <= 1e-12:
            raise ProbeDefinitionError(f"{name} sensitive axis cannot be zero.")
        axis = [component / magnitude for component in axis]
        channel = {
            "name": name,
            "output": str(raw.get("output", f"B{name}")).strip() or f"B{name}",
            "position_mm": position_mm,
            "sensitive_axis": axis,
            "colour": _colour(raw.get("colour"), defaults[index % len(defaults)]),
        }
        if raw.get("reference_from_edge_mm") is not None:
            try:
                reference = float(raw["reference_from_edge_mm"])
            except (TypeError, ValueError) as error:
                raise ProbeDefinitionError(f"{name} reference distance must be numeric.") from error
            if not math.isfinite(reference):
                raise ProbeDefinitionError(f"{name} reference distance must be finite.")
            channel["reference_from_edge_mm"] = reference
        channels.append(channel)

    raw_sources = value.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ProbeDefinitionError("Probe sources must be a list.")
    sources: list[dict[str, str]] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not title and not url:
            continue
        if url and not url.lower().startswith(("https://", "http://")):
            raise ProbeDefinitionError("Probe source links must use http:// or https://.")
        source = {"title": title or url, "url": url}
        for key in ("kind", "note"):
            if raw.get(key):
                source[key] = str(raw[key])
        sources.append(source)

    return {
        "schema": PROBE_DEFINITION_SCHEMA,
        "version": PROBE_DEFINITION_VERSION,
        "id": definition_id,
        "manufacturer": manufacturer,
        "model": model,
        "display_name": str(value.get("display_name", f"{manufacturer} {model}")).strip()
        or f"{manufacturer} {model}",
        "description": str(value.get("description", "")).strip(),
        "origin_description": str(value.get("origin_description", "Probe local origin.")).strip(),
        "coordinate_system": str(value.get("coordinate_system", "Probe-local right-handed XYZ.")).strip(),
        "verification_status": str(value.get("verification_status", "custom")).strip().lower() or "custom",
        "housing": {
            "shape": shape,
            "dimensions_mm": dimensions_mm,
            "center_mm": center_mm,
            "colour": _colour(housing_value.get("colour"), "#475569"),
            "opacity": opacity,
            "geometry_note": str(housing_value.get("geometry_note", "")).strip(),
            **({"sticker_face": sticker_face, "sticker_label": sticker_label or "STICKER SIDE"}
               if sticker_face else {}),
            **({"cable_face": cable_face} if cable_face else {}),
        },
        "channels": channels,
        "sources": sources,
    }


def builtin_probe_definitions() -> list[dict[str, Any]]:
    """Return validated detached copies of every bundled definition."""
    return [validate_probe_definition(copy.deepcopy(item)) for item in BUILTIN_PROBE_DEFINITIONS]


def load_probe_definition(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeDefinitionError(f"Could not read probe definition: {error}") from error
    return validate_probe_definition(value)


def save_probe_definition(path: str | Path, definition: dict[str, Any]) -> None:
    normalized = validate_probe_definition(definition)
    try:
        Path(path).write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ProbeDefinitionError(f"Could not save probe definition: {error}") from error
