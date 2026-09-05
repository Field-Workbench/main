"""Small, deliberately constrained interchange format for AI/manual scene creation.

The native ``.magpy.json`` scene remains authoritative. A Scene Recipe is a
human-reviewed interchange layer for Add/Replace construction: it cannot execute
code, read files, issue patch operations, or directly mutate existing objects.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any


RECIPE_FORMAT = "fieldworkbench-scene-recipe"
RECIPE_VERSION = 1

SUPPORTED_TYPES: tuple[str, ...] = (
    "group",
    "circular_coil",
    "racetrack_coil",
    "square_coil",
    "point_sensor",
    "axis_sensor",
    "box",
    "cylinder",
    "sphere",
    "block_magnet",
    "cylinder_magnet",
    "sphere_magnet",
    "generic_bust",
    "sri24_brain",
)

_COMMON_FIELDS = {
    "key",
    "type",
    "name",
    "parent",
    "visible",
    "position_mm",
    "rotation_deg",
}

_COIL_FIELDS = {
    "turns",
    "drive_current_mA",
    "awg",
    "enabled",
    "field_scale_factor",
    "current_mode",
    "lead_length_mm",
    "resistance_20_ohm_per_km",
    "enamel_radial_mm",
    "packing_factor",
    "temperature_c",
    "additional_mass_g",
    "construction",
}

_TYPE_FIELDS: dict[str, set[str]] = {
    "group": set(),
    "circular_coil": {"diameter_mm", "coil"} | _COIL_FIELDS,
    "racetrack_coil": {"end_diameter_mm", "straight_length_mm", "arc_segments", "coil"} | _COIL_FIELDS,
    "square_coil": {"side_mm", "coil"} | _COIL_FIELDS,
    "point_sensor": set(),
    "axis_sensor": {"start_mm", "end_mm", "samples"},
    "box": {"dimensions_mm", "role", "colour", "opacity", "clearance_mm"},
    "cylinder": {"diameter_mm", "height_mm", "role", "colour", "opacity", "clearance_mm"},
    "sphere": {"diameter_mm", "role", "colour", "opacity", "clearance_mm"},
    "block_magnet": {"dimensions_mm", "polarization_T"},
    "cylinder_magnet": {"diameter_mm", "height_mm", "polarization_T"},
    "sphere_magnet": {"diameter_mm", "polarization_T"},
    "generic_bust": set(),
    "sri24_brain": set(),
}

_CONSTRUCTION_FIELDS = {
    "enabled",
    "axial_width_mm",
    "radial_build_mode",
    "radial_build_mm",
    "bobbin_wall_thickness_mm",
    "flange_height_mm",
    "flange_thickness_mm",
}

_HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\Z")
_KEY = re.compile(r"[A-Za-z0-9_.-]+\Z")
_TOP_LEVEL_FIELDS = {
    "format",
    "version",
    "assistant_message",
    "assumptions",
    "suggested_next_steps",
    "requires_clarification",
    "clarifying_questions",
    "review_questions",
    "objects",
}

GUIDANCE_LEVELS: tuple[str, ...] = ("exploratory", "balanced", "strict")


class SceneRecipeError(ValueError):
    """Raised when pasted Scene Recipe JSON is malformed or unsupported."""


def _where(index: int | None, field: str | None = None) -> str:
    root = "recipe" if index is None else f"objects[{index}]"
    return f"{root}.{field}" if field else root


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneRecipeError(f"{path}: expected a JSON object.")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SceneRecipeError(f"{path}: expected true or false.")
    return bool(value)


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneRecipeError(f"{path}: expected a number.")
    result = float(value)
    if not math.isfinite(result):
        raise SceneRecipeError(f"{path}: value must be finite.")
    if positive and result <= 0:
        raise SceneRecipeError(f"{path}: value must be positive.")
    if minimum is not None and result < minimum:
        raise SceneRecipeError(f"{path}: value must be at least {minimum:g}.")
    if maximum is not None and result > maximum:
        raise SceneRecipeError(f"{path}: value must be no greater than {maximum:g}.")
    return result


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneRecipeError(f"{path}: expected an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SceneRecipeError(f"{path}: value must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise SceneRecipeError(f"{path}: value must be no greater than {maximum}.")
    return result


def _vector(value: Any, path: str, *, length: int = 3) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SceneRecipeError(f"{path}: expected an array of {length} numbers.")
    return [_number(item, f"{path}[{i}]") for i, item in enumerate(value)]


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SceneRecipeError(f"{path}: expected text.")
    result = value.strip()
    if not allow_empty and not result:
        raise SceneRecipeError(f"{path}: value cannot be empty.")
    return result


def _unknown_keys(source: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise SceneRecipeError(
            f"{path}: unsupported field(s): {', '.join(unknown)}. "
            "Remove fields that are not in the Scene Recipe guide."
        )


def _string_list(value: Any, path: str, *, maximum_items: int = 20) -> list[str]:
    if not isinstance(value, list):
        raise SceneRecipeError(f"{path}: expected an array of text strings.")
    if len(value) > maximum_items:
        raise SceneRecipeError(f"{path}: use at most {maximum_items} entries.")
    return [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _flatten_nested_coil_alias(
    source: dict[str, Any],
    object_type: str,
    index: int,
    warnings: list[str],
) -> dict[str, Any]:
    """Accept the common AI mistake of copying current-scene ``coil`` context.

    Current-scene summaries deliberately group coil inputs and derived values under
    ``coil`` for readability.  Scene Recipe v1 canonically keeps editable coil
    fields at the object top level.  External LLMs naturally mirror the context
    structure, so tolerate that wrapper and normalize it rather than rejecting an
    otherwise useful recipe.
    """
    if object_type not in {"circular_coil", "racetrack_coil", "square_coil"} or "coil" not in source:
        return source

    result = copy.deepcopy(source)
    nested = _object(result.pop("coil"), _where(index, "coil"))
    shape_fields = {
        "circular_coil": {"diameter_mm"},
        "racetrack_coil": {"end_diameter_mm", "straight_length_mm", "arc_segments"},
        "square_coil": {"side_mm"},
    }[object_type]
    allowed = _COIL_FIELDS | shape_fields | {"workbench_derived"}
    _unknown_keys(nested, allowed, _where(index, "coil"))

    if "workbench_derived" in nested:
        warnings.append(
            f"objects[{index}] {result.get('name', object_type)!r}: ignored nested "
            "coil.workbench_derived because Workbench-calculated values are context-only."
        )

    copied: list[str] = []
    for field, value in nested.items():
        if field == "workbench_derived":
            continue
        if field in result:
            if result[field] != value:
                raise SceneRecipeError(
                    f"{_where(index, field)}: conflicting values were supplied at the object "
                    f"top level and inside coil.{field}. Keep one unambiguous value."
                )
            continue
        result[field] = value
        copied.append(field)

    if copied:
        warnings.append(
            f"objects[{index}] {result.get('name', object_type)!r}: accepted nested coil "
            "compatibility wrapper and flattened its recipe fields to the object level."
        )
    return result


def parse_scene_recipe(text: str) -> dict[str, Any]:
    """Parse and strictly normalize one v1 recipe.

    Returns ``{"recipe": normalized_document, "warnings": [...]}``.
    """
    raw_text = str(text or "").strip()
    if not raw_text:
        raise SceneRecipeError("Paste a Field Workbench Scene Recipe before validating.")
    if raw_text.startswith("```"):
        raise SceneRecipeError(
            "Paste JSON only, without Markdown ``` fences. Ask the AI to return JSON only."
        )
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise SceneRecipeError(
            f"JSON parse error at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    document = _object(document, "recipe")
    _unknown_keys(document, _TOP_LEVEL_FIELDS, "recipe")
    if document.get("format") != RECIPE_FORMAT:
        raise SceneRecipeError(
            f"recipe.format: expected {RECIPE_FORMAT!r}."
        )
    version = document.get("version")
    if version != RECIPE_VERSION:
        raise SceneRecipeError(
            f"recipe.version: expected {RECIPE_VERSION}; received {version!r}."
        )
    advisory: dict[str, Any] = {}
    if "assistant_message" in document:
        advisory["assistant_message"] = _string(document["assistant_message"], "recipe.assistant_message")
    if "assumptions" in document:
        advisory["assumptions"] = _string_list(document["assumptions"], "recipe.assumptions")
    if "suggested_next_steps" in document:
        advisory["suggested_next_steps"] = _string_list(
            document["suggested_next_steps"], "recipe.suggested_next_steps"
        )
    if "review_questions" in document:
        advisory["review_questions"] = _string_list(
            document["review_questions"], "recipe.review_questions", maximum_items=20
        )
    requires_clarification = False
    if "requires_clarification" in document:
        requires_clarification = _bool(
            document["requires_clarification"], "recipe.requires_clarification"
        )
    clarifying_questions: list[str] = []
    if "clarifying_questions" in document:
        clarifying_questions = _string_list(
            document["clarifying_questions"], "recipe.clarifying_questions", maximum_items=20
        )
    if clarifying_questions and "requires_clarification" not in document:
        # Be forgiving of assistants that supplied the blocking questions but omitted
        # the companion flag. The normalized recipe still records the state clearly.
        requires_clarification = True
    if requires_clarification and not clarifying_questions:
        raise SceneRecipeError(
            "recipe.clarifying_questions: provide at least one concrete blocking question "
            "when requires_clarification is true."
        )
    advisory["requires_clarification"] = requires_clarification
    if clarifying_questions:
        advisory["clarifying_questions"] = clarifying_questions

    objects = document.get("objects")
    if not isinstance(objects, list):
        raise SceneRecipeError("recipe.objects: expected an array.")
    if len(objects) > 500:
        raise SceneRecipeError("recipe.objects: v1 accepts at most 500 objects per import.")
    if requires_clarification and objects:
        raise SceneRecipeError(
            "recipe.objects: must be empty when requires_clarification is true. "
            "Ask the blocking questions instead of returning speculative geometry."
        )

    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    keys: dict[str, int] = {}

    for index, raw in enumerate(objects):
        source = _object(raw, _where(index))
        object_type = _string(source.get("type"), _where(index, "type"))
        if object_type not in SUPPORTED_TYPES:
            raise SceneRecipeError(
                f"{_where(index, 'type')}: unsupported type {object_type!r}. "
                f"Supported types: {', '.join(SUPPORTED_TYPES)}."
            )
        source = _flatten_nested_coil_alias(source, object_type, index, warnings)
        _unknown_keys(source, _COMMON_FIELDS | _TYPE_FIELDS[object_type], _where(index))
        item: dict[str, Any] = {"type": object_type}

        if "key" in source:
            key = _string(source["key"], _where(index, "key"))
            if not _KEY.fullmatch(key):
                raise SceneRecipeError(
                    f"{_where(index, 'key')}: use only letters, numbers, _, -, or ."
                )
            if key in keys:
                raise SceneRecipeError(
                    f"{_where(index, 'key')}: duplicate key {key!r}; first used by objects[{keys[key]}]."
                )
            keys[key] = index
            item["key"] = key
        if "name" in source:
            item["name"] = _string(source["name"], _where(index, "name"))
        if "parent" in source:
            parent_value = source["parent"]
            # External LLMs commonly emit JSON null for an optional parent on
            # top-level objects. Treat that exactly like omission instead of
            # rejecting an otherwise valid recipe.
            if parent_value is None:
                warnings.append(
                    f"objects[{index}] {item.get('name', object_type)!r}: parent was null; "
                    "treated as a top-level object."
                )
            else:
                item["parent"] = _string(parent_value, _where(index, "parent"))
        if "visible" in source:
            item["visible"] = _bool(source["visible"], _where(index, "visible"))
        else:
            item["visible"] = True

        if object_type == "group":
            for field in ("position_mm", "rotation_deg"):
                if field not in source:
                    continue
                value = _vector(source[field], _where(index, field))
                # Scene Recipe v1 groups do not carry transforms, but LLMs often
                # emit explicit identity transforms for every object. Identity is
                # semantically equivalent to omission, so accept and discard it.
                if all(abs(component) <= 1e-12 for component in value):
                    warnings.append(
                        f"objects[{index}] {item.get('name', object_type)!r}: ignored identity "
                        f"{field} because group transforms are not part of Scene Recipe v1."
                    )
                    continue
                raise SceneRecipeError(
                    f"{_where(index, field)}: group transforms are not part of Scene Recipe v1; "
                    "omit this field or move/rotate the child objects instead."
                )
        elif object_type == "axis_sensor":
            if "position_mm" in source or "rotation_deg" in source:
                raise SceneRecipeError(
                    f"{_where(index)}: axis_sensor uses start_mm/end_mm rather than position_mm/rotation_deg."
                )
        else:
            item["position_mm"] = _vector(
                source.get("position_mm", [0.0, 0.0, 0.0]), _where(index, "position_mm")
            )
            item["rotation_deg"] = _vector(
                source.get("rotation_deg", [0.0, 0.0, 0.0]), _where(index, "rotation_deg")
            )

        if object_type in {"circular_coil", "racetrack_coil", "square_coil"}:
            item["turns"] = _integer(source.get("turns", 1), _where(index, "turns"), minimum=1, maximum=1_000_000)
            if "drive_current_mA" in source:
                item["drive_current_mA"] = _number(source["drive_current_mA"], _where(index, "drive_current_mA"))
            else:
                item["drive_current_mA"] = 0.0
                warnings.append(f"objects[{index}] {item.get('name', object_type)!r}: drive_current_mA omitted; using 0 mA.")
            item["awg"] = _integer(source.get("awg", 30), _where(index, "awg"), minimum=4, maximum=50)
            item["enabled"] = _bool(source.get("enabled", True), _where(index, "enabled"))
            item["field_scale_factor"] = _number(
                source.get("field_scale_factor", 1.0), _where(index, "field_scale_factor"), minimum=1e-6, maximum=1e6
            )
            current_mode = _string(source.get("current_mode", "rms_dc"), _where(index, "current_mode"))
            if current_mode not in {"rms_dc", "peak_sine"}:
                raise SceneRecipeError(f"{_where(index, 'current_mode')}: use 'rms_dc' or 'peak_sine'.")
            item["current_mode"] = current_mode
            item["lead_length_mm"] = _number(source.get("lead_length_mm", 200.0), _where(index, "lead_length_mm"), minimum=0.0)
            # These are editable winding-model inputs, not calculated outputs.
            # Keeping them in v1 makes an AI Replace round-trip capable of
            # faithfully preserving a physically configured coil.
            if "resistance_20_ohm_per_km" in source:
                resistance_value = source["resistance_20_ohm_per_km"]
                # AI-generated recipes sometimes use 0/null to mean
                # "no custom override". Zero is not a physically valid
                # copper resistance, so normalize those sentinel-like values
                # to omission and let the selected AWG supply its nominal
                # 20 °C value. Negative/non-numeric values remain errors.
                if resistance_value is None or (
                    not isinstance(resistance_value, bool)
                    and isinstance(resistance_value, (int, float))
                    and float(resistance_value) == 0.0
                ):
                    warnings.append(
                        f"objects[{index}] {item.get('name', object_type)!r}: "
                        "resistance_20_ohm_per_km was 0/null; using nominal "
                        f"20 °C copper resistance for AWG {item['awg']}."
                    )
                else:
                    item["resistance_20_ohm_per_km"] = _number(
                        resistance_value,
                        _where(index, "resistance_20_ohm_per_km"),
                        positive=True,
                    )
            item["enamel_radial_mm"] = _number(
                source.get("enamel_radial_mm", 0.025),
                _where(index, "enamel_radial_mm"),
                minimum=0.0,
                maximum=2.0,
            )
            item["packing_factor"] = _number(
                source.get("packing_factor", 0.75),
                _where(index, "packing_factor"),
                minimum=0.10,
                maximum=0.95,
            )
            item["temperature_c"] = _number(
                source.get("temperature_c", 20.0),
                _where(index, "temperature_c"),
                minimum=-100.0,
                maximum=300.0,
            )
            item["additional_mass_g"] = _number(
                source.get("additional_mass_g", 0.0),
                _where(index, "additional_mass_g"),
                minimum=0.0,
            )
            construction = source.get("construction", {})
            construction = _object(construction, _where(index, "construction"))
            _unknown_keys(construction, _CONSTRUCTION_FIELDS, _where(index, "construction"))
            default_axial_width = 65.0 if object_type == "racetrack_coil" else 0.0
            c: dict[str, Any] = {
                "enabled": _bool(construction.get("enabled", False), _where(index, "construction.enabled")),
                "axial_width_mm": _number(construction.get("axial_width_mm", default_axial_width), _where(index, "construction.axial_width_mm"), minimum=0.0),
                "radial_build_mode": _string(construction.get("radial_build_mode", "auto"), _where(index, "construction.radial_build_mode")),
                "radial_build_mm": _number(construction.get("radial_build_mm", 0.0), _where(index, "construction.radial_build_mm"), minimum=0.0),
                "bobbin_wall_thickness_mm": _number(construction.get("bobbin_wall_thickness_mm", 0.0), _where(index, "construction.bobbin_wall_thickness_mm"), minimum=0.0),
                "flange_height_mm": _number(construction.get("flange_height_mm", 0.0), _where(index, "construction.flange_height_mm"), minimum=0.0),
                "flange_thickness_mm": _number(construction.get("flange_thickness_mm", 0.0), _where(index, "construction.flange_thickness_mm"), minimum=0.0),
            }
            if c["radial_build_mode"] not in {"auto", "manual"}:
                raise SceneRecipeError(f"{_where(index, 'construction.radial_build_mode')}: use 'auto' or 'manual'.")
            if c["enabled"] and c["axial_width_mm"] <= 0.0:
                raise SceneRecipeError(
                    f"{_where(index, 'construction.axial_width_mm')}: enter a positive axial width when construction.enabled is true."
                )
            if c["enabled"] and c["radial_build_mode"] == "manual" and c["radial_build_mm"] <= 0.0:
                raise SceneRecipeError(
                    f"{_where(index, 'construction.radial_build_mm')}: enter a positive radial build when manual construction is enabled."
                )
            item["construction"] = c

        if object_type == "circular_coil":
            item["diameter_mm"] = _number(source.get("diameter_mm", 100.0), _where(index, "diameter_mm"), positive=True)
        elif object_type == "racetrack_coil":
            item["end_diameter_mm"] = _number(source.get("end_diameter_mm", 100.0), _where(index, "end_diameter_mm"), positive=True)
            item["straight_length_mm"] = _number(source.get("straight_length_mm", 100.0), _where(index, "straight_length_mm"), minimum=0.0)
            item["arc_segments"] = _integer(source.get("arc_segments", 32), _where(index, "arc_segments"), minimum=8, maximum=256)
        elif object_type == "square_coil":
            item["side_mm"] = _number(source.get("side_mm", 100.0), _where(index, "side_mm"), positive=True)
        elif object_type == "axis_sensor":
            item["start_mm"] = _vector(source.get("start_mm", [0.0, 0.0, -150.0]), _where(index, "start_mm"))
            item["end_mm"] = _vector(source.get("end_mm", [0.0, 0.0, 150.0]), _where(index, "end_mm"))
            item["samples"] = _integer(source.get("samples", 61), _where(index, "samples"), minimum=2, maximum=100_000)
        elif object_type == "box":
            item["dimensions_mm"] = [
                _number(value, f"{_where(index, 'dimensions_mm')}[{i}]", positive=True)
                for i, value in enumerate(_vector(source.get("dimensions_mm", [100.0, 100.0, 100.0]), _where(index, "dimensions_mm")))
            ]
        elif object_type in {"cylinder", "cylinder_magnet"}:
            item["diameter_mm"] = _number(source.get("diameter_mm", 100.0 if object_type == "cylinder" else 20.0), _where(index, "diameter_mm"), positive=True)
            item["height_mm"] = _number(source.get("height_mm", 100.0 if object_type == "cylinder" else 20.0), _where(index, "height_mm"), positive=True)
        elif object_type in {"sphere", "sphere_magnet"}:
            item["diameter_mm"] = _number(source.get("diameter_mm", 100.0 if object_type == "sphere" else 20.0), _where(index, "diameter_mm"), positive=True)
        elif object_type == "block_magnet":
            dims = _vector(source.get("dimensions_mm", [20.0, 20.0, 20.0]), _where(index, "dimensions_mm"))
            item["dimensions_mm"] = [
                _number(value, f"{_where(index, 'dimensions_mm')}[{i}]", positive=True)
                for i, value in enumerate(dims)
            ]

        if object_type in {"box", "cylinder", "sphere"}:
            role = _string(source.get("role", "visual"), _where(index, "role"))
            if role not in {"visual", "exclusion", "measurement"}:
                raise SceneRecipeError(f"{_where(index, 'role')}: use visual, exclusion, or measurement.")
            item["role"] = role
            colour = _string(source.get("colour", "#64748b"), _where(index, "colour"))
            if not _HEX_COLOUR.fullmatch(colour):
                raise SceneRecipeError(f"{_where(index, 'colour')}: use #RRGGBB notation.")
            item["colour"] = colour.lower()
            item["opacity"] = _number(source.get("opacity", 0.28), _where(index, "opacity"), minimum=0.02, maximum=1.0)
            if "clearance_mm" in source:
                item["clearance_mm"] = _number(source["clearance_mm"], _where(index, "clearance_mm"), minimum=0.0)
            else:
                item["clearance_mm"] = None

        if object_type in {"block_magnet", "cylinder_magnet", "sphere_magnet"}:
            if "polarization_T" in source:
                item["polarization_T"] = _vector(source["polarization_T"], _where(index, "polarization_T"))
            else:
                item["polarization_T"] = [0.0, 0.0, 0.0]
                warnings.append(f"objects[{index}] {item.get('name', object_type)!r}: polarization_T omitted; using [0,0,0] T.")

        normalized.append(item)

    # Parent references deliberately target recipe-created groups only.  This
    # avoids AI text silently mutating/reparenting existing scene objects.
    for index, item in enumerate(normalized):
        parent = item.get("parent")
        if not parent:
            continue
        if parent not in keys:
            raise SceneRecipeError(
                f"{_where(index, 'parent')}: no recipe object has key {parent!r}."
            )
        parent_index = keys[parent]
        if normalized[parent_index]["type"] != "group":
            raise SceneRecipeError(
                f"{_where(index, 'parent')}: {parent!r} must refer to a group object."
            )

    # Detect cycles among nested groups.
    for index, item in enumerate(normalized):
        if item["type"] != "group" or "key" not in item:
            continue
        seen = {item["key"]}
        parent = item.get("parent")
        while parent:
            if parent in seen:
                raise SceneRecipeError(f"objects[{index}].parent: group hierarchy contains a cycle.")
            seen.add(parent)
            parent_item = normalized[keys[parent]]
            parent = parent_item.get("parent")

    normalized_recipe: dict[str, Any] = {"format": RECIPE_FORMAT, "version": RECIPE_VERSION}
    normalized_recipe.update(advisory)
    normalized_recipe["objects"] = normalized
    if clarifying_questions and "requires_clarification" not in document:
        warnings.append(
            "clarifying_questions were supplied without requires_clarification; "
            "treated as a blocking clarification request."
        )

    return {
        "recipe": normalized_recipe,
        "warnings": warnings,
        "assistant_message": advisory.get("assistant_message"),
        "assumptions": list(advisory.get("assumptions", [])),
        "suggested_next_steps": list(advisory.get("suggested_next_steps", [])),
        "review_questions": list(advisory.get("review_questions", [])),
        "requires_clarification": bool(advisory.get("requires_clarification", False)),
        "clarifying_questions": list(advisory.get("clarifying_questions", [])),
    }


def scene_recipe_guide(
    current_scene: dict[str, Any] | None = None,
    *,
    import_mode: str = "add",
    guidance_level: str = "balanced",
) -> str:
    """Return the compact clipboard-oriented v1 prompt/format descriptor.

    This is intentionally a *working contract*, not a full manual.  External chat
    UIs often impose fairly small character limits, so the clipboard packet keeps
    the safety/ambiguity rules and complete field vocabulary while avoiding
    repeated explanation and large illustrative examples.
    """
    mode = str(import_mode).strip().lower()
    if mode not in {"add", "replace"}:
        raise ValueError("Scene Recipe import mode must be 'add' or 'replace'.")
    guidance = str(guidance_level).strip().lower()
    if guidance not in GUIDANCE_LEVELS:
        raise ValueError(
            "Scene Recipe guidance level must be 'exploratory', 'balanced', or 'strict'."
        )

    if guidance == "exploratory":
        ambiguity_policy = [
            "AMBIGUITY — EXPLORATORY",
            "- Build a useful, physically plausible first pass when possible. Source facts stay source facts; any unsourced scaffold/approximation must be clearly labelled and must not masquerade as the named design.",
            "- Ask at most ONE blocker unless the user's own requirements conflict. If source retrieval failed, prefer one request for the missing figure/excerpt rather than a parameter-by-parameter interrogation.",
            "- Block only when no useful non-misleading scene can be made; otherwise build and put worthwhile uncertainty in review_questions.",
        ]
    elif guidance == "strict":
        ambiguity_policy = [
            "AMBIGUITY — STRICT",
            "- Reproduce only evidence-backed geometry/electrical relationships; do not approximate unless requested. Any unresolved evidence-backed build-changing ambiguity blocks.",
            "- Clarification still must pass QUESTION ELIGIBILITY. Failed retrieval is not permission to invent alternatives or make the user restate a paper; ask for the smallest missing figure/excerpt/source material.",
            "- Global translation/rotation/naming never blocks unless registration to an external fixed reference matters physically.",
        ]
    else:
        ambiguity_policy = [
            "AMBIGUITY — BALANCED (DEFAULT)",
            "- Do the homework first, then build from the most defensible interpretation. Resolve from request, source/figures, relevant current scene and harmless conventions before asking.",
            "- Block only when the answer materially changes the apparatus AND passes QUESTION ELIGIBILITY; normally 1–3 blockers max. Otherwise build and use review_questions.",
            "- 'Based on/reproduce this paper' means use the source design; do not ask the user to choose between it and an approximation you invented.",
        ]

    lines = [
        "FIELD WORKBENCH SCENE RECIPE v1 — COMPACT CONTRACT",
        f"GUIDANCE LEVEL: {guidance.upper()}",
        "",
        "ROLE",
        "Translate ordinary-language magnetic/coil-design requests into physically plausible Field Workbench scene proposals. Put limits in assistant_message, non-blocking choices in assumptions, and useful Workbench actions/checks in suggested_next_steps.",
        "- Treat Scene Recipe as a reviewed proposal layer, not live control of Workbench. Read supplied scene/DRC context, reason about it, and return the desired recipe-compatible scene plus useful next actions.",
        "- Think like an experienced build-site foreman: inspect the whole assembly, use common sense, and ask only questions that matter at the selected guidance level.",
        "- One matched field magnitude does not prove equivalent exposure; direction, gradients, uniformity and target coverage may differ.",
        "- Never invent paper parameters, field results, DRC clearances, measurements or optimizer results.",
        "",
        "BUILD-SITE PREFLIGHT",
        "Before each object verify: identity/dimensions; physical plane/normal/in-plane axis RELATIVE TO THE ASSEMBLY; relative placement and spacing definition; stated pair/symmetry relationships; electrical sign when field-relevant; real-world feasibility; and whether a figure/photo is needed. Global origin and arbitrary +X/+Y/+Z mapping are bookkeeping unless tied to a fixed reference.",
        "- Numeric sanity: coordinates must agree with the dimensions you state. Convert units explicitly; for a pair separated by S and centered on the origin along one axis, the two coordinates differ by S (normally -S/2 and +S/2). Do not preserve old coordinates while claiming a new spacing.",
        "",
        "QUESTION ELIGIBILITY — CLARIFICATION IS LAST RESORT",
        "Ask a blocking question only if ALL are true: (1) its answer changes the real build; (2) it was not resolvable from the request, accessible source/figures/captions, or relevant current-scene relationships; (3) the ambiguity is evidence-backed, not created by your speculation; (4) it cannot be a harmless convention/assumption/review_question at this guidance level; (5) the question is the smallest useful blocker.",
        "- Never seed questions with unsupported candidate coil counts/shapes/topologies/dimensions/polarities/interpretations. If you do not know, ask neutrally. If retrieval/link access fails, say so and request the relevant figure/excerpt once; do not pretend the source itself was ambiguous or fill gaps from vague memory.",
        "",
        *ambiguity_policy,
        "- review_questions = worth confirming after a build (non-blocking). requires_clarification + clarifying_questions = genuine blockers and requires objects=[].",
        "- Named source design: do not infer geometry from its name/acronym. Retrieve title/DOI/URL plus relevant figures/captions when available before building.",
        "- Current scene is evidence, not scripture: clear target/plate/fixture relationships may resolve placement, but existing coils/orientations need not be preserved unless requested or continuity matters.",
        "",
        "SCENE INTERACTION / AUTHORITY",
        "- No live Workbench connection: never claim you already changed/rendered/analyzed/measured/exported anything unless the prompt supplied that result.",
        "- Complete-scene mode MAY propose preserve/change/move/add/remove operations by returning the COMPLETE desired recipe-compatible object set. Workbench validates it; the user then chooses Modify current scene or Paste into new scene. Omitted representable objects disappear on Modify current.",
        "- Native ids are context references only; recipe key/parent are local. recipe_type_hint marks a class v1 can reconstruct. Objects without it are context-only: do not silently approximate them; disclose the limit and prefer new-scene comparison when they must be retained.",
        "",
        "RESPONSE CONTRACT",
        "Return ONE JSON object only: no Markdown fences or prose outside JSON; no invented fields/trailing commas.",
        f'- format="{RECIPE_FORMAT}"; version={RECIPE_VERSION}; objects=array (max 500).',
        "- Optional top-level: assistant_message:string; assumptions:string[]; suggested_next_steps:string[]; review_questions:string[]; requires_clarification:boolean; clarifying_questions:string[] (arrays max 20).",
        "- requires_clarification=true requires clarifying_questions and objects=[]; Field Workbench will display the questions and refuse import.",
    ]

    if mode == "replace":
        lines.extend(
            [
                "- COMPLETE-SCENE workflow: return the COMPLETE desired recipe-compatible object scene. Preserve every representable current object/property the user wants retained, including unchanged objects needed in the final result. If no change is justified, preserve the useful representable scene.",
                "- Scene-wide notes/DRC settings survive Modify current scene, but object-specific properties survive only when represented by returned objects/defaults. Unsupported/context-only objects cannot be preserved by pretending they are recipe objects.",
            ]
        )
    else:
        lines.extend(
            [
                "- ADD mode: return only new objects. Do not duplicate current objects unless requested. If no addition is justified, objects may be empty.",
            ]
        )

    lines.extend(
        [
            "",
            "COORDINATES / LOCAL GEOMETRY",
            "- World: +X=right (LR), +Y=anterior (AP), +Z=superior (SI). Positions/dimensions mm; XYZ Euler rotation degrees; current mA; magnet polarization T.",
            "- At rotation_deg=[0,0,0], circular/racetrack/square coil current paths lie in the local XY plane; coil normal/axis is +Z. racetrack straight sections run parallel to local X; semicircular ends extend along local Y. Square sides follow local X/Y.",
            "- If source geometry is defined but Workbench axes/origin are not, choose a simple rigid mapping and disclose it. Ask only when registration to anatomy/fixtures/gravity/external coordinates makes the mapping physical.",
            "",
            "COMMON FIELDS",
            "- Every object: type required; key optional recipe-local [A-Za-z0-9_.-]+; name; parent=key of group created in this recipe; visible. For top-level objects OMIT parent; do not send parent:null (null is tolerated as omission).",
            "- position_mm=[x,y,z] and rotation_deg=[x,y,z] default [0,0,0] where applicable. group has no transform in v1: omit group position/rotation (identity [0,0,0] is tolerated and ignored). axis_sensor uses start/end instead.",
            "",
            "SUPPORTED TYPES / FIELDS",
            "- group",
            "- circular_coil: diameter_mm + COIL fields",
            "- racetrack_coil: end_diameter_mm, straight_length_mm, arc_segments + COIL fields",
            "- square_coil: side_mm + COIL fields",
            "- point_sensor: position_mm",
            "- axis_sensor (Linear axis sensor): start_mm, end_mm, samples",
            "- box: dimensions_mm[x,y,z], role, colour, opacity, clearance_mm",
            "- cylinder: diameter_mm, height_mm, role, colour, opacity, clearance_mm",
            "- sphere: diameter_mm, role, colour, opacity, clearance_mm",
            "- block_magnet: dimensions_mm[x,y,z], polarization_T[x,y,z]",
            "- cylinder_magnet: diameter_mm, height_mm, polarization_T[x,y,z]",
            "- sphere_magnet: diameter_mm, polarization_T[x,y,z]",
            "- generic_bust; sri24_brain: common transform fields only",
            "- Passive role='visual'|'exclusion'|'measurement'; colour='#RRGGBB'; opacity=0.02..1; clearance_mm>=0.",
            "",
            "COIL FIELDS / PHYSICAL INPUTS",
            "- IMPORTANT: in returned recipes, COIL fields are DIRECT fields of circular_coil/racetrack_coil/square_coil objects. Do NOT wrap them in a nested 'coil' object. Current-scene context groups them under 'coil' only as a readable context container.",
            "- COIL fields: turns, drive_current_mA, awg, enabled, field_scale_factor, current_mode ('rms_dc'|'peak_sine'), lead_length_mm, resistance_20_ohm_per_km, enamel_radial_mm, packing_factor, temperature_c, additional_mass_g, construction.",
            "- resistance_20_ohm_per_km is optional positive custom resistance at 20 C. OMIT to use nominal copper AWG. Do not emit 0 as a placeholder (0/null is tolerated as nominal-AWG fallback).",
            "- enamel_radial_mm 0..2 (default .025); packing_factor .10..95 (default .75); temperature_c -100..300 (default 20); additional_mass_g>=0.",
            "- NEW coils: omit unknown optional winding/construction fields; do not manufacture defaults. For coils intentionally retained in the complete scene, preserve supplied physical inputs.",
            "- construction optional: enabled:boolean; axial_width_mm>=0; radial_build_mode='auto'|'manual'; radial_build_mm>=0; bobbin_wall_thickness_mm>=0; flange_height_mm>=0; flange_thickness_mm>=0.",
            "- current-scene workbench_derived values are read-only numerical context: reason from them, but NEVER return workbench_derived or claim you recalculated those values.",
            "",
            "OBJECT CHECKS",
            "- Coils: resolve current-path dimensions, relative plane/normal/in-plane axis, pair/separation relationship and material current direction. Winding pack != centreline geometry. For a coaxial/Helmholtz-style pair, verify the centre-to-centre vector is parallel to BOTH coil normals; rotation_deg must actually produce that relationship.",
            "- Sensors/regions: distinguish plate footprint, well grid, thickness, stack spacing and actual sampled location/volume.",
            "- Repeated/symmetric objects: keep known dimensions consistent and verify transforms create the claimed relationship.",
            "- DRC: construction.enabled=true uses winding/bobbin envelope; otherwise only the zero-thickness current path is checked. Bare-path clear != physical certification; never invent build dimensions to enable DRC.",
            "",
            "WORKBENCH CAPABILITIES / WHAT TO SUGGEST",
            "- Inspect/visualize: point + Linear axis sensors; 2D contours/fluxlines; 3D Full Volume/Slices/fluxlines; LPBA40 Brain View regional summaries; waveform animation and video export. Detached 2D/3D/Brain viewers are frozen launch-time scene copies; reopen one after scene edits.",
            "- Analyze/design: measurement volumes or explicit samples (including built-in well-plate centres), target statistics, snapshots/comparison, bench-data Field-correction fitting, DRC of bare paths/physical coil envelopes/surface-region clearances, and the optimizer. Optimizer changes bounded properties only on coils already in its frozen launch scene; add/rebuild topology first. Never invent calculated outcomes.",
            "- Outside Recipe v1 Workbench also has Manual/WMM2025 Background fields, built-in culture plates, UPENN-GBM imports, arbitrary STL/OBJ meshes, and portable/OpenSCAD/FEMM/GetDP/COMSOL export. Recommend them when useful but do not emit unsupported recipe objects. Direct recipe creation is limited to the supported types above.",
            "- Recipe v1 has no script/network/file-fetch commands or native-id patch/delete commands; scene changes use Add or reviewed complete-scene replacement.",
            "- Omitted drive_current_mA defaults to 0 with warning; omitted magnet polarization defaults [0,0,0] T with warning; unknown types/fields are errors.",
        ]
    )

    if current_scene is not None:
        # Compact JSON is intentional: this clipboard packet is consumed by an
        # LLM and several free chat UIs have ~16k-character composer limits.
        compact_scene = json.dumps(
            current_scene,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        lines.extend(
            [
                "",
                "CURRENT FIELD WORKBENCH SCENE (context only; native ids are references, not recipe keys)",
                compact_scene,
            ]
        )
        if isinstance(current_scene.get("design_check"), dict):
            lines.extend(
                [
                    "DRC CONTEXT",
                    "- design_rules=active rules; design_check is usable when fresh=true; coil_status summarizes coils; issues contains measured/required clearances. negative actual_signed_clearance_mm means penetration/intersection. Preserve clear parts when practical and fix reported geometry rather than blindly redesigning everything.",
                ]
            )

    lines.extend(
        [
            "",
            "---------------- USER INSTRUCTIONS BELOW ----------------",
            "Type your request after this line:",
        ]
    )
    return "\n".join(lines) + "\n"
