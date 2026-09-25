"""Structured authoring recipes for generated Field Workbench waveforms.

The sampled time/value trace remains the playback representation.  This module
stores the *intent* used to create that trace separately, so later editor work
can reopen generated segments with meaningful controls instead of attempting to
reverse-engineer samples.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


AUTHORING_RECIPE_KEY = "authoring_recipe"
AUTHORING_RECIPE_FORMAT = "fieldworkbench.waveform-authoring"
AUTHORING_RECIPE_VERSION = 1

_SUPPORTED_UNITS = {
    "current": {"A", "mA"},
    "voltage": {"V", "mV"},
}
_SUPPORTED_GENERATORS = {"square", "sine", "action_potential", "free_form", "dc"}
_PARAMETER_KEYS = {
    "square": ("frequency_hz", "peak_to_peak", "offset", "duty_cycle_percent"),
    "sine": ("frequency_hz", "peak_to_peak", "offset", "phase_deg"),
    "action_potential": (
        "start_percent",
        "peak_percent",
        "trough_percent",
        "recovery_percent",
        "resting_level",
        "peak_level",
        "trough_level",
    ),
    "free_form": ("control_points", "curve"),
    "dc": ("level",),
}


class WaveformRecipeError(ValueError):
    """Raised when structured waveform authoring data is malformed."""


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise WaveformRecipeError(f"{label} must be numeric.") from error
    if not math.isfinite(number):
        raise WaveformRecipeError(f"{label} must be finite.")
    if minimum is not None and number < minimum:
        raise WaveformRecipeError(f"{label} must be at least {minimum:g}.")
    return number


def _signal_context(quantity: Any, unit: Any) -> dict[str, str]:
    quantity_text = str(quantity or "").strip().lower()
    unit_text = str(unit or "").strip()
    if quantity_text not in _SUPPORTED_UNITS or unit_text not in _SUPPORTED_UNITS[quantity_text]:
        raise WaveformRecipeError("Waveform recipes require Current (A/mA) or Voltage (V/mV).")
    return {"quantity": quantity_text, "unit": unit_text}


def _control_points(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 2:
        raise WaveformRecipeError("A free-form recipe needs at least two Bézier anchors.")
    result: list[dict[str, float]] = []
    previous = -math.inf
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, Mapping):
            raise WaveformRecipeError(f"Free-form anchor {index} must be an object.")
        point = {
            "time_percent": _finite(raw.get("time_percent"), label=f"Anchor {index} time"),
            "value": _finite(raw.get("value"), label=f"Anchor {index} value"),
            "handle_time_percent": _finite(
                raw.get("handle_time_percent", 0.0),
                label=f"Anchor {index} handle time",
                minimum=0.0,
            ),
            "handle_value": _finite(raw.get("handle_value", 0.0), label=f"Anchor {index} handle value"),
        }
        if not 0.0 <= point["time_percent"] <= 100.0:
            raise WaveformRecipeError("Free-form anchor times must be between 0% and 100%.")
        if point["time_percent"] <= previous:
            raise WaveformRecipeError("Free-form anchors must be strictly left-to-right.")
        previous = point["time_percent"]
        result.append(point)
    if not math.isclose(result[0]["time_percent"], 0.0, abs_tol=1e-9) or not math.isclose(
        result[-1]["time_percent"], 100.0, abs_tol=1e-9
    ):
        raise WaveformRecipeError("Free-form recipes must start at 0% and end at 100% time.")
    return result


def generated_segment_from_settings(settings: Mapping[str, Any], *, start_s: float) -> dict[str, Any]:
    """Create one canonical generator segment from a :class:`GeneratedTrace` settings map."""

    if not isinstance(settings, Mapping):
        raise WaveformRecipeError("Generated waveform settings must be an object.")
    generator = str(settings.get("shape") or "").strip().lower()
    if generator not in _SUPPORTED_GENERATORS:
        raise WaveformRecipeError(f"Unsupported waveform generator {generator!r}.")
    duration = _finite(settings.get("duration_s"), label="Generated segment duration", minimum=0.0)
    if duration <= 0.0:
        raise WaveformRecipeError("Generated segment duration must be greater than zero.")
    start = _finite(start_s, label="Generated segment start", minimum=0.0)

    parameters: dict[str, Any] = {}
    for key in _PARAMETER_KEYS[generator]:
        if key not in settings:
            raise WaveformRecipeError(f"Generated {generator.replace('_', ' ')} settings are missing {key!r}.")
        if key == "control_points":
            parameters[key] = _control_points(settings[key])
        elif key == "curve":
            parameters[key] = str(settings[key] or "").strip()
            if not parameters[key]:
                raise WaveformRecipeError("Free-form curve model cannot be blank.")
        else:
            parameters[key] = _finite(settings[key], label=key.replace("_", " ").title())

    return {
        "kind": "generator",
        "generator": generator,
        "start_s": start,
        "duration_s": duration,
        "parameters": parameters,
    }


def compiled_segment(*, start_s: float, duration_s: float, description: str = "Existing sampled waveform") -> dict[str, Any]:
    """Represent an authored composition section whose original recipe is unavailable."""

    start = _finite(start_s, label="Compiled segment start", minimum=0.0)
    duration = _finite(duration_s, label="Compiled segment duration", minimum=0.0)
    if duration <= 0.0:
        raise WaveformRecipeError("Compiled segment duration must be greater than zero.")
    return {
        "kind": "compiled",
        "start_s": start,
        "duration_s": duration,
        "description": str(description or "Existing sampled waveform").strip() or "Existing sampled waveform",
    }


def sampled_segment(
    time_s: Sequence[float],
    values: Sequence[float],
    *,
    start_s: float,
    description: str = "Imported waveform",
) -> dict[str, Any]:
    """Store one lossless sampled chunk so imports remain reorderable."""
    times = [_finite(value, label="Sample time") for value in time_s]
    samples = [_finite(value, label="Sample value") for value in values]
    if len(times) != len(samples) or len(times) < 2:
        raise WaveformRecipeError("A sampled segment needs at least two matching time/value samples.")
    if any(times[index + 1] <= times[index] for index in range(len(times) - 1)):
        raise WaveformRecipeError("Sampled segment time must increase strictly.")
    origin = times[0]
    relative = [value - origin for value in times]
    start = _finite(start_s, label="Sampled segment start", minimum=0.0)
    return {
        "kind": "sampled",
        "start_s": start,
        "duration_s": relative[-1],
        "description": str(description or "Imported waveform").strip() or "Imported waveform",
        "time_s": relative,
        "values": samples,
    }



def encoded_segment(
    raw_values: Sequence[float | int],
    *,
    start_s: float,
    sample_interval_s: float,
    profile: Mapping[str, Any],
    runtime_value_at_min: float | None = None,
    runtime_value_at_max: float | None = None,
    source_name: str = "",
    source_path: str = "",
) -> dict[str, Any]:
    """Store one editable import produced by a runtime-timed format profile.

    The exact source samples and the declarative encoding profile are retained.
    This is intentionally format-neutral: the bundled DAC encoding and any user
    profile tagged for import-time timing use this same component representation.
    """
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes, bytearray)):
        raise WaveformRecipeError("An encoded import needs a sequence of raw source samples.")
    samples = [_finite(value, label=f"Encoded sample {index}") for index, value in enumerate(raw_values, start=1)]
    if len(samples) < 2:
        raise WaveformRecipeError("An encoded import needs at least two source samples.")
    interval = _finite(sample_interval_s, label="Encoded sample interval", minimum=0.0)
    if interval <= 0.0:
        raise WaveformRecipeError("Encoded sample interval must be greater than zero.")
    if not isinstance(profile, Mapping):
        raise WaveformRecipeError("An encoded import needs its waveform format profile.")
    profile_copy = copy.deepcopy(dict(profile))
    start = _finite(start_s, label="Encoded import start", minimum=0.0)
    result: dict[str, Any] = {
        "kind": "encoded",
        "start_s": start,
        "duration_s": interval * float(len(samples) - 1),
        "raw_values": samples,
        "sample_interval_s": interval,
        "profile": profile_copy,
        "source_name": str(source_name or "").strip(),
        "source_path": str(source_path or "").strip(),
    }
    if runtime_value_at_min is not None:
        result["runtime_value_at_min"] = _finite(runtime_value_at_min, label="Runtime lower mapped value")
    if runtime_value_at_max is not None:
        result["runtime_value_at_max"] = _finite(runtime_value_at_max, label="Runtime upper mapped value")
    return result

def make_authoring_recipe(
    segments: Sequence[Mapping[str, Any]],
    *,
    quantity: str,
    unit: str,
    integrity: str = "exact",
    detached_reason: str = "",
) -> dict[str, Any]:
    """Build and validate the versioned authoring recipe container."""

    raw = {
        "format": AUTHORING_RECIPE_FORMAT,
        "version": AUTHORING_RECIPE_VERSION,
        "signal": _signal_context(quantity, unit),
        "integrity": str(integrity or "exact").strip().lower(),
        "segments": [copy.deepcopy(dict(segment)) for segment in segments],
    }
    if detached_reason:
        raw["detached_reason"] = str(detached_reason).strip()
    return normalize_authoring_recipe(raw)


def normalize_authoring_recipe(value: Any) -> dict[str, Any]:
    """Return a validated, JSON-safe canonical copy of one authoring recipe."""

    if not isinstance(value, Mapping):
        raise WaveformRecipeError("Waveform authoring recipe must be an object.")
    if value.get("format") != AUTHORING_RECIPE_FORMAT:
        raise WaveformRecipeError("Unsupported waveform authoring recipe format.")
    try:
        version = int(value.get("version", 0))
    except (TypeError, ValueError) as error:
        raise WaveformRecipeError("Waveform authoring recipe version must be an integer.") from error
    if version != AUTHORING_RECIPE_VERSION:
        raise WaveformRecipeError(
            f"Unsupported waveform authoring recipe version {version}; expected {AUTHORING_RECIPE_VERSION}."
        )

    signal = value.get("signal")
    if not isinstance(signal, Mapping):
        raise WaveformRecipeError("Waveform authoring recipe is missing its signal definition.")
    signal_copy = _signal_context(signal.get("quantity"), signal.get("unit"))

    integrity = str(value.get("integrity") or "exact").strip().lower()
    if integrity not in {"exact", "detached"}:
        raise WaveformRecipeError("Waveform authoring recipe integrity must be 'exact' or 'detached'.")

    raw_segments = value.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes, bytearray)) or not raw_segments:
        raise WaveformRecipeError("Waveform authoring recipe must contain at least one segment.")

    segments: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, Mapping):
            raise WaveformRecipeError(f"Waveform authoring segment {index} must be an object.")
        kind = str(raw.get("kind") or "").strip().lower()
        start = _finite(raw.get("start_s"), label=f"Segment {index} start", minimum=0.0)
        duration = _finite(raw.get("duration_s"), label=f"Segment {index} duration", minimum=0.0)
        if duration <= 0.0:
            raise WaveformRecipeError(f"Segment {index} duration must be greater than zero.")
        if index > 1 and start + 1e-12 < previous_end:
            raise WaveformRecipeError("Waveform authoring segments cannot overlap or run backward in time.")

        if kind == "generator":
            generator = str(raw.get("generator") or "").strip().lower()
            if generator not in _SUPPORTED_GENERATORS:
                raise WaveformRecipeError(f"Unsupported waveform generator {generator!r}.")
            parameters = raw.get("parameters")
            if not isinstance(parameters, Mapping):
                raise WaveformRecipeError(f"Generator segment {index} is missing its parameters.")
            settings = {
                "shape": generator,
                "duration_s": duration,
                **dict(parameters),
            }
            canonical = generated_segment_from_settings(settings, start_s=start)
        elif kind == "compiled":
            canonical = compiled_segment(
                start_s=start,
                duration_s=duration,
                description=str(raw.get("description") or "Existing sampled waveform"),
            )
        elif kind == "sampled":
            canonical = sampled_segment(
                raw.get("time_s", ()),
                raw.get("values", ()),
                start_s=start,
                description=str(raw.get("description") or "Imported waveform"),
            )
        elif kind == "encoded":
            canonical = encoded_segment(
                raw.get("raw_values", ()),
                start_s=start,
                sample_interval_s=raw.get("sample_interval_s"),
                profile=raw.get("profile", {}),
                runtime_value_at_min=raw.get("runtime_value_at_min"),
                runtime_value_at_max=raw.get("runtime_value_at_max"),
                source_name=str(raw.get("source_name") or ""),
                source_path=str(raw.get("source_path") or ""),
            )
        else:
            raise WaveformRecipeError(f"Unsupported waveform authoring segment kind {kind!r}.")

        segments.append(canonical)
        previous_end = max(previous_end, start + duration)

    result: dict[str, Any] = {
        "format": AUTHORING_RECIPE_FORMAT,
        "version": AUTHORING_RECIPE_VERSION,
        "signal": signal_copy,
        "integrity": integrity,
        "segments": segments,
    }
    reason = str(value.get("detached_reason") or "").strip()
    if integrity == "detached" and reason:
        result["detached_reason"] = reason
    return result


def recipe_is_exact(value: Any) -> bool:
    try:
        return normalize_authoring_recipe(value).get("integrity") == "exact"
    except WaveformRecipeError:
        return False


def recipe_is_fully_generated(value: Any) -> bool:
    """Whether every section can be reconstructed from generator parameters alone."""
    try:
        recipe = normalize_authoring_recipe(value)
    except WaveformRecipeError:
        return False
    return recipe.get("integrity") == "exact" and all(
        segment.get("kind") == "generator" for segment in recipe["segments"]
    )


def detach_authoring_recipe(value: Any, *, reason: str) -> dict[str, Any] | None:
    """Preserve prior authoring settings as provenance while marking them non-authoritative."""
    if value is None:
        return None
    try:
        recipe = normalize_authoring_recipe(value)
    except WaveformRecipeError:
        return None
    recipe["integrity"] = "detached"
    text = str(reason or "Waveform samples were edited outside the builder.").strip()
    recipe["detached_reason"] = text or "Waveform samples were edited outside the builder."
    return recipe


def update_recipe_signal(value: Any, *, quantity: str, unit: str) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        recipe = normalize_authoring_recipe(value)
    except WaveformRecipeError:
        return None
    recipe["signal"] = _signal_context(quantity, unit)
    return recipe


def recipe_summary(value: Any) -> tuple[int, list[str], bool] | None:
    """Return (generated_count, generator_names, exact) for compact UI provenance."""
    try:
        recipe = normalize_authoring_recipe(value)
    except WaveformRecipeError:
        return None
    generators = [
        str(segment.get("generator"))
        for segment in recipe["segments"]
        if segment.get("kind") == "generator"
    ]
    return len(generators), generators, recipe.get("integrity") == "exact"
