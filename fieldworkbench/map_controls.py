"""Versioned, presentation-only Maps control snapshots and cautious legacy recovery."""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QAbstractButton, QComboBox, QDoubleSpinBox, QSpinBox, QWidget

FORMAT = "fieldworkbench.map-controls"
VERSION = 1
AUX_KEY = "map_controls"
# Explicit bindings avoid restoring arbitrary widgets, filenames or model state.
CONTROLS = (
    "plane", "offset", "span", "resolution", "field", "component", "logarithmic",
    "display_mode", "flux_density", "colour_minimum_enabled", "colour_minimum",
    "colour_maximum_enabled", "colour_maximum", "show_grid", "show_contours",
    "contour_count", "contour_labels", "object_outlines", "sensor_placements",
    "well_outlines", "live_measurement_analysis", "slice_axis", "slice_count",
    "view_type", "include_scene", "intensity_opacity_enabled", "volume_opacity_lower",
    "volume_opacity_upper", "volume_opacity_lower_level", "volume_opacity_upper_level",
    "fluxline_minimum_field_enabled", "fluxline_minimum_field_value",
    "fluxline_conductor_cutoff_enabled", "fluxline_conductor_cutoff_value",
)
SCRIPTED = (
    "scripted_source_type", "scripted_exposure", "scripted_exposure_duration",
    "scripted_waveform", "scripted_amplitude", "scripted_frequency",
    "scripted_duration", "scripted_slew",
)


class MapControlsError(ValueError):
    pass


def _bindings(page):
    values = {name: getattr(page, name, None) for name in CONTROLS}
    for i, widget in enumerate(getattr(page, "coordinates", ())):
        if i < 3:
            values[f"coordinates.{i}"] = widget
    return values


def _capture(widgets):
    result = {}
    for name, widget in widgets.items():
        if isinstance(widget, QComboBox):
            if widget.currentIndex() < 0:
                continue
            data = widget.currentData()
            if data is None or type(data) in (str, int, float, bool):
                result[name] = {"kind": "choice", "data": data, "text": widget.currentText()}
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            result[name] = {"kind": "number", "value": widget.value()}
        elif isinstance(widget, QAbstractButton) and widget.isCheckable():
            result[name] = {"kind": "checked", "value": widget.isChecked()}
    return result


def capture_controls(page, *, mode: str, view: str, workspace=None):
    selections = {}
    for name, method in (("active_coil_ids", "selected_active_coil_ids"),
                         ("scene_object_ids", "selected_scene_object_ids"),
                         ("outline_object_ids", "selected_object_outline_ids")):
        getter = getattr(page, method, None)
        if callable(getter):
            selections[name] = list(getter())
    getter = getattr(page, "scene_object_opacity_values", None)
    if callable(getter):
        selections["scene_object_opacities"] = dict(getter())
    return {
        "format": FORMAT, "version": VERSION, "mode": mode, "view": view,
        "source_signature": ("sha256:" + hashlib.sha256(page.adapter.signature().encode("utf-8")).hexdigest()
                             if hasattr(page, "adapter") else None),
        "widgets": _capture(_bindings(page)), "selections": selections,
        "scripted_widgets": _capture({n: getattr(workspace, n, None) for n in SCRIPTED})
        if mode == "playback" and workspace is not None else {},
    }


def capture_page_controls(page):
    callback = getattr(page, "_map_controls_capture", None)
    if callable(callback):
        return callback()
    view = getattr(page, "_unified_workspace_view_key", "")
    if not view:
        view = "brain" if hasattr(page, "region_hierarchy") else (
            "2d" if hasattr(page, "plane") else "3d")
    return capture_controls(page, mode=getattr(page, "_unified_workspace_mode_key", "")
                            or getattr(page, "workspace_mode", "static"), view=view)


def attach_controls(widget, snapshot):
    if widget is None or snapshot is None:
        return
    widget._map_controls_snapshot = copy.deepcopy(snapshot)
    aux = copy.deepcopy(getattr(widget, "_saved_map_aux", {}))
    aux = aux if isinstance(aux, dict) else {}
    aux[AUX_KEY] = copy.deepcopy(snapshot)
    widget._saved_map_aux = aux


def recorded_controls(widget):
    value = getattr(widget, "_map_controls_snapshot", None)
    if value is not None:
        return copy.deepcopy(value)
    aux = getattr(widget, "_saved_map_aux", {})
    return copy.deepcopy(aux.get(AUX_KEY)) if isinstance(aux, dict) else None


def validate_snapshot(snapshot, *, mode, view):
    if not isinstance(snapshot, dict) or snapshot.get("format") != FORMAT:
        raise MapControlsError("This map has no valid controls record.")
    if type(snapshot.get("version")) is not int or snapshot["version"] != VERSION:
        raise MapControlsError("This map uses an unsupported controls-record version; controls were not changed.")
    if snapshot.get("mode") != mode or snapshot.get("view") != view:
        raise MapControlsError("The controls record does not match this map's mode/view; controls were not changed.")
    for key in ("widgets", "scripted_widgets", "selections"):
        if not isinstance(snapshot.get(key, {}), dict):
            raise MapControlsError(f"Invalid {key} controls record; controls were not changed.")
    return copy.deepcopy(snapshot)


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _plan(widgets, values, skipped, limits=None):
    actions = []
    for name, value in values.items():
        widget = widgets.get(name)
        if not isinstance(value, dict):
            skipped.append(f"{name}: invalid value")
            continue
        kind = value.get("kind")
        if isinstance(widget, QComboBox) and kind == "choice":
            data = value.get("data")
            if data is not None and type(data) not in (str, int, float, bool):
                skipped.append(f"{name}: invalid choice")
                continue
            index = widget.findData(data) if data is not None else widget.findText(str(value.get("text", "")))
            if index < 0 and all(widget.itemData(i) is None for i in range(widget.count())):
                index = widget.findText(str(value.get("text", "")))
            if index < 0:
                skipped.append(f"{name}: choice no longer available")
            else:
                actions.append((widget, widget.setCurrentIndex, index))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)) and kind == "number":
            number = value.get("value")
            minimum, maximum = (limits or {}).get(name, (widget.minimum(), widget.maximum()))
            if (not _finite_number(number)
                    or not minimum <= number <= maximum
                    or isinstance(widget, QSpinBox) and int(number) != number):
                skipped.append(f"{name}: number outside supported range")
            else:
                actions.append((widget, widget.setValue, int(number) if isinstance(widget, QSpinBox) else number))
        elif isinstance(widget, QAbstractButton) and widget.isCheckable() and kind == "checked" and type(value.get("value")) is bool:
            actions.append((widget, widget.setChecked, value["value"]))
        else:
            skipped.append(f"{name}: control unavailable or invalid type")
    return actions


def restore_controls(page, snapshot, *, mode, view, workspace=None):
    state = validate_snapshot(snapshot, mode=mode, view=view)
    skipped = []
    values = state.get("widgets", {})
    def target(name):
        record = values.get(name)
        return record.get("value") if isinstance(record, dict) else getattr(page, name).value()
    for lower, upper in (("volume_opacity_lower_level", "volume_opacity_upper_level"),
                         ("colour_minimum", "colour_maximum")):
        if not (hasattr(page, lower) and hasattr(page, upper)) or not ({lower, upper} & set(values)):
            continue
        lo, hi = target(lower), target(upper)
        def checked(name):
            value = values.get(name)
            return value.get("value", getattr(page, name).isChecked()) if isinstance(value, dict) else getattr(page, name).isChecked()
        enabled = lower.startswith("volume_") or (checked("colour_minimum_enabled") and checked("colour_maximum_enabled"))
        if enabled and _finite_number(lo) and _finite_number(hi) and (lo > hi or lo == hi and lower == "colour_minimum"):
            for name in (lower, upper, lower + "_enabled", upper + "_enabled"):
                values.pop(name, None)
            skipped.append(f"{lower}/{upper}: inconsistent bounds; unchanged")
    actions = _plan(_bindings(page), state.get("widgets", {}), skipped)
    source_values = copy.deepcopy(state.get("scripted_widgets", {}))
    source_type = source_values.get("scripted_source_type", {})
    limits, range_updates = {}, []
    if isinstance(source_type, dict) and source_type.get("data") == "exposure":
        choice = source_values.get("scripted_exposure", {})
        duration = source_values.get("scripted_exposure_duration", {})
        combo = getattr(workspace, "scripted_exposure", None)
        spin = getattr(workspace, "scripted_exposure_duration", None)
        valid = isinstance(choice, dict) and isinstance(duration, dict) and isinstance(combo, QComboBox) and isinstance(spin, QDoubleSpinBox)
        maximum = None
        if valid:
            valid = combo.findData(choice.get("data")) >= 0
        if valid:
            summaries = workspace.snapshot_adapter.list_exposures()
            maximum = next((x.get("duration_s") for x in summaries if x.get("id") == choice.get("data")), None)
            value = duration.get("value")
            valid = _finite_number(maximum) and _finite_number(value) and spin.minimum() <= value <= maximum
        if valid:
            limits["scripted_exposure_duration"] = (spin.minimum(), maximum)
            range_updates.append((spin, maximum))
        else:
            source_values.clear()
            skipped.append("saved exposure missing/changed or invalid preview duration; source inputs unchanged")
    actions += _plan({n: getattr(workspace, n, None) for n in SCRIPTED}, source_values, skipped, limits)
    # Signal-block all bound controls, including fields updated by unit helpers.
    widgets = list(_bindings(page).values()) + [getattr(workspace, n, None) for n in SCRIPTED]
    unique_widgets = {id(w): w for w in widgets if isinstance(w, QWidget)}
    blockers = [QSignalBlocker(w) for w in unique_widgets.values()]
    try:
        for widget, setter, value in actions:
            if isinstance(widget, QComboBox):
                setter(value)
        for widget, maximum in range_updates:
            widget.setMaximum(maximum)
        if any(widget is getattr(page, "field", None) for widget, _, _ in actions):
            colours = {name: getattr(page, name).value() for name in
                       ("colour_minimum", "colour_maximum") if hasattr(page, name)}
            update_units = getattr(page, "_update_colour_limit_units", None)
            if callable(update_units):
                update_units()
            # Missing metadata must not reset an existing numeric control.
            for name, value in colours.items():
                getattr(page, name).setValue(value)
        for widget, setter, value in actions:
            if not isinstance(widget, QComboBox):
                setter(value)
    except Exception:
        del blockers
        raise
    restored = len(actions)
    selections = state.get("selections", {})
    for key, attr, known_attr, refresh in (
        ("active_coil_ids", "_playback_selected_coil_ids", "_playback_known_coil_ids", "_refresh_playback_coils_menu"),
        ("scene_object_ids", "_scene_selected_object_ids", "_scene_known_object_ids", "_refresh_scene_objects_menu"),
        ("outline_object_ids", "_outline_selected_object_ids", "_outline_known_object_ids", "_refresh_object_outline_menu"),
    ):
        if key not in selections:
            continue
        ids = selections[key]
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or not hasattr(page, attr):
            skipped.append(f"{key}: invalid or unavailable selection")
            continue
        known = set(getattr(page, known_attr, ()))
        missing = set(ids) - known
        if missing:
            # Do not silently replace the selection with a different subset.
            skipped.append(f"{key}: missing objects ({', '.join(sorted(missing))}); selection unchanged")
            continue
        setattr(page, attr, set(ids))
        for flag in ("_playback_selection_initialized", "_scene_selection_initialized", "_outline_selection_initialized"):
            if hasattr(page, flag):
                setattr(page, flag, True)
        getattr(page, refresh)()
        restored += 1
    if "scene_object_opacities" in selections:
        opacities = selections["scene_object_opacities"]
        known = set(getattr(page, "_scene_known_object_ids", ()))
        if (not isinstance(opacities, dict) or any(
                k not in known or not _finite_number(v) or not 0 <= v <= 1
                for k, v in opacities.items()) or not hasattr(page, "_scene_object_opacity_overrides")):
            skipped.append("scene_object_opacities: invalid or missing objects; unchanged")
        else:
            page._scene_object_opacity_overrides = dict(opacities)
            page._refresh_scene_objects_menu()
            restored += 1
    # Synchronize enablement, not a rendered result or the underlying scene.
    for method in ("_update_view_type_controls",
                   "_update_colour_limit_availability", "_update_log_availability",
                   "_update_fluxline_options_visibility", "_apply_fluxline_options_to_adapter"):
        callback = getattr(page, method, None)
        if callable(callback):
            callback()
    if workspace is not None and mode == "playback":
        callback = getattr(workspace, "_scripted_source_changed", None)
        if callable(callback):
            callback()
    del blockers
    if state.get("source_signature") and hasattr(page, "adapter"):
        current_signature = page.adapter.signature()
        if str(state["source_signature"]).startswith("sha256:"):
            current_signature = "sha256:" + hashlib.sha256(current_signature.encode("utf-8")).hexdigest()
        if state["source_signature"] != current_signature:
            skipped.append("source scene differs from the map's recorded scene; only controls were restored")
    context = state.get("render_context", {})
    original_solver = context.get("coil_modeling_method") if isinstance(context, dict) else None
    current_solver = getattr(getattr(page, "adapter", None), "coil_modeling_method", None)
    if original_solver and current_solver and original_solver != current_solver:
        skipped.append(f"map used solver {original_solver}; current solver {current_solver} was left unchanged")
    missing = set(_capture(_bindings(page))) - set(state.get("widgets", {}))
    for key, getter in (("active_coil_ids", "selected_active_coil_ids"),
                        ("scene_object_ids", "selected_scene_object_ids"),
                        ("outline_object_ids", "selected_object_outline_ids"),
                        ("scene_object_opacities", "scene_object_opacity_values")):
        if callable(getattr(page, getter, None)) and key not in state.get("selections", {}):
            missing.add(key)
    if mode == "playback" and workspace is not None:
        missing |= set(_capture({n: getattr(workspace, n, None) for n in SCRIPTED})) - set(state.get("scripted_widgets", {}))
    if restored:
        page._selector_scene_dirty = True
        for method in ("preview_plane", "preview_slice_volume"):
            preview = getattr(page, method, None)
            if callable(preview):
                preview()  # selector geometry only; no numerical job or result replacement
                break
    return {"restored": restored, "skipped": skipped, "missing": sorted(missing),
            "partial": bool(state.get("legacy")) or bool(missing)}


def legacy_controls(content, aux, *, mode, view):
    """Recover only explicit native payload values; never guess currents/units."""
    state = {"format": FORMAT, "version": VERSION, "mode": mode, "view": view,
             "widgets": {}, "selections": {}, "scripted_widgets": {}, "legacy": True}
    if not isinstance(content, dict):
        return state
    values = content.get("map_settings", content)
    values = values if isinstance(values, dict) else {}
    mapping = {"span_mm": "span", "offset_mm": "offset", "plane": "plane",
               "resolution": "resolution", "field": "field", "component": "component",
               "slice_axis": "slice_axis", "slice_count": "slice_count", "view_type": "view_type",
               "logarithmic": "logarithmic", "show_grid": "show_grid", "include_scene": "include_scene"}
    def put(name, value, kind=None):
        kind = kind or ("checked" if type(value) is bool else "number")
        state["widgets"][name] = {"kind": kind, **({"data": value, "text": str(value)} if kind == "choice" else {"value": value})}
    for key, name in mapping.items():
        if key in values:
            put(name, values[key], "choice" if name in ("plane", "field", "component", "slice_axis", "view_type") else None)
    for key in ("active_coil_ids", "scene_object_ids", "outline_object_ids"):
        if isinstance(aux, dict) and key in aux:
            state["selections"][key] = copy.deepcopy(aux[key])
        elif key in values:
            state["selections"][key] = copy.deepcopy(values[key])
    if values.get("render_mode") in ("volume", "slices"):
        put("view_type", values["render_mode"], "choice")
    if "drive_coil_ids" in values:
        state["selections"]["active_coil_ids"] = copy.deepcopy(values["drive_coil_ids"])
    if isinstance(values.get("centre_mm"), list) and len(values["centre_mm"]) == 3:
        for i, value in enumerate(values["centre_mm"]):
            put(f"coordinates.{i}", value)
    # Old planar maps can retain their grid and semantic colourbar even though
    # the normal-axis offset and original Auto/manual flags were not saved.
    for trace in content.get("data", []) if isinstance(content.get("data", []), list) else []:
        if not isinstance(trace, dict) or trace.get("type") != "heatmap":
            continue
        x, y = trace.get("x"), trace.get("y")
        if (isinstance(x, list) and isinstance(y, list) and len(x) == len(y)
                and 2 <= len(x) <= 1000 and all(_finite_number(v) for v in x + y)
                and abs((max(x)-min(x))-(max(y)-min(y))) < 1e-8):
            put("span", max(x)-min(x));put("resolution", len(x))
        bar = trace.get("colorbar", {})
        title = bar.get("title", {}) if isinstance(bar, dict) else {}
        title = title.get("text", "") if isinstance(title, dict) else title
        first = title.split("<br>", 1)[0].split(";", 1)[0].strip() if isinstance(title, str) else ""
        semantic = None
        for field, unit in (("B", "µT"), ("H", "A/m")):
            for symbol, component in ((f"|{field}|", "magnitude"), (field+"x", "x"), (field+"y", "y"), (field+"z", "z")):
                if first in (f"{symbol} ({unit})", f"{symbol} {unit}"):
                    semantic = field, component
        if semantic:
            put("field", semantic[0], "choice");put("component", semantic[1], "choice")
            put("logarithmic", False)
            for name, key in (("colour_minimum", "zmin"), ("colour_maximum", "zmax")):
                if key in trace:
                    put(name, trace[key]);put(name+"_enabled", True)
        break
    # Old Plotly volumes contain explicit coordinate grids and semantic unit/component
    # metadata, but do not retain the original UI or custom opacity transfer curve.
    for trace in content.get("data", []) if isinstance(content.get("data", []), list) else []:
        if not isinstance(trace, dict) or trace.get("type") != "volume":
            continue
        try:
            axes = []
            for key in ("x", "y", "z"):
                arr = trace[key]
                if not isinstance(arr, list) or not 2 <= len(arr) <= 2_000_000 or any(type(x) not in (int, float) or not math.isfinite(x) for x in arr):
                    raise ValueError()
                axes.append(sorted(set(arr)))
            sizes = [len(axis) for axis in axes]
            widths = [axis[-1] - axis[0] for axis in axes]
            if len(set(sizes)) == 1 and min(widths) > 0 and max(widths) - min(widths) < 1e-8:
                put("view_type", "volume", "choice")
                put("resolution", sizes[0]);put("span", widths[0])
                for i, axis in enumerate(axes):
                    put(f"coordinates.{i}", (axis[0] + axis[-1]) / 2)
            meta = trace.get("meta", {})
            meta = meta if isinstance(meta, dict) else {}
            component = {"|B|": "magnitude", "Bx": "x", "By": "y", "Bz": "z"}.get(meta.get("component"))
            if meta.get("unit") == "µT" and component:
                put("field", "B", "choice");put("component", component, "choice")
                bar = trace.get("colorbar", {})
                title = bar.get("title", {}) if isinstance(bar, dict) else {}
                title = title.get("text", "") if isinstance(title, dict) else title
                # Old logarithmic colour values are exponents, not input µT.
                # Do not reinterpret them as linear limits.
                logarithmic = isinstance(title, str) and title.startswith("log₁₀")
                if logarithmic:
                    put("logarithmic", True)
                else:
                    if isinstance(title, str) and title.startswith(f"{meta.get('component')} (µT)"):
                        put("logarithmic", False)
                    for name, key in (("colour_minimum", "cmin"), ("colour_maximum", "cmax")):
                        if key in trace:
                            put(name, trace[key]);put(name + "_enabled", True)
        except (KeyError, TypeError, ValueError):
            pass
        break
    return state
