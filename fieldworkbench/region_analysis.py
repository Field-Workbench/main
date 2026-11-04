"""Shared playback/exposure analysis for resolved spatial regions.

Region providers supply an explicit hierarchy and weighted sample set through
the generic spatial-analysis contract. LPBA40, measurement volumes, and future
derived region sources therefore share the same frozen-basis calculations.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import numpy as np

from .spatial_analysis import (
    AnalysisRegionNode, AnalysisSampleSet, frame_region_metrics,
    playback_region_metrics, playback_region_metrics_streaming,
)


@dataclass(frozen=True)
class MeasurementRegions:
    children: tuple[AnalysisRegionNode, ...]
    key: str = "measurement_volumes"
    name: str = "Measurement volumes"
    label_ids: tuple[int, ...] = ()

    def walk(self):
        for child in self.children:
            yield from child.walk()


def encode(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def _emit_analysis_progress(callback, message, completed=None, total=None):
    if not callable(callback):
        return
    update = {"message": str(message)}
    if completed is not None:
        update["completed"] = int(completed)
    if total is not None:
        update["total"] = int(total)
    callback(update)


def _hierarchy_weighted_rms_moments(fixed, bases, samples, hierarchy):
    """Build every hierarchy row from one pass over the atomic region labels.

    A regional RMS quadratic form is a weighted mean of per-point second
    moments.  Parent rows can therefore be assembled exactly from the same
    unnormalised label accumulators as their children.  The older implementation
    rescanned the complete regional sample cloud once for *every* hierarchy row;
    this version scans each field/sample term once and only combines the small
    per-label accumulators afterwards.
    """
    fixed = np.asarray(fixed, dtype=np.float64).reshape(-1, 3)
    bases = np.asarray(bases, dtype=np.float64)
    labels = np.asarray(samples.label_ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(samples.weights_mm3, dtype=np.float64).reshape(-1)
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise ValueError("Regional waveform basis dimensions are invalid.")
    if labels.shape != (len(fixed),) or weights.shape != (len(fixed),):
        raise ValueError("Regional waveform labels and weights must match the field samples.")
    if not np.isfinite(fixed).all() or not np.isfinite(bases).all():
        raise ValueError("Regional waveform field bases contain non-finite values.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Regional waveform weights must be finite and positive.")

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    label_count = len(unique_labels)
    drive_count = bases.shape[0]
    mass = np.bincount(inverse, weights=weights, minlength=label_count).astype(np.float64)
    fixed_power = np.einsum("pk,pk->p", fixed, fixed, optimize=True)
    constant_sum = np.bincount(
        inverse, weights=weights * fixed_power, minlength=label_count
    ).astype(np.float64)

    fixed_basis = np.einsum("pk,cpk->pc", fixed, bases, optimize=True)
    linear_sum = np.empty((label_count, drive_count), dtype=np.float64)
    for drive_index in range(drive_count):
        linear_sum[:, drive_index] = np.bincount(
            inverse,
            weights=weights * fixed_basis[:, drive_index],
            minlength=label_count,
        )

    quadratic_sum = np.empty((label_count, drive_count, drive_count), dtype=np.float64)
    for left in range(drive_count):
        for right in range(left, drive_count):
            point_products = np.einsum(
                "pk,pk->p", bases[left], bases[right], optimize=True
            )
            accumulated = np.bincount(
                inverse, weights=weights * point_products, minlength=label_count
            )
            quadratic_sum[:, left, right] = accumulated
            quadratic_sum[:, right, left] = accumulated

    label_lookup = {int(label): index for index, label in enumerate(unique_labels)}
    moments = {}
    for node in hierarchy.walk():
        indices = [label_lookup[int(label)] for label in node.label_ids if int(label) in label_lookup]
        if not indices:
            continue
        indices_array = np.asarray(indices, dtype=np.int64)
        total_mass = float(np.sum(mass[indices_array]))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            continue
        constant = float(np.sum(constant_sum[indices_array]) / total_mass)
        linear = np.sum(linear_sum[indices_array], axis=0) / total_mass
        quadratic = np.sum(quadratic_sum[indices_array], axis=0) / total_mass
        moments[node.key] = {
            "constant": constant,
            "linear": np.asarray(linear, dtype=np.float64).tolist(),
            "quadratic": np.asarray(quadratic, dtype=np.float64).tolist(),
            "name": node.name,
        }
    return moments


def waveform_regions_payload(regions, timeline, drive_ids):
    """Attach a physical exposure timeline to an already-frozen regional basis.

    Regional second moments depend on the model/target/drive basis. Keeping the
    timeline assembly separate from field sampling lets the same compact payload
    machinery serve authoritative Exposure Analysis and viewer-side regional data.
    """
    moments = copy.deepcopy(regions) if isinstance(regions, dict) else {}
    if not moments:
        raise ValueError("Regional waveform basis is empty.")
    currents = np.asarray(timeline["analysis_currents"], dtype=np.float64)
    payload = {
        "times_b64": encode(timeline["analysis_times"], np.float64),
        "currents_b64": encode(currents, np.float64),
        "shape": list(currents.shape), "drive_ids": list(drive_ids),
        "regions": moments,
        "analysis_trace_reduced": bool(timeline.get("analysis_trace_reduced", False)),
        "analysis_spacing_s": timeline.get("analysis_spacing_s"),
        "analysis_compression": dict(timeline.get("analysis_compression") or {}),
        # Keep the analysis scope beside the frozen regional basis so every
        # analysis page can state exactly which physical interval its numbers
        # describe, independent of the viewer's current preview duration.
        "statistics_scope": str(timeline.get("statistics_scope", "full") or "full"),
        "statistics_duration_s": float(
            timeline.get(
                "statistics_duration_s",
                float(timeline["analysis_times"][-1]) - float(timeline["analysis_times"][0]),
            )
        ),
    }
    frozen = timeline.get("frozen_analysis_source")
    if not isinstance(frozen, dict):
        freeze = timeline.get("frozen_analysis_source_factory")
        frozen = freeze() if callable(freeze) else None
    if isinstance(frozen, dict):
        payload["authoritative_analysis"] = copy.deepcopy(frozen)
    return payload


def waveform_basis_payload(fixed, bases, samples, hierarchy, timeline, drive_ids):
    """Retain vector second moments, not an FFT of scalar |B|.

    This compact quadratic form preserves spatially averaged vector spectral
    power, including opposed fields which would cancel in a mean vector.
    Hierarchy rows are assembled from per-label accumulators so parent/child
    overlap does not repeatedly rescan the atlas sample cloud.
    """
    moments = _hierarchy_weighted_rms_moments(fixed, bases, samples, hierarchy)
    return waveform_regions_payload(moments, timeline, drive_ids)


class _FrozenExposureAnalysisAdapter:
    """Minimal saved-result shim for the chronological exposure resolver."""

    def __init__(self, source: dict):
        self._source = source
        self._trace_cache = {}

    def _waveform_source_trace(self, waveform_id):
        key = str(waveform_id)
        if key in self._trace_cache:
            times, values, metadata = self._trace_cache[key]
            return times.copy(), values.copy(), copy.deepcopy(metadata)
        raw = self._source.get("source_traces", {}).get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"Frozen analysis is missing waveform source {key!r}.")
        try:
            count = int(raw["sample_count"])
            times = np.frombuffer(
                base64.b64decode(raw["times_b64"]), dtype=np.float64
            ).copy()
            values = np.frombuffer(
                base64.b64decode(raw["values_b64"]), dtype=np.float64
            ).copy()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Frozen waveform source {key!r} is malformed.") from error
        if (
            count < 2
            or len(times) != count
            or len(values) != count
            or not np.isfinite(times).all()
            or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Frozen waveform source {key!r} has invalid samples.")
        metadata = copy.deepcopy(
            raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        )
        self._trace_cache[key] = (times, values, metadata)
        return times.copy(), values.copy(), copy.deepcopy(metadata)

    def coil_electrical_load(self, coil_id):
        key = str(coil_id)
        raw = self._source.get("coil_loads", {}).get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"Frozen analysis is missing electrical load data for coil {key!r}.")
        return copy.deepcopy(raw)

    def resolve_exposure_source_network(self, groups, **kwargs):
        """Reuse the live shared-source electrical solver with frozen inputs.

        The chronological resolver calls this method on its adapter.  Saved-map
        replay deliberately implements only the small adapter surface needed by
        that resolver, so forward the network solve to :class:`StudioAdapter`
        while keeping waveform traces and coil loads sourced from this frozen
        object.
        """
        from .studio_adapter import StudioAdapter

        return StudioAdapter.resolve_exposure_source_network(self, groups, **kwargs)


_AUTHORITATIVE_ANALYSIS_SCHEMA = "fieldworkbench.authoritative-exposure-analysis"
_AUTHORITATIVE_ANALYSIS_VERSION = 1
_AUTHORITATIVE_ANALYSIS_MODES = {"verified_packets", "replay_recipe"}


def _authoritative_analysis_source(payload):
    """Validate and return the frozen exposure-analysis source.

    The regional field basis and the frozen current stream are coupled by drive
    column order.  Treat that order as part of the saved-result integrity
    contract so a malformed/corrupted payload fails closed instead of applying
    one coil's current to another coil's field basis.
    """
    frozen = payload.get("authoritative_analysis") if isinstance(payload, dict) else None
    if not isinstance(frozen, dict):
        raise ValueError(
            "This saved result does not contain an authoritative analysis source. "
            "Regenerate the map with a newer Field Workbench version."
        )
    try:
        version = int(frozen.get("version", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The saved result uses an unsupported authoritative analysis format."
        ) from error
    if (
        frozen.get("schema") != _AUTHORITATIVE_ANALYSIS_SCHEMA
        or version != _AUTHORITATIVE_ANALYSIS_VERSION
    ):
        raise ValueError(
            "The saved result uses an unsupported authoritative analysis format."
        )

    mode = str(frozen.get("mode", ""))
    if mode not in _AUTHORITATIVE_ANALYSIS_MODES:
        raise ValueError(f"Unknown authoritative analysis replay mode: {mode!r}.")

    raw_drive_ids = frozen.get("drive_ids", [])
    if not isinstance(raw_drive_ids, (list, tuple)):
        raise ValueError("The saved result contains malformed authoritative drive metadata.")
    frozen_drive_ids = [str(value) for value in raw_drive_ids]
    if len(set(frozen_drive_ids)) != len(frozen_drive_ids):
        raise ValueError("The saved result contains duplicate authoritative drive ids.")

    raw_payload_drive_ids = payload.get("drive_ids", []) if isinstance(payload, dict) else []
    if raw_payload_drive_ids is not None and not isinstance(raw_payload_drive_ids, (list, tuple)):
        raise ValueError("The saved result contains malformed regional drive metadata.")
    payload_drive_ids = [str(value) for value in (raw_payload_drive_ids or [])]
    if payload_drive_ids and frozen_drive_ids != payload_drive_ids:
        raise ValueError(
            "The authoritative analysis drive order does not match the frozen regional "
            "field basis. Regenerate the map."
        )
    return frozen


def authoritative_analysis_packet_factory(payload):
    """Return a replayable full-resolution packet factory from a saved result.

    Verified periodic results replay their frozen transient/cycle/remainder
    packets directly. Other exposures regenerate the original event-aware
    chronological stream from the frozen protocol, waveform commands and coil
    electrical loads, without consulting the live project.
    """
    frozen = _authoritative_analysis_source(payload)
    drive_ids = [str(value) for value in frozen.get("drive_ids", [])]
    mode = str(frozen.get("mode", ""))

    def verified_packets():
        packets = frozen.get("packets")
        if not isinstance(packets, list) or not packets:
            raise ValueError("Verified frozen analysis does not contain reusable packets.")
        for raw in packets:
            if not isinstance(raw, dict):
                raise ValueError("Verified frozen analysis contains a malformed packet.")
            try:
                times = np.frombuffer(
                    base64.b64decode(raw["times_b64"]), dtype=np.float64
                ).copy()
                shape = tuple(int(value) for value in raw["shape"])
                currents = np.frombuffer(
                    base64.b64decode(raw["currents_b64"]), dtype=np.float64
                ).copy().reshape(shape)
                repeat_count = int(raw.get("repeat_count", 1))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Verified frozen analysis contains an invalid packet.") from error
            if (
                len(shape) != 2
                or currents.shape[0] != len(times)
                or currents.shape[1] != len(drive_ids)
                or repeat_count < 1
                or len(times) < 1
                or not np.isfinite(times).all()
                or not np.isfinite(currents).all()
                or (len(times) > 1 and np.any(np.diff(times) <= 0.0))
            ):
                raise ValueError("Verified frozen analysis packet dimensions are invalid.")
            yield times, currents, repeat_count

    def replay_recipe():
        from .exposures import normalize_exposure_protocol
        from .studio_adapter import StudioAdapter
        from .waveform import _exposure_analysis_clock, _iter_exposure_analysis_time_chunks

        protocol = frozen.get("protocol")
        try:
            state = normalize_exposure_protocol(protocol)
            spacing = float(frozen["analysis_spacing_s"])
        except Exception as error:  # noqa: BLE001 - saved-result validation boundary
            raise ValueError("Frozen exposure analysis recipe is malformed.") from error
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("Frozen exposure analysis spacing is invalid.")
        adapter = _FrozenExposureAnalysisAdapter(frozen)
        _computed_spacing, source_clocks = _exposure_analysis_clock(adapter, state)
        chunks = _iter_exposure_analysis_time_chunks(state, spacing, source_clocks)
        for resolved in StudioAdapter.iter_exposure_currents_at_times(
            adapter, state, chunks
        ):
            times = np.asarray(resolved.get("time_s", []), dtype=np.float64).reshape(-1)
            current_map = resolved.get("coil_currents_a", {})
            if not isinstance(current_map, dict):
                raise ValueError("Frozen exposure replay returned invalid coil-current data.")
            columns = []
            for coil_id in drive_ids:
                values = np.asarray(
                    current_map.get(coil_id, []), dtype=np.float64
                ).reshape(-1)
                if len(values) != len(times):
                    raise ValueError(
                        f"Frozen exposure replay did not reproduce coil {coil_id!r}."
                    )
                columns.append(values)
            currents = (
                np.column_stack(columns)
                if columns
                else np.zeros((len(times), 0), dtype=np.float64)
            )
            yield times, currents

    if mode == "verified_packets":
        return verified_packets
    if mode == "replay_recipe":
        return replay_recipe
    raise ValueError(f"Unknown authoritative analysis replay mode: {mode!r}.")




def measurement_playback_payload(
    adapter, timeline, drive_ids, controlled_ids, progress_callback=None, *,
    trace=None, waveform_metadata=None, sequence_steps=None, sequence_loop=False,
    frame_only=False,
):
    # Local import avoids a waveform <-> region_analysis import cycle.
    from .waveform import (
        field_bases_for_drives, MAX_BRAIN_FRAME_METRIC_CACHE_BYTES,
        _brain_playback_run_metadata, _emit_render_task_progress,
        _render_task_progress_relay,
    )

    nodes, points, labels, weights, descriptions, volumes = [], [], [], [], [], []
    measurement_entries = []
    for entry in adapter.list_objects():
        obj = adapter.get_object(entry["id"])
        if obj.get("derived") or obj.get("role") != "measurement":
            continue
        if not adapter.supports_measurement(entry["id"]):
            continue
        measurement_entries.append(entry)

    if measurement_entries:
        _emit_render_task_progress(
            progress_callback,
            task_id="measurement_basis",
            stage="Preparing measurement-volume samples…",
            completed=0,
            total=max(1, (len(drive_ids) + 1) * 1000),
            unit="basis work",
            detail=f"Preparing samples for {len(measurement_entries):,} measurement volume(s).",
        )

    point_indices = []
    point_keys = []
    used_keys = {str(entry["id"]) for entry in measurement_entries}
    next_point_label = len(measurement_entries) + 1
    point_offset = 0
    for measurement_index, entry in enumerate(measurement_entries, start=1):
        object_id = str(entry["id"])
        object_label = str(entry.get("label") or object_id)
        _emit_render_task_progress(
            progress_callback,
            task_id="measurement_basis",
            stage=f"Preparing measurement samples — {object_label}",
            completed=0,
            total=max(1, (len(drive_ids) + 1) * 1000),
            unit="basis work",
            detail=f"Measurement volume {measurement_index:,}/{len(measurement_entries):,}: {object_label}",
        )
        sample = adapter.measurement_sample_points(object_id)
        positions = np.asarray(sample["world_points_m"], dtype=float).reshape(-1, 3)
        if not len(positions):
            raise ValueError(f"Measurement volume {sample['label']} contains no sample points.")
        label = len(nodes) + 1
        user_points = sample["sampling_mode"] == "user_defined"
        name = str(sample["label"]) + (" [samples]" if user_points else "")
        nodes.append(AnalysisRegionNode(str(entry["id"]), name, (label,), ()))
        points.append(positions)
        labels.extend([label] * len(positions))
        # Weights are equal within an object. Geometric volume is separately
        # reported; it is never inferred from custom point density.
        volume = float(sample["volume_cm3"])
        weights.extend([max(volume * 1000, 1.0) / len(positions)] * len(positions))
        volumes.append(volume)
        descriptions.append({"key": str(entry["id"]), "name": name,
                             "label_id": label, "sampling_mode": sample["sampling_mode"],
                             "quality": sample.get("quality"), "sample_count": len(positions),
                             "volume_cm3": volume})
        if frame_only and user_points:
            well = entry.get("well_plate") or {}
            well_labels = well.get("well_labels", [])
            well_positions = np.asarray(well.get("sample_points_mm", []), dtype=float)
            local_mm = np.asarray(sample["local_points_m"]) * 1000
            named_wells = (
                len(well_labels) == len(positions)
                and well_positions.shape == local_mm.shape
                and np.allclose(well_positions, local_mm, rtol=0, atol=1e-9)
            )
            children = []
            # Bound table size for large imported point sets. The aggregate
            # still uses every point, including points with no individual row.
            indices = np.linspace(0, len(positions) - 1, min(len(positions), 2000), dtype=int)
            for index in indices:
                key = f"{object_id}:point:{index + 1}"
                while key in used_keys:
                    key += ":point"
                used_keys.add(key)
                child = {
                    "key": key,
                    "name": str(well_labels[index]) if named_wells else f"Point {index + 1}",
                    "label_id": next_point_label,
                    "position_mm": (positions[index] * 1000).tolist(),
                }
                children.append(child)
                point_indices.append(point_offset + index)
                point_keys.append(key)
                next_point_label += 1
            descriptions[-1]["children"] = children
        point_offset += len(positions)
    if not nodes:
        return {"objects": [], "metrics": None, "waveform": None}
    points = np.concatenate(points)
    _emit_render_task_progress(
        progress_callback,
        task_id="measurement_basis",
        stage="Measurement-volume samples ready; calculating field bases…",
        completed=0,
        total=max(1, (len(drive_ids) + 1) * 1000),
        unit="basis work",
        detail=f"Prepared {len(points):,} sample point(s) across {len(nodes):,} measurement volume(s).",
    )
    labels = np.asarray(labels, dtype=np.int32)
    counts = {int(label): int(np.sum(labels == label)) for label in range(1, len(nodes)+1)}
    label_volumes_mm3 = {
        int(index + 1): max(float(volume) * 1000.0, 1.0)
        for index, volume in enumerate(volumes)
    }
    samples = AnalysisSampleSet(
        points, labels, np.asarray(weights), np.arange(len(points)), counts, counts, 1.0,
        label_volumes_mm3=label_volumes_mm3,
    )
    hierarchy = MeasurementRegions(tuple(nodes))
    measurement_basis_progress = _render_task_progress_relay(
        progress_callback,
        "measurement_basis",
        default_unit="basis work",
    )
    fixed, bases = field_bases_for_drives(
        adapter, coil_ids=drive_ids, zero_coil_ids=controlled_ids,
        points_m=points, field="B", progress_callback=measurement_basis_progress,
    )
    if not np.isfinite(fixed).all() or not np.isfinite(bases).all():
        raise ValueError("Measurement samples contain non-finite magnetic fields; check source intersections.")
    source_count = len(timeline["frame_times"])
    max_frames = max(2, MAX_BRAIN_FRAME_METRIC_CACHE_BYTES // max(1, (len(nodes) + len(point_indices)) * 7 * 4))
    indices = np.unique(np.rint(np.linspace(0, source_count-1, min(source_count, max_frames))).astype(int))

    def frame_progress(completed, total):
        _emit_render_task_progress(
            progress_callback,
            task_id="measurement_frames",
            stage="Building measurement-volume frame metrics…",
            completed=completed,
            total=total,
            unit="frames",
            detail=f"Measurement playback frames {completed:,}/{total:,}",
        )

    metric_currents = np.asarray(timeline["frame_currents"])[indices]
    frames = frame_region_metrics(
        fixed, bases, metric_currents, samples, hierarchy,
        progress_callback=frame_progress,
    )
    frames["values"][:, :, 6] = volumes
    if point_indices:
        point_fixed = fixed[point_indices]
        point_bases = bases[:, point_indices]
        values = np.full(
            (len(indices), len(nodes) + len(point_indices), 7), np.nan, dtype=np.float32
        )
        values[:, :len(nodes)] = frames["values"]
        for frame_index, current in enumerate(metric_currents):
            vectors = point_fixed + np.einsum("c,cpk->pk", current, point_bases)
            magnitude_uT = np.linalg.norm(vectors, axis=1) * 1e6
            # RMS, mean, P95 and peak are all the point's field magnitude.
            # Spatial uniformity, alignment and volume require a region.
            values[frame_index, len(nodes):, :4] = magnitude_uT[:, None]
        frames["values"] = values
        frames["keys"].extend(point_keys)
        frames["names"].extend(child["name"] for obj in descriptions for child in obj.get("children", []))
        frames["sample_counts"].extend([1] * len(point_indices))
        frames["voxel_counts"].extend([0] * len(point_indices))
    metrics = {
        "keys": frames["keys"],
        "names": frames["names"],
        "shape": list(frames["values"].shape),
        "values_b64": encode(frames["values"], np.float32),
        "sample_counts": frames["sample_counts"],
        "voxel_counts": frames["voxel_counts"],
        "source_frame_count": source_count,
        "sampled_frame_indices_b64": encode(indices, np.int32),
    }
    # Retain at most 2000 highlight points per object; numerical samples are untouched.
    highlights = []
    for node in nodes:
        selected = points[labels == node.label_ids[0]] * 1000
        stride = max(1, int(np.ceil(len(selected)/2000)))
        highlights.append({"label_id": node.label_ids[0], "positions": selected[::stride].tolist()})
    highlights.extend(
        {"label_id": child["label_id"], "positions": [child["position_mm"]]}
        for obj in descriptions
        for child in obj.get("children", [])
    )
    result = {"objects": descriptions, "metrics": metrics, "highlights": highlights}
    if frame_only:
        return result

    statistics_scope = str(timeline.get("statistics_scope", "full") or "full").strip().lower()
    statistics_label = "preview" if statistics_scope == "preview" else "full-exposure"

    def streaming_summary_progress(completed, total):
        _emit_render_task_progress(
            progress_callback,
            task_id="measurement_summary",
            stage=f"Summarizing {statistics_label} measurement statistics…",
            completed=completed,
            total=total,
            unit="logical samples",
            detail=f"Authoritative {statistics_label} samples {completed:,}/{total:,}",
        )

    def region_summary_progress(completed, total):
        _emit_render_task_progress(
            progress_callback,
            task_id="measurement_summary",
            stage=f"Summarizing {statistics_label} measurement statistics…",
            completed=completed,
            total=total,
            unit="regions",
            detail=f"Measurement regions {completed:,}/{total:,}",
        )

    if callable(timeline.get("analysis_packet_factory") or timeline.get("analysis_chunk_factory")):
        analysis_factory = timeline.get("analysis_packet_factory") or timeline["analysis_chunk_factory"]
        summary = playback_region_metrics_streaming(
            fixed, bases, analysis_factory, samples, hierarchy,
            progress_callback=streaming_summary_progress,
            analysis_sample_count=int(timeline.get("analysis_sample_count_estimate", 0) or 0),
        )
    else:
        summary = playback_region_metrics(
            fixed, bases, timeline["analysis_currents"], timeline["analysis_times"],
            frames, np.asarray(timeline["frame_times"])[indices], samples, hierarchy,
            full_timeline=False, progress_callback=region_summary_progress,
        )
    rows = [summary[node.key] for node in nodes]
    summary_values = np.asarray([(r.playback_rms_uT, r.mean_uT, r.max_p95_uT, r.absolute_peak_uT,
                                  r.b_time_uT_s, r.b2_time_uT2_s, r.volume_cm3) for r in rows])
    metrics["playback_summary"] = {"shape": list(summary_values.shape), "values_b64": encode(summary_values, np.float64),
                    "peak_times_s_b64": encode([r.peak_time_s for r in rows], np.float64),
                    "playback": {"exposure_time_s": float(timeline.get("physical_duration_s", timeline["analysis_times"][-1]-timeline["analysis_times"][0])),
                                 "preview_duration_s": float(timeline.get("preview_duration_s", timeline["analysis_times"][-1]-timeline["analysis_times"][0])),
                                 "statistics_scope": str(timeline.get("statistics_scope", "full")),
                                 "statistics_duration_s": float(timeline.get("statistics_duration_s", timeline["analysis_times"][-1]-timeline["analysis_times"][0])),
                                 "frame_count": source_count, "analysis_sample_count": int(timeline.get("analysis_sample_count_estimate", len(timeline["analysis_times"]))),
                                 "analysis_sample_count_estimated": callable(timeline.get("analysis_chunk_factory")),
                                 "regional_frame_sample_count": len(indices), "atlas_sample_count": len(points)}}
    if trace is not None:
        metrics["playback_summary"]["playback"] = _brain_playback_run_metadata(
            trace, frame_count=source_count, analysis_sample_count=int(timeline.get("analysis_sample_count_estimate", len(timeline["analysis_times"]))),
            summary_sample_count=len(indices), atlas_sample_count=len(points),
            analysis_sample_count_estimated=callable(timeline.get("analysis_chunk_factory")),
            waveform_metadata=waveform_metadata, sequence_steps=sequence_steps, sequence_loop=sequence_loop,
        )
    result["waveform"] = waveform_basis_payload(fixed, bases, samples, hierarchy, timeline, drive_ids)
    return result


MAX_RETAINED_INTERVALS = 2000


def current_threshold_intervals(times, values, threshold):
    """Threshold crossings of the piecewise-linear signed current record."""
    intervals = []
    tolerance = np.finfo(float).eps * max(1., float(np.max(np.abs(times)))) * 8
    for sign in (1., -1.):
        a, b = sign * values[:-1], sign * values[1:]
        mask = np.maximum(a, b) > threshold
        if not np.any(mask):
            continue
        fraction = np.divide(threshold-a, b-a, out=np.zeros_like(a), where=(b != a))
        starts = (times[:-1] + np.where(a > threshold, 0., fraction) * np.diff(times))[mask]
        ends = (times[:-1] + np.where(b > threshold, 1., fraction) * np.diff(times))[mask]
        breaks = np.flatnonzero(starts[1:] > ends[:-1] + tolerance) + 1
        intervals.extend(zip(starts[np.r_[0, breaks]], ends[np.r_[breaks-1, len(ends)-1]]))
    return sorted((float(start), float(end)) for start, end in intervals)


def _time_domain_packet_factory(payload):
    """Return the authoritative time-domain source for regional analyses.

    Exposure maps created by the Phase-1 format carry a replayable authoritative
    source and always use it, even when their retained display trace happens to
    contain every sample. Non-exposure waveform maps do not have that source, so
    an unreduced retained trace remains their authoritative timeline.
    """
    if isinstance(payload.get("authoritative_analysis"), dict):
        return authoritative_analysis_packet_factory(payload)
    if payload.get("analysis_trace_reduced", False):
        raise ValueError(
            "This saved result contains only a reduced display trace and no "
            "authoritative analysis source. Regenerate the map."
        )
    try:
        times = np.frombuffer(
            base64.b64decode(payload["times_b64"]), dtype=np.float64
        ).copy()
        currents = np.frombuffer(
            base64.b64decode(payload["currents_b64"]), dtype=np.float64
        ).copy().reshape(payload["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Regional waveform analysis payload is malformed.") from error
    drive_count = len(payload.get("drive_ids", []))
    if (
        times.ndim != 1
        or len(times) < 2
        or currents.shape != (len(times), drive_count)
        or not np.isfinite(times).all()
        or not np.isfinite(currents).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            "Waveform analysis needs at least two increasing physical time samples."
        )

    def retained_factory():
        yield times.copy(), currents.copy(), 1

    return retained_factory


def _iter_time_domain_packets(payload):
    factory = _time_domain_packet_factory(payload)
    drive_count = len(payload.get("drive_ids", []))
    for packet in factory():
        if not isinstance(packet, (tuple, list)) or len(packet) not in (2, 3):
            raise ValueError("Authoritative regional analysis returned an invalid packet.")
        repeat_count = int(packet[2]) if len(packet) == 3 else 1
        times = np.asarray(packet[0], dtype=np.float64).reshape(-1)
        currents = np.asarray(packet[1], dtype=np.float64)
        if (
            repeat_count < 1
            or len(times) < 1
            or currents.shape != (len(times), drive_count)
            or not np.isfinite(times).all()
            or not np.isfinite(currents).all()
            or np.any(np.diff(times) < 0.0)
            or (
                repeat_count > 1
                and (len(times) < 2 or float(times[-1]) <= float(times[0]))
            )
        ):
            raise ValueError("Authoritative regional analysis packet dimensions are invalid.")
        yield times, currents, repeat_count


def _time_domain_work_sample_count(payload):
    """Return deterministic sample-work units for time-domain progress.

    Verified-cycle payloads store the exact packets that will be evaluated, so
    their retained packet shapes give an exact work count regardless of logical
    repeat_count. Retained short traces are exact as well. Replay recipes retain
    a conservative event-aware sample estimate; that path may finish with a
    small final jump to 100%, but never pretends repeated logical cycles are
    being expanded when they are not.
    """
    frozen = payload.get("authoritative_analysis") if isinstance(payload, dict) else None
    if isinstance(frozen, dict):
        validated = _authoritative_analysis_source(payload)
        mode = str(validated.get("mode", ""))
        if mode == "verified_packets":
            packets = validated.get("packets")
            if isinstance(packets, list) and packets:
                total = 0
                for raw in packets:
                    if not isinstance(raw, dict):
                        break
                    shape = raw.get("shape")
                    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                        break
                    try:
                        rows = int(shape[0])
                    except (TypeError, ValueError):
                        break
                    if rows < 1:
                        break
                    total += rows
                else:
                    if total > 0:
                        return int(total)
        elif mode == "replay_recipe":
            try:
                estimate = int(validated.get("analysis_sample_count_estimate", 0) or 0)
            except (TypeError, ValueError):
                estimate = 0
            if estimate > 0:
                return estimate

    try:
        shape = payload.get("shape", [])
        retained = int(shape[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        retained = 0
    return max(1, retained)


def _region_moments(payload, key, drive_count):
    try:
        region = payload["regions"][key]
        constant = float(region["constant"])
        linear = np.asarray(region["linear"], dtype=np.float64)
        quadratic = np.asarray(region["quadratic"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Regional waveform second moments are malformed.") from error
    if linear.shape != (drive_count,) or quadratic.shape != (drive_count, drive_count):
        raise ValueError("Region waveform moment dimensions are invalid.")
    return constant, linear, quadratic


def _region_spatial_rms_values_direct(constant, linear, quadratic, currents):
    currents = np.asarray(currents, dtype=np.float64)
    power_t2 = (
        constant
        + 2.0 * currents @ linear
        + np.einsum(
            "tc,cd,td->t", currents, quadratic, currents, optimize=True
        )
    )
    return np.sqrt(np.maximum(0.0, power_t2)) * 1e6


def _region_spatial_rms_values(constant, linear, quadratic, currents):
    """Evaluate exact RMS values while collapsing repeated current states.

    DAC/burst exposures commonly contain long plateaus and revisit the same
    current vectors.  The quadratic regional field expression only depends on
    the current vector, so exact repeated states can be evaluated once without
    changing the numerical result.  Continuously varying signals take the fast
    vectorised direct path with only one adjacent-state comparison.
    """
    currents = np.asarray(currents, dtype=np.float64)
    if currents.ndim != 2 or len(currents) < 3:
        return _region_spatial_rms_values_direct(constant, linear, quadratic, currents)

    changed = np.ones(len(currents), dtype=bool)
    changed[1:] = np.any(currents[1:] != currents[:-1], axis=1)
    run_starts = np.flatnonzero(changed)
    if len(run_starts) == len(currents):
        return _region_spatial_rms_values_direct(constant, linear, quadratic, currents)
    # A lone repeated endpoint in an otherwise continuous signal is not worth
    # sorting/deduplicating. Keep the direct BLAS/einsum path unless plateaus
    # remove a meaningful fraction of the rows (small packets stay cheap).
    if len(currents) >= 32 and len(run_starts) > int(len(currents) * 0.90):
        return _region_spatial_rms_values_direct(constant, linear, quadratic, currents)

    run_currents = currents[run_starts]
    if len(run_currents) >= 4:
        unique_currents, inverse = np.unique(run_currents, axis=0, return_inverse=True)
        if len(unique_currents) < len(run_currents):
            run_values = _region_spatial_rms_values_direct(
                constant, linear, quadratic, unique_currents
            )[inverse]
        else:
            run_values = _region_spatial_rms_values_direct(
                constant, linear, quadratic, run_currents
            )
    else:
        run_values = _region_spatial_rms_values_direct(
            constant, linear, quadratic, run_currents
        )
    run_lengths = np.diff(np.append(run_starts, len(currents)))
    return np.repeat(run_values, run_lengths)


def region_spatial_rms_trace(payload, key):
    """Return a complete retained regional spatial-RMS trace when one exists.

    This helper intentionally remains a retained-trace operation for spectral/UI
    callers. Long-exposure cumulative analyses use the packet-stream routines
    below instead of materializing the physical exposure.
    """
    times, currents = _waveform_timeline(payload)
    constant, linear, quadratic = _region_moments(payload, key, currents.shape[1])
    rms_uT = _region_spatial_rms_values(constant, linear, quadratic, currents)
    return np.asarray(times, dtype=np.float64), np.asarray(rms_uT, dtype=np.float64)


def _time_tolerance(*values):
    finite = [abs(float(value)) for value in values if value is not None and np.isfinite(value)]
    return np.finfo(float).eps * max([1.0, *finite]) * 16.0


def _empty_stats():
    return {"count": 0, "total": 0.0, "min": np.inf, "max": -np.inf}


def _stats_add(stats, value, count=1):
    count = int(count)
    if count <= 0:
        return
    value = float(value)
    stats["count"] += count
    stats["total"] += value * count
    stats["min"] = min(float(stats["min"]), value)
    stats["max"] = max(float(stats["max"]), value)


def _stats_merge(*stats_values):
    result = _empty_stats()
    for stats in stats_values:
        if not stats or int(stats.get("count", 0)) <= 0:
            continue
        result["count"] += int(stats["count"])
        result["total"] += float(stats["total"])
        result["min"] = min(float(result["min"]), float(stats["min"]))
        result["max"] = max(float(result["max"]), float(stats["max"]))
    return result


def _interval_group_from_intervals(intervals, *, preview_limit=MAX_RETAINED_INTERVALS):
    clean = [(float(start), float(end)) for start, end in intervals if float(end) > float(start)]
    if not clean:
        return {
            "count": 0,
            "total_s": 0.0,
            "first": None,
            "last": None,
            "internal_widths": _empty_stats(),
            "gaps": _empty_stats(),
            "preview": [],
        }
    widths = [end - start for start, end in clean]
    internal = _empty_stats()
    for width in widths[1:-1]:
        _stats_add(internal, width)
    gaps = _empty_stats()
    for left, right in zip(clean[:-1], clean[1:], strict=True):
        gap = float(right[0] - left[1])
        if gap < -_time_tolerance(left[1], right[0]):
            raise ValueError("Threshold intervals overlap unexpectedly.")
        _stats_add(gaps, max(0.0, gap))
    return {
        "count": len(clean),
        "total_s": float(sum(widths)),
        "first": clean[0],
        "last": clean[-1],
        "internal_widths": internal,
        "gaps": gaps,
        "preview": clean[:preview_limit],
    }


def _interval_width_stats(group):
    stats = _stats_merge(group.get("internal_widths"))
    count = int(group.get("count", 0))
    if count <= 0:
        return stats
    first = group["first"]
    _stats_add(stats, float(first[1] - first[0]))
    if count > 1:
        last = group["last"]
        _stats_add(stats, float(last[1] - last[0]))
    return stats


def _preview_append(preview, interval, limit):
    start, end = map(float, interval)
    if end <= start:
        return
    if preview:
        previous_start, previous_end = preview[-1]
        tolerance = _time_tolerance(previous_end, start)
        if start <= previous_end + tolerance:
            preview[-1] = (previous_start, max(previous_end, end))
            return
    if len(preview) < limit:
        preview.append((start, end))


def _combine_interval_groups(left, right, *, preview_limit=MAX_RETAINED_INTERVALS):
    if int(left.get("count", 0)) <= 0:
        return copy.deepcopy(right)
    if int(right.get("count", 0)) <= 0:
        return copy.deepcopy(left)

    a_count = int(left["count"])
    b_count = int(right["count"])
    a_first, a_last = left["first"], left["last"]
    b_first, b_last = right["first"], right["last"]
    gap = float(b_first[0] - a_last[1])
    tolerance = _time_tolerance(a_last[1], b_first[0])
    if gap < -tolerance:
        raise ValueError("Threshold interval groups overlap unexpectedly.")
    merge_boundary = gap <= tolerance

    internal = _stats_merge(left["internal_widths"], right["internal_widths"])
    gaps = _stats_merge(left["gaps"], right["gaps"])
    total_s = float(left["total_s"] + right["total_s"])

    if merge_boundary:
        merged = (float(a_last[0]), float(b_first[1]))
        total_s += gap
        count = a_count + b_count - 1
        if a_count > 1 and b_count > 1:
            _stats_add(internal, merged[1] - merged[0])
        first = merged if a_count == 1 else a_first
        last = merged if b_count == 1 else b_last
    else:
        count = a_count + b_count
        _stats_add(gaps, gap)
        if a_count > 1:
            _stats_add(internal, a_last[1] - a_last[0])
        if b_count > 1:
            _stats_add(internal, b_first[1] - b_first[0])
        first, last = a_first, b_last

    preview = [tuple(value) for value in left.get("preview", [])[:preview_limit]]
    # If all left intervals are represented, the boundary interaction and as
    # many right intervals as fit can be represented exactly as well.
    if a_count <= preview_limit:
        for interval in right.get("preview", []):
            _preview_append(preview, interval, preview_limit)
            if len(preview) >= preview_limit and int(right["count"]) > len(right.get("preview", [])):
                break

    return {
        "count": count,
        "total_s": total_s,
        "first": first,
        "last": last,
        "internal_widths": internal,
        "gaps": gaps,
        "preview": preview[:preview_limit],
    }


def _repeat_pattern_preview(intervals, cycle_start, period, repeat_count, boundary_merge, limit):
    if not intervals or limit <= 0:
        return []
    intervals = [(float(start), float(end)) for start, end in intervals]
    count = len(intervals)
    if boundary_merge and count == 1:
        start, end = intervals[0]
        return [(start, end + (repeat_count - 1) * period)]

    preview = []
    if boundary_merge:
        first, last = intervals[0], intervals[-1]
        _preview_append(preview, first, limit)
        for cycle_index in range(repeat_count):
            shift = cycle_index * period
            for interval in intervals[1:-1]:
                _preview_append(
                    preview, (interval[0] + shift, interval[1] + shift), limit
                )
                if len(preview) >= limit:
                    return preview
            if cycle_index < repeat_count - 1:
                merged = (
                    last[0] + shift,
                    first[1] + (cycle_index + 1) * period,
                )
                _preview_append(preview, merged, limit)
                if len(preview) >= limit:
                    return preview
        _preview_append(
            preview,
            (
                last[0] + (repeat_count - 1) * period,
                last[1] + (repeat_count - 1) * period,
            ),
            limit,
        )
        return preview

    for cycle_index in range(repeat_count):
        shift = cycle_index * period
        for interval in intervals:
            _preview_append(
                preview, (interval[0] + shift, interval[1] + shift), limit
            )
            if len(preview) >= limit:
                return preview
    return preview


def _repeat_interval_group(
    intervals, cycle_start, cycle_end, repeat_count, *, preview_limit=MAX_RETAINED_INTERVALS
):
    repeat_count = int(repeat_count)
    if repeat_count < 1:
        raise ValueError("Repeated threshold analysis requires a positive repeat count.")
    if repeat_count == 1:
        return _interval_group_from_intervals(intervals, preview_limit=preview_limit)
    cycle_start = float(cycle_start)
    cycle_end = float(cycle_end)
    period = cycle_end - cycle_start
    if period <= 0.0:
        raise ValueError("Repeated threshold analysis requires a positive cycle span.")
    intervals = [(float(start), float(end)) for start, end in intervals]
    if not intervals:
        return _interval_group_from_intervals([], preview_limit=preview_limit)

    widths = [end - start for start, end in intervals]
    first = intervals[0]
    last = intervals[-1]
    boundary_gap = float(first[0] + period - last[1])
    tolerance = _time_tolerance(first[0], last[1], cycle_start, cycle_end)
    if boundary_gap < -tolerance:
        raise ValueError("Repeated threshold intervals overlap the cycle boundary.")
    boundary_merge = boundary_gap <= tolerance

    if boundary_merge and len(intervals) == 1:
        merged = (first[0], last[1] + (repeat_count - 1) * period)
        return _interval_group_from_intervals([merged], preview_limit=preview_limit)

    internal = _empty_stats()
    gaps = _empty_stats()
    if boundary_merge:
        for width in widths[1:-1]:
            _stats_add(internal, width, repeat_count)
        merged_width = widths[-1] + boundary_gap + widths[0]
        _stats_add(internal, merged_width, repeat_count - 1)
        for left, right in zip(intervals[:-1], intervals[1:], strict=True):
            _stats_add(gaps, max(0.0, right[0] - left[1]), repeat_count)
        count = repeat_count * len(intervals) - (repeat_count - 1)
        total_s = repeat_count * sum(widths) + boundary_gap * (repeat_count - 1)
    else:
        for index, width in enumerate(widths):
            multiplicity = repeat_count
            if index == 0:
                multiplicity -= 1
            if index == len(widths) - 1:
                multiplicity -= 1
            _stats_add(internal, width, multiplicity)
        for left, right in zip(intervals[:-1], intervals[1:], strict=True):
            _stats_add(gaps, max(0.0, right[0] - left[1]), repeat_count)
        _stats_add(gaps, boundary_gap, repeat_count - 1)
        count = repeat_count * len(intervals)
        total_s = repeat_count * sum(widths)

    shifted_last = (
        last[0] + (repeat_count - 1) * period,
        last[1] + (repeat_count - 1) * period,
    )
    preview = _repeat_pattern_preview(
        intervals,
        cycle_start,
        period,
        repeat_count,
        boundary_merge,
        preview_limit,
    )
    return {
        "count": int(count),
        "total_s": float(total_s),
        "first": first,
        "last": shifted_last,
        "internal_widths": internal,
        "gaps": gaps,
        "preview": preview,
    }


def _scalar_threshold_intervals(times, values, threshold):
    """Intervals where a scalar, piecewise-linear record is strictly above threshold."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    threshold = float(threshold)
    if times.ndim != 1 or values.shape != times.shape or len(times) < 2:
        raise ValueError("Threshold analysis requires matching scalar values and time samples.")
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(times)))) * 8.0
    pieces = []
    for index in range(len(times) - 1):
        t0, t1 = float(times[index]), float(times[index + 1])
        v0, v1 = float(values[index]), float(values[index + 1])
        if max(v0, v1) <= threshold:
            continue
        start, end = t0, t1
        if v0 <= threshold < v1:
            fraction = (threshold - v0) / (v1 - v0)
            start = t0 + fraction * (t1 - t0)
        elif v1 <= threshold < v0:
            fraction = (threshold - v0) / (v1 - v0)
            end = t0 + fraction * (t1 - t0)
        if end > start + tolerance:
            pieces.append((start, end))
    if not pieces:
        return []
    merged = [list(pieces[0])]
    for start, end in pieces[1:]:
        if start <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def _logical_packet_end(times, repeat_count):
    if repeat_count > 1:
        return float(times[0]) + repeat_count * float(times[-1] - times[0])
    return float(times[-1])


def analyse_region_exposure_summary(payload, key, progress_callback=None):
    """Stream cumulative regional spatial-RMS statistics over physical time."""
    drive_count = len(payload.get("drive_ids", []))
    constant, linear, quadratic = _region_moments(payload, key, drive_count)
    work_total = _time_domain_work_sample_count(payload)
    work_done = 0
    _emit_analysis_progress(
        progress_callback,
        "Exposure summary — evaluating regional field…",
        0,
        work_total,
    )
    b_time = 0.0
    b2_time = 0.0
    first_time = None
    last_time = None
    previous_value = None
    peak_value = -np.inf
    peak_time = None
    logical_samples = 0
    max_interval = 0.0

    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        values = _region_spatial_rms_values(constant, linear, quadratic, currents)
        if first_time is None:
            first_time = float(times[0])
        elif last_time is not None:
            tolerance = _time_tolerance(last_time, times[0])
            if float(times[0]) < last_time - tolerance:
                raise ValueError("Authoritative regional analysis moved backwards in time.")
            join_dt = float(times[0]) - last_time
            if join_dt > 0.0:
                b_time += (previous_value + float(values[0])) * 0.5 * join_dt
                b2_time += (previous_value**2 + float(values[0])**2) * 0.5 * join_dt
                max_interval = max(max_interval, join_dt)

        local_index = int(np.argmax(values))
        candidate = float(values[local_index])
        if candidate > peak_value:
            peak_value = candidate
            peak_time = float(times[local_index])

        if len(times) >= 2:
            dt = np.diff(times)
            cycle_b = float(np.sum((values[:-1] + values[1:]) * 0.5 * dt))
            cycle_b2 = float(np.sum((values[:-1] ** 2 + values[1:] ** 2) * 0.5 * dt))
            b_time += cycle_b * repeat_count
            b2_time += cycle_b2 * repeat_count
            max_interval = max(max_interval, float(np.max(dt)))

        previous_value = float(values[-1])
        last_time = _logical_packet_end(times, repeat_count)
        logical_samples += len(times) * repeat_count
        work_done += len(times)
        _emit_analysis_progress(
            progress_callback,
            "Exposure summary — evaluating regional field…",
            min(work_done, work_total),
            work_total,
        )

    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError("Exposure summary requires a positive physical duration.")
    duration = float(last_time - first_time)
    _emit_analysis_progress(progress_callback, "Exposure summary ready", work_total, work_total)
    return {
        "duration_s": duration,
        "mean_spatial_rms_uT": b_time / duration,
        "playback_rms_uT": float(np.sqrt(max(0.0, b2_time / duration))),
        "peak_spatial_rms_uT": float(peak_value),
        "peak_time_s": float(peak_time),
        "b_time_uT_s": float(b_time),
        "b2_time_uT2_s": float(b2_time),
        "source_sample_count": int(logical_samples),
        "max_source_interval_s": float(max_interval),
        "analysis_source": "authoritative_packets" if isinstance(payload.get("authoritative_analysis"), dict) else "full_retained_trace",
    }


def analyse_regions_exposure_summary(payload, progress_callback=None):
    """Stream cumulative regional spatial-RMS statistics over physical time."""
    drive_count = len(payload.get("drive_ids", []))
    keys = tuple(payload.get("regions", {}))
    if not keys:
        raise ValueError("No prepared regions are available.")
    moments = [_region_moments(payload, key, drive_count) for key in keys]
    work_total = _time_domain_work_sample_count(payload)
    work_done = 0
    _emit_analysis_progress(
        progress_callback,
        "Exposure summary — evaluating regional field…",
        0,
        work_total,
    )
    b_time = np.zeros(len(keys))
    b2_time = np.zeros(len(keys))
    first_time = None
    last_time = None
    previous_value = None
    peak_value = np.full(len(keys), -np.inf)
    peak_time = np.zeros(len(keys))
    logical_samples = 0
    max_interval = 0.0

    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        values = np.column_stack([_region_spatial_rms_values(*m, currents) for m in moments])
        if first_time is None:
            first_time = float(times[0])
        elif last_time is not None:
            tolerance = _time_tolerance(last_time, times[0])
            if float(times[0]) < last_time - tolerance:
                raise ValueError("Authoritative regional analysis moved backwards in time.")
            join_dt = float(times[0]) - last_time
            if join_dt > 0.0:
                b_time += (previous_value + values[0]) * 0.5 * join_dt
                b2_time += (previous_value**2 + values[0]**2) * 0.5 * join_dt
                max_interval = max(max_interval, join_dt)

        local_indices = np.argmax(values, axis=0)
        candidates = values[local_indices, np.arange(len(keys))]
        improved = candidates > peak_value
        peak_value[improved] = candidates[improved]
        peak_time[improved] = times[local_indices[improved]]

        if len(times) >= 2:
            dt = np.diff(times)
            cycle_b = np.sum((values[:-1] + values[1:]) * 0.5 * dt[:, None], axis=0)
            cycle_b2 = np.sum((values[:-1] ** 2 + values[1:] ** 2) * 0.5 * dt[:, None], axis=0)
            b_time += cycle_b * repeat_count
            b2_time += cycle_b2 * repeat_count
            max_interval = max(max_interval, float(np.max(dt)))

        previous_value = values[-1].copy()
        last_time = _logical_packet_end(times, repeat_count)
        logical_samples += len(times) * repeat_count
        work_done += len(times)
        _emit_analysis_progress(
            progress_callback,
            "Exposure summary — evaluating regional field…",
            min(work_done, work_total),
            work_total,
        )

    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError("Exposure summary requires a positive physical duration.")
    duration = float(last_time - first_time)
    _emit_analysis_progress(progress_callback, "Exposure summary ready", work_total, work_total)
    return {key: {
        "duration_s": duration,
        "mean_spatial_rms_uT": b_time[index] / duration,
        "playback_rms_uT": float(np.sqrt(max(0.0, b2_time[index] / duration))),
        "peak_spatial_rms_uT": float(peak_value[index]),
        "peak_time_s": float(peak_time[index]),
        "b_time_uT_s": float(b_time[index]),
        "b2_time_uT2_s": float(b2_time[index]),
        "source_sample_count": int(logical_samples),
        "max_source_interval_s": float(max_interval),
        "analysis_source": "authoritative_packets" if isinstance(payload.get("authoritative_analysis"), dict) else "full_retained_trace",
    } for index, key in enumerate(keys)}


def _threshold_group_for_region(payload, key, threshold, progress_callback=None):
    drive_count = len(payload.get("drive_ids", []))
    constant, linear, quadratic = _region_moments(payload, key, drive_count)
    work_samples = _time_domain_work_sample_count(payload)
    work_total = max(1, work_samples * 2)
    work_done = 0
    _emit_analysis_progress(
        progress_callback,
        "Threshold analysis — evaluating field and crossings…",
        0,
        work_total,
    )
    combined = _interval_group_from_intervals([])
    first_time = None
    last_time = None
    previous_value = None
    logical_samples = 0
    max_interval = 0.0
    peak_value = -np.inf
    peak_time = None

    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        values = _region_spatial_rms_values(constant, linear, quadratic, currents)
        work_done += len(times)
        _emit_analysis_progress(
            progress_callback,
            "Threshold analysis — evaluating regional field…",
            min(work_done, work_total),
            work_total,
        )
        if first_time is None:
            first_time = float(times[0])
        elif last_time is not None:
            tolerance = _time_tolerance(last_time, times[0])
            if float(times[0]) < last_time - tolerance:
                raise ValueError("Authoritative threshold analysis moved backwards in time.")
            join_dt = float(times[0]) - last_time
            if join_dt > 0.0:
                join_intervals = _scalar_threshold_intervals(
                    np.asarray([last_time, float(times[0])]),
                    np.asarray([previous_value, float(values[0])]),
                    threshold,
                )
                combined = _combine_interval_groups(
                    combined, _interval_group_from_intervals(join_intervals)
                )
                max_interval = max(max_interval, join_dt)

        local_index = int(np.argmax(values))
        candidate = float(values[local_index])
        if candidate > peak_value:
            peak_value = candidate
            peak_time = float(times[local_index])

        if len(times) >= 2:
            intervals = _scalar_threshold_intervals(times, values, threshold)
            if repeat_count > 1:
                group = _repeat_interval_group(
                    intervals, times[0], times[-1], repeat_count
                )
            else:
                group = _interval_group_from_intervals(intervals)
            combined = _combine_interval_groups(combined, group)
            max_interval = max(max_interval, float(np.max(np.diff(times))))

        work_done += len(times)
        _emit_analysis_progress(
            progress_callback,
            "Threshold analysis — finding excursions…",
            min(work_done, work_total),
            work_total,
        )
        previous_value = float(values[-1])
        last_time = _logical_packet_end(times, repeat_count)
        logical_samples += len(times) * repeat_count

    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError("Threshold analysis requires a positive physical duration.")
    _emit_analysis_progress(progress_callback, "Threshold analysis ready", work_total, work_total)
    return combined, first_time, last_time, peak_value, peak_time, logical_samples, max_interval


def analyse_region_threshold_duration(payload, key, threshold_uT, progress_callback=None):
    """Time-above-threshold descriptors from the authoritative packet stream."""
    threshold = float(threshold_uT)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Field threshold must be a finite value of 0 µT or greater.")
    (
        group,
        first_time,
        last_time,
        peak_value,
        peak_time,
        logical_samples,
        max_interval,
    ) = _threshold_group_for_region(
        payload, key, threshold, progress_callback=progress_callback
    )
    return _threshold_result(payload, threshold, group, first_time, last_time,
                             peak_value, peak_time, logical_samples, max_interval)


def _threshold_result(payload, threshold, group, first_time, last_time,
                      peak_value, peak_time, logical_samples, max_interval):
    duration = float(last_time - first_time)
    width_stats = _interval_width_stats(group)
    above_s = float(group["total_s"])
    return {
        "threshold_uT": threshold,
        "duration_s": duration,
        "time_above_s": above_s,
        "fraction_above": above_s / duration if duration > 0.0 else 0.0,
        "percent_above": 100.0 * above_s / duration if duration > 0.0 else 0.0,
        "longest_interval_s": float(width_stats["max"]) if width_stats["count"] else 0.0,
        "excursion_count": int(group["count"]),
        "intervals_s": [tuple(value) for value in group["preview"]],
        "intervals_truncated": int(group["count"]) > len(group["preview"]),
        "interval_preview_limit": MAX_RETAINED_INTERVALS,
        "peak_spatial_rms_uT": float(peak_value),
        "peak_time_s": float(peak_time),
        "source_sample_count": int(logical_samples),
        "max_source_interval_s": float(max_interval),
        "analysis_source": "authoritative_packets" if isinstance(payload.get("authoritative_analysis"), dict) else "full_retained_trace",
    }


def analyse_regions_threshold_duration(payload, threshold_uT, progress_callback=None):
    """Stream each packet once; keep only bounded interval summaries per region."""
    threshold = float(threshold_uT)
    if not np.isfinite(threshold) or threshold < 0.:
        raise ValueError("Field threshold must be a finite value of 0 µT or greater.")
    keys = tuple(payload.get("regions", {}))
    if not keys:
        raise ValueError("No prepared regions are available.")
    moments = {k: _region_moments(payload, k, len(payload.get("drive_ids", []))) for k in keys}
    states = {k: {"group": _interval_group_from_intervals([]), "previous": None,
                  "peak": -np.inf, "when": None} for k in keys}
    first_time = last_time = None
    logical_samples = 0
    max_interval = 0.
    work_total = max(1, _time_domain_work_sample_count(payload) * len(keys))
    work_done = 0
    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        if first_time is None:
            first_time = float(times[0])
        if last_time is not None and times[0] < last_time - _time_tolerance(last_time, times[0]):
            raise ValueError("Authoritative threshold analysis moved backwards in time.")
        join_dt = float(times[0]) - last_time if last_time is not None else 0.
        if join_dt > 0.:
            max_interval = max(max_interval, join_dt)
        if len(times) >= 2:
            max_interval = max(max_interval, float(np.max(np.diff(times))))
        for key, state in states.items():
            values = _region_spatial_rms_values(*moments[key], currents)
            if join_dt > 0.:
                intervals = _scalar_threshold_intervals(np.asarray([last_time, times[0]]),
                    np.asarray([state["previous"], values[0]]), threshold)
                state["group"] = _combine_interval_groups(state["group"],
                                                          _interval_group_from_intervals(intervals))
            index = int(np.argmax(values))
            if values[index] > state["peak"]:
                state["peak"], state["when"] = float(values[index]), float(times[index])
            if len(times) >= 2:
                intervals = _scalar_threshold_intervals(times, values, threshold)
                group = (_repeat_interval_group(intervals, times[0], times[-1], repeat_count)
                         if repeat_count > 1 else _interval_group_from_intervals(intervals))
                state["group"] = _combine_interval_groups(state["group"], group)
            state["previous"] = float(values[-1])
            work_done += len(times)
            _emit_analysis_progress(progress_callback, "Finding excursions across all regions…",
                                    min(work_done, work_total), work_total)
        last_time = _logical_packet_end(times, repeat_count)
        logical_samples += len(times) * repeat_count
    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError("Threshold analysis requires a positive physical duration.")
    return {key: _threshold_result(payload, threshold, state["group"], first_time, last_time,
                                   state["peak"], state["when"], logical_samples, max_interval)
            for key, state in states.items()}


FFT_WINDOWS = ("rectangular", "hann", "hamming", "blackman")

# Frequency-dependent external magnetic-flux-density screening curves.  The
# values below are RMS B reference/action levels in microtesla and the power
# is the exponent in ``coefficient * f**power``.  Field Workbench currently
# evaluates the low-frequency 1 Hz--100 kHz portion only.
FREQUENCY_WEIGHTING_PROFILES = {
    "icnirp2010_public": {
        "label": "ICNIRP 2010 — General public B reference levels",
        "source": "ICNIRP LF Guidelines (2010), Table 4",
        "segments": (
            (1.0, 8.0, 4.0e4, -2),
            (8.0, 25.0, 5.0e3, -1),
            (25.0, 400.0, 2.0e2, 0),
            (400.0, 3.0e3, 8.0e4, -1),
            (3.0e3, 1.0e5, 27.0, 0),
        ),
    },
    "icnirp2010_occupational": {
        "label": "ICNIRP 2010 — Occupational B reference levels",
        "source": "ICNIRP LF Guidelines (2010), Table 3",
        "segments": (
            (1.0, 8.0, 2.0e5, -2),
            (8.0, 25.0, 2.5e4, -1),
            (25.0, 300.0, 1.0e3, 0),
            (300.0, 3.0e3, 3.0e5, -1),
            (3.0e3, 1.0e5, 1.0e2, 0),
        ),
    },
    "eu2013_low": {
        "label": "EU 2013/35/EU — Low action level (B)",
        "source": "Directive 2013/35/EU, Annex II Table B2",
        "segments": (
            (1.0, 8.0, 2.0e5, -2),
            (8.0, 25.0, 2.5e4, -1),
            (25.0, 300.0, 1.0e3, 0),
            (300.0, 3.0e3, 3.0e5, -1),
            (3.0e3, 1.0e5, 1.0e2, 0),
        ),
    },
    "eu2013_high": {
        "label": "EU 2013/35/EU — High action level (B)",
        "source": "Directive 2013/35/EU, Annex II Table B2",
        "segments": (
            (1.0, 3.0e3, 3.0e5, -1),
            (3.0e3, 1.0e5, 1.0e2, 0),
        ),
    },
    "eu2013_limb": {
        "label": "EU 2013/35/EU — Limb action level (B)",
        "source": "Directive 2013/35/EU, Annex II Table B2",
        "segments": (
            (1.0, 3.0e3, 9.0e5, -1),
            (3.0e3, 1.0e5, 3.0e2, 0),
        ),
    },
}
FREQUENCY_WEIGHTING_REFERENCE_HZ = 50.0
FREQUENCY_WEIGHTING_MIN_HZ = 1.0
FREQUENCY_WEIGHTING_MAX_HZ = 100_000.0


def _waveform_timeline(payload):
    if payload.get("analysis_trace_reduced", False):
        raise ValueError(
            "This long exposure retains a reduced trace for display only. "
            "This retained-trace helper cannot materialize the full physical record; "
            "use an authoritative long-exposure analysis path instead."
        )
    times = np.frombuffer(base64.b64decode(payload["times_b64"]), dtype=np.float64)
    currents = np.frombuffer(base64.b64decode(payload["currents_b64"]), dtype=np.float64).reshape(payload["shape"])
    if len(times) < 3 or currents.shape[0] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Waveform analysis needs at least three increasing physical time samples.")
    return times, currents


def _fft_window(name, sample_count):
    key = str(name).strip().lower()
    if key not in FFT_WINDOWS:
        raise ValueError(f"Unknown FFT window: {name}")
    if key == "rectangular":
        return np.ones(sample_count, dtype=np.float64)
    # Use the periodic form of the NumPy windows for a DFT record.  Generating
    # N+1 points and dropping the duplicate endpoint avoids treating both ends
    # of the FFT interval as independent samples.
    generator = {"hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}[key]
    return np.asarray(generator(sample_count + 1)[:-1], dtype=np.float64)


MAX_SEGMENTED_FFT_SAMPLES = 16_777_216
MAX_SEGMENTED_FFT_SEGMENTS = 4096
MAX_FFT_SAMPLES = 65_536


def _fft_sample_count(value, maximum_samples):
    maximum_samples = max(2, int(maximum_samples))
    if value is None:
        return maximum_samples
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("FFT sample count must be an integer.") from error
    if count < 2 or count > maximum_samples:
        raise ValueError(
            f"FFT sample count must be between 2 and {maximum_samples:,}."
        )
    return count


def _fft_region_power(region, uniform, *, window, detrend):
    """Return one energy-normalized one-sided vector-field periodogram."""
    values = np.asarray(uniform, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("FFT analysis needs at least two uniformly sampled values.")
    n = len(values)
    q = np.asarray(region["quadratic"], dtype=np.float64)
    linear = np.asarray(region["linear"], dtype=np.float64)
    weights = _fft_window(window, n)
    normalization = float(np.sqrt(n * np.dot(weights, weights)))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("FFT window has zero energy for this sample count.")

    if bool(detrend):
        centred = values - values.mean(axis=0)
        coeff = np.fft.rfft(weights[:, None] * centred, axis=0) / normalization
        power = np.einsum("fc,...cd,fd->f...", coeff.conj(), q, coeff).real
    else:
        augmented = np.column_stack([np.ones(n, dtype=np.float64), values])
        moment = np.empty(q.shape[:-2] + (q.shape[-2] + 1, q.shape[-1] + 1), dtype=np.float64)
        moment[..., 0, 0] = region["constant"]
        moment[..., 0, 1:] = linear
        moment[..., 1:, 0] = linear
        moment[..., 1:, 1:] = q
        coeff = np.fft.rfft(weights[:, None] * augmented, axis=0) / normalization
        power = np.einsum("fc,...cd,fd->f...", coeff.conj(), moment, coeff).real

    power = np.maximum(0.0, np.asarray(power, dtype=np.float64))
    if len(power) > 1:
        power[1:] *= 2.0
        if n % 2 == 0:
            power[-1] /= 2.0
    return power


def _dc_spatial_rms_uT(region, mean_currents):
    mean = np.asarray(mean_currents, dtype=np.float64).reshape(-1)
    q = np.asarray(region["quadratic"], dtype=np.float64)
    linear = np.asarray(region["linear"], dtype=np.float64)
    value = (np.asarray(region["constant"]) + 2.0 * np.einsum("c,...c->...", mean, linear)
             + np.einsum("c,...cd,d->...", mean, q, mean))
    return np.sqrt(np.maximum(0.0, value)) * 1e6


def _authoritative_spectral_clock(payload):
    frozen = _authoritative_analysis_source(payload)
    try:
        spacing = float(frozen["analysis_spacing_s"])
        protocol = frozen["protocol"]
        duration = float(protocol["duration_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "The saved result does not contain a valid authoritative spectral clock."
        ) from error
    if not np.isfinite(spacing) or spacing <= 0.0 or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("The saved result contains an invalid authoritative spectral clock.")
    intervals = max(2, int(np.ceil(duration / spacing)))
    source_spacing = duration / intervals
    return duration, intervals, source_spacing


def _verified_periodic_spectral_packet(payload):
    """Return the repeated verified steady-state cycle, if one exists."""
    frozen = _authoritative_analysis_source(payload)
    if str(frozen.get("mode", "")) != "verified_packets":
        return None
    repeated = []
    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        if int(repeat_count) > 1:
            repeated.append((times, currents, int(repeat_count)))
    if not repeated:
        return None
    if len(repeated) != 1:
        raise ValueError(
            "Verified long-exposure analysis contains more than one repeated spectral cycle."
        )
    return repeated[0]


def _replay_recipe_uniform_segments(payload, *, duration, segment_samples, segment_count):
    """Resolve only the uniform FFT samples for a frozen non-periodic exposure.

    This deliberately bypasses the denser event-aware time-domain packet stream.
    The electrical resolver still walks every source/routing event internally,
    but only the bounded FFT grid is retained by this caller.
    """
    frozen = _authoritative_analysis_source(payload)
    if str(frozen.get("mode", "")) != "replay_recipe":
        raise ValueError("Authoritative FFT replay recipe is unavailable.")
    from .exposures import normalize_exposure_protocol
    from .studio_adapter import StudioAdapter

    try:
        state = normalize_exposure_protocol(frozen["protocol"])
    except Exception as error:  # noqa: BLE001 - frozen-result validation boundary
        raise ValueError("Frozen exposure FFT recipe is malformed.") from error
    adapter = _FrozenExposureAnalysisAdapter(frozen)
    drive_ids = [str(value) for value in frozen.get("drive_ids", [])]
    n = int(segment_samples)
    segments = int(segment_count)
    spacing = float(duration) / float(n * segments)

    def sample_chunks():
        for segment_index in range(segments):
            first_sample = segment_index * n
            yield (first_sample + np.arange(n, dtype=np.float64)) * spacing

    produced = 0
    for resolved in StudioAdapter.iter_exposure_currents_at_times(adapter, state, sample_chunks()):
        times = np.asarray(resolved.get("time_s", []), dtype=np.float64).reshape(-1)
        if len(times) != n:
            raise ValueError("Frozen exposure FFT replay returned an invalid sample count.")
        current_map = resolved.get("coil_currents_a", {})
        if not isinstance(current_map, dict):
            raise ValueError("Frozen exposure FFT replay returned invalid current data.")
        columns = []
        for coil_id in drive_ids:
            values = np.asarray(current_map.get(coil_id, []), dtype=np.float64).reshape(-1)
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError(
                    f"Frozen exposure FFT replay did not reproduce coil {coil_id!r}."
                )
            columns.append(values)
        currents = (
            np.column_stack(columns)
            if columns
            else np.zeros((n, 0), dtype=np.float64)
        )
        produced += 1
        yield times, currents
    if produced != segments:
        raise ValueError("Frozen exposure FFT replay did not produce every spectral record.")



def _packet_uniform_segments(payload, *, duration, segment_samples, segment_count):
    """Resample an authoritative packet stream into bounded equal FFT records.

    The output grid spans the complete physical duration with exactly
    ``segment_count * segment_samples`` samples.  That count is never larger
    than the authoritative source interval count selected by the caller, so
    interpolation cannot claim source bandwidth that was never present.
    """
    n = int(segment_samples)
    segments = int(segment_count)
    total_samples = n * segments
    if n < 2 or segments < 1 or total_samples < 2:
        raise ValueError("Segmented FFT configuration is invalid.")
    spacing = float(duration) / float(total_samples)
    packet_iter = iter(_iter_time_domain_packets(payload))
    drive_count = len(payload.get("drive_ids", []))
    buffer_times = np.empty(0, dtype=np.float64)
    buffer_currents = np.empty((0, drive_count), dtype=np.float64)
    exhausted = False

    def append_until(target):
        nonlocal buffer_times, buffer_currents, exhausted
        tolerance = _time_tolerance(target)
        while (not len(buffer_times) or float(buffer_times[-1]) < target - tolerance) and not exhausted:
            try:
                times, currents, repeat_count = next(packet_iter)
            except StopIteration:
                exhausted = True
                break
            if int(repeat_count) != 1:
                raise ValueError(
                    "Segmented FFT cannot expand a repeated packet; use the verified-cycle spectral path."
                )
            times = np.asarray(times, dtype=np.float64).reshape(-1)
            currents = np.asarray(currents, dtype=np.float64)
            if len(buffer_times):
                keep = times > float(buffer_times[-1]) + _time_tolerance(buffer_times[-1])
                if not np.any(keep):
                    continue
                times = times[keep]
                currents = currents[keep]
            if len(times):
                buffer_times = np.concatenate((buffer_times, times))
                buffer_currents = np.vstack((buffer_currents, currents))

    for segment_index in range(segments):
        first_sample = segment_index * n
        query = (first_sample + np.arange(n, dtype=np.float64)) * spacing
        append_until(float(query[-1]))
        if not len(buffer_times) or float(buffer_times[0]) > float(query[0]) + _time_tolerance(query[0]):
            raise ValueError("Authoritative FFT stream does not cover the requested physical time.")
        if float(buffer_times[-1]) < float(query[-1]) - _time_tolerance(query[-1]):
            raise ValueError("Authoritative FFT stream ended before the exposure duration.")
        uniform = np.column_stack(
            [np.interp(query, buffer_times, buffer_currents[:, column]) for column in range(drive_count)]
        ) if drive_count else np.zeros((n, 0), dtype=np.float64)
        yield query, uniform

        next_time = float((segment_index + 1) * n * spacing)
        if len(buffer_times):
            keep_from = max(0, int(np.searchsorted(buffer_times, next_time, side="right")) - 1)
            if keep_from:
                buffer_times = buffer_times[keep_from:]
                buffer_currents = buffer_currents[keep_from:]


def _authoritative_uniform_segments(payload, *, duration, segment_samples, segment_count):
    frozen = _authoritative_analysis_source(payload)
    mode = str(frozen.get("mode", ""))
    if mode == "replay_recipe":
        return _replay_recipe_uniform_segments(
            payload,
            duration=duration,
            segment_samples=segment_samples,
            segment_count=segment_count,
        )
    return _packet_uniform_segments(
        payload,
        duration=duration,
        segment_samples=segment_samples,
        segment_count=segment_count,
    )



def analyse_region_spectrum(payload, key, *, fft_samples=None, window="rectangular", detrend=True, progress_callback=None, _region_keys=None):
    """Return a bounded, source-honest regional vector spectrum.

    Short/unreduced records keep the historical whole-record FFT exactly.
    Reduced exposures use their Phase-1 authoritative source:

    * a verified periodic exposure analyses one verified steady-state physical
      cycle at the source-supported clock, without expanding its repeats;
    * another long exposure is divided into equal non-overlapping records that
      span the complete physical duration, and their vector spectral powers are
      averaged.  The resampling clock is never faster than the authoritative
      source clock and the total amount of spectral work is bounded.

    ``bin_rms_uT`` remains an RMS contribution per one-sided frequency bin.
    For segmented records it is the square root of the mean bin power across
    all records, analogous to a bounded Welch-style periodogram without overlap.
    """
    _emit_analysis_progress(progress_callback, "Preparing spectrum…")
    if str(window).strip().lower() not in FFT_WINDOWS:
        raise ValueError(f"Unknown FFT window: {window}")
    window_key = str(window).strip().lower()
    drive_count = len(payload.get("drive_ids", []))
    try:
        region = payload["regions"][key]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown regional FFT target.") from error
    _constant, _linear, _quadratic = _region_moments(payload, key, drive_count)
    if _region_keys is not None:
        moments = [_region_moments(payload, k, drive_count) for k in _region_keys]
        region = dict(zip(("constant", "linear", "quadratic"),
                          (np.stack(values) for values in zip(*moments))))

    # Preserve the established exact short-record behaviour.  Exposure maps do
    # not need replay merely because they happen to carry an authoritative
    # source if their complete physical trace was already retained.
    if not bool(payload.get("analysis_trace_reduced", False)):
        times, currents = _waveform_timeline(payload)
        maximum_samples = min(len(times) - 1, MAX_FFT_SAMPLES)
        n = _fft_sample_count(fft_samples, maximum_samples)
        duration = float(times[-1] - times[0])
        spacing = duration / n
        grid = times[0] + np.arange(n, dtype=np.float64) * spacing
        uniform = np.column_stack(
            [np.interp(grid, times, column) for column in currents.T]
        ) if drive_count else np.zeros((n, 0), dtype=np.float64)
        power = _fft_region_power(region, uniform, window=window_key, detrend=detrend)
        mean = uniform.mean(axis=0)
        frequency = np.fft.rfftfreq(n, spacing)
        _emit_analysis_progress(progress_callback, "Spectrum ready", 1, 1)
        return {
            "frequency_hz": frequency,
            "bin_rms_uT": np.sqrt(power) * 1e6,
            "dc_rms_uT": _dc_spatial_rms_uT(region, mean),
            "fft_samples": n,
            "fft_bins": len(frequency),
            "max_fft_samples": maximum_samples,
            "frequency_resolution_hz": 1.0 / duration,
            "nyquist_hz": 1.0 / (2.0 * spacing),
            "sample_spacing_s": spacing,
            "source_sample_spacing_s": float(np.max(np.diff(times))),
            "source_nyquist_hz": 1.0 / (2.0 * float(np.max(np.diff(times)))),
            "window": window_key,
            "detrended": bool(detrend),
            "analysis_mode": "whole_record",
            "analysis_mode_label": "Whole physical record",
            "physical_duration_s": duration,
            "spectral_record_duration_s": duration,
            "segment_count": 1,
            "spectral_sampling_reduced": bool(n < len(times) - 1),
        }

    frozen = _authoritative_analysis_source(payload)

    # A verified repeated physical cycle is the preferred long-exposure FFT:
    # it preserves the true steady-state waveform bandwidth and harmonic
    # structure without allowing exposure duration to dilute the sample clock.
    periodic = _verified_periodic_spectral_packet(payload)
    if periodic is not None:
        times, currents, repeat_count = periodic
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            raise ValueError("Verified spectral cycle has zero duration.")
        raw_spacing = frozen.get("analysis_spacing_s")
        if raw_spacing is not None:
            try:
                nominal_spacing = float(raw_spacing)
            except (TypeError, ValueError):
                nominal_spacing = np.nan
        else:
            nominal_spacing = np.nan
        if np.isfinite(nominal_spacing) and nominal_spacing > 0.0:
            source_intervals = max(2, int(np.ceil(duration / nominal_spacing)))
            source_spacing = duration / source_intervals
        else:
            source_intervals = max(2, len(times) - 1)
            source_spacing = duration / source_intervals
        maximum_samples = min(source_intervals, MAX_FFT_SAMPLES)
        n = _fft_sample_count(fft_samples, maximum_samples)
        spacing = duration / n
        grid = float(times[0]) + np.arange(n, dtype=np.float64) * spacing
        uniform = np.column_stack(
            [np.interp(grid, times, column) for column in currents.T]
        ) if drive_count else np.zeros((n, 0), dtype=np.float64)
        power = _fft_region_power(region, uniform, window=window_key, detrend=detrend)
        mean = uniform.mean(axis=0)
        frequency = np.fft.rfftfreq(n, spacing)
        compression = frozen.get("compression") if isinstance(frozen.get("compression"), dict) else {}
        physical_duration = None
        try:
            physical_duration = float(frozen.get("protocol", {}).get("duration_s"))
        except (TypeError, ValueError):
            physical_duration = None
        if physical_duration is None or not np.isfinite(physical_duration):
            physical_duration = duration * int(repeat_count)
        _emit_analysis_progress(progress_callback, "Spectrum ready", 1, 1)
        return {
            "frequency_hz": frequency,
            "bin_rms_uT": np.sqrt(power) * 1e6,
            "dc_rms_uT": _dc_spatial_rms_uT(region, mean),
            "fft_samples": n,
            "fft_bins": len(frequency),
            "max_fft_samples": maximum_samples,
            "frequency_resolution_hz": 1.0 / duration,
            "nyquist_hz": 1.0 / (2.0 * spacing),
            "sample_spacing_s": spacing,
            "source_sample_spacing_s": source_spacing,
            "source_nyquist_hz": 1.0 / (2.0 * source_spacing),
            "window": window_key,
            "detrended": bool(detrend),
            "analysis_mode": "verified_periodic_cycle",
            "analysis_mode_label": "Verified steady-state cycle",
            "physical_duration_s": float(physical_duration),
            "spectral_record_duration_s": duration,
            "segment_count": 1,
            "repeat_count": int(repeat_count),
            "stable_start_s": compression.get("stable_start_s"),
            "spectral_sampling_reduced": bool(n < source_intervals),
        }

    # Non-periodic long records cannot have an exact bounded whole-record DFT
    # while preserving arbitrary source bandwidth.  Tile the entire physical
    # duration with equal records, average their vector periodograms, and cap
    # the total source evaluations.  Fewer samples means a lower *reported*
    # Nyquist; it never manufactures bandwidth by interpolation.
    duration, source_intervals, source_spacing = _authoritative_spectral_clock(payload)
    maximum_samples = min(MAX_FFT_SAMPLES, source_intervals)
    if fft_samples is None:
        # Auto mode maximizes represented source bandwidth within both the
        # per-record and total-work bounds.  Choosing the record count first
        # avoids throwing away nearly half the source clock when the interval
        # count is only slightly larger than 65,536.
        spectral_budget = min(source_intervals, MAX_SEGMENTED_FFT_SAMPLES)
        segment_count = max(1, int(np.ceil(spectral_budget / MAX_FFT_SAMPLES)))
        segment_count = min(segment_count, MAX_SEGMENTED_FFT_SEGMENTS)
        n = max(2, min(maximum_samples, spectral_budget // segment_count))
        segment_count = max(1, min(segment_count, source_intervals // n))
    else:
        n = _fft_sample_count(fft_samples, maximum_samples)
        possible_segments = max(1, source_intervals // n)
        work_segments = max(1, MAX_SEGMENTED_FFT_SAMPLES // n)
        segment_count = min(possible_segments, MAX_SEGMENTED_FFT_SEGMENTS, work_segments)
    total_samples = segment_count * n
    spacing = duration / total_samples
    if spacing + _time_tolerance(spacing, source_spacing) < source_spacing:
        raise ValueError("Segmented FFT attempted to exceed the authoritative source bandwidth.")

    accumulated_power = None
    current_sum = np.zeros(drive_count, dtype=np.float64)
    actual_segments = 0
    _emit_analysis_progress(
        progress_callback,
        f"Calculating spectrum — 0/{segment_count} records",
        0,
        segment_count,
    )
    for segment_index, (_grid, uniform) in enumerate(_authoritative_uniform_segments(
        payload,
        duration=duration,
        segment_samples=n,
        segment_count=segment_count,
    ), start=1):
        segment_power = _fft_region_power(
            region, uniform, window=window_key, detrend=detrend
        )
        if accumulated_power is None:
            accumulated_power = np.zeros_like(segment_power)
        accumulated_power += segment_power
        current_sum += np.sum(uniform, axis=0)
        actual_segments += 1
        _emit_analysis_progress(
            progress_callback,
            f"Calculating spectrum — {segment_index}/{segment_count} records",
            segment_index,
            segment_count,
        )
    if accumulated_power is None or actual_segments != segment_count:
        raise ValueError("Authoritative segmented FFT did not produce the expected records.")
    power = accumulated_power / float(actual_segments)
    mean = current_sum / float(total_samples)
    record_duration = n * spacing
    frequency = np.fft.rfftfreq(n, spacing)
    return {
        "frequency_hz": frequency,
        "bin_rms_uT": np.sqrt(np.maximum(0.0, power)) * 1e6,
        "dc_rms_uT": _dc_spatial_rms_uT(region, mean),
        "fft_samples": n,
        "fft_bins": len(frequency),
        "max_fft_samples": maximum_samples,
        "frequency_resolution_hz": 1.0 / record_duration,
        "nyquist_hz": 1.0 / (2.0 * spacing),
        "sample_spacing_s": spacing,
        "source_sample_spacing_s": source_spacing,
        "source_nyquist_hz": 1.0 / (2.0 * source_spacing),
        "window": window_key,
        "detrended": bool(detrend),
        "analysis_mode": "segmented_full_duration",
        "analysis_mode_label": "Segmented full-duration spectrum",
        "physical_duration_s": duration,
        "spectral_record_duration_s": record_duration,
        "segment_count": segment_count,
        "source_interval_count": source_intervals,
        "spectral_sample_count": total_samples,
        "spectral_sampling_reduced": bool(total_samples < source_intervals),
    }


def frequency_weighting_reference_uT(frequency_hz, profile_key):
    """Return RMS external-B reference/action levels for one weighting profile.

    Frequencies outside the implemented 1 Hz--100 kHz LF screening band return
    infinity so they contribute zero to the normalized weighting.
    """
    profile = FREQUENCY_WEIGHTING_PROFILES.get(str(profile_key))
    if profile is None:
        raise ValueError(f"Unknown frequency-weighting profile: {profile_key}")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    result = np.full(frequency.shape, np.inf, dtype=np.float64)
    for low, high, coefficient, power in profile["segments"]:
        # The tabulated curves are continuous at their boundaries, so assigning
        # the shared endpoint to either adjoining segment gives the same value.
        mask = (frequency >= low) & (frequency <= high)
        if np.any(mask):
            result[mask] = coefficient * np.power(frequency[mask], power)
    return result


def _frequency_weighting_phase_rad(frequency_hz, profile_key):
    """Exact piecewise phase prescribed by the ICNIRP weighted-peak method.

    If the tabulated exposure limit varies as 1/f², 1/f, f⁰ or f, the
    corresponding weighting phase is +180°, +90°, 0° or -90°.
    """
    profile = FREQUENCY_WEIGHTING_PROFILES.get(str(profile_key))
    if profile is None:
        raise ValueError(f"Unknown frequency-weighting profile: {profile_key}")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    result = np.zeros(frequency.shape, dtype=np.float64)
    phase_for_power = {-2: np.pi, -1: np.pi / 2.0, 0: 0.0, 1: -np.pi / 2.0}
    for low, high, _coefficient, power in profile["segments"]:
        mask = (frequency >= low) & (frequency <= high)
        if np.any(mask):
            result[mask] = phase_for_power[int(power)]
    return result


def _frequency_weighting_transfer(sample_count, spacing_s, profile):
    """Return the discrete phase-aware weighting transfer for one FFT block."""
    n = int(sample_count)
    spacing = float(spacing_s)
    if n < 2 or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Frequency weighting requires a valid uniform sample clock.")
    frequency = np.fft.rfftfreq(n, spacing)
    references = frequency_weighting_reference_uT(frequency, profile)
    phases = _frequency_weighting_phase_rad(frequency, profile)
    reference_at_fref = float(
        frequency_weighting_reference_uT(
            np.asarray([FREQUENCY_WEIGHTING_REFERENCE_HZ]), profile
        )[0]
    )
    valid = (
        (frequency >= FREQUENCY_WEIGHTING_MIN_HZ)
        & (frequency <= FREQUENCY_WEIGHTING_MAX_HZ)
        & np.isfinite(references)
        & (references > 0.0)
    )
    gain = np.zeros_like(frequency)
    gain[valid] = reference_at_fref / references[valid]
    transfer = gain * np.exp(1j * phases)
    return frequency, references, transfer, reference_at_fref


def _frequency_weighted_current_block(uniform, spacing_s, profile, transfer=None):
    """Apply the published gain/phase curve to one uniform current block."""
    values = np.asarray(uniform, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Frequency weighting needs at least two uniform samples.")
    if transfer is None:
        _frequency, _references, transfer, _reference = _frequency_weighting_transfer(
            len(values), spacing_s, profile
        )
    spectrum = np.fft.rfft(values, axis=0)
    return np.fft.irfft(spectrum * transfer[:, None], n=len(values), axis=0)


def _weighted_region_spatial_rms_uT(quadratic, weighted_currents):
    currents = np.asarray(weighted_currents, dtype=np.float64)
    q = np.asarray(quadratic, dtype=np.float64)
    return (
        np.sqrt(
            np.maximum(
                0.0,
                np.einsum("tc,cd,td->t", currents, q, currents, optimize=True),
            )
        )
        * 1e6
    )


def _frequency_weighting_preview_append(
    preview_times, preview_values, grid, values, *, global_offset, total_samples
):
    """Retain an evenly distributed bounded trace without affecting peak search."""
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(grid) != len(values):
        raise ValueError("Frequency-weighted preview dimensions are inconsistent.")
    stride = max(1, int(np.ceil(float(total_samples) / float(MAX_FFT_SAMPLES))))
    indices = np.arange(len(grid), dtype=np.int64)
    keep = ((indices + int(global_offset)) % stride) == 0
    if np.any(keep):
        preview_times.extend(grid[keep].tolist())
        preview_values.extend(values[keep].tolist())
    return stride


def analyse_region_frequency_weighted(payload, key, *, profile="icnirp2010_public", fft_samples=None, spectrum=None, progress_callback=None, _block_observer=None):
    """Return phase-aware weighted-peak and spectral-sum regional screening.

    Short complete records preserve the historical whole-record calculation.
    Reduced verified-periodic exposures analyse the already-verified steady-state
    physical cycle without expanding repeats. Other reduced long exposures use
    bounded three-record-context FFT blocks over the complete physical duration;
    only the centre record is scored so circular block boundaries are kept one
    record away from the reported samples.

    The frozen payload stores spatial second moments of the vector field.
    Frequency weighting is linear, so it is applied to the drive currents first
    and the regional spatial RMS is then evaluated without repeating the field
    solve. The result is a regional screening descriptor, not pointwise
    compliance certification.
    """
    if str(profile) not in FREQUENCY_WEIGHTING_PROFILES:
        raise ValueError(f"Unknown frequency-weighting profile: {profile}")
    drive_count = len(payload.get("drive_ids", []))
    try:
        region = payload["regions"][key]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown regional frequency-weighting target.") from error
    _constant, _linear, q = _region_moments(payload, key, drive_count)

    # Reuse the Phase-3 source-honest spectral plan. This determines the exact
    # source-supported record length, sampling interval and long-exposure mode,
    # and also supplies the spectral-summation cross-check. A caller may pass a
    # session-cached spectrum so opening this page does not repeat the same FFT.
    if not isinstance(spectrum, dict):
        _emit_analysis_progress(progress_callback, "Preparing frequency-weighted spectrum…")
        spectrum = analyse_region_spectrum(
            payload, key, fft_samples=fft_samples, window="rectangular", detrend=True,
            progress_callback=progress_callback,
        )
    else:
        _emit_analysis_progress(progress_callback, "Using cached spectrum; applying weighting…")
    mode = str(spectrum.get("analysis_mode", "whole_record"))
    n = int(spectrum["fft_samples"])
    spacing = float(spectrum["sample_spacing_s"])
    peak_limit_uT = np.sqrt(2.0) * float(
        frequency_weighting_reference_uT(
            np.asarray([FREQUENCY_WEIGHTING_REFERENCE_HZ]), profile
        )[0]
    )
    reference_at_fref = peak_limit_uT / np.sqrt(2.0)

    preview_times: list[float] = []
    preview_values: list[float] = []
    weighted_peak_value = -np.inf
    weighted_peak_time = None
    weighting_frequency_resolution = None
    weighting_context_records = 1
    weighting_sample_count = n
    weighted_trace_reduced = False

    if mode in {"whole_record", "verified_periodic_cycle"}:
        _emit_analysis_progress(progress_callback, "Applying frequency weighting…", 0, 1)
        if mode == "whole_record":
            times, currents = _waveform_timeline(payload)
        else:
            periodic = _verified_periodic_spectral_packet(payload)
            if periodic is None:
                raise ValueError("Verified frequency-weighting cycle is unavailable.")
            times, currents, _repeat_count = periodic
        grid = float(times[0]) + np.arange(n, dtype=np.float64) * spacing
        uniform = (
            np.column_stack(
                [np.interp(grid, times, column) for column in currents.T]
            )
            if drive_count
            else np.zeros((n, 0), dtype=np.float64)
        )
        _wf, _wr, transfer, _reference = _frequency_weighting_transfer(
            n, spacing, profile
        )
        weighted_currents = _frequency_weighted_current_block(
            uniform, spacing, profile, transfer=transfer
        )
        if _block_observer is not None:
            _block_observer(grid, weighted_currents, 0, n)
        weighted_values = _weighted_region_spatial_rms_uT(q, weighted_currents)
        peak_index = int(np.argmax(weighted_values)) if len(weighted_values) else 0
        weighted_peak_value = float(weighted_values[peak_index])
        weighted_peak_time = float(grid[peak_index])
        preview_times = grid.tolist()
        preview_values = weighted_values.tolist()
        weighting_frequency_resolution = 1.0 / (n * spacing)
        _emit_analysis_progress(progress_callback, "Frequency weighting ready", 1, 1)
    elif mode == "segmented_full_duration":
        segment_count = int(spectrum["segment_count"])
        duration = float(spectrum["physical_duration_s"])
        total_samples = n * segment_count
        if segment_count < 1 or total_samples < 2:
            raise ValueError("Segmented frequency-weighting plan is invalid.")
        _emit_analysis_progress(
            progress_callback,
            f"Applying frequency weighting — 0/{segment_count} records",
            0,
            segment_count,
        )

        segment_iter = iter(
            _authoritative_uniform_segments(
                payload,
                duration=duration,
                segment_samples=n,
                segment_count=segment_count,
            )
        )
        try:
            first_item = next(segment_iter)
        except StopIteration as error:
            raise ValueError("Authoritative frequency weighting produced no records.") from error
        weighting_context_records = 3 if segment_count > 1 else 1
        context_n = n * weighting_context_records
        _wf, _wr, context_transfer, _reference = _frequency_weighting_transfer(
            context_n, spacing, profile
        )
        weighting_frequency_resolution = 1.0 / (context_n * spacing)
        weighting_sample_count = context_n

        scored_segments = 0

        def score_segment(segment_index, item, left_values=None, right_values=None):
            nonlocal weighted_peak_value, weighted_peak_time, weighted_trace_reduced, scored_segments
            current_grid, current_values = item
            current_values = np.asarray(current_values, dtype=np.float64)
            if segment_count == 1:
                block = current_values
                centre_slice = slice(0, n)
            else:
                block = np.vstack((left_values, current_values, right_values))
                centre_slice = slice(n, 2 * n)
            weighted_block = _frequency_weighted_current_block(
                block, spacing, profile, transfer=context_transfer
            )
            if _block_observer is not None:
                _block_observer(current_grid, weighted_block[centre_slice], segment_index * n, total_samples)
            weighted_values = _weighted_region_spatial_rms_uT(
                q, weighted_block[centre_slice]
            )
            local_peak = int(np.argmax(weighted_values)) if len(weighted_values) else 0
            candidate = float(weighted_values[local_peak])
            if candidate > weighted_peak_value:
                weighted_peak_value = candidate
                weighted_peak_time = float(current_grid[local_peak])
            stride = _frequency_weighting_preview_append(
                preview_times,
                preview_values,
                current_grid,
                weighted_values,
                global_offset=segment_index * n,
                total_samples=total_samples,
            )
            weighted_trace_reduced = bool(weighted_trace_reduced or stride > 1)
            scored_segments += 1
            _emit_analysis_progress(
                progress_callback,
                f"Applying frequency weighting — {scored_segments}/{segment_count} records",
                scored_segments,
                segment_count,
            )

        if segment_count == 1:
            score_segment(0, first_item)
        else:
            # Match the circular boundary convention of the historical
            # whole-record DFT without retaining the whole long exposure. The
            # first record is scored after the last record is known; all middle
            # records stream with one real record of context on each side.
            try:
                second_item = next(segment_iter)
            except StopIteration as error:
                raise ValueError("Segmented frequency weighting is missing its second record.") from error
            previous_item = first_item
            current_item = second_item
            for segment_index in range(1, segment_count - 1):
                try:
                    next_item = next(segment_iter)
                except StopIteration as error:
                    raise ValueError(
                        "Authoritative frequency weighting ended before the physical duration."
                    ) from error
                score_segment(
                    segment_index,
                    current_item,
                    np.asarray(previous_item[1], dtype=np.float64),
                    np.asarray(next_item[1], dtype=np.float64),
                )
                previous_item, current_item = current_item, next_item

            # current_item is now the final record. Use the first record as its
            # right context, then score the deferred first record with the final
            # record as left context and the second as right context.
            last_index = segment_count - 1
            score_segment(
                last_index,
                current_item,
                np.asarray(previous_item[1], dtype=np.float64),
                np.asarray(first_item[1], dtype=np.float64),
            )
            score_segment(
                0,
                first_item,
                np.asarray(current_item[1], dtype=np.float64),
                np.asarray(second_item[1], dtype=np.float64),
            )
            order = np.argsort(np.asarray(preview_times, dtype=np.float64))
            preview_times[:] = np.asarray(preview_times, dtype=np.float64)[order].tolist()
            preview_values[:] = np.asarray(preview_values, dtype=np.float64)[order].tolist()
    else:
        raise ValueError(f"Unknown frequency-weighting analysis mode: {mode!r}.")

    if not np.isfinite(weighted_peak_value) or weighted_peak_time is None:
        raise ValueError("Frequency weighting did not produce a finite regional peak.")

    grid = np.asarray(preview_times, dtype=np.float64)
    weighted_rms_uT = np.asarray(preview_values, dtype=np.float64)
    normalized_trace = weighted_rms_uT / peak_limit_uT
    weighted_peak_index = weighted_peak_value / peak_limit_uT

    # The spectral-sum descriptor uses the corresponding Phase-3 spectrum. For
    # segmented records this is the RMS spectral power averaged across the full
    # physical duration, not a spectrum of the bounded display preview.
    frequency = np.asarray(spectrum["frequency_hz"], dtype=np.float64)
    bin_rms = np.asarray(spectrum["bin_rms_uT"], dtype=np.float64)
    references = frequency_weighting_reference_uT(frequency, profile)
    spectral_valid = (
        (frequency >= FREQUENCY_WEIGHTING_MIN_HZ)
        & (frequency <= FREQUENCY_WEIGHTING_MAX_HZ)
        & np.isfinite(references)
        & (references > 0.0)
    )
    spectral_ratio = np.zeros_like(bin_rms)
    spectral_ratio[spectral_valid] = bin_rms[spectral_valid] / references[spectral_valid]

    below = (frequency > 0.0) & (frequency < FREQUENCY_WEIGHTING_MIN_HZ)
    above = frequency > FREQUENCY_WEIGHTING_MAX_HZ
    spectral_resolution = float(spectrum["frequency_resolution_hz"])
    nyquist = float(spectrum["nyquist_hz"])
    resolved_min = max(
        FREQUENCY_WEIGHTING_MIN_HZ,
        spectral_resolution,
        float(weighting_frequency_resolution),
    )
    resolved_max = min(FREQUENCY_WEIGHTING_MAX_HZ, nyquist)
    full_band = bool(
        resolved_min <= FREQUENCY_WEIGHTING_MIN_HZ * (1.0 + 1e-12)
        and resolved_max >= FREQUENCY_WEIGHTING_MAX_HZ * (1.0 - 1e-12)
    )

    result = {
        "profile": str(profile),
        "profile_label": FREQUENCY_WEIGHTING_PROFILES[str(profile)]["label"],
        "source": FREQUENCY_WEIGHTING_PROFILES[str(profile)]["source"],
        "reference_frequency_hz": FREQUENCY_WEIGHTING_REFERENCE_HZ,
        "reference_rms_uT": reference_at_fref,
        "reference_peak_uT": peak_limit_uT,
        "time_s": grid,
        "weighted_spatial_rms_uT": weighted_rms_uT,
        "weighted_index": normalized_trace,
        "weighted_peak_index": float(weighted_peak_index),
        "weighted_peak_percent": float(weighted_peak_index * 100.0),
        "weighted_peak_time_s": float(weighted_peak_time),
        "weighted_peak_equivalent_uT": float(weighted_peak_value),
        "spectral_sum_index": float(np.sum(spectral_ratio)),
        "spectral_sum_percent": float(np.sum(spectral_ratio) * 100.0),
        "frequency_hz": frequency,
        "bin_rms_uT": bin_rms,
        "reference_rms_by_bin_uT": references,
        "spectral_ratio_by_bin": spectral_ratio,
        "fft_samples": n,
        "fft_bins": len(frequency),
        "max_fft_samples": int(spectrum["max_fft_samples"]),
        "frequency_resolution_hz": spectral_resolution,
        "weighting_frequency_resolution_hz": float(weighting_frequency_resolution),
        "nyquist_hz": nyquist,
        "sample_spacing_s": spacing,
        "source_sample_spacing_s": float(spectrum["source_sample_spacing_s"]),
        "source_nyquist_hz": float(spectrum["source_nyquist_hz"]),
        "band_min_hz": FREQUENCY_WEIGHTING_MIN_HZ,
        "band_max_hz": FREQUENCY_WEIGHTING_MAX_HZ,
        "resolved_band_min_hz": float(resolved_min),
        "resolved_band_max_hz": float(resolved_max),
        "full_band_resolved": full_band,
        "below_band_rms_uT": float(np.sqrt(np.sum(bin_rms[below] ** 2))),
        "above_band_rms_uT": float(np.sqrt(np.sum(bin_rms[above] ** 2))),
        "max_source_interval_s": float(spectrum["source_sample_spacing_s"]),
        "analysis_mode": mode,
        "analysis_mode_label": str(spectrum.get("analysis_mode_label", mode)),
        "physical_duration_s": float(spectrum["physical_duration_s"]),
        "spectral_record_duration_s": float(spectrum["spectral_record_duration_s"]),
        "segment_count": int(spectrum.get("segment_count", 1)),
        "weighting_context_records": int(weighting_context_records),
        "weighting_fft_samples": int(weighting_sample_count),
        "weighted_trace_reduced": bool(weighted_trace_reduced),
        "spectral_sampling_reduced": bool(spectrum.get("spectral_sampling_reduced", False)),
    }
    if "repeat_count" in spectrum:
        result["repeat_count"] = int(spectrum["repeat_count"])
    if "stable_start_s" in spectrum:
        result["stable_start_s"] = spectrum.get("stable_start_s")
    return result


def analyse_regions_frequency_weighted(payload, *, profile="icnirp2010_public", fft_samples=None,
                                       progress_callback=None, spectra=None):
    """Weight one profile for all prepared regions, sharing replay and FFT work.

    Spectra share one current FFT per record. Weighting streams each context
    block once, then scores every regional quadratic form. Only bounded display
    traces and spectra are retained; the full exposure is never held in memory.
    The single-region entry point remains available for legacy callers.
    """
    keys = tuple(payload.get("regions", {}))
    if not keys:
        raise ValueError("No prepared regions are available.")
    if profile not in FREQUENCY_WEIGHTING_PROFILES:
        raise ValueError(f"Unknown frequency-weighting profile: {profile}")
    if spectra is None:
        spectra = analyse_regions_spectrum(payload, fft_samples=fft_samples,
            window="rectangular", detrend=True, progress_callback=progress_callback)
    if set(spectra) != set(keys):
        raise ValueError("The saved spectrum does not contain every prepared region.")
    first_spectrum = spectra[keys[0]]
    states = {k: {"times": [], "values": [], "peak": -np.inf, "peak_time": None}
              for k in keys[1:]}
    quadratics = {k: np.asarray(payload["regions"][k]["quadratic"], dtype=np.float64)
                  for k in states}

    def score(grid, currents, offset, total):
        for k, state in states.items():
            values = _weighted_region_spatial_rms_uT(quadratics[k], currents)
            index = int(np.argmax(values))
            if float(values[index]) > state["peak"]:
                state["peak"] = float(values[index])
                state["peak_time"] = float(grid[index])
            _frequency_weighting_preview_append(state["times"], state["values"], grid, values,
                                                global_offset=offset, total_samples=total)

    first = analyse_region_frequency_weighted(payload, keys[0], profile=profile,
        fft_samples=fft_samples, spectrum=first_spectrum, progress_callback=progress_callback,
        _block_observer=score)
    results = {keys[0]: first}
    frequency = first["frequency_hz"]
    refs = first["reference_rms_by_bin_uT"]
    valid = ((frequency >= FREQUENCY_WEIGHTING_MIN_HZ) &
             (frequency <= FREQUENCY_WEIGHTING_MAX_HZ) & np.isfinite(refs) & (refs > 0))
    below = (frequency > 0) & (frequency < FREQUENCY_WEIGHTING_MIN_HZ)
    above = frequency > FREQUENCY_WEIGHTING_MAX_HZ
    for column, k in enumerate(keys[1:], start=1):
        state = states[k]
        if not np.isfinite(state["peak"]) or state["peak_time"] is None:
            raise ValueError(f"Frequency weighting did not produce a finite peak for {k}.")
        order = np.argsort(state["times"], kind="stable")
        values = np.asarray(state["values"])[order]
        bins = spectra[k]["bin_rms_uT"]
        ratios = np.zeros_like(bins)
        ratios[valid] = bins[valid] / refs[valid]
        peak = state["peak"] / first["reference_peak_uT"]
        results[k] = dict(first, time_s=np.asarray(state["times"])[order],
            weighted_spatial_rms_uT=values, weighted_index=values / first["reference_peak_uT"],
            weighted_peak_index=peak, weighted_peak_percent=peak * 100,
            weighted_peak_time_s=state["peak_time"], weighted_peak_equivalent_uT=state["peak"],
            bin_rms_uT=bins, spectral_ratio_by_bin=ratios,
            spectral_sum_index=float(np.sum(ratios)), spectral_sum_percent=float(np.sum(ratios) * 100),
            below_band_rms_uT=float(np.sqrt(np.sum(bins[below] ** 2))),
            above_band_rms_uT=float(np.sqrt(np.sum(bins[above] ** 2))))
    return results


def analyse_region_waveform(payload, key, threshold_percent=1.0, progress_callback=None, *, _slew_observer=None):
    """Authoritative regional slew and drive-current pulse timing.

    This intentionally no longer performs an FFT. Spectral analysis has a
    different sampling problem and remains isolated in ``analyse_region_spectrum``.
    """
    try:
        threshold_percent = float(threshold_percent)
    except (TypeError, ValueError) as error:
        raise ValueError("Drive pulse threshold must be a finite percentage.") from error
    if not np.isfinite(threshold_percent) or threshold_percent < 0.0:
        raise ValueError("Drive pulse threshold must be a finite percentage of 0 or greater.")

    drive_ids = [str(value) for value in payload.get("drive_ids", [])]
    drive_count = len(drive_ids)
    _constant, _linear, quadratic = _region_moments(payload, key, drive_count)

    # Pulse thresholds are defined as a percentage of each drive's true peak
    # absolute current, so establish those peaks from the full packet stream
    # before classifying crossings. Progress work units deliberately follow the
    # retained packet samples actually evaluated, not the potentially enormous
    # logical repeat count represented by a verified cycle.
    work_samples = _time_domain_work_sample_count(payload)
    work_total = max(1, work_samples * (drive_count + 2))
    work_done = 0
    _emit_analysis_progress(
        progress_callback, "Timing pass 1/2 — finding drive peaks", 0, work_total
    )
    peak_currents = np.zeros(drive_count, dtype=np.float64)
    for _times, currents, _repeat_count in _iter_time_domain_packets(payload):
        if len(currents):
            peak_currents = np.maximum(
                peak_currents, np.max(np.abs(currents), axis=0)
            )
        work_done += len(currents)
        _emit_analysis_progress(
            progress_callback,
            "Timing pass 1/2 — finding drive peaks",
            min(work_done, work_total),
            work_total,
        )
    thresholds = peak_currents * threshold_percent / 100.0
    _emit_analysis_progress(
        progress_callback,
        "Timing pass 2/2 — measuring transitions",
        min(work_done, work_total),
        work_total,
    )

    groups = [_interval_group_from_intervals([]) for _ in range(drive_count)]
    first_time = None
    last_time = None
    previous_currents = None
    max_interval = 0.0
    peak_slew = -np.inf
    peak_slew_time = None
    logical_samples = 0

    def consider_slew(start_time, dt, delta):
        nonlocal peak_slew, peak_slew_time, max_interval
        dt = float(dt)
        if dt <= 0.0:
            return
        derivative = np.asarray(delta, dtype=np.float64) / dt
        if _slew_observer is not None:
            _slew_observer(np.asarray([start_time]), derivative[None, :])
        value = float(
            np.sqrt(
                max(
                    0.0,
                    float(derivative @ quadratic @ derivative),
                )
            )
            * 1e6
        )
        if value > peak_slew:
            peak_slew = value
            peak_slew_time = float(start_time)
        max_interval = max(max_interval, dt)

    for times, currents, repeat_count in _iter_time_domain_packets(payload):
        if first_time is None:
            first_time = float(times[0])
        elif last_time is not None:
            tolerance = _time_tolerance(last_time, times[0])
            if float(times[0]) < last_time - tolerance:
                raise ValueError("Authoritative timing analysis moved backwards in time.")
            join_dt = float(times[0]) - last_time
            if join_dt > 0.0:
                consider_slew(last_time, join_dt, currents[0] - previous_currents)
                join_times = np.asarray([last_time, float(times[0])], dtype=np.float64)
                for index in range(drive_count):
                    intervals = current_threshold_intervals(
                        join_times,
                        np.asarray(
                            [previous_currents[index], currents[0, index]],
                            dtype=np.float64,
                        ),
                        thresholds[index],
                    )
                    groups[index] = _combine_interval_groups(
                        groups[index], _interval_group_from_intervals(intervals)
                    )

        if len(times) >= 2:
            dt = np.diff(times)
            derivatives = np.diff(currents, axis=0) / dt[:, None]
            if _slew_observer is not None:
                _slew_observer(times[:-1], derivatives)
            slew = (
                np.sqrt(
                    np.maximum(
                        0.0,
                        np.einsum(
                            "tc,cd,td->t",
                            derivatives,
                            quadratic,
                            derivatives,
                            optimize=True,
                        ),
                    )
                )
                * 1e6
            )
            local_index = int(np.argmax(slew))
            local_peak = float(slew[local_index])
            if local_peak > peak_slew:
                peak_slew = local_peak
                peak_slew_time = float(times[local_index])
            max_interval = max(max_interval, float(np.max(dt)))

            for index in range(drive_count):
                intervals = current_threshold_intervals(
                    times, currents[:, index], thresholds[index]
                )
                if repeat_count > 1:
                    group = _repeat_interval_group(
                        intervals, times[0], times[-1], repeat_count
                    )
                else:
                    group = _interval_group_from_intervals(intervals)
                groups[index] = _combine_interval_groups(groups[index], group)

        work_done += len(times) * (drive_count + 1)
        _emit_analysis_progress(
            progress_callback,
            "Timing pass 2/2 — measuring transitions",
            min(work_done, work_total),
            work_total,
        )
        previous_currents = currents[-1].copy()
        last_time = _logical_packet_end(times, repeat_count)
        logical_samples += len(times) * repeat_count

    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError("Timing analysis requires a positive physical duration.")
    duration = float(last_time - first_time)
    if not np.isfinite(peak_slew):
        peak_slew = 0.0
        peak_slew_time = first_time

    drive_timing = []
    for drive_id, threshold, group in zip(
        drive_ids, thresholds, groups, strict=True
    ):
        widths = _interval_width_stats(group)
        gaps = group["gaps"]
        preview = [tuple(value) for value in group.get("preview", [])]
        preview_widths = [end - start for start, end in preview]
        preview_gaps = [
            max(0.0, right[0] - left[1])
            for left, right in zip(preview[:-1], preview[1:], strict=True)
        ]
        active_time = float(group["total_s"])
        drive_timing.append(
            {
                "id": drive_id,
                "threshold_a": float(threshold),
                "pulses": int(group["count"]),
                "active_s": active_time,
                "inactive_s": max(0.0, duration - active_time),
                "pulse_width_min_s": float(widths["min"]) if widths["count"] else 0.0,
                "pulse_width_max_s": float(widths["max"]) if widths["count"] else 0.0,
                "pulse_width_mean_s": float(widths["total"] / widths["count"]) if widths["count"] else 0.0,
                "gap_count": int(gaps["count"]),
                "gap_width_min_s": float(gaps["min"]) if gaps["count"] else 0.0,
                "gap_width_max_s": float(gaps["max"]) if gaps["count"] else 0.0,
                "gap_width_mean_s": float(gaps["total"] / gaps["count"]) if gaps["count"] else 0.0,
                "pulse_widths_s": preview_widths,
                "gap_widths_s": preview_gaps,
                "timing_preview_truncated": int(group["count"]) > len(preview),
            }
        )

    _emit_analysis_progress(progress_callback, "Timing analysis ready", work_total, work_total)
    return {
        "peak_spatial_rms_slew_uT_s": float(peak_slew),
        "slew_time_s": float(peak_slew_time),
        "max_interval_s": float(max_interval),
        "duration_s": duration,
        "source_sample_count": int(logical_samples),
        "analysis_source": "authoritative_packets" if isinstance(payload.get("authoritative_analysis"), dict) else "full_retained_trace",
        "drive_timing": drive_timing,
    }



def analyse_regions_spectrum(payload, *, fft_samples=None, window="rectangular", detrend=True,
                             progress_callback=None):
    """One source replay and FFT per record, scored for every prepared region."""
    keys = tuple(payload.get("regions", {}))
    if not keys:
        raise ValueError("No prepared regions are available.")
    result = analyse_region_spectrum(payload, keys[0], fft_samples=fft_samples,
        window=window, detrend=detrend, progress_callback=progress_callback, _region_keys=keys)
    return {key: dict(result, bin_rms_uT=result["bin_rms_uT"][:, i],
                     dc_rms_uT=result["dc_rms_uT"][i]) for i, key in enumerate(keys)}


def analyse_regions_waveform(payload, threshold_percent=1.0, progress_callback=None):
    """Share current peaks, pulse intervals and derivatives across all regions."""
    keys = tuple(payload.get("regions", {}))
    if not keys:
        raise ValueError("No prepared regions are available.")
    quadratics = {key: _region_moments(payload, key, len(payload.get("drive_ids", [])))[2]
                  for key in keys[1:]}
    peaks = {key: (-np.inf, None) for key in quadratics}

    def score(times, derivatives):
        for key, quadratic in quadratics.items():
            values = np.sqrt(np.maximum(0., np.einsum("tc,cd,td->t", derivatives,
                             quadratic, derivatives, optimize=True))) * 1e6
            index = int(np.argmax(values))
            if values[index] > peaks[key][0]:
                peaks[key] = (float(values[index]), float(times[index]))

    first = analyse_region_waveform(payload, keys[0], threshold_percent,
                                   progress_callback, _slew_observer=score)
    results = {keys[0]: first}
    for key, (peak, when) in peaks.items():
        if not np.isfinite(peak):
            peak, when = 0., first["slew_time_s"]
        results[key] = dict(first, peak_spatial_rms_slew_uT_s=peak, slew_time_s=when)
    return results
