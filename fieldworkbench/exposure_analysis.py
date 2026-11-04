"""Dedicated saved-exposure analysis workflow.

Authoritative exposure statistics deliberately live outside Scripted playback.
This module prepares a frozen regional field basis + physical exposure timeline
from the Exposure Editor, then hands that payload to the existing guided
analysis workspace (summary, spectrum/FFT, timing, threshold and ICNIRP/EU
frequency-weighted screening).

Targets include LPBA40 Brain Areas, geometry assigned the Measurement role, and
closed meshes with authored surface regions. Surface-region targets are resolved
into configurable inward near-surface volumes before entering the same generic
regional-analysis pipeline. Each target retains its own sampling, preparation and
derived-result identity, independent of playback. Prepared payloads carry dependency
fingerprints and are maintained automatically as first-class project analysis caches.
"""
from __future__ import annotations

import copy
import numpy as np
from typing import Any, Callable

from .brain_analysis import lpba40_region_hierarchy, sample_lpba40_voxels
from .exposures import exposure_coil_ids, normalize_exposure_protocol
from .region_analysis import waveform_basis_payload
from .spatial_analysis import (
    AnalysisRegionNode, AnalysisRegionSet, AnalysisSampleSet,
    REGIONAL_ANALYSIS_SCHEMA_VERSION, analysis_region_hierarchy_payload,
    analysis_region_set_identity,
)
from .sri24_atlas import load_sri24_parcellation
from .studio_adapter import StudioAdapter, StudioOperationError
from .waveform import _prepare_exposure_current_timeline, field_bases_for_drives


# Match Brain View's authoritative LPBA40 analysis density.  Keep this local to
# the analysis engine rather than importing brain_view.py (which imports the
# full Maps UI and would create a circular dependency through dialogs.py).
EXPOSURE_BRAIN_SAMPLES_PER_LABEL = 900
EXPOSURE_ANALYSIS_RETAINED_TRACE_S = 2.0


class ExposureAnalysisCancelled(RuntimeError):
    """Raised internally when a user cancels analysis preparation."""


def brain_exposure_analysis_targets(adapter: StudioAdapter) -> list[dict[str, str]]:
    """Return SRI24 brain objects that can anchor an LPBA40 exposure analysis."""

    targets: list[dict[str, str]] = []
    for entry in adapter.list_objects():
        object_id = str(entry.get("id", "") or "").strip()
        if not object_id:
            continue
        try:
            obj = adapter.get_object(object_id)
        except Exception:  # noqa: BLE001 - catalogue boundary
            continue
        if str(obj.get("builtin_template", "") or "") != "sri24_brain":
            continue
        targets.append(
            {
                "id": object_id,
                "label": str(entry.get("label") or obj.get("label") or "SRI24 brain"),
                "kind": "brain_lpba40",
                "atlas": "LPBA40",
            }
        )
    return targets


def _is_surface_region_analysis_target(adapter: StudioAdapter, target_id: str) -> bool:
    """Return whether an object provides authored near-surface analysis regions.

    Authored surface regions are intrinsic analysis geometry and therefore take
    precedence over the object's scene role.  Assigning the same bust the
    Measurement role must not replace Face/Scalp/etc. with a whole-object
    Measurement volume inside Exposure Analysis.
    """
    try:
        target = adapter.get_object(str(target_id))
    except Exception:
        return False
    if target.get("derived"):
        return False
    return bool(adapter.supports_surface_region_volume(str(target_id)))


def exposure_analysis_targets(adapter: StudioAdapter) -> list[dict[str, str]]:
    """Eligible brain, near-surface, and Measurement objects by stable object id."""
    targets = brain_exposure_analysis_targets(adapter)
    brain_ids = {target["id"] for target in targets}
    for entry in adapter.list_objects():
        key = str(entry.get("id", ""))
        if not key or key in brain_ids:
            continue
        obj = adapter.get_object(key)
        if _is_surface_region_analysis_target(adapter, key):
            targets.append({
                "id": key,
                "label": str(entry.get("label") or key),
                "kind": "surface_region_volume",
                "atlas": "",
            })
        elif (not obj.get("derived") and obj.get("role") == "measurement"
                and adapter.supports_measurement(key)):
            targets.append({"id": key, "label": str(entry.get("label") or key),
                            "kind": "measurement_volume", "atlas": ""})
    return targets


def _measurement_target_samples(adapter, target_id):
    target = adapter.get_object(str(target_id))
    if target.get("role") != "measurement" or target.get("derived"):
        raise StudioOperationError("Choose a measurement volume from the current model.")
    sample = adapter.measurement_sample_points(str(target_id))
    points = np.asarray(sample["world_points_m"], dtype=float).reshape(-1, 3)
    count = len(points)
    if not count or not np.isfinite(points).all():
        raise StudioOperationError("The measurement volume has no valid sample points.")
    # Generated centres represent equal volumes. User points have equal sample
    # weights; their distribution must not be presented as a volume integral.
    volume_mm3 = max(float(sample["volume_cm3"]) * 1000., 1.)
    samples = AnalysisSampleSet(
        points, np.ones(count, dtype=np.int32),
        np.full(count, volume_mm3 / count), np.arange(count),
        {1: count}, {1: count}, 1.0, label_volumes_mm3={1: volume_mm3},
    )
    name = str(sample["label"])
    if sample["sampling_mode"] == "user_defined":
        name += " [samples]"
    return AnalysisRegionNode(str(target_id), name, (1,), ()), samples


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    cancel_requested: Callable[[], bool] | None,
    update: dict[str, Any],
) -> None:
    if callable(cancel_requested) and cancel_requested():
        raise ExposureAnalysisCancelled("Exposure analysis preparation was cancelled.")
    if callable(callback):
        callback(dict(update))


def _brain_target_samples(adapter: StudioAdapter, target_id: str):
    try:
        target = adapter.get_object(str(target_id))
    except KeyError as error:
        raise StudioOperationError("The selected brain analysis target is no longer present.") from error
    if str(target.get("builtin_template", "") or "") != "sri24_brain":
        raise StudioOperationError("Exposure Brain Areas analysis requires a built-in SRI24 brain target.")

    parcellation = load_sri24_parcellation("lpba40")
    hierarchy = lpba40_region_hierarchy(parcellation)
    mesh = adapter._mesh(str(target_id))
    position_m, rotation = adapter._mesh_world_transform(mesh)
    scale_xyz = adapter._mesh_scale_xyz(mesh)
    samples = sample_lpba40_voxels(
        parcellation,
        position_m=position_m,
        rotation=rotation,
        scale_xyz=scale_xyz,
        maximum_samples_per_label=EXPOSURE_BRAIN_SAMPLES_PER_LABEL,
    )
    return hierarchy, samples


def _cached_region_set_for_target(adapter: StudioAdapter, target_id: str):
    identity = adapter.analysis_region_geometry_cache_identity(str(target_id))
    cached = adapter.cached_analysis_region_set(identity)
    return identity, cached


def _refresh_cached_target_metadata(
    region_set: AnalysisRegionSet, *, target_name: str
) -> AnalysisRegionSet:
    metadata = dict(region_set.metadata or {})
    metadata["target_name"] = str(target_name)
    return AnalysisRegionSet(
        key=region_set.key,
        name=region_set.name,
        kind=region_set.kind,
        hierarchy=region_set.hierarchy,
        samples=region_set.samples,
        sampling_mode=region_set.sampling_mode,
        metadata=metadata,
    )


def _brain_region_set(adapter: StudioAdapter, target_id: str) -> AnalysisRegionSet:
    target = adapter.get_object(str(target_id))
    target_name = str(target.get("label") or "SRI24 brain")
    cache_identity, cached = _cached_region_set_for_target(adapter, target_id)
    if isinstance(cached, AnalysisRegionSet):
        return _refresh_cached_target_metadata(cached, target_name=target_name)
    hierarchy, samples = _brain_target_samples(adapter, target_id)
    result = AnalysisRegionSet(
        key="sri24.lpba40",
        name="LPBA40 brain regions",
        kind="atlas_volume",
        hierarchy=hierarchy,
        samples=samples,
        sampling_mode="atlas",
        metadata={
            "target_kind": "brain_lpba40",
            "target_name": target_name,
            "atlas": "LPBA40",
            "provider": "sri24_lpba40",
            "provider_version": 1,
            "geometry_cache_identity": cache_identity,
        },
    )
    adapter.cache_analysis_region_set(
        cache_identity, result, target_id=str(target_id), target_name=target_name
    )
    return result


def _measurement_region_set(adapter: StudioAdapter, target_id: str) -> AnalysisRegionSet:
    target = adapter.get_object(str(target_id))
    target_name = str(target.get("label") or target_id)
    cache_identity, cached = _cached_region_set_for_target(adapter, target_id)
    if isinstance(cached, AnalysisRegionSet):
        return _refresh_cached_target_metadata(cached, target_name=target_name)
    hierarchy, samples = _measurement_target_samples(adapter, target_id)
    settings = adapter.measurement_properties(str(target_id))
    result = AnalysisRegionSet(
        key=f"measurement:{target_id}",
        name=str(target.get("label") or "Measurement volume"),
        kind="measurement_volume",
        hierarchy=hierarchy,
        samples=samples,
        sampling_mode=str(settings["sampling_mode"]),
        metadata={
            "target_kind": "measurement_volume",
            "target_name": target_name,
            "atlas": "",
            "provider": "measurement_volume",
            "provider_version": 1,
            "geometry_cache_identity": cache_identity,
        },
    )
    adapter.cache_analysis_region_set(
        cache_identity, result, target_id=str(target_id), target_name=target_name
    )
    return result


def _surface_region_set(
    adapter: StudioAdapter,
    target_id: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> AnalysisRegionSet:
    target = adapter.get_object(str(target_id))
    target_name = str(target.get("label") or target_id)
    try:
        result = adapter.surface_region_volume_set(
            str(target_id),
            progress_callback=progress_callback,
            cancel_requested=cancel_requested,
        )
    except StudioOperationError:
        if callable(cancel_requested) and cancel_requested():
            raise ExposureAnalysisCancelled(
                "Exposure analysis preparation was cancelled."
            )
        raise
    if callable(cancel_requested) and cancel_requested():
        raise ExposureAnalysisCancelled("Exposure analysis preparation was cancelled.")
    metadata = dict(result.metadata or {})
    metadata.update({
        "target_kind": "surface_region_volume",
        "target_name": target_name,
        "atlas": "",
        "provider": "surface_region_volume",
    })
    try:
        depth_mm = float(metadata.get("depth_mm", metadata.get("sampling_depth_mm")))
        set_name = f"{target_name} surface regions ({depth_mm:g} mm inward)"
    except (TypeError, ValueError):
        set_name = f"{target_name} surface regions"
    return AnalysisRegionSet(
        key=result.key,
        name=set_name,
        kind=result.kind,
        hierarchy=result.hierarchy,
        samples=result.samples,
        sampling_mode=result.sampling_mode,
        metadata=metadata,
    )


def resolve_exposure_analysis_region_set(
    adapter: StudioAdapter,
    target_id: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> AnalysisRegionSet:
    """Resolve any supported object into the shared regional-analysis contract."""
    target = adapter.get_object(str(target_id))
    if str(target.get("builtin_template", "") or "") == "sri24_brain":
        return _brain_region_set(adapter, str(target_id))
    if _is_surface_region_analysis_target(adapter, str(target_id)):
        return _surface_region_set(
            adapter,
            str(target_id),
            progress_callback=progress_callback,
            cancel_requested=cancel_requested,
        )
    if (
        not target.get("derived")
        and target.get("role") == "measurement"
        and adapter.supports_measurement(str(target_id))
    ):
        return _measurement_region_set(adapter, str(target_id))
    raise StudioOperationError("Choose an eligible regional analysis target.")


def prepare_exposure_analysis_payload(
    adapter: StudioAdapter,
    exposure: dict[str, Any],
    *,
    target_id: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build the frozen regional payload consumed by Exposure Analysis.

    The returned object contains only the compact regional second moments,
    a bounded retained display trace, and the replayable authoritative exposure
    source.  It does *not* build animation frames or a WebGL map.
    """

    target = adapter.get_object(str(target_id))
    is_brain = str(target.get("builtin_template", "")) == "sri24_brain"
    is_surface_region = _is_surface_region_analysis_target(adapter, str(target_id))
    is_measurement = (
        not target.get("derived")
        and target.get("role") == "measurement"
        and adapter.supports_measurement(str(target_id))
    )
    if not is_brain and not is_surface_region and not is_measurement:
        raise StudioOperationError("Choose an eligible regional analysis target.")
    source = copy.deepcopy(exposure if isinstance(exposure, dict) else {})
    protocol = normalize_exposure_protocol(source.get("protocol"))
    duration_s = float(protocol["duration_s"])
    drive_ids = exposure_coil_ids(protocol)
    if not drive_ids:
        raise StudioOperationError("The selected exposure does not route any physical coils.")
    missing = [coil_id for coil_id in drive_ids if not adapter.is_coil(coil_id)]
    if missing:
        labels = protocol.get("coil_labels", {}) if isinstance(protocol.get("coil_labels"), dict) else {}
        names = [str(labels.get(value) or value) for value in missing]
        raise StudioOperationError(
            "The exposure references coil(s) not present in the current model: " + ", ".join(names)
        )

    exposure_id = str(source.get("id", "") or "")
    dependency_signature = adapter.exposure_analysis_dependency_signature(
        exposure_id, str(target_id)
    )

    _emit_progress(
        progress_callback,
        cancel_requested,
        {
            "stage": "Preparing the physical exposure timeline…",
            "completed": 0,
            "total": 1000,
            "detail": f"{len(drive_ids):,} driven coil(s) • full exposure {duration_s:.6g} s",
        },
    )

    def timeline_progress(update: dict[str, Any]) -> None:
        forwarded = dict(update) if isinstance(update, dict) else {"stage": str(update)}
        forwarded["stage"] = str(forwarded.get("stage") or "Preparing exposure timeline…")
        _emit_progress(progress_callback, cancel_requested, forwarded)

    # Exposure Analysis is always authoritative for the complete physical exposure.
    # Only the lightweight display trace is bounded; replayable analysis packets
    # still represent the full saved duration.
    retained_stop_s = min(duration_s, EXPOSURE_ANALYSIS_RETAINED_TRACE_S)
    timeline = _prepare_exposure_current_timeline(
        adapter,
        protocol,
        coil_ids=drive_ids,
        start_s=0.0,
        stop_s=retained_stop_s,
        frame_count=2,
        analysis_samples=2,
        statistics_scope="full",
        include_authoritative_analysis=True,
        progress_callback=timeline_progress,
    )

    _emit_progress(
        progress_callback,
        cancel_requested,
        {
            "stage": "Preparing target samples…",
            "completed": 0,
            "total": 1,
            "detail": "Using the selected object’s placement, rotation and scale.",
        },
    )
    def region_progress(update: dict[str, Any]) -> None:
        forwarded = dict(update) if isinstance(update, dict) else {"stage": str(update)}
        forwarded["stage"] = str(
            forwarded.get("stage") or "Preparing regional target geometry…"
        )
        _emit_progress(progress_callback, cancel_requested, forwarded)

    region_set = resolve_exposure_analysis_region_set(
        adapter,
        str(target_id),
        progress_callback=region_progress,
        cancel_requested=cancel_requested,
    )
    cache_identities = adapter.exposure_analysis_cache_identities(
        exposure_id, str(target_id)
    )
    region_content_identity = analysis_region_set_identity(region_set)
    hierarchy = region_set.hierarchy
    samples = region_set.samples
    sample_count = int(samples.point_count)
    sample_detail = f"{sample_count:,} sample points."
    if str(region_set.kind) == "surface_region_volume":
        depth_mm = (region_set.metadata or {}).get("depth_mm", (region_set.metadata or {}).get("sampling_depth_mm"))
        try:
            sample_detail = (
                f"{sample_count:,} retained field samples across "
                f"{len(region_set.hierarchy.children):,} near-surface region(s) at "
                f"{float(depth_mm):g} mm depth."
            )
        except (TypeError, ValueError):
            pass
    _emit_progress(
        progress_callback,
        cancel_requested,
        {
            "stage": "Target samples ready",
            "completed": 1,
            "total": 1,
            "detail": sample_detail,
        },
    )

    def basis_progress(update: dict[str, Any]) -> None:
        forwarded = dict(update) if isinstance(update, dict) else {"stage": str(update)}
        forwarded["stage"] = "Field calculation — " + str(
            forwarded.get("stage") or "sampling magnetic field"
        )
        _emit_progress(progress_callback, cancel_requested, forwarded)

    fixed, bases = field_bases_for_drives(
        adapter,
        coil_ids=drive_ids,
        zero_coil_ids=drive_ids,
        points_m=samples.points_m,
        field="B",
        progress_callback=basis_progress,
    )

    _emit_progress(
        progress_callback,
        cancel_requested,
        {
            "stage": "Preparing regional results…",
            "completed": 0,
            "total": 1,
            "detail": "Preparing the regions for the selected exposure.",
        },
    )
    payload = waveform_basis_payload(
        fixed,
        bases,
        samples,
        hierarchy,
        timeline,
        drive_ids,
    )
    # ``waveform_basis_payload`` is also used by viewer-side regional data and
    # therefore carries generic Preview/Full bookkeeping. Exposure Analysis has
    # only one meaning now: the complete physical exposure. Keep that duration in
    # its dedicated context instead of persisting a redundant scope selector.
    payload.pop("statistics_scope", None)
    payload.pop("statistics_duration_s", None)
    region_metadata = dict(region_set.metadata or {})
    context = {
        "exposure_id": exposure_id,
        "exposure_name": str(source.get("name") or "Exposure"),
        "target_kind": str(region_metadata.get("target_kind") or region_set.kind),
        "target_id": str(target_id),
        "target_name": str(region_metadata.get("target_name") or region_set.name),
        "atlas": str(region_metadata.get("atlas") or ""),
        "sampling_mode": str(region_set.sampling_mode),
        "sampling_depth_mm": None,
        "sample_count": int(sample_count),
        "drive_ids": list(drive_ids),
        "physical_duration_s": duration_s,
        "dependency_signature": dependency_signature,
        "geometry_cache_identity": str(
            region_metadata.get("geometry_cache_identity")
            or cache_identities["geometry_identity"]
        ),
        "region_geometry_content_identity": region_content_identity,
        "exposure_data_identity": cache_identities["exposure_data_identity"],
        "preparation_identity": cache_identities["preparation_identity"],
        "analysis_engine": "regional_analysis_v2",
        "regional_schema_version": REGIONAL_ANALYSIS_SCHEMA_VERSION,
        "region_set": {
            "key": str(region_set.key),
            "name": str(region_set.name),
            "kind": str(region_set.kind),
            "sampling_mode": str(region_set.sampling_mode),
            "provider": str(region_metadata.get("provider") or region_set.kind),
            "provider_version": int(region_metadata.get("provider_version", 1) or 1),
        },
        "region_hierarchy": analysis_region_hierarchy_payload(hierarchy),
    }
    if str(region_set.kind) == "surface_region_volume":
        try:
            context["sampling_depth_mm"] = float(
                region_metadata.get(
                    "depth_mm", region_metadata.get("sampling_depth_mm")
                )
            )
        except (TypeError, ValueError):
            context["sampling_depth_mm"] = None
    payload["exposure_analysis_context"] = context
    _emit_progress(
        progress_callback,
        cancel_requested,
        {
            "stage": "Exposure analysis is ready",
            "completed": 1,
            "total": 1,
            "detail": (
                f"{len(payload.get('regions', {})):,} regions • "
                f"{int(timeline.get('analysis_sample_count_estimate', 0) or 0):,} logical analysis samples"
            ),
        },
    )
    return payload
