"""Generic spatial-region analysis primitives and field statistics.

This module owns the numerical contract shared by atlas regions, ordinary
measurement volumes, and derived region providers such as near-surface bust
shells.  Region providers are responsible only for resolving a hierarchy
and a weighted sample set; downstream regional calculations are source-agnostic.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np


REGIONAL_ANALYSIS_SCHEMA_VERSION = 1
REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION = 1


class RegionalAnalysisError(ValueError):
    """Raised when a generic regional definition or calculation is invalid."""


@dataclass(frozen=True)
class AnalysisRegionNode:
    """One node in a source-agnostic regional hierarchy.

    ``label_ids`` identify the atomic labels owned by this row.  A region
    provider may attach lightweight metadata for presentation/provenance; the
    numerical engine only requires ``key``, ``name``, ``label_ids`` and
    ``children``.
    """

    key: str
    name: str
    label_ids: tuple[int, ...]
    children: tuple["AnalysisRegionNode", ...] = ()
    region_kind: str = "volume"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def walk(self) -> Iterator["AnalysisRegionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True, init=False)
class AnalysisSampleSet:
    """Weighted spatial samples resolved by any analysis-region provider.

    ``weights_mm3`` are the numerical integration weights used by regional
    statistics. ``source_counts`` and ``sampled_counts`` preserve source/retained
    population metadata. ``label_volumes_mm3`` gives authoritative physical
    volume independently of sample density, which is important for non-atlas
    providers. Legacy ``source_voxel_counts``/``voxel_volume_mm3`` aliases are
    retained for the LPBA40-facing API.
    """

    points_m: np.ndarray
    label_ids: np.ndarray
    weights_mm3: np.ndarray
    source_indices: np.ndarray
    source_counts: dict[int, int]
    sampled_counts: dict[int, int]
    nominal_element_volume_mm3: float
    label_volumes_mm3: dict[int, float]

    def __init__(
        self,
        points_m,
        label_ids,
        weights_mm3,
        source_indices,
        source_counts=None,
        sampled_counts=None,
        nominal_element_volume_mm3=1.0,
        label_volumes_mm3=None,
        *,
        source_voxel_counts=None,
        voxel_volume_mm3=None,
    ) -> None:
        if source_counts is None:
            source_counts = source_voxel_counts
        elif source_voxel_counts is not None:
            raise RegionalAnalysisError("Specify source_counts or source_voxel_counts, not both.")
        if source_counts is None:
            source_counts = {}
        if sampled_counts is None:
            sampled_counts = {}
        if voxel_volume_mm3 is not None:
            nominal_element_volume_mm3 = voxel_volume_mm3

        points = np.asarray(points_m, dtype=float)
        labels = np.asarray(label_ids, dtype=np.int32)
        weights = np.asarray(weights_mm3, dtype=float)
        source_idx = np.asarray(source_indices, dtype=np.int64)
        source = {int(label): int(count) for label, count in dict(source_counts).items()}
        sampled = {int(label): int(count) for label, count in dict(sampled_counts).items()}
        nominal = float(nominal_element_volume_mm3)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise RegionalAnalysisError("Regional sample points must be an N×3 array.")
        if labels.shape != (len(points),) or weights.shape != (len(points),) or source_idx.shape != (len(points),):
            raise RegionalAnalysisError("Regional sample labels and weights must match the points.")
        if np.any(source_idx < 0):
            raise RegionalAnalysisError("Regional sample source indices cannot be negative.")
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise RegionalAnalysisError("Regional samples contain non-finite coordinates or weights.")
        if np.any(weights <= 0.0):
            raise RegionalAnalysisError("Regional sample weights must be positive.")
        unique, counts = np.unique(labels, return_counts=True)
        actual = {int(label): int(count) for label, count in zip(unique, counts, strict=True)}
        present = set(actual)
        if present != set(source) or present != set(sampled):
            raise RegionalAnalysisError("Regional sample count dictionaries must match the retained labels.")
        if any(value <= 0 for value in source.values()) or any(value <= 0 for value in sampled.values()):
            raise RegionalAnalysisError("Regional sample counts must be positive.")
        if any(sampled[label] != actual[label] for label in present):
            raise RegionalAnalysisError("Regional sampled counts do not match the label array.")
        if not math.isfinite(nominal) or nominal <= 0.0:
            raise RegionalAnalysisError("Regional nominal element volume must be positive.")

        if label_volumes_mm3 is None:
            volumes = {label: float(source[label]) * nominal for label in present}
        else:
            volumes = {int(label): float(volume) for label, volume in dict(label_volumes_mm3).items()}
            if set(volumes) != present:
                raise RegionalAnalysisError("Regional label volumes must match the retained labels.")
            if any(not math.isfinite(value) or value <= 0.0 for value in volumes.values()):
                raise RegionalAnalysisError("Regional label volumes must be finite and positive.")

        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "label_ids", labels)
        object.__setattr__(self, "weights_mm3", weights)
        object.__setattr__(self, "source_indices", source_idx)
        object.__setattr__(self, "source_counts", source)
        object.__setattr__(self, "sampled_counts", sampled)
        object.__setattr__(self, "nominal_element_volume_mm3", nominal)
        object.__setattr__(self, "label_volumes_mm3", volumes)

    @property
    def point_count(self) -> int:
        return int(len(self.points_m))

    @property
    def points_mm(self) -> np.ndarray:
        return np.asarray(self.points_m, dtype=float) * 1000.0

    @property
    def source_voxel_counts(self) -> dict[int, int]:
        return self.source_counts

    @property
    def voxel_volume_mm3(self) -> float:
        return self.nominal_element_volume_mm3

    def volume_mm3_for_labels(self, labels: Iterable[int]) -> float:
        return float(sum(self.label_volumes_mm3.get(int(label), 0.0) for label in set(labels)))


@dataclass(frozen=True)
class AnalysisRegionSet:
    """Resolved region source consumed by the shared Exposure Analysis backend."""

    key: str
    name: str
    kind: str
    hierarchy: AnalysisRegionNode
    samples: AnalysisSampleSet
    sampling_mode: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


def _cache_json_value(value: Any) -> Any:
    """Return a deterministic JSON-safe value for cache identity metadata."""

    if isinstance(value, dict):
        return {str(key): _cache_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cache_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_cache_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _cache_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _analysis_region_node_payload(node: AnalysisRegionNode) -> dict[str, Any]:
    return {
        "key": str(node.key),
        "name": str(node.name),
        "label_ids": [int(value) for value in node.label_ids],
        "children": [_analysis_region_node_payload(child) for child in node.children],
        "region_kind": str(node.region_kind),
        "metadata": _cache_json_value(node.metadata),
    }


def analysis_region_hierarchy_payload(node: AnalysisRegionNode) -> dict[str, Any]:
    """Return a JSON-safe hierarchy description for saved analysis UI metadata."""
    if not isinstance(node, AnalysisRegionNode):
        raise RegionalAnalysisError("An AnalysisRegionNode hierarchy is required.")
    return _analysis_region_node_payload(node)


def analysis_region_hierarchy_from_payload(value: Any) -> AnalysisRegionNode:
    """Restore a hierarchy description saved with an Exposure Analysis payload."""
    return _analysis_region_node_from_payload(value)


def _analysis_region_node_from_payload(value: Any) -> AnalysisRegionNode:
    if not isinstance(value, dict):
        raise RegionalAnalysisError("Cached regional hierarchy node must be an object.")
    raw_children = value.get("children", [])
    if not isinstance(raw_children, list):
        raise RegionalAnalysisError("Cached regional hierarchy children must be an array.")
    try:
        labels = tuple(int(item) for item in value.get("label_ids", []))
    except (TypeError, ValueError) as error:
        raise RegionalAnalysisError("Cached regional hierarchy labels are invalid.") from error
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return AnalysisRegionNode(
        key=str(value.get("key", "")),
        name=str(value.get("name", value.get("key", ""))),
        label_ids=labels,
        children=tuple(_analysis_region_node_from_payload(child) for child in raw_children),
        region_kind=str(value.get("region_kind", "volume") or "volume"),
        metadata=dict(metadata),
    )


def analysis_region_set_cache_payload(region_set: AnalysisRegionSet) -> dict[str, Any]:
    """Serialize one resolved region set for the shared geometry-cache layer.

    Numeric arrays intentionally remain ndarrays here.  The project persistence
    codec stores them in its binary array pool rather than expanding them to JSON.
    """

    if not isinstance(region_set, AnalysisRegionSet):
        raise RegionalAnalysisError("A resolved AnalysisRegionSet is required.")
    samples = region_set.samples
    return {
        "schema_version": REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION,
        "regional_schema_version": REGIONAL_ANALYSIS_SCHEMA_VERSION,
        "key": str(region_set.key),
        "name": str(region_set.name),
        "kind": str(region_set.kind),
        "sampling_mode": str(region_set.sampling_mode),
        "metadata": _cache_json_value(region_set.metadata),
        "hierarchy": _analysis_region_node_payload(region_set.hierarchy),
        "samples": {
            "points_m": np.asarray(samples.points_m, dtype=np.float64),
            "label_ids": np.asarray(samples.label_ids, dtype=np.int32),
            "weights_mm3": np.asarray(samples.weights_mm3, dtype=np.float64),
            "source_indices": np.asarray(samples.source_indices, dtype=np.int64),
            "source_counts": {str(key): int(value) for key, value in samples.source_counts.items()},
            "sampled_counts": {str(key): int(value) for key, value in samples.sampled_counts.items()},
            "nominal_element_volume_mm3": float(samples.nominal_element_volume_mm3),
            "label_volumes_mm3": {
                str(key): float(value) for key, value in samples.label_volumes_mm3.items()
            },
        },
    }


def analysis_region_set_from_cache_payload(value: Any) -> AnalysisRegionSet:
    """Restore a resolved region set from :func:`analysis_region_set_cache_payload`."""

    if not isinstance(value, dict):
        raise RegionalAnalysisError("Cached regional geometry must be an object.")
    try:
        cache_version = int(value.get("schema_version", 0))
        regional_version = int(value.get("regional_schema_version", 0))
    except (TypeError, ValueError) as error:
        raise RegionalAnalysisError("Cached regional geometry version is invalid.") from error
    if cache_version != REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION:
        raise RegionalAnalysisError(
            f"Unsupported regional geometry cache schema {cache_version}; "
            f"expected {REGIONAL_GEOMETRY_CACHE_SCHEMA_VERSION}."
        )
    if regional_version != REGIONAL_ANALYSIS_SCHEMA_VERSION:
        raise RegionalAnalysisError(
            f"Cached regional geometry uses regional schema {regional_version}; "
            f"expected {REGIONAL_ANALYSIS_SCHEMA_VERSION}."
        )
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, dict):
        raise RegionalAnalysisError("Cached regional geometry samples are missing.")

    def counts(name: str) -> dict[int, int]:
        raw = raw_samples.get(name, {})
        if not isinstance(raw, dict):
            raise RegionalAnalysisError(f"Cached regional {name} must be an object.")
        try:
            return {int(key): int(item) for key, item in raw.items()}
        except (TypeError, ValueError) as error:
            raise RegionalAnalysisError(f"Cached regional {name} is invalid.") from error

    raw_volumes = raw_samples.get("label_volumes_mm3", {})
    if not isinstance(raw_volumes, dict):
        raise RegionalAnalysisError("Cached regional label volumes must be an object.")
    try:
        volumes = {int(key): float(item) for key, item in raw_volumes.items()}
        nominal = float(raw_samples.get("nominal_element_volume_mm3", 1.0))
    except (TypeError, ValueError) as error:
        raise RegionalAnalysisError("Cached regional sample volumes are invalid.") from error
    samples = AnalysisSampleSet(
        np.asarray(raw_samples.get("points_m"), dtype=float),
        np.asarray(raw_samples.get("label_ids"), dtype=np.int32),
        np.asarray(raw_samples.get("weights_mm3"), dtype=float),
        np.asarray(raw_samples.get("source_indices"), dtype=np.int64),
        counts("source_counts"),
        counts("sampled_counts"),
        nominal,
        label_volumes_mm3=volumes,
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return AnalysisRegionSet(
        key=str(value.get("key", "")),
        name=str(value.get("name", value.get("key", ""))),
        kind=str(value.get("kind", "regional") or "regional"),
        hierarchy=_analysis_region_node_from_payload(value.get("hierarchy")),
        samples=samples,
        sampling_mode=str(value.get("sampling_mode", "provider") or "provider"),
        metadata=dict(metadata),
    )


def analysis_region_set_identity(region_set: AnalysisRegionSet) -> str:
    """Return a content identity for resolved geometry/sampling membership.

    This is intentionally independent of any exposure or analysis-page settings.
    It is suitable for provenance and corruption checks.  Providers may also
    expose a cheaper *request* identity used to locate this cache before the
    expensive geometry has been regenerated.
    """

    payload = analysis_region_set_cache_payload(region_set)
    samples = payload.pop("samples")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    for name, dtype in (
        ("points_m", np.float64),
        ("label_ids", np.int32),
        ("weights_mm3", np.float64),
        ("source_indices", np.int64),
    ):
        array = np.ascontiguousarray(np.asarray(samples[name], dtype=dtype))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    scalar_samples = {
        key: value for key, value in samples.items()
        if key not in {"points_m", "label_ids", "weights_mm3", "source_indices"}
    }
    digest.update(
        json.dumps(scalar_samples, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class RegionMetrics:
    rms_uT: float
    mean_uT: float
    p95_uT: float
    peak_uT: float
    uniformity_pct: float
    directional_consistency_pct: float
    volume_cm3: float
    sample_count: int
    voxel_count: int


@dataclass(frozen=True)
class RegionPlaybackMetrics:
    playback_rms_uT: float
    mean_uT: float
    max_p95_uT: float
    absolute_peak_uT: float
    b_time_uT_s: float
    b2_time_uT2_s: float
    volume_cm3: float
    peak_time_s: float
    sample_count: int
    voxel_count: int


def _require_hierarchy(hierarchy: AnalysisRegionNode | None) -> AnalysisRegionNode:
    if hierarchy is None:
        raise RegionalAnalysisError("A regional hierarchy is required.")
    return hierarchy


def _weighted_percentiles(
    values: np.ndarray,
    weights: np.ndarray,
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    """Return deterministic weighted nearest-rank percentiles with one sort."""

    data = np.asarray(values, dtype=float).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if data.shape != mass.shape:
        raise RegionalAnalysisError("Weighted percentile arrays must have equal lengths.")
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return tuple(math.nan for _ in percentiles)
    data = data[valid]
    mass = mass[valid]
    order = np.argsort(data, kind="stable")
    data = data[order]
    cumulative = np.cumsum(mass[order])
    total = float(cumulative[-1])
    result: list[float] = []
    for percentile in percentiles:
        target = max(0.0, min(100.0, float(percentile))) / 100.0 * total
        index = int(np.searchsorted(cumulative, target, side="left"))
        result.append(float(data[min(index, len(data) - 1)]))
    return tuple(result)


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    """Return a deterministic weighted nearest-rank percentile."""

    return _weighted_percentiles(values, weights, (percentile,))[0]


def _metrics_for_mask(
    vectors_t: np.ndarray,
    samples: AnalysisSampleSet,
    mask: np.ndarray,
) -> RegionMetrics:
    vectors = np.asarray(vectors_t, dtype=float)[mask]
    weights = np.asarray(samples.weights_mm3, dtype=float)[mask]
    labels = np.asarray(samples.label_ids, dtype=np.int32)[mask]
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    vectors = vectors[valid]
    weights = weights[valid]
    labels = labels[valid]
    if not len(vectors):
        return RegionMetrics(
            *(math.nan,) * 7,
            sample_count=0,
            voxel_count=0,
        )

    magnitude_uT = np.linalg.norm(vectors, axis=1) * 1e6
    weight_total = float(np.sum(weights))
    mean_uT = float(np.sum(weights * magnitude_uT) / weight_total)
    rms_uT = float(math.sqrt(np.sum(weights * magnitude_uT**2) / weight_total))
    p5_uT, p95_uT = _weighted_percentiles(magnitude_uT, weights, (5.0, 95.0))
    peak_uT = float(np.max(magnitude_uT))
    if mean_uT > 1e-30 and math.isfinite(p5_uT) and math.isfinite(p95_uT):
        uniformity = max(0.0, min(100.0, 100.0 * (1.0 - (p95_uT - p5_uT) / mean_uT)))
    else:
        uniformity = 100.0 if p95_uT <= 1e-30 else 0.0

    magnitudes_t = np.linalg.norm(vectors, axis=1)
    directional = magnitudes_t > 1e-30
    if np.any(directional):
        direction_weights = weights[directional]
        unit_vectors = vectors[directional] / magnitudes_t[directional, None]
        mean_direction = np.sum(
            unit_vectors * direction_weights[:, None], axis=0
        ) / float(np.sum(direction_weights))
        directional_consistency = max(
            0.0, min(100.0, float(np.linalg.norm(mean_direction) * 100.0))
        )
    else:
        directional_consistency = 100.0

    unique_labels = set(int(value) for value in labels)
    voxel_count = sum(samples.source_voxel_counts[value] for value in unique_labels)
    volume_cm3 = (
        samples.volume_mm3_for_labels(unique_labels) / 1000.0
    )
    return RegionMetrics(
        rms_uT=rms_uT,
        mean_uT=mean_uT,
        p95_uT=p95_uT,
        peak_uT=peak_uT,
        uniformity_pct=uniformity,
        directional_consistency_pct=directional_consistency,
        volume_cm3=volume_cm3,
        sample_count=int(len(vectors)),
        voxel_count=int(voxel_count),
    )


def region_metrics(
    vectors_t: np.ndarray,
    samples: AnalysisSampleSet,
    hierarchy: AnalysisRegionNode | None = None,
) -> dict[str, RegionMetrics]:
    """Calculate metrics for every node in the regional hierarchy."""

    vectors = np.asarray(vectors_t, dtype=float)
    if vectors.shape != (samples.point_count, 3):
        raise RegionalAnalysisError(
            "Regional field vectors must contain one XYZ vector per region sample."
        )
    root = _require_hierarchy(hierarchy)
    result: dict[str, RegionMetrics] = {}
    for node in root.walk():
        mask = np.isin(samples.label_ids, np.asarray(node.label_ids, dtype=np.int32))
        result[node.key] = _metrics_for_mask(vectors, samples, mask)
    return result



def frame_region_metrics(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    frame_currents_a: np.ndarray,
    samples: AnalysisSampleSet,
    hierarchy: AnalysisRegionNode | None = None,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Precompute regional table metrics for every rendered waveform frame.

    The returned numerical cube is ``frame × hierarchy row × metric`` using
    the same volume-weighted definitions as :func:`region_metrics`.
    Keeping this work in the isolated render worker lets the Qt table switch
    frames instantly while playback or the timeline slider moves.
    """

    fixed = np.asarray(fixed_vectors_t, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors_t_per_a, dtype=float)
    currents = np.asarray(frame_currents_a, dtype=float)
    if fixed.shape != (samples.point_count, 3):
        raise RegionalAnalysisError(
            "Regional waveform metrics require one fixed XYZ vector per region sample."
        )
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise RegionalAnalysisError("Regional waveform metric basis dimensions are invalid.")
    if currents.ndim != 2 or currents.shape[1] != bases.shape[0]:
        raise RegionalAnalysisError("Regional waveform metric current dimensions are invalid.")

    root = _require_hierarchy(hierarchy)
    nodes = list(root.walk())
    labels = np.asarray(samples.label_ids, dtype=np.int32)
    weights = np.asarray(samples.weights_mm3, dtype=float)
    prepared: list[tuple[np.ndarray, np.ndarray, float, int, int, float]] = []
    for node in nodes:
        indices = np.flatnonzero(
            np.isin(labels, np.asarray(node.label_ids, dtype=np.int32))
        )
        node_weights = weights[indices]
        weight_total = float(np.sum(node_weights)) if len(indices) else 0.0
        unique_labels = set(int(value) for value in labels[indices])
        voxel_count = sum(samples.source_voxel_counts[value] for value in unique_labels)
        volume_cm3 = samples.volume_mm3_for_labels(unique_labels) / 1000.0
        prepared.append(
            (indices, node_weights, weight_total, int(voxel_count), int(len(indices)), float(volume_cm3))
        )

    # RMS, mean, P95, peak, uniformity, directional consistency, volume.
    # Playback timelines often revisit the exact same coil-current state (for
    # example DAC plateaus, pauses, or repeated routing steps). Regional
    # metrics are a pure function of that current vector, so evaluate each exact
    # state once and scatter the result back to every rendered frame.  This is
    # lossless: near-equal floating-point states are intentionally *not* merged.
    if len(currents):
        unique_currents, inverse = np.unique(currents, axis=0, return_inverse=True)
    else:
        unique_currents = currents
        inverse = np.empty(0, dtype=np.int64)
    unique_values = np.full(
        (len(unique_currents), len(nodes), 7), np.nan, dtype=np.float32
    )
    total_frames = len(currents)
    total_states = max(1, len(unique_currents))
    last_reported = 0
    for state_index, current in enumerate(unique_currents):
        vectors = fixed + np.einsum("c,cpk->pk", current, bases, optimize=True)
        magnitude_t = np.linalg.norm(vectors, axis=1)
        magnitude_uT = magnitude_t * 1e6
        for node_index, (indices, node_weights, weight_total, voxel_count, sample_count, volume_cm3) in enumerate(prepared):
            if sample_count <= 0 or weight_total <= 0.0:
                continue
            mags = magnitude_uT[indices]
            rms_uT = float(math.sqrt(np.sum(node_weights * mags * mags) / weight_total))
            mean_uT = float(np.sum(node_weights * mags) / weight_total)
            p5_uT, p95_uT = _weighted_percentiles(mags, node_weights, (5.0, 95.0))
            peak_uT = float(np.max(mags))
            if mean_uT > 1e-30 and math.isfinite(p5_uT) and math.isfinite(p95_uT):
                uniformity = max(
                    0.0,
                    min(100.0, 100.0 * (1.0 - (p95_uT - p5_uT) / mean_uT)),
                )
            else:
                uniformity = 100.0 if p95_uT <= 1e-30 else 0.0

            direction_magnitudes = magnitude_t[indices]
            directional = direction_magnitudes > 1e-30
            if np.any(directional):
                direction_weights = node_weights[directional]
                node_vectors = vectors[indices]
                unit_vectors = (
                    node_vectors[directional]
                    / direction_magnitudes[directional, None]
                )
                mean_direction = np.sum(
                    unit_vectors * direction_weights[:, None], axis=0
                ) / float(np.sum(direction_weights))
                directional_consistency = max(
                    0.0, min(100.0, float(np.linalg.norm(mean_direction) * 100.0))
                )
            else:
                directional_consistency = 100.0

            unique_values[state_index, node_index] = (
                rms_uT,
                mean_uT,
                p95_uT,
                peak_uT,
                uniformity,
                directional_consistency,
                volume_cm3,
            )

        if callable(progress_callback) and total_frames:
            completed = min(
                total_frames,
                max(last_reported + 1, int(round((state_index + 1) * total_frames / total_states))),
            )
            progress_callback(completed, total_frames)
            last_reported = completed

    values = unique_values[inverse] if len(inverse) else unique_values

    return {
        "keys": [node.key for node in nodes],
        "names": [node.name for node in nodes],
        "values": values,
        "sample_counts": [entry[4] for entry in prepared],
        "voxel_counts": [entry[3] for entry in prepared],
    }

def weighted_rms_coefficients(
    fixed_vectors: np.ndarray,
    basis_vectors: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the quadratic coefficients for weighted spatial RMS magnitude.

    For current vector ``c``, mean(|B|²) is ``constant + 2*c·linear +
    cᵀ*quadratic*c``. This lets animated region RMS traces be evaluated without
    allocating a time×voxel×XYZ tensor.
    """

    fixed = np.asarray(fixed_vectors, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors, dtype=float)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise RegionalAnalysisError("Regional waveform basis dimensions are invalid.")
    if mass.shape != (len(fixed),) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise RegionalAnalysisError("Regional waveform weights are invalid.")
    total = float(np.sum(mass))
    normalized = mass / total
    constant = float(np.sum(normalized * np.sum(fixed * fixed, axis=1)))
    linear = np.asarray(
        [np.sum(normalized * np.sum(fixed * basis, axis=1)) for basis in bases],
        dtype=float,
    )
    quadratic = np.einsum(
        "p,cpk,dpk->cd", normalized, bases, bases, optimize=True
    )
    return constant, linear, np.asarray(quadratic, dtype=float)


def weighted_rms_from_coefficients(
    currents_a: np.ndarray,
    coefficients: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Evaluate RMS magnitude in tesla for one or more current states."""

    current = np.asarray(currents_a, dtype=float)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    constant, linear, quadratic = coefficients
    if current.ndim != 2 or current.shape[1] != len(linear):
        raise RegionalAnalysisError("Regional waveform current dimensions are invalid.")
    mean_square = (
        float(constant)
        + 2.0 * np.einsum("tc,c->t", current, linear, optimize=True)
        + np.einsum("tc,cd,td->t", current, quadratic, current, optimize=True)
    )
    return np.sqrt(np.maximum(mean_square, 0.0))


def playback_region_metrics_streaming(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    analysis_chunk_factory,
    samples: AnalysisSampleSet,
    hierarchy: AnalysisRegionNode | None = None,
    *,
    progress_callback=None,
    analysis_sample_count: int | None = None,
) -> dict[str, RegionPlaybackMetrics]:
    """Accumulate authoritative regional metrics from bounded exposure packets.

    Factories may yield ordinary ``(times, currents)`` chunks or verified
    ``(times, currents, repeat_count)`` complete cycles. Repeated cycles have
    their B-time/B²-time/mean integrals multiplied while extrema are evaluated
    once. This preserves the full physical exposure duration without replaying
    identical steady-state cycles.
    """
    fixed = np.asarray(fixed_vectors_t, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors_t_per_a, dtype=float)
    if fixed.shape != (samples.point_count, 3):
        raise RegionalAnalysisError("Regional playback summary requires one fixed XYZ vector per region sample.")
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise RegionalAnalysisError("Regional playback summary basis dimensions are invalid.")
    root = _require_hierarchy(hierarchy)
    nodes = list(root.walk())
    if not nodes:
        return {}

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with older NumPy
        integrate = np.trapz
    b_integrals = np.zeros(len(nodes), dtype=float)
    b2_integrals = np.zeros(len(nodes), dtype=float)
    mean_integrals = np.zeros(len(nodes), dtype=float)
    max_p95 = np.full(len(nodes), -np.inf, dtype=float)
    absolute_peak = np.full(len(nodes), -np.inf, dtype=float)
    peak_times = np.full(len(nodes), np.nan, dtype=float)
    volumes = np.full(len(nodes), np.nan, dtype=float)
    sample_counts: list[int] | None = None
    voxel_counts: list[int] | None = None
    first_time: float | None = None
    last_time: float | None = None
    previous_values: np.ndarray | None = None
    processed = 0

    for packet in analysis_chunk_factory():
        if not isinstance(packet, (tuple, list)) or len(packet) not in (2, 3):
            raise RegionalAnalysisError("Streamed regional playback received an invalid analysis packet.")
        raw_times, raw_currents = packet[0], packet[1]
        repeat_count = int(packet[2]) if len(packet) == 3 else 1
        if repeat_count < 1:
            raise RegionalAnalysisError("Streamed regional playback requires a positive repeat count.")
        times = np.asarray(raw_times, dtype=float).reshape(-1)
        currents = np.asarray(raw_currents, dtype=float)
        if len(times) < 1 or currents.shape != (len(times), bases.shape[0]):
            raise RegionalAnalysisError("Streamed regional playback chunks have invalid current dimensions.")
        if not np.isfinite(times).all() or not np.isfinite(currents).all() or np.any(np.diff(times) < 0.0):
            raise RegionalAnalysisError("Streamed regional playback chunks require finite increasing samples.")
        if repeat_count > 1 and (len(times) < 2 or float(times[-1]) <= float(times[0])):
            raise RegionalAnalysisError("Repeated regional playback packets require a positive cycle span.")
        chunk = frame_region_metrics(fixed, bases, currents, samples, root)
        values = np.asarray(chunk["values"], dtype=float)
        if values.shape != (len(times), len(nodes), 7) or not np.isfinite(values).all():
            raise RegionalAnalysisError("Streamed regional playback regional metrics are incomplete.")
        if sample_counts is None:
            sample_counts = [int(value) for value in chunk["sample_counts"]]
            voxel_counts = [int(value) for value in chunk["voxel_counts"]]
            volumes[:] = values[0, :, 6]
        if first_time is None:
            first_time = float(times[0])

        if previous_values is not None and last_time is not None:
            if float(times[0]) < last_time - max(1e-12, abs(last_time) * 1e-12):
                raise RegionalAnalysisError("Streamed regional playback chunks moved backwards in time.")
            join_dt = float(times[0]) - last_time
            if join_dt > 0.0:
                b_integrals += (previous_values[:, 0] + values[0, :, 0]) * 0.5 * join_dt
                b2_integrals += (previous_values[:, 0] ** 2 + values[0, :, 0] ** 2) * 0.5 * join_dt
                mean_integrals += (previous_values[:, 1] + values[0, :, 1]) * 0.5 * join_dt

        if len(times) >= 2 and float(times[-1]) > float(times[0]):
            cycle_b = np.asarray(integrate(values[:, :, 0], times, axis=0), dtype=float)
            cycle_b2 = np.asarray(integrate(values[:, :, 0] ** 2, times, axis=0), dtype=float)
            cycle_mean = np.asarray(integrate(values[:, :, 1], times, axis=0), dtype=float)
            b_integrals += cycle_b * repeat_count
            b2_integrals += cycle_b2 * repeat_count
            mean_integrals += cycle_mean * repeat_count

        chunk_p95 = np.max(values[:, :, 2], axis=0)
        max_p95 = np.maximum(max_p95, chunk_p95)
        for node_index in range(len(nodes)):
            local_index = int(np.argmax(values[:, node_index, 3]))
            candidate = float(values[local_index, node_index, 3])
            if candidate > absolute_peak[node_index]:
                absolute_peak[node_index] = candidate
                peak_times[node_index] = float(times[local_index])
        previous_values = values[-1]
        if repeat_count > 1:
            last_time = float(times[0]) + repeat_count * float(times[-1] - times[0])
        else:
            last_time = float(times[-1])
        processed += len(times) * repeat_count
        if callable(progress_callback):
            total = int(analysis_sample_count or max(processed, 1))
            progress_callback(min(processed, total), total)

    if callable(progress_callback) and processed:
        total = int(analysis_sample_count or processed)
        progress_callback(total, total)
    if first_time is None or last_time is None or previous_values is None:
        raise RegionalAnalysisError("Streamed regional playback did not produce any analysis samples.")
    duration = float(last_time - first_time)
    if duration <= 0.0:
        raise RegionalAnalysisError("Regional playback summary requires positive exposure time.")
    assert sample_counts is not None and voxel_counts is not None
    result: dict[str, RegionPlaybackMetrics] = {}
    for index, node in enumerate(nodes):
        result[node.key] = RegionPlaybackMetrics(
            playback_rms_uT=float(math.sqrt(max(0.0, b2_integrals[index] / duration))),
            mean_uT=float(mean_integrals[index] / duration),
            max_p95_uT=float(max_p95[index]),
            absolute_peak_uT=float(absolute_peak[index]),
            b_time_uT_s=float(b_integrals[index]),
            b2_time_uT2_s=float(b2_integrals[index]),
            volume_cm3=float(volumes[index]),
            peak_time_s=float(peak_times[index]),
            sample_count=sample_counts[index],
            voxel_count=voxel_counts[index],
        )
    return result


def playback_region_metrics(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    analysis_currents_a: np.ndarray,
    analysis_times_s: np.ndarray,
    frame_metrics: dict[str, object],
    frame_times_s: np.ndarray,
    samples: AnalysisSampleSet,
    hierarchy: AnalysisRegionNode | None = None,
    *,
    progress_callback=None,
    full_timeline: bool = False,
) -> dict[str, RegionPlaybackMetrics]:
    """Summarize every regional hierarchy row over one physical playback pass.

    The cumulative spatial-RMS quantities use the higher-density analysis
    timeline and the exact quadratic field-basis form.  Mean magnitude, spatial
    P95, and retained-sample peak reuse the already-calculated regional frame
    cache, time-weighted using its physical source times.  Consequently changing
    preview FPS or exported-video slow motion does not change exposure duration.
    """

    fixed = np.asarray(fixed_vectors_t, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors_t_per_a, dtype=float)
    currents = np.asarray(analysis_currents_a, dtype=float)
    analysis_times = np.asarray(analysis_times_s, dtype=float).reshape(-1)
    if full_timeline:
        # Stream regional statistics across the complete analysis timeline.
        # Adjacent chunks share one boundary sample for trapezoidal integration.
        from dataclasses import replace
        result = {}
        total_time = float(analysis_times[-1] - analysis_times[0])
        mean_integrals = {}
        for start in range(0, len(analysis_times) - 1, 255):
            stop = min(start + 256, len(analysis_times))
            chunk_times = analysis_times[start:stop]
            chunk_currents = currents[start:stop]
            chunk_frames = frame_region_metrics(fixed, bases, chunk_currents, samples, hierarchy)
            rows = playback_region_metrics(fixed, bases, chunk_currents, chunk_times,
                                                chunk_frames, chunk_times, samples, hierarchy)
            dt = float(chunk_times[-1] - chunk_times[0])
            for key, row in rows.items():
                mean_integrals[key] = mean_integrals.get(key, 0.0) + row.mean_uT * dt
                old = result.get(key)
                if old is None:
                    result[key] = row
                else:
                    result[key] = replace(old,
                        max_p95_uT=max(old.max_p95_uT, row.max_p95_uT),
                        absolute_peak_uT=max(old.absolute_peak_uT, row.absolute_peak_uT),
                        peak_time_s=row.peak_time_s if row.absolute_peak_uT > old.absolute_peak_uT else old.peak_time_s,
                        b_time_uT_s=old.b_time_uT_s + row.b_time_uT_s,
                        b2_time_uT2_s=old.b2_time_uT2_s + row.b2_time_uT2_s)
            if callable(progress_callback):
                progress_callback(stop, len(analysis_times))
        return {key: replace(row, mean_uT=mean_integrals[key] / total_time,
                             playback_rms_uT=math.sqrt(max(0.0, row.b2_time_uT2_s / total_time)))
                for key, row in result.items()}
    frame_times = np.asarray(frame_times_s, dtype=float).reshape(-1)
    frame_values = np.asarray(frame_metrics.get("values"), dtype=float)
    if fixed.shape != (samples.point_count, 3):
        raise RegionalAnalysisError(
            "Regional playback summary requires one fixed XYZ vector per region sample."
        )
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise RegionalAnalysisError("Regional playback summary basis dimensions are invalid.")
    if currents.ndim != 2 or currents.shape != (len(analysis_times), bases.shape[0]):
        raise RegionalAnalysisError("Regional playback summary current dimensions are invalid.")
    if len(analysis_times) < 2 or not np.isfinite(analysis_times).all():
        raise RegionalAnalysisError("Regional playback summary requires finite analysis times.")
    if np.any(np.diff(analysis_times) <= 0.0):
        raise RegionalAnalysisError("Regional playback summary analysis times must increase.")
    duration = float(analysis_times[-1] - analysis_times[0])
    if duration <= 0.0:
        raise RegionalAnalysisError("Regional playback summary requires positive exposure time.")

    root = _require_hierarchy(hierarchy)
    nodes = list(root.walk())
    frame_keys = [str(value) for value in frame_metrics.get("keys", [])]
    if frame_keys != [node.key for node in nodes]:
        raise RegionalAnalysisError(
            "Regional playback frame metrics do not match the regional hierarchy."
        )
    if frame_values.shape != (len(frame_times), len(nodes), 7):
        raise RegionalAnalysisError("Regional playback frame metrics have an invalid shape.")
    if len(frame_times) < 2 or not np.isfinite(frame_times).all():
        raise RegionalAnalysisError("Regional playback summary requires finite frame times.")
    if np.any(np.diff(frame_times) <= 0.0):
        raise RegionalAnalysisError("Regional playback summary frame times must increase.")
    frame_duration = float(frame_times[-1] - frame_times[0])
    if frame_duration <= 0.0 or not np.isfinite(frame_values).all():
        raise RegionalAnalysisError("Regional playback frame statistics are incomplete.")

    sample_counts = [int(value) for value in frame_metrics.get("sample_counts", [])]
    voxel_counts = [int(value) for value in frame_metrics.get("voxel_counts", [])]
    if len(sample_counts) != len(nodes) or len(voxel_counts) != len(nodes):
        raise RegionalAnalysisError("Regional playback frame sample counts are incomplete.")

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with older NumPy
        integrate = np.trapz
    labels = np.asarray(samples.label_ids, dtype=np.int32)
    result: dict[str, RegionPlaybackMetrics] = {}
    for node_index, node in enumerate(nodes):
        indices = np.flatnonzero(
            np.isin(labels, np.asarray(node.label_ids, dtype=np.int32))
        )
        if not len(indices):
            raise RegionalAnalysisError(
                f"Regional playback summary region {node.name!r} has no retained samples."
            )
        coefficients = weighted_rms_coefficients(
            fixed[indices], bases[:, indices, :], samples.weights_mm3[indices]
        )
        spatial_rms_uT = (
            weighted_rms_from_coefficients(currents, coefficients) * 1e6
        )
        b_time = float(integrate(spatial_rms_uT, analysis_times))
        b2_time = float(integrate(spatial_rms_uT * spatial_rms_uT, analysis_times))

        frame = frame_values[:, node_index, :]
        time_mean = float(integrate(frame[:, 1], frame_times) / frame_duration)
        peak_index = int(np.argmax(frame[:, 3]))
        result[node.key] = RegionPlaybackMetrics(
            playback_rms_uT=float(math.sqrt(max(0.0, b2_time / duration))),
            mean_uT=time_mean,
            max_p95_uT=float(np.max(frame[:, 2])),
            absolute_peak_uT=float(frame[peak_index, 3]),
            b_time_uT_s=b_time,
            b2_time_uT2_s=b2_time,
            volume_cm3=float(frame[0, 6]),
            peak_time_s=float(frame_times[peak_index]),
            sample_count=sample_counts[node_index],
            voxel_count=voxel_counts[node_index],
        )
        if callable(progress_callback):
            progress_callback(node_index + 1, len(nodes))
    return result
