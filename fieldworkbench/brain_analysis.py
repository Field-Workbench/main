"""LPBA40 brain-region sampling, hierarchy, and field statistics.

SRI24's LPBA40 lookup table is deliberately flat.  Field Workbench adds a
small, documented presentation hierarchy so the viewer can browse meaningful
bilateral structures without claiming that the grouping is part of the
upstream atlas.  All numerical work remains tied to the original label ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .anatomy_frames import sri24_ras_mm_to_head_m
from .sri24_atlas import SRI24Parcellation, load_sri24_parcellation


class BrainAnalysisError(ValueError):
    """Raised when atlas placement, sampling, or statistics are invalid."""


@dataclass(frozen=True)
class BrainRegionNode:
    """One row in the application-authored LPBA40 presentation hierarchy."""

    key: str
    name: str
    label_ids: tuple[int, ...]
    children: tuple["BrainRegionNode", ...] = ()

    def walk(self) -> Iterator["BrainRegionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class BrainSampleSet:
    """Deterministic, volume-weighted atlas voxel-centre samples."""

    points_m: np.ndarray
    label_ids: np.ndarray
    weights_mm3: np.ndarray
    source_indices: np.ndarray
    source_voxel_counts: dict[int, int]
    sampled_counts: dict[int, int]
    voxel_volume_mm3: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=float)
        labels = np.asarray(self.label_ids, dtype=np.int32)
        weights = np.asarray(self.weights_mm3, dtype=float)
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        source_counts = {
            int(label): int(count)
            for label, count in self.source_voxel_counts.items()
        }
        sampled_counts = {
            int(label): int(count) for label, count in self.sampled_counts.items()
        }
        voxel_volume = float(self.voxel_volume_mm3)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise BrainAnalysisError("Brain sample points must be an N×3 array.")
        if (
            labels.shape != (len(points),)
            or weights.shape != (len(points),)
            or source_indices.shape != (len(points),)
        ):
            raise BrainAnalysisError("Brain sample labels and weights must match the points.")
        if np.any(source_indices < 0):
            raise BrainAnalysisError("Brain sample source indices cannot be negative.")
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise BrainAnalysisError("Brain samples contain non-finite coordinates or weights.")
        if np.any(weights <= 0.0):
            raise BrainAnalysisError("Brain sample weights must be positive.")
        unique_labels, actual_count_values = np.unique(labels, return_counts=True)
        actual_counts = {
            int(label): int(count)
            for label, count in zip(
                unique_labels, actual_count_values, strict=True
            )
        }
        present = set(actual_counts)
        if present != set(source_counts) or present != set(sampled_counts):
            raise BrainAnalysisError(
                "Brain sample count dictionaries must match the retained labels."
            )
        if any(value <= 0 for value in source_counts.values()) or any(
            value <= 0 for value in sampled_counts.values()
        ):
            raise BrainAnalysisError("Brain sample counts must be positive.")
        if any(sampled_counts[label] != actual_counts[label] for label in present):
            raise BrainAnalysisError("Brain sampled counts do not match the label array.")
        if not math.isfinite(voxel_volume) or voxel_volume <= 0.0:
            raise BrainAnalysisError("Brain atlas voxel volume must be positive.")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "label_ids", labels)
        object.__setattr__(self, "weights_mm3", weights)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "source_voxel_counts", source_counts)
        object.__setattr__(self, "sampled_counts", sampled_counts)
        object.__setattr__(self, "voxel_volume_mm3", voxel_volume)

    @property
    def point_count(self) -> int:
        return int(len(self.points_m))

    @property
    def points_mm(self) -> np.ndarray:
        return np.asarray(self.points_m, dtype=float) * 1000.0


@dataclass(frozen=True)
class BrainRegionMetrics:
    """Magnitude and directional statistics over volume-weighted atlas samples."""

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
class BrainRegionPlaybackMetrics:
    """Duration-weighted regional statistics over one physical playback pass.

    ``playback_rms_uT``, ``b_time_uT_s``, and ``b2_time_uT2_s`` integrate the
    instantaneous volume-weighted spatial RMS magnitude.  The remaining field
    statistics summarize the existing rendered-frame regional cache.  These
    quantities are comparative magnetic-field exposure indices, not a
    biological dose model.
    """

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


# This grouping is a Field Workbench browsing aid, not an upstream LPBA40
# taxonomy.  Every LPBA40 non-background label appears exactly once.
_LPBA40_DIVISIONS: tuple[
    tuple[str, str, tuple[str, ...]], ...
] = (
    (
        "frontal",
        "Frontal lobe",
        (
            "superior_frontal_gyrus",
            "middle_frontal_gyrus",
            "inferior_frontal_gyrus",
            "precentral_gyrus",
            "middle_orbitofrontal_gyrus",
            "lateral_orbitofrontal_gyrus",
            "gyrus_rectus",
        ),
    ),
    (
        "parietal",
        "Parietal lobe",
        (
            "postcentral_gyrus",
            "superior_parietal_gyrus",
            "supramarginal_gyrus",
            "angular_gyrus",
            "precuneus",
        ),
    ),
    (
        "occipital",
        "Occipital lobe",
        (
            "superior_occipital_gyrus",
            "middle_occipital_gyrus",
            "inferior_occipital_gyrus",
            "cuneus",
            "lingual_gyrus",
        ),
    ),
    (
        "temporal",
        "Temporal lobe",
        (
            "superior_temporal_gyrus",
            "middle_temporal_gyrus",
            "inferior_temporal_gyrus",
            "parahippocampal_gyrus",
            "fusiform_gyrus",
        ),
    ),
    ("insula", "Insula", ("insular_cortex",)),
    ("limbic", "Limbic structures", ("cingulate_gyrus", "hippocampus")),
    ("deep_gray", "Deep gray matter", ("caudate", "putamen")),
    ("hindbrain", "Cerebellum and brainstem", ("cerebellum", "brainstem")),
)


def _display_region_name(stem: str) -> str:
    return str(stem).replace("_", " ").capitalize()


def lpba40_region_hierarchy(
    parcellation: SRI24Parcellation | None = None,
) -> BrainRegionNode:
    """Return the complete Workbench browsing tree for LPBA40.

    Bilateral anatomical rows contain Left and Right leaves. Cerebellum and
    brainstem are unpaired in the upstream atlas and therefore remain leaves.
    """

    atlas = parcellation or load_sri24_parcellation("lpba40")
    if atlas.key != "lpba40":
        raise BrainAnalysisError("The LPBA40 hierarchy requires the LPBA40 parcellation.")
    values_by_name = {
        definition.name: int(value)
        for value, definition in atlas.labels.items()
        if int(value) in atlas.present_label_ids and int(value) != 0
    }
    assigned: list[int] = []
    divisions: list[BrainRegionNode] = []
    for division_key, division_name, stems in _LPBA40_DIVISIONS:
        regions: list[BrainRegionNode] = []
        for stem in stems:
            left = values_by_name.get(f"L_{stem}")
            right = values_by_name.get(f"R_{stem}")
            if left is not None or right is not None:
                if left is None or right is None:
                    raise BrainAnalysisError(
                        f"LPBA40 has an incomplete bilateral definition for {stem}."
                    )
                label_ids = (left, right)
                children = (
                    BrainRegionNode(f"{stem}.left", "Left", (left,)),
                    BrainRegionNode(f"{stem}.right", "Right", (right,)),
                )
            else:
                value = values_by_name.get(stem)
                if value is None:
                    raise BrainAnalysisError(
                        f"LPBA40 does not contain the expected structure {stem}."
                    )
                label_ids = (value,)
                children = ()
            assigned.extend(label_ids)
            regions.append(
                BrainRegionNode(
                    key=stem,
                    name=_display_region_name(stem),
                    label_ids=tuple(label_ids),
                    children=children,
                )
            )
        division_ids = tuple(
            value for region in regions for value in region.label_ids
        )
        divisions.append(
            BrainRegionNode(
                key=division_key,
                name=division_name,
                label_ids=division_ids,
                children=tuple(regions),
            )
        )

    expected = sorted(
        int(value) for value in atlas.present_label_ids if int(value) != 0
    )
    if sorted(assigned) != expected or len(set(assigned)) != len(assigned):
        missing = sorted(set(expected) - set(assigned))
        repeated = sorted(
            value for value in set(assigned) if assigned.count(value) > 1
        )
        raise BrainAnalysisError(
            "The Workbench LPBA40 hierarchy is not an exact label partition"
            f" (missing={missing}, repeated={repeated})."
        )
    return BrainRegionNode(
        key="whole_brain",
        name="Whole brain",
        label_ids=tuple(expected),
        children=tuple(divisions),
    )


def _validate_placement(
    position_m: Iterable[float],
    rotation: np.ndarray,
    scale_xyz: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(position_m, dtype=float)
    orientation = np.asarray(rotation, dtype=float)
    scale = np.asarray(scale_xyz, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise BrainAnalysisError("Brain placement requires a finite XYZ position.")
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        raise BrainAnalysisError("Brain placement requires a finite 3×3 rotation.")
    if not np.allclose(orientation @ orientation.T, np.eye(3), atol=1e-6):
        raise BrainAnalysisError("Brain placement rotation is not orthonormal.")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise BrainAnalysisError("Brain placement scale must contain three positive values.")
    return position, orientation, scale


def sample_lpba40_voxels(
    parcellation: SRI24Parcellation | None = None,
    *,
    position_m: Iterable[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    scale_xyz: Iterable[float] = (1.0, 1.0, 1.0),
    maximum_samples_per_label: int = 1200,
) -> BrainSampleSet:
    """Sample every LPBA40 label deterministically in the placed brain frame.

    Each retained point represents an equal share of its source label's voxels.
    Group statistics can therefore combine labels without bias from the per-label
    cap. Atlas voxel centres are mapped SRI24 RAS mm -> canonical head metres ->
    the selected brain object's scale/rotation/translation.
    """

    atlas = parcellation or load_sri24_parcellation("lpba40")
    if atlas.key != "lpba40":
        raise BrainAnalysisError("Brain sampling currently supports LPBA40 only.")
    cap = int(maximum_samples_per_label)
    if cap < 1:
        raise BrainAnalysisError("At least one sample per LPBA40 label is required.")
    position, orientation, scale = _validate_placement(
        position_m, np.eye(3) if rotation is None else rotation, scale_xyz
    )

    flat = np.asarray(atlas.volume).reshape(-1)
    nonzero_indices = np.flatnonzero(flat)
    nonzero_labels = np.asarray(flat[nonzero_indices], dtype=np.int32)
    order = np.argsort(nonzero_labels, kind="stable")
    grouped_indices = nonzero_indices[order]
    grouped_labels = nonzero_labels[order]
    unique_labels, starts, counts = np.unique(
        grouped_labels, return_index=True, return_counts=True
    )

    sample_flat_indices: list[np.ndarray] = []
    sample_label_ids: list[np.ndarray] = []
    source_counts: dict[int, int] = {}
    sampled_counts: dict[int, int] = {}
    for raw_label, raw_start, raw_count in zip(
        unique_labels, starts, counts, strict=True
    ):
        label_id = int(raw_label)
        count = int(raw_count)
        start = int(raw_start)
        retained = min(count, cap)
        if retained == count:
            selected = grouped_indices[start : start + count]
        else:
            offsets = np.floor(
                (np.arange(retained, dtype=float) + 0.5) * count / retained
            ).astype(np.int64)
            selected = grouped_indices[start + offsets]
        sample_flat_indices.append(np.asarray(selected, dtype=np.int64))
        sample_label_ids.append(np.full(retained, label_id, dtype=np.int32))
        source_counts[label_id] = count
        sampled_counts[label_id] = retained

    selected_flat = np.concatenate(sample_flat_indices)
    labels = np.concatenate(sample_label_ids)
    voxel_indices = np.column_stack(
        np.unravel_index(selected_flat, atlas.volume.shape, order="C")
    ).astype(float)
    atlas_ras_mm = (
        voxel_indices @ np.asarray(atlas.affine_ras_mm[:3, :3], dtype=float).T
        + np.asarray(atlas.affine_ras_mm[:3, 3], dtype=float)
    )
    canonical_head_m = sri24_ras_mm_to_head_m(atlas_ras_mm)
    points_m = (canonical_head_m * scale) @ orientation.T + position

    source_voxel_volume_mm3 = abs(
        float(np.linalg.det(np.asarray(atlas.affine_ras_mm[:3, :3], dtype=float)))
    )
    placed_voxel_volume_mm3 = source_voxel_volume_mm3 * abs(float(np.prod(scale)))
    weights = np.asarray(
        [
            source_counts[int(label)]
            * placed_voxel_volume_mm3
            / sampled_counts[int(label)]
            for label in labels
        ],
        dtype=float,
    )
    return BrainSampleSet(
        points_m=np.asarray(points_m, dtype=float),
        label_ids=labels,
        weights_mm3=weights,
        source_indices=np.arange(len(labels), dtype=np.int64),
        source_voxel_counts=source_counts,
        sampled_counts=sampled_counts,
        voxel_volume_mm3=placed_voxel_volume_mm3,
    )


def downsample_brain_samples(
    samples: BrainSampleSet,
    *,
    maximum_points: int = 14000,
) -> BrainSampleSet:
    """Return a smaller deterministic set for interactive 3D rendering."""

    limit = int(maximum_points)
    labels = sorted(int(value) for value in samples.sampled_counts)
    if limit < len(labels):
        raise BrainAnalysisError(
            "The display sample limit must retain at least one point per atlas label."
        )
    if samples.point_count <= limit:
        return samples

    # Allocate the budget proportionally to source volume while guaranteeing
    # every structure remains selectable in the viewer.
    source_total = max(1, sum(samples.source_voxel_counts.values()))
    allocations = {
        label: max(
            1,
            min(
                samples.sampled_counts[label],
                int(math.floor(limit * samples.source_voxel_counts[label] / source_total)),
            ),
        )
        for label in labels
    }
    remaining = limit - sum(allocations.values())
    while remaining < 0:
        candidates = [label for label in labels if allocations[label] > 1]
        if not candidates:
            break
        candidates.sort(
            key=lambda label: (
                samples.source_voxel_counts[label] / allocations[label],
                label,
            )
        )
        for label in candidates:
            if remaining >= 0:
                break
            allocations[label] -= 1
            remaining += 1
    while remaining > 0:
        candidates = [
            label
            for label in labels
            if allocations[label] < samples.sampled_counts[label]
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda label: (
                samples.source_voxel_counts[label] / (allocations[label] + 1),
                -label,
            ),
            reverse=True,
        )
        for label in candidates:
            if remaining <= 0:
                break
            allocations[label] += 1
            remaining -= 1

    chosen_parts: list[np.ndarray] = []
    for label in labels:
        available = np.flatnonzero(samples.label_ids == label)
        retained = allocations[label]
        if retained >= len(available):
            chosen = available
        else:
            offsets = np.floor(
                (np.arange(retained, dtype=float) + 0.5) * len(available) / retained
            ).astype(np.int64)
            chosen = available[offsets]
        chosen_parts.append(chosen)
    chosen_indices = np.sort(np.concatenate(chosen_parts))
    chosen_labels = np.asarray(samples.label_ids[chosen_indices], dtype=np.int32)
    chosen_counts = {
        label: int(np.count_nonzero(chosen_labels == label)) for label in labels
    }
    weights = np.asarray(
        [
            samples.source_voxel_counts[int(label)]
            * samples.voxel_volume_mm3
            / chosen_counts[int(label)]
            for label in chosen_labels
        ],
        dtype=float,
    )
    return BrainSampleSet(
        points_m=np.asarray(samples.points_m[chosen_indices], dtype=float),
        label_ids=chosen_labels,
        weights_mm3=weights,
        source_indices=np.asarray(samples.source_indices[chosen_indices], dtype=np.int64),
        source_voxel_counts=dict(samples.source_voxel_counts),
        sampled_counts=chosen_counts,
        voxel_volume_mm3=float(samples.voxel_volume_mm3),
    )


def _weighted_percentiles(
    values: np.ndarray,
    weights: np.ndarray,
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    """Return deterministic weighted nearest-rank percentiles with one sort."""

    data = np.asarray(values, dtype=float).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if data.shape != mass.shape:
        raise BrainAnalysisError("Weighted percentile arrays must have equal lengths.")
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
    samples: BrainSampleSet,
    mask: np.ndarray,
) -> BrainRegionMetrics:
    vectors = np.asarray(vectors_t, dtype=float)[mask]
    weights = np.asarray(samples.weights_mm3, dtype=float)[mask]
    labels = np.asarray(samples.label_ids, dtype=np.int32)[mask]
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    vectors = vectors[valid]
    weights = weights[valid]
    labels = labels[valid]
    if not len(vectors):
        return BrainRegionMetrics(
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
        voxel_count * float(samples.voxel_volume_mm3) / 1000.0
    )
    return BrainRegionMetrics(
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


def lpba40_region_metrics(
    vectors_t: np.ndarray,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
) -> dict[str, BrainRegionMetrics]:
    """Calculate metrics for every node in the LPBA40 browsing hierarchy."""

    vectors = np.asarray(vectors_t, dtype=float)
    if vectors.shape != (samples.point_count, 3):
        raise BrainAnalysisError(
            "Brain field vectors must contain one XYZ vector per atlas sample."
        )
    root = hierarchy or lpba40_region_hierarchy()
    result: dict[str, BrainRegionMetrics] = {}
    for node in root.walk():
        mask = np.isin(samples.label_ids, np.asarray(node.label_ids, dtype=np.int32))
        result[node.key] = _metrics_for_mask(vectors, samples, mask)
    return result



def lpba40_frame_region_metrics(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    frame_currents_a: np.ndarray,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Precompute Brain Areas table metrics for every rendered waveform frame.

    The returned numerical cube is ``frame × hierarchy row × metric`` using
    the same volume-weighted definitions as :func:`lpba40_region_metrics`.
    Keeping this work in the isolated render worker lets the Qt table switch
    frames instantly while playback or the timeline slider moves.
    """

    fixed = np.asarray(fixed_vectors_t, dtype=float).reshape(-1, 3)
    bases = np.asarray(basis_vectors_t_per_a, dtype=float)
    currents = np.asarray(frame_currents_a, dtype=float)
    if fixed.shape != (samples.point_count, 3):
        raise BrainAnalysisError(
            "Brain waveform metrics require one fixed XYZ vector per atlas sample."
        )
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise BrainAnalysisError("Brain waveform metric basis dimensions are invalid.")
    if currents.ndim != 2 or currents.shape[1] != bases.shape[0]:
        raise BrainAnalysisError("Brain waveform metric current dimensions are invalid.")

    root = hierarchy or lpba40_region_hierarchy()
    nodes = list(root.walk())
    labels = np.asarray(samples.label_ids, dtype=np.int32)
    weights = np.asarray(samples.weights_mm3, dtype=float)
    prepared: list[tuple[np.ndarray, np.ndarray, float, int, int]] = []
    for node in nodes:
        indices = np.flatnonzero(
            np.isin(labels, np.asarray(node.label_ids, dtype=np.int32))
        )
        node_weights = weights[indices]
        weight_total = float(np.sum(node_weights)) if len(indices) else 0.0
        unique_labels = set(int(value) for value in labels[indices])
        voxel_count = sum(samples.source_voxel_counts[value] for value in unique_labels)
        volume_cm3 = voxel_count * float(samples.voxel_volume_mm3) / 1000.0
        prepared.append(
            (indices, node_weights, weight_total, int(voxel_count), int(len(indices)))
        )

    # RMS, mean, P95, peak, uniformity, directional consistency, volume.
    # Playback timelines often revisit the exact same coil-current state (for
    # example DAC plateaus, pauses, or repeated routing steps).  Brain-region
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
        for node_index, (indices, node_weights, weight_total, voxel_count, sample_count) in enumerate(prepared):
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

            volume_cm3 = voxel_count * float(samples.voxel_volume_mm3) / 1000.0
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
        raise BrainAnalysisError("Brain waveform basis dimensions are invalid.")
    if mass.shape != (len(fixed),) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise BrainAnalysisError("Brain waveform weights are invalid.")
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
        raise BrainAnalysisError("Brain waveform current dimensions are invalid.")
    mean_square = (
        float(constant)
        + 2.0 * np.einsum("tc,c->t", current, linear, optimize=True)
        + np.einsum("tc,cd,td->t", current, quadratic, current, optimize=True)
    )
    return np.sqrt(np.maximum(mean_square, 0.0))


def lpba40_playback_region_metrics_streaming(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    analysis_chunk_factory,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
    *,
    progress_callback=None,
    analysis_sample_count: int | None = None,
) -> dict[str, BrainRegionPlaybackMetrics]:
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
        raise BrainAnalysisError("Brain playback summary requires one fixed XYZ vector per atlas sample.")
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise BrainAnalysisError("Brain playback summary basis dimensions are invalid.")
    root = hierarchy or lpba40_region_hierarchy()
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
            raise BrainAnalysisError("Streamed brain playback received an invalid analysis packet.")
        raw_times, raw_currents = packet[0], packet[1]
        repeat_count = int(packet[2]) if len(packet) == 3 else 1
        if repeat_count < 1:
            raise BrainAnalysisError("Streamed brain playback requires a positive repeat count.")
        times = np.asarray(raw_times, dtype=float).reshape(-1)
        currents = np.asarray(raw_currents, dtype=float)
        if len(times) < 1 or currents.shape != (len(times), bases.shape[0]):
            raise BrainAnalysisError("Streamed brain playback chunks have invalid current dimensions.")
        if not np.isfinite(times).all() or not np.isfinite(currents).all() or np.any(np.diff(times) < 0.0):
            raise BrainAnalysisError("Streamed brain playback chunks require finite increasing samples.")
        if repeat_count > 1 and (len(times) < 2 or float(times[-1]) <= float(times[0])):
            raise BrainAnalysisError("Repeated brain playback packets require a positive cycle span.")
        chunk = lpba40_frame_region_metrics(fixed, bases, currents, samples, root)
        values = np.asarray(chunk["values"], dtype=float)
        if values.shape != (len(times), len(nodes), 7) or not np.isfinite(values).all():
            raise BrainAnalysisError("Streamed brain playback regional metrics are incomplete.")
        if sample_counts is None:
            sample_counts = [int(value) for value in chunk["sample_counts"]]
            voxel_counts = [int(value) for value in chunk["voxel_counts"]]
            volumes[:] = values[0, :, 6]
        if first_time is None:
            first_time = float(times[0])

        if previous_values is not None and last_time is not None:
            if float(times[0]) < last_time - max(1e-12, abs(last_time) * 1e-12):
                raise BrainAnalysisError("Streamed brain playback chunks moved backwards in time.")
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
        raise BrainAnalysisError("Streamed brain playback did not produce any analysis samples.")
    duration = float(last_time - first_time)
    if duration <= 0.0:
        raise BrainAnalysisError("Brain playback summary requires positive exposure time.")
    assert sample_counts is not None and voxel_counts is not None
    result: dict[str, BrainRegionPlaybackMetrics] = {}
    for index, node in enumerate(nodes):
        result[node.key] = BrainRegionPlaybackMetrics(
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


def lpba40_playback_region_metrics(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    analysis_currents_a: np.ndarray,
    analysis_times_s: np.ndarray,
    frame_metrics: dict[str, object],
    frame_times_s: np.ndarray,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
    *,
    progress_callback=None,
    full_timeline: bool = False,
) -> dict[str, BrainRegionPlaybackMetrics]:
    """Summarize every LPBA40 hierarchy row over one physical playback pass.

    The cumulative spatial-RMS quantities use the higher-density analysis
    timeline and the exact quadratic field-basis form.  Mean magnitude, spatial
    P95, and retained-sample peak reuse the already-calculated Brain Areas frame
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
            chunk_frames = lpba40_frame_region_metrics(fixed, bases, chunk_currents, samples, hierarchy)
            rows = lpba40_playback_region_metrics(fixed, bases, chunk_currents, chunk_times,
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
        raise BrainAnalysisError(
            "Brain playback summary requires one fixed XYZ vector per atlas sample."
        )
    if bases.ndim != 3 or bases.shape[1:] != fixed.shape:
        raise BrainAnalysisError("Brain playback summary basis dimensions are invalid.")
    if currents.ndim != 2 or currents.shape != (len(analysis_times), bases.shape[0]):
        raise BrainAnalysisError("Brain playback summary current dimensions are invalid.")
    if len(analysis_times) < 2 or not np.isfinite(analysis_times).all():
        raise BrainAnalysisError("Brain playback summary requires finite analysis times.")
    if np.any(np.diff(analysis_times) <= 0.0):
        raise BrainAnalysisError("Brain playback summary analysis times must increase.")
    duration = float(analysis_times[-1] - analysis_times[0])
    if duration <= 0.0:
        raise BrainAnalysisError("Brain playback summary requires positive exposure time.")

    root = hierarchy or lpba40_region_hierarchy()
    nodes = list(root.walk())
    frame_keys = [str(value) for value in frame_metrics.get("keys", [])]
    if frame_keys != [node.key for node in nodes]:
        raise BrainAnalysisError(
            "Brain playback frame metrics do not match the LPBA40 hierarchy."
        )
    if frame_values.shape != (len(frame_times), len(nodes), 7):
        raise BrainAnalysisError("Brain playback frame metrics have an invalid shape.")
    if len(frame_times) < 2 or not np.isfinite(frame_times).all():
        raise BrainAnalysisError("Brain playback summary requires finite frame times.")
    if np.any(np.diff(frame_times) <= 0.0):
        raise BrainAnalysisError("Brain playback summary frame times must increase.")
    frame_duration = float(frame_times[-1] - frame_times[0])
    if frame_duration <= 0.0 or not np.isfinite(frame_values).all():
        raise BrainAnalysisError("Brain playback frame statistics are incomplete.")

    sample_counts = [int(value) for value in frame_metrics.get("sample_counts", [])]
    voxel_counts = [int(value) for value in frame_metrics.get("voxel_counts", [])]
    if len(sample_counts) != len(nodes) or len(voxel_counts) != len(nodes):
        raise BrainAnalysisError("Brain playback frame sample counts are incomplete.")

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with older NumPy
        integrate = np.trapz
    labels = np.asarray(samples.label_ids, dtype=np.int32)
    result: dict[str, BrainRegionPlaybackMetrics] = {}
    for node_index, node in enumerate(nodes):
        indices = np.flatnonzero(
            np.isin(labels, np.asarray(node.label_ids, dtype=np.int32))
        )
        if not len(indices):
            raise BrainAnalysisError(
                f"Brain playback summary region {node.name!r} has no retained samples."
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
        result[node.key] = BrainRegionPlaybackMetrics(
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
