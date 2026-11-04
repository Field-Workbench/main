"""LPBA40 brain-region sampling, hierarchy, and field statistics.

SRI24's LPBA40 lookup table is deliberately flat.  Field Workbench adds a
small, documented presentation hierarchy so the viewer can browse meaningful
bilateral structures without claiming that the grouping is part of the
upstream atlas.  All numerical work remains tied to the original label ids.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .anatomy_frames import sri24_ras_mm_to_head_m
from .sri24_atlas import SRI24Parcellation, load_sri24_parcellation


# Brain-specific code supplies atlas hierarchy/sampling; all numerical region
# structures and statistics live in the source-agnostic spatial engine.  Keep
# the historical Brain* names as compatibility aliases for Brain View callers.
from .spatial_analysis import (
    AnalysisRegionNode, AnalysisSampleSet, RegionMetrics, RegionPlaybackMetrics,
    RegionalAnalysisError, frame_region_metrics as _frame_region_metrics,
    playback_region_metrics as _playback_region_metrics,
    playback_region_metrics_streaming as _playback_region_metrics_streaming,
    region_metrics as _region_metrics, weighted_percentile,
    weighted_rms_coefficients as _weighted_rms_coefficients,
    weighted_rms_from_coefficients as _weighted_rms_from_coefficients,
)

BrainAnalysisError = RegionalAnalysisError
BrainRegionNode = AnalysisRegionNode
BrainSampleSet = AnalysisSampleSet
BrainRegionMetrics = RegionMetrics
BrainRegionPlaybackMetrics = RegionPlaybackMetrics


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


def lpba40_region_metrics(
    vectors_t: np.ndarray,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
) -> dict[str, BrainRegionMetrics]:
    """Calculate metrics for every LPBA40 browsing row using the shared engine."""
    return _region_metrics(vectors_t, samples, hierarchy or lpba40_region_hierarchy())


def lpba40_frame_region_metrics(
    fixed_vectors_t: np.ndarray,
    basis_vectors_t_per_a: np.ndarray,
    frame_currents_a: np.ndarray,
    samples: BrainSampleSet,
    hierarchy: BrainRegionNode | None = None,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Precompute LPBA40 frame metrics through the shared regional engine."""
    return _frame_region_metrics(
        fixed_vectors_t, basis_vectors_t_per_a, frame_currents_a, samples,
        hierarchy or lpba40_region_hierarchy(), progress_callback=progress_callback,
    )


def weighted_rms_coefficients(
    fixed_vectors: np.ndarray,
    basis_vectors: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the shared weighted regional RMS coefficients."""
    return _weighted_rms_coefficients(fixed_vectors, basis_vectors, weights)


def weighted_rms_from_coefficients(
    currents_a: np.ndarray,
    coefficients: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compatibility wrapper for the shared weighted regional RMS evaluator."""
    return _weighted_rms_from_coefficients(currents_a, coefficients)


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
    """Stream LPBA40 playback statistics through the shared regional engine."""
    return _playback_region_metrics_streaming(
        fixed_vectors_t, basis_vectors_t_per_a, analysis_chunk_factory, samples,
        hierarchy or lpba40_region_hierarchy(),
        progress_callback=progress_callback,
        analysis_sample_count=analysis_sample_count,
    )


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
    """Summarize LPBA40 playback through the shared regional engine."""
    return _playback_region_metrics(
        fixed_vectors_t, basis_vectors_t_per_a, analysis_currents_a, analysis_times_s,
        frame_metrics, frame_times_s, samples, hierarchy or lpba40_region_hierarchy(),
        progress_callback=progress_callback, full_timeline=full_timeline,
    )
