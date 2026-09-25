"""Bundled SRI24 v2.0 anatomical label resources.

The official NIfTI label volumes remain in their native SRI24 RAS millimetre
frame.  This module validates that frame, parses the upstream colour tables,
and exposes complete label definitions without requiring nibabel, CMTK, SPM,
or a network connection.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .nifti import NiftiReadError, read_nifti_volume


class SRI24AtlasError(ValueError):
    """Raised when a bundled SRI24 atlas asset is missing or inconsistent."""


SRI24_LABEL_ASSET_DIR = Path(__file__).with_name("assets") / "sri24_labels"
SRI24_LABEL_MANIFEST = SRI24_LABEL_ASSET_DIR / "manifest.json"
SRI24_GRID_SHAPE = (240, 240, 155)
SRI24_LABEL_AFFINE_RAS_MM = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SRI24AtlasLabel:
    """One integer atlas value with its upstream name and display colour."""

    value: int
    name: str
    rgba: tuple[int, int, int, int]


@dataclass(frozen=True)
class SRI24Parcellation:
    """One validated SRI24 label volume and its complete lookup table."""

    key: str
    display_name: str
    volume: np.ndarray
    affine_ras_mm: np.ndarray
    labels: dict[int, SRI24AtlasLabel]
    present_label_ids: tuple[int, ...]
    unused_table_label_ids: tuple[int, ...]
    supplemental_label_ids: tuple[int, ...]
    duplicate_table_label_ids: tuple[int, ...]

    @property
    def non_background_labels(self) -> tuple[SRI24AtlasLabel, ...]:
        """Return non-background definitions that occur in this volume."""

        return tuple(self.labels[value] for value in self.present_label_ids if value != 0)


_PARCELLATION_SPECS: dict[str, dict[str, str]] = {
    "tzo116plus": {
        "display_name": "AAL-derived tzo116plus",
        "volume": "tzo116plus.nii.gz",
        "labels": "SRI24-tzo116plus.txt",
    },
    "lpba40": {
        "display_name": "LPBA40",
        "volume": "lpba40.nii.gz",
        "labels": "LPBA40-labels.txt",
    },
}
_PARCELLATION_ALIASES = {
    "aal": "tzo116plus",
    "aal116": "tzo116plus",
    "tzo": "tzo116plus",
    "lpba": "lpba40",
}


def sri24_parcellation_choices() -> tuple[dict[str, str], ...]:
    """Return stable keys and user-facing names for bundled parcellations."""

    return tuple(
        {"key": key, "label": spec["display_name"]}
        for key, spec in _PARCELLATION_SPECS.items()
    )


def _parse_label_table(
    path: Path,
) -> tuple[dict[int, SRI24AtlasLabel], tuple[int, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SRI24AtlasError(f"Could not read {path.name}: {error}") from error
    labels: dict[int, SRI24AtlasLabel] = {}
    duplicates: list[int] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        try:
            fields = shlex.split(raw_line)
        except ValueError as error:
            raise SRI24AtlasError(
                f"{path.name} line {line_number} has invalid quoting."
            ) from error
        if len(fields) < 6:
            raise SRI24AtlasError(
                f"{path.name} line {line_number} does not contain ID, name, and RGBA values."
            )
        try:
            value = int(fields[0])
            rgba = tuple(int(component) for component in fields[2:6])
        except ValueError as error:
            raise SRI24AtlasError(
                f"{path.name} line {line_number} contains a non-integer ID or colour."
            ) from error
        if len(rgba) != 4 or any(component < 0 or component > 255 for component in rgba):
            raise SRI24AtlasError(
                f"{path.name} line {line_number} contains an invalid RGBA colour."
            )
        definition = SRI24AtlasLabel(value, str(fields[1]), rgba)
        previous = labels.get(value)
        if previous is not None:
            if previous != definition:
                raise SRI24AtlasError(
                    f"{path.name} contains conflicting definitions for label {value}."
                )
            duplicates.append(value)
            continue
        labels[value] = definition
    if 0 not in labels:
        raise SRI24AtlasError(f"{path.name} does not define background label 0.")
    return labels, tuple(sorted(set(duplicates)))


def _tzo_supplemental_labels() -> dict[int, SRI24AtlasLabel]:
    """Fill nine spatial labels omitted from the official v2.0 text table.

    The upstream volume encodes ventricular slices in the label value.  The
    missing names and colours therefore follow the same deterministic sequence
    as the adjacent official rows; the upstream table itself remains unmodified.
    """

    labels: dict[int, SRI24AtlasLabel] = {}
    for value in (422, 424, 426, 428, 430, 432, 434):
        y_index = 48 + (value - 201) // 2
        green = y_index + 90
        labels[value] = SRI24AtlasLabel(
            value,
            f"LateralVentricle_R_y{y_index}",
            (0, green, 0, 255),
        )
    for value in (476, 478):
        y_index = 98 + (value - 491) // 2
        cyan = y_index + 40
        labels[value] = SRI24AtlasLabel(
            value,
            f"ThirdVentricle_R_y{y_index}",
            (0, cyan, cyan, 255),
        )
    return labels


def _validated_sri24_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        volume, affine = read_nifti_volume(path)
    except NiftiReadError as error:
        raise SRI24AtlasError(str(error)) from error
    if volume.shape != SRI24_GRID_SHAPE:
        raise SRI24AtlasError(
            f"{path.name} uses grid {volume.shape!r}; expected {SRI24_GRID_SHAPE!r}."
        )
    if not np.issubdtype(volume.dtype, np.integer):
        raise SRI24AtlasError(f"{path.name} is not an integer label volume.")
    if not np.allclose(affine, SRI24_LABEL_AFFINE_RAS_MM, atol=1e-6):
        raise SRI24AtlasError(
            f"{path.name} does not use the expected SRI24 v2.0 RAS affine."
        )
    return volume, affine


def load_sri24_parcellation(key: str) -> SRI24Parcellation:
    """Load and validate one bundled SRI24 anatomical parcellation."""

    normalized = str(key).strip().lower()
    normalized = _PARCELLATION_ALIASES.get(normalized, normalized)
    spec = _PARCELLATION_SPECS.get(normalized)
    if spec is None:
        raise SRI24AtlasError(
            "Unknown SRI24 parcellation. Choose tzo116plus or lpba40."
        )
    volume, affine = _validated_sri24_volume(
        SRI24_LABEL_ASSET_DIR / spec["volume"]
    )
    labels, duplicates = _parse_label_table(
        SRI24_LABEL_ASSET_DIR / spec["labels"]
    )
    supplemental: dict[int, SRI24AtlasLabel] = {}
    if normalized == "tzo116plus":
        supplemental = _tzo_supplemental_labels()
        for value, definition in supplemental.items():
            if value in labels and labels[value] != definition:
                raise SRI24AtlasError(
                    f"The supplemental tzo116plus label {value} conflicts with the official table."
                )
            labels.setdefault(value, definition)

    present = tuple(int(value) for value in np.unique(volume))
    missing = sorted(set(present) - set(labels))
    if missing:
        raise SRI24AtlasError(
            f"{spec['volume']} contains labels with no definitions: {missing}."
        )
    unused = tuple(sorted(set(labels) - set(present)))
    return SRI24Parcellation(
        key=normalized,
        display_name=spec["display_name"],
        volume=volume,
        affine_ras_mm=affine,
        labels=labels,
        present_label_ids=present,
        unused_table_label_ids=unused,
        supplemental_label_ids=tuple(sorted(supplemental)),
        duplicate_table_label_ids=duplicates,
    )


def load_sri24_supratentorial_mask() -> tuple[np.ndarray, np.ndarray]:
    """Load the unmodified support mask included in the official label archive."""

    volume, affine = _validated_sri24_volume(SRI24_LABEL_ASSET_DIR / "suptent.nii.gz")
    values = set(int(value) for value in np.unique(volume))
    if values != {0, 1, 2}:
        raise SRI24AtlasError(
            f"suptent.nii.gz contains unexpected values: {sorted(values)}."
        )
    return volume, affine


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SRI24AtlasError(f"Could not hash {path.name}: {error}") from error
    return digest.hexdigest()


def _sha256_gzip_payload(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, gzip.BadGzipFile, EOFError) as error:
        raise SRI24AtlasError(f"Could not decompress {path.name}: {error}") from error
    return digest.hexdigest()


def _laterality_check(parcellation: SRI24Parcellation) -> tuple[int, tuple[str, ...]]:
    names_to_values = {definition.name: value for value, definition in parcellation.labels.items()}
    pairs: list[tuple[str, str]] = []
    for definition in parcellation.non_background_labels:
        name = definition.name
        if name.startswith("L_"):
            right_name = "R_" + name[2:]
        elif name.endswith("_L"):
            right_name = name[:-2] + "_R"
        else:
            continue
        right_value = names_to_values.get(right_name)
        if right_value in parcellation.present_label_ids:
            pairs.append((name, right_name))

    max_value = max(parcellation.present_label_ids)
    counts = np.zeros(max_value + 1, dtype=np.int64)
    weighted_x = np.zeros(max_value + 1, dtype=float)
    for x_index in range(parcellation.volume.shape[0]):
        slice_counts = np.bincount(
            np.asarray(parcellation.volume[x_index], dtype=np.int64).ravel(),
            minlength=max_value + 1,
        )
        counts += slice_counts[: max_value + 1]
        world_x = float(
            parcellation.affine_ras_mm[0, 0] * x_index
            + parcellation.affine_ras_mm[0, 3]
        )
        weighted_x += world_x * slice_counts[: max_value + 1]

    failures: list[str] = []
    for left_name, right_name in pairs:
        left_value = names_to_values[left_name]
        right_value = names_to_values[right_name]
        left_x = weighted_x[left_value] / counts[left_value]
        right_x = weighted_x[right_value] / counts[right_value]
        if not left_x < right_x:
            failures.append(f"{left_name}/{right_name}")
    return len(pairs), tuple(failures)


def validate_sri24_asset_bundle() -> dict[str, Any]:
    """Perform exhaustive integrity, lookup-coverage, and laterality checks."""

    try:
        manifest = json.loads(SRI24_LABEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SRI24AtlasError(f"Could not read the SRI24 asset manifest: {error}") from error
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise SRI24AtlasError("The SRI24 asset manifest has no asset table.")
    for filename, metadata in assets.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            raise SRI24AtlasError(f"The manifest entry for {filename} has no SHA-256.")
        actual = _sha256(SRI24_LABEL_ASSET_DIR / filename)
        if actual != metadata["sha256"]:
            raise SRI24AtlasError(f"Bundled SRI24 asset checksum mismatch: {filename}.")
        original_hash = metadata.get("uncompressed_sha256")
        if isinstance(original_hash, str):
            actual_original = _sha256_gzip_payload(SRI24_LABEL_ASSET_DIR / filename)
            if actual_original != original_hash:
                raise SRI24AtlasError(
                    f"Bundled SRI24 asset does not reproduce its source payload: {filename}."
                )

    report: dict[str, Any] = {
        "atlas_release": manifest.get("atlas_release"),
        "source_archive_sha256": manifest.get("source", {}).get("archive_sha256"),
        "maps": {},
    }
    mesh_metadata = assets.get("lpba40_meshes.npz")
    if not isinstance(mesh_metadata, dict):
        raise SRI24AtlasError("The SRI24 manifest has no LPBA40 mesh-cache entry.")
    if mesh_metadata.get("source_volume_sha256") != assets["lpba40.nii.gz"].get(
        "sha256"
    ):
        raise SRI24AtlasError(
            "The LPBA40 mesh cache does not reference the bundled atlas checksum."
        )
    if mesh_metadata.get("source_label_table_sha256") != assets[
        "LPBA40-labels.txt"
    ].get("sha256"):
        raise SRI24AtlasError(
            "The LPBA40 mesh cache does not reference the bundled label-table checksum."
        )
    report["lpba40_mesh_cache"] = {
        key: mesh_metadata.get(key)
        for key in (
            "mesh_version",
            "source_stride",
            "source_shape",
            "coordinate_frame",
            "label_count",
            "vertex_count",
            "triangle_count",
            "generator",
        )
    }
    for key in _PARCELLATION_SPECS:
        parcellation = load_sri24_parcellation(key)
        metadata = assets[_PARCELLATION_SPECS[key]["volume"]]
        if str(parcellation.volume.dtype) != metadata.get("dtype"):
            raise SRI24AtlasError(
                f"{parcellation.display_name} datatype disagrees with the manifest."
            )
        if len(parcellation.present_label_ids) != metadata.get("unique_label_count"):
            raise SRI24AtlasError(
                f"{parcellation.display_name} label count disagrees with the manifest."
            )
        if int(np.count_nonzero(parcellation.volume)) != metadata.get("nonzero_voxel_count"):
            raise SRI24AtlasError(
                f"{parcellation.display_name} voxel count disagrees with the manifest."
            )
        expected_supplements = tuple(metadata.get("volume_ids_absent_from_table", ()))
        if parcellation.supplemental_label_ids != expected_supplements:
            raise SRI24AtlasError(
                f"{parcellation.display_name} supplemental labels disagree with the manifest."
            )
        expected_unused = tuple(metadata.get("table_ids_absent_from_volume", ()))
        if parcellation.unused_table_label_ids != expected_unused:
            raise SRI24AtlasError(
                f"{parcellation.display_name} unused table labels disagree with the manifest."
            )
        expected_duplicates = tuple(metadata.get("table_duplicate_ids", ()))
        if parcellation.duplicate_table_label_ids != expected_duplicates:
            raise SRI24AtlasError(
                f"{parcellation.display_name} duplicate table labels disagree with the manifest."
            )
        checked_pairs, failures = _laterality_check(parcellation)
        if failures:
            raise SRI24AtlasError(
                f"{parcellation.display_name} failed left/right checks: {', '.join(failures)}."
            )
        report["maps"][key] = {
            "present_label_count": len(parcellation.present_label_ids),
            "non_background_label_count": len(parcellation.non_background_labels),
            "supplemental_label_ids": list(parcellation.supplemental_label_ids),
            "unused_table_label_ids": list(parcellation.unused_table_label_ids),
            "duplicate_table_label_ids": list(parcellation.duplicate_table_label_ids),
            "laterality_pairs_checked": checked_pairs,
        }
    support, _affine = load_sri24_supratentorial_mask()
    support_metadata = assets["suptent.nii.gz"]
    if str(support.dtype) != support_metadata.get("dtype"):
        raise SRI24AtlasError("The supratentorial support-mask datatype is unexpected.")
    if int(np.count_nonzero(support)) != support_metadata.get("nonzero_voxel_count"):
        raise SRI24AtlasError("The supratentorial support-mask count disagrees with the manifest.")
    report["suptent_values"] = [int(value) for value in np.unique(support)]
    return report
