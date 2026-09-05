"""Lightweight UPENN-GBM case download and atlas-mesh conversion.

The supported release assets are preprocessed NIfTI segmentations in the same
SRI24 RAS space as the built-in brain.  This module deliberately uses only the
standard library and NumPy: importing one target should not pull a full DICOM
database stack into the desktop build.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .anatomy_frames import sri24_ras_mm_to_head_m, sri24_ras_mm_to_head_mm
from .nifti import NiftiReadError, read_nifti_volume

UPENN_GBM_COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/upenn-gbm/"
UPENN_GBM_DATASET_DOI = "10.7937/TCIA.709X-DN49"
UPENN_GBM_RELEASE_URL = "https://github.com/data-nih/tcia/releases/tag/upenn-gbm"
UPENN_GBM_RELEASE_API_URL = (
    "https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm"
)
UPENN_GBM_ASSET_BASE_URL = (
    "https://github.com/data-nih/tcia/releases/download/upenn-gbm"
)
UPENN_GBM_LICENSE = "CC BY 4.0"
UPENN_GBM_CATALOGUE_CACHE_FILENAME = "catalogue.json"
UPENN_GBM_CATALOGUE_SCHEMA_VERSION = 2

_SEGMENTATION_ASSET_PATTERN = re.compile(r"^sub-(\d{3})_seg\.nii\.gz$")
_MAX_CATALOGUE_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_SEGMENTATION_BYTES = 64 * 1024 * 1024
_EXPECTED_SRI24_AFFINE_RAS_MM = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
# Axis-aligned bounds of the bundled SRI24 GM+WM brain after applying the
# canonical SRI24 RAS -> NAS/LPA/RPA transform.  The lower/upper thirds provide
# deliberately coarse atlas-relative anterior/posterior and inferior/superior
# descriptors; they are not a substitute for a labelled anatomical atlas.
_SRI24_BRAIN_BOUNDS_HEAD_MM = np.asarray(
    [
        [-64.991, -80.265, -13.263],
        [67.406, 90.879, 112.757],
    ],
    dtype=float,
)
_SRI24_BRAIN_LOWER_THIRD_HEAD_MM = (
    _SRI24_BRAIN_BOUNDS_HEAD_MM[0]
    + (_SRI24_BRAIN_BOUNDS_HEAD_MM[1] - _SRI24_BRAIN_BOUNDS_HEAD_MM[0]) / 3.0
)
_SRI24_BRAIN_UPPER_THIRD_HEAD_MM = (
    _SRI24_BRAIN_BOUNDS_HEAD_MM[0]
    + 2.0 * (_SRI24_BRAIN_BOUNDS_HEAD_MM[1] - _SRI24_BRAIN_BOUNDS_HEAD_MM[0]) / 3.0
)

# The data-nih/tcia release currently exposes per-case NIfTI segmentation
# assets for these subjects.  Keeping the small catalogue in the application
# makes the case browser instant and avoids a GitHub API request before every
# import.  The segmentation file itself is still fetched on demand.
_SEGMENTATION_CASE_NUMBERS = tuple(
    int(value)
    for value in """
002 006 008 009 011 013 014 016 017 018 020 021 026 029 030 031 033 035
040 041 043 054 059 060 062 066 069 073 075 076 080 082 083 086 088 091
093 096 100 101 102 105 106 107 108 112 113 114 115 117 118 119 121 122
124 131 134 135 136 137 138 139 140 141 143 144 146 147 148 149 151 154
156 158 166 172 173 174 176 178 180 189 192 193 196 197 201 205 206 208
215 217 226 227 228 238 240 249 251 252 253 254 256 259 261 262 264 266
270 272 274 275 277 284 285 290 294 307 312 330 344 351 356 360 362 367
368 371 373 375 376 380 384 388 390 391 393 398 402 404 418 428 430 437
438 439 474
""".split()
)


@dataclass(frozen=True)
class UpennGbmCase:
    """One supported atlas-normalized UPENN-GBM segmentation asset."""

    subject_number: int
    asset_filename: str | None = None
    asset_download_url: str | None = None
    asset_size_bytes: int | None = None
    asset_updated_at: str | None = None
    compatibility: str = "snapshot"
    compatibility_detail: str = (
        "Bundled catalogue entry; its NIfTI is validated when imported."
    )
    approximate_location: str | None = None
    centroid_head_mm: tuple[float, float, float] | None = None
    left_fraction: float | None = None
    right_fraction: float | None = None
    tumor_core_volume_ml: float | None = None
    whole_tumor_volume_ml: float | None = None
    extent_head_mm: tuple[float, float, float] | None = None

    @property
    def case_id(self) -> str:
        return f"UPENN-GBM-{self.subject_number:05d}"

    @property
    def short_id(self) -> str:
        return f"sub-{self.subject_number:03d}"

    @property
    def filename(self) -> str:
        return self.asset_filename or f"{self.short_id}_seg.nii.gz"

    @property
    def download_url(self) -> str:
        return self.asset_download_url or f"{UPENN_GBM_ASSET_BASE_URL}/{self.filename}"

    @property
    def is_importable(self) -> bool:
        return self.compatibility != "incompatible"

    @property
    def status_label(self) -> str:
        return {
            "compatible": "Compatible",
            "incompatible": "Incompatible",
            "unverified": "Unverified",
            "snapshot": "Bundled snapshot",
        }.get(self.compatibility, "Unverified")

    @property
    def has_case_statistics(self) -> bool:
        return (
            self.approximate_location is not None
            and self.centroid_head_mm is not None
            and self.left_fraction is not None
            and self.right_fraction is not None
            and self.tumor_core_volume_ml is not None
            and self.whole_tumor_volume_ml is not None
            and self.extent_head_mm is not None
        )


@dataclass(frozen=True)
class UpennGbmCatalogue:
    """A bundled, cached, or freshly refreshed segmentation catalogue."""

    cases: tuple[UpennGbmCase, ...]
    source: str = "bundled"
    refreshed_at: str | None = None
    release_updated_at: str | None = None
    release_id: int | None = None
    etag: str | None = None

    @property
    def compatible_count(self) -> int:
        return sum(case.compatibility == "compatible" for case in self.cases)

    @property
    def incompatible_count(self) -> int:
        return sum(case.compatibility == "incompatible" for case in self.cases)

    @property
    def unverified_count(self) -> int:
        return sum(
            case.compatibility in {"unverified", "snapshot"} for case in self.cases
        )


UPENN_GBM_CASES = tuple(UpennGbmCase(value) for value in _SEGMENTATION_CASE_NUMBERS)
UPENN_GBM_BUNDLED_CATALOGUE = UpennGbmCatalogue(cases=UPENN_GBM_CASES)


@dataclass(frozen=True)
class TumorRegion:
    key: str
    label: str
    short_label: str
    source_values: tuple[int, ...]
    colour: str


# The mirrored NIfTI assets use compact values 1/2/3 for the three independent
# subregions.  Derived core and whole-tumor masks are unions, matching the
# definitions used by the UPENN-GBM publication.
UPENN_GBM_REGIONS: tuple[TumorRegion, ...] = (
    TumorRegion(
        "tumor_core",
        "Tumor core (NCR/NET + enhancing)",
        "Tumor core",
        (1, 3),
        "#d946ef",
    ),
    TumorRegion(
        "enhancing_tumor",
        "Enhancing tumor (ET)",
        "Enhancing tumor",
        (3,),
        "#f59e0b",
    ),
    TumorRegion(
        "necrotic_core",
        "Necrotic/non-enhancing core (NCR/NET)",
        "Necrotic/non-enhancing core",
        (1,),
        "#dc2626",
    ),
    TumorRegion(
        "edema",
        "Peritumoral edema/infiltration (ED)",
        "Edema/infiltration",
        (2,),
        "#22c55e",
    ),
    TumorRegion(
        "whole_tumor",
        "Whole tumor (all labels)",
        "Whole tumor",
        (1, 2, 3),
        "#06b6d4",
    ),
)
UPENN_GBM_REGIONS_BY_KEY = {region.key: region for region in UPENN_GBM_REGIONS}


class UpennGbmImportError(RuntimeError):
    """Raised when a case cannot be downloaded or converted safely."""


class UpennGbmImportCancelled(UpennGbmImportError):
    """Raised when the user cancels a case import."""


ProgressCallback = Callable[[dict[str, Any]], None]


def find_upenn_gbm_case(
    case_id: str | int | UpennGbmCase,
    cases: Iterable[UpennGbmCase] | None = None,
) -> UpennGbmCase:
    """Resolve either a numeric subject or a displayed UPENN case id."""
    if isinstance(case_id, UpennGbmCase):
        return case_id
    if isinstance(case_id, int):
        number = case_id
    else:
        digits = "".join(character for character in str(case_id) if character.isdigit())
        if not digits:
            raise UpennGbmImportError(f"Unrecognized UPENN-GBM case id: {case_id}")
        number = int(digits[-5:])
    for case in UPENN_GBM_CASES if cases is None else cases:
        if case.subject_number == number:
            return case
    raise UpennGbmImportError(
        f"{case_id} does not have a per-case segmentation in this catalogue."
    )


def _catalogue_cache_path(cache_directory: str | Path) -> Path:
    return Path(cache_directory) / UPENN_GBM_CATALOGUE_CACHE_FILENAME


def _is_supported_asset_url(url: str, filename: str) -> bool:
    parsed = urlparse(str(url))
    return (
        parsed.scheme == "https"
        and parsed.netloc.casefold() == "github.com"
        and parsed.path
        == f"/data-nih/tcia/releases/download/upenn-gbm/{filename}"
        and not parsed.query
        and not parsed.fragment
    )


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Catalogue statistic is not finite.")
    return result


def _optional_finite_triplet(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Catalogue statistic is not a three-value coordinate.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Catalogue coordinate is not finite.")
    return result


def _case_from_cache(value: object) -> UpennGbmCase:
    if not isinstance(value, dict):
        raise ValueError("Catalogue case entry is not an object.")
    subject_number = int(value["subject_number"])
    filename = str(value["filename"])
    expected_filename = f"sub-{subject_number:03d}_seg.nii.gz"
    if not 0 < subject_number < 1000 or filename != expected_filename:
        raise ValueError("Catalogue case id and filename do not agree.")
    download_url = str(value["download_url"])
    if not _is_supported_asset_url(download_url, filename):
        raise ValueError("Catalogue case has an unsupported download URL.")
    compatibility = str(value.get("compatibility", "unverified"))
    if compatibility not in {"compatible", "incompatible", "unverified"}:
        compatibility = "unverified"
    size_value = value.get("size_bytes")
    size_bytes = int(size_value) if size_value is not None else None
    if size_bytes is not None and not 0 < size_bytes <= _MAX_SEGMENTATION_BYTES:
        raise ValueError("Catalogue case has an invalid asset size.")
    case = UpennGbmCase(
        subject_number=subject_number,
        asset_filename=filename,
        asset_download_url=download_url,
        asset_size_bytes=size_bytes,
        asset_updated_at=(
            str(value["updated_at"]) if value.get("updated_at") is not None else None
        ),
        compatibility=compatibility,
        compatibility_detail=str(value.get("compatibility_detail", "")),
        approximate_location=(
            str(value["approximate_location"])
            if value.get("approximate_location") is not None
            else None
        ),
        centroid_head_mm=_optional_finite_triplet(value.get("centroid_head_mm")),
        left_fraction=_optional_finite_float(value.get("left_fraction")),
        right_fraction=_optional_finite_float(value.get("right_fraction")),
        tumor_core_volume_ml=_optional_finite_float(
            value.get("tumor_core_volume_ml")
        ),
        whole_tumor_volume_ml=_optional_finite_float(
            value.get("whole_tumor_volume_ml")
        ),
        extent_head_mm=_optional_finite_triplet(value.get("extent_head_mm")),
    )
    if case.compatibility == "compatible" and not case.has_case_statistics:
        raise ValueError("Compatible catalogue case is missing derived statistics.")
    if case.has_case_statistics:
        if not (
            0.0 <= case.left_fraction <= 1.0
            and 0.0 <= case.right_fraction <= 1.0
            and math.isclose(
                case.left_fraction + case.right_fraction, 1.0, abs_tol=1e-6
            )
            and case.tumor_core_volume_ml >= 0.0
            and case.whole_tumor_volume_ml > 0.0
            and all(value > 0.0 for value in case.extent_head_mm)
        ):
            raise ValueError("Catalogue case statistics are outside valid ranges.")
    return case


def load_upenn_gbm_catalogue(cache_directory: str | Path) -> UpennGbmCatalogue:
    """Load the last verified live catalogue, or the bundled offline snapshot."""
    path = _catalogue_cache_path(cache_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or int(payload.get("schema_version", 0))
            != UPENN_GBM_CATALOGUE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported catalogue cache schema.")
        cases = tuple(_case_from_cache(value) for value in payload["cases"])
        if not cases or len({case.subject_number for case in cases}) != len(cases):
            raise ValueError("Catalogue cache is empty or contains duplicate cases.")
        return UpennGbmCatalogue(
            cases=tuple(sorted(cases, key=lambda case: case.subject_number)),
            source="cache",
            refreshed_at=(
                str(payload["refreshed_at"])
                if payload.get("refreshed_at") is not None
                else None
            ),
            release_updated_at=(
                str(payload["release_updated_at"])
                if payload.get("release_updated_at") is not None
                else None
            ),
            release_id=(
                int(payload["release_id"])
                if payload.get("release_id") is not None
                else None
            ),
            etag=str(payload["etag"]) if payload.get("etag") else None,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return UPENN_GBM_BUNDLED_CATALOGUE


def _write_upenn_gbm_catalogue(
    cache_directory: str | Path, catalogue: UpennGbmCatalogue
) -> None:
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    destination = _catalogue_cache_path(cache)
    partial = destination.with_suffix(destination.suffix + ".part")
    payload = {
        "schema_version": UPENN_GBM_CATALOGUE_SCHEMA_VERSION,
        "refreshed_at": catalogue.refreshed_at,
        "release_updated_at": catalogue.release_updated_at,
        "release_id": catalogue.release_id,
        "etag": catalogue.etag,
        "cases": [
            {
                "subject_number": case.subject_number,
                "filename": case.filename,
                "download_url": case.download_url,
                "size_bytes": case.asset_size_bytes,
                "updated_at": case.asset_updated_at,
                "compatibility": case.compatibility,
                "compatibility_detail": case.compatibility_detail,
                "approximate_location": case.approximate_location,
                "centroid_head_mm": case.centroid_head_mm,
                "left_fraction": case.left_fraction,
                "right_fraction": case.right_fraction,
                "tumor_core_volume_ml": case.tumor_core_volume_ml,
                "whole_tumor_volume_ml": case.whole_tumor_volume_ml,
                "extent_head_mm": case.extent_head_mm,
            }
            for case in catalogue.cases
        ],
    }
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, destination)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise UpennGbmImportError(
            f"Could not save the refreshed UPENN-GBM catalogue: {error}"
        ) from error


def _read_release_payload(
    cache_directory: str | Path,
    *,
    cancel_event: threading.Event | None = None,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any] | None, str | None]:
    cached = load_upenn_gbm_catalogue(cache_directory)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "FieldWorkbench-UPENN-GBM-catalogue",
    }
    if cached.source == "cache" and cached.etag:
        headers["If-None-Match"] = cached.etag
    request = urllib.request.Request(UPENN_GBM_RELEASE_API_URL, headers=headers)
    try:
        with urlopen(request, timeout=45) as response:
            if int(getattr(response, "status", response.getcode())) == 304:
                return None, cached.etag
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > _MAX_CATALOGUE_RESPONSE_BYTES:
                raise UpennGbmImportError("The catalogue response exceeds 64 MiB.")
            chunks: list[bytes] = []
            received = 0
            while True:
                _check_cancel(cancel_event)
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > _MAX_CATALOGUE_RESPONSE_BYTES:
                    raise UpennGbmImportError("The catalogue response exceeds 64 MiB.")
                chunks.append(chunk)
            etag = response.headers.get("ETag")
        payload = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The release response is not an object.")
        return payload, str(etag) if etag else None
    except urllib.error.HTTPError as error:
        if error.code == 304 and cached.source == "cache":
            return None, cached.etag
        raise UpennGbmImportError(
            f"Could not refresh the UPENN-GBM release catalogue: {error}"
        ) from error
    except UpennGbmImportCancelled:
        raise
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise UpennGbmImportError(
            f"Could not refresh the UPENN-GBM release catalogue: {error}"
        ) from error


def _cases_from_release_payload(payload: dict[str, Any]) -> tuple[UpennGbmCase, ...]:
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise UpennGbmImportError("The release response does not contain an asset list.")
    cases: dict[int, UpennGbmCase] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        filename = str(asset.get("name", ""))
        match = _SEGMENTATION_ASSET_PATTERN.fullmatch(filename)
        if match is None:
            continue
        subject_number = int(match.group(1))
        download_url = str(asset.get("browser_download_url", ""))
        if not _is_supported_asset_url(download_url, filename):
            continue
        size_value = asset.get("size")
        try:
            size_bytes = int(size_value) if size_value is not None else None
        except (TypeError, ValueError):
            size_bytes = None
        if subject_number in cases:
            continue
        compatibility = "unverified"
        detail = "Not yet checked."
        if size_bytes is not None and not 352 < size_bytes <= _MAX_SEGMENTATION_BYTES:
            compatibility = "incompatible"
            detail = "The asset size is empty or exceeds the 64 MiB safety limit."
        cases[subject_number] = UpennGbmCase(
            subject_number=subject_number,
            asset_filename=filename,
            asset_download_url=download_url,
            asset_size_bytes=size_bytes,
            asset_updated_at=(
                str(asset["updated_at"]) if asset.get("updated_at") is not None else None
            ),
            compatibility=compatibility,
            compatibility_detail=detail,
        )
    if not cases:
        raise UpennGbmImportError(
            "The release currently exposes no compatible-looking per-case segmentation assets."
        )
    return tuple(cases[number] for number in sorted(cases))


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise UpennGbmImportCancelled("UPENN-GBM import cancelled.")


def _emit(progress_callback: ProgressCallback | None, **update: Any) -> None:
    if progress_callback is not None:
        progress_callback(update)


def download_upenn_gbm_segmentation(
    case: UpennGbmCase,
    cache_directory: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    force: bool = False,
) -> Path:
    """Download one small segmentation asset into the application cache."""
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / case.filename
    if destination.is_file() and destination.stat().st_size > 352 and not force:
        _emit(
            progress_callback,
            phase="download",
            percent=100,
            message=f"Using cached {case.filename}",
        )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        case.download_url,
        headers={"User-Agent": "FieldWorkbench-UPENN-GBM-importer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            total = int(response.headers.get("Content-Length") or 0)
            if total > _MAX_SEGMENTATION_BYTES:
                raise UpennGbmImportError(
                    "The case segmentation exceeds the 64 MiB safety limit."
                )
            received = 0
            digest = hashlib.sha256()
            with partial.open("wb") as handle:
                while True:
                    _check_cancel(cancel_event)
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > _MAX_SEGMENTATION_BYTES:
                        raise UpennGbmImportError(
                            "The case segmentation exceeds the 64 MiB safety limit."
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                    percent = min(99, round(received * 100 / total)) if total else 0
                    _emit(
                        progress_callback,
                        phase="download",
                        percent=percent,
                        message=f"Downloading {case.case_id} segmentation…",
                    )
        if received <= 352:
            raise UpennGbmImportError("The downloaded segmentation is empty or incomplete.")
        os.replace(partial, destination)
        _emit(
            progress_callback,
            phase="download",
            percent=100,
            message=f"Downloaded {case.filename}",
            sha256=digest.hexdigest(),
        )
        return destination
    except UpennGbmImportCancelled:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        partial.unlink(missing_ok=True)
        raise UpennGbmImportError(
            f"Could not download {case.case_id}: {error}"
        ) from error


def read_nifti_segmentation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the subset of NIfTI-1 required by the UPENN segmentation assets."""
    try:
        volume, affine = read_nifti_volume(path)
    except NiftiReadError as error:
        raise UpennGbmImportError(str(error)) from error
    if volume.shape != (240, 240, 155):
        raise UpennGbmImportError(
            "The downloaded mask is not on the expected 240×240×155 SRI24 grid."
        )
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(affine).all() or not np.allclose(spacing, 1.0, atol=0.01):
        raise UpennGbmImportError(
            "The downloaded mask is not on the expected 1 mm SRI24 physical grid."
        )
    if not np.allclose(affine, _EXPECTED_SRI24_AFFINE_RAS_MM, atol=0.01):
        raise UpennGbmImportError(
            "The downloaded mask does not use the expected SRI24 atlas affine/orientation."
        )
    rounded = np.rint(volume).astype(np.int16, copy=False)
    labels = set(map(int, np.unique(rounded)))
    if not labels.issubset({0, 1, 2, 3}) or labels <= {0}:
        raise UpennGbmImportError(
            f"Unexpected UPENN segmentation labels: {sorted(labels)}."
        )
    return rounded, affine


def calculate_upenn_gbm_case_statistics(
    segmentation: np.ndarray,
    affine_ras_mm: np.ndarray,
) -> dict[str, Any]:
    """Calculate compact atlas-relative measurements for the case chooser."""
    volume = np.asarray(segmentation)
    whole_mask = np.isin(volume, (1, 2, 3))
    whole_indices = np.argwhere(whole_mask)
    if not len(whole_indices):
        raise UpennGbmImportError("The UPENN segmentation contains no tumor voxels.")

    ras_mm = whole_indices @ affine_ras_mm[:3, :3].T + affine_ras_mm[:3, 3]
    head_mm = sri24_ras_mm_to_head_mm(ras_mm)
    centroid = np.mean(head_mm, axis=0)

    # Add one oriented voxel's projected width to the centre-to-centre span so
    # a one-voxel region correctly reports a non-zero physical extent.
    ras_origin = affine_ras_mm[:3, 3]
    head_origin = sri24_ras_mm_to_head_mm([ras_origin])[0]
    head_steps = sri24_ras_mm_to_head_mm(
        ras_origin + affine_ras_mm[:3, :3].T
    ) - head_origin
    voxel_aabb_width = np.sum(np.abs(head_steps), axis=0)
    extent = np.ptp(head_mm, axis=0) + voxel_aabb_width

    right_fraction = float(np.mean(head_mm[:, 0] >= 0.0))
    left_fraction = 1.0 - right_fraction
    if right_fraction >= 0.80:
        hemisphere = "Right"
    elif left_fraction >= 0.80:
        hemisphere = "Left"
    elif right_fraction >= 0.60:
        hemisphere = "Mostly right"
    elif left_fraction >= 0.60:
        hemisphere = "Mostly left"
    else:
        hemisphere = "Bilateral"

    if centroid[1] < _SRI24_BRAIN_LOWER_THIRD_HEAD_MM[1]:
        anterior_posterior = "posterior"
    elif centroid[1] > _SRI24_BRAIN_UPPER_THIRD_HEAD_MM[1]:
        anterior_posterior = "anterior"
    else:
        anterior_posterior = "central"
    if centroid[2] < _SRI24_BRAIN_LOWER_THIRD_HEAD_MM[2]:
        inferior_superior = "inferior"
    elif centroid[2] > _SRI24_BRAIN_UPPER_THIRD_HEAD_MM[2]:
        inferior_superior = "superior"
    else:
        inferior_superior = "middle"

    voxel_volume_ml = abs(float(np.linalg.det(affine_ras_mm[:3, :3]))) / 1000.0
    tumor_core_voxels = int(np.count_nonzero(np.isin(volume, (1, 3))))
    whole_tumor_voxels = int(len(whole_indices))
    return {
        "approximate_location": (
            f"{hemisphere} · {anterior_posterior} · {inferior_superior}"
        ),
        "centroid_head_mm": tuple(float(value) for value in centroid),
        "left_fraction": left_fraction,
        "right_fraction": right_fraction,
        "tumor_core_volume_ml": tumor_core_voxels * voxel_volume_ml,
        "whole_tumor_volume_ml": whole_tumor_voxels * voxel_volume_ml,
        "extent_head_mm": tuple(float(value) for value in extent),
    }


def _verify_upenn_gbm_case(
    case: UpennGbmCase,
    cache_directory: str | Path,
    cancel_event: threading.Event | None,
) -> UpennGbmCase:
    if case.compatibility == "incompatible":
        return case
    cache = Path(cache_directory)
    destination = cache / case.filename
    was_cached = destination.is_file() and destination.stat().st_size > 352
    force = bool(
        was_cached
        and case.asset_size_bytes is not None
        and destination.stat().st_size != case.asset_size_bytes
    )
    try:
        path = download_upenn_gbm_segmentation(
            case,
            cache,
            cancel_event=cancel_event,
            force=force,
        )
    except UpennGbmImportCancelled:
        raise
    except UpennGbmImportError as error:
        return replace(
            case,
            compatibility="unverified",
            compatibility_detail=f"Could not download for verification: {error}",
        )
    try:
        segmentation, affine = read_nifti_segmentation(path)
    except UpennGbmImportError as first_error:
        if was_cached and not force:
            path.unlink(missing_ok=True)
            try:
                path = download_upenn_gbm_segmentation(
                    case,
                    cache,
                    cancel_event=cancel_event,
                    force=True,
                )
            except UpennGbmImportCancelled:
                raise
            except UpennGbmImportError as error:
                return replace(
                    case,
                    compatibility="unverified",
                    compatibility_detail=f"Could not replace an invalid cached file: {error}",
                )
            try:
                segmentation, affine = read_nifti_segmentation(path)
            except UpennGbmImportError as error:
                return replace(
                    case,
                    compatibility="incompatible",
                    compatibility_detail=str(error),
                )
        else:
            return replace(
                case,
                compatibility="incompatible",
                compatibility_detail=str(first_error),
            )
    statistics = calculate_upenn_gbm_case_statistics(segmentation, affine)
    return replace(
        case,
        compatibility="compatible",
        compatibility_detail="Validated SRI24 grid, affine, labels, and non-empty mask.",
        **statistics,
    )


def refresh_upenn_gbm_catalogue(
    cache_directory: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> UpennGbmCatalogue:
    """Discover release masks, validate them in parallel, and cache the result."""
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    _emit(
        progress_callback,
        phase="catalogue",
        percent=1,
        message="Checking the UPENN-GBM release catalogue…",
    )
    payload, etag = _read_release_payload(
        cache,
        cancel_event=cancel_event,
        urlopen=urlopen,
    )
    if payload is None:
        cached = load_upenn_gbm_catalogue(cache)
        if cached.source != "cache":
            raise UpennGbmImportError(
                "The release reported no change, but no cached catalogue is available."
            )
        _emit(
            progress_callback,
            phase="catalogue",
            percent=100,
            message="Catalogue is already current.",
        )
        return cached

    cases = _cases_from_release_payload(payload)
    previous = load_upenn_gbm_catalogue(cache)
    previous_by_number = (
        {case.subject_number: case for case in previous.cases}
        if previous.source == "cache"
        else {}
    )
    verified: dict[int, UpennGbmCase] = {}
    to_verify: list[UpennGbmCase] = []
    for case in cases:
        old_case = previous_by_number.get(case.subject_number)
        same_asset = bool(
            old_case is not None
            and old_case.filename == case.filename
            and old_case.download_url == case.download_url
            and old_case.asset_size_bytes == case.asset_size_bytes
            and old_case.asset_updated_at == case.asset_updated_at
            and (
                old_case.compatibility == "incompatible"
                or old_case.has_case_statistics
            )
        )
        if same_asset and old_case.compatibility in {"compatible", "incompatible"}:
            verified[case.subject_number] = replace(
                case,
                compatibility=old_case.compatibility,
                compatibility_detail=old_case.compatibility_detail,
                approximate_location=old_case.approximate_location,
                centroid_head_mm=old_case.centroid_head_mm,
                left_fraction=old_case.left_fraction,
                right_fraction=old_case.right_fraction,
                tumor_core_volume_ml=old_case.tumor_core_volume_ml,
                whole_tumor_volume_ml=old_case.whole_tumor_volume_ml,
                extent_head_mm=old_case.extent_head_mm,
            )
        else:
            to_verify.append(case)
    _emit(
        progress_callback,
        phase="catalogue",
        percent=5,
        message=(
            f"Found {len(cases)} masks; checking {len(to_verify)} new, changed, "
            "or previously unverified assets…"
        ),
    )
    if to_verify:
        with ThreadPoolExecutor(max_workers=min(4, len(to_verify))) as executor:
            futures = {
                executor.submit(_verify_upenn_gbm_case, case, cache, cancel_event): case
                for case in to_verify
            }
            try:
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    _check_cancel(cancel_event)
                    case = future.result()
                    verified[case.subject_number] = case
                    percent = 5 + round(90 * completed_count / len(to_verify))
                    _emit(
                        progress_callback,
                        phase="catalogue",
                        percent=min(95, percent),
                        message=(
                            f"Checked {completed_count} of {len(to_verify)} changed masks "
                            f"({case.case_id}: {case.status_label.lower()})…"
                        ),
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    else:
        _emit(
            progress_callback,
            phase="catalogue",
            percent=95,
            message="All discovered masks match the verified cache.",
        )
    refreshed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    release_id_value = payload.get("id")
    try:
        release_id = int(release_id_value) if release_id_value is not None else None
    except (TypeError, ValueError):
        release_id = None
    catalogue = UpennGbmCatalogue(
        cases=tuple(verified[number] for number in sorted(verified)),
        source="live",
        refreshed_at=refreshed_at,
        release_updated_at=(
            str(payload["updated_at"]) if payload.get("updated_at") is not None else None
        ),
        release_id=release_id,
        etag=etag,
    )
    _write_upenn_gbm_catalogue(cache, catalogue)
    _emit(
        progress_callback,
        phase="catalogue",
        percent=100,
        message=(
            f"Catalogue refreshed: {catalogue.compatible_count} compatible, "
            f"{catalogue.incompatible_count} incompatible, "
            f"{catalogue.unverified_count} unverified."
        ),
    )
    return catalogue


_CUBE_CORNERS = np.asarray(
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ],
    dtype=np.int16,
)

# A globally consistent six-tetrahedra split around cube diagonal 0→7.  Unlike
# a raw voxel-face mesh, this resolves diagonal contacts without non-manifold
# edges, so the resulting target remains eligible for measurement/exclusion.
_CUBE_TETRAHEDRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)


def binary_mask_to_surface(
    mask: np.ndarray,
    *,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed, consistently oriented surface with marching tetrahedra."""
    binary = np.asarray(mask, dtype=bool)
    nonzero = np.argwhere(binary)
    if not len(nonzero):
        raise UpennGbmImportError("The requested tumor region is empty in this case.")

    lower = np.maximum(nonzero.min(axis=0) - 1, 0)
    upper = np.minimum(nonzero.max(axis=0) + 2, binary.shape)
    cropped = binary[
        tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))
    ]
    padded = np.pad(cropped, 1, mode="constant", constant_values=False)
    cell_shape = np.asarray(padded.shape, dtype=int) - 1
    corner_values = np.stack(
        [
            padded[
                corner[0] : corner[0] + cell_shape[0],
                corner[1] : corner[1] + cell_shape[1],
                corner[2] : corner[2] + cell_shape[2],
            ]
            for corner in _CUBE_CORNERS
        ],
        axis=-1,
    )
    active = np.any(corner_values, axis=-1) & ~np.all(corner_values, axis=-1)
    cells = np.argwhere(active)
    values = corner_values[active]
    if not len(cells):
        raise UpennGbmImportError("No boundary could be generated for this tumor region.")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    edge_vertices: dict[tuple[int, int], int] = {}
    grid_shape = padded.shape

    def flat_index(point: np.ndarray) -> int:
        return (
            (int(point[0]) * grid_shape[1] + int(point[1])) * grid_shape[2]
            + int(point[2])
        )

    def vertex_index(first: np.ndarray, second: np.ndarray) -> int:
        first_index = flat_index(first)
        second_index = flat_index(second)
        edge = (
            (first_index, second_index)
            if first_index < second_index
            else (second_index, first_index)
        )
        existing = edge_vertices.get(edge)
        if existing is not None:
            return existing
        index = len(vertices)
        edge_vertices[edge] = index
        vertices.append((0.5 * (first + second)).tolist())
        return index

    total = len(cells)
    for cell_index, (cell, cube_values) in enumerate(zip(cells, values, strict=True)):
        if cell_index % 512 == 0:
            _check_cancel(cancel_event)
            if progress_callback is not None:
                progress_callback(round(cell_index * 100 / total))
        points = cell + _CUBE_CORNERS
        for tetrahedron in _CUBE_TETRAHEDRA:
            inside = [index for index in tetrahedron if cube_values[index]]
            outside = [index for index in tetrahedron if not cube_values[index]]
            inside_count = len(inside)
            if inside_count in {0, 4}:
                continue
            if inside_count == 1:
                crossings = [(inside[0], index) for index in outside]
            elif inside_count == 3:
                crossings = [(outside[0], index) for index in inside]
            else:
                first_inside, second_inside = inside
                first_outside, second_outside = outside
                crossings = [
                    (first_inside, first_outside),
                    (first_inside, second_outside),
                    (second_inside, second_outside),
                    (second_inside, first_outside),
                ]
            polygon = [
                vertex_index(points[first], points[second])
                for first, second in crossings
            ]
            triangles = (
                [polygon]
                if len(polygon) == 3
                else [
                    [polygon[0], polygon[1], polygon[2]],
                    [polygon[0], polygon[2], polygon[3]],
                ]
            )
            outward = points[outside].mean(axis=0) - points[inside].mean(axis=0)
            for triangle in triangles:
                coordinates = np.asarray([vertices[index] for index in triangle])
                normal = np.cross(
                    coordinates[1] - coordinates[0],
                    coordinates[2] - coordinates[0],
                )
                if float(np.dot(normal, outward)) < 0.0:
                    triangle = [triangle[0], triangle[2], triangle[1]]
                faces.append(triangle)

    _check_cancel(cancel_event)
    if progress_callback is not None:
        progress_callback(100)
    vertex_array = np.asarray(vertices, dtype=float)
    vertex_array += lower - 1
    return vertex_array, np.asarray(faces, dtype=np.int32)


def _transform_voxel_vertices_to_head_m(
    vertices: np.ndarray, affine_ras_mm: np.ndarray
) -> np.ndarray:
    ras_mm = vertices @ affine_ras_mm[:3, :3].T + affine_ras_mm[:3, 3]
    return sri24_ras_mm_to_head_m(ras_mm)


def build_upenn_gbm_region_meshes(
    case: UpennGbmCase,
    segmentation: np.ndarray,
    affine_ras_mm: np.ndarray,
    region_keys: Iterable[str],
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> list[dict[str, Any]]:
    """Build selected case regions as closed head-frame triangle meshes."""
    keys = list(dict.fromkeys(str(key) for key in region_keys))
    if not keys:
        raise UpennGbmImportError("Select at least one tumor region.")
    unknown = [key for key in keys if key not in UPENN_GBM_REGIONS_BY_KEY]
    if unknown:
        raise UpennGbmImportError(f"Unknown tumor region: {unknown[0]}")

    voxel_volume_ml = abs(float(np.linalg.det(affine_ras_mm[:3, :3]))) / 1000.0
    results: list[dict[str, Any]] = []
    for region_index, key in enumerate(keys):
        _check_cancel(cancel_event)
        region = UPENN_GBM_REGIONS_BY_KEY[key]
        mask = np.isin(segmentation, region.source_values)
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            raise UpennGbmImportError(
                f"{case.case_id} contains no voxels for {region.short_label}."
            )
        _emit(
            progress_callback,
            phase="mesh",
            percent=0,
            region_index=region_index,
            region_count=len(keys),
            message=f"Building {region.short_label} surface…",
        )

        def mesh_progress(percent: int) -> None:
            _emit(
                progress_callback,
                phase="mesh",
                percent=percent,
                region_index=region_index,
                region_count=len(keys),
                message=f"Building {region.short_label} surface…",
            )

        voxel_vertices, faces = binary_mask_to_surface(
            mask,
            cancel_event=cancel_event,
            progress_callback=mesh_progress,
        )
        vertices_head_m = _transform_voxel_vertices_to_head_m(
            voxel_vertices, affine_ras_mm
        )
        triangles = vertices_head_m[faces]
        signed_volume_m3 = float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
            )
            / 6.0
        )
        if signed_volume_m3 < 0.0:
            faces = faces[:, [0, 2, 1]]
        results.append(
            {
                "region_key": region.key,
                "region_label": region.short_label,
                "source_label_values": list(region.source_values),
                "colour": region.colour,
                "voxel_count": voxel_count,
                "segmented_volume_ml": voxel_count * voxel_volume_ml,
                "vertices": vertices_head_m.tolist(),
                "faces": faces.tolist(),
            }
        )
    return results


def import_upenn_gbm_case(
    case_id: str | int | UpennGbmCase,
    region_keys: Iterable[str],
    cache_directory: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Download, validate, and surface selected regions for one case."""
    case = find_upenn_gbm_case(case_id)
    if not case.is_importable:
        raise UpennGbmImportError(
            f"{case.case_id} is incompatible: {case.compatibility_detail}"
        )
    path = download_upenn_gbm_segmentation(
        case,
        cache_directory,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    try:
        segmentation, affine = read_nifti_segmentation(path)
    except UpennGbmImportError:
        # A stale/incomplete cache entry should heal itself once before the
        # error reaches the user.
        path.unlink(missing_ok=True)
        path = download_upenn_gbm_segmentation(
            case,
            cache_directory,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            force=True,
        )
        segmentation, affine = read_nifti_segmentation(path)
    meshes = build_upenn_gbm_region_meshes(
        case,
        segmentation,
        affine,
        region_keys,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    return {
        "case_id": case.case_id,
        "short_id": case.short_id,
        "source_filename": case.filename,
        "source_url": case.download_url,
        "dataset_url": UPENN_GBM_COLLECTION_URL,
        "dataset_doi": UPENN_GBM_DATASET_DOI,
        "release_url": UPENN_GBM_RELEASE_URL,
        "license": UPENN_GBM_LICENSE,
        "coordinate_frame": "NAS/LPA/RPA head frame",
        "source_coordinate_frame": "UPENN-GBM preprocessed SRI24 RAS millimetres",
        "meshes": meshes,
    }
