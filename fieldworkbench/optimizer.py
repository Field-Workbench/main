"""Reusable genetic-search primitives for Field Workbench.

The search runner deliberately knows nothing about Magpylib or Qt.  The
application adapter supplies coil generation, field evaluation, and final DRC
callbacks; this module owns deterministic evolution, Pareto filtering, and
near-duplicate removal. The adapter supplies either generated equal-volume
samples or exact user-defined points through the same target boundary.
"""

from __future__ import annotations

import copy
import math
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares


class OptimizationCancelled(RuntimeError):
    """Raised cooperatively when the user cancels a running search."""


@dataclass
class VectorFieldStamp:
    """Interpolated coil-local vector field reusable under rigid transforms.

    ``vectors_t_per_a`` stores the complete local ``(Bx, By, Bz)`` response on
    three monotonically increasing local-coordinate axes.  The field can then be
    sampled at arbitrary world-space points after translating/rotating the source
    without another electromagnetic solve.

    ``geometric_scale`` additionally supports a *uniformly* scaled copy of the
    source geometry.  Magnetostatic Biot--Savart fields obey

        B_s(r) = B_1(r/s) / s

    for the same current when every source length is scaled by ``s``.  Workbench
    uses this exact relation for centreline circular-loop diameter changes; it is
    intentionally not applied to non-similar shape edits such as independent
    racetrack straight/end dimensions.
    """

    axes_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    vectors_t_per_a: np.ndarray

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=float).reshape(-1) for axis in self.axes_m)
        if len(axes) != 3 or any(len(axis) < 2 for axis in axes):
            raise ValueError("A vector-field stamp requires three axes with at least two samples each.")
        if any(not np.all(np.diff(axis) > 0.0) for axis in axes):
            raise ValueError("Vector-field stamp axes must be strictly increasing.")
        values = np.asarray(self.vectors_t_per_a, dtype=float)
        expected = (len(axes[0]), len(axes[1]), len(axes[2]), 3)
        if values.shape != expected:
            raise ValueError(
                f"Vector-field stamp values have shape {values.shape}; expected {expected}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Vector-field stamp values must be finite.")
        self.axes_m = axes
        self.vectors_t_per_a = values
        self._interpolator = RegularGridInterpolator(
            axes,
            values,
            bounds_error=False,
            fill_value=np.nan,
        )

    def sample_world(
        self,
        points_m: np.ndarray,
        *,
        position_m: np.ndarray,
        rotation_matrix: np.ndarray,
        geometric_scale: float = 1.0,
    ) -> np.ndarray:
        """Return world-space vectors at ``points_m`` for one ampere of drive.

        The stored field is expressed in the source's local frame.  Query points
        are transformed into that frame, interpolation happens there, and the
        resulting vectors are rotated back to world coordinates.  A uniform
        geometry scale uses the exact magnetostatic similarity relation described
        in the class docstring.
        """
        points = np.asarray(points_m, dtype=float).reshape(-1, 3)
        position = np.asarray(position_m, dtype=float).reshape(3)
        rotation = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
        scale = float(geometric_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Vector-field stamp geometric scale must be positive.")
        if not np.isfinite(points).all() or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Vector-field stamp transform inputs must be finite.")

        # Row-vector convention: world = local @ R.T + position, therefore
        # local = (world - position) @ R.
        local_points = (points - position) @ rotation
        local_vectors = np.asarray(self._interpolator(local_points / scale), dtype=float)
        if not np.isfinite(local_vectors).all():
            raise ValueError("Vector-field stamp query fell outside the cached volume.")
        local_vectors /= scale
        return local_vectors @ rotation.T


MIN_OPTIMIZER_WORKERS = 1
MAX_OPTIMIZER_WORKERS = 32
MAX_OPTIMIZER_BENCHMARK_WORKERS = 32


def _clean_cpu_description(value: Any) -> str:
    """Collapse a platform CPU brand string into one compact report line."""
    return " ".join(str(value or "").replace("\x00", " ").split())


def reported_cpu_description() -> str:
    """Return the shortest useful CPU brand description available locally.

    Workbench optimizer reports only need a human-readable machine fingerprint,
    not a full hardware inventory.  Prefer the OS-provided processor brand string
    on each supported desktop platform and fall back to Python's platform helpers.
    """

    system = platform.system().lower()

    if system == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
                fallback = ""
                for raw_line in handle:
                    key, separator, value = raw_line.partition(":")
                    if not separator:
                        continue
                    key = key.strip().lower()
                    cleaned = _clean_cpu_description(value)
                    if not cleaned:
                        continue
                    if key == "model name":
                        return cleaned
                    if key in {"hardware", "model"} and not fallback:
                        fallback = cleaned
                if fallback:
                    return fallback
        except OSError:
            pass

    elif system == "windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
            cleaned = _clean_cpu_description(value)
            if cleaned:
                return cleaned
        except (ImportError, OSError):
            pass
        cleaned = _clean_cpu_description(os.environ.get("PROCESSOR_IDENTIFIER"))
        if cleaned:
            return cleaned

    elif system == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            cleaned = _clean_cpu_description(completed.stdout)
            if completed.returncode == 0 and cleaned:
                return cleaned
        except (OSError, subprocess.SubprocessError):
            pass

    for value in (platform.processor(), platform.machine()):
        cleaned = _clean_cpu_description(value)
        if cleaned:
            return cleaned
    return "Unknown CPU"


def reported_logical_cpu_count() -> int:
    """Return the logical CPU count reported by the operating system."""
    return max(MIN_OPTIMIZER_WORKERS, int(os.cpu_count() or 1))


def available_optimizer_workers() -> int:
    """Return the supported optimizer process-worker ceiling.

    Process pools are allowed to oversubscribe the machine deliberately: some
    field workloads can still benefit from more worker processes than reported
    logical CPUs because the native numerical kernels do not occupy a Python
    process uniformly.  The benchmark is the preferred way to discover the
    useful count for a particular system.
    """
    return MAX_OPTIMIZER_WORKERS


def default_optimizer_worker_count() -> int:
    """Use modest concurrency until the machine-specific benchmark is run."""
    return min(2, reported_logical_cpu_count(), available_optimizer_workers())


DEFAULT_OPTIMIZER_WORKER_COUNT = default_optimizer_worker_count()


def normalize_optimizer_worker_count(value: Any) -> int:
    """Validate a persisted/public optimizer worker count."""
    try:
        workers = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Optimizer worker count must be an integer.") from error
    supported = available_optimizer_workers()
    if not MIN_OPTIMIZER_WORKERS <= workers <= supported:
        raise ValueError(
            f"Optimizer worker count must be between {MIN_OPTIMIZER_WORKERS} "
            f"and {supported}."
        )
    return workers


def optimizer_benchmark_worker_counts(max_workers: int | None = None) -> list[int]:
    """Return the bounded worker counts used by the representative benchmark."""
    limit = MAX_OPTIMIZER_BENCHMARK_WORKERS if max_workers is None else int(max_workers)
    limit = max(
        MIN_OPTIMIZER_WORKERS,
        min(MAX_OPTIMIZER_BENCHMARK_WORKERS, available_optimizer_workers(), limit),
    )
    return list(range(MIN_OPTIMIZER_WORKERS, limit + 1))


def recommend_optimizer_worker_count(results: list[dict[str, Any]]) -> int:
    """Choose the smallest count within 8% of the benchmark's fastest result."""
    valid = [
        item
        for item in results
        if int(item.get("workers", 0)) >= MIN_OPTIMIZER_WORKERS
        and math.isfinite(float(item.get("elapsed_seconds", math.inf)))
        and float(item.get("elapsed_seconds", math.inf)) > 0.0
    ]
    if not valid:
        return MIN_OPTIMIZER_WORKERS
    fastest = min(float(item["elapsed_seconds"]) for item in valid)
    near_fastest = [
        item
        for item in valid
        if float(item["elapsed_seconds"]) <= fastest * 1.08
    ]
    return min(int(item["workers"]) for item in near_fastest)


def default_optimizer_settings() -> dict[str, Any]:
    """Return conservative first-release settings for a 1--4 coil search."""
    return {
        "target_object_id": "",
        "anatomy_object_id": None,
        "direction_mode": "magnitude",
        "coil_count_optimize": True,
        "coil_count_fixed": 2,
        "coil_count_min": 1,
        "coil_count_max": 4,
        "placement_optimize": True,
        "shell_radius_min_mm": 95.0,
        "shell_radius_max_mm": 145.0,
        "stand_off_mm": 5.0,
        "tilt_limit_deg": 20.0,
        "diameter_optimize": True,
        "diameter_fixed_mm": 100.0,
        "diameter_min_mm": 35.0,
        "diameter_max_mm": 220.0,
        "turns_optimize": True,
        "turns_fixed": 300,
        "turns_min": 50,
        "turns_max": 700,
        "awg_optimize": True,
        "awg_fixed": 30,
        "awg_min": 26,
        "awg_max": 36,
        "axial_width_optimize": True,
        "axial_width_fixed_mm": 8.0,
        "axial_width_min_mm": 4.0,
        "axial_width_max_mm": 18.0,
        "max_current_a": 0.07,
        "max_voltage_v": 12.0,
        "max_power_per_coil_w": 2.0,
        "max_total_power_w": 6.0,
        "max_current_density_a_mm2": 3.0,
        "max_total_mass_g": 1000.0,
        "minimum_in_band_pct": 70.0,
        "population": 24,
        "generations": 12,
        "finalist_count": 8,
        "seed": 20260802,
        "mutation_rate": 0.24,
        "coarse_quality": "preview",
        "final_quality": "standard",
        "finite_pack_order": 3,
        "worker_count": default_optimizer_worker_count(),
    }


def validate_optimizer_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the public settings dictionary and reject unsafe ranges."""
    settings = default_optimizer_settings()
    settings.update(copy.deepcopy(raw))
    bool_names = (
        "coil_count_optimize",
        "placement_optimize",
        "diameter_optimize",
        "turns_optimize",
        "awg_optimize",
        "axial_width_optimize",
    )
    int_names = (
        "coil_count_fixed",
        "coil_count_min",
        "coil_count_max",
        "turns_fixed",
        "turns_min",
        "turns_max",
        "awg_fixed",
        "awg_min",
        "awg_max",
        "population",
        "generations",
        "finalist_count",
        "seed",
        "finite_pack_order",
        "worker_count",
    )
    float_names = (
        "shell_radius_min_mm",
        "shell_radius_max_mm",
        "stand_off_mm",
        "tilt_limit_deg",
        "diameter_fixed_mm",
        "diameter_min_mm",
        "diameter_max_mm",
        "axial_width_fixed_mm",
        "axial_width_min_mm",
        "axial_width_max_mm",
        "max_current_a",
        "max_voltage_v",
        "max_power_per_coil_w",
        "max_total_power_w",
        "max_current_density_a_mm2",
        "max_total_mass_g",
        "minimum_in_band_pct",
        "mutation_rate",
    )
    try:
        for name in bool_names:
            value = settings[name]
            if isinstance(value, str):
                if value.strip().lower() not in {"true", "false"}:
                    raise ValueError(f"{name} must be true or false")
                value = value.strip().lower() == "true"
            settings[name] = bool(value)
        for name in int_names:
            settings[name] = int(settings[name])
        for name in float_names:
            settings[name] = float(settings[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid optimizer setting: {error}") from error

    settings["target_object_id"] = str(settings.get("target_object_id", ""))
    anatomy = settings.get("anatomy_object_id")
    settings["anatomy_object_id"] = None if anatomy in {None, "", "none"} else str(anatomy)
    settings["direction_mode"] = str(settings.get("direction_mode", "magnitude")).lower()
    if settings["direction_mode"] not in {"magnitude", "x", "y", "z"}:
        raise ValueError("Target direction must be magnitude, X, Y, or Z.")

    for low, high, label in (
        ("coil_count_min", "coil_count_max", "coil count"),
        ("turns_min", "turns_max", "turn count"),
        ("awg_min", "awg_max", "AWG"),
        ("shell_radius_min_mm", "shell_radius_max_mm", "search radius"),
        ("diameter_min_mm", "diameter_max_mm", "mean diameter"),
        ("axial_width_min_mm", "axial_width_max_mm", "winding width"),
    ):
        if settings[low] > settings[high]:
            raise ValueError(f"Minimum {label} cannot exceed its maximum.")
    if not 1 <= settings["coil_count_min"] <= settings["coil_count_max"] <= 4:
        raise ValueError("This optimizer supports between one and four coils.")
    if not 1 <= settings["coil_count_fixed"] <= 4:
        raise ValueError("Fixed coil count must be between one and four.")
    if not 1 <= settings["turns_min"] <= settings["turns_max"] <= 1_000_000:
        raise ValueError("Turn limits must be between 1 and 1,000,000.")
    if not 1 <= settings["turns_fixed"] <= 1_000_000:
        raise ValueError("Fixed turns must be between 1 and 1,000,000.")
    if not 4 <= settings["awg_min"] <= settings["awg_max"] <= 50:
        raise ValueError("AWG limits must be between 4 and 50.")
    if not 4 <= settings["awg_fixed"] <= 50:
        raise ValueError("Fixed AWG must be between 4 and 50.")
    positive = (
        "shell_radius_min_mm",
        "shell_radius_max_mm",
        "diameter_fixed_mm",
        "diameter_min_mm",
        "diameter_max_mm",
        "axial_width_fixed_mm",
        "axial_width_min_mm",
        "axial_width_max_mm",
        "max_current_a",
        "max_voltage_v",
        "max_power_per_coil_w",
        "max_total_power_w",
        "max_current_density_a_mm2",
        "max_total_mass_g",
    )
    if any(not math.isfinite(settings[name]) or settings[name] <= 0 for name in positive):
        raise ValueError("Optimizer dimensions and hard limits must be finite and positive.")
    if not 0 <= settings["stand_off_mm"] <= 1e6:
        raise ValueError("Stand-off must be between 0 and 1,000,000 mm.")
    if not 0 <= settings["tilt_limit_deg"] <= 90:
        raise ValueError("Tilt limit must be between 0 and 90 degrees.")
    if not 0 <= settings["minimum_in_band_pct"] <= 100:
        raise ValueError("Minimum target coverage must be between 0 and 100 percent.")
    if not 8 <= settings["population"] <= 1000:
        raise ValueError("Population must be between 8 and 1,000.")
    if not 1 <= settings["generations"] <= 10_000:
        raise ValueError("Generations must be between 1 and 10,000.")
    if not 1 <= settings["finalist_count"] <= 50:
        raise ValueError("Finalist count must be between 1 and 50.")
    if not 0.01 <= settings["mutation_rate"] <= 1.0:
        raise ValueError("Mutation rate must be between 0.01 and 1.0.")
    if settings["finite_pack_order"] not in {1, 3, 5}:
        raise ValueError("Finite winding-pack order must be 1, 3, or 5.")
    settings["worker_count"] = normalize_optimizer_worker_count(
        settings["worker_count"]
    )
    if str(settings["coarse_quality"]) not in {"preview", "standard"}:
        raise ValueError("Coarse sampling must be Preview or Standard.")
    if str(settings["final_quality"]) not in {"preview", "standard", "fine"}:
        raise ValueError("Final sampling must be Preview, Standard, or Fine.")
    return settings


def field_statistics(
    vectors_t: np.ndarray,
    target_uT: float,
    tolerance_pct: float,
) -> dict[str, float | int | list[float]]:
    """Calculate the optimizer's field and target-coverage statistics."""
    vectors = np.asarray(vectors_t, dtype=float).reshape(-1, 3)
    magnitudes = np.linalg.norm(vectors, axis=1) * 1e6
    finite = np.isfinite(vectors).all(axis=1) & np.isfinite(magnitudes)
    values = magnitudes[finite]
    valid_vectors = vectors[finite]
    if not len(values):
        raise ValueError("No finite target-field samples were returned.")
    percentiles = np.percentile(values, [5, 50, 95])
    mean = float(np.mean(values))
    std = float(np.std(values))
    low = float(target_uT) * (1.0 - float(tolerance_pct) / 100.0)
    high = float(target_uT) * (1.0 + float(tolerance_pct) / 100.0)
    in_band = (values >= low) & (values <= high)
    mean_vector = np.mean(valid_vectors, axis=0) * 1e6
    vector_magnitudes = np.linalg.norm(valid_vectors, axis=1)
    direction_mask = vector_magnitudes > max(1e-18, float(np.max(vector_magnitudes)) * 1e-12)
    if np.any(direction_mask):
        units = valid_vectors[direction_mask] / vector_magnitudes[direction_mask, None]
        directional_consistency_pct = 100.0 * float(np.linalg.norm(np.mean(units, axis=0)))
    else:
        directional_consistency_pct = 0.0
    return {
        "mean_uT": mean,
        "min_uT": float(np.min(values)),
        "max_uT": float(np.max(values)),
        "p5_uT": float(percentiles[0]),
        "median_uT": float(percentiles[1]),
        "p95_uT": float(percentiles[2]),
        "std_uT": std,
        "coefficient_variation_pct": 100.0 * std / abs(mean) if mean else math.inf,
        "uniformity_spread_pct": (
            100.0 * float(percentiles[2] - percentiles[0]) / abs(mean)
            if mean
            else math.inf
        ),
        "rms_target_error_uT": float(np.sqrt(np.mean((values - target_uT) ** 2))),
        "mean_absolute_target_error_uT": float(np.mean(np.abs(values - target_uT))),
        "max_target_error_uT": float(np.max(np.abs(values - target_uT))),
        "in_band_pct": 100.0 * float(np.mean(in_band)),
        "below_target_pct": 100.0 * float(np.mean(values < low)),
        "above_target_pct": 100.0 * float(np.mean(values > high)),
        "target_uT": float(target_uT),
        "tolerance_pct": float(tolerance_pct),
        "target_low_uT": low,
        "target_high_uT": high,
        "mean_vector_uT": mean_vector.tolist(),
        "directional_consistency_pct": directional_consistency_pct,
        "sample_count": int(len(vectors)),
        "valid_sample_count": int(len(values)),
    }


def fit_signed_currents(
    basis_t_per_a: np.ndarray,
    target_uT: float,
    current_limits_a: np.ndarray,
    direction_mode: str = "magnitude",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit bounded signed drive currents for one candidate coil array.

    Signed current is the first release's explicit 0/180-degree phase model.
    Magnitude mode minimizes ``|B|`` error.  Directional modes fit the complete
    vector against a field along the requested world axis.
    """
    basis = np.asarray(basis_t_per_a, dtype=float)
    if basis.ndim != 3 or basis.shape[2] != 3:
        raise ValueError("Field basis must have shape coils × samples × 3.")
    limits = np.asarray(current_limits_a, dtype=float).reshape(-1)
    if len(limits) != basis.shape[0] or np.any(~np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("Each coil needs one finite, positive current limit.")
    target_t = float(target_uT) * 1e-6
    residual_scale = max(abs(target_t), 1e-12)
    direction_mode = str(direction_mode).lower()

    def combined(currents: np.ndarray) -> np.ndarray:
        return np.tensordot(currents, basis, axes=(0, 0))

    if direction_mode == "magnitude":
        def residual(currents: np.ndarray) -> np.ndarray:
            return (np.linalg.norm(combined(currents), axis=1) - target_t) / residual_scale
    else:
        axis = {"x": 0, "y": 1, "z": 2}[direction_mode]
        target_vector = np.zeros(3, dtype=float)
        target_vector[axis] = target_t

        def residual(currents: np.ndarray) -> np.ndarray:
            return ((combined(currents) - target_vector) / residual_scale).reshape(-1)

    centre_basis = basis[:, len(basis[0]) // 2, :]
    strengths = np.linalg.norm(centre_basis, axis=1)
    useful = strengths > max(1e-18, float(np.max(strengths, initial=0.0)) * 1e-12)
    start = np.zeros(len(limits), dtype=float)
    if np.any(useful):
        start[useful] = np.minimum(
            limits[useful] * 0.5,
            target_t / max(float(np.sum(strengths[useful])), 1e-18),
        )
    solution = least_squares(
        residual,
        start,
        bounds=(-limits, limits),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=120,
    )
    currents = np.asarray(solution.x, dtype=float)
    vectors = combined(currents)
    return currents, vectors, {
        "success": bool(solution.success),
        "cost": float(solution.cost),
        "evaluations": int(solution.nfev),
        "message": str(solution.message),
    }


def _objective_vector(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    return (
        float(metrics["rms_target_error_uT"]),
        float(metrics["total_mass_g"]),
        float(metrics["coil_count"]),
        float(metrics["max_voltage_v"]),
        float(metrics["total_power_w"]),
        -float(metrics["in_band_pct"]),
    )


def dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Tolerance-aware Pareto dominance for validated feasible designs."""
    a = _objective_vector(first)
    b = _objective_vector(second)
    tolerances = (1e-6, 1e-3, 0.0, 1e-6, 1e-6, 1e-6)
    no_worse = all(left <= right + tol for left, right, tol in zip(a, b, tolerances, strict=True))
    better = any(left < right - tol for left, right, tol in zip(a, b, tolerances, strict=True))
    return no_worse and better


def pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic non-dominated candidates."""
    feasible = [item for item in candidates if item.get("feasible", False)]
    front = [
        candidate
        for index, candidate in enumerate(feasible)
        if not any(
            other_index != index and dominates(other, candidate)
            for other_index, other in enumerate(feasible)
        )
    ]
    return sorted(front, key=lambda item: (_objective_vector(item), str(item.get("id", ""))))


def _recipe_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    recipe = []
    for coil in candidate.get("coils", []):
        recipe.append(
            (
                round(float(coil.get("diameter_mm", 0.0)), 1),
                int(coil.get("turns", 0)),
                int(coil.get("awg", 0)),
                round(float(coil.get("axial_width_mm", 0.0)), 1),
            )
        )
    return tuple(sorted(recipe))


def filter_near_duplicates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove candidates with essentially identical recipe and outcomes."""
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (float(item.get("fitness", math.inf)), _objective_vector(item)),
    ):
        metrics = candidate["metrics"]
        duplicate = False
        for other in kept:
            other_metrics = other["metrics"]
            if _recipe_signature(candidate) != _recipe_signature(other):
                continue
            if (
                abs(float(metrics["rms_target_error_uT"]) - float(other_metrics["rms_target_error_uT"]))
                <= max(0.05, 0.01 * float(metrics["target_uT"]))
                and abs(float(metrics["total_mass_g"]) - float(other_metrics["total_mass_g"]))
                <= max(0.5, 0.01 * float(metrics["total_mass_g"]))
                and abs(float(metrics["in_band_pct"]) - float(other_metrics["in_band_pct"])) <= 1.0
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def tag_finalists(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select a readable mix of Pareto and best-in-category finalists."""
    feasible = [item for item in candidates if item.get("feasible", False)]
    if not feasible:
        return []
    for item in feasible:
        item["tags"] = []
    def identity(item: dict[str, Any]) -> str:
        return str(item.get("genome_signature") or item.get("id") or id(item))
    selectors = (
        ("Balanced", lambda item: float(item.get("fitness", math.inf))),
        ("Best uniformity", lambda item: float(item["metrics"]["rms_target_error_uT"])),
        ("Lightest", lambda item: float(item["metrics"]["total_mass_g"])),
        ("Fewest coils", lambda item: (int(item["metrics"]["coil_count"]), float(item["metrics"]["rms_target_error_uT"]))),
        ("Best coverage", lambda item: (-float(item["metrics"]["in_band_pct"]), float(item["metrics"]["total_mass_g"]))),
    )
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    for tag, key in selectors:
        winner = min(feasible, key=key)
        winner["tags"].append(tag)
        if identity(winner) not in chosen_ids:
            chosen.append(winner)
            chosen_ids.add(identity(winner))
    for item in pareto_front(feasible):
        if identity(item) not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(identity(item))
    chosen = filter_near_duplicates(chosen)
    for item in sorted(feasible, key=lambda value: float(value.get("fitness", math.inf))):
        if len(chosen) >= limit:
            break
        if identity(item) not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(identity(item))
    return chosen[:limit]


@dataclass
class SearchCallbacks:
    """Application-owned operations used by :class:`GeneticSearchRunner`."""

    create: Callable[[np.random.Generator], dict[str, Any]]
    mutate: Callable[[dict[str, Any], np.random.Generator], dict[str, Any]]
    crossover: Callable[[dict[str, Any], dict[str, Any], np.random.Generator], dict[str, Any]]
    evaluate: Callable[[dict[str, Any]], dict[str, Any]]
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    signature: Callable[[dict[str, Any]], str]


class GeneticSearchRunner:
    """Deterministic, cancellable two-stage genetic/Pareto search."""

    def __init__(
        self,
        settings: dict[str, Any],
        callbacks: SearchCallbacks,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: Any | None = None,
        pause_event: Any | None = None,
    ):
        self.settings = validate_optimizer_settings(settings)
        self.callbacks = callbacks
        self.progress = progress
        self.cancel_event = cancel_event
        self.pause_event = pause_event

    def _checkpoint(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise OptimizationCancelled("Optimization cancelled.")
        while self.pause_event is not None and self.pause_event.is_set():
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise OptimizationCancelled("Optimization cancelled.")
            time.sleep(0.05)

    @staticmethod
    def _tournament(
        evaluated: list[dict[str, Any]], rng: np.random.Generator
    ) -> dict[str, Any]:
        count = min(3, len(evaluated))
        indices = rng.choice(len(evaluated), size=count, replace=False)
        return min((evaluated[int(index)] for index in indices), key=lambda item: float(item["fitness"]))

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        rng = np.random.default_rng(self.settings["seed"])
        population = [self.callbacks.create(rng) for _ in range(self.settings["population"])]
        archive: dict[str, dict[str, Any]] = {}
        evaluated_count = 0
        feasible_count = 0
        rejected_count = 0

        for generation in range(self.settings["generations"]):
            evaluated_generation: list[dict[str, Any]] = []
            for individual_index, genome in enumerate(population):
                self._checkpoint()
                signature = self.callbacks.signature(genome)
                cached = archive.get(signature)
                if cached is None:
                    candidate = self.callbacks.evaluate(copy.deepcopy(genome))
                    candidate["genome"] = copy.deepcopy(genome)
                    candidate["genome_signature"] = signature
                    archive[signature] = candidate
                    evaluated_count += 1
                    if candidate.get("feasible", False):
                        feasible_count += 1
                    else:
                        rejected_count += 1
                else:
                    candidate = cached
                evaluated_generation.append(candidate)
                if self.progress is not None:
                    best = min(
                        archive.values(), key=lambda item: float(item.get("fitness", math.inf))
                    )
                    self.progress(
                        {
                            "phase": "search",
                            "generation": generation + 1,
                            "generations": self.settings["generations"],
                            "individual": individual_index + 1,
                            "population": self.settings["population"],
                            "evaluated": evaluated_count,
                            "feasible": feasible_count,
                            "rejected": rejected_count,
                            "best_rms_uT": float(best.get("metrics", {}).get("rms_target_error_uT", math.inf)),
                        }
                    )

            evaluated_generation.sort(key=lambda item: float(item.get("fitness", math.inf)))
            elite_count = max(2, round(0.18 * self.settings["population"]))
            next_population = [copy.deepcopy(item["genome"]) for item in evaluated_generation[:elite_count]]
            while len(next_population) < self.settings["population"]:
                first = self._tournament(evaluated_generation, rng)
                second = self._tournament(evaluated_generation, rng)
                child = self.callbacks.crossover(first["genome"], second["genome"], rng)
                child = self.callbacks.mutate(child, rng)
                next_population.append(child)
            population = next_population

        preliminary = [item for item in archive.values() if item.get("feasible", False)]
        preliminary.sort(key=lambda item: float(item.get("fitness", math.inf)))
        frontier = pareto_front(preliminary)
        validation_pool: list[dict[str, Any]] = []
        validation_signatures: set[str] = set()
        for item in frontier + preliminary:
            signature = str(item.get("genome_signature", id(item)))
            if signature not in validation_signatures:
                validation_pool.append(item)
                validation_signatures.add(signature)
            if len(validation_pool) >= max(self.settings["finalist_count"] * 4, 12):
                break

        validated: list[dict[str, Any]] = []
        for index, item in enumerate(validation_pool):
            self._checkpoint()
            if self.progress is not None:
                self.progress(
                    {
                        "phase": "validation",
                        "candidate": index + 1,
                        "candidate_count": len(validation_pool),
                        "evaluated": evaluated_count,
                        "feasible": feasible_count,
                        "rejected": rejected_count,
                    }
                )
            checked = self.callbacks.validate(copy.deepcopy(item))
            if checked.get("feasible", False):
                validated.append(checked)

        finalists = tag_finalists(validated, self.settings["finalist_count"])
        for index, finalist in enumerate(finalists, 1):
            finalist["id"] = f"finalist_{index}"
        return {
            "settings": copy.deepcopy(self.settings),
            "finalists": finalists,
            "pareto_count": len(pareto_front(validated)),
            "validated_candidate_count": len(validated),
            "generated_candidate_count": len(archive),
            "evaluated_candidate_count": evaluated_count,
            "feasible_candidate_count": feasible_count,
            "rejected_candidate_count": rejected_count,
            "elapsed_seconds": time.monotonic() - started,
            "cancelled": False,
        }
