from __future__ import annotations

import json
import math
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

import magpylib as magpy
import numpy as np
from scipy.constants import mu_0
from scipy.spatial.transform import Rotation

from fieldworkbench.studio_adapter import StudioAdapter


FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")


def circular_loop_axis_field_t(
    radius_m: float,
    ampere_turns: float,
    axial_offset_m: np.ndarray | list[float] | float,
) -> np.ndarray:
    """Independent closed-form B magnitude for a thin circular loop on axis."""
    z = np.asarray(axial_offset_m, dtype=float)
    return (
        mu_0
        * float(ampere_turns)
        * float(radius_m) ** 2
        / (2.0 * (float(radius_m) ** 2 + z**2) ** 1.5)
    )


def field_vectors(adapter: StudioAdapter, points_m: np.ndarray) -> np.ndarray:
    values = np.asarray(adapter.field_fingerprint(points_m.tolist()), dtype=float)
    return values.reshape(-1, 3)


def add_circle(
    adapter: StudioAdapter,
    *,
    diameter_m: float = 0.1,
    turns: int = 1,
    drive_current_a: float = 1.0,
    field_scale_factor: float = 1.0,
    position_m: list[float] | None = None,
    orientation_rotvec_deg: list[float] | None = None,
) -> str:
    object_id = adapter.add_template("circle")
    operations = [
        {
            "method": "set_param",
            "params": {
                "object_id": object_id,
                "name": "diameter",
                "value": float(diameter_m),
            },
        },
        {
            "method": "set_coil_physical",
            "params": {
                "object_id": object_id,
                "turns": int(turns),
                "drive_current_a": float(drive_current_a),
                "field_scale_factor": float(field_scale_factor),
            },
        },
    ]
    if position_m is not None or orientation_rotvec_deg is not None:
        operations.append(
            {
                "method": "set_transform",
                "params": {
                    "object_id": object_id,
                    "position": list(position_m or [0.0, 0.0, 0.0]),
                    "orientation": list(
                        orientation_rotvec_deg or [0.0, 0.0, 0.0]
                    ),
                },
            }
        )
    adapter.apply_operations(operations)
    return object_id


def direct_magpylib_source(adapter: StudioAdapter, object_id: str):
    """Rebuild a supported Workbench coil directly with Magpylib's public API."""
    object_type = adapter.get_object(object_id)["type"]
    params = {
        item["name"]: item["value"] for item in adapter.get_params(object_id)
    }
    transform = adapter.get_transform(object_id)
    position = np.asarray(transform.get("position", [0.0, 0.0, 0.0]), dtype=float)
    if position.ndim > 1:
        position = position[-1]
    euler = np.asarray(transform.get("euler", [0.0, 0.0, 0.0]), dtype=float)
    if euler.ndim > 1:
        euler = euler[-1]
    common = {
        "current": float(params["current"]),
        "position": position,
        "orientation": Rotation.from_euler("xyz", euler, degrees=True),
    }
    if object_type == "current.Circle":
        return magpy.current.Circle(diameter=float(params["diameter"]), **common)
    if object_type == "current.Polyline":
        return magpy.current.Polyline(
            vertices=np.asarray(params["vertices"], dtype=float), **common
        )
    raise AssertionError(f"Unsupported parity source: {object_type}")


class DependencyPinTests(unittest.TestCase):
    """Make every result identify the exact calculation engines under test."""

    def test_installed_engines_match_requirements(self):
        self.assertEqual(version("magpylib"), "5.2.3")
        self.assertEqual(version("magpylib-studio"), "0.1.2")


class IndependentPhysicsTests(unittest.TestCase):
    """Compare Workbench results with equations that do not call Magpylib."""

    def setUp(self):
        self.adapter = StudioAdapter({"objects": []})

    def test_helmholtz_pair_matches_closed_form_at_and_around_centre(self):
        radius_m = 0.1
        turns = 80
        drive_current_a = 0.035
        ampere_turns = turns * drive_current_a
        add_circle(
            self.adapter,
            diameter_m=2.0 * radius_m,
            turns=turns,
            drive_current_a=drive_current_a,
            position_m=[0.0, 0.0, -radius_m / 2.0],
        )
        add_circle(
            self.adapter,
            diameter_m=2.0 * radius_m,
            turns=turns,
            drive_current_a=drive_current_a,
            position_m=[0.0, 0.0, radius_m / 2.0],
        )

        z_m = np.asarray([-0.025, 0.0, 0.025, 0.075])
        points = np.column_stack([np.zeros_like(z_m), np.zeros_like(z_m), z_m])
        expected_bz = sum(
            circular_loop_axis_field_t(radius_m, ampere_turns, z_m - coil_z)
            for coil_z in (-radius_m / 2.0, radius_m / 2.0)
        )
        values = field_vectors(self.adapter, points)

        np.testing.assert_allclose(values[:, :2], 0.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(values[:, 2], expected_bz, rtol=1e-9, atol=1e-14)
        expected_centre = (
            mu_0 * ampere_turns / radius_m * (4.0 / 5.0) ** 1.5
        )
        self.assertAlmostEqual(values[1, 2], expected_centre, places=14)

    def test_translation_and_rotation_preserve_the_axis_solution(self):
        centre = np.asarray([0.03, -0.02, 0.01])
        radius_m = 0.05
        turns = 25
        current_a = 0.08
        coil_id = add_circle(
            self.adapter,
            diameter_m=2.0 * radius_m,
            turns=turns,
            drive_current_a=current_a,
            position_m=centre.tolist(),
            orientation_rotvec_deg=[0.0, 60.0, 0.0],
        )
        transform = self.adapter.get_transform(coil_id)
        normal = Rotation.from_euler(
            "xyz", transform["euler"], degrees=True
        ).apply([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            normal, [math.sqrt(3.0) / 2.0, 0.0, 0.5], atol=1e-12
        )

        offsets = np.asarray([0.0, 0.02, 0.05])
        points = centre + offsets[:, None] * normal
        expected_magnitude = circular_loop_axis_field_t(
            radius_m, turns * current_a, offsets
        )
        expected = expected_magnitude[:, None] * normal
        np.testing.assert_allclose(
            field_vectors(self.adapter, points), expected, rtol=1e-9, atol=1e-14
        )

    def test_current_reversal_superposition_and_cancellation_are_linear(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025], [0.012, -0.008, 0.035]]
        )
        coil_id = add_circle(
            self.adapter, turns=20, drive_current_a=0.05
        )
        positive = field_vectors(self.adapter, points)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": coil_id, "drive_current_a": -0.05},
                }
            ]
        )
        reversed_field = field_vectors(self.adapter, points)
        np.testing.assert_allclose(reversed_field, -positive, rtol=1e-12, atol=1e-15)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": coil_id, "drive_current_a": 0.05},
                }
            ]
        )
        second_id = self.adapter.duplicate(coil_id)
        doubled = field_vectors(self.adapter, points)
        np.testing.assert_allclose(doubled, 2.0 * positive, rtol=1e-12, atol=1e-15)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": second_id, "drive_current_a": -0.05},
                }
            ]
        )
        cancelled = field_vectors(self.adapter, points)
        np.testing.assert_allclose(cancelled, 0.0, rtol=0.0, atol=1e-15)

    def test_b_h_millimetre_and_microtesla_conversions_are_consistent(self):
        add_circle(self.adapter, turns=10, drive_current_a=0.03)
        point_m = np.asarray([0.0, 0.0, 0.025])
        vectorized_b = field_vectors(self.adapter, point_m.reshape(1, 3))[0]
        probe_b = np.asarray(
            self.adapter.field_at_mm((point_m * 1000.0).tolist(), "B")["values"][0],
            dtype=float,
        )
        probe_h = np.asarray(
            self.adapter.field_at_mm((point_m * 1000.0).tolist(), "H")["values"][0],
            dtype=float,
        )
        np.testing.assert_allclose(probe_b, vectorized_b, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(probe_b, mu_0 * probe_h, rtol=1e-9, atol=1e-14)

        box = self.adapter.add_template("guide_box")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_param",
                    "params": {
                        "object_id": box,
                        "name": "dimension",
                        "value": [0.01, 0.01, 0.01],
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": box,
                        "position": point_m.tolist(),
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "method": "set_geometry_style",
                    "params": {"object_id": box, "role": "measurement"},
                },
                {
                    "method": "set_measurement_settings",
                    "params": {"object_id": box, "quality": "preview"},
                },
            ]
        )
        result = self.adapter.analyze_measurement(box)
        local_points = np.asarray(result["local_points_m"], dtype=float)
        centre_index = int(np.argmin(np.linalg.norm(local_points, axis=1)))
        self.assertLess(np.linalg.norm(local_points[centre_index]), 1e-15)
        np.testing.assert_allclose(
            result["values_t"][centre_index], probe_b, rtol=1e-12, atol=1e-15
        )
        self.assertAlmostEqual(
            result["magnitudes_uT"][centre_index],
            np.linalg.norm(probe_b) * 1e6,
            places=12,
        )

    def test_current_convention_changes_rms_heating_not_field_amplitude(self):
        coil_id = add_circle(self.adapter, turns=100, drive_current_a=0.07)
        point = np.asarray([[0.0, 0.0, 0.025]])
        rms_field = field_vectors(self.adapter, point)
        rms_properties = self.adapter.coil_physical_properties(coil_id)

        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {"object_id": coil_id, "current_mode": "peak_sine"},
                }
            ]
        )
        peak_field = field_vectors(self.adapter, point)
        peak_properties = self.adapter.coil_physical_properties(coil_id)

        np.testing.assert_allclose(peak_field, rms_field, rtol=1e-13, atol=1e-16)
        self.assertAlmostEqual(
            peak_properties["calculations"]["copper_loss_w"],
            rms_properties["calculations"]["copper_loss_w"] / 2.0,
        )
        self.assertAlmostEqual(
            peak_properties["calculations"]["current_density_a_mm2"],
            rms_properties["calculations"]["current_density_a_mm2"]
            / math.sqrt(2.0),
        )
        self.assertAlmostEqual(
            peak_properties["calculations"]["resistive_voltage_v"],
            rms_properties["calculations"]["resistive_voltage_v"],
        )


class DirectMagpylibParityTests(unittest.TestCase):
    """Compare Workbench integration with direct public Magpylib calls."""

    def setUp(self):
        self.adapter = StudioAdapter({"objects": []})
        self.points = np.asarray(
            [
                [-0.04, 0.015, 0.025],
                [0.0, 0.0, 0.04],
                [0.025, -0.03, -0.02],
                [0.07, 0.02, 0.06],
                [-0.015, -0.055, 0.01],
            ]
        )

    def assert_parity(self, object_ids: list[str]) -> None:
        sources = [direct_magpylib_source(self.adapter, item) for item in object_ids]
        direct_source = sources[0] if len(sources) == 1 else magpy.Collection(*sources)
        expected = np.asarray(direct_source.getB(self.points), dtype=float).reshape(-1, 3)
        actual = field_vectors(self.adapter, self.points)
        # Studio exposes rotations as Euler angles after accepting a rotation
        # vector. Reconstructing a direct source from that public transform can
        # introduce a few femtotesla of round-trip noise for a polyline.
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=5e-15)

    def test_transformed_corrected_circle_matches_direct_magpylib_vectors(self):
        coil_id = add_circle(
            self.adapter,
            diameter_m=0.12,
            turns=48,
            drive_current_a=0.027,
            field_scale_factor=1.75,
            position_m=[0.015, -0.01, 0.02],
            orientation_rotvec_deg=[18.0, -27.0, 11.0],
        )
        self.assert_parity([coil_id])

    def test_transformed_polyline_matches_direct_magpylib_vectors(self):
        coil_id = self.adapter.add_template("polyline")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": coil_id,
                        "turns": 40,
                        "drive_current_a": 0.05,
                        "field_scale_factor": 1.3,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": coil_id,
                        "position": [-0.012, 0.018, -0.008],
                        "orientation": [-14.0, 22.0, 9.0],
                    },
                },
            ]
        )
        self.assert_parity([coil_id])

    def test_mixed_source_sum_matches_direct_magpylib_collection(self):
        circle_id = add_circle(
            self.adapter,
            diameter_m=0.09,
            turns=30,
            drive_current_a=0.04,
            position_m=[0.02, 0.0, -0.025],
            orientation_rotvec_deg=[0.0, 35.0, 0.0],
        )
        polyline_id = self.adapter.add_template("polyline")
        self.adapter.apply_operations(
            [
                {
                    "method": "set_coil_physical",
                    "params": {
                        "object_id": polyline_id,
                        "turns": 22,
                        "drive_current_a": -0.031,
                        "field_scale_factor": 0.85,
                    },
                },
                {
                    "method": "set_transform",
                    "params": {
                        "object_id": polyline_id,
                        "position": [-0.025, 0.015, 0.03],
                        "orientation": [12.0, 0.0, -20.0],
                    },
                },
            ]
        )
        self.assert_parity([circle_id, polyline_id])


class SavedReferenceSceneTests(unittest.TestCase):
    """Keep representative saved scenes and their complete B-vector grids stable."""

    def test_saved_reference_scene_vectors_and_roundtrip(self):
        manifest = json.loads(
            (FIXTURE_DIRECTORY / "reference_fields.json").read_text(encoding="utf-8")
        )
        for case in manifest["cases"]:
            with self.subTest(case=case["name"]):
                scene_path = FIXTURE_DIRECTORY / case["scene"]
                adapter = StudioAdapter(
                    json.loads(scene_path.read_text(encoding="utf-8"))
                )
                points = np.asarray(case["points_m"], dtype=float)
                expected = np.asarray(case["expected_b_t"], dtype=float)
                actual = field_vectors(adapter, points)
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-9, atol=1e-14
                )

                with tempfile.TemporaryDirectory() as directory:
                    roundtrip_path = Path(directory) / scene_path.name
                    adapter.save_file(roundtrip_path)
                    reopened = StudioAdapter({"objects": []})
                    reopened.load_file(roundtrip_path)
                    np.testing.assert_array_equal(
                        field_vectors(reopened, points), actual
                    )


if __name__ == "__main__":
    unittest.main()
