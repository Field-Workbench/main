"""Project-context/resource-registry tests that do not require Qt or Magpylib."""

from __future__ import annotations

import gc
import unittest
from pathlib import Path

from fieldworkbench.project_context import (
    ProjectContext,
    ProjectResourceRegistry,
    RESOURCE_DATASET,
    RESOURCE_EXPOSURE,
    RESOURCE_EXPOSURE_ANALYSIS,
    RESOURCE_MODEL_OBJECT,
    RESOURCE_SAVED_MAP,
    RESOURCE_SNAPSHOT,
    RESOURCE_WAVEFORM,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.version = 1
        self.objects = [
            {"id": "coil_1", "label": "Drive coil", "type": "current.Circle"},
            {"id": "brain", "label": "Brain", "type": "workbench.Mesh"},
        ]
        self.snapshots = [
            {"id": "snap_1", "name": "Guide sphere", "status": "current", "source_object_id": "brain"}
        ]
        self.saved_maps = [
            {"id": "map_1", "name": "Centre field", "kind": "static", "view": "2d"},
            {"id": "map_2", "name": "Brain playback", "kind": "scripted", "view": "brain"},
        ]
        self.exposures = [
            {"id": "exp_1", "name": "SAM360"},
            {"id": "exp_2", "name": "Sine"},
        ]
        self.exposure_analyses = [
            {
                "id": "analysis_1",
                "name": "SAM360 — Full exposure",
                "exposure_id": "exp_1",
                "target_id": "brain",
                "payload_b64": "opaque",
            }
        ]
        self.datasets = [
            {"id": "data_1", "name": "FGM3D grid"},
        ]
        self.waveforms = [
            {"id": "wave_1", "name": "Project waveform", "quantity": "current", "unit": "mA", "sample_count": 3, "duration_s": 0.002},
        ]
        self.exposure_payloads = {
            "exp_1": {
                "id": "exp_1",
                "protocol": {
                    "sequence": {
                        "loop": False,
                        "steps": [{
                            "id": "step_1",
                            "duration_s": 1.0,
                            "drive_groups": [{
                                "id": "drive_1",
                                "waveform_id": "wave_1",
                                "coil_ids": ["coil_1"],
                                "connection": "series",
                            }],
                        }],
                    }
                },
            },
            "exp_2": {"id": "exp_2", "protocol": {"sequence": {"loop": False, "steps": []}}},
        }

    def signature(self) -> str:
        return f"sig-{self.version}"

    def list_objects(self):
        return [dict(item) for item in self.objects]

    def object_type_label(self, object_id: str) -> str:
        return "Circular coil" if object_id == "coil_1" else "Imported mesh"

    def list_snapshots(self):
        return [dict(item) for item in self.snapshots]

    def list_saved_maps(self):
        return [dict(item) for item in self.saved_maps]

    def list_exposures(self):
        return [dict(item) for item in self.exposures]

    def exposure(self, exposure_id: str):
        return self.exposure_payloads[exposure_id]

    def list_measurement_datasets(self):
        return [dict(item) for item in self.datasets]

    def list_waveforms(self):
        return [dict(item) for item in self.waveforms]


class ProjectContextTests(unittest.TestCase):
    def test_registry_uses_namespaced_native_resource_ids(self) -> None:
        adapter = FakeAdapter()
        registry = ProjectResourceRegistry(adapter)

        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_MODEL_OBJECT)],
            ["model_object:coil_1", "model_object:brain"],
        )
        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_SNAPSHOT)],
            ["snapshot:snap_1"],
        )
        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_SAVED_MAP)],
            ["saved_map:map_1", "saved_map:map_2"],
        )
        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_EXPOSURE)],
            ["exposure:exp_1", "exposure:exp_2"],
        )
        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_EXPOSURE_ANALYSIS)],
            ["exposure_analysis:analysis_1"],
        )
        self.assertEqual(
            [item.key for item in registry.by_kind(RESOURCE_DATASET)],
            ["dataset:data_1"],
        )

    def test_only_first_class_project_waveforms_are_addressable(self) -> None:
        registry = ProjectResourceRegistry(FakeAdapter())
        waveforms = registry.by_kind(RESOURCE_WAVEFORM)
        self.assertEqual(len(waveforms), 1)
        self.assertEqual(waveforms[0].key, "waveform:wave_1")
        self.assertTrue(waveforms[0].persistent)

    def test_resource_signature_changes_when_project_resources_change(self) -> None:
        adapter = FakeAdapter()
        registry = ProjectResourceRegistry(adapter)
        before = registry.signature()
        adapter.datasets.append({"id": "data_2", "name": "Second grid"})
        after = registry.signature()
        self.assertNotEqual(before, after)
        self.assertEqual(registry.counts()[RESOURCE_DATASET], 2)

    def test_context_owns_project_identity_path_and_save_baseline(self) -> None:
        adapter = FakeAdapter()
        context = ProjectContext(adapter, untitled_number=2)
        self.assertEqual(context.display_name(), "Untitled 2")
        self.assertFalse(context.is_dirty())

        adapter.version = 2
        self.assertTrue(context.is_dirty())
        context.mark_saved(Path("example.magpy.json"))
        self.assertFalse(context.is_dirty())
        self.assertEqual(context.display_name(), "example.magpy.json")

    def test_unsaved_context_can_use_a_suggested_display_label(self) -> None:
        context = ProjectContext(FakeAdapter(), untitled_number=3, display_label="Optimizer finalist 2")
        self.assertEqual(context.display_name(), "Optimizer finalist 2")
        context.mark_saved(Path("optimized.magpy.json"))
        self.assertEqual(context.display_name(), "optimized.magpy.json")

    def test_context_tracks_windows_weakly_and_tags_project_identity(self) -> None:
        context = ProjectContext(FakeAdapter())

        class Window:
            pass

        window = Window()
        context.register_window("maps_analysis", window)
        self.assertEqual(context.windows("maps_analysis"), (window,))
        self.assertIs(window.project_context, context)
        self.assertEqual(window.project_id, context.project_id)
        self.assertEqual(window.project_role, "maps_analysis")

        del window
        gc.collect()
        self.assertEqual(context.windows("maps_analysis"), ())


if __name__ == "__main__":
    unittest.main()
