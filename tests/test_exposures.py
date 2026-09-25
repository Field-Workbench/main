from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fieldworkbench.exposures import (
    EXPOSURE_FILE_SCHEMA,
    EXPOSURE_FILE_VERSION,
    EXPOSURE_FORMAT_VERSION,
    ExposureError,
    exposure_coil_ids,
    exposure_summary,
    exposure_waveform_ids,
    new_exposure,
    normalize_exposure,
    normalize_exposure_protocol,
    read_exposure_file,
    write_exposure_file,
)


class ExposureSchemaTests(unittest.TestCase):
    def test_new_exposure_uses_only_drive_group_protocol(self):
        exposure = new_exposure("Treatment A")
        self.assertTrue(exposure["id"].startswith("exposure_"))
        self.assertEqual(exposure["name"], "Treatment A")
        self.assertEqual(exposure["format_version"], EXPOSURE_FORMAT_VERSION)
        self.assertEqual(exposure["protocol"]["sequence"], {"loop": True, "steps": []})
        for retired in ("active_coil_ids", "drive", "waveform", "sequencer", "traces"):
            self.assertNotIn(retired, exposure["protocol"])

    def test_drive_groups_normalize_and_summarize(self):
        protocol = normalize_exposure_protocol(
            {
                "sequence": {
                    "loop": False,
                    "steps": [
                        {
                            "id": "step_a",
                            "duration_s": 3.0,
                            "drive_groups": [
                                {
                                    "id": "drive_a",
                                    "waveform_id": "wave_thomas",
                                    "coil_ids": ["coil_3", "coil_4"],
                                    "connection": "series",
                                }
                            ],
                        },
                        {
                            "id": "step_b",
                            "duration_s": 2.0,
                            "drive_groups": [
                                {
                                    "id": "drive_b",
                                    "waveform_id": "wave_thomas",
                                    "coil_ids": ["coil_1", "coil_2"],
                                    "connection": "parallel",
                                },
                                {
                                    "id": "drive_c",
                                    "waveform_id": "wave_dc",
                                    "coil_ids": ["coil_3", "coil_4"],
                                    "connection": "series",
                                },
                            ],
                        },
                    ],
                }
            }
        )
        self.assertEqual(exposure_waveform_ids(protocol), ["wave_thomas", "wave_dc"])
        self.assertEqual(exposure_coil_ids(protocol), ["coil_3", "coil_4", "coil_1", "coil_2"])
        self.assertEqual(protocol["sequence"]["steps"][1]["drive_groups"][0]["connection"], "parallel")
        self.assertNotIn("waveform_name", protocol["sequence"]["steps"][0]["drive_groups"][0])
        exposure = normalize_exposure(
            {
                "id": "exp_groups",
                "name": "Groups",
                "format_version": EXPOSURE_FORMAT_VERSION,
                "protocol": protocol,
            }
        )
        summary = exposure_summary(exposure)
        self.assertEqual(summary["source_count"], 2)
        self.assertEqual(summary["drive_group_count"], 3)
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["coil_count"], 4)
        self.assertAlmostEqual(summary["duration_s"], 5.0)


    def test_source_playback_modes_normalize(self):
        base = {
            "duration_s": 2.0,
            "sources": [{
                "id": "source_a", "name": "A", "waveform_id": "wave_a",
                "playback_mode": "once_per_step", "phase_offset_s": 0.25,
            }],
            "sequence": {"loop": True, "steps": [{
                "duration_s": 1.0,
                "drive_groups": [{
                    "id": "drive_a", "source_id": "source_a",
                    "coil_ids": ["coil_1"], "connection": "series",
                }],
            }]},
        }
        protocol = normalize_exposure_protocol(base)
        self.assertEqual(protocol["sources"][0]["playback_mode"], "once_per_step")
        self.assertAlmostEqual(protocol["sources"][0]["phase_offset_s"], 0.25)

        continuous = json.loads(json.dumps(base))
        continuous["sources"][0]["playback_mode"] = "continuous"
        self.assertEqual(
            normalize_exposure_protocol(continuous)["sources"][0]["playback_mode"],
            "continuous",
        )

        former_repeat = json.loads(json.dumps(base))
        former_repeat["sources"][0]["playback_mode"] = "repeat"
        self.assertEqual(
            normalize_exposure_protocol(former_repeat)["sources"][0]["playback_mode"],
            "continuous",
        )

        invalid = json.loads(json.dumps(base))
        invalid["sources"][0]["playback_mode"] = "mystery"
        with self.assertRaisesRegex(ExposureError, "continuous or once_per_step"):
            normalize_exposure_protocol(invalid)


    def test_drive_group_load_overrides_normalize_per_selected_coil(self):
        protocol = normalize_exposure_protocol({
            "duration_s": 1.0,
            "sources": [{"id": "source_a", "name": "A", "waveform_id": "wave_a"}],
            "sequence": {"loop": False, "steps": [{
                "duration_s": 1.0,
                "drive_groups": [{
                    "id": "drive_a", "source_id": "source_a",
                    "coil_ids": ["coil_1", "coil_2"], "connection": "series",
                    "load_overrides": {
                        "coil_1": {"resistance_ohm": 12.5, "inductance_h": 0.145},
                        "coil_2": {"inductance_h": 0.150},
                    },
                }],
            }]},
        })
        group = protocol["sequence"]["steps"][0]["drive_groups"][0]
        self.assertEqual(group["load_overrides"]["coil_1"]["resistance_ohm"], 12.5)
        self.assertAlmostEqual(group["load_overrides"]["coil_1"]["inductance_h"], 0.145)
        self.assertNotIn("resistance_ohm", group["load_overrides"]["coil_2"])

        invalid = json.loads(json.dumps(protocol))
        invalid["sequence"]["steps"][0]["drive_groups"][0]["load_overrides"]["coil_3"] = {
            "resistance_ohm": 1.0
        }
        with self.assertRaisesRegex(ExposureError, "not selected"):
            normalize_exposure_protocol(invalid)

        invalid_l = json.loads(json.dumps(protocol))
        invalid_l["sequence"]["steps"][0]["drive_groups"][0]["load_overrides"]["coil_1"]["inductance_h"] = 0.0
        with self.assertRaisesRegex(ExposureError, "greater than zero"):
            normalize_exposure_protocol(invalid_l)

    def test_drive_group_rejects_unknown_connection(self):
        with self.assertRaises(ExposureError):
            normalize_exposure_protocol(
                {
                    "sequence": {
                        "steps": [
                            {
                                "duration_s": 1.0,
                                "drive_groups": [
                                    {
                                        "waveform_id": "wave_1",
                                        "coil_ids": ["coil_1"],
                                        "connection": "mystery",
                                    }
                                ],
                            }
                        ]
                    }
                }
            )

    def test_one_coil_cannot_be_connected_to_two_sources_in_same_step(self):
        with self.assertRaisesRegex(ExposureError, "only one source"):
            normalize_exposure_protocol(
                {
                    "sequence": {
                        "steps": [
                            {
                                "duration_s": 1.0,
                                "drive_groups": [
                                    {"waveform_id": "a", "coil_ids": ["coil_1"], "connection": "series"},
                                    {"waveform_id": "b", "coil_ids": ["coil_1"], "connection": "series"},
                                ],
                            }
                        ]
                    }
                }
            )

    def test_one_source_can_feed_multiple_parallel_drive_groups_in_same_step(self):
        protocol = normalize_exposure_protocol({
            "duration_s": 1.0,
            "sources": [{"id": "source_a", "name": "A", "waveform_id": "wave_a"}],
            "sequence": {"loop": False, "steps": [{
                "duration_s": 1.0,
                "drive_groups": [
                    {"id": "g1", "source_id": "source_a", "coil_ids": ["c1", "c2"], "connection": "series"},
                    {"id": "g2", "source_id": "source_a", "coil_ids": ["c3", "c4"], "connection": "series"},
                    {"id": "g3", "source_id": "source_a", "coil_ids": ["c5", "c6"], "connection": "series"},
                ],
            }]},
        })
        groups = protocol["sequence"]["steps"][0]["drive_groups"]
        self.assertEqual(len(groups), 3)
        self.assertEqual({group["source_id"] for group in groups}, {"source_a"})

    def test_unknown_protocol_fields_are_rejected_without_compatibility_paths(self):
        with self.assertRaisesRegex(ExposureError, "Unsupported exposure protocol field"):
            normalize_exposure_protocol({"old_schema": {"anything": True}})
        with self.assertRaisesRegex(ExposureError, "Unsupported exposure sequence field"):
            normalize_exposure_protocol({"sequence": {"steps": [], "legacy": True}})
        with self.assertRaisesRegex(ExposureError, "Unsupported exposure step 1 field"):
            normalize_exposure_protocol({
                "sequence": {"steps": [{"duration_s": 1.0, "drive_groups": [], "legacy": True}]}
            })
        with self.assertRaisesRegex(ExposureError, "Unsupported exposure step 1 drive group 1 field"):
            normalize_exposure_protocol({
                "sequence": {"steps": [{
                    "duration_s": 1.0,
                    "drive_groups": [{
                        "waveform_id": "wave_1", "coil_ids": ["coil_1"],
                        "connection": "series", "waveform_name": "stale cache",
                    }],
                }]}
            })

    def test_drive_group_requires_source_and_coil_load(self):
        for group, message in (
            ({"coil_ids": ["coil_1"], "connection": "series"}, "waveform source"),
            ({"waveform_id": "wave_1", "coil_ids": [], "connection": "series"}, "at least one coil"),
        ):
            with self.assertRaisesRegex(ExposureError, message):
                normalize_exposure_protocol({
                    "sequence": {"steps": [{"duration_s": 1.0, "drive_groups": [group]}]}
                })

    def test_portable_exposure_requires_all_referenced_waveforms(self):
        protocol = normalize_exposure_protocol({
            "sequence": {"steps": [{
                "duration_s": 1.0,
                "drive_groups": [{
                    "waveform_id": "wave_a",
                    "coil_ids": ["c1"],
                    "connection": "series",
                }],
            }]}
        })
        exposure = normalize_exposure({
            "id": "exp_missing_bundle",
            "name": "Missing bundle",
            "format_version": EXPOSURE_FORMAT_VERSION,
            "protocol": protocol,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.fwbexposure.json"
            with self.assertRaisesRegex(ExposureError, "every referenced waveform"):
                write_exposure_file(path, exposure)
            with self.assertRaisesRegex(ExposureError, "missing referenced waveform"):
                write_exposure_file(path, exposure, waveform_resources=[])

    def test_portable_drive_group_exposure_can_bundle_multiple_waveforms(self):
        protocol = normalize_exposure_protocol(
            {
                "sequence": {
                    "steps": [
                        {
                            "duration_s": 1.0,
                            "drive_groups": [
                                {"waveform_id": "wave_a", "coil_ids": ["c1"], "connection": "series"},
                                {"waveform_id": "wave_b", "coil_ids": ["c2"], "connection": "series"},
                            ],
                        }
                    ]
                }
            }
        )
        exposure = normalize_exposure(
            {
                "id": "exp_bundle",
                "name": "Bundle",
                "format_version": EXPOSURE_FORMAT_VERSION,
                "protocol": protocol,
            }
        )
        resources = [{"id": "wave_a", "document": {}}, {"id": "wave_b", "document": {}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.fwbexposure.json"
            write_exposure_file(path, exposure, waveform_resources=resources)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["version"], EXPOSURE_FILE_VERSION)
            self.assertEqual(len(raw["waveform_resources"]), 2)
            self.assertNotIn("waveform_resource", raw)
            restored = read_exposure_file(path)
        self.assertEqual(len(restored["_bundled_waveform_resources"]), 2)
        self.assertEqual(exposure_waveform_ids(restored["protocol"]), ["wave_a", "wave_b"])

    def test_portable_exposure_round_trip(self):
        exposure = new_exposure("Portable")
        exposure["description"] = "Derived from experiment A.\n\nReference: https://doi.org/10.1234/example\nTiming: 500 ms per step."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portable.fwbexposure.json"
            write_exposure_file(path, exposure)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], EXPOSURE_FILE_SCHEMA)
            self.assertEqual(payload["version"], EXPOSURE_FILE_VERSION)
            restored = read_exposure_file(path)
        self.assertEqual(restored["name"], "Portable")
        self.assertEqual(restored["description"], "Derived from experiment A.\n\nReference: https://doi.org/10.1234/example\nTiming: 500 ms per step.")
        self.assertEqual(restored["protocol"], exposure["protocol"])


if __name__ == "__main__":
    unittest.main()
