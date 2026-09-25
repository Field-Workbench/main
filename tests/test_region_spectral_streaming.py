"""Dependency-light regressions for Phase-3 authoritative regional FFTs."""
import base64
import unittest

import numpy as np

from fieldworkbench.region_analysis import analyse_region_spectrum


def _encode(values):
    return base64.b64encode(np.asarray(values, dtype=np.float64).tobytes()).decode("ascii")


def _base_payload(display_duration):
    display_times = np.asarray([0.0, float(display_duration)], dtype=np.float64)
    display_currents = np.zeros((2, 1), dtype=np.float64)
    return {
        "times_b64": _encode(display_times),
        "currents_b64": _encode(display_currents),
        "shape": [2, 1],
        "drive_ids": ["coil"],
        "regions": {
            "r": {
                "name": "R",
                "constant": 0.0,
                "linear": [0.0],
                "quadratic": [[1e-12]],
            }
        },
        "analysis_trace_reduced": True,
    }


class RegionSpectralStreamingTests(unittest.TestCase):
    def test_replay_recipe_spectral_path_validates_authoritative_schema_before_replay(self):
        payload = _base_payload(2.0)
        payload["authoritative_analysis"] = {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 99,
            "mode": "replay_recipe",
            "drive_ids": ["coil"],
            "analysis_spacing_s": 0.001,
            "protocol": {"duration_s": 2.0},
            "source_traces": {},
            "coil_loads": {},
        }
        with self.assertRaisesRegex(ValueError, "unsupported authoritative analysis format"):
            analyse_region_spectrum(payload, "r")

    def test_long_periodic_20_hz_uses_verified_cycle_without_repeat_expansion(self):
        period = 0.05
        cycle_intervals = 1000
        cycle_times = np.linspace(0.0, period, cycle_intervals + 1)
        cycle_currents = np.sin(2.0 * np.pi * 20.0 * cycle_times)[:, None]
        repeats = 36_000  # 30 minutes at 20 Hz.
        payload = _base_payload(period * repeats)
        payload["authoritative_analysis"] = {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 1,
            "mode": "verified_packets",
            "drive_ids": ["coil"],
            "analysis_spacing_s": period / cycle_intervals,
            "protocol": {"duration_s": period * repeats},
            "compression": {
                "enabled": True,
                "period_s": period,
                "stable_start_s": 0.0,
                "repeat_count": repeats,
                "remainder_s": 0.0,
            },
            "packets": [
                {
                    "times_b64": _encode(cycle_times),
                    "currents_b64": _encode(cycle_currents),
                    "shape": list(cycle_currents.shape),
                    "repeat_count": repeats,
                }
            ],
        }

        result = analyse_region_spectrum(payload, "r")
        peak = int(np.argmax(result["bin_rms_uT"][1:]) + 1)
        self.assertEqual(result["analysis_mode"], "verified_periodic_cycle")
        self.assertEqual(result["repeat_count"], repeats)
        self.assertAlmostEqual(result["physical_duration_s"], 1800.0)
        self.assertAlmostEqual(result["spectral_record_duration_s"], period)
        self.assertAlmostEqual(result["frequency_resolution_hz"], 20.0)
        self.assertAlmostEqual(result["frequency_hz"][peak], 20.0)
        self.assertAlmostEqual(result["bin_rms_uT"][peak], 1.0 / np.sqrt(2.0), places=9)
        self.assertGreater(result["nyquist_hz"], 20.0)

    def test_short_unreduced_fft_keeps_existing_whole_record_math(self):
        times = np.linspace(0.0, 1.0, 1025)
        currents = (1.0 + np.sin(2.0 * np.pi * 10.0 * times))[:, None]
        payload = {
            "times_b64": _encode(times),
            "currents_b64": _encode(currents),
            "shape": list(currents.shape),
            "drive_ids": ["coil"],
            "regions": {
                "r": {
                    "name": "R",
                    "constant": 4e-12,
                    "linear": [2e-12],
                    "quadratic": [[1e-12]],
                }
            },
            "analysis_trace_reduced": False,
        }
        result = analyse_region_spectrum(
            payload, "r", fft_samples=512, window="hann", detrend=True
        )
        self.assertEqual(result["analysis_mode"], "whole_record")
        self.assertEqual(result["fft_samples"], 512)
        self.assertAlmostEqual(result["dc_rms_uT"], 3.0)
        peak = int(np.argmax(result["bin_rms_uT"][1:]) + 1)
        self.assertAlmostEqual(result["frequency_hz"][peak], 10.0)
        self.assertAlmostEqual(
            np.sqrt(np.sum(result["bin_rms_uT"][1:] ** 2)),
            1.0 / np.sqrt(2.0),
            delta=(1.0 / np.sqrt(2.0)) * 1e-6,
        )

    def test_segmented_auto_uses_nearly_all_source_clock_across_fft_boundary(self):
        duration = 7.0
        source_intervals = 70_000
        times = np.linspace(0.0, duration, source_intervals + 1)
        currents = np.sin(2.0 * np.pi * 20.0 * times)[:, None]
        payload = _base_payload(duration)
        payload["authoritative_analysis"] = {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 1,
            "mode": "verified_packets",
            "drive_ids": ["coil"],
            "analysis_spacing_s": duration / source_intervals,
            "protocol": {"duration_s": duration},
            "packets": [
                {
                    "times_b64": _encode(times),
                    "currents_b64": _encode(currents),
                    "shape": list(currents.shape),
                    "repeat_count": 1,
                }
            ],
        }
        result = analyse_region_spectrum(payload, "r")
        self.assertEqual(result["analysis_mode"], "segmented_full_duration")
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["fft_samples"], 35_000)
        self.assertEqual(result["spectral_sample_count"], source_intervals)
        self.assertFalse(result["spectral_sampling_reduced"])
        self.assertAlmostEqual(result["nyquist_hz"], result["source_nyquist_hz"], places=10)


    def test_nonperiodic_reduced_record_uses_full_duration_segmented_power(self):
        duration = 4.0
        source_intervals = 4096
        times = np.linspace(0.0, duration, source_intervals + 1)
        currents = np.sin(2.0 * np.pi * 20.0 * times)[:, None]
        payload = _base_payload(duration)
        payload["authoritative_analysis"] = {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 1,
            "mode": "verified_packets",
            "drive_ids": ["coil"],
            "analysis_spacing_s": duration / source_intervals,
            "protocol": {"duration_s": duration},
            "packets": [
                {
                    "times_b64": _encode(times),
                    "currents_b64": _encode(currents),
                    "shape": list(currents.shape),
                    "repeat_count": 1,
                }
            ],
        }

        result = analyse_region_spectrum(payload, "r", fft_samples=1024)
        peak = int(np.argmax(result["bin_rms_uT"][1:]) + 1)
        self.assertEqual(result["analysis_mode"], "segmented_full_duration")
        self.assertEqual(result["segment_count"], 4)
        self.assertEqual(result["spectral_sample_count"], 4096)
        self.assertFalse(result["spectral_sampling_reduced"])
        self.assertAlmostEqual(result["physical_duration_s"], 4.0)
        self.assertAlmostEqual(result["spectral_record_duration_s"], 1.0)
        self.assertAlmostEqual(result["frequency_resolution_hz"], 1.0)
        self.assertAlmostEqual(result["nyquist_hz"], 512.0)
        self.assertAlmostEqual(result["frequency_hz"][peak], 20.0)
        self.assertAlmostEqual(result["bin_rms_uT"][peak], 1.0 / np.sqrt(2.0), places=8)

    def test_segmented_fft_never_claims_more_bandwidth_than_source_clock(self):
        duration = 3.7
        source_intervals = 5000
        times = np.linspace(0.0, duration, source_intervals + 1)
        currents = np.cos(2.0 * np.pi * 11.0 * times)[:, None]
        payload = _base_payload(duration)
        payload["authoritative_analysis"] = {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 1,
            "mode": "verified_packets",
            "drive_ids": ["coil"],
            "analysis_spacing_s": duration / source_intervals,
            "protocol": {"duration_s": duration},
            "packets": [
                {
                    "times_b64": _encode(times),
                    "currents_b64": _encode(currents),
                    "shape": list(currents.shape),
                    "repeat_count": 1,
                }
            ],
        }
        result = analyse_region_spectrum(payload, "r", fft_samples=1024, window="hann")
        self.assertLessEqual(result["nyquist_hz"], result["source_nyquist_hz"] * (1.0 + 1e-12))
        self.assertGreaterEqual(result["sample_spacing_s"], result["source_sample_spacing_s"] * (1.0 - 1e-12))
        self.assertTrue(result["spectral_sampling_reduced"])


if __name__ == "__main__":
    unittest.main()
