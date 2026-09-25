"""Dependency-light regressions for Phase-4 authoritative frequency weighting."""
import base64
import unittest

import numpy as np

from fieldworkbench.region_analysis import analyse_region_frequency_weighted


def _encode(values):
    return base64.b64encode(np.asarray(values, dtype=np.float64).tobytes()).decode("ascii")


def _base_payload(display_duration):
    return {
        "times_b64": _encode([0.0, float(display_duration)]),
        "currents_b64": _encode(np.zeros((2, 1), dtype=np.float64)),
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


def _packet_source(times, currents, duration, *, repeat_count=1, spacing=None):
    times = np.asarray(times, dtype=np.float64)
    currents = np.asarray(currents, dtype=np.float64)
    if spacing is None:
        spacing = float(times[-1] - times[0]) / max(1, len(times) - 1)
    return {
        "schema": "fieldworkbench.authoritative-exposure-analysis",
        "version": 1,
        "mode": "verified_packets",
        "drive_ids": ["coil"],
        "analysis_spacing_s": float(spacing),
        "protocol": {"duration_s": float(duration)},
        "compression": {
            "enabled": repeat_count > 1,
            "period_s": float(times[-1] - times[0]),
            "stable_start_s": float(times[0]),
            "repeat_count": int(repeat_count),
            "remainder_s": 0.0,
        },
        "packets": [
            {
                "times_b64": _encode(times),
                "currents_b64": _encode(currents),
                "shape": list(currents.shape),
                "repeat_count": int(repeat_count),
            }
        ],
    }


class RegionFrequencyWeightingStreamingTests(unittest.TestCase):
    def test_30_minute_periodic_tone_uses_verified_cycle(self):
        period = 0.05
        intervals = 1000
        times = np.linspace(0.0, period, intervals + 1)
        # 100 uT RMS in a 1 uT/A regional basis.
        currents = (100.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 20.0 * times))[:, None]
        repeats = 36_000
        payload = _base_payload(period * repeats)
        payload["authoritative_analysis"] = _packet_source(
            times, currents, period * repeats, repeat_count=repeats,
            spacing=period / intervals,
        )

        result = analyse_region_frequency_weighted(payload, "r")
        # ICNIRP 2010 public reference at 20 Hz is 250 uT RMS.
        self.assertEqual(result["analysis_mode"], "verified_periodic_cycle")
        self.assertEqual(result["repeat_count"], repeats)
        self.assertAlmostEqual(result["physical_duration_s"], 1800.0)
        self.assertAlmostEqual(result["weighted_peak_index"], 0.4, places=10)
        self.assertAlmostEqual(result["spectral_sum_index"], 0.4, places=10)
        self.assertAlmostEqual(result["weighted_peak_equivalent_uT"], 80.0 * np.sqrt(2.0), places=9)
        self.assertAlmostEqual(result["resolved_band_min_hz"], 20.0)
        self.assertFalse(result["full_band_resolved"])
        self.assertLessEqual(len(result["time_s"]), 65536)

    def test_segmented_full_duration_weighting_matches_integer_cycle_tone(self):
        duration = 8.0
        intervals = 8192
        times = np.linspace(0.0, duration, intervals + 1)
        currents = (100.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 20.0 * times))[:, None]
        payload = _base_payload(duration)
        payload["authoritative_analysis"] = _packet_source(
            times, currents, duration, spacing=duration / intervals
        )

        result = analyse_region_frequency_weighted(
            payload, "r", fft_samples=2048
        )
        self.assertEqual(result["analysis_mode"], "segmented_full_duration")
        self.assertEqual(result["segment_count"], 4)
        self.assertEqual(result["weighting_context_records"], 3)
        self.assertAlmostEqual(result["weighted_peak_index"], 0.4, places=9)
        self.assertAlmostEqual(result["spectral_sum_index"], 0.4, places=9)
        self.assertAlmostEqual(result["frequency_resolution_hz"], 0.5)
        self.assertAlmostEqual(result["weighting_frequency_resolution_hz"], 1.0 / 6.0)

    def test_segmented_weighting_retains_bounded_plot_trace(self):
        duration = 100.0
        intervals = 100_000
        times = np.linspace(0.0, duration, intervals + 1)
        currents = (10.0 * np.sin(2.0 * np.pi * 10.0 * times))[:, None]
        payload = _base_payload(duration)
        payload["authoritative_analysis"] = _packet_source(
            times, currents, duration, spacing=duration / intervals
        )

        result = analyse_region_frequency_weighted(payload, "r", fft_samples=1024)
        self.assertEqual(result["analysis_mode"], "segmented_full_duration")
        self.assertTrue(result["weighted_trace_reduced"])
        self.assertLessEqual(len(result["time_s"]), 65536)
        self.assertEqual(len(result["time_s"]), len(result["weighted_index"]))
        self.assertTrue(np.isfinite(result["weighted_peak_index"]))
        self.assertGreater(result["weighted_peak_index"], 0.0)


if __name__ == "__main__":
    unittest.main()
