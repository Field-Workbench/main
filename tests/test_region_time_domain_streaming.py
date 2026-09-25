"""Lightweight regressions for authoritative regional time-domain analysis."""
import base64
import unittest

import numpy as np

from fieldworkbench.region_analysis import (
    MAX_RETAINED_INTERVALS,
    analyse_region_exposure_summary,
    analyse_region_frequency_weighted,
    analyse_region_spectrum,
    analyse_region_threshold_duration,
    analyse_region_waveform,
)


def _encode(values):
    return base64.b64encode(np.asarray(values, dtype=np.float64).tobytes()).decode("ascii")


def _verified_payload(values, *, repeat_count=4, times=(0.0, 0.25, 0.5)):
    times = np.asarray(times, dtype=np.float64)
    currents = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    display_times = np.asarray([0.0, max(1.0, float(times[-1]) * repeat_count)], dtype=np.float64)
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
        "authoritative_analysis": {
            "schema": "fieldworkbench.authoritative-exposure-analysis",
            "version": 1,
            "mode": "verified_packets",
            "drive_ids": ["coil"],
            "packets": [
                {
                    "times_b64": _encode(times),
                    "currents_b64": _encode(currents),
                    "shape": list(currents.shape),
                    "repeat_count": int(repeat_count),
                }
            ],
        },
    }


class RegionTimeDomainStreamingTests(unittest.TestCase):
    def test_reduced_display_trace_uses_authoritative_packets_for_time_domain_tools(self):
        payload = _verified_payload([0.0, 2.0, 0.0], repeat_count=4)

        summary = analyse_region_exposure_summary(payload, "r")
        self.assertEqual(summary["analysis_source"], "authoritative_packets")
        self.assertAlmostEqual(summary["duration_s"], 2.0)
        self.assertAlmostEqual(summary["b_time_uT_s"], 2.0)
        self.assertAlmostEqual(summary["b2_time_uT2_s"], 4.0)
        self.assertAlmostEqual(summary["playback_rms_uT"], np.sqrt(2.0))

        threshold = analyse_region_threshold_duration(payload, "r", 1.0)
        self.assertEqual(threshold["excursion_count"], 4)
        self.assertAlmostEqual(threshold["time_above_s"], 1.0)
        self.assertAlmostEqual(threshold["longest_interval_s"], 0.25)
        self.assertFalse(threshold["intervals_truncated"])

        timing = analyse_region_waveform(payload, "r", 50.0)
        drive = timing["drive_timing"][0]
        self.assertEqual(drive["pulses"], 4)
        self.assertAlmostEqual(drive["active_s"], 1.0)
        self.assertAlmostEqual(drive["pulse_width_mean_s"], 0.25)
        self.assertAlmostEqual(drive["gap_width_mean_s"], 0.25)
        self.assertAlmostEqual(timing["peak_spatial_rms_slew_uT_s"], 8.0)

    def test_repeated_boundary_excursions_merge_without_expansion(self):
        payload = _verified_payload([2.0, 0.0, 2.0], repeat_count=4)
        threshold = analyse_region_threshold_duration(payload, "r", 1.0)
        self.assertEqual(threshold["excursion_count"], 5)
        self.assertAlmostEqual(threshold["time_above_s"], 1.0)
        self.assertAlmostEqual(threshold["longest_interval_s"], 0.25)
        self.assertEqual(
            threshold["intervals_s"],
            [(0.0, 0.125), (0.375, 0.625), (0.875, 1.125), (1.375, 1.625), (1.875, 2.0)],
        )

    def test_large_repeat_keeps_threshold_table_bounded(self):
        repeats = 100_000
        payload = _verified_payload([0.0, 2.0, 0.0], repeat_count=repeats)
        threshold = analyse_region_threshold_duration(payload, "r", 1.0)
        self.assertEqual(threshold["excursion_count"], repeats)
        self.assertEqual(len(threshold["intervals_s"]), MAX_RETAINED_INTERVALS)
        self.assertTrue(threshold["intervals_truncated"])
        self.assertAlmostEqual(threshold["time_above_s"], repeats * 0.25)

        timing = analyse_region_waveform(payload, "r", 50.0)["drive_timing"][0]
        self.assertEqual(timing["pulses"], repeats)
        self.assertAlmostEqual(timing["pulse_width_min_s"], 0.25)
        self.assertAlmostEqual(timing["pulse_width_max_s"], 0.25)
        self.assertTrue(timing["timing_preview_truncated"])


    def test_unreduced_non_exposure_trace_keeps_short_waveform_behavior(self):
        times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        currents = np.array([[0.0], [2.0], [2.0], [0.0]], dtype=np.float64)
        payload = {
            "times_b64": _encode(times),
            "currents_b64": _encode(currents),
            "shape": [4, 1],
            "drive_ids": ["coil"],
            "regions": {
                "r": {
                    "name": "R",
                    "constant": 0.0,
                    "linear": [0.0],
                    "quadratic": [[1e-12]],
                }
            },
            "analysis_trace_reduced": False,
        }
        summary = analyse_region_exposure_summary(payload, "r")
        self.assertEqual(summary["analysis_source"], "full_retained_trace")
        self.assertAlmostEqual(summary["b_time_uT_s"], 4.0)
        self.assertAlmostEqual(summary["b2_time_uT2_s"], 8.0)
        threshold = analyse_region_threshold_duration(payload, "r", 1.0)
        self.assertEqual(threshold["intervals_s"], [(0.5, 2.5)])
        timing = analyse_region_waveform(payload, "r", 75.0)["drive_timing"][0]
        self.assertEqual(timing["pulses"], 1)
        self.assertAlmostEqual(timing["active_s"], 1.5)

    def test_compressed_cycles_match_explicit_short_expansion(self):
        rng = np.random.default_rng(5)
        for _case in range(25):
            sample_count = int(rng.integers(3, 9))
            repeat_count = int(rng.integers(2, 7))
            middle = np.sort(rng.uniform(0.01, 0.99, sample_count - 2))
            cycle_times = np.r_[0.0, middle, 1.0]
            cycle_values = rng.uniform(-2.0, 2.0, sample_count)
            cycle_values[-1] = cycle_values[0]
            compressed = _verified_payload(
                cycle_values, repeat_count=repeat_count, times=cycle_times
            )

            explicit_times = []
            explicit_values = []
            for cycle_index in range(repeat_count):
                shifted = cycle_times + cycle_index
                values = cycle_values
                if cycle_index:
                    shifted = shifted[1:]
                    values = values[1:]
                explicit_times.extend(shifted)
                explicit_values.extend(values)
            explicit_currents = np.asarray(explicit_values, dtype=np.float64)[:, None]
            explicit = {
                "times_b64": _encode(np.asarray(explicit_times)),
                "currents_b64": _encode(explicit_currents),
                "shape": list(explicit_currents.shape),
                "drive_ids": ["coil"],
                "regions": compressed["regions"],
                "analysis_trace_reduced": False,
            }

            threshold_uT = float(rng.uniform(0.0, 1.5))
            pulse_percent = float(rng.uniform(0.1, 90.0))
            summary_a = analyse_region_exposure_summary(compressed, "r")
            summary_b = analyse_region_exposure_summary(explicit, "r")
            for key in ("duration_s", "b_time_uT_s", "b2_time_uT2_s", "playback_rms_uT"):
                self.assertAlmostEqual(summary_a[key], summary_b[key], places=10)

            threshold_a = analyse_region_threshold_duration(compressed, "r", threshold_uT)
            threshold_b = analyse_region_threshold_duration(explicit, "r", threshold_uT)
            self.assertEqual(threshold_a["excursion_count"], threshold_b["excursion_count"] )
            for key in ("time_above_s", "longest_interval_s"):
                self.assertAlmostEqual(threshold_a[key], threshold_b[key], places=10)

            timing_a = analyse_region_waveform(compressed, "r", pulse_percent)
            timing_b = analyse_region_waveform(explicit, "r", pulse_percent)
            np.testing.assert_allclose(
                timing_a["peak_spatial_rms_slew_uT_s"],
                timing_b["peak_spatial_rms_slew_uT_s"],
                rtol=1e-12, atol=1e-12,
            )
            drive_a = timing_a["drive_timing"][0]
            drive_b = timing_b["drive_timing"][0]
            self.assertEqual(drive_a["pulses"], drive_b["pulses"] )
            self.assertEqual(drive_a["gap_count"], drive_b["gap_count"] )
            for key in (
                "active_s", "inactive_s", "pulse_width_min_s",
                "pulse_width_max_s", "pulse_width_mean_s",
                "gap_width_min_s", "gap_width_max_s", "gap_width_mean_s",
            ):
                self.assertAlmostEqual(drive_a[key], drive_b[key], places=10)


    def test_time_domain_progress_is_determinate_for_verified_packets(self):
        payload = _verified_payload([0.0, 2.0, 0.0], repeat_count=100_000)

        def check_progress(analyse, *args, require_intermediate=True):
            updates = []
            analyse(payload, "r", *args, progress_callback=updates.append)
            numeric = [
                update for update in updates
                if "completed" in update and "total" in update
            ]
            self.assertTrue(numeric)
            totals = {int(update["total"]) for update in numeric}
            self.assertEqual(len(totals), 1)
            total = totals.pop()
            self.assertGreater(total, 0)
            completed = [int(update["completed"]) for update in numeric]
            self.assertEqual(completed, sorted(completed))
            self.assertEqual(completed[0], 0)
            self.assertEqual(completed[-1], total)
            if require_intermediate:
                self.assertTrue(any(0 < value < total for value in completed))
            return updates

        summary_updates = check_progress(
            analyse_region_exposure_summary, require_intermediate=False
        )
        self.assertEqual(summary_updates[-1]["message"], "Exposure summary ready")

        threshold_updates = check_progress(analyse_region_threshold_duration, 1.0)
        self.assertEqual(threshold_updates[-1]["message"], "Threshold analysis ready")

        timing_updates = check_progress(analyse_region_waveform, 50.0)
        self.assertEqual(timing_updates[-1]["message"], "Timing analysis ready")

    def test_authoritative_drive_order_must_match_regional_basis(self):
        payload = _verified_payload([0.0, 2.0, 0.0], repeat_count=4)
        payload["authoritative_analysis"]["drive_ids"] = ["wrong-coil"]
        cases = (
            (analyse_region_exposure_summary, ()),
            (analyse_region_threshold_duration, (1.0,)),
            (analyse_region_waveform, (50.0,)),
            (analyse_region_spectrum, ()),
            (analyse_region_frequency_weighted, ()),
        )
        for analyse, args in cases:
            with self.subTest(analyse=analyse.__name__):
                with self.assertRaisesRegex(ValueError, "drive order"):
                    analyse(payload, "r", *args)

    def test_fft_and_weighting_use_verified_cycle_without_repeat_expansion(self):
        payload = _verified_payload([0.0, 2.0, 0.0], repeat_count=4)
        spectrum = analyse_region_spectrum(payload, "r")
        self.assertEqual(spectrum["analysis_mode"], "verified_periodic_cycle")
        self.assertEqual(spectrum["repeat_count"], 4)
        self.assertAlmostEqual(spectrum["spectral_record_duration_s"], 0.5)
        weighted = analyse_region_frequency_weighted(payload, "r")
        self.assertEqual(weighted["analysis_mode"], "verified_periodic_cycle")
        self.assertEqual(weighted["repeat_count"], 4)
        self.assertLessEqual(len(weighted["time_s"]), 65536)


if __name__ == "__main__":
    unittest.main()
