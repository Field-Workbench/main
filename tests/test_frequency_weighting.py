"""Frequency-weighted regional screening math regression checks."""
import unittest

import numpy as np

from fieldworkbench.brain_analysis import BrainRegionNode, BrainSampleSet
from fieldworkbench.region_analysis import (
    analyse_region_frequency_weighted,
    frequency_weighting_reference_uT,
    waveform_basis_payload,
)


class FrequencyWeightingTests(unittest.TestCase):
    @staticmethod
    def _single_tone_payload(frequency_hz: float, rms_uT: float, duration_s: float = 1.0):
        t = np.linspace(0.0, duration_s, 10001)
        samples = BrainSampleSet(
            np.zeros((1, 3)), np.ones(1, dtype=int), np.ones(1),
            np.arange(1), {1: 1}, {1: 1}, 1,
        )
        root = BrainRegionNode("r", "R", (1,))
        bases = np.array([[[1e-6, 0.0, 0.0]]])  # 1 µT/A
        currents = (
            rms_uT * np.sqrt(2.0) * np.sin(2.0 * np.pi * frequency_hz * t)
        )[:, None]
        return waveform_basis_payload(
            np.zeros((1, 3)), bases, samples, root,
            {"analysis_times": t, "analysis_currents": currents}, ["coil"],
        )

    def test_single_tone_weighted_peak_and_spectral_sum_match_reference_ratio(self):
        payload = self._single_tone_payload(10.0, 100.0)
        result = analyse_region_frequency_weighted(
            payload, "r", profile="icnirp2010_public"
        )
        # ICNIRP 2010 public B reference at 10 Hz is 500 µT RMS.
        self.assertAlmostEqual(result["weighted_peak_index"], 0.2, delta=2e-6)
        self.assertAlmostEqual(result["spectral_sum_index"], 0.2, delta=2e-6)
        self.assertAlmostEqual(result["reference_rms_uT"], 200.0)

    def test_reference_profiles_match_tabulated_breakpoints(self):
        public = frequency_weighting_reference_uT(
            np.array([1.0, 8.0, 25.0, 50.0, 400.0, 3000.0, 100000.0]),
            "icnirp2010_public",
        )
        np.testing.assert_allclose(
            public, [40000.0, 625.0, 200.0, 200.0, 200.0, 27.0, 27.0]
        )
        occupational = frequency_weighting_reference_uT(
            np.array([1.0, 8.0, 25.0, 300.0, 3000.0, 100000.0]),
            "icnirp2010_occupational",
        )
        np.testing.assert_allclose(
            occupational, [200000.0, 3125.0, 1000.0, 1000.0, 100.0, 100.0]
        )

    def test_frequency_outside_implemented_lf_band_is_not_weighted(self):
        refs = frequency_weighting_reference_uT(
            np.array([0.5, 1.0, 100000.0, 100001.0]), "icnirp2010_public"
        )
        self.assertTrue(np.isinf(refs[0]))
        self.assertTrue(np.isfinite(refs[1]))
        self.assertTrue(np.isfinite(refs[2]))
        self.assertTrue(np.isinf(refs[3]))


if __name__ == "__main__":
    unittest.main()
