from __future__ import annotations

import importlib.util
import unittest
from datetime import date

import numpy as np

from fieldworkbench.geomagnetism import (
    decimal_year,
    latlon_to_utm,
    scene_vector_from_ned,
    utm_to_latlon,
    wmm2025_field,
)


class GeomagnetismTests(unittest.TestCase):
    def test_decimal_year_tracks_leap_year_calendar(self):
        self.assertAlmostEqual(decimal_year(date(2025, 1, 1)), 2025.0, places=12)
        self.assertAlmostEqual(decimal_year(date(2026, 7, 2)), 2026.0 + 182.0 / 365.0, places=12)

    def test_wgs84_utm_roundtrip_is_sub_meter_scale(self):
        latitude = 43.6532
        longitude = -79.3832
        utm = latlon_to_utm(latitude, longitude)
        self.assertEqual(utm["zone"], 17)
        self.assertEqual(utm["hemisphere"], "N")
        restored = utm_to_latlon(
            utm["zone"], utm["easting_m"], utm["northing_m"], utm["hemisphere"]
        )
        self.assertAlmostEqual(restored[0], latitude, places=6)
        self.assertAlmostEqual(restored[1], longitude, places=6)

    def test_high_arctic_negative_longitude_keeps_normal_utm_zone(self):
        utm = latlon_to_utm(83.5, -40.0)
        self.assertEqual(utm["zone"], 24)
        restored = utm_to_latlon(
            utm["zone"], utm["easting_m"], utm["northing_m"], utm["hemisphere"]
        )
        self.assertAlmostEqual(restored[0], 83.5, places=5)
        self.assertAlmostEqual(restored[1], -40.0, places=5)

    def test_zero_orientation_maps_ned_to_workbench_east_north_up(self):
        vector = scene_vector_from_ned(10.0, 20.0, 30.0)
        np.testing.assert_allclose(vector, [20.0, 10.0, -30.0], atol=1e-12)

    def test_heading_90_degrees_rotates_geographic_north_to_minus_scene_x(self):
        vector = scene_vector_from_ned(10.0, 0.0, 0.0, heading_deg=90.0)
        np.testing.assert_allclose(vector, [-10.0, 0.0, 0.0], atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("pygeomag"), "pygeomag not installed")
    def test_wmm2025_matches_official_noaa_reference_vector(self):
        # NOAA/NCEI WMM2025 official test row: 2025.0, 0 km, 80 N, 0 E.
        result = wmm2025_field(
            latitude_deg=80.0,
            longitude_deg=0.0,
            altitude_m=0.0,
            when=date(2025, 1, 1),
        )
        self.assertAlmostEqual(result["north_uT"], 6.5216, places=3)
        self.assertAlmostEqual(result["east_uT"], 0.1459, places=3)
        self.assertAlmostEqual(result["down_uT"], 54.7915, places=3)
        np.testing.assert_allclose(
            result["scene_vector_uT"],
            [0.1459, 6.5216, -54.7915],
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
