from __future__ import annotations

import unittest

from fieldworkbench.field_grid import (
    FIELD_VOLUME_AXIS_LABELS,
    field_volume_grid_spec,
)


class FieldGridTests(unittest.TestCase):
    def test_grid_contract_has_shared_axes_ticks_and_box_edges(self) -> None:
        spec = field_volume_grid_spec([10.0, -20.0, 30.0], 80.0)

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["box_min_mm"], [-30.0, -60.0, -10.0])
        self.assertEqual(spec["box_max_mm"], [50.0, 20.0, 70.0])
        self.assertEqual(len(spec["box_segments_mm"]), 12)
        self.assertEqual(
            [axis["label"] for axis in spec["axes"]],
            list(FIELD_VOLUME_AXIS_LABELS),
        )
        for axis in spec["axes"]:
            self.assertEqual(len(axis["tick_values_mm"]), 5)
            self.assertEqual(len(axis["tick_labels"]), 5)
            self.assertEqual(axis["tick_values_mm"][0], axis["minimum_mm"])
            self.assertEqual(axis["tick_values_mm"][-1], axis["maximum_mm"])

    def test_disabled_grid_keeps_geometry_contract_but_marks_it_hidden(self) -> None:
        spec = field_volume_grid_spec([0.0, 0.0, 0.0], 100.0, enabled=False)
        self.assertFalse(spec["enabled"])
        self.assertEqual(len(spec["box_segments_mm"]), 12)


if __name__ == "__main__":
    unittest.main()
