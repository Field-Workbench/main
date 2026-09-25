from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fieldworkbench"
    / "numeric_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("fieldworkbench_numeric_inputs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib contract guard
    raise RuntimeError("Unable to load numeric input helpers")
NUMERIC_INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMERIC_INPUTS)


class NumericInputTests(unittest.TestCase):
    def test_fixed_decimal_text_hides_only_insignificant_padding(self) -> None:
        compact = NUMERIC_INPUTS.compact_fixed_decimal_text
        self.assertEqual(compact("12.000000"), "12")
        self.assertEqual(compact("-12.340000"), "-12.34")
        self.assertEqual(compact("0.000001"), "0.000001")
        self.assertEqual(compact("12,5000", decimal_point=","), "12,5")
        self.assertEqual(compact("12.000000", minimum_decimals=2), "12.00")
        self.assertEqual(compact("12.300000", minimum_decimals=2), "12.30")
        self.assertEqual(compact("12.345000", minimum_decimals=2), "12.345")
        self.assertEqual(
            compact("12,300000", decimal_point=",", minimum_decimals=2),
            "12,30",
        )

    def test_playback_speed_arrows_follow_requested_ratio_ladder(self) -> None:
        step = NUMERIC_INPUTS.stepped_playback_speed
        value = 1.0
        expected = [0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        for next_value in expected:
            value = step(value, -1, decimals=9)
            self.assertAlmostEqual(value, next_value, places=9)

        ladder = NUMERIC_INPUTS.playback_speed_ladder(decimals=9)
        self.assertEqual(ladder[0], NUMERIC_INPUTS.PLAYBACK_SPEED_MINIMUM)
        self.assertGreater(ladder[1], ladder[0])
        self.assertEqual(step(ladder[0], -1, decimals=9), ladder[0])
        self.assertEqual(step(ladder[0], 1, decimals=9), ladder[1])

    def test_arbitrary_typed_speed_rejoins_pattern_in_arrow_direction(self) -> None:
        step = NUMERIC_INPUTS.stepped_playback_speed
        self.assertEqual(step(0.6, -1, decimals=9), 0.5)
        self.assertEqual(step(0.6, 1, decimals=9), 0.75)
        self.assertEqual(step(0.1, -1, decimals=9), 0.0625)
        self.assertEqual(step(0.1, 1, decimals=9), 0.125)
        self.assertEqual(step(1.1, -1, decimals=9), 1.0)
        self.assertEqual(step(1.1, 1, decimals=9), 1.25)


if __name__ == "__main__":
    unittest.main()
