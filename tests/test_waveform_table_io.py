"""Focused regression coverage for Phase-5 ordinary text/CSV waveform I/O."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fieldworkbench.waveform_io import WaveformDocument, WaveformIOError, WaveformSeries
from fieldworkbench.waveform_table_io import (
    TabularSignalSpec,
    documents_from_text_table,
    parse_text_table,
    read_text_table,
    write_waveform_csv,
)


class WaveformTableIOTests(unittest.TestCase):
    def test_auto_csv_header_and_multiple_signal_columns(self):
        table = parse_text_table(
            "Time (ms),Coil current,Generator voltage\n0,0,1\n1,10,2\n2,-5,3\n",
            delimiter_mode="auto",
            header_mode="auto",
            source_path="scope.csv",
        )
        self.assertTrue(table.has_header)
        self.assertEqual(table.delimiter_mode, "comma")
        documents = documents_from_text_table(
            table,
            (
                TabularSignalSpec(1, "Thomas current", "current", "mA"),
                TabularSignalSpec(2, "Drive voltage", "voltage", "V", scale=2.0, offset=-1.0),
            ),
            time_column=0,
            time_unit="ms",
        )
        self.assertEqual(len(documents), 2)
        np.testing.assert_allclose(documents[0].signal.time_s, [0.0, 0.001, 0.002])
        np.testing.assert_allclose(documents[0].signal.values, [0.0, 10.0, -5.0])
        np.testing.assert_allclose(documents[1].signal.values, [1.0, 3.0, 5.0])
        self.assertEqual(documents[1].signal.quantity, "voltage")
        self.assertEqual(documents[1].metadata["import_settings"]["signal_header"], "Generator voltage")
        self.assertEqual(documents[1].metadata["import_settings"]["time_header"], "Time (ms)")

    def test_whitespace_table_can_supply_fixed_interval(self):
        table = parse_text_table("1 2\n3 4\n5 6\n", delimiter_mode="auto", header_mode="no")
        documents = documents_from_text_table(
            table,
            (TabularSignalSpec(0, "A", "current", "A"), TabularSignalSpec(1, "B", "current", "mA")),
            time_column=None,
            sample_interval=0.5,
            sample_interval_unit="ms",
        )
        np.testing.assert_allclose(documents[0].signal.time_s, [0.0, 0.0005, 0.001])
        self.assertEqual(documents[0].metadata["import_settings"]["time_mode"], "fixed_interval")

    def test_selected_columns_must_be_numeric_and_time_monotonic(self):
        bad_values = parse_text_table("t,x\n0,1\n1,nope\n", delimiter_mode="comma", header_mode="yes")
        with self.assertRaisesRegex(WaveformIOError, "non-numeric"):
            documents_from_text_table(
                bad_values, (TabularSignalSpec(1, "x", "current", "mA"),), time_column=0
            )
        bad_time = parse_text_table("t,x\n0,1\n0,2\n", delimiter_mode="comma", header_mode="yes")
        with self.assertRaisesRegex(WaveformIOError, "strictly increasing"):
            documents_from_text_table(
                bad_time, (TabularSignalSpec(1, "x", "current", "mA"),), time_column=0
            )

    def test_csv_export_is_simple_physical_table_and_reimports(self):
        document = WaveformDocument(
            [WaveformSeries("Source", [0.0, 0.01, 0.02], [0.0, 5.0, -2.0], "current", "mA")],
            title="Thomas-1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_waveform_csv(Path(tmp) / "wave.csv", document)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Time (s)", text)
            self.assertIn("Thomas-1 (mA)", text)
            table = read_text_table(path)
        imported = documents_from_text_table(
            table,
            (TabularSignalSpec(1, "Thomas-1", "current", "mA"),),
            time_column=0,
            time_unit="s",
        )[0]
        np.testing.assert_allclose(imported.signal.time_s, document.signal.time_s)
        np.testing.assert_allclose(imported.signal.values, document.signal.values)


if __name__ == "__main__":
    unittest.main()
