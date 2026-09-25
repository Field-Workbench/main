from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fieldworkbench.waveform_io import (
    NATIVE_WAVEFORM_FORMAT,
    NATIVE_WAVEFORM_READER_ID,
    NATIVE_WAVEFORM_VERSION,
    NativeWaveformJSON,
    WaveformDocument,
    WaveformIOError,
    WaveformIORegistry,
    WaveformSeries,
    build_default_waveform_io_registry,
    convert_signal_values,
    read_project_waveform_documents,
    read_waveform_document,
    series_values_si,
    waveform_document_from_common_time,
    write_waveform_document,
    waveform_document_from_payload,
    waveform_signal_to_payload,
)


class WaveformIOTests(unittest.TestCase):
    def test_native_waveform_round_trip_preserves_single_electrical_signal_and_metadata(self) -> None:
        document = WaveformDocument(
            [
                WaveformSeries(
                    "Drive A",
                    [0.0, 0.001, 0.002],
                    [0.0, 12.5, 0.0],
                    quantity="current",
                    unit="mA",
                    metadata={"scope_channel": "CH1"},
                ),
            ],
            title="Measured exposure",
            metadata={"instrument": "example"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.fwbwave.json"
            write_waveform_document(path, document)
            decoded = read_waveform_document(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(raw["format"], NATIVE_WAVEFORM_FORMAT)
        self.assertEqual(raw["version"], NATIVE_WAVEFORM_VERSION)
        self.assertIn("signal", raw)
        self.assertNotIn("channels", raw)
        self.assertEqual(decoded.source_format, NATIVE_WAVEFORM_READER_ID)
        self.assertEqual(decoded.title, "Measured exposure")
        self.assertEqual(decoded.metadata["instrument"], "example")
        self.assertEqual(decoded.channel_count, 1)
        self.assertEqual(decoded.channel("drive a").unit, "mA")
        self.assertEqual(decoded.channel(0).metadata["scope_channel"], "CH1")
        np.testing.assert_allclose(decoded.signal.time_s, [0.0, 0.001, 0.002])
        np.testing.assert_allclose(decoded.signal.values, [0.0, 12.5, 0.0])

    def test_project_waveform_import_preserves_each_resource_and_skips_invalid_entries(self) -> None:
        current = WaveformDocument(
            [WaveformSeries("I", [0.0, 0.001], [0.0, 12.0], quantity="current", unit="mA")],
            title="Old current title",
            metadata={"notes": "current note", "electrical": {"slew_rate_per_s": 4.0}},
        )
        voltage = WaveformDocument(
            [WaveformSeries("V", [0.0, 0.002], [0.0, 1.0], quantity="voltage", unit="V")],
            title="Old voltage title",
            metadata={
                "notes": "voltage note",
                "electrical": {"slew_rate_per_s": 8.0, "output_resistance_ohm": 50.0},
            },
        )
        project = {
            "objects": [],
            "field_workbench": {
                "version": 23,
                "waveforms": [
                    {"id": "w1", "name": "Project current", "document": waveform_signal_to_payload(current)},
                    {"id": "broken", "name": "Broken", "document": {"format": "not-a-waveform"}},
                    {"id": "w2", "name": "Project voltage", "document": waveform_signal_to_payload(voltage)},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.magpy.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            documents = read_project_waveform_documents(path)

        self.assertEqual([item.title for item in documents], ["Project current", "Project voltage"])
        self.assertEqual([item.signal.quantity for item in documents], ["current", "voltage"])
        self.assertEqual(documents[0].metadata["notes"], "current note")
        self.assertEqual(documents[1].metadata["electrical"]["output_resistance_ohm"], 50.0)
        self.assertEqual(documents[0].source_path, str(path))
        self.assertEqual(documents[0].source_format, "fieldworkbench.project-waveform")

    def test_native_payload_is_strictly_single_electrical_signal(self) -> None:
        document = WaveformDocument(
            [WaveformSeries("Control", [0.0, 0.001, 0.002], [0.0, 0.5, 1.0], quantity="voltage", unit="V")],
            title="Builder waveform",
            metadata={"origin": "test"},
        )
        payload = waveform_signal_to_payload(document)
        decoded = waveform_document_from_payload(payload)
        self.assertEqual(payload["format"], NATIVE_WAVEFORM_FORMAT)
        self.assertEqual(payload["version"], NATIVE_WAVEFORM_VERSION)
        self.assertIn("signal", payload)
        self.assertNotIn("channels", payload)
        self.assertEqual(decoded.title, "Builder waveform")
        self.assertEqual(decoded.channels[0].quantity, "voltage")
        np.testing.assert_allclose(decoded.channels[0].values, [0.0, 0.5, 1.0])

        with self.assertRaisesRegex(WaveformIOError, "exactly one signal"):
            waveform_signal_to_payload(
                WaveformDocument(
                    [
                        WaveformSeries("A", [0.0, 0.001], [0.0, 1.0], quantity="current", unit="A"),
                        WaveformSeries("B", [0.0, 0.001], [0.0, 1.0], quantity="voltage", unit="V"),
                    ]
                )
            )

        with self.assertRaisesRegex(WaveformIOError, "Current .* or Voltage"):
            waveform_signal_to_payload(
                WaveformDocument(
                    [
                        WaveformSeries(
                            "Control",
                            [0.0, 0.001],
                            [0.0, 1.0],
                            quantity="arbitrary",
                            unit="normalized",
                        )
                    ]
                )
            )

        legacy_v1 = {
            "format": NATIVE_WAVEFORM_FORMAT,
            "version": 1,
            "title": "old",
            "channels": [],
        }
        with self.assertRaisesRegex(WaveformIOError, "Unsupported native waveform version 1"):
            waveform_document_from_payload(legacy_v1)

    def test_catalogue_voltage_source_output_resistance_is_canonical(self) -> None:
        voltage = WaveformDocument(
            [WaveformSeries("V", [0.0, 0.001], [0.0, 1.0], quantity="voltage", unit="V")],
            metadata={
                "electrical": {
                    "slew_rate_per_s": 12.0,
                    "output_resistance_ohm": 50.0,
                }
            },
        )
        payload = waveform_signal_to_payload(voltage)
        electrical = payload["metadata"]["electrical"]
        self.assertEqual(electrical["output_resistance_ohm"], 50.0)
        decoded = waveform_document_from_payload(payload)
        self.assertEqual(decoded.metadata["electrical"]["output_resistance_ohm"], 50.0)

        for invalid_field in ("resistance_ohm", "inductance_h", "manual_resistance_ohm"):
            invalid_voltage = WaveformDocument(
                [WaveformSeries("V", [0.0, 0.001], [0.0, 1.0], quantity="voltage", unit="V")],
                metadata={"electrical": {invalid_field: 1.0}},
            )
            with self.assertRaisesRegex(WaveformIOError, "Unsupported waveform electrical field"):
                waveform_signal_to_payload(invalid_voltage)

        current = WaveformDocument(
            [WaveformSeries("I", [0.0, 0.001], [0.0, 1.0], quantity="current", unit="A")],
            metadata={"electrical": {"output_resistance_ohm": 50.0}},
        )
        with self.assertRaisesRegex(WaveformIOError, "Unsupported waveform electrical field"):
            waveform_signal_to_payload(current)

        for invalid in (-1.0, float("nan")):
            voltage.metadata["electrical"]["output_resistance_ohm"] = invalid
            with self.assertRaisesRegex(WaveformIOError, "output resistance"):
                waveform_signal_to_payload(voltage)

    def test_common_time_constructor_normalizes_time_units(self) -> None:
        document = waveform_document_from_common_time(
            [0.0, 1.0, 2.0],
            [
                ("CH1", [0.0, 1.0, 0.0], "voltage", "V"),
                ("CH2", [0.0, 10.0, 0.0], "current", "mA"),
            ],
            time_unit="ms",
        )
        np.testing.assert_allclose(document.common_time_s(), [0.0, 0.001, 0.002])
        self.assertEqual(document.channel_count, 2)

    def test_signal_unit_helpers_preserve_imported_values_and_offer_si_mapping(self) -> None:
        series = WaveformSeries(
            "sense",
            [0.0, 1.0],
            [0.0, 250.0],
            quantity="voltage",
            unit="mV",
        )
        values_si, quantity, unit = series_values_si(series)
        self.assertEqual(quantity, "voltage")
        self.assertEqual(unit, "V")
        np.testing.assert_allclose(values_si, [0.0, 0.25])
        np.testing.assert_allclose(series.values, [0.0, 250.0])
        np.testing.assert_allclose(
            convert_signal_values([0.0, 250.0], from_unit="mV", to_unit="V"),
            [0.0, 0.25],
        )

    def test_series_requires_strict_physical_time(self) -> None:
        with self.assertRaisesRegex(WaveformIOError, "strictly increasing"):
            WaveformSeries("bad", [0.0, 0.0, 1.0], [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(WaveformIOError, "at least two"):
            WaveformSeries("bad", [0.0], [1.0])

    def test_registry_requires_explicit_reader_and_rejects_duplicate_registration(self) -> None:
        registry = build_default_waveform_io_registry()
        with self.assertRaisesRegex(WaveformIOError, "already registered"):
            registry.register_reader(NativeWaveformJSON())

        document = WaveformDocument(
            [WaveformSeries("A", [0.0, 1.0], [0.0, 1.0], quantity="current", unit="A")]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mystery.json"
            registry.write(path, document)
            with self.assertRaisesRegex(TypeError, "format_id"):
                registry.read(path)
            decoded = registry.read(path, format_id=NATIVE_WAVEFORM_READER_ID)
        self.assertEqual(decoded.source_format, NATIVE_WAVEFORM_READER_ID)

    def test_registry_is_format_module_extensible(self) -> None:
        class DummyReader:
            format_id = "tests.dummy"
            display_name = "Dummy waveform"
            suffixes = (".dummy",)

            def read(self, path: Path) -> WaveformDocument:
                return WaveformDocument(
                    [WaveformSeries("Dummy", [0.0, 1.0], [1.0, 2.0], quantity="arbitrary")],
                    title=path.stem,
                )

        registry = WaveformIORegistry()
        registry.register_reader(DummyReader())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.dummy"
            path.write_bytes(b"DUMMY\n")
            document = registry.read(path, format_id="tests.dummy")
        self.assertEqual(document.source_format, "tests.dummy")
        self.assertEqual(document.title, "capture")
        self.assertEqual(document.channel(0).sample_count, 2)


if __name__ == "__main__":
    unittest.main()
