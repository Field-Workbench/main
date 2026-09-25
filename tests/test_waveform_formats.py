from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from fieldworkbench.waveform_formats import (
    FORMAT_PROFILE_SCHEMA,
    WaveformFormatProfile,
    WaveformFormatStore,
    builtin_unsigned_8bit_dac_profile,
    builtin_waveform_format_profiles,
    parse_waveform_bytes,
)
from fieldworkbench.waveform_io import WaveformIOError


class WaveformFormatTests(unittest.TestCase):
    def test_text_profile_decodes_unsigned_8bit_newline_values_and_scales(self) -> None:
        profile = WaveformFormatProfile(
            profile_id="tests.dac",
            name="DAC-like text",
            data_mode="text",
            number_format="uint8",
            text_record_separator="newline",
            text_field_separator="none",
            time_mode="fixed_interval",
            sample_interval=2.0,
            sample_interval_unit="ms",
            count_mode="eof",
            output_quantity="voltage",
            output_unit="V",
            scale=10.0 / 255.0,
            offset=-5.0,
        )
        parsed = parse_waveform_bytes(b"0\n128\n255\n", profile)
        np.testing.assert_allclose(parsed.time_s, [0.0, 0.002, 0.004])
        np.testing.assert_allclose(parsed.raw_values, [0.0, 128.0, 255.0])
        np.testing.assert_allclose(parsed.output_values[[0, 2]], [-5.0, 5.0])
        document = parsed.document(title="example")
        self.assertEqual(document.channel(0).quantity, "voltage")
        self.assertEqual(document.channel(0).unit, "V")

    def test_text_profile_supports_header_count_and_explicit_time_field(self) -> None:
        profile = WaveformFormatProfile(
            profile_id="tests.csv",
            name="Header CSV",
            data_mode="text",
            number_format="float64",
            text_field_separator="comma",
            text_skip_records=2,
            text_value_field=2,
            time_mode="field",
            text_time_field=1,
            time_unit="ms",
            count_mode="text_record",
            count_text_record=1,
            count_text_field=2,
            count_number_format="uint16",
            output_quantity="current",
            output_unit="mA",
        )
        payload = b"points,3\ntime,value\n0,1.5\n2,2.5\n4,3.5\n"
        parsed = parse_waveform_bytes(payload, profile)
        np.testing.assert_allclose(parsed.time_s, [0.0, 0.002, 0.004])
        np.testing.assert_allclose(parsed.output_values, [1.5, 2.5, 3.5])

    def test_binary_profile_supports_header_stride_endianness_and_count(self) -> None:
        header = struct.pack("<I", 3) + b"HEAD"
        records = b"".join(
            struct.pack("<fH", time_ms, raw) + b"xx"
            for time_ms, raw in ((0.0, 10), (1.0, 20), (2.0, 30))
        )
        profile = WaveformFormatProfile(
            profile_id="tests.binary",
            name="Binary records",
            data_mode="binary",
            number_format="uint16",
            byte_order="little",
            binary_header_bytes=8,
            binary_record_bytes=8,
            binary_value_offset=4,
            time_mode="field",
            binary_time_offset=0,
            binary_time_format="float32",
            time_unit="ms",
            count_mode="binary_header",
            count_binary_offset=0,
            count_number_format="uint32",
            count_byte_order="little",
            output_quantity="current",
            output_unit="mA",
            scale=0.5,
        )
        parsed = parse_waveform_bytes(header + records, profile)
        np.testing.assert_allclose(parsed.time_s, [0.0, 0.001, 0.002], rtol=1e-7)
        np.testing.assert_allclose(parsed.raw_values, [10, 20, 30])
        np.testing.assert_allclose(parsed.output_values, [5, 10, 15])


    def test_builtin_dac_profile_duplicates_through_generic_profile_api(self) -> None:
        profile = builtin_unsigned_8bit_dac_profile().copy_as_new()
        self.assertTrue(profile.profile_id.startswith("user."))
        self.assertEqual(profile.file_suffixes, (".dac",))
        self.assertEqual(profile.data_mode, "text")
        self.assertEqual(profile.number_format, "uint8")
        self.assertEqual(profile.text_record_separator, "newline")
        self.assertEqual(profile.text_field_separator, "none")
        self.assertEqual(profile.text_skip_records, 1)
        self.assertEqual(profile.count_mode, "text_record")
        self.assertEqual(profile.count_text_record, 1)
        self.assertEqual(profile.count_text_field, 1)
        self.assertEqual(profile.count_number_format, "uint32")
        self.assertEqual(profile.time_mode, "runtime_interval")
        self.assertEqual(profile.conversion_mode, "runtime_linear")
        self.assertEqual(profile.sample_interval_unit, "ms")
        payload = b"3\n0\n128\n255\n"
        with self.assertRaisesRegex(WaveformIOError, "requires a sample interval"):
            parse_waveform_bytes(payload, profile)
        parsed = parse_waveform_bytes(
            payload,
            profile,
            runtime_sample_interval_s=0.002,
            runtime_value_at_min=-5.0,
            runtime_value_at_max=5.0,
        )
        np.testing.assert_allclose(parsed.time_s, [0.0, 0.002, 0.004])
        np.testing.assert_allclose(parsed.raw_values, [0.0, 128.0, 255.0])
        np.testing.assert_allclose(parsed.output_values[[0, 2]], [-5.0, 5.0])


    def test_builtin_dac_header_count_is_enforced(self) -> None:
        profile = builtin_unsigned_8bit_dac_profile()
        with self.assertRaisesRegex(WaveformIOError, "requests 4 samples but only 3"):
            parse_waveform_bytes(
                b"4\n0\n128\n255\n",
                profile,
                runtime_sample_interval_s=0.001,
                runtime_value_at_min=0.0,
                runtime_value_at_max=1.0,
            )

    def test_bundled_dac_is_an_ordinary_runtime_format_profile(self) -> None:
        profiles = builtin_waveform_format_profiles()
        self.assertEqual(len(profiles), 1)
        dac = profiles[0]
        self.assertEqual(dac.profile_id, "builtin.dac_8bit")
        self.assertEqual(dac.time_mode, "runtime_interval")
        self.assertEqual(dac.conversion_mode, "runtime_linear")
        duplicate = dac.copy_as_new()
        self.assertTrue(duplicate.profile_id.startswith("user."))
        self.assertEqual(duplicate.time_mode, dac.time_mode)
        self.assertEqual(duplicate.conversion_mode, dac.conversion_mode)

    def test_user_profile_can_request_runtime_timing_without_dac_specific_logic(self) -> None:
        profile = WaveformFormatProfile(
            profile_id="tests.runtime",
            name="Runtime bytes",
            data_mode="binary",
            number_format="uint8",
            binary_record_bytes=1,
            time_mode="runtime_interval",
            sample_interval=2.0,
            sample_interval_unit="ms",
            conversion_mode="fixed_linear",
            scale=0.5,
            output_quantity="current",
            output_unit="mA",
        )
        parsed = parse_waveform_bytes(bytes((2, 4, 6)), profile, runtime_sample_interval_s=0.005)
        np.testing.assert_allclose(parsed.time_s, [0.0, 0.005, 0.010])
        np.testing.assert_allclose(parsed.output_values, [1.0, 2.0, 3.0])

    def test_profile_store_round_trip_and_portable_schema(self) -> None:
        profile = WaveformFormatProfile.new("My lab format")
        with tempfile.TemporaryDirectory() as directory:
            store = WaveformFormatStore(directory)
            path = store.save(profile)
            loaded = store.profiles()
            portable = WaveformFormatProfile.load(path)
            self.assertEqual(portable.to_dict()["schema"], FORMAT_PROFILE_SCHEMA)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].profile_id, profile.profile_id)
            store.delete(profile.profile_id)
            self.assertEqual(store.profiles(), ())

    def test_integer_range_and_invalid_binary_record_are_rejected(self) -> None:
        text_profile = WaveformFormatProfile(
            profile_id="tests.u8",
            name="u8",
            number_format="uint8",
            sample_interval=1,
            sample_interval_unit="ms",
        )
        with self.assertRaisesRegex(WaveformIOError, "outside the uint8 range"):
            parse_waveform_bytes(b"0\n300\n", text_profile)
        binary_profile = WaveformFormatProfile(
            profile_id="tests.badbin",
            name="bad",
            data_mode="binary",
            number_format="uint16",
            binary_record_bytes=1,
            sample_interval=1,
            sample_interval_unit="ms",
        )
        with self.assertRaisesRegex(WaveformIOError, "too small"):
            parse_waveform_bytes(b"\x00\x01", binary_profile)


if __name__ == "__main__":
    unittest.main()
