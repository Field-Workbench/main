"""Declarative waveform format profiles and general sample decoding.

A format profile describes *where sampled values live in a file*.  It does not
know anything about scene coils or playback.  Text and binary profiles share the
same normalized result so later vendor modules can either use this machinery or
provide a dedicated reader when a format is too complex to express declaratively.

Waveform import uses explicit format selection.  There is no automatic
probing/detection path here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import struct
from typing import Any
from uuid import uuid4

import numpy as np

from .waveform_io import WaveformDocument, WaveformIOError, WaveformSeries, time_scale_to_seconds


FORMAT_PROFILE_SCHEMA = "fieldworkbench.waveform-format"
FORMAT_PROFILE_VERSION = 1
FORMAT_PROFILE_SUFFIX = ".fwbformat.json"

_NUMBER_FORMATS: dict[str, tuple[str, int | None, bool | None]] = {
    "uint8": ("B", 1, False),
    "int8": ("b", 1, True),
    "uint16": ("H", 2, False),
    "int16": ("h", 2, True),
    "uint32": ("I", 4, False),
    "int32": ("i", 4, True),
    "uint64": ("Q", 8, False),
    "int64": ("q", 8, True),
    "float32": ("f", 4, None),
    "float64": ("d", 8, None),
}


@dataclass(frozen=True)
class WaveformFormatProfile:
    """Portable description of a simple sampled waveform file.

    Field/record numbers exposed to users are one-based.  Binary offsets are
    byte offsets.  ``scale`` and ``offset`` define the final engineering value:
    ``output = raw * scale + offset``.
    """

    profile_id: str
    name: str
    description: str = ""
    file_suffixes: tuple[str, ...] = ()
    data_mode: str = "text"  # text | binary

    # Raw sample representation.
    number_format: str = "float64"
    byte_order: str = "little"  # little | big

    # Text layout.
    text_encoding: str = "utf-8-sig"
    text_record_separator: str = "newline"  # newline | lf | crlf | custom
    text_custom_record_separator: str = ""
    text_field_separator: str = "none"  # none | whitespace | comma | tab | semicolon | custom
    text_custom_field_separator: str = ""
    text_comment_prefix: str = ""
    text_skip_records: int = 0
    text_value_field: int = 1

    # Binary layout.
    binary_header_bytes: int = 0
    binary_record_bytes: int = 0  # 0 = minimum packed record size
    binary_value_offset: int = 0

    # Timeline.  ``runtime_interval`` means the file does not encode sample
    # timing; the importer asks for the interval when the file is opened.
    time_mode: str = "fixed_interval"  # fixed_interval | field | runtime_interval
    sample_interval: float = 1.0
    sample_interval_unit: str = "ms"
    time_unit: str = "s"
    text_time_field: int = 1
    binary_time_offset: int = 0
    binary_time_format: str = "float64"

    # Record count / termination.
    count_mode: str = "eof"  # eof | fixed | text_record | binary_header
    fixed_count: int = 0
    count_text_record: int = 1
    count_text_field: int = 1
    count_binary_offset: int = 0
    count_number_format: str = "uint32"
    count_byte_order: str = "little"

    # Meaning/conversion of the parsed raw sample. Runtime linear mapping lets
    # any encoding ask for the engineering values corresponding to two raw
    # reference values when the file is imported.
    output_quantity: str = "current"  # current | voltage
    output_unit: str = "mA"
    conversion_mode: str = "fixed_linear"  # fixed_linear | runtime_linear
    scale: float = 1.0
    offset: float = 0.0
    runtime_raw_min: float = 0.0
    runtime_raw_max: float = 255.0
    runtime_value_at_min: float = 0.0
    runtime_value_at_max: float = 1.0

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id or "").strip()
        name = str(self.name or "").strip()
        if not profile_id:
            raise WaveformIOError("Waveform format profiles require a stable profile id.")
        if not name:
            raise WaveformIOError("Waveform format profiles require a name.")
        if self.data_mode not in {"text", "binary"}:
            raise WaveformIOError("Waveform format data mode must be text or binary.")
        if self.number_format not in _NUMBER_FORMATS:
            raise WaveformIOError(f"Unsupported sample number format {self.number_format!r}.")
        if self.byte_order not in {"little", "big"}:
            raise WaveformIOError("Binary byte order must be little or big endian.")
        if self.time_mode not in {"fixed_interval", "field", "runtime_interval"}:
            raise WaveformIOError("Waveform time source must be fixed_interval, field, or runtime_interval.")
        if self.binary_time_format not in _NUMBER_FORMATS:
            raise WaveformIOError(f"Unsupported binary time format {self.binary_time_format!r}.")
        if self.count_mode not in {"eof", "fixed", "text_record", "binary_header"}:
            raise WaveformIOError(f"Unsupported sample-count mode {self.count_mode!r}.")
        if self.count_number_format not in _NUMBER_FORMATS:
            raise WaveformIOError(f"Unsupported count number format {self.count_number_format!r}.")
        if self.count_byte_order not in {"little", "big"}:
            raise WaveformIOError("Count byte order must be little or big endian.")
        if self.text_record_separator not in {"newline", "lf", "crlf", "custom"}:
            raise WaveformIOError("Unsupported text record separator.")
        if self.text_field_separator not in {"none", "whitespace", "comma", "tab", "semicolon", "custom"}:
            raise WaveformIOError("Unsupported text field separator.")
        for label, value in (
            ("text_skip_records", self.text_skip_records),
            ("binary_header_bytes", self.binary_header_bytes),
            ("binary_record_bytes", self.binary_record_bytes),
            ("binary_value_offset", self.binary_value_offset),
            ("binary_time_offset", self.binary_time_offset),
            ("count_binary_offset", self.count_binary_offset),
        ):
            if int(value) < 0:
                raise WaveformIOError(f"{label} cannot be negative.")
        for label, value in (
            ("text_value_field", self.text_value_field),
            ("text_time_field", self.text_time_field),
            ("count_text_record", self.count_text_record),
            ("count_text_field", self.count_text_field),
        ):
            if int(value) < 1:
                raise WaveformIOError(f"{label} must be 1 or greater.")
        if self.count_mode == "fixed" and int(self.fixed_count) < 1:
            raise WaveformIOError("A fixed sample count must be at least 1.")
        if self.time_mode in {"fixed_interval", "runtime_interval"}:
            try:
                interval = float(self.sample_interval)
            except (TypeError, ValueError) as error:
                raise WaveformIOError("Sample interval must be numeric.") from error
            if not math.isfinite(interval) or interval <= 0.0:
                raise WaveformIOError("Sample interval must be finite and greater than zero.")
            time_scale_to_seconds(self.sample_interval_unit)
        else:
            time_scale_to_seconds(self.time_unit)
        if self.conversion_mode not in {"fixed_linear", "runtime_linear"}:
            raise WaveformIOError("Waveform conversion mode must be fixed_linear or runtime_linear.")
        for label, value in (("scale", self.scale), ("offset", self.offset)):
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise WaveformIOError(f"Waveform {label} must be numeric.") from error
            if not math.isfinite(number):
                raise WaveformIOError(f"Waveform {label} must be finite.")
        for label, value in (
            ("runtime_raw_min", self.runtime_raw_min),
            ("runtime_raw_max", self.runtime_raw_max),
            ("runtime_value_at_min", self.runtime_value_at_min),
            ("runtime_value_at_max", self.runtime_value_at_max),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise WaveformIOError(f"Waveform {label} must be numeric.") from error
            if not math.isfinite(number):
                raise WaveformIOError(f"Waveform {label} must be finite.")
        if self.conversion_mode == "runtime_linear" and math.isclose(
            float(self.runtime_raw_min), float(self.runtime_raw_max), abs_tol=0.0
        ):
            raise WaveformIOError("Runtime linear mapping requires two different raw reference values.")
        suffixes = []
        for suffix in self.file_suffixes:
            text = str(suffix or "").strip().lower()
            if not text:
                continue
            if not text.startswith("."):
                text = "." + text
            if text not in suffixes:
                suffixes.append(text)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "file_suffixes", tuple(suffixes))

    @classmethod
    def new(cls, name: str = "New waveform format") -> "WaveformFormatProfile":
        return cls(profile_id=f"user.{uuid4().hex}", name=name)

    def copy_as_new(self, *, name: str | None = None) -> "WaveformFormatProfile":
        return replace(
            self,
            profile_id=f"user.{uuid4().hex}",
            name=str(name or f"{self.name} copy").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["file_suffixes"] = list(self.file_suffixes)
        return {
            "schema": FORMAT_PROFILE_SCHEMA,
            "version": FORMAT_PROFILE_VERSION,
            "profile": payload,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WaveformFormatProfile":
        if not isinstance(payload, dict):
            raise WaveformIOError("Waveform format profile must be a JSON object.")
        if payload.get("schema") != FORMAT_PROFILE_SCHEMA:
            raise WaveformIOError("Not a Field Workbench waveform format profile.")
        try:
            version = int(payload.get("version", 0))
        except (TypeError, ValueError) as error:
            raise WaveformIOError("Waveform format profile version must be an integer.") from error
        if version != FORMAT_PROFILE_VERSION:
            raise WaveformIOError(
                f"Unsupported waveform format profile version {version}; expected {FORMAT_PROFILE_VERSION}."
            )
        raw = payload.get("profile")
        if not isinstance(raw, dict):
            raise WaveformIOError("Waveform format profile payload is missing.")
        data = dict(raw)
        data["file_suffixes"] = tuple(data.get("file_suffixes") or ())
        try:
            return cls(**data)
        except TypeError as error:
            raise WaveformIOError(f"Invalid waveform format profile fields: {error}") from error

    @classmethod
    def load(cls, path: str | Path) -> "WaveformFormatProfile":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WaveformIOError(f"Unable to read waveform format profile {source.name!r}: {error}") from error
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            raise WaveformIOError(f"Unable to save waveform format profile {destination.name!r}: {error}") from error



def builtin_unsigned_8bit_dac_profile() -> WaveformFormatProfile:
    """The ordinary declarative profile used for legacy unsigned 8-bit DAC files.

    Nothing in the importer keys off the filename or this profile id.  Its
    editable-at-import behaviour follows solely from the same runtime timing and
    runtime linear-mapping flags exposed by the format builder.
    """
    return WaveformFormatProfile(
        profile_id="builtin.dac_8bit",
        name="Unsigned 8-bit DAC",
        description=(
            "Line-oriented text DAC: the first line is the point count and each "
            "following line is one decimal 8-bit sample value (0..255)."
        ),
        file_suffixes=(".dac",),
        data_mode="text",
        number_format="uint8",
        text_encoding="utf-8-sig",
        text_record_separator="newline",
        text_field_separator="none",
        text_skip_records=1,
        text_value_field=1,
        time_mode="runtime_interval",
        sample_interval=1.0,
        sample_interval_unit="ms",
        count_mode="text_record",
        count_text_record=1,
        count_text_field=1,
        count_number_format="uint32",
        output_quantity="current",
        output_unit="mA",
        conversion_mode="runtime_linear",
        runtime_raw_min=0.0,
        runtime_raw_max=255.0,
        runtime_value_at_min=0.0,
        runtime_value_at_max=1.0,
    )


def builtin_waveform_format_profiles() -> tuple[WaveformFormatProfile, ...]:
    """Return bundled profiles handled by exactly the same declarative path as user profiles."""
    return (builtin_unsigned_8bit_dac_profile(),)


@dataclass(frozen=True)
class ParsedWaveformData:
    """Decoded samples before they are mapped onto a playback drive."""

    time_s: np.ndarray
    raw_values: np.ndarray
    output_values: np.ndarray
    profile: WaveformFormatProfile

    def document(self, *, title: str = "", source_path: str = "") -> WaveformDocument:
        return WaveformDocument(
            [
                WaveformSeries(
                    self.profile.name,
                    self.time_s,
                    self.output_values,
                    quantity=self.profile.output_quantity,
                    unit=self.profile.output_unit,
                    metadata={"format_profile_id": self.profile.profile_id},
                )
            ],
            title=title or self.profile.name,
            metadata={
                "format_profile": self.profile.name,
                "import_settings": {
                    "format": "format_profile",
                    "profile": self.profile.to_dict(),
                },
            },
            source_format=f"profile:{self.profile.profile_id}",
            source_path=source_path,
        )


def _byte_order_prefix(order: str) -> str:
    return "<" if order == "little" else ">"


def _number_size(number_format: str) -> int:
    try:
        return int(_NUMBER_FORMATS[number_format][1] or 0)
    except KeyError as error:
        raise WaveformIOError(f"Unsupported number format {number_format!r}.") from error


def _unpack_number(payload: bytes, offset: int, number_format: str, byte_order: str) -> float:
    code, size, _signed = _NUMBER_FORMATS[number_format]
    assert size is not None
    if offset < 0 or offset + size > len(payload):
        raise WaveformIOError("Binary field extends beyond the available record/header bytes.")
    return float(struct.unpack_from(_byte_order_prefix(byte_order) + code, payload, offset)[0])


def _parse_text_number(text: str, number_format: str) -> float:
    token = str(text).strip()
    if not token:
        raise WaveformIOError("Encountered an empty numeric field.")
    _code, size, signed = _NUMBER_FORMATS[number_format]
    if signed is None:
        try:
            value = float(token)
        except ValueError as error:
            raise WaveformIOError(f"{token!r} is not a valid floating-point sample.") from error
        if not math.isfinite(value):
            raise WaveformIOError("Waveform samples must be finite.")
        return value
    try:
        value = int(token, 0) if re.match(r"^[+-]?0[xob]", token, flags=re.IGNORECASE) else int(token, 10)
    except ValueError as error:
        raise WaveformIOError(f"{token!r} is not a valid integer sample.") from error
    bits = int(size or 1) * 8
    if signed:
        minimum, maximum = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        minimum, maximum = 0, 2**bits - 1
    if value < minimum or value > maximum:
        raise WaveformIOError(
            f"Integer sample {value} is outside the {number_format} range {minimum} to {maximum}."
        )
    return float(value)


def _split_text_records(text: str, profile: WaveformFormatProfile) -> list[str]:
    mode = profile.text_record_separator
    if mode == "newline":
        records = text.splitlines()
    elif mode == "lf":
        records = text.split("\n")
    elif mode == "crlf":
        records = text.split("\r\n")
    else:
        separator = bytes(profile.text_custom_record_separator, "utf-8").decode("unicode_escape")
        if not separator:
            raise WaveformIOError("Custom text record separator cannot be empty.")
        records = text.split(separator)
    prefix = profile.text_comment_prefix
    if prefix:
        return [record for record in records if not record.lstrip().startswith(prefix)]
    return records


def _split_text_fields(record: str, profile: WaveformFormatProfile) -> list[str]:
    mode = profile.text_field_separator
    if mode == "none":
        return [record.strip()]
    if mode == "whitespace":
        return record.split()
    separators = {"comma": ",", "tab": "\t", "semicolon": ";"}
    separator = separators.get(mode)
    if mode == "custom":
        separator = bytes(profile.text_custom_field_separator, "utf-8").decode("unicode_escape")
        if not separator:
            raise WaveformIOError("Custom text field separator cannot be empty.")
    assert separator is not None
    return [field.strip() for field in record.split(separator)]


def _text_field(record: str, field_number: int, profile: WaveformFormatProfile) -> str:
    fields = _split_text_fields(record, profile)
    index = int(field_number) - 1
    if index < 0 or index >= len(fields):
        raise WaveformIOError(
            f"Text record has {len(fields)} field(s), but profile requests field {field_number}."
        )
    return fields[index]


def _sample_count_from_text(records: list[str], profile: WaveformFormatProfile) -> int | None:
    if profile.count_mode == "eof":
        return None
    if profile.count_mode == "fixed":
        return int(profile.fixed_count)
    if profile.count_mode != "text_record":
        raise WaveformIOError("A text profile cannot use a binary-header sample count.")
    index = int(profile.count_text_record) - 1
    if index < 0 or index >= len(records):
        raise WaveformIOError(
            f"Sample-count record {profile.count_text_record} is outside the text file."
        )
    value = _parse_text_number(
        _text_field(records[index], profile.count_text_field, profile),
        profile.count_number_format,
    )
    count = int(value)
    if value != count or count < 1:
        raise WaveformIOError("Sample count stored in the text file must be a positive integer.")
    return count


def _parse_text(payload: bytes, profile: WaveformFormatProfile) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        text = payload.decode(profile.text_encoding)
    except (LookupError, UnicodeDecodeError) as error:
        raise WaveformIOError(
            f"Unable to decode text waveform using {profile.text_encoding!r}: {error}"
        ) from error
    records = _split_text_records(text, profile)
    requested_count = _sample_count_from_text(records, profile)
    data_records = records[int(profile.text_skip_records) :]
    values: list[float] = []
    times: list[float] = []
    for record_number, record in enumerate(data_records, start=int(profile.text_skip_records) + 1):
        if not record.strip():
            continue
        try:
            values.append(
                _parse_text_number(_text_field(record, profile.text_value_field, profile), profile.number_format)
            )
            if profile.time_mode == "field":
                times.append(float(_text_field(record, profile.text_time_field, profile)))
        except WaveformIOError as error:
            raise WaveformIOError(f"Text record {record_number}: {error}") from error
        if requested_count is not None and len(values) >= requested_count:
            break
    if not values:
        raise WaveformIOError("The selected text file did not yield any waveform samples.")
    if requested_count is not None and len(values) != requested_count:
        raise WaveformIOError(
            f"Format requests {requested_count} samples but only {len(values)} data records were decoded."
        )
    return np.asarray(values, dtype=float), np.asarray(times, dtype=float) if times else None


def _sample_count_from_binary(payload: bytes, profile: WaveformFormatProfile, record_bytes: int) -> int:
    available = len(payload) - int(profile.binary_header_bytes)
    if available < 0:
        raise WaveformIOError("Binary header size exceeds the selected file size.")
    if profile.count_mode == "fixed":
        return int(profile.fixed_count)
    if profile.count_mode == "binary_header":
        value = _unpack_number(
            payload,
            int(profile.count_binary_offset),
            profile.count_number_format,
            profile.count_byte_order,
        )
        count = int(value)
        if value != count or count < 1:
            raise WaveformIOError("Binary header sample count must be a positive integer.")
        return count
    if profile.count_mode != "eof":
        raise WaveformIOError("A binary profile cannot use a text-record sample count.")
    if record_bytes <= 0:
        raise WaveformIOError("Binary record size must be greater than zero.")
    if available % record_bytes:
        raise WaveformIOError(
            f"Binary payload has {available} data bytes, which is not an exact multiple of the {record_bytes}-byte record size."
        )
    return available // record_bytes


def _parse_binary(payload: bytes, profile: WaveformFormatProfile) -> tuple[np.ndarray, np.ndarray | None]:
    value_size = _number_size(profile.number_format)
    minimum_record = int(profile.binary_value_offset) + value_size
    if profile.time_mode == "field":
        minimum_record = max(
            minimum_record,
            int(profile.binary_time_offset) + _number_size(profile.binary_time_format),
        )
    record_bytes = int(profile.binary_record_bytes) or minimum_record
    if record_bytes < minimum_record:
        raise WaveformIOError(
            f"Binary record size {record_bytes} is too small for the configured fields ({minimum_record} bytes required)."
        )
    count = _sample_count_from_binary(payload, profile, record_bytes)
    start = int(profile.binary_header_bytes)
    required = start + count * record_bytes
    if required > len(payload):
        raise WaveformIOError(
            f"Format requests {count} binary records ({required} bytes total) but file contains only {len(payload)} bytes."
        )
    values = np.empty(count, dtype=float)
    times = np.empty(count, dtype=float) if profile.time_mode == "field" else None
    for index in range(count):
        base = start + index * record_bytes
        values[index] = _unpack_number(
            payload,
            base + int(profile.binary_value_offset),
            profile.number_format,
            profile.byte_order,
        )
        if times is not None:
            times[index] = _unpack_number(
                payload,
                base + int(profile.binary_time_offset),
                profile.binary_time_format,
                profile.byte_order,
            )
    return values, times


def parse_waveform_bytes(
    payload: bytes,
    profile: WaveformFormatProfile,
    *,
    runtime_sample_interval_s: float | None = None,
    runtime_value_at_min: float | None = None,
    runtime_value_at_max: float | None = None,
) -> ParsedWaveformData:
    """Decode bytes with an explicitly selected declarative format profile.

    Runtime values are accepted only because the selected profile requests
    them; callers never infer them from a file suffix or payload.
    """
    if not isinstance(profile, WaveformFormatProfile):
        raise WaveformIOError("A WaveformFormatProfile is required for general-format decoding.")
    data = bytes(payload)
    if not data:
        raise WaveformIOError("The selected waveform file is empty.")
    raw_values, explicit_times = (
        _parse_text(data, profile) if profile.data_mode == "text" else _parse_binary(data, profile)
    )
    if explicit_times is None:
        if profile.time_mode == "runtime_interval":
            if runtime_sample_interval_s is None:
                raise WaveformIOError(f"{profile.name} requires a sample interval at import time.")
            dt = float(runtime_sample_interval_s)
            if not math.isfinite(dt) or dt <= 0.0:
                raise WaveformIOError("Runtime sample interval must be finite and greater than zero.")
        else:
            dt = float(profile.sample_interval) * time_scale_to_seconds(profile.sample_interval_unit)
        time_s = np.arange(len(raw_values), dtype=float) * dt
    else:
        time_s = explicit_times * time_scale_to_seconds(profile.time_unit)
    if len(time_s) < 2:
        raise WaveformIOError("Waveform import requires at least two samples.")
    if not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0):
        raise WaveformIOError("Imported waveform time values must be finite and strictly increasing.")
    if profile.conversion_mode == "runtime_linear":
        low = float(profile.runtime_value_at_min if runtime_value_at_min is None else runtime_value_at_min)
        high = float(profile.runtime_value_at_max if runtime_value_at_max is None else runtime_value_at_max)
        raw_low = float(profile.runtime_raw_min)
        raw_high = float(profile.runtime_raw_max)
        scale = (high - low) / (raw_high - raw_low)
        offset = low - raw_low * scale
        output = raw_values * scale + offset
    else:
        output = raw_values * float(profile.scale) + float(profile.offset)
    if not np.isfinite(output).all():
        raise WaveformIOError("Scaled waveform values must be finite.")
    return ParsedWaveformData(time_s, raw_values, output, profile)


def parse_waveform_file(
    path: str | Path,
    profile: WaveformFormatProfile,
    *,
    runtime_sample_interval_s: float | None = None,
    runtime_value_at_min: float | None = None,
    runtime_value_at_max: float | None = None,
) -> ParsedWaveformData:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise WaveformIOError(f"Unable to read waveform file {source}: {error}") from error
    return parse_waveform_bytes(
        payload,
        profile,
        runtime_sample_interval_s=runtime_sample_interval_s,
        runtime_value_at_min=runtime_value_at_min,
        runtime_value_at_max=runtime_value_at_max,
    )


class WaveformFormatStore:
    """Directory-backed collection of user-created declarative format profiles."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def profiles(self) -> tuple[WaveformFormatProfile, ...]:
        if not self.directory.exists():
            return ()
        profiles: list[WaveformFormatProfile] = []
        for path in sorted(self.directory.glob(f"*{FORMAT_PROFILE_SUFFIX}")):
            try:
                profiles.append(WaveformFormatProfile.load(path))
            except WaveformIOError:
                continue
        return tuple(sorted(profiles, key=lambda profile: profile.name.casefold()))

    def path_for(self, profile: WaveformFormatProfile) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile.profile_id).strip("._") or "format"
        return self.directory / f"{safe_id}{FORMAT_PROFILE_SUFFIX}"

    def save(self, profile: WaveformFormatProfile) -> Path:
        path = self.path_for(profile)
        profile.save(path)
        return path

    def delete(self, profile_id: str) -> None:
        for profile in self.profiles():
            if profile.profile_id == profile_id:
                try:
                    self.path_for(profile).unlink()
                except FileNotFoundError:
                    pass
                return


def number_format_labels() -> tuple[tuple[str, str], ...]:
    return (
        ("Unsigned 8-bit integer", "uint8"),
        ("Signed 8-bit integer", "int8"),
        ("Unsigned 16-bit integer", "uint16"),
        ("Signed 16-bit integer", "int16"),
        ("Unsigned 32-bit integer", "uint32"),
        ("Signed 32-bit integer", "int32"),
        ("Unsigned 64-bit integer", "uint64"),
        ("Signed 64-bit integer", "int64"),
        ("32-bit floating point", "float32"),
        ("64-bit floating point", "float64"),
    )
