"""Format-neutral waveform import/export infrastructure.

This module intentionally has no Qt dependency.  File-format readers normalize
external data into :class:`WaveformDocument` / :class:`WaveformSeries`; the
playback UI can then map those normalized channels onto Field Workbench drives.

The registry, normalized data model, and native JSON interchange keep external
file details outside playback. Ordinary text/CSV column mapping lives at the
Waveform Editor import boundary, while advanced/vendor readers can be registered
without teaching the playback engine about individual file formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .waveform_recipe import (
    AUTHORING_RECIPE_KEY,
    WaveformRecipeError,
    normalize_authoring_recipe,
)

NATIVE_WAVEFORM_FORMAT = "fieldworkbench.waveform"
NATIVE_WAVEFORM_VERSION = 2  # One canonical electrical source signal
NATIVE_WAVEFORM_READER_ID = "fieldworkbench.waveform-json"
PROJECT_WAVEFORM_SOURCE_FORMAT = "fieldworkbench.project-waveform"


class WaveformIOError(ValueError):
    """Raised when a waveform file cannot be decoded, normalized, or encoded."""


def _copy_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WaveformIOError("Waveform metadata must be a mapping/object.")
    return {str(key): item for key, item in value.items()}


def _normalise_quantity(value: str | None) -> str:
    text = str(value or "arbitrary").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "amps": "current",
        "amp": "current",
        "amperes": "current",
        "ampere": "current",
        "current": "current",
        "volts": "voltage",
        "volt": "voltage",
        "voltage": "voltage",
        "magnetic_field": "magnetic_flux_density",
        "field": "magnetic_flux_density",
        "b": "magnetic_flux_density",
        "magnetic_flux_density": "magnetic_flux_density",
        "arbitrary": "arbitrary",
        "unknown": "arbitrary",
        "": "arbitrary",
    }
    return aliases.get(text, "arbitrary")


def _finite_vector(values: Any, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError) as error:
        raise WaveformIOError(f"{label} must contain numeric values.") from error
    if not len(array):
        raise WaveformIOError(f"{label} cannot be empty.")
    if not np.isfinite(array).all():
        raise WaveformIOError(f"{label} must contain only finite values.")
    return array


@dataclass(frozen=True)
class WaveformSeries:
    """One normalized sampled waveform channel.

    ``time_s`` is always physical time in seconds.  ``values`` deliberately
    retain the channel's declared ``unit`` so importers do not silently change
    source data.  Consumers can use :func:`convert_signal_values` when mapping
    the series onto an SI playback drive.
    """

    name: str
    time_s: np.ndarray
    values: np.ndarray
    quantity: str = "arbitrary"
    unit: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "Waveform").strip() or "Waveform"
        times = _finite_vector(self.time_s, label=f"{name} time axis")
        values = _finite_vector(self.values, label=f"{name} samples")
        if len(times) != len(values):
            raise WaveformIOError(
                f"Waveform channel {name!r} has {len(times)} time values but {len(values)} samples."
            )
        if len(times) < 2:
            raise WaveformIOError(f"Waveform channel {name!r} must contain at least two samples.")
        if np.any(np.diff(times) <= 0.0):
            raise WaveformIOError(
                f"Waveform channel {name!r} time values must be strictly increasing."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "quantity", _normalise_quantity(self.quantity))
        object.__setattr__(self, "unit", str(self.unit or "").strip())
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def sample_count(self) -> int:
        return int(len(self.values))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def nominal_sample_interval_s(self) -> float | None:
        differences = np.diff(self.time_s)
        if not len(differences):
            return None
        candidate = float(np.median(differences))
        return candidate if math.isfinite(candidate) and candidate > 0.0 else None


@dataclass(frozen=True)
class WaveformDocument:
    """Format-neutral waveform file containing one or more sampled channels."""

    channels: Sequence[WaveformSeries]
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_format: str = ""
    source_path: str = ""

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels:
            raise WaveformIOError("A waveform document must contain at least one channel.")
        if not all(isinstance(channel, WaveformSeries) for channel in channels):
            raise WaveformIOError("Waveform document channels must be WaveformSeries objects.")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "title", str(self.title or "").strip())
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
        object.__setattr__(self, "source_format", str(self.source_format or "").strip())
        object.__setattr__(self, "source_path", str(self.source_path or "").strip())

    @property
    def signal(self) -> WaveformSeries:
        """The sole signal in a catalogue waveform; never silently select one."""
        if len(self.channels) != 1:
            raise WaveformIOError("A catalogue waveform must contain exactly one signal. Split the imported signals first.")
        return self.channels[0]

    def split_signals(self) -> tuple["WaveformDocument", ...]:
        """Split file input into independent waveforms without losing provenance."""
        return tuple(
            WaveformDocument(
                [series],
                title=self.title if len(self.channels) == 1 else f"{self.title or 'Waveform'} — {series.name}",
                metadata=self.metadata, source_format=self.source_format, source_path=self.source_path,
            )
            for series in self.channels
        )

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def channel(self, selector: int | str) -> WaveformSeries:
        if isinstance(selector, int):
            try:
                return self.channels[selector]
            except IndexError as error:
                raise WaveformIOError(f"Waveform channel index {selector} is out of range.") from error
        name = str(selector).strip()
        for channel in self.channels:
            if channel.name == name:
                return channel
        lowered = name.casefold()
        matches = [channel for channel in self.channels if channel.name.casefold() == lowered]
        if len(matches) == 1:
            return matches[0]
        raise WaveformIOError(f"Waveform channel {selector!r} was not found.")

    def common_time_s(self) -> np.ndarray | None:
        """Return a copy of the shared time axis, or ``None`` for mixed timelines."""
        reference = self.channels[0].time_s
        for channel in self.channels[1:]:
            if channel.time_s.shape != reference.shape or not np.array_equal(channel.time_s, reference):
                return None
        return np.asarray(reference, dtype=float).copy()


class WaveformReader(Protocol):
    """Protocol implemented by one explicitly selected import-format module."""

    format_id: str
    display_name: str
    suffixes: Sequence[str]

    def read(self, path: Path) -> WaveformDocument:
        """Decode one file into the normalized document model."""


class WaveformWriter(Protocol):
    """Protocol implemented by one export-format module."""

    format_id: str
    display_name: str
    suffixes: Sequence[str]

    def write(self, path: Path, document: WaveformDocument) -> None:
        """Encode a normalized document."""


def _normalise_suffixes(suffixes: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for suffix in suffixes:
        text = str(suffix or "").strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = "." + text
        if text not in result:
            result.append(text)
    return tuple(result)


def _path_matches_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _normalise_suffixes(suffixes))


class WaveformIORegistry:
    """Registry for explicitly selected waveform reader and writer modules.

    Format selection remains explicit: the UI presents registered/native or
    reusable profile choices and the user selects the intended interpretation.
    """

    def __init__(self) -> None:
        self._readers: dict[str, WaveformReader] = {}
        self._writers: dict[str, WaveformWriter] = {}

    @staticmethod
    def _format_id(component: Any) -> str:
        format_id = str(getattr(component, "format_id", "") or "").strip()
        if not format_id:
            raise WaveformIOError("Waveform reader/writer modules must define a non-empty format_id.")
        return format_id

    def register_reader(self, reader: WaveformReader, *, replace: bool = False) -> None:
        format_id = self._format_id(reader)
        if format_id in self._readers and not replace:
            raise WaveformIOError(f"Waveform reader {format_id!r} is already registered.")
        self._readers[format_id] = reader

    def register_writer(self, writer: WaveformWriter, *, replace: bool = False) -> None:
        format_id = self._format_id(writer)
        if format_id in self._writers and not replace:
            raise WaveformIOError(f"Waveform writer {format_id!r} is already registered.")
        self._writers[format_id] = writer

    @property
    def readers(self) -> tuple[WaveformReader, ...]:
        return tuple(self._readers.values())

    @property
    def writers(self) -> tuple[WaveformWriter, ...]:
        return tuple(self._writers.values())

    def reader(self, format_id: str) -> WaveformReader:
        try:
            return self._readers[str(format_id)]
        except KeyError as error:
            raise WaveformIOError(f"Unknown waveform import format {format_id!r}.") from error

    def writer(self, format_id: str) -> WaveformWriter:
        try:
            return self._writers[str(format_id)]
        except KeyError as error:
            raise WaveformIOError(f"Unknown waveform export format {format_id!r}.") from error

    def read(self, path: str | Path, *, format_id: str) -> WaveformDocument:
        source = Path(path)
        if not str(format_id or "").strip():
            raise WaveformIOError("Choose a waveform import format explicitly.")
        reader = self.reader(format_id)
        try:
            document = reader.read(source)
        except WaveformIOError:
            raise
        except Exception as error:  # noqa: BLE001 - reader boundary
            raise WaveformIOError(
                f"Unable to import {source.name!r} with {reader.display_name}: {error}"
            ) from error
        if not isinstance(document, WaveformDocument):
            raise WaveformIOError(
                f"Waveform reader {reader.format_id!r} did not return a WaveformDocument."
            )
        if document.source_format and document.source_format != reader.format_id:
            return document
        return WaveformDocument(
            document.channels,
            title=document.title,
            metadata=document.metadata,
            source_format=reader.format_id,
            source_path=document.source_path or str(source),
        )

    def write(
        self,
        path: str | Path,
        document: WaveformDocument,
        *,
        format_id: str = NATIVE_WAVEFORM_READER_ID,
    ) -> None:
        destination = Path(path)
        writer = self.writer(format_id)
        try:
            writer.write(destination, document)
        except WaveformIOError:
            raise
        except Exception as error:  # noqa: BLE001 - writer boundary
            raise WaveformIOError(
                f"Unable to export {destination.name!r} with {writer.display_name}: {error}"
            ) from error

    def import_filters(self) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Return ``(format_id, display_name, suffixes)`` tuples for a future file picker."""
        return tuple(
            (
                reader.format_id,
                str(reader.display_name),
                _normalise_suffixes(getattr(reader, "suffixes", ())),
            )
            for reader in self.readers
        )

    def export_filters(self) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        return tuple(
            (
                writer.format_id,
                str(writer.display_name),
                _normalise_suffixes(getattr(writer, "suffixes", ())),
            )
            for writer in self.writers
        )


_TIME_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1.0e-3,
    "millisecond": 1.0e-3,
    "milliseconds": 1.0e-3,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "μs": 1.0e-6,
    "microsecond": 1.0e-6,
    "microseconds": 1.0e-6,
    "ns": 1.0e-9,
    "nanosecond": 1.0e-9,
    "nanoseconds": 1.0e-9,
}

_SIGNAL_TO_SI: dict[str, tuple[str, float]] = {
    "a": ("current", 1.0),
    "ma": ("current", 1.0e-3),
    "ua": ("current", 1.0e-6),
    "µa": ("current", 1.0e-6),
    "μa": ("current", 1.0e-6),
    "na": ("current", 1.0e-9),
    "v": ("voltage", 1.0),
    "mv": ("voltage", 1.0e-3),
    "uv": ("voltage", 1.0e-6),
    "µv": ("voltage", 1.0e-6),
    "μv": ("voltage", 1.0e-6),
    "nv": ("voltage", 1.0e-9),
    "t": ("magnetic_flux_density", 1.0),
    "mt": ("magnetic_flux_density", 1.0e-3),
    "ut": ("magnetic_flux_density", 1.0e-6),
    "µt": ("magnetic_flux_density", 1.0e-6),
    "μt": ("magnetic_flux_density", 1.0e-6),
    "nt": ("magnetic_flux_density", 1.0e-9),
}

_SI_UNITS = {
    "current": "A",
    "voltage": "V",
    "magnetic_flux_density": "T",
}


def time_scale_to_seconds(unit: str) -> float:
    """Return the multiplier that converts a recognized time unit to seconds."""
    key = str(unit or "").strip().lower().replace("μ", "µ")
    try:
        return _TIME_TO_SECONDS[key]
    except KeyError as error:
        raise WaveformIOError(f"Unsupported waveform time unit {unit!r}.") from error


def signal_unit_to_si(unit: str, *, quantity: str | None = None) -> tuple[str, str, float]:
    """Return ``(quantity, SI unit, multiplier)`` for a recognized signal unit."""
    key = str(unit or "").strip().lower().replace("μ", "µ").replace(" ", "")
    try:
        detected_quantity, scale = _SIGNAL_TO_SI[key]
    except KeyError as error:
        raise WaveformIOError(f"Unsupported waveform signal unit {unit!r}.") from error
    requested = _normalise_quantity(quantity)
    if quantity is not None and requested not in {"arbitrary", detected_quantity}:
        raise WaveformIOError(
            f"Waveform unit {unit!r} represents {detected_quantity}, not {requested}."
        )
    return detected_quantity, _SI_UNITS[detected_quantity], scale


def convert_signal_values(
    values: Any,
    *,
    from_unit: str,
    to_unit: str,
    quantity: str | None = None,
) -> np.ndarray:
    """Convert current, voltage, or B samples between recognized engineering units."""
    array = _finite_vector(values, label="Waveform samples")
    from_quantity, _from_si, from_scale = signal_unit_to_si(from_unit, quantity=quantity)
    to_quantity, _to_si, to_scale = signal_unit_to_si(to_unit, quantity=quantity)
    if from_quantity != to_quantity:
        raise WaveformIOError(
            f"Cannot convert waveform values from {from_unit!r} to incompatible unit {to_unit!r}."
        )
    return array * (from_scale / to_scale)


def series_values_si(series: WaveformSeries) -> tuple[np.ndarray, str, str]:
    """Return ``(values, quantity, SI unit)`` without mutating the imported series."""
    quantity, si_unit, scale = signal_unit_to_si(series.unit, quantity=series.quantity)
    return np.asarray(series.values, dtype=float) * scale, quantity, si_unit


def waveform_document_from_common_time(
    time_values: Any,
    channels: Iterable[tuple[str, Any, str, str]],
    *,
    time_unit: str = "s",
    title: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> WaveformDocument:
    """Convenience constructor for simple tabular importers.

    Each channel tuple is ``(name, values, quantity, unit)``. This remains a
    convenient bridge for table/vendor readers at the import boundary.
    """
    times = _finite_vector(time_values, label="Waveform time column") * time_scale_to_seconds(time_unit)
    series = [
        WaveformSeries(name, times, values, quantity=quantity, unit=unit)
        for name, values, quantity, unit in channels
    ]
    return WaveformDocument(series, title=title, metadata=metadata or {})


def _canonical_source_electrical_metadata(
    value: Any,
    *,
    quantity: str,
) -> dict[str, Any]:
    """Validate the one authoritative electrical-source metadata object.

    Catalogue waveforms store source characteristics only. Current and voltage
    sources may define slew; voltage sources additionally define non-negative
    series output resistance (0 ohm = ideal source). Load resistance/inductance
    belongs exclusively to model coils, so unknown electrical fields are errors
    rather than compatibility data to migrate or silently discard.
    """
    if value is None:
        electrical: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        electrical = dict(value)
    else:
        raise WaveformIOError("Waveform electrical metadata must be an object.")

    allowed = {"slew_rate_per_s"}
    if quantity == "voltage":
        allowed.add("output_resistance_ohm")
    unknown = sorted(set(electrical) - allowed)
    if unknown:
        raise WaveformIOError(
            "Unsupported waveform electrical field(s): " + ", ".join(unknown) + "."
        )

    slew = electrical.get("slew_rate_per_s", 0.0)
    try:
        slew = float(slew)
    except (TypeError, ValueError) as error:
        raise WaveformIOError("Waveform slew rate must be a finite non-negative number.") from error
    if not math.isfinite(slew) or slew < 0.0:
        raise WaveformIOError("Waveform slew rate must be a finite non-negative number.")
    result: dict[str, Any] = {"slew_rate_per_s": slew}

    if quantity == "voltage":
        resistance = electrical.get("output_resistance_ohm", 0.0)
        try:
            resistance = float(resistance)
        except (TypeError, ValueError) as error:
            raise WaveformIOError(
                "Voltage-source output resistance must be a finite non-negative number."
            ) from error
        if not math.isfinite(resistance) or resistance < 0.0:
            raise WaveformIOError(
                "Voltage-source output resistance must be a finite non-negative number."
            )
        result["output_resistance_ohm"] = resistance
    return result

def waveform_signal_to_payload(document: WaveformDocument) -> dict[str, Any]:
    """Canonical single-signal project/native waveform record (version 2).

    Multi-signal documents exist only at file-import boundaries. Voltage-source
    output resistance is source metadata; physical coil/load R/L is excluded.
    """
    signal = document.signal
    units = {"current": {"A", "mA"}, "voltage": {"V", "mV"}}
    if signal.quantity not in units or signal.unit not in units[signal.quantity]:
        raise WaveformIOError("Catalogue waveforms require Current (A/mA) or Voltage (V/mV).")
    payload = {
        "format": NATIVE_WAVEFORM_FORMAT,
        "version": NATIVE_WAVEFORM_VERSION,
        "title": document.title,
        "metadata": dict(document.metadata),
        "signal": {
            "name": signal.name,
            "quantity": signal.quantity,
            "unit": signal.unit,
            "time_s": [float(value) for value in signal.time_s],
            "values": [float(value) for value in signal.values],
            "metadata": dict(signal.metadata),
        },
    }
    payload["metadata"]["electrical"] = _canonical_source_electrical_metadata(
        payload["metadata"].get("electrical"),
        quantity=signal.quantity,
    )
    signal_metadata = payload["signal"]["metadata"]
    if "electrical" in signal_metadata:
        raise WaveformIOError(
            "Catalogue waveform electrical metadata belongs to the waveform, not the sampled signal."
        )

    # The versioned authoring recipe belongs to the waveform as a whole, not
    # to its sampled signal payload. Validate/canonicalize it at every project
    # and native-file serialization boundary so stale UI metadata cannot become
    # an undocumented second waveform format.
    payload["signal"]["metadata"].pop(AUTHORING_RECIPE_KEY, None)
    if AUTHORING_RECIPE_KEY in payload["metadata"]:
        try:
            payload["metadata"][AUTHORING_RECIPE_KEY] = normalize_authoring_recipe(
                payload["metadata"][AUTHORING_RECIPE_KEY]
            )
        except WaveformRecipeError as error:
            raise WaveformIOError(str(error)) from error
    return payload


def waveform_document_from_payload(
    payload: Mapping[str, Any],
    *,
    source_format: str = NATIVE_WAVEFORM_READER_ID,
    source_path: str = "",
) -> WaveformDocument:
    """Decode the native waveform mapping used by files and project resources."""
    if not isinstance(payload, Mapping):
        raise WaveformIOError("Native waveform data must contain one object.")
    if payload.get("format") != NATIVE_WAVEFORM_FORMAT:
        raise WaveformIOError(
            f"Unsupported waveform format marker {payload.get('format')!r}; expected {NATIVE_WAVEFORM_FORMAT!r}."
        )
    try:
        version = int(payload.get("version", 0))
    except (TypeError, ValueError) as error:
        raise WaveformIOError("Native waveform format version must be an integer.") from error
    if version != NATIVE_WAVEFORM_VERSION:
        raise WaveformIOError(
            f"Unsupported native waveform version {version}; expected {NATIVE_WAVEFORM_VERSION}."
        )
    if "channels" in payload or not isinstance(payload.get("signal"), Mapping):
        raise WaveformIOError(
            "Native waveform data must contain exactly one signal object and no channels list."
        )
    raw = payload["signal"]
    channels = [
        WaveformSeries(
            str(raw.get("name") or "Waveform"),
            raw.get("time_s", []),
            raw.get("values", []),
            quantity=str(raw.get("quantity") or ""),
            unit=str(raw.get("unit") or ""),
            metadata=raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {},
        )
    ]
    metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}
    quantity = channels[0].quantity
    units = {"current": {"A", "mA"}, "voltage": {"V", "mV"}}
    if quantity not in units or channels[0].unit not in units[quantity]:
        raise WaveformIOError("Native waveforms require Current (A/mA) or Voltage (V/mV).")
    metadata["electrical"] = _canonical_source_electrical_metadata(
        metadata.get("electrical"), quantity=quantity
    )
    if "electrical" in channels[0].metadata:
        raise WaveformIOError(
            "Catalogue waveform electrical metadata belongs to the waveform, not the sampled signal."
        )
    if AUTHORING_RECIPE_KEY in metadata:
        try:
            metadata[AUTHORING_RECIPE_KEY] = normalize_authoring_recipe(metadata[AUTHORING_RECIPE_KEY])
        except WaveformRecipeError as error:
            raise WaveformIOError(str(error)) from error
    return WaveformDocument(
        channels,
        title=str(payload.get("title") or ""),
        metadata=metadata,
        source_format=source_format,
        source_path=source_path,
    )


def read_project_waveform_documents(path: str | Path) -> tuple[WaveformDocument, ...]:
    """Read saved waveform resources directly from a Field Workbench project.

    Only the project's ``field_workbench.waveforms`` metadata is decoded; model
    objects, meshes, saved maps, and other project resources are not constructed.
    Malformed optional waveform records are ignored, matching normal project load
    behaviour. Each returned document keeps its own source/electrical/authoring
    metadata and uses the source catalogue resource name as its title.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WaveformIOError(
            f"Unable to read Field Workbench project {source.name!r}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise WaveformIOError("The selected file is not a Field Workbench project document.")
    metadata = payload.get("field_workbench")
    if not isinstance(metadata, Mapping):
        raise WaveformIOError("The selected project does not contain any saved waveforms.")
    resources = metadata.get("waveforms", [])
    if not isinstance(resources, list):
        raise WaveformIOError("The selected project has malformed waveform metadata.")
    if not resources:
        raise WaveformIOError("The selected project does not contain any saved waveforms.")

    documents: list[WaveformDocument] = []
    for raw in resources:
        if not isinstance(raw, Mapping):
            continue
        try:
            document = waveform_document_from_payload(
                raw.get("document"),
                source_format=PROJECT_WAVEFORM_SOURCE_FORMAT,
                source_path=str(source),
            )
        except WaveformIOError:
            continue
        resource_name = str(
            raw.get("name") or document.title or document.signal.name or "Waveform"
        ).strip() or "Waveform"
        if document.title != resource_name:
            document = WaveformDocument(
                document.channels,
                title=resource_name,
                metadata=document.metadata,
                source_format=document.source_format,
                source_path=document.source_path,
            )
        documents.append(document)
    if not documents:
        raise WaveformIOError("The selected project does not contain any valid saved waveforms.")
    return tuple(documents)


class NativeWaveformJSON:
    """Reader/writer for the documented Field Workbench waveform interchange file."""

    format_id = NATIVE_WAVEFORM_READER_ID
    display_name = "Field Workbench waveform JSON"
    suffixes = (".fwbwave", ".fwbwave.json")

    def read(self, path: Path) -> WaveformDocument:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WaveformIOError(f"Unable to decode native waveform JSON: {error}") from error
        return waveform_document_from_payload(
            payload, source_format=self.format_id, source_path=str(path)
        )

    def write(self, path: Path, document: WaveformDocument) -> None:
        payload = waveform_signal_to_payload(document)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            raise WaveformIOError(f"Unable to write native waveform JSON: {error}") from error


def build_default_waveform_io_registry() -> WaveformIORegistry:
    """Return a fresh registry containing the formats bundled with this release."""
    registry = WaveformIORegistry()
    native = NativeWaveformJSON()
    registry.register_reader(native)
    registry.register_writer(native)
    return registry


DEFAULT_WAVEFORM_IO_REGISTRY = build_default_waveform_io_registry()


def read_waveform_document(
    path: str | Path,
    *,
    format_id: str = NATIVE_WAVEFORM_READER_ID,
    registry: WaveformIORegistry | None = None,
) -> WaveformDocument:
    """Decode a waveform through an explicitly selected registered format.

    The convenience wrapper defaults to the native FWB interchange reader; the
    importer UI always passes the user's selected format id.
    """
    return (registry or DEFAULT_WAVEFORM_IO_REGISTRY).read(path, format_id=format_id)


def write_waveform_document(
    path: str | Path,
    document: WaveformDocument,
    *,
    format_id: str = NATIVE_WAVEFORM_READER_ID,
    registry: WaveformIORegistry | None = None,
) -> None:
    """Encode a waveform through the default (or supplied) registry."""
    (registry or DEFAULT_WAVEFORM_IO_REGISTRY).write(path, document, format_id=format_id)
