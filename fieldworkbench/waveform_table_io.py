"""Simple tabular waveform import/export helpers.

This module is intentionally Qt-free.  Ordinary CSV/TXT import is described in
terms a user can see in the file: rows, columns, time, electrical units, and an
optional linear value conversion.  Device encodings and binary layouts remain
in the explicit format-profile / DAC boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .waveform_io import WaveformDocument, WaveformIOError, WaveformSeries, time_scale_to_seconds


_DELIMITERS = {
    "comma": ",",
    "tab": "\t",
    "semicolon": ";",
    "whitespace": None,
}


@dataclass(frozen=True)
class TextTable:
    """Decoded rectangular text table before electrical interpretation."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    delimiter_mode: str
    has_header: bool
    source_path: str = ""

    @property
    def column_count(self) -> int:
        return len(self.headers)

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class TabularSignalSpec:
    """Map one table column into one electrical waveform resource."""

    column: int
    name: str
    quantity: str
    unit: str
    scale: float = 1.0
    offset: float = 0.0


def _numeric(text: str) -> bool:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _split_line(line: str, delimiter_mode: str) -> list[str]:
    if delimiter_mode == "whitespace":
        return line.strip().split()
    try:
        delimiter = _DELIMITERS[delimiter_mode]
    except KeyError as error:
        raise WaveformIOError(f"Unsupported text-table delimiter mode {delimiter_mode!r}.") from error
    if delimiter is None:
        return line.strip().split()
    return next(csv.reader([line], delimiter=delimiter))


def _detect_delimiter(lines: Sequence[str]) -> str:
    """Choose a common table delimiter without pretending to detect file type."""
    sample = list(lines[:20])
    if not sample:
        raise WaveformIOError("The selected text file does not contain any data rows.")
    best_mode = "whitespace"
    best_score = (-1, -1)
    for mode in ("comma", "tab", "semicolon", "whitespace"):
        widths = [len(_split_line(line, mode)) for line in sample]
        useful = [width for width in widths if width > 1]
        if not useful:
            score = (0, 0)
        else:
            common = max(set(useful), key=useful.count)
            score = (sum(width == common for width in widths), common)
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode


def _detect_header(rows: Sequence[Sequence[str]]) -> bool:
    if len(rows) < 2:
        return False
    first = rows[0]
    second = rows[1]
    # A human header normally contains at least one non-number where the next
    # record contains a number.  This avoids relying on csv.Sniffer.has_header,
    # whose heuristic is surprisingly unstable for short instrument files.
    return any((not _numeric(a)) and _numeric(b) for a, b in zip(first, second))


def parse_text_table(
    text: str,
    *,
    delimiter_mode: str = "auto",
    header_mode: str = "auto",
    source_path: str = "",
) -> TextTable:
    """Parse a rectangular TXT/CSV table while retaining raw cell strings."""
    lines = _nonempty_lines(str(text))
    if not lines:
        raise WaveformIOError("The selected text file does not contain any waveform data.")
    if delimiter_mode == "auto":
        delimiter_mode = _detect_delimiter(lines)
    if delimiter_mode not in _DELIMITERS:
        raise WaveformIOError(f"Unsupported text-table delimiter mode {delimiter_mode!r}.")

    rows = [_split_line(line, delimiter_mode) for line in lines]
    width = len(rows[0])
    if width < 1:
        raise WaveformIOError("The selected text file does not contain any columns.")
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise WaveformIOError(
                f"Text row {index} contains {len(row)} columns; expected {width}. "
                "Choose the correct delimiter or clean the source table."
            )

    if header_mode == "auto":
        has_header = _detect_header(rows)
    elif header_mode == "yes":
        has_header = True
    elif header_mode == "no":
        has_header = False
    else:
        raise WaveformIOError(f"Unsupported text-table header mode {header_mode!r}.")

    if has_header:
        raw_headers = rows.pop(0)
        headers = tuple(str(value).strip() or f"Column {index + 1}" for index, value in enumerate(raw_headers))
    else:
        headers = tuple(f"Column {index + 1}" for index in range(width))
    if not rows:
        raise WaveformIOError("The selected text file contains a header but no data rows.")
    return TextTable(
        headers=headers,
        rows=tuple(tuple(str(value).strip() for value in row) for row in rows),
        delimiter_mode=delimiter_mode,
        has_header=has_header,
        source_path=str(source_path or ""),
    )


def read_text_table(
    path: str | Path,
    *,
    delimiter_mode: str = "auto",
    header_mode: str = "auto",
    encoding: str = "utf-8-sig",
) -> TextTable:
    source = Path(path)
    try:
        text = source.read_text(encoding=encoding)
    except (OSError, UnicodeError, LookupError) as error:
        raise WaveformIOError(f"Unable to read text waveform {source.name!r}: {error}") from error
    return parse_text_table(
        text,
        delimiter_mode=delimiter_mode,
        header_mode=header_mode,
        source_path=str(source),
    )


def _column_values(table: TextTable, column: int, *, label: str) -> np.ndarray:
    if column < 0 or column >= table.column_count:
        raise WaveformIOError(f"{label} column is outside the imported table.")
    values: list[float] = []
    for data_index, row in enumerate(table.rows, start=1):
        row_number = data_index + int(table.has_header)
        token = row[column]
        try:
            value = float(token)
        except ValueError as error:
            raise WaveformIOError(
                f"{label} column {table.headers[column]!r} contains non-numeric value {token!r} at row {row_number}."
            ) from error
        if not math.isfinite(value):
            raise WaveformIOError(
                f"{label} column {table.headers[column]!r} contains a non-finite value at row {row_number}."
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def documents_from_text_table(
    table: TextTable,
    signal_specs: Iterable[TabularSignalSpec],
    *,
    time_column: int | None,
    time_unit: str = "s",
    sample_interval: float = 1.0,
    sample_interval_unit: str = "ms",
) -> tuple[WaveformDocument, ...]:
    """Translate selected table columns into independent catalogue waveforms."""
    specs = tuple(signal_specs)
    if not specs:
        raise WaveformIOError("Select at least one signal column to import.")
    if time_column is None:
        try:
            interval = float(sample_interval) * time_scale_to_seconds(sample_interval_unit)
        except (TypeError, ValueError) as error:
            raise WaveformIOError("Fixed sample interval must be numeric.") from error
        if not math.isfinite(interval) or interval <= 0.0:
            raise WaveformIOError("Fixed sample interval must be finite and greater than zero.")
        time_s = np.arange(table.row_count, dtype=float) * interval
        original_time_offset = 0.0
    else:
        raw_time = _column_values(table, int(time_column), label="Time")
        factor = time_scale_to_seconds(time_unit)
        time_s = raw_time * factor
        if len(time_s) < 2:
            raise WaveformIOError("Waveform import requires at least two data rows.")
        if np.any(np.diff(time_s) <= 0.0):
            raise WaveformIOError("Imported time values must be strictly increasing.")
        original_time_offset = float(time_s[0])
        time_s = time_s - original_time_offset

    if len(time_s) < 2:
        raise WaveformIOError("Waveform import requires at least two data rows.")

    documents: list[WaveformDocument] = []
    source_name = Path(table.source_path).name if table.source_path else "text table"
    for spec in specs:
        column = int(spec.column)
        if time_column is not None and column == int(time_column):
            raise WaveformIOError("The selected time column cannot also be imported as a signal.")
        quantity = str(spec.quantity or "").strip().lower()
        valid_units = {"current": {"A", "mA"}, "voltage": {"V", "mV"}}
        if quantity not in valid_units or spec.unit not in valid_units[quantity]:
            raise WaveformIOError("Each imported signal must be Current (A/mA) or Voltage (V/mV).")
        scale = float(spec.scale)
        offset = float(spec.offset)
        if not math.isfinite(scale) or not math.isfinite(offset):
            raise WaveformIOError("Text/CSV signal scale and offset must be finite.")
        raw = _column_values(table, column, label="Signal")
        values = raw * scale + offset
        name = str(spec.name or table.headers[column] or f"Signal {column + 1}").strip() or f"Signal {column + 1}"
        import_settings = {
            "format": "text_csv_table",
            "source_file": source_name,
            "delimiter": table.delimiter_mode,
            "header": bool(table.has_header),
            "signal_column": column + 1,
            "signal_header": table.headers[column],
            "quantity": quantity,
            "unit": spec.unit,
            "scale": scale,
            "offset": offset,
        }
        if time_column is None:
            import_settings.update({
                "time_mode": "fixed_interval",
                "sample_interval": float(sample_interval),
                "sample_interval_unit": sample_interval_unit,
            })
        else:
            import_settings.update({
                "time_mode": "column",
                "time_column": int(time_column) + 1,
                "time_header": table.headers[int(time_column)],
                "time_unit": time_unit,
                "original_time_offset_s": original_time_offset,
            })
        documents.append(
            WaveformDocument(
                [WaveformSeries(name, time_s, values, quantity=quantity, unit=spec.unit)],
                title=name,
                metadata={"import_settings": import_settings},
                source_format="text_csv_table",
                source_path=table.source_path,
            )
        )
    return tuple(documents)


def write_waveform_csv(path: str | Path, document: WaveformDocument) -> Path:
    """Write one saved waveform as a simple physical time/value CSV table."""
    signal = document.signal
    destination = Path(path)
    header_name = str(document.title or signal.name or "Waveform").strip() or "Waveform"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Time (s)", f"{header_name} ({signal.unit})"])
            for time_value, signal_value in zip(signal.time_s, signal.values):
                writer.writerow([f"{float(time_value):.17g}", f"{float(signal_value):.17g}"])
    except OSError as error:
        raise WaveformIOError(f"Unable to write waveform CSV {destination.name!r}: {error}") from error
    return destination
