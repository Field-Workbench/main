"""Waveform import/export boundary dialogs.

Normal waveform authoring is embedded directly in the Waveform Editor.  This
module now contains only dialogs that translate an external device/file encoding
at a catalogue import/export boundary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from ...waveform_io import WaveformSeries
from ..core.dac import write_dac
from ..core.native import quantize_to_unsigned_codes, uniform_sample_interval_s
from ..core.rigol import CsvSettings, RafSettings, RigolExportError, build_raf, write_csv
from .base_window import BaseDialog, BaseWindow
from .widgets import NumericSpec, StatusBar, action_button


_VALUE_MIN = -1.0e12
_VALUE_MAX = 1.0e12


def _value_spec(label: str, value: float = 0.0) -> NumericSpec:
    return NumericSpec(label, value, _VALUE_MIN, _VALUE_MAX, 0.001, 11, "float", True)


class DacExportDialog(BaseDialog):
    """Quantize one native waveform signal into the unsigned 8-bit DAC format."""

    def __init__(self, owner: BaseWindow, series: WaveformSeries) -> None:
        super().__init__(owner, title="Export DAC", modal=True)
        self.series = series
        values = np.asarray(series.values, dtype=float)
        low = float(np.min(values))
        high = float(np.max(values))
        if math.isclose(low, high):
            high = low + 1.0
        self.setMinimumWidth(590)
        root = QVBoxLayout(self)
        intro = QLabel(
            "Legacy DAC files are unsigned 8-bit. Choose which native engineering values map to code 0 and code 255. "
            "Values outside that range are clipped only in the exported copy.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        form_host = QGroupBox("Export mapping", self)
        form = QFormLayout(form_host)
        self.zero_value = self.numeric_field(form_host, _value_spec("", low))
        self.full_value = self.numeric_field(form_host, _value_spec("", high))
        form.addRow("Code 0 =", self.zero_value)
        form.addRow("Code 255 =", self.full_value)
        form.addRow("Signal", QLabel(f"{series.name} · {series.quantity.title()} {series.unit}", form_host))
        root.addWidget(form_host)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(action_button("Export…", self, self.export, role="primary"))
        actions.addWidget(action_button("Close", self, self.close_window))
        root.addLayout(actions)
        self.status = StatusBar(self)
        root.addWidget(self.status)

    def export(self) -> None:
        if not self.fields_valid():
            return
        destination, _ = QFileDialog.getSaveFileName(self, "Save DAC file", "", "DAC files (*.dac);;All files (*)")
        if not destination:
            return
        path = Path(destination)
        if path.suffix.lower() != ".dac":
            path = path.with_suffix(".dac")
        try:
            points = quantize_to_unsigned_codes(
                self.series.values,
                bit_depth=8,
                value_at_zero=self.zero_value.get_float(),
                value_at_full_scale=self.full_value.get_float(),
                clip=True,
            )
            write_dac(path, [int(value) for value in points])
        except Exception as error:  # noqa: BLE001
            self.show_error("DAC export failed", str(error))
            return
        self.status.set(f"Saved {path.name} as 8-bit DAC data.")


class NativeRigolExportDialog(BaseDialog):
    """Translate one native waveform signal at the Rigol export boundary."""

    def __init__(self, owner: BaseWindow, series: WaveformSeries) -> None:
        super().__init__(owner, title="Export to Rigol format", modal=True)
        self.series = series
        values = np.asarray(series.values, dtype=float)
        low = float(np.min(values))
        high = float(np.max(values))
        if math.isclose(low, high):
            high = low + 1.0
        self.setMinimumWidth(760)
        root = QVBoxLayout(self)
        intro = QLabel(
            "Rigol output uses the generator's ±5 V full-scale range. Map waveform values to that range here; "
            "the project waveform itself remains unchanged.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        row = QHBoxLayout()
        mapping = QGroupBox("Native → Rigol mapping", self)
        mapping_form = QFormLayout(mapping)
        self.minus5 = self.numeric_field(mapping, _value_spec("", low))
        self.plus5 = self.numeric_field(mapping, _value_spec("", high))
        self.rest = self.numeric_field(mapping, _value_spec("", 0.0 if low <= 0 <= high else low))
        mapping_form.addRow("-5 V =", self.minus5)
        mapping_form.addRow("+5 V =", self.plus5)
        mapping_form.addRow("Cycle-rest value", self.rest)
        row.addWidget(mapping)

        timing = QGroupBox("Output", self)
        timing_form = QFormLayout(timing)
        self.cycle_delay = self.numeric_field(timing, NumericSpec("", 0.0, 0.0, 500.0, 0.1, 8, "float", True))
        self.slew_rate = self.numeric_field(timing, NumericSpec("", 1.0, 0.001, 1000.0, 0.01, 8, "float", True))
        self.target_millions = self.numeric_field(timing, NumericSpec("", 8, 1, 16))
        self.raf_radio = QRadioButton("Binary / RAF", timing)
        self.csv_radio = QRadioButton("Text / CSV", timing)
        self.raf_radio.setChecked(True)
        group = QButtonGroup(timing)
        group.addButton(self.raf_radio)
        group.addButton(self.csv_radio)
        timing_form.addRow("Cycle delay (ms)", self.cycle_delay)
        timing_form.addRow("Slew rate (V/µs)", self.slew_rate)
        timing_form.addRow("CSV target (million points)", self.target_millions)
        timing_form.addRow(self.raf_radio)
        timing_form.addRow(self.csv_radio)
        row.addWidget(timing)
        root.addLayout(row)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(action_button("Export…", self, self.export, role="primary"))
        actions.addWidget(action_button("Close", self, self.close_window))
        root.addLayout(actions)
        self.status = StatusBar(self)
        root.addWidget(self.status)

    def _codes_and_timing(self) -> tuple[list[int], int, float]:
        interval_s = uniform_sample_interval_s(self.series)
        point_delay_ms = interval_s * 1000.0
        if not 0.00002 <= point_delay_ms <= 500.0:
            raise RigolExportError("Rigol export requires a sample interval between 0.00002 and 500 ms.")
        codes = quantize_to_unsigned_codes(
            self.series.values,
            bit_depth=16,
            value_at_zero=self.minus5.get_float(),
            value_at_full_scale=self.plus5.get_float(),
            clip=True,
        )
        rest_code = int(quantize_to_unsigned_codes(
            [self.rest.get_float()],
            bit_depth=16,
            value_at_zero=self.minus5.get_float(),
            value_at_full_scale=self.plus5.get_float(),
            clip=True,
        )[0])
        return [int(value) for value in codes], rest_code, point_delay_ms

    def export(self) -> None:
        if not self.fields_valid():
            return
        try:
            points, rest_code, point_delay_ms = self._codes_and_timing()
        except Exception as error:  # noqa: BLE001
            self.show_error("Rigol export failed", str(error))
            return
        if self.raf_radio.isChecked():
            destination, _ = QFileDialog.getSaveFileName(self, "Save Rigol RAF", "", "Rigol RAF (*.RAF);;All files (*)")
            if not destination:
                return
            path = Path(destination)
            if path.suffix.lower() != ".raf":
                path = path.with_suffix(".RAF")
            settings = RafSettings(
                point_delay_ms=point_delay_ms,
                cycle_delay_ms=self.cycle_delay.get_float(),
                rest_value=rest_code,
                bit_depth=16,
                slew_rate_v_per_us=self.slew_rate.get_float(),
            )
            try:
                build = build_raf(points, file_name=path.name, settings=settings)
                path.write_bytes(build.file_bytes)
            except (OSError, RigolExportError, OverflowError) as error:
                self.show_error("RAF export failed", str(error))
                return
            self.status.set(f"Saved {path.name} · {build.total_points:,} output samples.")
            return

        destination, _ = QFileDialog.getSaveFileName(self, "Save Rigol CSV", "", "Rigol CSV (*.csv);;All files (*)")
        if not destination:
            return
        path = Path(destination)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        settings = CsvSettings(
            point_delay_ms=point_delay_ms,
            cycle_delay_ms=self.cycle_delay.get_float(),
            rest_value=rest_code,
            bit_depth=16,
            slew_rate_v_per_us=self.slew_rate.get_float(),
            target_points=self.target_millions.get_int() * 1_000_000,
        )
        try:
            plan = write_csv(path, points, settings)
        except (OSError, RigolExportError) as error:
            self.show_error("CSV export failed", str(error))
            return
        self.status.set(f"Saved {path.name} · {plan.total_points:,} output samples.")
