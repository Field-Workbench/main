"""Analysis dialogs used by the first Field Workbench prototype."""

from __future__ import annotations

import copy
import csv
import html
import math
import pickle
import shutil
import signal
import tempfile
import threading
import time
from datetime import date
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .geomagnetism import (
    WMM2025_END_DATE,
    WMM2025_START_DATE,
    latlon_to_utm,
    utm_to_latlon,
    wmm2025_field,
)
from PySide6.QtCore import (
    QDate,
    QEvent,
    QUrl,
    QObject,
    QPointF,
    QRectF,
    QSize,
    QStandardPaths,
    QSettings,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDateEdit,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

try:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - optional Qt component in source-only installs
    QWebEnginePage = None
    QWebEngineProfile = None
    QWebEngineView = None

from .optimizer import (
    MAX_OPTIMIZER_BENCHMARK_WORKERS,
    MIN_OPTIMIZER_WORKERS,
    OptimizationCancelled,
    available_optimizer_workers,
    normalize_optimizer_worker_count,
    reported_logical_cpu_count,
)
from .camera_timeline import (
    default_2d_camera,
    normalize_camera_timeline,
    scripted_timeline_duration_s,
    plotly_view_to_video_camera,
    video_camera_to_plotly_view,
)
from .plot_view import PlotView
from .theme import current_theme, plot_theme_colors
from .process_ipc import field_render_process_context
from .render_process import run_field_render_process
from .waveform_gl_view import WaveformGLView
from .probe_models import (
    ProbeDefinitionError,
    builtin_probe_definitions,
    load_probe_definition,
    validate_probe_definition,
)
from .portable_scene import (
    TRANSLATOR_COMSOL,
    TRANSLATOR_FEMM,
    TRANSLATOR_GETDP,
    TRANSLATOR_OPENSCAD,
    TRANSLATOR_PORTABLE,
    _femm_alignment,
    _femm_direction_label,
    _femm_reference_half_basis,
    _femm_reference_slice_basis,
    _femm_reference_slice_offset_from_axis,
    build_portable_scene,
    export_catalog,
    femm_selection_status,
    getdp_selection_status,
    translator_description,
)
from .getdp_export import GETDP_MESH_PRESETS, getdp_export_plan
from .studio_adapter import (
    COIL_MODELING_LABELS,
    FIELD_HEAT_COLOURSCALE,
    FIELD_SIGNED_COLOURSCALE,
    _field_colour_at_fraction,
    FieldCalculationCancelled,
    StudioAdapter,
    StudioOperationError,
    field_component_is_magnitude,
)
from .upenn_gbm import (
    UPENN_GBM_COLLECTION_URL,
    UPENN_GBM_REGIONS,
    UpennGbmCase,
    UpennGbmCatalogue,
    UpennGbmImportCancelled,
    import_upenn_gbm_case,
    load_upenn_gbm_catalogue,
    refresh_upenn_gbm_catalogue,
)
from .window_utils import (
    CompactDoubleSpinBox,
    PlaybackSpeedSpinBox,
    capture_window_default_settings,
    centre_window_on_parent,
    enable_standard_window_controls,
    reset_window_to_default_settings,
)
from .waveform import drive_coil_catalog, validate_animation_field_basis
from .video_export import DEFAULT_CAMERA_FOV_DEG, default_video_camera, validate_video_camera
from .waveform_dialog import (
    WaveformPlaybackDialog,
    remembered_waveform_timing,
)
from .waveform_video_dialog import WaveformVideoExportDialog


DEFAULT_BACKGROUND_LATITUDE_DEG = 46.4939
DEFAULT_BACKGROUND_LONGITUDE_DEG = -80.9954
SELECTOR_PREVIEW_DEBOUNCE_MS = 250


class LocationMapPickerDialog(QDialog):
    """Small click-to-select geographic location picker using OpenStreetMap tiles."""

    def __init__(self, latitude_deg: float, longitude_deg: float, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Choose location on map")
        self.resize(900, 650)
        self._latitude = float(latitude_deg)
        self._longitude = float(longitude_deg)
        self._profile = None

        root = QVBoxLayout(self)
        hint = QLabel(
            "Click the map to choose a WGS84 latitude/longitude. Map tiles require an "
            "internet connection; the coordinate fields remain usable without one."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        root.addWidget(hint)

        coordinate_row = QHBoxLayout()
        self.latitude = CompactDoubleSpinBox()
        self.latitude.setRange(-90.0, 90.0)
        self.latitude.setDecimals(7)
        self.latitude.setValue(self._latitude)
        self.longitude = CompactDoubleSpinBox()
        self.longitude.setRange(-180.0, 180.0)
        self.longitude.setDecimals(7)
        self.longitude.setValue(self._longitude)
        coordinate_row.addWidget(QLabel("Latitude"))
        coordinate_row.addWidget(self.latitude)
        coordinate_row.addWidget(QLabel("Longitude"))
        coordinate_row.addWidget(self.longitude)
        update_button = QPushButton("Update marker")
        update_button.clicked.connect(self._update_marker_from_fields)
        coordinate_row.addWidget(update_button)
        root.addLayout(coordinate_row)

        if QWebEngineView is None:
            unavailable = QLabel(
                "Qt WebEngine is not available in this Python environment. Enter coordinates above."
            )
            unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
            unavailable.setWordWrap(True)
            root.addWidget(unavailable, 1)
            self.map_view = None
        else:
            self.map_view = QWebEngineView(self)
            if QWebEngineProfile is not None and QWebEnginePage is not None:
                self._profile = QWebEngineProfile(self)
                self._profile.setHttpUserAgent("FieldWorkbench/0.13 QtWebEngine location-picker")
                self.map_view.setPage(QWebEnginePage(self._profile, self.map_view))
            self.map_view.titleChanged.connect(self._map_title_changed)
            self.map_view.setHtml(
                self._map_html(self._latitude, self._longitude),
                QUrl("https://www.openstreetmap.org/"),
            )
            root.addWidget(self.map_view, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _map_html(latitude_deg: float, longitude_deg: float) -> str:
        latitude = max(-85.0, min(85.0, float(latitude_deg)))
        longitude = max(-180.0, min(180.0, float(longitude_deg)))
        return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
<style>html,body,#map{{height:100%;margin:0}} body{{background:#e5e7eb}}</style>
</head><body><div id='map'></div>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<script>
const map = L.map('map').setView([{latitude:.8f},{longitude:.8f}], 7);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}}).addTo(map);
let marker = L.marker([{latitude:.8f},{longitude:.8f}]).addTo(map);
function choose(lat,lng) {{
  marker.setLatLng([lat,lng]);
  document.title = 'FWB_PICK:' + lat.toFixed(8) + ',' + lng.toFixed(8);
}}
map.on('click', e => choose(e.latlng.lat, e.latlng.lng));
window.fwbSetPoint = function(lat,lng) {{ marker.setLatLng([lat,lng]); map.panTo([lat,lng]); }};
</script></body></html>"""

    def _map_title_changed(self, title: str) -> None:
        if not title.startswith("FWB_PICK:"):
            return
        try:
            latitude, longitude = title.split(":", 1)[1].split(",", 1)
            self._latitude = float(latitude)
            self._longitude = float(longitude)
        except (TypeError, ValueError):
            return
        self.latitude.setValue(self._latitude)
        self.longitude.setValue(self._longitude)

    def _update_marker_from_fields(self) -> None:
        self._latitude = self.latitude.value()
        self._longitude = self.longitude.value()
        if self.map_view is not None:
            self.map_view.page().runJavaScript(
                f"if (window.fwbSetPoint) fwbSetPoint({self._latitude:.10f},{self._longitude:.10f});"
            )

    def location(self) -> tuple[float, float]:
        return float(self.latitude.value()), float(self.longitude.value())


class BackgroundFieldDialog(QDialog):
    """Create or recalculate a uniform manual/WMM2025 scene background field."""

    def __init__(self, parent=None, *, initial: dict[str, Any] | None = None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Background field")
        self.resize(660, 650)
        self._definition: dict[str, Any] | None = None
        initial = copy.deepcopy(initial or {})
        source = initial.get("source", {}) if isinstance(initial.get("source"), dict) else {}

        root = QVBoxLayout(self)
        intro = QLabel(
            "Add one uniform static vector to every B/H field evaluation in this scene. "
            "Manual values are entered directly in Workbench X/Y/Z. Earth field uses WMM2025 "
            "and stores the resolved vector so saved scenes stay reproducible."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        manual = QWidget()
        manual_form = QFormLayout(manual)
        vector = initial.get("vector_uT", [0.0, 0.0, 0.0])
        self.manual_spins: list[QDoubleSpinBox] = []
        for index, axis in enumerate(("Bx", "By", "Bz")):
            spin = CompactDoubleSpinBox()
            spin.setRange(-1.0e9, 1.0e9)
            spin.setDecimals(6)
            spin.setSingleStep(1.0)
            spin.setSuffix(" µT")
            try:
                spin.setValue(float(vector[index]))
            except (IndexError, TypeError, ValueError):
                pass
            manual_form.addRow(axis, spin)
            self.manual_spins.append(spin)
        manual_note = QLabel(
            "These components use the Workbench scene axes directly and are independent of position."
        )
        manual_note.setWordWrap(True)
        manual_note.setObjectName("hint")
        manual_form.addRow(manual_note)
        self.tabs.addTab(manual, "Manual vector")

        earth = QWidget()
        earth_layout = QVBoxLayout(earth)
        coordinate_group = QGroupBox("Location (WGS84)")
        coordinate_layout = QVBoxLayout(coordinate_group)
        coordinate_mode_row = QHBoxLayout()
        coordinate_mode_row.addWidget(QLabel("Input"))
        self.coordinate_mode = QComboBox()
        self.coordinate_mode.addItem("Latitude / longitude", "latlon")
        self.coordinate_mode.addItem("UTM", "utm")
        coordinate_mode_row.addWidget(self.coordinate_mode, 1)
        self.map_button = QPushButton("Choose on map…")
        self.map_button.clicked.connect(self._choose_map_location)
        coordinate_mode_row.addWidget(self.map_button)
        coordinate_layout.addLayout(coordinate_mode_row)

        self.coordinate_stack = QStackedWidget()
        latlon_widget = QWidget()
        latlon_form = QFormLayout(latlon_widget)
        self.latitude = CompactDoubleSpinBox()
        self.latitude.setRange(-90.0, 90.0)
        self.latitude.setDecimals(7)
        self.latitude.setValue(float(source.get("latitude_deg", DEFAULT_BACKGROUND_LATITUDE_DEG)))
        self.longitude = CompactDoubleSpinBox()
        self.longitude.setRange(-180.0, 180.0)
        self.longitude.setDecimals(7)
        self.longitude.setValue(float(source.get("longitude_deg", DEFAULT_BACKGROUND_LONGITUDE_DEG)))
        latlon_form.addRow("Latitude", self.latitude)
        latlon_form.addRow("Longitude", self.longitude)
        self.coordinate_stack.addWidget(latlon_widget)

        utm_widget = QWidget()
        utm_form = QFormLayout(utm_widget)
        self.utm_zone = QSpinBox()
        self.utm_zone.setRange(1, 60)
        self.utm_hemisphere = QComboBox()
        self.utm_hemisphere.addItems(["N", "S"])
        self.utm_easting = CompactDoubleSpinBox()
        self.utm_easting.setRange(0.0, 1_000_000.0)
        self.utm_easting.setDecimals(3)
        self.utm_easting.setSuffix(" m")
        self.utm_northing = CompactDoubleSpinBox()
        self.utm_northing.setRange(0.0, 10_000_000.0)
        self.utm_northing.setDecimals(3)
        self.utm_northing.setSuffix(" m")
        utm_form.addRow("Zone", self.utm_zone)
        utm_form.addRow("Hemisphere", self.utm_hemisphere)
        utm_form.addRow("Easting", self.utm_easting)
        utm_form.addRow("Northing", self.utm_northing)
        self.coordinate_stack.addWidget(utm_widget)
        coordinate_layout.addWidget(self.coordinate_stack)
        earth_layout.addWidget(coordinate_group)

        calculation_group = QGroupBox("WMM2025 calculation")
        calculation_form = QFormLayout(calculation_group)
        self.altitude = CompactDoubleSpinBox()
        self.altitude.setRange(-1000.0, 850000.0)
        self.altitude.setDecimals(1)
        self.altitude.setSuffix(" m")
        self.altitude.setValue(float(source.get("altitude_m", 0.0)))
        self.when = QDateEdit()
        self.when.setCalendarPopup(True)
        self.when.setDisplayFormat("yyyy-MM-dd")
        self.when.setMinimumDate(QDate(WMM2025_START_DATE.year, 1, 1))
        self.when.setMaximumDate(QDate(WMM2025_END_DATE.year, 12, 31))
        date_text = str(source.get("date", ""))
        initial_date = QDate.fromString(date_text, "yyyy-MM-dd") if date_text else QDate.currentDate()
        if not initial_date.isValid():
            initial_date = QDate.currentDate()
        if initial_date < self.when.minimumDate():
            initial_date = self.when.minimumDate()
        elif initial_date > self.when.maximumDate():
            initial_date = self.when.maximumDate()
        self.when.setDate(initial_date)
        calculation_form.addRow("Altitude", self.altitude)
        calculation_form.addRow("Date", self.when)
        earth_layout.addWidget(calculation_group)

        orientation_group = QGroupBox("Scene orientation")
        orientation_form = QFormLayout(orientation_group)
        self.heading = CompactDoubleSpinBox()
        self.heading.setRange(0.0, 360.0)
        self.heading.setDecimals(2)
        self.heading.setSuffix("°")
        self.heading.setValue(float(source.get("heading_deg", 0.0)) % 360.0)
        self.pitch = CompactDoubleSpinBox()
        self.pitch.setRange(-180.0, 180.0)
        self.pitch.setDecimals(2)
        self.pitch.setSuffix("°")
        self.pitch.setValue(float(source.get("pitch_deg", 0.0)))
        self.roll = CompactDoubleSpinBox()
        self.roll.setRange(-180.0, 180.0)
        self.roll.setDecimals(2)
        self.roll.setSuffix("°")
        self.roll.setValue(float(source.get("roll_deg", 0.0)))
        orientation_form.addRow("Heading (clockwise from true north)", self.heading)
        orientation_form.addRow("Pitch about scene +X", self.pitch)
        orientation_form.addRow("Roll about scene +Y", self.roll)
        orientation_note = QLabel("At 0° / 0° / 0°: +X = east, +Y = true north, +Z = up.")
        orientation_note.setWordWrap(True)
        orientation_note.setObjectName("hint")
        orientation_form.addRow(orientation_note)
        earth_layout.addWidget(orientation_group)

        result_group = QGroupBox("Calculated field")
        result_form = QFormLayout(result_group)
        self.scene_result = QLabel("Not calculated")
        self.ned_result = QLabel("—")
        self.di_result = QLabel("—")
        for label in (self.scene_result, self.ned_result, self.di_result):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            label.setWordWrap(True)
        result_form.addRow("Workbench B", self.scene_result)
        result_form.addRow("Geographic N/E/Down", self.ned_result)
        result_form.addRow("D / I / |B|", self.di_result)
        calculate = QPushButton("Calculate / refresh WMM2025")
        calculate.clicked.connect(self._calculate_wmm)
        result_form.addRow(calculate)
        earth_layout.addWidget(result_group)
        earth_layout.addStretch(1)
        self.tabs.addTab(earth, "Earth field — WMM2025")

        self.coordinate_mode.currentIndexChanged.connect(self._coordinate_mode_changed)
        self.coordinate_mode.setCurrentIndex(
            max(0, self.coordinate_mode.findData(str(source.get("coordinate_input", "latlon"))))
        )
        self._sync_utm_from_latlon(source.get("utm"))

        if str(initial.get("mode", "manual")) == "wmm2025":
            self.tabs.setCurrentIndex(1)
            self._show_existing_wmm(source, vector)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_definition)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _coordinate_mode_changed(self, index: int) -> None:
        self.coordinate_stack.setCurrentIndex(index)
        if self.coordinate_mode.currentData() == "utm":
            self._sync_utm_from_latlon()
        else:
            self._sync_latlon_from_utm()

    def _sync_utm_from_latlon(self, stored_utm: Any = None) -> None:
        try:
            utm = stored_utm if isinstance(stored_utm, dict) else latlon_to_utm(
                self.latitude.value(), self.longitude.value()
            )
            self.utm_zone.setValue(int(utm["zone"]))
            self.utm_hemisphere.setCurrentText(str(utm["hemisphere"]).upper())
            self.utm_easting.setValue(float(utm["easting_m"]))
            self.utm_northing.setValue(float(utm["northing_m"]))
        except (KeyError, TypeError, ValueError):
            pass

    def _sync_latlon_from_utm(self) -> None:
        try:
            latitude, longitude = utm_to_latlon(
                self.utm_zone.value(),
                self.utm_easting.value(),
                self.utm_northing.value(),
                self.utm_hemisphere.currentText(),
            )
            self.latitude.setValue(latitude)
            self.longitude.setValue(longitude)
        except ValueError:
            pass

    def _resolved_latlon(self) -> tuple[float, float]:
        if self.coordinate_mode.currentData() == "utm":
            return utm_to_latlon(
                self.utm_zone.value(),
                self.utm_easting.value(),
                self.utm_northing.value(),
                self.utm_hemisphere.currentText(),
            )
        return self.latitude.value(), self.longitude.value()

    def _choose_map_location(self) -> None:
        try:
            latitude, longitude = self._resolved_latlon()
        except ValueError:
            latitude, longitude = self.latitude.value(), self.longitude.value()
        dialog = LocationMapPickerDialog(latitude, longitude, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        latitude, longitude = dialog.location()
        self.latitude.setValue(latitude)
        self.longitude.setValue(longitude)
        latlon_index = max(0, self.coordinate_mode.findData("latlon"))
        previous_block = self.coordinate_mode.blockSignals(True)
        self.coordinate_mode.setCurrentIndex(latlon_index)
        self.coordinate_mode.blockSignals(previous_block)
        self.coordinate_stack.setCurrentIndex(latlon_index)
        self._sync_utm_from_latlon()

    def _date_value(self) -> date:
        value = self.when.date()
        return date(value.year(), value.month(), value.day())

    def _calculate_wmm(self, *, warn: bool = True) -> dict[str, Any] | None:
        try:
            latitude, longitude = self._resolved_latlon()
            result = wmm2025_field(
                latitude_deg=latitude,
                longitude_deg=longitude,
                altitude_m=self.altitude.value(),
                when=self._date_value(),
                heading_deg=self.heading.value(),
                pitch_deg=self.pitch.value(),
                roll_deg=self.roll.value(),
            )
            utm = latlon_to_utm(latitude, longitude) if -80.0 <= latitude <= 84.0 else None
        except (RuntimeError, ValueError) as error:
            if warn:
                QMessageBox.warning(self, "WMM2025", str(error))
            return None

        vector = result["scene_vector_uT"]
        self.scene_result.setText(
            f"Bx {vector[0]:.4f} µT   •   By {vector[1]:.4f} µT   •   Bz {vector[2]:.4f} µT"
        )
        self.ned_result.setText(
            f"N {result['north_uT']:.4f} µT   •   E {result['east_uT']:.4f} µT   •   Down {result['down_uT']:.4f} µT"
        )
        self.di_result.setText(
            f"D {result['declination_deg']:.3f}°   •   I {result['inclination_deg']:.3f}°   •   |B| {result['total_uT']:.4f} µT"
        )
        result["kind"] = "wmm2025"
        result["coordinate_input"] = str(self.coordinate_mode.currentData())
        if utm is not None:
            result["utm"] = utm
        return result

    def _show_existing_wmm(self, source: dict[str, Any], vector: Any) -> None:
        try:
            self.scene_result.setText(
                f"Bx {float(vector[0]):.4f} µT   •   By {float(vector[1]):.4f} µT   •   Bz {float(vector[2]):.4f} µT"
            )
            self.ned_result.setText(
                f"N {float(source['north_uT']):.4f} µT   •   E {float(source['east_uT']):.4f} µT   •   Down {float(source['down_uT']):.4f} µT"
            )
            self.di_result.setText(
                f"D {float(source['declination_deg']):.3f}°   •   I {float(source['inclination_deg']):.3f}°   •   |B| {float(source['total_uT']):.4f} µT"
            )
        except (KeyError, IndexError, TypeError, ValueError):
            pass

    def _accept_definition(self) -> None:
        if self.tabs.currentIndex() == 0:
            self._definition = {
                "mode": "manual",
                "vector_uT": [spin.value() for spin in self.manual_spins],
                "source": {"kind": "manual"},
            }
        else:
            source = self._calculate_wmm()
            if source is None:
                return
            self._definition = {
                "mode": "wmm2025",
                "vector_uT": list(source["scene_vector_uT"]),
                "source": source,
            }
        self.accept()

    def field_definition(self) -> dict[str, Any]:
        if self._definition is None:
            raise RuntimeError("Background field dialog was not accepted.")
        return copy.deepcopy(self._definition)


class CustomProbeDialog(QDialog):
    """Create one scene-portable triaxial or multi-channel probe definition."""

    def __init__(self, parent=None, *, initial: dict[str, Any] | None = None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Create custom magnetometer probe")
        self.resize(900, 720)
        self._definition: dict[str, Any] | None = None
        initial = copy.deepcopy(initial or {})
        housing = initial.get("housing", {}) if isinstance(initial.get("housing"), dict) else {}

        root = QVBoxLayout(self)
        intro = QLabel(
            "Define the housing envelope and each physical measurement channel. Positions and "
            "sensitive directions use the probe's local right-handed XYZ coordinates. The "
            "validated definition will be embedded in the scene."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        content_layout = QVBoxLayout(contents)
        scroll.setWidget(contents)
        root.addWidget(scroll, 1)

        identity = QGroupBox("Identity and provenance")
        identity_form = QFormLayout(identity)
        self.manufacturer = QLineEdit(str(initial.get("manufacturer", "Custom")))
        self.model = QLineEdit(str(initial.get("model", "Three-axis probe")))
        self.description = QPlainTextEdit(str(initial.get("description", "")))
        self.description.setMaximumHeight(72)
        self.origin_description = QLineEdit(
            str(initial.get("origin_description", "Probe local origin."))
        )
        self.coordinate_system = QLineEdit(
            str(initial.get("coordinate_system", "Probe-local right-handed XYZ."))
        )
        sources = initial.get("sources", []) if isinstance(initial.get("sources"), list) else []
        source = sources[0] if sources and isinstance(sources[0], dict) else {}
        self.source_title = QLineEdit(str(source.get("title", "")))
        self.source_url = QLineEdit(str(source.get("url", "")))
        self.source_url.setPlaceholderText("https://…")
        identity_form.addRow("Manufacturer", self.manufacturer)
        identity_form.addRow("Model", self.model)
        identity_form.addRow("Description", self.description)
        identity_form.addRow("Local origin", self.origin_description)
        identity_form.addRow("Axis convention", self.coordinate_system)
        identity_form.addRow("Source title", self.source_title)
        identity_form.addRow("Source link", self.source_url)
        content_layout.addWidget(identity)

        housing_box = QGroupBox("Housing envelope")
        housing_form = QFormLayout(housing_box)

        def vector_spins(values: Any, *, positive: bool) -> list[QDoubleSpinBox]:
            try:
                vector = [float(value) for value in values]
            except (TypeError, ValueError):
                vector = [10.0, 10.0, 50.0] if positive else [0.0, 0.0, 0.0]
            if len(vector) != 3:
                vector = [10.0, 10.0, 50.0] if positive else [0.0, 0.0, 0.0]
            result: list[QDoubleSpinBox] = []
            for value in vector:
                spin = CompactDoubleSpinBox()
                spin.setRange(0.001 if positive else -1.0e6, 1.0e6)
                spin.setDecimals(4)
                spin.setSuffix(" mm")
                spin.setValue(value)
                result.append(spin)
            return result

        self.housing_dimensions = vector_spins(
            housing.get("dimensions_mm", [10.0, 10.0, 50.0]), positive=True
        )
        self.housing_center = vector_spins(
            housing.get("center_mm", [0.0, 0.0, 0.0]), positive=False
        )
        self.sticker_face = QComboBox()
        self.sticker_face.setObjectName("customProbeStickerFace")
        self.sticker_face.addItem("None", "")
        for face in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            self.sticker_face.addItem(face, face)
        initial_sticker_face = str(housing.get("sticker_face", "")).strip().upper()
        sticker_index = self.sticker_face.findData(initial_sticker_face)
        self.sticker_face.setCurrentIndex(max(0, sticker_index))
        self.sticker_label = QLineEdit(str(housing.get("sticker_label", "STICKER SIDE")))
        self.sticker_label.setPlaceholderText("STICKER SIDE")
        self.cable_face = QComboBox()
        self.cable_face.setObjectName("customProbeCableFace")
        self.cable_face.addItem("None", "")
        for face in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            self.cable_face.addItem(face, face)
        initial_cable_face = str(housing.get("cable_face", "")).strip().upper()
        cable_index = self.cable_face.findData(initial_cable_face)
        self.cable_face.setCurrentIndex(max(0, cable_index))
        dimensions_row = QHBoxLayout()
        center_row = QHBoxLayout()
        for axis, spin in zip("XYZ", self.housing_dimensions, strict=True):
            dimensions_row.addWidget(QLabel(axis))
            dimensions_row.addWidget(spin)
        for axis, spin in zip("XYZ", self.housing_center, strict=True):
            center_row.addWidget(QLabel(axis))
            center_row.addWidget(spin)
        housing_form.addRow("Dimensions", dimensions_row)
        housing_form.addRow("Centre from origin", center_row)
        housing_form.addRow("Sticker/label face", self.sticker_face)
        housing_form.addRow("Sticker marker text", self.sticker_label)
        housing_form.addRow("Cable connection face", self.cable_face)
        self.geometry_note = QLineEdit(str(housing.get("geometry_note", "")))
        housing_form.addRow("Geometry note", self.geometry_note)
        content_layout.addWidget(housing_box)

        channels_box = QGroupBox("Measurement channels")
        channels_layout = QVBoxLayout(channels_box)
        channel_note = QLabel(
            "Each reported value is the full world field at that point projected onto the "
            "channel's sensitive direction. Directions are normalized when the definition is saved."
        )
        channel_note.setWordWrap(True)
        channel_note.setObjectName("hint")
        channels_layout.addWidget(channel_note)
        self.channel_table = QTableWidget(0, 7)
        self.channel_table.setObjectName("customProbeChannelTable")
        self.channel_table.setHorizontalHeaderLabels(
            ["Channel", "Point X (mm)", "Point Y (mm)", "Point Z (mm)", "Axis X", "Axis Y", "Axis Z"]
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.channel_table.setMinimumHeight(210)
        channels_layout.addWidget(self.channel_table)
        row_buttons = QHBoxLayout()
        add_channel = QPushButton("Add channel")
        add_channel.clicked.connect(lambda: self._add_channel("Channel", [0, 0, 0], [0, 0, 1]))
        remove_channel = QPushButton("Remove selected")
        remove_channel.clicked.connect(self._remove_selected_channel)
        row_buttons.addWidget(add_channel)
        row_buttons.addWidget(remove_channel)
        row_buttons.addStretch(1)
        channels_layout.addLayout(row_buttons)
        content_layout.addWidget(channels_box)
        content_layout.addStretch(1)

        raw_channels = initial.get("channels", []) if isinstance(initial.get("channels"), list) else []
        if raw_channels:
            for channel in raw_channels:
                if isinstance(channel, dict):
                    self._add_channel(
                        str(channel.get("name", "Channel")),
                        channel.get("position_mm", [0, 0, 0]),
                        channel.get("sensitive_axis", [0, 0, 1]),
                    )
        else:
            self._add_channel("X", [0, 0, 0], [1, 0, 0])
            self._add_channel("Y", [0, 0, 0], [0, 1, 0])
            self._add_channel("Z", [0, 0, 0], [0, 0, 1])

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_definition)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _add_channel(self, name: str, position: Any, axis: Any) -> None:
        row = self.channel_table.rowCount()
        self.channel_table.insertRow(row)
        values = [name, *list(position)[:3], *list(axis)[:3]]
        while len(values) < 7:
            values.append(0.0)
        for column, value in enumerate(values[:7]):
            self.channel_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _remove_selected_channel(self) -> None:
        rows = sorted({index.row() for index in self.channel_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.channel_table.removeRow(row)

    def _channel_definitions(self) -> list[dict[str, Any]]:
        channels: list[dict[str, Any]] = []
        for row in range(self.channel_table.rowCount()):
            values = [
                self.channel_table.item(row, column).text().strip()
                if self.channel_table.item(row, column) is not None
                else ""
                for column in range(7)
            ]
            channels.append(
                {
                    "name": values[0],
                    "output": f"B{values[0]}",
                    "position_mm": [float(value) for value in values[1:4]],
                    "sensitive_axis": [float(value) for value in values[4:7]],
                }
            )
        return channels

    def _accept_definition(self) -> None:
        sources = []
        if self.source_title.text().strip() or self.source_url.text().strip():
            sources.append(
                {
                    "title": self.source_title.text().strip(),
                    "url": self.source_url.text().strip(),
                    "kind": "user-source",
                }
            )
        raw = {
            "manufacturer": self.manufacturer.text().strip(),
            "model": self.model.text().strip(),
            "display_name": " ".join(
                part for part in (self.manufacturer.text().strip(), self.model.text().strip()) if part
            ),
            "description": self.description.toPlainText().strip(),
            "origin_description": self.origin_description.text().strip(),
            "coordinate_system": self.coordinate_system.text().strip(),
            "verification_status": "custom",
            "housing": {
                "shape": "box",
                "dimensions_mm": [spin.value() for spin in self.housing_dimensions],
                "center_mm": [spin.value() for spin in self.housing_center],
                "sticker_face": str(self.sticker_face.currentData() or ""),
                "sticker_label": self.sticker_label.text().strip() or "STICKER SIDE",
                "cable_face": str(self.cable_face.currentData() or ""),
                "colour": "#475569",
                "opacity": 0.58,
                "geometry_note": self.geometry_note.text().strip(),
            },
            "channels": self._channel_definitions(),
            "sources": sources,
        }
        try:
            self._definition = validate_probe_definition(raw)
        except (ProbeDefinitionError, ValueError) as error:
            QMessageBox.warning(self, "Custom probe", str(error))
            return
        self.accept()

    def probe_definition(self) -> dict[str, Any]:
        if self._definition is None:
            raise RuntimeError("Custom probe dialog was not accepted.")
        return copy.deepcopy(self._definition)


class ProbeLibraryDialog(QDialog):
    """First-draft browser for bundled, imported, and newly authored probes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Magnetometer probe library")
        self.resize(900, 590)
        self._definitions = builtin_probe_definitions()
        self._definition: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        intro = QLabel(
            "Choose a verified bundled model, import a probe-definition JSON file, or create a "
            "scene-specific custom model. The complete definition is copied into the scene."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        body = QSplitter(Qt.Orientation.Horizontal)
        self.probe_list = QTreeWidget()
        self.probe_list.setObjectName("magnetometerProbeLibraryList")
        self.probe_list.setHeaderLabels(["Manufacturer", "Model", "Status"])
        self.probe_list.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.probe_list.currentItemChanged.connect(self._selection_changed)
        self.probe_list.itemDoubleClicked.connect(lambda *_args: self._accept_selected())
        body.addWidget(self.probe_list)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        body.addWidget(self.details)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        root.addWidget(body, 1)

        for index, definition in enumerate(self._definitions):
            item = QTreeWidgetItem(
                [
                    definition["manufacturer"],
                    definition["model"],
                    definition["verification_status"].replace("-", " ").title(),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            self.probe_list.addTopLevelItem(item)
        action_row = QHBoxLayout()
        custom = QPushButton("Create custom…")
        custom.setObjectName("createCustomProbeButton")
        custom.clicked.connect(self._create_custom)
        import_button = QPushButton("Import definition…")
        import_button.clicked.connect(self._import_definition)
        action_row.addWidget(custom)
        action_row.addWidget(import_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.add_button = buttons.addButton("Add probe to scene", QDialogButtonBox.ButtonRole.AcceptRole)
        self.add_button.setObjectName("addMagnetometerProbeButton")
        self.add_button.clicked.connect(self._accept_selected)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if self.probe_list.topLevelItemCount():
            self.probe_list.setCurrentItem(self.probe_list.topLevelItem(0))
        else:
            self.add_button.setEnabled(False)

    def _selected_definition(self) -> dict[str, Any] | None:
        item = self.probe_list.currentItem()
        if item is None:
            return None
        try:
            index = int(item.data(0, Qt.ItemDataRole.UserRole))
            return self._definitions[index]
        except (TypeError, ValueError, IndexError):
            return None

    def _selection_changed(self, *_args) -> None:
        definition = self._selected_definition()
        self.add_button.setEnabled(definition is not None)
        if definition is None:
            self.details.clear()
            return
        housing = definition["housing"]
        channel_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(channel["name"])),
                html.escape(", ".join(f"{value:g}" for value in channel["position_mm"])),
                html.escape(", ".join(f"{value:g}" for value in channel["sensitive_axis"])),
                (
                    f"{float(channel['reference_from_edge_mm']):g} mm"
                    if channel.get("reference_from_edge_mm") is not None
                    else "—"
                ),
            )
            for channel in definition["channels"]
        )
        source_rows = "".join(
            f"<li><a href='{html.escape(source['url'])}'>{html.escape(source['title'])}</a></li>"
            for source in definition.get("sources", [])
            if source.get("url")
        )
        self.details.setHtml(
            f"<h2>{html.escape(definition['display_name'])}</h2>"
            f"<p>{html.escape(definition.get('description', ''))}</p>"
            f"<p><b>Status:</b> {html.escape(definition['verification_status'])}<br>"
            f"<b>Housing envelope:</b> {' × '.join(f'{value:g}' for value in housing['dimensions_mm'])} mm<br>"
            f"<b>Origin:</b> {html.escape(definition['origin_description'])}<br>"
            f"<b>Axes:</b> {html.escape(definition['coordinate_system'])}<br>"
            f"<b>Sticker/label face:</b> {html.escape(str(housing.get('sticker_face') or 'Not specified'))}<br>"
            f"<b>Cable connection face:</b> {html.escape(str(housing.get('cable_face') or 'Not specified'))}</p>"
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<tr><th>Channel</th><th>Local point (mm)</th><th>Sensitive axis</th>"
            "<th>Reference from edge</th></tr>"
            f"{channel_rows}</table>"
            f"<h3>Sources</h3><ul>{source_rows or '<li>No external source supplied.</li>'}</ul>"
        )

    def _create_custom(self) -> None:
        dialog = CustomProbeDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._definition = dialog.probe_definition()
        self.accept()

    def _import_definition(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import magnetometer probe definition",
            "",
            "Field Workbench probe definition (*.fwprobe.json *.json);;All files (*)",
        )
        if not path:
            return
        try:
            definition = load_probe_definition(path)
        except ProbeDefinitionError as error:
            QMessageBox.warning(self, "Import probe definition", str(error))
            return
        self._definitions.append(definition)
        index = len(self._definitions) - 1
        item = QTreeWidgetItem(
            [definition["manufacturer"], definition["model"], "Imported"]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, index)
        self.probe_list.addTopLevelItem(item)
        self.probe_list.setCurrentItem(item)

    def _accept_selected(self) -> None:
        definition = self._selected_definition()
        if definition is None:
            return
        self._definition = copy.deepcopy(definition)
        self.accept()

    def probe_definition(self) -> dict[str, Any]:
        if self._definition is None:
            raise RuntimeError("Probe library dialog was not accepted.")
        return copy.deepcopy(self._definition)


class AiHelpDialog(QDialog):
    """Manual clipboard bridge between Field Workbench and an external AI chat."""

    def __init__(self, adapter: StudioAdapter, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.imported_ids: list[str] = []
        self.replaced_scene = False
        self.pasted_into_new_scene = False
        self.new_scene_adapter: StudioAdapter | None = None
        self._validated_text = ""
        self._activity_started_at: float | None = None
        self._activity_message = "Ready"
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(100)
        self._activity_timer.timeout.connect(self._refresh_activity_elapsed)
        self.setWindowTitle("AI Help — Scene Recipe")
        self.resize(900, 820)

        root = QVBoxLayout(self)
        # Keep the dialog from being squeezed below the combined minimum height of its
        # three workflow sections. Qt can otherwise be forced to overlap child widgets
        # when a top-level window is made shorter than their minimum-size requirements.
        root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        heading = QLabel("AI Help — Field Workbench Scene Recipe v1")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        intro = QLabel(
            "Use Field Workbench with an AI assistant in three steps. The transfer is manual copy/paste: "
            "the AI can reason about copied scene context and propose a changed complete scene, but it never "
            "controls the live Workbench. Nothing changes until the returned Scene Recipe validates and you "
            "choose Modify current scene or Paste into new scene. Pasted content is never executed as code."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        # Step 1: build the context packet that the user carries to an external AI chat.
        copy_group = QGroupBox("1 — Copy scene + AI hints")
        copy_layout = QVBoxLayout(copy_group)
        copy_hint = QLabel(
            "Choose what the AI should know, then copy the packet. It includes the current Scene Recipe "
            "capabilities and important Workbench tools the AI may recommend. Paste it into your preferred "
            "AI assistant and type your request beneath the USER INSTRUCTIONS marker at the end."
        )
        copy_hint.setWordWrap(True)
        copy_hint.setObjectName("hint")
        copy_layout.addWidget(copy_hint)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("AI guidance"))
        self.guidance_level = QComboBox()
        self.guidance_level.addItem("Exploratory — make a reasonable first pass", "exploratory")
        self.guidance_level.addItem("Balanced — ask only when it changes the build", "balanced")
        self.guidance_level.addItem("Strict — reproduce only what is known", "strict")
        self.guidance_level.setCurrentIndex(1)
        self.guidance_level.setToolTip(
            "Controls how readily the copied AI prompt stops for clarification. All levels require "
            "evidence-backed questions rather than invented alternatives. Exploratory favors a useful "
            "first pass (normally at most one blocker), Balanced asks only about unresolved choices that "
            "materially change the build, and Strict stops on unresolved source-faithful build geometry."
        )
        mode_row.addWidget(self.guidance_level)
        mode_row.addStretch(1)
        copy_layout.addLayout(mode_row)

        copy_row = QHBoxLayout()
        self.include_scene = QCheckBox("Include current scene context")
        self.include_scene.setChecked(True)
        self.include_scene.setToolTip(
            "Adds a compact scene summary to the clipboard guide. Triangle meshes and long sensor "
            "paths are summarized rather than copied in full."
        )
        self.include_scene.toggled.connect(self._context_options_changed)
        copy_row.addWidget(self.include_scene)
        self.include_drc = QCheckBox("Include DRC rules/results")
        self.include_drc.setChecked(True)
        self.include_drc.setToolTip(
            "Includes regional DRC rules, per-coil status, and actionable clearance/collision issues. "
            "A still-fresh completed Check design report is reused; otherwise a new DRC pass runs."
        )
        copy_row.addWidget(self.include_drc)
        copy_row.addStretch(1)
        self.copy_guide_button = QPushButton("Copy scene + AI hints")
        self.copy_guide_button.setObjectName("primaryButton")
        self.copy_guide_button.clicked.connect(self.copy_format_guide)
        copy_row.addWidget(self.copy_guide_button)
        copy_layout.addLayout(copy_row)

        self.copy_status = QLabel("")
        self.copy_status.setObjectName("hint")
        copy_layout.addWidget(self.copy_status)
        root.addWidget(copy_group)

        handoff_hint = QLabel(
            "↓ Paste that packet into your preferred AI assistant, add your request, then bring the "
            "returned Scene Recipe back here."
        )
        handoff_hint.setWordWrap(True)
        handoff_hint.setObjectName("sectionHeading")
        root.addWidget(handoff_hint)

        # Step 2: the JSON is primarily a transport envelope. Keep it visible/editable, but compact;
        # most users care more about the interpreted validation/advice in Step 3.
        paste_group = QGroupBox("2 — Paste the AI response")
        paste_layout = QVBoxLayout(paste_group)
        paste_hint = QLabel(
            "Paste the complete returned Scene Recipe JSON below. You do not need to read or edit the "
            "JSON unless you want to; Validate turns it into a plain-language review."
        )
        paste_hint.setWordWrap(True)
        paste_hint.setObjectName("hint")
        paste_layout.addWidget(paste_hint)
        self.recipe_edit = QPlainTextEdit()
        self.recipe_edit.setPlaceholderText(
            '{\n  "format": "fieldworkbench-scene-recipe",\n  "version": 1,\n  "objects": [...]\n}'
        )
        self.recipe_edit.setTabChangesFocus(False)
        self.recipe_edit.setMinimumHeight(80)
        self.recipe_edit.setMaximumHeight(140)
        self.recipe_edit.textChanged.connect(self._recipe_changed)
        paste_layout.addWidget(self.recipe_edit)

        validate_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate AI response")
        self.validate_button.setObjectName("primaryButton")
        self.validate_button.clicked.connect(self.validate_recipe)
        validate_row.addWidget(self.validate_button)
        validate_row.addStretch(1)
        paste_layout.addLayout(validate_row)
        root.addWidget(paste_group)

        # Step 3 gets the majority of the remaining vertical space: this is where Shining Star's
        # interpretation, assumptions, next steps, warnings, and object summary are meant to be read.
        review_group = QGroupBox("3 — Review the AI interpretation and choose destination")
        review_layout = QVBoxLayout(review_group)
        review_hint = QLabel(
            "Review what the AI thinks you asked for, its assumptions and suggested next steps, then "
            "either modify the current scene or paste the validated result into a new scene."
        )
        review_hint.setWordWrap(True)
        review_hint.setObjectName("hint")
        review_layout.addWidget(review_hint)
        self.validation_output = QPlainTextEdit()
        self.validation_output.setReadOnly(True)
        # This pane still receives most of the expandable vertical space, but keep its
        # hard minimum modest so the import controls always remain in their own row.
        self.validation_output.setMinimumHeight(120)
        self.validation_output.setPlaceholderText("Paste the AI response above, then click Validate AI response.")
        review_layout.addWidget(self.validation_output, 1)

        import_row = QHBoxLayout()
        self.modify_current_button = QPushButton("Modify current scene")
        self.modify_current_button.setObjectName("primaryButton")
        self.modify_current_button.setEnabled(False)
        self.modify_current_button.setToolTip(
            "Replace the current scene objects with the validated complete scene. Scene-wide notes and DRC settings are preserved."
        )
        self.modify_current_button.clicked.connect(self.modify_current_scene)
        self.paste_new_scene_button = QPushButton("Paste into new scene")
        self.paste_new_scene_button.setEnabled(False)
        self.paste_new_scene_button.setToolTip(
            "Create a new scene tab from the validated complete scene, leaving the current scene untouched."
        )
        self.paste_new_scene_button.clicked.connect(self.paste_into_new_scene)
        import_row.addStretch(1)
        import_row.addWidget(self.modify_current_button)
        import_row.addWidget(self.paste_new_scene_button)
        review_layout.addLayout(import_row)
        root.addWidget(review_group, 1)

        self.activity_progress = QProgressBar()
        self.activity_progress.setRange(0, 100)
        self.activity_progress.setValue(0)
        self.activity_progress.setTextVisible(True)
        self.activity_progress.setFormat("Ready")
        self.activity_progress.setMaximumHeight(20)
        self.activity_progress.setToolTip(
            "Shows scene-summary, fresh DRC pair-check, clipboard, and recipe-validation progress."
        )
        root.addWidget(self.activity_progress)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _current_guidance_level(self) -> str:
        return str(self.guidance_level.currentData() or "balanced")

    def _recipe_changed(self) -> None:
        self._validated_text = ""
        self.modify_current_button.setEnabled(False)
        self.paste_new_scene_button.setEnabled(False)

    def _context_options_changed(self) -> None:
        self.include_drc.setEnabled(self.include_scene.isChecked())

    def _activity_elapsed_seconds(self) -> float:
        if self._activity_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._activity_started_at)

    def _refresh_activity_elapsed(self) -> None:
        if self._activity_started_at is None:
            return
        self.activity_progress.setFormat(
            f"{self._activity_message}  •  elapsed {_format_elapsed_time(self._activity_elapsed_seconds())}"
        )

    def _start_activity(self, text: str) -> None:
        self._activity_started_at = time.monotonic()
        self._activity_message = str(text)
        self._activity_timer.start()
        self._set_activity_progress(0, text)

    def _finish_activity(self, text: str, *, value: int = 100) -> None:
        self._activity_message = str(text)
        self._set_activity_progress(value, text)
        self._activity_timer.stop()
        # Keep the final elapsed duration visible after the operation completes.
        self._refresh_activity_elapsed()
        self._activity_started_at = None

    def _set_activity_progress(self, value: int, text: str) -> None:
        """Update the shared AI-help activity bar and paint it before long synchronous work."""
        self.activity_progress.setRange(0, 100)
        self.activity_progress.setValue(max(0, min(100, int(value))))
        self._activity_message = str(text)
        if self._activity_started_at is None:
            self.activity_progress.setFormat(self._activity_message)
        else:
            self._refresh_activity_elapsed()
        # Prompt generation can include a full DRC pass and recipe validation/import can
        # build a temporary scene. Force status updates through the event queue so the
        # elapsed counter and determinate progress remain visibly alive.
        self.activity_progress.repaint()
        QApplication.processEvents()

    def _recipe_operation_progress(self, update: dict[str, Any]) -> None:
        self._set_activity_progress(
            int(update.get("percent", 0)),
            str(update.get("message", "Working with Scene Recipe…")),
        )

    def _set_prompt_busy(self, busy: bool) -> None:
        """Prevent context options changing while one clipboard prompt is being assembled."""
        self.copy_guide_button.setEnabled(not busy)
        self.include_scene.setEnabled(not busy)
        self.include_drc.setEnabled(not busy and self.include_scene.isChecked())
        self.guidance_level.setEnabled(not busy)

    def _prompt_progress(self, update: dict[str, Any]) -> None:
        """Mirror adapter scene/DRC prompt progress into the shared AI Help bar."""
        self._set_activity_progress(
            int(update.get("percent", 0)),
            str(update.get("message", "Preparing AI prompt…")),
        )

    def copy_format_guide(self) -> None:
        include_scene = self.include_scene.isChecked()
        include_drc = include_scene and self.include_drc.isChecked()
        activity = (
            "Preparing prompt + fresh DRC…"
            if include_drc
            else "Preparing AI prompt…"
        )
        self._start_activity(activity)
        self._set_prompt_busy(True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            text = self.adapter.scene_recipe_help_text(
                include_current_scene=include_scene,
                include_drc=include_drc,
                import_mode="replace",
                guidance_level=self._current_guidance_level(),
                progress_callback=self._prompt_progress,
            )
            self._set_activity_progress(99, "Copying prompt to clipboard…")
            QApplication.clipboard().setText(text)
        except Exception:
            self._finish_activity("Prompt copy failed")
            raise
        finally:
            QApplication.restoreOverrideCursor()
            self._set_prompt_busy(False)
        details: list[str] = []
        if include_scene:
            details.append("current-scene context")
        if include_drc:
            details.append("fresh DRC context")
        detail = f" with {' + '.join(details)}" if details else ""
        guidance_name = self.guidance_level.currentText().split(" — ", 1)[0]
        self.copy_status.setText(
            f"Copied scene + AI hints ({guidance_name} guidance){detail} to the clipboard. "
            "Paste it into your AI chat and add your request at the bottom."
        )
        self._finish_activity("Format guide copied")

    def validate_recipe(self) -> bool:
        text = self.recipe_edit.toPlainText()
        self._start_activity("Validating recipe…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.adapter.validate_scene_recipe(
                text,
                mode="replace",
                progress_callback=self._recipe_operation_progress,
            )
        except StudioOperationError as error:
            self._validated_text = ""
            self.modify_current_button.setEnabled(False)
            self.paste_new_scene_button.setEnabled(False)
            self.validation_output.setPlainText(f"ERROR\n{error}")
            self._finish_activity("Validation failed")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._set_activity_progress(92, "Preparing validation report…")
        lines: list[str] = []
        assistant_message = str(result.get("assistant_message") or "").strip()
        assumptions = list(result.get("assumptions", []))
        next_steps = list(result.get("suggested_next_steps", []))
        review_questions = list(result.get("review_questions", []))
        requires_clarification = bool(result.get("requires_clarification", False))
        clarifying_questions = list(result.get("clarifying_questions", []))
        if assistant_message:
            lines.extend(["AI interpretation", assistant_message, ""])
        if requires_clarification:
            lines.append("CLARIFICATION REQUIRED — scene import is disabled")
            lines.extend(f"• {item}" for item in clarifying_questions)
            lines.append("")
        if assumptions:
            lines.append("Assumptions")
            lines.extend(f"• {item}" for item in assumptions)
            lines.append("")
        if review_questions:
            lines.append("Worth confirming")
            lines.extend(f"• {item}" for item in review_questions)
            lines.append("")
        if next_steps:
            lines.append("Suggested next steps")
            lines.extend(f"• {item}" for item in next_steps)
            lines.append("")
        if requires_clarification:
            lines.append("✓ Clarification request recognized. No scene changes will be imported.")
        else:
            lines.append(f"✓ {result['object_count']} object(s) recognized and buildable.")
        warnings = list(result.get("warnings", []))
        if warnings:
            lines.append("")
            lines.append(f"Warnings ({len(warnings)}):")
            lines.extend(f"• {warning}" for warning in warnings)
        else:
            lines.append("✓ No validation warnings.")
        if result.get("objects"):
            lines.append("")
            lines.append("Validated complete scene:")
            lines.extend(
                f"• {entry.get('name', entry.get('id', 'Object'))} — {entry.get('type', 'Object')}"
                for entry in result["objects"]
            )
        self.validation_output.setPlainText("\n".join(lines))
        self._validated_text = text
        self.modify_current_button.setEnabled(not requires_clarification)
        self.paste_new_scene_button.setEnabled(not requires_clarification)
        self._finish_activity("Clarification required" if requires_clarification else "Validation complete")
        return True

    def _ensure_validated_recipe(self) -> str | None:
        text = self.recipe_edit.toPlainText()
        if text != self._validated_text and not self.validate_recipe():
            return None
        if not self._validated_text:
            return None
        return text

    def modify_current_scene(self) -> None:
        text = self._ensure_validated_recipe()
        if text is None:
            return
        answer = QMessageBox.question(
            self,
            "Modify current scene?",
            "This will replace all current scene objects with the validated complete scene. "
            "Scene-wide DRC settings and notes are preserved. The change is one undoable step.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_activity("Modifying current scene from recipe…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.adapter.import_scene_recipe(
                text,
                mode="replace",
                progress_callback=self._recipe_operation_progress,
            )
        except StudioOperationError as error:
            self._validated_text = ""
            self.modify_current_button.setEnabled(False)
            self.paste_new_scene_button.setEnabled(False)
            self.validation_output.setPlainText(f"ERROR\n{error}")
            self._finish_activity("Import failed")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.imported_ids = list(result.get("object_ids", []))
        self.replaced_scene = True
        self.pasted_into_new_scene = False
        self.new_scene_adapter = None
        self._finish_activity("Current scene modified")
        self.accept()

    def paste_into_new_scene(self) -> None:
        text = self._ensure_validated_recipe()
        if text is None:
            return
        self._start_activity("Building new scene from recipe…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            adapter = StudioAdapter()
            adapter.new_empty()
            result = adapter.import_scene_recipe(
                text,
                mode="replace",
                progress_callback=self._recipe_operation_progress,
            )
        except StudioOperationError as error:
            self._validated_text = ""
            self.modify_current_button.setEnabled(False)
            self.paste_new_scene_button.setEnabled(False)
            self.validation_output.setPlainText(f"ERROR\n{error}")
            self._finish_activity("New scene import failed")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.imported_ids = list(result.get("object_ids", []))
        self.replaced_scene = False
        self.pasted_into_new_scene = True
        self.new_scene_adapter = adapter
        self._finish_activity("New scene ready")
        self.accept()




class SceneExportDialog(QDialog):
    """Choose scene objects and a translator for portable-scene export."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        preview_base_figure: dict[str, Any] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self._updating_checks = False
        self._items: dict[str, QTreeWidgetItem] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self.setWindowTitle("Export scene")
        self.resize(1320, 760)

        root = QVBoxLayout(self)
        heading = QLabel("Export portable scene")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        intro = QLabel(
            "Field Workbench normalizes the checked physical scene into a portable model, then "
            "the selected translator writes the export independently of editor and optimizer state."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("sceneExportContentSplitter")
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.content_splitter.addWidget(left_panel)
        root.addWidget(self.content_splitter, 1)

        translator_box = QGroupBox("Translator")
        translator_form = QFormLayout(translator_box)
        self.translator_selector = QComboBox()
        self.translator_selector.setObjectName("sceneExportTranslatorSelector")
        self.translator_selector.addItem("OpenSCAD model (.scad)", TRANSLATOR_OPENSCAD)
        self.translator_selector.addItem("FEMM axisymmetric builder (.lua)", TRANSLATOR_FEMM)
        self.translator_selector.addItem(
            "Gmsh/GetDP 3D magnetostatic project (.getdp.zip)", TRANSLATOR_GETDP
        )
        self.translator_selector.addItem("COMSOL 6.4 Java builder (.java)", TRANSLATOR_COMSOL)
        self.translator_selector.addItem("Portable scene package (.fwpkg)", TRANSLATOR_PORTABLE)
        self.translator_selector.currentIndexChanged.connect(self._translator_changed)
        translator_form.addRow("Export as", self.translator_selector)
        self.translator_hint = QLabel()
        self.translator_hint.setObjectName("hint")
        self.translator_hint.setWordWrap(True)
        translator_form.addRow(self.translator_hint)
        left_layout.addWidget(translator_box)

        object_box = QGroupBox("Scene objects")
        object_layout = QVBoxLayout(object_box)
        self.object_tree = QTreeWidget()
        self.object_tree.setObjectName("sceneExportObjectTree")
        self.object_tree.setHeaderLabels(["Export / object", "Type", "Translator status"])
        self.object_tree.setRootIsDecorated(True)
        self.object_tree.setAlternatingRowColors(True)
        self.object_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.object_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.object_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.object_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.object_tree.itemChanged.connect(self._item_check_changed)
        object_layout.addWidget(self.object_tree, 1)

        selection_row = QHBoxLayout()
        self.select_visible_button = QPushButton("Select visible")
        self.select_visible_button.clicked.connect(self._select_visible)
        selection_row.addWidget(self.select_visible_button)
        self.select_all_button = QPushButton("Select all supported")
        self.select_all_button.clicked.connect(self._select_all_supported)
        selection_row.addWidget(self.select_all_button)
        self.clear_selection_button = QPushButton("Clear")
        self.clear_selection_button.clicked.connect(self._clear_selection)
        selection_row.addWidget(self.clear_selection_button)
        selection_row.addStretch(1)
        object_layout.addLayout(selection_row)
        self.selection_validation_hint = QLabel()
        self.selection_validation_hint.setObjectName("hint")
        self.selection_validation_hint.setWordWrap(True)
        object_layout.addWidget(self.selection_validation_hint)
        left_layout.addWidget(object_box, 1)

        options_box = QGroupBox("Translator options")
        options_layout = QVBoxLayout(options_box)
        self.include_construction_checkbox = QCheckBox(
            "Use saved winding / bobbin / flange construction where available"
        )
        self.include_construction_checkbox.setObjectName("sceneExportIncludeConstruction")
        self.include_construction_checkbox.setChecked(True)
        self.include_construction_checkbox.setToolTip(
            "For OpenSCAD, export the same physical assembly solids used by Workbench rendering "
            "and DRC. Coils without enabled construction are emitted as thin centreline references."
        )
        options_layout.addWidget(self.include_construction_checkbox)
        self.options_hint = QLabel(
            "The portable package always retains the physical construction description; this option "
            "only controls how the OpenSCAD translator renders coils."
        )
        self.options_hint.setObjectName("hint")
        self.options_hint.setWordWrap(True)
        options_layout.addWidget(self.options_hint)

        self.apply_field_scale_checkbox = QCheckBox(
            "Apply empirical Field correction to exported magnetic excitation"
        )
        self.apply_field_scale_checkbox.setObjectName("sceneExportApplyFieldScale")
        self.apply_field_scale_checkbox.setChecked(False)
        self.apply_field_scale_checkbox.setToolTip(
            "Off by default so FEMM/GetDP/COMSOL can act as independent physics checks. When enabled, "
            "each exported coil excitation is multiplied by its saved empirical Field correction."
        )
        options_layout.addWidget(self.apply_field_scale_checkbox)
        self.magnetic_options_hint = QLabel(
            "FEMM, GetDP and COMSOL use saved current × turns. Leave correction off for an "
            "independent solver check; enable it only to carry the calibrated Workbench amplitude."
        )
        self.magnetic_options_hint.setObjectName("hint")
        self.magnetic_options_hint.setWordWrap(True)
        options_layout.addWidget(self.magnetic_options_hint)

        self.getdp_options_box = QGroupBox("GetDP project options")
        self.getdp_options_box.setObjectName("sceneExportGetDPOptions")
        getdp_box_layout = QVBoxLayout(self.getdp_options_box)
        getdp_box_layout.setContentsMargins(6, 6, 6, 6)
        self.getdp_options_scroll = QScrollArea()
        self.getdp_options_scroll.setObjectName("sceneExportGetDPOptionsScroll")
        self.getdp_options_scroll.setWidgetResizable(True)
        self.getdp_options_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.getdp_options_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.getdp_options_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.getdp_options_scroll.setMinimumHeight(150)
        self.getdp_options_scroll.setMaximumHeight(300)
        self.getdp_options_contents = QWidget()
        self.getdp_options_contents.setObjectName("sceneExportGetDPOptionsContents")
        getdp_layout = QVBoxLayout(self.getdp_options_contents)
        getdp_layout.setContentsMargins(2, 2, 4, 2)
        getdp_box_layout.addWidget(self.getdp_options_scroll)
        self.getdp_options_scroll.setWidget(self.getdp_options_contents)
        self.getdp_use_physical_checkbox = QCheckBox(
            "Use saved physical winding construction when available"
        )
        self.getdp_use_physical_checkbox.setObjectName(
            "sceneExportGetDPUsePhysicalConstruction"
        )
        self.getdp_use_physical_checkbox.setChecked(True)
        self.getdp_use_physical_checkbox.setToolTip(
            "When enabled, valid saved winding-pack dimensions become the finite GetDP source. "
            "Coils without physical construction still use a compact numerical surrogate. "
            "Turn this off to force numerical winding surrogates for every selected coil."
        )
        self.getdp_use_physical_checkbox.toggled.connect(self._refresh_export_button)
        getdp_layout.addWidget(self.getdp_use_physical_checkbox)

        getdp_preset_form = QFormLayout()
        self.getdp_mesh_preset_selector = QComboBox()
        self.getdp_mesh_preset_selector.setObjectName("sceneExportGetDPMeshPreset")
        for key in ("fast", "standard", "validation"):
            label = str(GETDP_MESH_PRESETS[key]["label"])
            self.getdp_mesh_preset_selector.addItem(label, key)
        self.getdp_mesh_preset_selector.addItem("Custom", "custom")
        self.getdp_mesh_preset_selector.setCurrentIndex(
            self.getdp_mesh_preset_selector.findData("standard")
        )
        self.getdp_mesh_preset_selector.setToolTip(
            "Fast is intended for quick geometry checks. Standard is the normal engineering-quality "
            "mesh and is tuned to remain practical on 16 GB-class systems for representative scenes. "
            "Validation concentrates the strongest refinement around sources and samples, automatically "
            "grades very fine winding elements into the surrounding air, uses Gmsh HXT for parallel 3-D "
            "tetrahedralization, and may require a higher-memory workstation. All presets use the same "
            "automatically sized infinite-space boundary."
        )
        self.getdp_mesh_preset_selector.currentIndexChanged.connect(
            self._getdp_mesh_preset_changed
        )
        getdp_preset_form.addRow("Mesh quality", self.getdp_mesh_preset_selector)
        getdp_layout.addLayout(getdp_preset_form)
        self.getdp_mesh_hint = QLabel(
            "FWB automatically sizes the local air region and surrounds it with a GetDP infinite shell. "
            "Mesh quality changes local discretization, not how far away an artificial boundary is placed."
        )
        self.getdp_mesh_hint.setObjectName("hint")
        self.getdp_mesh_hint.setWordWrap(True)
        getdp_layout.addWidget(self.getdp_mesh_hint)

        self.getdp_advanced_toggle = QToolButton()
        self.getdp_advanced_toggle.setObjectName("sceneExportGetDPAdvancedToggle")
        self.getdp_advanced_toggle.setText("Advanced mesh controls")
        self.getdp_advanced_toggle.setCheckable(True)
        self.getdp_advanced_toggle.setChecked(False)
        self.getdp_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.getdp_advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        getdp_layout.addWidget(self.getdp_advanced_toggle)

        self.getdp_advanced_widget = QWidget()
        self.getdp_advanced_widget.setObjectName("sceneExportGetDPAdvanced")
        getdp_advanced_form = QFormLayout(self.getdp_advanced_widget)
        getdp_advanced_form.setContentsMargins(0, 0, 0, 0)

        self.getdp_source_elements_spin = CompactDoubleSpinBox()
        self.getdp_source_elements_spin.setObjectName("sceneExportGetDPSourceElements")
        self.getdp_source_elements_spin.setRange(1.0, 12.0)
        self.getdp_source_elements_spin.setDecimals(1)
        self.getdp_source_elements_spin.setSingleStep(0.5)
        self.getdp_source_elements_spin.setToolTip(
            "Approximate number of Gmsh characteristic lengths across the smallest winding-pack "
            "dimension. Higher values make the source mesh finer."
        )
        getdp_advanced_form.addRow("Winding elements", self.getdp_source_elements_spin)

        self.getdp_near_divisions_spin = QSpinBox()
        self.getdp_near_divisions_spin.setObjectName("sceneExportGetDPNearDivisions")
        self.getdp_near_divisions_spin.setRange(4, 128)
        self.getdp_near_divisions_spin.setToolTip(
            "Nominal divisions per largest coil characteristic radius in the source/sample "
            "neighbourhood. Higher values make that region finer."
        )
        getdp_advanced_form.addRow("Near-region divisions", self.getdp_near_divisions_spin)

        self.getdp_far_divisions_spin = QSpinBox()
        self.getdp_far_divisions_spin.setObjectName("sceneExportGetDPFarDivisions")
        self.getdp_far_divisions_spin.setRange(4, 128)
        self.getdp_far_divisions_spin.setToolTip(
            "Nominal divisions across the automatically sized ordinary-air radius. Higher values "
            "reduce the coarsening allowed away from sources and samples."
        )
        getdp_advanced_form.addRow("Far-air divisions", self.getdp_far_divisions_spin)

        getdp_layout.addWidget(self.getdp_advanced_widget)
        self.getdp_advanced_widget.setVisible(False)
        self.getdp_advanced_toggle.toggled.connect(self._getdp_advanced_toggled)

        for editor in (
            self.getdp_source_elements_spin,
            self.getdp_near_divisions_spin,
            self.getdp_far_divisions_spin,
        ):
            editor.valueChanged.connect(self._getdp_custom_mesh_changed)

        self.getdp_plan_hint = QLabel()
        self.getdp_plan_hint.setObjectName("hint")
        self.getdp_plan_hint.setWordWrap(True)
        getdp_layout.addWidget(self.getdp_plan_hint)

        # Treat the wheel as navigation while the pointer is over GetDP form
        # editors.  Spin boxes and closed combo boxes otherwise consume wheel
        # events and can silently alter export settings while the user is only
        # trying to move through the nested options pane.
        self._getdp_options_wheel_filter = EditorWheelScrollFilter(
            self.getdp_options_scroll
        )
        for editor_type in (QDoubleSpinBox, QSpinBox, QComboBox):
            for editor in self.getdp_options_contents.findChildren(editor_type):
                editor.installEventFilter(self._getdp_options_wheel_filter)

        options_layout.addWidget(self.getdp_options_box)
        self._updating_getdp_mesh_controls = False
        self._apply_getdp_mesh_preset("standard")

        self.femm_slice_row = QWidget()
        femm_slice_layout = QFormLayout(self.femm_slice_row)
        femm_slice_layout.setContentsMargins(0, 0, 0, 0)
        self.femm_slice_plane_selector = QComboBox()
        self.femm_slice_plane_selector.setObjectName("sceneExportFemmSlicePlane")
        self.femm_slice_plane_selector.addItem("Auto — best principal source plane", "auto")
        self.femm_slice_plane_selector.addItem("XZ — source X/Z section", "xz")
        self.femm_slice_plane_selector.addItem("YZ — source Y/Z section", "yz")
        self.femm_slice_plane_selector.addItem("XY — source X/Y section", "xy")
        self.femm_slice_plane_selector.setToolTip(
            "Choose the literal source-world slice orientation used for passive reference outlines. "
            "This now matches the 2D viewer plane semantics exactly; the outline is never silently "
            "rotated into a different meridional plane."
        )
        self.femm_slice_plane_selector.currentIndexChanged.connect(self._refresh_export_button)
        femm_slice_layout.addRow("Reference slice", self.femm_slice_plane_selector)
        self.femm_slice_offset = QLineEdit("—")
        self.femm_slice_offset.setObjectName("sceneExportFemmSliceOffset")
        self.femm_slice_offset.setReadOnly(True)
        self.femm_slice_offset.setToolTip(
            "Automatically derived from the selected coils' common axis. The chosen principal "
            "plane is shifted so it passes through that axis; there is no manual offset to tune."
        )
        femm_slice_layout.addRow("Axis-aligned offset", self.femm_slice_offset)
        self.femm_half_selector = QComboBox()
        self.femm_half_selector.setObjectName("sceneExportFemmHalf")
        self.femm_half_selector.addItem("Side 1", "positive")
        self.femm_half_selector.addItem("Side 2", "negative")
        self.femm_half_selector.setToolTip(
            "FEMM only has r ≥ 0. Choose which side of the source slice's common coil axis is "
            "retained and mapped onto FEMM +r."
        )
        self.femm_half_selector.currentIndexChanged.connect(self._refresh_export_button)
        femm_slice_layout.addRow("Export half", self.femm_half_selector)
        self.femm_slice_hint = QLabel(
            "Selected passive geometry is intersected with the exact source-world plane. Its "
            "offset follows the selected coils' common axis automatically. Choose which source "
            "half is folded onto FEMM's required r ≥ 0 side; the retained cross-section is then "
            "mapped with rigid 2D rotation/translation only, preserving its shape."
        )
        self.femm_slice_hint.setObjectName("hint")
        self.femm_slice_hint.setWordWrap(True)
        femm_slice_layout.addRow(self.femm_slice_hint)
        options_layout.addWidget(self.femm_slice_row)
        left_layout.addWidget(options_box)

        preview_box = QGroupBox("Preview")
        preview_box.setObjectName("sceneExportPreviewBox")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setObjectName("sceneExportPreviewTabs")
        self.scene_preview_plot = PlotView(
            allow_fullscreen=False, image_filename="scene-export-preview"
        )
        self.slice_preview_plot = PlotView(
            allow_fullscreen=False, image_filename="scene-export-slice-preview"
        )
        self.scene_preview_tab_index = self.preview_tabs.addTab(self.scene_preview_plot, "Scene")
        self.slice_preview_tab_index = self.preview_tabs.addTab(self.slice_preview_plot, "Slice")
        preview_layout.addWidget(self.preview_tabs, 1)
        self.preview_hint = QLabel(
            "Checked objects are highlighted. For FEMM, the translucent plane is the exact "
            "source slice and the Slice tab shows the same cross-section logic as the 2D viewer."
        )
        self.preview_hint.setObjectName("hint")
        self.preview_hint.setWordWrap(True)
        preview_layout.addWidget(self.preview_hint)
        self.content_splitter.addWidget(preview_box)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([780, 500])
        self._preview_base_figure: dict[str, Any] | None = preview_base_figure
        self.finished.connect(lambda _code: self._cleanup_previews())

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.export_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.export_button.setText("Export…")
        self.export_button.setObjectName("primaryButton")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._translator_changed()
        self._select_visible()

    def translator(self) -> str:
        return str(self.translator_selector.currentData() or TRANSLATOR_OPENSCAD)

    def include_construction(self) -> bool:
        return bool(self.include_construction_checkbox.isChecked())

    def apply_field_scale(self) -> bool:
        return bool(self.apply_field_scale_checkbox.isChecked())

    def getdp_use_physical_construction(self) -> bool:
        return bool(self.getdp_use_physical_checkbox.isChecked())

    def getdp_mesh_preset(self) -> str:
        return str(self.getdp_mesh_preset_selector.currentData() or "standard")

    def getdp_source_elements(self) -> float:
        return float(self.getdp_source_elements_spin.value())

    def getdp_near_radius_divisions(self) -> float:
        return float(self.getdp_near_divisions_spin.value())

    def getdp_far_radius_divisions(self) -> float:
        return float(self.getdp_far_divisions_spin.value())

    def _apply_getdp_mesh_preset(self, preset: str) -> None:
        values = GETDP_MESH_PRESETS.get(str(preset))
        if values is None:
            return
        self._updating_getdp_mesh_controls = True
        try:
            self.getdp_source_elements_spin.setValue(float(values["source_elements"]))
            self.getdp_near_divisions_spin.setValue(
                int(round(float(values["near_radius_divisions"])))
            )
            self.getdp_far_divisions_spin.setValue(
                int(round(float(values["far_radius_divisions"])))
            )
        finally:
            self._updating_getdp_mesh_controls = False

    def _getdp_mesh_preset_changed(self, _index: int = -1) -> None:
        preset = self.getdp_mesh_preset()
        if preset != "custom":
            self._apply_getdp_mesh_preset(preset)
        self._refresh_export_button()

    def _getdp_custom_mesh_changed(self, _value: Any = None) -> None:
        if self._updating_getdp_mesh_controls:
            return
        custom_index = self.getdp_mesh_preset_selector.findData("custom")
        if custom_index >= 0 and self.getdp_mesh_preset_selector.currentIndex() != custom_index:
            self._updating_getdp_mesh_controls = True
            try:
                self.getdp_mesh_preset_selector.setCurrentIndex(custom_index)
            finally:
                self._updating_getdp_mesh_controls = False
        self._refresh_export_button()

    def _getdp_advanced_toggled(self, checked: bool) -> None:
        self.getdp_advanced_widget.setVisible(bool(checked))
        self.getdp_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _refresh_getdp_plan_hint(self) -> None:
        if self.translator() != TRANSLATOR_GETDP:
            self.getdp_plan_hint.clear()
            return
        selected = self.selected_object_ids()
        if not selected:
            self.getdp_plan_hint.setText("Select at least one supported coil to preview the GetDP mesh plan.")
            return
        try:
            scene = build_portable_scene(self.adapter, selected)
            plan = getdp_export_plan(
                scene,
                apply_field_scale=self.apply_field_scale(),
                use_physical_construction=self.getdp_use_physical_construction(),
                mesh_preset=self.getdp_mesh_preset(),
                source_elements=self.getdp_source_elements(),
                near_radius_divisions=self.getdp_near_radius_divisions(),
                far_radius_divisions=self.getdp_far_radius_divisions(),
            )
        except Exception as error:  # noqa: BLE001 - selection validation owns blocking errors
            self.getdp_plan_hint.setText(f"Mesh preview unavailable: {error}")
            return
        physical = int(plan["physical_source_count"])
        numerical = int(plan["numerical_source_count"])
        source_note = []
        if physical:
            source_note.append(f"{physical} physical")
        if numerical:
            source_note.append(f"{numerical} numerical")
        source_text = " + ".join(source_note) or "no"
        self.getdp_plan_hint.setText(
            f"Planned sources: {source_text}. Derived characteristic lengths: winding "
            f"{float(plan['source_mesh_mm']):.3g} mm • near/sample {float(plan['near_mesh_mm']):.3g} mm • "
            f"far air {float(plan['far_mesh_mm']):.3g} mm • infinite shell {float(plan['shell_mesh_mm']):.3g} mm. "
            + (
                f"Automatic winding-to-air grading: air interface {float(plan['source_grading_air_min_mm']):.3g} mm "
                f"through {float(plan['source_grading_end_mm']):.3g} mm. "
                if bool(plan.get("source_grading_enabled"))
                else ""
            )
            + (
                "3-D mesher HXT (system threads). "
                if str(plan.get("gmsh_mesher")) == "HXT"
                else ""
            )
            + f"Automatic air/shell radii {float(plan['inner_air_radius_mm']):.3g} / "
            f"{float(plan['shell_outer_radius_mm']):.3g} mm; {int(plan['sample_count'])} sample point"
            f"{'s' if int(plan['sample_count']) != 1 else ''}."
        )

    def femm_reference_slice_plane(self) -> str:
        return str(self.femm_slice_plane_selector.currentData() or "auto")

    def femm_reference_half(self) -> str:
        return str(self.femm_half_selector.currentData() or "positive")

    def _femm_slice_context(self) -> tuple[dict[str, Any], dict[str, Any], float]:
        """Resolve the exact source plane, axis-derived offset, and selected half."""
        portable = build_portable_scene(self.adapter, self.selected_object_ids())
        alignment = _femm_alignment(portable)
        basis = _femm_reference_slice_basis(alignment, self.femm_reference_slice_plane())
        offset_mm = _femm_reference_slice_offset_from_axis(alignment, basis)
        basis = _femm_reference_half_basis(basis, self.femm_reference_half())
        return alignment, basis, offset_mm

    def femm_reference_slice_offset_mm(self) -> float:
        """Return the read-only slice offset currently implied by the common coil axis."""
        try:
            _alignment, _basis, offset_mm = self._femm_slice_context()
        except Exception:  # noqa: BLE001 - UI display fallback
            return 0.0
        return float(offset_mm)

    def _refresh_femm_slice_controls(self) -> None:
        """Show the automatic axis offset and human-readable source-side choices."""
        try:
            _alignment, basis, offset_mm = self._femm_slice_context()
            base_basis = _femm_reference_slice_basis(
                _alignment, self.femm_reference_slice_plane()
            )
            positive = _femm_direction_label(np.asarray(base_basis["radial_world"], dtype=float))
            negative = _femm_direction_label(-np.asarray(base_basis["radial_world"], dtype=float))
            self.femm_slice_offset.setText(f"{offset_mm:.2f} mm — common axis")
            self.femm_half_selector.setItemText(0, f"{positive} side")
            self.femm_half_selector.setItemText(1, f"{negative} side")
        except Exception:  # noqa: BLE001 - invalid selection is explained by validation label
            self.femm_slice_offset.setText("—")
            self.femm_half_selector.setItemText(0, "Side 1")
            self.femm_half_selector.setItemText(1, "Side 2")

    def selected_object_ids(self) -> list[str]:
        result: list[str] = []
        for object_id, item in self._items.items():
            record = self._records.get(object_id, {})
            if record.get("container") or not record.get("supported"):
                continue
            if item.checkState(0) == Qt.CheckState.Checked:
                result.append(object_id)
        return result

    def _checked_supported_ids(self) -> set[str]:
        return set(self.selected_object_ids())

    def _translator_changed(self, _index: int = -1) -> None:
        previous = self._checked_supported_ids()
        translator = self.translator()
        catalog = export_catalog(self.adapter, translator)
        self._records = {str(record["id"]): record for record in catalog}
        self.translator_hint.setText(translator_description(translator))
        openscad = translator == TRANSLATOR_OPENSCAD
        magnetic = translator in {TRANSLATOR_FEMM, TRANSLATOR_GETDP, TRANSLATOR_COMSOL}
        self.include_construction_checkbox.setEnabled(openscad)
        self.include_construction_checkbox.setVisible(openscad)
        self.options_hint.setVisible(openscad)
        self.apply_field_scale_checkbox.setEnabled(magnetic)
        self.apply_field_scale_checkbox.setVisible(magnetic)
        self.magnetic_options_hint.setVisible(magnetic)
        self.getdp_options_box.setVisible(translator == TRANSLATOR_GETDP)
        self.femm_slice_row.setVisible(translator == TRANSLATOR_FEMM)
        self.selection_validation_hint.setVisible(
            translator in {TRANSLATOR_FEMM, TRANSLATOR_GETDP}
        )
        self.preview_tabs.setTabVisible(
            self.slice_preview_tab_index, translator == TRANSLATOR_FEMM
        )

        self._updating_checks = True
        try:
            self.object_tree.clear()
            self._items = {}
            for record in catalog:
                object_id = str(record["id"])
                item = QTreeWidgetItem(
                    [
                        str(record.get("label") or object_id),
                        str(record.get("type_label") or record.get("type") or "Object"),
                        str(record.get("status") or ""),
                    ]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, object_id)
                item.setToolTip(0, object_id)
                item.setToolTip(2, str(record.get("status") or ""))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if record.get("container"):
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                elif record.get("supported"):
                    item.setCheckState(
                        0,
                        Qt.CheckState.Checked if object_id in previous else Qt.CheckState.Unchecked,
                    )
                else:
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self._items[object_id] = item

            for record in catalog:
                object_id = str(record["id"])
                item = self._items[object_id]
                parent_item = self._items.get(str(record.get("parent") or ""))
                if parent_item is None:
                    self.object_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
            self.object_tree.expandAll()
            self._refresh_all_containers()
        finally:
            self._updating_checks = False
        self._refresh_export_button()

    def _set_descendants(self, item: QTreeWidgetItem, checked: bool) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            object_id = str(child.data(0, Qt.ItemDataRole.UserRole) or "")
            record = self._records.get(object_id, {})
            if record.get("container"):
                child.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self._set_descendants(child, checked)
            elif record.get("supported"):
                child.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _refresh_container(self, item: QTreeWidgetItem) -> None:
        states: list[Qt.CheckState] = []
        for index in range(item.childCount()):
            child = item.child(index)
            object_id = str(child.data(0, Qt.ItemDataRole.UserRole) or "")
            record = self._records.get(object_id, {})
            if record.get("container") or record.get("supported"):
                states.append(child.checkState(0))
        if not states or all(state == Qt.CheckState.Unchecked for state in states):
            item.setCheckState(0, Qt.CheckState.Unchecked)
        elif all(state == Qt.CheckState.Checked for state in states):
            item.setCheckState(0, Qt.CheckState.Checked)
        else:
            item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _refresh_ancestors(self, item: QTreeWidgetItem | None) -> None:
        while item is not None:
            object_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
            if self._records.get(object_id, {}).get("container"):
                self._refresh_container(item)
            item = item.parent()

    def _refresh_all_containers(self) -> None:
        containers = [
            item
            for object_id, item in self._items.items()
            if self._records.get(object_id, {}).get("container")
        ]
        for item in reversed(containers):
            self._refresh_container(item)

    def _item_check_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_checks or column != 0:
            return
        object_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        record = self._records.get(object_id, {})
        self._updating_checks = True
        try:
            if record.get("container") and item.checkState(0) != Qt.CheckState.PartiallyChecked:
                self._set_descendants(item, item.checkState(0) == Qt.CheckState.Checked)
            self._refresh_ancestors(item.parent())
        finally:
            self._updating_checks = False
        self._refresh_export_button()

    def _select_visible(self) -> None:
        self._updating_checks = True
        try:
            for object_id, item in self._items.items():
                record = self._records.get(object_id, {})
                if record.get("container"):
                    continue
                if record.get("supported"):
                    item.setCheckState(
                        0,
                        Qt.CheckState.Checked if record.get("visible", True) else Qt.CheckState.Unchecked,
                    )
            self._refresh_all_containers()
        finally:
            self._updating_checks = False
        self._refresh_export_button()

    def _select_all_supported(self) -> None:
        self._updating_checks = True
        try:
            for object_id, item in self._items.items():
                record = self._records.get(object_id, {})
                if not record.get("container") and record.get("supported"):
                    item.setCheckState(0, Qt.CheckState.Checked)
            self._refresh_all_containers()
        finally:
            self._updating_checks = False
        self._refresh_export_button()

    def _clear_selection(self) -> None:
        self._updating_checks = True
        try:
            for object_id, item in self._items.items():
                record = self._records.get(object_id, {})
                if record.get("container") or record.get("supported"):
                    item.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            self._updating_checks = False
        self._refresh_export_button()

    def _refresh_export_button(self, *_args) -> None:
        selected = self.selected_object_ids()
        enabled = bool(selected)
        if self.translator() == TRANSLATOR_FEMM:
            valid, message = femm_selection_status(
                self.adapter,
                selected,
                self.femm_reference_slice_plane(),
                self.femm_reference_half(),
            )
            self.selection_validation_hint.setText(message)
            enabled = enabled and valid
        elif self.translator() == TRANSLATOR_GETDP:
            valid, message = getdp_selection_status(
                self.adapter,
                selected,
                use_physical_construction=self.getdp_use_physical_construction(),
            )
            self.selection_validation_hint.setText(message)
            enabled = enabled and valid
        else:
            self.selection_validation_hint.clear()
        self.export_button.setEnabled(enabled)
        self._refresh_getdp_plan_hint()
        self._refresh_femm_slice_controls()
        self._refresh_preview()

    def _cleanup_previews(self) -> None:
        self.scene_preview_plot.cleanup()
        self.slice_preview_plot.cleanup()

    def _preview_plane_span_mm(self) -> float:
        """Choose a useful source-scene plane size for the compact 3-D preview."""
        figure = self._preview_base_figure or {}
        maximum = 0.0
        for trace in figure.get("data", []):
            if not isinstance(trace, dict):
                continue
            for key in ("x", "y", "z"):
                values = trace.get(key)
                if values is None:
                    continue
                try:
                    array = np.asarray(
                        [value for value in values if value is not None], dtype=float
                    )
                except (TypeError, ValueError):
                    continue
                finite = array[np.isfinite(array)]
                if finite.size:
                    maximum = max(maximum, float(np.max(np.abs(finite))))
        return max(300.0, min(5000.0, 2.4 * maximum if maximum > 0.0 else 300.0))

    def _refresh_preview(self) -> None:
        """Synchronize the export preview with checks and translator options."""
        if not hasattr(self, "scene_preview_plot"):
            return
        if self._preview_base_figure is None:
            try:
                self._preview_base_figure = self.adapter.base_figure()
            except Exception:  # noqa: BLE001 - preview must never block export
                self._preview_base_figure = {"data": [], "layout": {"template": "plotly_white"}}
        selected = self.selected_object_ids()
        overlay = None
        femm_context = None
        if self.translator() == TRANSLATOR_FEMM:
            try:
                alignment, basis, offset_mm = self._femm_slice_context()
                plane = str(basis["plane"])
                femm_context = (alignment, basis, offset_mm)
            except Exception:  # noqa: BLE001 - preview fallback only
                plane = self.femm_reference_slice_plane()
                if plane == "auto":
                    plane = "xz"
                offset_mm = 0.0
            overlay = {
                "kind": "plane",
                "plane": plane,
                "offset_mm": offset_mm,
                "span_mm": self._preview_plane_span_mm(),
            }
        try:
            figure = self.adapter.compose_figure(
                self._preview_base_figure,
                selected_ids=selected,
                analysis_overlay=overlay,
            )
            self.scene_preview_plot.set_figure(figure)
        except Exception:  # noqa: BLE001 - preview must never block export
            pass

        if self.translator() != TRANSLATOR_FEMM:
            return
        plane = str((overlay or {}).get("plane", "xz"))
        offset_mm = float((overlay or {}).get("offset_mm", 0.0))
        try:
            traces = self.adapter._scene_cross_section_traces_2d(  # noqa: SLF001
                plane=plane,
                offset_mm=offset_mm,
                span_mm=10000.0,
                object_ids=selected,
            )
            labels = {
                "xy": ("X (mm)", "Y (mm)"),
                "xz": ("X (mm)", "Z (mm)"),
                "yz": ("Y (mm)", "Z (mm)"),
            }[plane]
            shapes: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            half_note = ""
            if femm_context is not None:
                alignment, basis, _resolved_offset = femm_context
                u_world = np.asarray(basis["u_world"], dtype=float)
                v_world = np.asarray(basis["v_world"], dtype=float)
                origin = np.asarray(alignment["origin_world_mm"], dtype=float)
                anchor_uv = np.asarray(
                    [float(np.dot(origin, u_world)), float(np.dot(origin, v_world))],
                    dtype=float,
                )
                radial_uv = np.asarray(basis["radial_uv"], dtype=float)
                half_label = _femm_direction_label(np.asarray(basis["radial_world"], dtype=float))
                half_note = f" · export {half_label} half"
                arrow_length = max(20.0, min(60.0, 0.12 * self._preview_plane_span_mm()))
                tip = anchor_uv + arrow_length * radial_uv
                annotations.append(
                    {
                        "x": float(tip[0]),
                        "y": float(tip[1]),
                        "ax": float(anchor_uv[0]),
                        "ay": float(anchor_uv[1]),
                        "xref": "x",
                        "yref": "y",
                        "axref": "x",
                        "ayref": "y",
                        "text": f"FEMM +r: {half_label}",
                        "showarrow": True,
                    }
                )
                if basis.get("axis_parallel_to_slice", False):
                    axis_world = np.asarray(alignment["axis_world"], dtype=float)
                    axis_uv = np.asarray(
                        [float(np.dot(axis_world, u_world)), float(np.dot(axis_world, v_world))],
                        dtype=float,
                    )
                    norm = float(np.linalg.norm(axis_uv))
                    if norm > 1e-12:
                        axis_uv /= norm
                        span = self._preview_plane_span_mm()
                        first = anchor_uv - span * axis_uv
                        second = anchor_uv + span * axis_uv
                        shapes.append(
                            {
                                "type": "line",
                                "x0": float(first[0]),
                                "y0": float(first[1]),
                                "x1": float(second[0]),
                                "y1": float(second[1]),
                                "line": {"dash": "dash", "width": 1},
                            }
                        )
            slice_figure = {
                "data": traces,
                "layout": {
                    "template": "plotly_white",
                    "margin": {"l": 48, "r": 12, "t": 34, "b": 44},
                    "title": {
                        "text": f"{plane.upper()} axis slice @ {offset_mm:g} mm{half_note}",
                        "x": 0.5,
                        "font": {"size": 13},
                    },
                    "showlegend": False,
                    "shapes": shapes,
                    "annotations": annotations,
                    "xaxis": {"title": labels[0], "zeroline": True},
                    "yaxis": {
                        "title": labels[1],
                        "zeroline": True,
                        "scaleanchor": "x",
                        "scaleratio": 1,
                    },
                },
            }
            self.slice_preview_plot.set_figure(slice_figure)
        except Exception:  # noqa: BLE001 - preview must never block export
            pass


class SceneObjectImportDialog(QDialog):
    """Checkbox tree for copying selected objects from another saved scene."""

    def __init__(self, catalog: list[dict[str, Any]], source_name: str, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.catalog = [copy.deepcopy(item) for item in catalog]
        self._updating_checks = False
        self.setWindowTitle("Import objects from scene")
        self.resize(650, 520)

        root = QVBoxLayout(self)
        heading = QLabel("Import objects from saved scene")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            f"Choose objects to copy from {source_name}. Selecting a group selects its contents; "
            "checking an individual child without its group imports that object at the scene root. "
            "Scene notes, snapshots, and scene-wide DRC settings are not merged."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Type"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.tree, 1)

        self._items: dict[str, QTreeWidgetItem] = {}
        for record in self.catalog:
            object_id = str(record.get("id", ""))
            if not object_id:
                continue
            item = QTreeWidgetItem(
                [str(record.get("label") or object_id), str(record.get("type_label") or record.get("type") or "Object")]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, object_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setToolTip(0, object_id)
            self._items[object_id] = item

        for record in self.catalog:
            object_id = str(record.get("id", ""))
            item = self._items.get(object_id)
            if item is None:
                continue
            parent_id = str(record.get("parent") or "")
            parent_item = self._items.get(parent_id)
            if parent_item is None:
                self.tree.addTopLevelItem(item)
            else:
                parent_item.addChild(item)
        self.tree.expandAll()
        self.tree.itemChanged.connect(self._item_check_changed)

        selection_row = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda _checked=False: self._set_all(Qt.CheckState.Checked))
        clear_all = QPushButton("Clear")
        clear_all.clicked.connect(lambda _checked=False: self._set_all(Qt.CheckState.Unchecked))
        selection_row.addWidget(select_all)
        selection_row.addWidget(clear_all)
        selection_row.addStretch(1)
        root.addLayout(selection_row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.import_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.import_button.setText("Import selected")
        self.import_button.setObjectName("primaryButton")
        self.import_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

    def _set_descendants(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            child.setCheckState(0, state)
            self._set_descendants(child, state)

    def _refresh_ancestors(self, item: QTreeWidgetItem | None) -> None:
        while item is not None:
            states = [item.child(index).checkState(0) for index in range(item.childCount())]
            if states and all(state == Qt.CheckState.Checked for state in states):
                state = Qt.CheckState.Checked
            elif states and all(state == Qt.CheckState.Unchecked for state in states):
                state = Qt.CheckState.Unchecked
            else:
                state = Qt.CheckState.PartiallyChecked
            item.setCheckState(0, state)
            item = item.parent()

    def _item_check_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_checks or column != 0:
            return
        self._updating_checks = True
        try:
            state = item.checkState(0)
            if state in {Qt.CheckState.Checked, Qt.CheckState.Unchecked}:
                self._set_descendants(item, state)
            self._refresh_ancestors(item.parent())
        finally:
            self._updating_checks = False
        self._update_import_button()

    def _set_all(self, state: Qt.CheckState) -> None:
        self._updating_checks = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(index)
                item.setCheckState(0, state)
                self._set_descendants(item, state)
        finally:
            self._updating_checks = False
        self._update_import_button()

    def checked_ids(self) -> list[str]:
        selected: list[str] = []
        for record in self.catalog:
            object_id = str(record.get("id", ""))
            item = self._items.get(object_id)
            if item is not None and item.checkState(0) == Qt.CheckState.Checked:
                selected.append(object_id)
        return selected

    def _update_import_button(self) -> None:
        self.import_button.setEnabled(bool(self.checked_ids()))


class UpennGbmImportWorker(QObject):
    """Download and surface one case away from the GUI thread."""

    progress = Signal(dict)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        case: UpennGbmCase,
        region_keys: list[str],
        cache_directory: str,
    ):
        super().__init__()
        self.case = case
        self.region_keys = list(region_keys)
        self.cache_directory = str(cache_directory)
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            result = import_upenn_gbm_case(
                self.case,
                self.region_keys,
                self.cache_directory,
                progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
            )
        except UpennGbmImportCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class UpennGbmCatalogueWorker(QObject):
    """Refresh and validate the release catalogue away from the GUI thread."""

    progress = Signal(dict)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cache_directory: str):
        super().__init__()
        self.cache_directory = str(cache_directory)
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            catalogue = refresh_upenn_gbm_catalogue(
                self.cache_directory,
                progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
            )
        except UpennGbmImportCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(catalogue)


class _UpennCaseTreeItem(QTreeWidgetItem):
    """Use hidden typed values when the user sorts case-statistic columns."""

    SORT_ROLE = Qt.ItemDataRole.UserRole + 1

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        left = self.data(column, self.SORT_ROLE)
        right = other.data(column, self.SORT_ROLE)
        if left is not None and right is not None:
            try:
                return left < right
            except TypeError:
                return str(left).casefold() < str(right).casefold()
        return super().__lt__(other)


class UpennGbmImportDialog(QDialog):
    """Browse supported UPENN-GBM segmentations and import selected regions."""

    def __init__(self, parent=None, *, cache_directory: str | Path | None = None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Import UPENN-GBM case")
        self.resize(1120, 720)
        self.setMinimumWidth(940)
        self.thread: QThread | None = None
        self.worker: UpennGbmImportWorker | UpennGbmCatalogueWorker | None = None
        self._operation: str | None = None
        self._result: dict | None = None
        self._requested_role = "measurement"
        self._finish_action: str | None = None
        self._pending_failure: str | None = None
        self._pending_catalogue: UpennGbmCatalogue | None = None
        if cache_directory is None:
            cache_root = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.CacheLocation
            )
            if not cache_root:
                cache_root = str(Path.home() / ".cache" / "Field Workbench")
            cache_directory = Path(cache_root) / "upenn_gbm"
        self._cache_directory = str(cache_directory)
        self._catalogue = load_upenn_gbm_catalogue(self._cache_directory)

        root = QVBoxLayout(self)
        heading = QLabel("UPENN-GBM atlas case importer")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        self.intro = QLabel()
        self.intro.setWordWrap(True)
        self.intro.setOpenExternalLinks(True)
        self.intro.setObjectName("hint")
        root.addWidget(self.intro)

        warning = QLabel(
            "These masks are registered to the population SRI24 atlas and are placed through "
            "the same fixed SRI24→NAS/LPA/RPA transform as the built-in brain. They preserve "
            "case tumor size and atlas location, but are not patient-specific scalp geometry "
            "and are not intended for diagnosis or treatment planning."
        )
        warning.setWordWrap(True)
        warning.setObjectName("warningPanel")
        root.addWidget(warning)

        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Filter case id, asset, or status; for example 002 or compatible"
        )
        self.search.textChanged.connect(self._filter_cases)
        search_row = QHBoxLayout()
        search_row.addWidget(self.search, 1)
        self.refresh_button = QPushButton("Refresh catalogue")
        self.refresh_button.setToolTip(
            "Discover the current release masks, validate their SRI24 compatibility, "
            "and rebuild the cached catalogue"
        )
        self.refresh_button.clicked.connect(self._start_refresh)
        search_row.addWidget(self.refresh_button)
        root.addLayout(search_row)

        self.catalogue_status = QLabel()
        self.catalogue_status.setObjectName("hint")
        root.addWidget(self.catalogue_status)

        self.cases = QTreeWidget()
        self.cases.setHeaderLabels(
            [
                "Case",
                "Approximate location",
                "Tumor core (mL)",
                "Whole tumor (mL)",
                "Extent X × Y × Z (mm)",
                "Status",
            ]
        )
        self.cases.setRootIsDecorated(False)
        self.cases.setAlternatingRowColors(True)
        self.cases.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._populate_cases(self._catalogue)
        header = self.cases.header()
        for column in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.cases.setSortingEnabled(True)
        self.cases.sortItems(0, Qt.SortOrder.AscendingOrder)
        root.addWidget(self.cases, 1)

        options = QGroupBox("Import options")
        options_layout = QVBoxLayout(options)
        self.region_checks: dict[str, QCheckBox] = {}
        for region in UPENN_GBM_REGIONS:
            check = QCheckBox(region.label)
            check.setChecked(region.key == "tumor_core")
            check.setStyleSheet(f"QCheckBox {{ color: {region.colour}; font-weight: 600; }}")
            self.region_checks[region.key] = check
            options_layout.addWidget(check)
        overlap_note = QLabel(
            "Tumor core and whole tumor are derived unions. Selecting them together with their "
            "component regions deliberately creates overlapping objects."
        )
        overlap_note.setWordWrap(True)
        overlap_note.setObjectName("hint")
        options_layout.addWidget(overlap_note)
        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("Scene role"))
        self.role = QComboBox()
        self.role.addItem("Measurement volume", "measurement")
        self.role.addItem("Visual only", "visual")
        self.role.addItem("Exclusion zone", "exclusion")
        role_row.addWidget(self.role, 1)
        options_layout.addLayout(role_row)
        root.addWidget(options)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        root.addWidget(self.progress_bar)

        self.buttons = QDialogButtonBox()
        self.import_button = self.buttons.addButton(
            "Download && add to scene", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.import_button.setObjectName("primaryButton")
        self.cancel_button = self.buttons.addButton(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._start_import)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

    def _filter_cases(self, text: str) -> None:
        query = str(text).strip().casefold()
        first_visible = None
        for index in range(self.cases.topLevelItemCount()):
            item = self.cases.topLevelItem(index)
            case = item.data(0, Qt.ItemDataRole.UserRole)
            searchable = " ".join(
                [item.text(column) for column in range(self.cases.columnCount())]
                + (
                    [case.filename, case.compatibility_detail]
                    if isinstance(case, UpennGbmCase)
                    else []
                )
            ).casefold()
            visible = not query or query in searchable
            item.setHidden(not visible)
            if visible and first_visible is None and bool(
                item.flags() & Qt.ItemFlag.ItemIsSelectable
            ):
                first_visible = item
        current = self.cases.currentItem()
        if first_visible is not None and (current is None or current.isHidden()):
            self.cases.setCurrentItem(first_visible)

    def _populate_cases(self, catalogue: UpennGbmCatalogue) -> None:
        selected_number = None
        current = self.cases.currentItem()
        if current is not None:
            current_case = current.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(current_case, UpennGbmCase):
                selected_number = current_case.subject_number
        self.cases.clear()
        first_importable = None
        selected_item = None
        for case in catalogue.cases:
            core_volume = (
                f"{case.tumor_core_volume_ml:.1f}"
                if case.tumor_core_volume_ml is not None
                else "—"
            )
            whole_volume = (
                f"{case.whole_tumor_volume_ml:.1f}"
                if case.whole_tumor_volume_ml is not None
                else "—"
            )
            extent = (
                " × ".join(f"{value:.0f}" for value in case.extent_head_mm)
                if case.extent_head_mm is not None
                else "—"
            )
            item = _UpennCaseTreeItem(
                [
                    case.case_id,
                    case.approximate_location or "—",
                    core_volume,
                    whole_volume,
                    extent,
                    case.status_label,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, case)
            sort_values = (
                case.subject_number,
                case.approximate_location or "",
                case.tumor_core_volume_ml,
                case.whole_tumor_volume_ml,
                max(case.extent_head_mm) if case.extent_head_mm is not None else None,
                case.status_label,
            )
            for column, value in enumerate(sort_values):
                item.setData(column, _UpennCaseTreeItem.SORT_ROLE, value)
            for column in (2, 3):
                item.setTextAlignment(
                    column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            for column in (4, 5):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignCenter)
            details = self._case_tooltip(case)
            for column in range(6):
                item.setToolTip(column, details)
            if not case.is_importable:
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsSelectable
                    & ~Qt.ItemFlag.ItemIsEnabled
                )
                for column in range(6):
                    item.setForeground(column, QBrush(QColor("#9ca3af")))
            elif first_importable is None:
                first_importable = item
            if case.subject_number == selected_number and case.is_importable:
                selected_item = item
            self.cases.addTopLevelItem(item)
        if selected_item is not None:
            self.cases.setCurrentItem(selected_item)
        elif first_importable is not None:
            self.cases.setCurrentItem(first_importable)
        self._catalogue = catalogue
        count = len(catalogue.cases)
        self.intro.setText(
            f"Browse {count} cases with per-case tumor masks, download one segmentation "
            "on demand, and convert selected regions into closed scene meshes. "
            f'<a href="{UPENN_GBM_COLLECTION_URL}">Collection and citation</a>'
        )
        if catalogue.source == "bundled":
            summary = f"Bundled offline snapshot • {count} cases • compatibility checked on import"
        else:
            updated = catalogue.refreshed_at or "unknown time"
            summary = (
                f"Updated {updated} • {catalogue.compatible_count} compatible • "
                f"{catalogue.incompatible_count} incompatible • "
                f"{catalogue.unverified_count} unverified"
            )
        self.catalogue_status.setText(summary)
        self._filter_cases(self.search.text())

    @staticmethod
    def _case_tooltip(case: UpennGbmCase) -> str:
        lines = [case.case_id, f"Asset: {case.filename}"]
        if case.has_case_statistics:
            centroid = case.centroid_head_mm
            extent = case.extent_head_mm
            lines.extend(
                [
                    f"Approximate location: {case.approximate_location}",
                    (
                        "Centroid in canonical head frame: "
                        f"X {centroid[0]:.1f}, Y {centroid[1]:.1f}, "
                        f"Z {centroid[2]:.1f} mm"
                    ),
                    (
                        f"Whole-tumor distribution: {case.left_fraction * 100:.1f}% left, "
                        f"{case.right_fraction * 100:.1f}% right"
                    ),
                    f"Tumor core: {case.tumor_core_volume_ml:.3f} mL",
                    f"Whole tumor: {case.whole_tumor_volume_ml:.3f} mL",
                    (
                        "Head-axis extent: "
                        f"{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm"
                    ),
                ]
            )
        else:
            lines.append("Case measurements are calculated by Refresh catalogue.")
        lines.extend(
            [
                f"Status: {case.status_label}",
                case.compatibility_detail,
            ]
        )
        return "\n".join(lines)

    def selected_region_keys(self) -> list[str]:
        return [
            region.key
            for region in UPENN_GBM_REGIONS
            if self.region_checks[region.key].isChecked()
        ]

    def requested_role(self) -> str:
        return str(self._requested_role)

    def import_result(self) -> dict:
        if self._result is None:
            raise RuntimeError("The UPENN-GBM import has not completed.")
        return self._result

    def _set_running(self, running: bool) -> None:
        self.search.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.cases.setEnabled(not running)
        self.role.setEnabled(not running)
        for check in self.region_checks.values():
            check.setEnabled(not running)
        self.import_button.setEnabled(not running)
        if running:
            self.cancel_button.setText(
                "Cancel refresh" if self._operation == "refresh" else "Cancel import"
            )
        else:
            self.cancel_button.setText("Cancel")
            self.cancel_button.setEnabled(True)

    def _start_refresh(self) -> None:
        if self.thread is not None:
            return
        self._operation = "refresh"
        self._finish_action = None
        self._pending_failure = None
        self._pending_catalogue = None
        self._set_running(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Checking the release catalogue…")

        self.thread = QThread(self)
        self.worker = UpennGbmCatalogueWorker(self._cache_directory)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._catalogue_completed)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _start_import(self) -> None:
        if self.thread is not None:
            return
        item = self.cases.currentItem()
        if item is None or item.isHidden():
            QMessageBox.information(self, "UPENN-GBM case", "Select a case to import.")
            return
        regions = self.selected_region_keys()
        if not regions:
            QMessageBox.information(
                self, "UPENN-GBM regions", "Select at least one tumor region."
            )
            return
        case = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(case, UpennGbmCase) or not case.is_importable:
            QMessageBox.information(
                self,
                "UPENN-GBM case",
                "Select a compatible or unverified case to import.",
            )
            return
        case_id = case.case_id
        self._requested_role = str(self.role.currentData())
        self._operation = "import"
        self._finish_action = None
        self._pending_failure = None
        self._set_running(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Preparing {case_id}…")

        self.thread = QThread(self)
        self.worker = UpennGbmImportWorker(case, regions, self._cache_directory)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(dict)
    def _progress(self, update: dict) -> None:
        phase = str(update.get("phase", ""))
        percent = max(0, min(100, int(update.get("percent", 0))))
        if phase == "catalogue":
            overall = percent
        elif phase == "download":
            overall = round(percent * 0.2)
        else:
            region_count = max(1, int(update.get("region_count", 1)))
            region_index = max(0, int(update.get("region_index", 0)))
            overall = round(20 + 80 * (region_index + percent / 100.0) / region_count)
        self.progress_bar.setValue(max(0, min(100, overall)))
        self.progress_bar.setFormat(str(update.get("message", "Importing…")))

    @Slot(object)
    def _completed(self, result: object) -> None:
        self._result = dict(result) if isinstance(result, dict) else None
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Case ready — adding to scene…")
        self._finish_action = "accept"

    @Slot(object)
    def _catalogue_completed(self, result: object) -> None:
        if not isinstance(result, UpennGbmCatalogue):
            self._pending_failure = "The refreshed catalogue result is malformed."
            return
        self._pending_catalogue = result
        self._finish_action = "refresh"


    @Slot(str)
    def _failed(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(
            "Catalogue refresh failed" if self._operation == "refresh" else "Import failed"
        )
        self._pending_failure = str(message)

    @Slot()
    def _cancelled(self) -> None:
        self.progress_bar.setValue(0)
        if self._operation == "refresh":
            self.progress_bar.setFormat("Catalogue refresh cancelled")
            self._finish_action = "refresh_cancelled"
        else:
            self.progress_bar.setFormat("Import cancelled")
            self._finish_action = "reject"

    @Slot()
    def _thread_finished(self) -> None:
        operation = self._operation
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self._operation = None
        self._set_running(False)
        if self._finish_action == "accept" and self._result is not None:
            super().accept()
        elif self._finish_action == "reject":
            super().reject()
        elif self._finish_action == "refresh" and self._pending_catalogue is not None:
            catalogue = self._pending_catalogue
            self._pending_catalogue = None
            self._populate_cases(catalogue)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(
                f"Catalogue ready — {len(catalogue.cases)} cases found"
            )
        elif self._finish_action == "refresh_cancelled":
            self._finish_action = None
        elif self._pending_failure:
            message = self._pending_failure
            self._pending_failure = None
            QMessageBox.warning(
                self,
                "UPENN-GBM catalogue" if operation == "refresh" else "UPENN-GBM import",
                message,
            )

    def reject(self) -> None:
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.progress_bar.setFormat("Cancelling safely…")
            self.cancel_button.setEnabled(False)
            self._finish_action = (
                "refresh_cancelled" if self._operation == "refresh" else "reject"
            )
            return
        super().reject()


def _update_field_component_combo_labels(combo: QComboBox, field_name: str) -> None:
    """Keep B/H component formulas explicit without changing stored item data."""
    symbol = str(field_name).strip().upper() or "B"
    labels = {
        "magnitude": f"Magnitude |{symbol}|",
        "abs_x": f"X magnitude |{symbol}x|",
        "abs_y": f"Y magnitude |{symbol}y|",
        "abs_z": f"Z magnitude |{symbol}z|",
        "x": f"X signed {symbol}x",
        "y": f"Y signed {symbol}y",
        "z": f"Z signed {symbol}z",
    }
    for data, label in labels.items():
        index = combo.findData(data)
        if index >= 0:
            combo.setItemText(index, label)


def _spin(
    value=0.0,
    minimum=-100000.0,
    maximum=100000.0,
    decimals=3,
    suffix="",
    *,
    spinbox_type=CompactDoubleSpinBox,
):
    widget = spinbox_type()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    widget.setKeyboardTracking(False)
    return widget


def _format_elapsed_time(seconds: float) -> str:
    """Format a field-map duration compactly for progress and status displays."""
    elapsed = max(0.0, float(seconds))
    if elapsed < 60.0:
        return f"{elapsed:.1f} s"
    whole_seconds = int(elapsed)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _field_map_elapsed(progress: QProgressDialog | None) -> float:
    if progress is None:
        return 0.0
    try:
        started = float(getattr(progress, "_field_map_started_at"))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, time.perf_counter() - started)


def _coil_solver_summary(report: dict | None) -> str:
    """Summarize the actual per-coil representation used by the last field call."""
    report = report if isinstance(report, dict) else {}
    requested = str(report.get("requested", "centreline")).strip().lower()
    requested_label = COIL_MODELING_LABELS.get(requested, requested.title())
    coils = report.get("coils", [])
    actual_values = [
        str(item.get("actual", requested)).strip().lower()
        for item in coils
        if isinstance(item, dict)
    ]
    if not actual_values:
        return requested_label
    def empirical_tolerance_warning(item: dict) -> bool:
        auto = item.get("auto")
        if not isinstance(auto, dict):
            return False
        validation = auto.get("validation")
        return bool(
            isinstance(validation, dict)
            and validation.get("status") == "complete"
            and auto.get("empirical_tolerance_met") is False
        )

    above_tolerance = sum(
        empirical_tolerance_warning(item) for item in coils if isinstance(item, dict)
    )
    tolerance_suffix = ""
    if requested == "auto" and above_tolerance:
        noun = "coil" if above_tolerance == 1 else "coils"
        tolerance_suffix = f"; {above_tolerance} {noun} above empirical tolerance"
    counts = Counter(actual_values)
    if len(counts) == 1:
        actual = next(iter(counts))
        actual_label = COIL_MODELING_LABELS.get(actual, actual.title())
        if requested == "auto":
            fallback_count = sum(
                bool(item.get("fallback_reason"))
                for item in coils
                if isinstance(item, dict)
            )
            suffix = f"; {fallback_count} fallback" if fallback_count else ""
            if fallback_count != 1 and fallback_count:
                suffix += "s"
            return f"Auto → {actual_label}{suffix}{tolerance_suffix}"
        if actual == requested:
            return actual_label
        return f"{actual_label} (fallback from {requested_label})"
    fallback_count = counts.get("centreline", 0) if requested != "centreline" else 0
    if fallback_count and counts.get(requested, 0) + fallback_count == len(actual_values):
        noun = "coil" if len(actual_values) == 1 else "coils"
        return (
            f"{requested_label} with Centreline fallback "
            f"({fallback_count}/{len(actual_values)} {noun})"
        )
    parts = [
        f"{COIL_MODELING_LABELS.get(method, method.title())} ×{count}"
        for method, count in sorted(counts.items())
    ]
    if requested == "auto":
        fallback_count = sum(
            bool(item.get("fallback_reason"))
            for item in coils
            if isinstance(item, dict)
        )
        suffix = f"; {fallback_count} fallback" if fallback_count else ""
        if fallback_count != 1 and fallback_count:
            suffix += "s"
        return "Auto: " + ", ".join(parts) + suffix + tolerance_suffix
    return "Mixed: " + ", ".join(parts)


def _field_map_status_text(adapter: StudioAdapter, elapsed_seconds: float) -> str:
    return (
        f"Solver: {_coil_solver_summary(adapter.coil_modeling_report())}  •  "
        f"Total elapsed: {_format_elapsed_time(elapsed_seconds)}"
    )


class _FieldMapProgressDialog(QProgressDialog):
    """Modal progress window whose close action requests safe batch cancellation."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._field_map_started_at = time.perf_counter()
        self._field_map_cancel_requested = False
        self._field_map_allow_close = False

    def cancellation_requested(self) -> bool:
        return bool(self._field_map_cancel_requested)

    def request_cancellation(self) -> None:
        if self._field_map_cancel_requested or self._field_map_allow_close:
            return
        self._field_map_cancel_requested = True
        self.setWindowTitle("Cancelling field map")
        self.setLabelText(
            "Cancelling after the current solver batch finishes…\n"
            f"Elapsed: {_format_elapsed_time(_field_map_elapsed(self))}"
        )
        bar = self.findChild(QProgressBar)
        if bar is not None:
            bar.setFormat("Cancelling after current batch…")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._field_map_allow_close:
            super().closeEvent(event)
            return
        self.request_cancellation()
        event.ignore()

    def reject(self) -> None:
        if self._field_map_allow_close:
            super().reject()
            return
        self.request_cancellation()

    def finish_and_close(self) -> None:
        self._field_map_allow_close = True
        self.close()


def _centre_field_map_progress(progress: QWidget, parent: QWidget) -> None:
    """Place a progress dialog over its map window, clamped to the active screen."""
    centre_window_on_parent(progress, parent)


def _raise_if_field_map_cancelled(
    progress: _FieldMapProgressDialog | None,
) -> None:
    if progress is not None and progress.cancellation_requested():
        raise FieldCalculationCancelled(
            "The field-map calculation was cancelled by the user."
        )


def _show_field_map_progress(
    parent: QWidget, message: str
) -> _FieldMapProgressDialog:
    """Show a centered determinate field-map progress window."""
    progress = _FieldMapProgressDialog(parent)
    progress.setObjectName("fieldMapCalculationProgress")
    progress.setWindowTitle("Calculating field map")
    progress.setLabelText(message)
    progress.setCancelButton(None)
    progress.setRange(0, 1)
    progress.setValue(0)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumWidth(470)
    bar = progress.findChild(QProgressBar)
    if bar is not None:
        bar.setFormat("Preparing field samples…  •  elapsed 0.0 s")
    progress.show()
    # The solver runs synchronously in bounded batches. Flush one event pass so
    # the dialog has its final size, then explicitly centre it over either map
    # window before the first Magpylib batch begins.
    QApplication.processEvents()
    _centre_field_map_progress(progress, parent)
    progress.raise_()
    progress.activateWindow()
    QApplication.processEvents()
    return progress


def _update_field_map_progress(
    progress: _FieldMapProgressDialog | None,
    update: dict,
) -> None:
    """Apply one solver progress update and keep the synchronous UI responsive."""
    if progress is None:
        return
    _raise_if_field_map_cancelled(progress)
    try:
        total = max(1, int(update.get("total", 1)))
        completed = min(total, max(0, int(update.get("completed", 0))))
    except (TypeError, ValueError):
        total, completed = 1, 0
    stage = str(update.get("stage", "Calculating field map…"))
    unit = str(update.get("unit", "field samples"))
    detail = str(update.get("detail", "")).strip()
    memory_bits: list[str] = []
    try:
        rss_mb = float(update.get("rss_mb"))
        if math.isfinite(rss_mb):
            memory_bits.append(f"process RSS {rss_mb:,.0f} MB")
    except (TypeError, ValueError):
        pass
    try:
        budget_mb = int(update.get("memory_budget_mb"))
        if budget_mb > 0:
            memory_bits.append(f"kernel budget {budget_mb:,} MB")
    except (TypeError, ValueError):
        pass
    label_lines = [stage]
    if detail:
        label_lines.append(detail)
    if memory_bits:
        label_lines.append(" • ".join(memory_bits))
    elapsed_text = _format_elapsed_time(_field_map_elapsed(progress))
    label_lines.append(f"Elapsed: {elapsed_text}")
    progress.setRange(0, total)
    progress.setValue(completed)
    progress.setLabelText("\n".join(label_lines))
    bar = progress.findChild(QProgressBar)
    if bar is not None:
        percentage = int(round(100.0 * completed / total))
        bar.setFormat(
            f"{completed:,} / {total:,} {unit} completed — {percentage}%  •  "
            f"elapsed {elapsed_text}"
        )
    QApplication.processEvents()
    _raise_if_field_map_cancelled(progress)


def _set_field_map_rendering(
    progress: _FieldMapProgressDialog | None,
    map_name: str,
) -> None:
    if progress is None:
        return
    _raise_if_field_map_cancelled(progress)
    elapsed_text = _format_elapsed_time(_field_map_elapsed(progress))
    progress.setValue(progress.maximum())
    progress.setLabelText(f"Rendering {map_name}…\nElapsed: {elapsed_text}")
    progress_bar = progress.findChild(QProgressBar)
    if progress_bar is not None:
        progress_bar.setFormat(
            f"Field sampling complete — rendering…  •  elapsed {elapsed_text}"
        )
    QApplication.processEvents()
    _raise_if_field_map_cancelled(progress)


def _finish_field_map_progress(progress: _FieldMapProgressDialog | None) -> None:
    if progress is None:
        return
    progress.finish_and_close()
    progress.deleteLater()
    QApplication.processEvents()


_ACTIVE_FIELD_RENDER_THREADS: set[QThread] = set()
_FIELD_RENDER_SHUTDOWN_CONNECTED = False
# Starting a spawned process briefly serializes a large immutable scene into the
# child. Keep that handshake one-at-a-time while allowing the native calculations
# themselves to overlap according to the user's worker preference.
_FIELD_RENDER_PROCESS_START_LOCK = threading.Lock()
_FIELD_RENDER_CANCEL_GRACE_SECONDS = 1.0
# Give Qt WebEngine one event-loop turn plus a short renderer-shutdown window
# after hidden result pages are discarded, before the next large native solve
# starts allocating its field arrays.
_FIELD_RENDER_BROWSER_RELEASE_DELAY_MS = 125


class _FieldRenderConcurrencyLimiter:
    """FIFO, dynamically configurable process slots shared by all map windows."""

    def __init__(self, max_workers: int = 1):
        self._condition = threading.Condition()
        self._max_workers = normalize_optimizer_worker_count(max_workers)
        self._active = 0
        self._waiting: list[object] = []

    def configure(self, max_workers: int) -> int:
        """Apply a new admission cap without interrupting active calculations."""

        normalized = normalize_optimizer_worker_count(max_workers)
        with self._condition:
            self._max_workers = normalized
            self._condition.notify_all()
        return normalized

    def snapshot(self) -> tuple[int, int, int]:
        """Return active count, queued count, and the current limit."""

        with self._condition:
            return self._active, len(self._waiting), self._max_workers

    def acquire(
        self,
        cancel_event: threading.Event,
        wait_callback: Callable[[int, int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Wait cooperatively for the next FIFO slot and return active/limit."""

        token = object()
        last_status: tuple[int, int, int] | None = None
        with self._condition:
            if cancel_event.is_set():
                raise FieldCalculationCancelled("Calculation cancelled while queued.")
            self._waiting.append(token)
            while True:
                if cancel_event.is_set():
                    self._waiting.remove(token)
                    self._condition.notify_all()
                    raise FieldCalculationCancelled(
                        "Calculation cancelled while queued."
                    )
                if (
                    self._waiting
                    and self._waiting[0] is token
                    and self._active < self._max_workers
                ):
                    self._waiting.pop(0)
                    self._active += 1
                    self._condition.notify_all()
                    return self._active, self._max_workers

                status = (
                    self._active,
                    self._max_workers,
                    self._waiting.index(token) + 1,
                )
                if wait_callback is not None and status != last_status:
                    wait_callback(*status)
                    last_status = status
                self._condition.wait(timeout=0.1)

    def release(self) -> None:
        """Release one admitted slot and wake the next queued job."""

        with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify_all()


_FIELD_RENDER_CONCURRENCY = _FieldRenderConcurrencyLimiter()


def configure_field_render_worker_limit(max_workers: int) -> int:
    """Set the global map-render process cap used by every map window."""

    return _FIELD_RENDER_CONCURRENCY.configure(max_workers)


def _field_render_process_exit_message(
    exit_code: int | None,
    *,
    last_stage: str | None = None,
    crash_log_path: str | None = None,
    diagnostic_path: str | None = None,
) -> str:
    """Explain an abnormal isolated solver exit without losing diagnostics."""

    code = int(exit_code) if exit_code is not None else None
    reason = f"exit code {code}" if code is not None else "an unknown exit status"
    if code is not None and code < 0:
        try:
            signal_name = signal.Signals(-code).name
        except (ValueError, OSError):
            signal_name = None
        if signal_name == "SIGSEGV":
            reason = "SIGSEGV (segmentation fault)"
        elif signal_name == "SIGKILL":
            reason = "SIGKILL (often operating-system memory pressure)"
        elif signal_name == "SIGABRT":
            reason = "SIGABRT (native library abort)"
        elif signal_name:
            reason = signal_name
    if code is not None:
        windows_status = code & 0xFFFFFFFF
        windows_reasons = {
            0xC0000005: "Windows access violation (0xC0000005)",
            0xC0000017: "Windows out-of-memory status (0xC0000017)",
            0xC0000409: "Windows stack-buffer overrun (0xC0000409)",
        }
        reason = windows_reasons.get(windows_status, reason)

    lines = [
        f"The isolated field-solver process crashed with {reason}.",
        (
            "Field Workbench stayed open and this render was stopped before its "
            "native failure could terminate the GUI."
        ),
    ]
    if last_stage:
        lines.append(f"Last reported stage: {last_stage}")
    if crash_log_path:
        lines.append(f"Crash trace: {crash_log_path}")
    if diagnostic_path:
        lines.append(f"Solver diagnostics: {diagnostic_path}")
    lines.append(
        "Retrying is safe; if the same stage fails repeatedly, reduce model detail "
        "or map resolution and keep the crash trace for diagnosis."
    )
    return "\n".join(lines)


def _field_render_staging_directory(result: dict[str, Any]) -> Path | None:
    """Return a validated render staging directory below the system temp root."""

    raw_directory = str(result.get("work_directory") or "").strip()
    if not raw_directory:
        return None
    try:
        directory = Path(raw_directory).resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
    except (OSError, RuntimeError):
        return None
    if (
        directory.parent != temporary_root
        or not directory.name.startswith("field-workbench-render-")
    ):
        return None
    return directory


def _cleanup_field_render_staging(result: dict[str, Any]) -> None:
    """Remove only a render-job directory created beneath the system temp root."""

    directory = _field_render_staging_directory(result)
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


def _load_field_render_result(result: dict[str, Any]) -> dict[str, Any]:
    """Load a trusted isolated-process result, then release its staging files."""

    directory = _field_render_staging_directory(result)
    if directory is None:
        raise StudioOperationError("The render worker returned an invalid staging path.")
    try:
        raw_result_path = str(result.get("result_path") or "").strip()
        if not raw_result_path:
            raise StudioOperationError("The render worker returned no result path.")
        result_path = Path(raw_result_path).resolve()
        expected_path = directory / "render-result.pickle"
        if result_path != expected_path:
            raise StudioOperationError("The render worker returned an invalid result path.")
        with result_path.open("rb") as stream:
            isolated_result = pickle.load(stream)
        if not isinstance(isolated_result, dict):
            raise StudioOperationError("The render worker returned a malformed result.")
        return isolated_result
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _stop_field_render_threads_for_shutdown() -> None:
    """Cancel and join render workers before Qt tears their thread objects down."""

    threads = list(_ACTIVE_FIELD_RENDER_THREADS)
    for thread in threads:
        worker = getattr(thread, "_field_render_worker", None)
        if isinstance(worker, _FieldRenderWorker):
            worker.request_cancellation()
        # quit() is thread-safe and takes effect as soon as a running worker
        # returns to its QThread event loop.
        thread.quit()
    for thread in threads:
        if thread.isRunning():
            thread.wait()


def _ensure_field_render_shutdown_hook() -> None:
    global _FIELD_RENDER_SHUTDOWN_CONNECTED
    if _FIELD_RENDER_SHUTDOWN_CONNECTED:
        return
    application = QApplication.instance()
    if application is None:
        return
    application.aboutToQuit.connect(_stop_field_render_threads_for_shutdown)
    _FIELD_RENDER_SHUTDOWN_CONNECTED = True


class _FieldRenderJobPage(QWidget):
    """In-tab execution plan for one non-blocking map-render job."""

    action_requested = Signal()

    _TASK_ICONS = {
        "pending": "○",
        "active": "▶",
        "complete": "✓",
        "skipped": "–",
        "cancelled": "×",
        "failed": "!",
    }

    def __init__(
        self,
        job_label: str,
        parent=None,
        *,
        render_kind: str = "static",
        request: dict[str, Any] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("fieldRenderJobPage")
        self.job_label = str(job_label)
        self._started_at = time.perf_counter()
        self._terminal = False
        self._cancel_requested = False
        self._active_task_id: str | None = None
        self._visible_progress_fraction = 0.0
        self._render_kind = str(render_kind).strip().lower()
        self._viewer_name = "Plotly" if self._render_kind == "2d_static" else "WebGL"
        self._last_stage = "Waiting for a render slot…"
        self._task_plan = self._build_task_plan(render_kind, request or {})
        self._task_by_id = {
            str(task["id"]): task for task in self._task_plan
        }
        self._task_items: dict[str, QTreeWidgetItem] = {}

        root = QVBoxLayout(self)
        # Render-job tabs are already their own dedicated work surface.  Keep
        # the progress UI essentially edge-to-edge inside the tab instead of
        # nesting it in a heavily inset dialog-style card.
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        card = QGroupBox()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # The global QGroupBox theme reserves title margin/rounded corners.
        # This group has no title and is the tab surface itself, so suppress
        # those outer-card affordances while retaining the theme background.
        card.setStyleSheet(
            "QGroupBox { margin-top: 0px; border-radius: 0px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        self.heading = QLabel(f"Calculating {self.job_label}")
        self.heading.setObjectName("dialogHeading")
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.heading)

        self.detail = QLabel(
            "Waiting for an available render slot.\n"
            "Task progress and elapsed time will update as processing advances."
        )
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        card_layout.addWidget(self.detail)

        self.task_list = QTreeWidget()
        self.task_list.setColumnCount(4)
        self.task_list.setHeaderLabels(["", "Render task", "Work", "Time"])
        self.task_list.setRootIsDecorated(False)
        self.task_list.setItemsExpandable(False)
        self.task_list.setUniformRowHeights(True)
        self.task_list.setAlternatingRowColors(True)
        self.task_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.task_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.task_list.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.task_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.task_list.setMinimumHeight(min(250, 54 + 27 * len(self._task_plan)))
        for task in self._task_plan:
            task_id = str(task["id"])
            item = QTreeWidgetItem(["○", str(task["label"]), "—", "…"])
            item.setData(0, Qt.ItemDataRole.UserRole, task_id)
            item.setTextAlignment(
                2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            item.setTextAlignment(
                3, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.task_list.addTopLevelItem(item)
            self._task_items[task_id] = item
        card_layout.addWidget(self.task_list, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% • Waiting for render slot")
        card_layout.addWidget(self.progress_bar)

        self.elapsed = QLabel("Elapsed: 0.0 s")
        self.elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed.setObjectName("hint")
        card_layout.addWidget(self.elapsed)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.action_button = QPushButton("Cancel calculation")
        self.action_button.clicked.connect(
            lambda *_args: self.action_requested.emit()
        )
        actions.addWidget(self.action_button)
        actions.addStretch(1)
        card_layout.addLayout(actions)

        root.addWidget(card, 1)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)
        self._elapsed_timer.start()
        self._activate_task("queue")

    @staticmethod
    def _new_task(task_id: str, label: str, weight: float) -> dict[str, Any]:
        return {
            "id": str(task_id),
            "label": str(label),
            "progress_weight": max(0.0, float(weight)),
            "status": "pending",
            "fraction": 0.0,
            "completed": None,
            "total": None,
            "unit": "",
            "stage": "",
            "started_at": None,
            "elapsed_accum": 0.0,
        }

    @classmethod
    def _build_task_plan(
        cls, render_kind: str, request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return a stable weighted plan whose measured parts use real counters."""

        kind = str(render_kind).strip().lower()
        map_settings = request.get("map_settings", {})
        if kind == "brain_static":
            specs = (
                ("queue", "Acquire render slot", 0.02),
                ("launch", "Start render worker", 0.03),
                ("grid", "Prepare LPBA40 samples", 0.05),
                ("field", "Calculate brain-region field", 0.72),
                ("assemble", "Build Brain View WebGL map", 0.13),
                ("viewer", "Open WebGL viewer", 0.05),
            )
            return [cls._new_task(*spec) for spec in specs]
        if kind == "2d_static":
            display_mode = str(
                map_settings.get("display_mode", "heatmap")
                if isinstance(map_settings, dict)
                else "heatmap"
            ).strip().lower()
            include_heatmap = display_mode in {"heatmap", "both"}
            include_flux = display_mode in {"flux", "both"}
            specs: list[tuple[str, str, float]] = [
                ("queue", "Acquire render slot", 0.02),
                ("launch", "Start render worker", 0.03),
                ("grid", "Prepare sample grid", 0.05),
            ]
            if include_heatmap and include_flux:
                specs.extend(
                    [
                        ("field", "Calculate heatmap field", 0.43),
                        ("flux", "Calculate fluxline field grid", 0.29),
                    ]
                )
            elif include_flux:
                specs.append(("flux", "Calculate fluxline field grid", 0.72))
            else:
                specs.append(("field", "Calculate heatmap field", 0.72))
            specs.extend(
                [
                    ("assemble", "Assemble Plotly map", 0.13),
                    ("viewer", "Open Plotly viewer", 0.05),
                ]
            )
            return [cls._new_task(*spec) for spec in specs]

        if kind == "waveform":
            brain_frames = bool(
                isinstance(map_settings, dict)
                and map_settings.get("precompute_brain_region_frames", False)
            )
            specs = (
                (
                    ("queue", "Acquire render slot", 0.02),
                    ("launch", "Start render worker", 0.03),
                    ("grid", "Prepare sample grid", 0.04),
                    ("field", "Calculate fixed field and coil bases", 0.57),
                    ("brain_metrics", "Precompute Brain Areas frame metrics", 0.18),
                    ("assemble", "Build playback and WebGL data", 0.11),
                    ("viewer", "Open WebGL viewer", 0.05),
                )
                if brain_frames
                else (
                    ("queue", "Acquire render slot", 0.02),
                    ("launch", "Start render worker", 0.03),
                    ("grid", "Prepare sample grid", 0.04),
                    ("field", "Calculate fixed field and coil bases", 0.72),
                    ("assemble", "Build playback and WebGL data", 0.14),
                    ("viewer", "Open WebGL viewer", 0.05),
                )
            )
            return [cls._new_task(*spec) for spec in specs]

        display_mode = str(
            map_settings.get("display_mode", "slices")
            if isinstance(map_settings, dict)
            else "slices"
        ).strip().lower()
        include_flux = display_mode in {"flux", "both"}
        specs: list[tuple[str, str, float]] = [
            ("queue", "Acquire render slot", 0.02),
            ("launch", "Start render worker", 0.03),
            ("grid", "Prepare sample grid", 0.04 if include_flux else 0.05),
            ("field", "Calculate map field", 0.43 if include_flux else 0.68),
        ]
        if include_flux:
            specs.append(("flux", "Calculate fluxline field grid", 0.28))
        specs.extend(
            [
                ("assemble", "Assemble WebGL scene", 0.15 if include_flux else 0.17),
                ("viewer", "Open WebGL viewer", 0.05),
            ]
        )
        return [cls._new_task(*spec) for spec in specs]

    @property
    def terminal(self) -> bool:
        return bool(self._terminal)

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self._started_at)

    def _refresh_elapsed(self) -> None:
        self.elapsed.setText(
            f"Elapsed: {_format_elapsed_time(self.elapsed_seconds())}"
        )
        self._refresh_task_list()

    def _task_elapsed_seconds(self, task: dict[str, Any]) -> float:
        elapsed = max(0.0, float(task.get("elapsed_accum", 0.0) or 0.0))
        started = task.get("started_at")
        if task.get("status") == "active" and started is not None:
            elapsed += max(0.0, time.perf_counter() - float(started))
        return elapsed

    def _finish_task_clock(self, task: dict[str, Any]) -> None:
        started = task.get("started_at")
        if started is not None:
            task["elapsed_accum"] = self._task_elapsed_seconds(task)
        task["started_at"] = None

    def _task_index(self, task_id: str) -> int:
        return next(
            (
                index
                for index, task in enumerate(self._task_plan)
                if str(task.get("id")) == str(task_id)
            ),
            -1,
        )

    def _activate_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._task_by_id.get(str(task_id))
        if task is None:
            return None
        target_index = self._task_index(str(task_id))
        active_index = self._task_index(self._active_task_id or "")
        if active_index > target_index >= 0:
            return task

        now = time.perf_counter()
        active = self._task_by_id.get(self._active_task_id or "")
        if active is not None and active is not task:
            self._finish_task_clock(active)
            if active.get("status") == "active":
                active["status"] = "complete"
                active["fraction"] = 1.0
                if active.get("total") is not None:
                    active["completed"] = active["total"]

        for earlier in self._task_plan[: max(0, target_index)]:
            if earlier.get("status") in {"pending", "active"}:
                self._finish_task_clock(earlier)
                earlier["status"] = "complete"
                earlier["fraction"] = 1.0
                if earlier.get("total") is not None:
                    earlier["completed"] = earlier["total"]

        if task.get("status") == "pending":
            task["status"] = "active"
            task["started_at"] = now
        elif task.get("status") == "active" and task.get("started_at") is None:
            task["started_at"] = now
        self._active_task_id = str(task_id) if task.get("status") == "active" else None
        self._refresh_task_list()
        self._refresh_determinate_progress_bar()
        return task

    def _complete_task(self, task_id: str) -> None:
        task = self._task_by_id.get(str(task_id))
        if task is None:
            return
        self._finish_task_clock(task)
        task["status"] = "complete"
        task["fraction"] = 1.0
        if task.get("total") is not None:
            task["completed"] = task["total"]
        if self._active_task_id == str(task_id):
            self._active_task_id = None
        self._refresh_task_list()
        self._refresh_determinate_progress_bar()

    def _task_for_stage(self, stage: str) -> str:
        lowered = str(stage).strip().lower()
        if "queued" in lowered or "waiting for a render slot" in lowered or "waiting for a 3d render slot" in lowered:
            return "queue"
        if "isolated" in lowered or "render slot admitted" in lowered:
            return "launch"
        if "fluxline" in lowered and "flux" in self._task_by_id:
            return "flux"
        if "sample grid" in lowered and "field" not in lowered:
            return "grid"
        if any(word in lowered for word in ("webgl", "playback data", "render package")):
            return "assemble"
        if any(word in lowered for word in ("field", "basis", "grid")):
            return "field"
        return self._active_task_id or "launch"

    @staticmethod
    def _short_unit(unit: str) -> str:
        normalized = str(unit).strip()
        return {
            "field samples": "samples",
            "basis work": "basis",
            "render steps": "steps",
            "render step": "steps",
            "render job": "job",
        }.get(normalized, normalized)

    def _task_work_text(self, task: dict[str, Any]) -> str:
        status = str(task.get("status", "pending"))
        completed = task.get("completed")
        total = task.get("total")
        if completed is not None and total is not None:
            try:
                current = max(0, int(completed))
                maximum = max(1, int(total))
                suffix = self._short_unit(str(task.get("unit") or ""))
                return f"{current:,}/{maximum:,}{' ' + suffix if suffix else ''}"
            except (TypeError, ValueError):
                pass
        if status == "complete":
            return "done"
        if status == "active":
            return "in progress"
        if status in {"skipped", "cancelled"}:
            return "—"
        if status == "failed":
            return "failed"
        return "…"

    def _task_time_text(self, task: dict[str, Any]) -> str:
        status = str(task.get("status", "pending"))
        if status in {"pending", "skipped"}:
            return "…" if status == "pending" else "—"
        return _format_elapsed_time(self._task_elapsed_seconds(task))

    def _refresh_task_list(self) -> None:
        if not hasattr(self, "task_list"):
            return
        for task in self._task_plan:
            task_id = str(task["id"])
            item = self._task_items.get(task_id)
            if item is None:
                continue
            status = str(task.get("status", "pending"))
            item.setText(0, self._TASK_ICONS.get(status, "○"))
            item.setText(1, str(task.get("label", "Render task")))
            item.setText(2, self._task_work_text(task))
            item.setText(3, self._task_time_text(task))
            item.setToolTip(1, str(task.get("stage") or task.get("label") or ""))
            font = item.font(1)
            font.setBold(status == "active")
            item.setFont(0, font)
            item.setFont(1, font)

    def _task_plan_progress_fraction(self) -> float:
        weights = [
            max(0.0, float(task.get("progress_weight", 0.0) or 0.0))
            for task in self._task_plan
        ]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            return 0.0
        progress = 0.0
        for task, weight in zip(self._task_plan, weights, strict=True):
            status = str(task.get("status", "pending"))
            if status in {"complete", "skipped"}:
                fraction = 1.0
            else:
                fraction = max(0.0, min(1.0, float(task.get("fraction", 0.0) or 0.0)))
            progress += weight * fraction
        return max(0.0, min(1.0, progress / weight_sum))

    def _refresh_determinate_progress_bar(self, *, force_complete: bool = False) -> None:
        raw_fraction = 1.0 if force_complete else self._task_plan_progress_fraction()
        if not force_complete:
            raw_fraction = min(raw_fraction, 0.99)
        self._visible_progress_fraction = max(
            self._visible_progress_fraction,
            max(0.0, min(1.0, raw_fraction)),
        )
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(
            int(round(1000.0 * self._visible_progress_fraction))
        )
        active = self._task_by_id.get(self._active_task_id or "")
        if force_complete:
            label = "Render ready"
        elif self._cancel_requested:
            label = "Cancelling safely"
        elif active is not None:
            label = str(active.get("label") or self._last_stage)
        else:
            label = self._last_stage
        self.progress_bar.setFormat(f"%p% • {label}")

    def update_progress(self, update: dict[str, Any]) -> None:
        if self._terminal:
            return
        try:
            total = max(1, int(update.get("total", 1)))
            completed = min(total, max(0, int(update.get("completed", 0))))
        except (TypeError, ValueError):
            total, completed = 1, 0
        stage = str(update.get("stage") or "Calculating field map…")
        self._last_stage = stage
        task_id = str(
            update.get("render_task") or self._task_for_stage(stage)
        )
        task = self._activate_task(task_id)
        if task is not None:
            previous_completed = task.get("completed")
            previous_total = task.get("total")
            if previous_total == total and previous_completed is not None:
                task["completed"] = max(int(previous_completed), completed)
            else:
                task["completed"] = completed
            task["total"] = total
            task["unit"] = str(update.get("unit") or task.get("unit") or "")
            task["stage"] = stage
            fraction = max(0.0, min(1.0, completed / total))
            task["fraction"] = max(float(task.get("fraction", 0.0) or 0.0), fraction)
            if bool(update.get("render_task_complete")):
                self._complete_task(task_id)
        detail = str(update.get("detail") or "").strip()
        memory_bits: list[str] = []
        try:
            rss_mb = float(update.get("rss_mb"))
            if math.isfinite(rss_mb):
                memory_bits.append(f"process RSS {rss_mb:,.0f} MB")
        except (TypeError, ValueError):
            pass
        try:
            budget_mb = int(update.get("memory_budget_mb"))
            if budget_mb > 0:
                memory_bits.append(f"kernel budget {budget_mb:,} MB")
        except (TypeError, ValueError):
            pass
        lines = [stage]
        if detail:
            lines.append(detail)
        if memory_bits:
            lines.append(" • ".join(memory_bits))
        self.detail.setText("\n".join(lines))
        self._refresh_task_list()
        self._refresh_determinate_progress_bar()
        self._refresh_elapsed()

    def set_rendering(self) -> None:
        if self._terminal:
            return
        self.heading.setText(f"Opening {self.job_label}")
        self.detail.setText(
            f"Field sampling complete — creating the {self._viewer_name} viewer…"
        )
        self._last_stage = f"Creating the {self._viewer_name} viewer…"
        viewer = self._activate_task("viewer")
        if viewer is not None:
            viewer["fraction"] = max(float(viewer.get("fraction", 0.0) or 0.0), 0.15)
            viewer["stage"] = self._last_stage
        self._refresh_task_list()
        self._refresh_determinate_progress_bar()
        self._refresh_elapsed()

    def set_ready(self) -> None:
        """Resolve the final plan sliver once the WebGL widget exists."""

        for task in self._task_plan:
            self._finish_task_clock(task)
            task["status"] = "complete"
            task["fraction"] = 1.0
            if task.get("total") is not None:
                task["completed"] = task["total"]
        self._active_task_id = None
        self._last_stage = "Render ready"
        self._refresh_task_list()
        self._refresh_determinate_progress_bar(force_complete=True)

    def set_cancelling(self) -> None:
        if self._terminal:
            return
        self._cancel_requested = True
        self.heading.setText(f"Cancelling {self.job_label}")
        self.detail.setText("Stopping the render calculation…")
        self._last_stage = "Stopping render worker…"
        self._refresh_determinate_progress_bar()
        self.action_button.setEnabled(False)
        self._refresh_elapsed()

    def _set_terminal(
        self,
        heading: str,
        detail: str,
        bar_text: str,
        terminal_status: str,
    ) -> None:
        self._terminal = True
        self._elapsed_timer.stop()
        active = self._task_by_id.get(self._active_task_id or "")
        if active is not None:
            self._finish_task_clock(active)
            active["status"] = str(terminal_status)
        for task in self._task_plan:
            if task.get("status") == "pending":
                task["status"] = "skipped"
        self._active_task_id = None
        self.heading.setText(heading)
        self.detail.setText(detail)
        self.progress_bar.setFormat(bar_text)
        self.action_button.setText("Close tab")
        self.action_button.setEnabled(True)
        self._refresh_task_list()
        self._refresh_elapsed()

    def set_cancelled(self) -> None:
        self._set_terminal(
            f"{self.job_label} cancelled",
            "No result was added. The settings in Selector are unchanged.",
            "Cancelled",
            "cancelled",
        )

    def set_failed(self, message: str) -> None:
        self._set_terminal(
            f"{self.job_label} failed",
            str(message),
            "Calculation failed",
            "failed",
        )


class _FieldRenderWorker(QObject):
    """Supervise one crash-isolated map-render process from a lightweight QThread."""

    progress = Signal(int, dict)
    completed = Signal(int, dict)
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(
        self,
        *,
        job_id: int,
        document: dict[str, Any],
        preferences: dict[str, Any],
        render_kind: str,
        request: dict[str, Any],
    ):
        super().__init__()
        self.job_id = int(job_id)
        # The owning 3D dialog shares one immutable scene snapshot among queued
        # jobs. StudioAdapter deep-copies it only when this job reaches the solver.
        self.document = document
        self.preferences = copy.deepcopy(preferences)
        self.render_kind = str(render_kind)
        self.request = copy.deepcopy(request)
        self.cancel_event = threading.Event()
        self._process_cancel_connection: Any | None = None
        self._process_cancel_lock = threading.Lock()
        self._process_cancel_sent = False
        self._process: Any | None = None

    def request_cancellation(self) -> None:
        self.cancel_event.set()
        self._signal_process_cancellation()

    def _signal_process_cancellation(self) -> None:
        """Send one non-blocking cancellation byte to the isolated child."""

        with self._process_cancel_lock:
            connection = self._process_cancel_connection
            if connection is None or self._process_cancel_sent:
                return
            self._process_cancel_sent = True
            try:
                connection.send_bytes(b"\x01")
            except (BrokenPipeError, EOFError, OSError, ValueError):
                pass

    def _render_dimension_label(self) -> str:
        if self.render_kind == "2d_static":
            return "2D"
        if self.render_kind == "waveform":
            map_kind = str(self.request.get("map_kind", "3d")).strip().lower()
            return "2D" if map_kind == "2d" else "3D"
        return "3D"

    def _acquire_solver(self) -> tuple[int, int]:
        """Wait cooperatively for a Preferences-limited global render slot."""

        def report_waiting(active: int, limit: int, position: int) -> None:
            self.progress.emit(
                self.job_id,
                {
                    "stage": "Queued — waiting for a render slot…",
                    "render_task": "queue",
                    "detail": (
                        f"{active} of {limit} allowed render calculations are active. "
                        f"FIFO queue position: {position}."
                    ),
                    "completed": 0,
                    "total": 1,
                    "unit": "render job",
                },
            )

        return _FIELD_RENDER_CONCURRENCY.acquire(
            self.cancel_event,
            report_waiting,
        )

    def _run_isolated_process(self, started: float) -> dict[str, Any]:
        """Spawn, supervise, and collect one native render outside the GUI process."""

        context = field_render_process_context()
        cancel_receive_connection, cancel_send_connection = context.Pipe(
            duplex=False
        )
        receive_connection, send_connection = context.Pipe(duplex=False)
        work_directory = Path(
            tempfile.mkdtemp(prefix=f"field-workbench-render-{self.job_id}-")
        )
        result_path = work_directory / "render-result.pickle"
        crash_log_path = work_directory / "native-crash.log"
        process = context.Process(
            target=run_field_render_process,
            args=(
                send_connection,
                str(result_path),
                str(crash_log_path),
                self.document,
                self.preferences,
                self.render_kind,
                self.request,
                cancel_receive_connection,
            ),
            name=f"FieldRenderSolver-{self.job_id}-{self.render_kind}",
        )
        with self._process_cancel_lock:
            self._process_cancel_connection = cancel_send_connection
            self._process_cancel_sent = False
        self._process = process
        terminal_kind: str | None = None
        failure_message: str | None = None
        last_stage: str | None = None
        diagnostic_path: str | None = None
        cancellation_started: float | None = None
        process_started = False
        pipe_open = True
        preserve_crash_log = False
        preserve_result = False

        def handle_message(message: Any) -> None:
            nonlocal terminal_kind, failure_message, last_stage, diagnostic_path
            if not isinstance(message, dict):
                return
            kind = str(message.get("kind", ""))
            if kind == "progress":
                update = message.get("update")
                if isinstance(update, dict):
                    stage = str(update.get("stage", "")).strip()
                    if stage:
                        last_stage = stage
                    self.progress.emit(self.job_id, dict(update))
            elif kind == "diagnostics":
                path = str(message.get("path", "")).strip()
                diagnostic_path = path or diagnostic_path
            elif kind in {"completed", "cancelled", "failed"}:
                terminal_kind = kind
                if kind == "failed":
                    failure_message = str(
                        message.get("message") or "The isolated render failed."
                    )

        try:
            if self.cancel_event.is_set():
                self._signal_process_cancellation()
            self.progress.emit(
                self.job_id,
                {
                    "stage": f"Starting {self._render_dimension_label()} render worker…",
                    "render_task": "launch",
                    "detail": "Launching the render process and transferring the job data.",
                    "completed": 0,
                    "total": 1,
                    "unit": "render job",
                },
            )
            # The process-launch handshake serializes the scene and creates native
            # process resources. Guard only that short operation; once started,
            # calculations overlap up to the configured global worker limit.
            with _FIELD_RENDER_PROCESS_START_LOCK:
                process.start()
            process_started = True
            send_connection.close()
            cancel_receive_connection.close()
            # The launch has serialized the immutable request into the child. Release
            # the QThread's references before either process allocates field arrays.
            self.document = {}
            self.request = {}

            while True:
                if self.cancel_event.is_set():
                    self._signal_process_cancellation()
                    if cancellation_started is None:
                        cancellation_started = time.monotonic()

                if pipe_open:
                    try:
                        if receive_connection.poll(0.05):
                            handle_message(receive_connection.recv())
                    except (EOFError, OSError):
                        pipe_open = False
                else:
                    time.sleep(0.02)

                if (
                    cancellation_started is not None
                    and process.is_alive()
                    and time.monotonic() - cancellation_started
                    >= _FIELD_RENDER_CANCEL_GRACE_SECONDS
                ):
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive() and hasattr(process, "kill"):
                        process.kill()

                if not process.is_alive():
                    process.join()
                    if pipe_open:
                        while True:
                            try:
                                if not receive_connection.poll(0):
                                    break
                                handle_message(receive_connection.recv())
                            except (EOFError, OSError):
                                pipe_open = False
                                break
                    break

            exit_code = process.exitcode
            if self.cancel_event.is_set() or terminal_kind == "cancelled":
                raise FieldCalculationCancelled("Calculation cancelled.")
            if terminal_kind == "failed":
                message = failure_message or "The isolated render failed."
                if diagnostic_path:
                    message += f"\nSolver diagnostics: {diagnostic_path}"
                raise StudioOperationError(message)
            if exit_code != 0:
                preserve_crash_log = True
                raise StudioOperationError(
                    _field_render_process_exit_message(
                        exit_code,
                        last_stage=last_stage,
                        crash_log_path=str(crash_log_path),
                        diagnostic_path=diagnostic_path,
                    )
                )
            if not result_path.is_file():
                raise StudioOperationError(
                    "The isolated field solver exited without producing a render result."
                )
            preserve_result = True
            return {
                "result_path": str(result_path),
                "work_directory": str(work_directory),
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            self._process = None
            with self._process_cancel_lock:
                parent_cancel_connection = self._process_cancel_connection
                self._process_cancel_connection = None
                self._process_cancel_sent = False
            if parent_cancel_connection is not None:
                try:
                    parent_cancel_connection.close()
                except (OSError, ValueError):
                    pass
            try:
                cancel_receive_connection.close()
            except (OSError, ValueError):
                pass
            try:
                receive_connection.close()
            except (OSError, ValueError):
                pass
            if not process_started:
                try:
                    send_connection.close()
                except (OSError, ValueError):
                    pass
                try:
                    process.close()
                except (AttributeError, ValueError):
                    pass
            if process_started and process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=0.5)
            if process_started and not process.is_alive():
                try:
                    process.close()
                except (AttributeError, ValueError):
                    pass
            if preserve_crash_log:
                for path in (
                    result_path,
                    result_path.with_name(result_path.name + ".tmp"),
                ):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            elif not preserve_result:
                shutil.rmtree(work_directory, ignore_errors=True)

    @Slot()
    def run(self) -> None:
        started = time.perf_counter()
        solver_acquired = False
        try:
            if self.cancel_event.is_set():
                raise FieldCalculationCancelled("Calculation cancelled.")
            active, limit = self._acquire_solver()
            solver_acquired = True
            if self.cancel_event.is_set():
                raise FieldCalculationCancelled("Calculation cancelled.")
            self.progress.emit(
                self.job_id,
                {
                    "stage": f"{self._render_dimension_label()} render slot admitted…",
                    "render_task": "launch",
                    "detail": (
                        f"Running {active} of up to {limit} simultaneous render "
                        "calculations allowed by Preferences."
                    ),
                    "completed": 0,
                    "total": 1,
                    "unit": "render job",
                },
            )
            result = self._run_isolated_process(started)
            self.completed.emit(self.job_id, result)
        except FieldCalculationCancelled:
            self.cancelled.emit(self.job_id)
        except StudioOperationError as error:
            if self.cancel_event.is_set() or "cancelled" in str(error).lower():
                self.cancelled.emit(self.job_id)
            else:
                self.failed.emit(self.job_id, str(error))
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(
                self.job_id,
                f"Unexpected render error ({type(error).__name__}): {error}",
            )
        finally:
            self.document = {}
            self.request = {}
            if solver_acquired:
                _FIELD_RENDER_CONCURRENCY.release()


class MeasurementAnalysisProgressDialog(_FieldMapProgressDialog):
    """Determinate, cancellable progress window for measurement-volume analysis."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("measurementAnalysisProgress")
        self.setWindowTitle("Analyzing measurement volume")
        self.setLabelText("Preparing measurement-volume samples…")
        self.setCancelButton(None)
        self.setRange(0, 1)
        self.setValue(0)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(470)
        bar = self.findChild(QProgressBar)
        if bar is not None:
            bar.setFormat("Preparing volume samples…  •  elapsed 0.0 s")
        self.show()
        QApplication.processEvents()
        if parent is not None:
            _centre_field_map_progress(self, parent)
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def update_from_solver(self, update: dict) -> None:
        payload = dict(update)
        payload.setdefault("unit", "field samples")
        _update_field_map_progress(self, payload)

    def request_cancellation(self) -> None:
        if self._field_map_cancel_requested or self._field_map_allow_close:
            return
        self._field_map_cancel_requested = True
        self.setWindowTitle("Cancelling volume analysis")
        self.setLabelText(
            "Cancelling after the current solver batch finishes…\n"
            f"Elapsed: {_format_elapsed_time(self.elapsed_seconds())}"
        )
        bar = self.findChild(QProgressBar)
        if bar is not None:
            bar.setFormat("Cancelling after current batch…")

    def elapsed_seconds(self) -> float:
        return _field_map_elapsed(self)

    def finish(self) -> None:
        _finish_field_map_progress(self)


SELECTABLE_TEXT_FLAGS = (
    Qt.TextInteractionFlag.TextSelectableByMouse
    | Qt.TextInteractionFlag.TextSelectableByKeyboard
)


def _selectable_label(
    text: object,
    *,
    word_wrap: bool = False,
    alignment: Qt.AlignmentFlag | None = None,
) -> QLabel:
    """Create a result label whose contents can be selected and copied."""
    label = QLabel(str(text))
    label.setTextInteractionFlags(SELECTABLE_TEXT_FLAGS)
    label.setWordWrap(word_wrap)
    if alignment is not None:
        label.setAlignment(alignment)
    return label


def _make_result_text_selectable(widget: QWidget) -> None:
    """Make every existing label in a result surface mouse/keyboard selectable."""
    for label in widget.findChildren(QLabel):
        label.setTextInteractionFlags(label.textInteractionFlags() | SELECTABLE_TEXT_FLAGS)


class SensorPathProgressDialog(_FieldMapProgressDialog):
    """Determinate, cancellable progress window for sensor-path evaluation."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("sensorPathCalculationProgress")
        self.setWindowTitle("Calculating sensor path")
        self.setLabelText("Preparing sensor-path samples…")
        self.setCancelButton(None)
        self.setRange(0, 1)
        self.setValue(0)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(470)
        bar = self.findChild(QProgressBar)
        if bar is not None:
            bar.setFormat("Preparing sensor samples…  •  elapsed 0.0 s")
        self.show()
        QApplication.processEvents()
        if parent is not None:
            _centre_field_map_progress(self, parent)
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def update_from_solver(self, update: dict) -> None:
        payload = dict(update)
        payload.setdefault("unit", "field samples")
        _update_field_map_progress(self, payload)

    def request_cancellation(self) -> None:
        if self._field_map_cancel_requested or self._field_map_allow_close:
            return
        self._field_map_cancel_requested = True
        self.setWindowTitle("Cancelling sensor path")
        self.setLabelText(
            "Cancelling after the current solver batch finishes…\n"
            f"Elapsed: {_format_elapsed_time(self.elapsed_seconds())}"
        )
        bar = self.findChild(QProgressBar)
        if bar is not None:
            bar.setFormat("Cancelling after current batch…")

    def elapsed_seconds(self) -> float:
        return _field_map_elapsed(self)

    def finish(self) -> None:
        _finish_field_map_progress(self)


def _copyable_statistics_group(
    title: str,
    rows: list[tuple[str, str]],
) -> QGroupBox:
    """Build a selectable statistics section with an exact plain-text copy action."""
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    form = QFormLayout()
    for row_label, value in rows:
        form.addRow(_selectable_label(row_label), _selectable_label(value, word_wrap=True))
    layout.addLayout(form)

    actions = QHBoxLayout()
    actions.addStretch(1)
    copy_button = QPushButton("Copy text")
    copy_button.setObjectName("copyStatisticsSectionButton")
    copy_button.setProperty("statisticsSection", title)
    copy_button.setToolTip(f"Copy the {title} section as plain text")

    def copy_section(_checked=False):
        report = title + "\n" + "\n".join(
            f"{row_label}: {value}" for row_label, value in rows
        )
        QApplication.clipboard().setText(report)
        copy_button.setText("Copied")

    copy_button.clicked.connect(copy_section)
    actions.addWidget(copy_button)
    layout.addLayout(actions)
    return group


class CollapsibleSection(QWidget):
    """Compact disclosure section used by dense map setup sidebars."""

    def __init__(self, title: str, parent=None, *, expanded: bool = True):
        super().__init__(parent)
        self._title = str(title)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self.header = QToolButton()
        self.header.setText(self._title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.header.setAutoRaise(True)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setStyleSheet(
            "QToolButton { font-weight: 600; text-align: left; padding: 5px 2px; }"
        )
        root.addWidget(self.header)

        self.content = QWidget()
        self.content.setMinimumWidth(0)
        self.content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.content.setVisible(bool(expanded))
        root.addWidget(self.content)
        self.header.toggled.connect(self._set_expanded)

    def setContentLayout(self, layout: QLayout) -> None:  # noqa: N802 - Qt-style helper
        self.content.setLayout(layout)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))


class EditorWheelScrollFilter(QObject):
    """Route wheel gestures over form editors to their enclosing page.

    Spin boxes and closed combo boxes normally consume the wheel and silently
    change values.  On a long setup page the wheel is navigation instead;
    values remain editable through typing, arrow buttons, and opened menus.
    """

    def __init__(self, scroll_area: QScrollArea):
        super().__init__(scroll_area)
        self.scroll_area = scroll_area

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if event.type() != QEvent.Type.Wheel or not isinstance(
            watched, (QDoubleSpinBox, QSpinBox, QComboBox)
        ):
            return super().eventFilter(watched, event)

        scroll_bar = self.scroll_area.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            distance = pixel_delta
        else:
            wheel_steps = event.angleDelta().y() / 120.0
            distance = wheel_steps * max(36, scroll_bar.singleStep() * 3)
        scroll_bar.setValue(scroll_bar.value() - round(distance))
        event.accept()
        return True


class MeasurementPreviewWidget(QWidget):
    """Small non-interactive local-frame preview for a measurement result."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.setMinimumHeight(190)
        self.setToolTip(
            "Static preview of the measurement geometry and sampled field in local coordinates"
        )

    @staticmethod
    def _colour(value: float, low: float, high: float) -> QColor:
        if not math.isfinite(value):
            return QColor("#94a3b8")
        t = 0.5 if high <= low else max(0.0, min(1.0, (value - low) / (high - low)))
        stops = [
            (0.0, QColor("#2563eb")),
            (0.38, QColor("#14b8a6")),
            (0.68, QColor("#eab308")),
            (1.0, QColor("#dc2626")),
        ]
        for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
            if t <= x1:
                fraction = (t - x0) / (x1 - x0)
                return QColor(
                    round(c0.red() + fraction * (c1.red() - c0.red())),
                    round(c0.green() + fraction * (c1.green() - c0.green())),
                    round(c0.blue() + fraction * (c1.blue() - c0.blue())),
                    205,
                )
        return QColor(stops[-1][1])

    def paintEvent(self, event):  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        canvas = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        preview_colours = plot_theme_colors(current_theme())
        border_colour = "#56616b" if current_theme() == "dark" else "#d9e0e8"
        painter.setPen(QPen(QColor(border_colour), 1.0))
        painter.setBrush(QColor(preview_colours["canvas_outer"]))
        painter.drawRoundedRect(canvas, 6.0, 6.0)

        points = self.result.get("local_points_m")
        if points is None:
            points = self.result.get("points_m", [])
        try:
            points = [list(map(float, point)) for point in points]
            magnitudes = [float(value) for value in self.result.get("magnitudes_uT", [])]
        except (TypeError, ValueError):
            points, magnitudes = [], []
        if not points or len(points) != len(magnitudes):
            painter.setPen(QColor(preview_colours["text"]))
            painter.drawText(canvas, Qt.AlignmentFlag.AlignCenter, "Preview unavailable")
            return

        if len(points) > 1100:
            step = max(1, len(points) // 1100)
            indices = list(range(0, len(points), step))[:1100]
            draw_points = [points[index] for index in indices]
            draw_magnitudes = [magnitudes[index] for index in indices]
        else:
            draw_points = points
            draw_magnitudes = magnitudes

        dimensions_mm = self.result.get("dimensions_mm", [])
        try:
            dimensions = [float(value) / 1000.0 for value in dimensions_mm]
        except (TypeError, ValueError):
            dimensions = []
        if len(dimensions) != 3 or not all(value > 0 for value in dimensions):
            axis_count = max(1, int(self.result.get("axis_count", 1)))
            spacing = [
                float(value) / 1000.0 for value in self.result.get("spacing_mm", [])
            ]
            spacing = (spacing + [0.0, 0.0, 0.0])[:3]
            dimensions = [
                max(spacing[index] * axis_count, 1e-12) for index in range(3)
            ]
        half = [value / 2.0 for value in dimensions]
        try:
            bounds_centre = [
                float(value)
                for value in self.result.get(
                    "local_bounds_centre_m",
                    self.result.get("local_centre_m", []),
                )
            ]
        except (TypeError, ValueError):
            bounds_centre = []
        if len(bounds_centre) != 3 or not all(math.isfinite(value) for value in bounds_centre):
            bounds_centre = [
                0.5 * (
                    min(point[axis] for point in points)
                    + max(point[axis] for point in points)
                )
                for axis in range(3)
            ]
        bounds_corners = [
            [
                bounds_centre[0] + sx * half[0],
                bounds_centre[1] + sy * half[1],
                bounds_centre[2] + sz * half[2],
            ]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        extent_points = points + bounds_corners
        projected = [
            (
                point[0] - 0.62 * point[1],
                0.34 * point[0] + 0.34 * point[1] - point[2],
            )
            for point in extent_points
        ]
        u_values = [value[0] for value in projected]
        v_values = [value[1] for value in projected]
        u_low, u_high = min(u_values), max(u_values)
        v_low, v_high = min(v_values), max(v_values)
        u_span = max(1e-12, u_high - u_low)
        v_span = max(1e-12, v_high - v_low)
        plot_rect = canvas.adjusted(30.0, 17.0, -30.0, -25.0)
        scale = min(plot_rect.width() / u_span, plot_rect.height() / v_span) * 0.88
        u_mid = 0.5 * (u_low + u_high)
        v_mid = 0.5 * (v_low + v_high)

        def project(point) -> QPointF:
            u = point[0] - 0.62 * point[1]
            v = 0.34 * point[0] + 0.34 * point[1] - point[2]
            return QPointF(
                plot_rect.center().x() + (u - u_mid) * scale,
                plot_rect.center().y() + (v - v_mid) * scale,
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(preview_colours["reference"]), 1.2))
        shape = str(self.result.get("shape", "")).lower()
        if shape == "sphere":
            radius = max(half)
            for plane in range(3):
                polygon = QPolygonF()
                for index in range(73):
                    angle = 2.0 * math.pi * index / 72.0
                    point = list(bounds_centre)
                    axes = [axis for axis in range(3) if axis != plane]
                    point[axes[0]] = bounds_centre[axes[0]] + radius * math.cos(angle)
                    point[axes[1]] = bounds_centre[axes[1]] + radius * math.sin(angle)
                    polygon.append(project(point))
                painter.drawPolyline(polygon)
        elif shape == "cylinder":
            radius = max(half[0], half[1])
            z_values = (
                bounds_centre[2] - half[2],
                bounds_centre[2] + half[2],
            )
            for z_value in z_values:
                polygon = QPolygonF()
                for index in range(73):
                    angle = 2.0 * math.pi * index / 72.0
                    polygon.append(
                        project(
                            [
                                bounds_centre[0] + radius * math.cos(angle),
                                bounds_centre[1] + radius * math.sin(angle),
                                z_value,
                            ]
                        )
                    )
                painter.drawPolyline(polygon)
            for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
                x_value = bounds_centre[0] + radius * math.cos(angle)
                y_value = bounds_centre[1] + radius * math.sin(angle)
                painter.drawLine(
                    project([x_value, y_value, z_values[0]]),
                    project([x_value, y_value, z_values[1]]),
                )
        else:
            corners = [
                [
                    bounds_centre[0] + sx * half[0],
                    bounds_centre[1] + sy * half[1],
                    bounds_centre[2] + sz * half[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
            for first, second in (
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
            ):
                painter.drawLine(project(corners[first]), project(corners[second]))

        finite_magnitudes = [value for value in magnitudes if math.isfinite(value)]
        if finite_magnitudes:
            ordered = sorted(finite_magnitudes)
            low = ordered[max(0, round(0.05 * (len(ordered) - 1)))]
            high = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        else:
            low, high = 0.0, 1.0
        order = sorted(range(len(draw_points)), key=lambda index: sum(draw_points[index]))
        painter.setPen(Qt.PenStyle.NoPen)
        radius = 2.2 if len(draw_points) < 500 else 1.65
        for index in order:
            painter.setBrush(self._colour(draw_magnitudes[index], low, high))
            location = project(draw_points[index])
            painter.drawEllipse(location, radius, radius)

        painter.setPen(QColor(preview_colours["text"]))
        painter.drawText(
            QRectF(canvas.left() + 10, canvas.bottom() - 21, canvas.width() - 20, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{len(points):,} samples • colour shows |B| • local-coordinate view",
        )


class DrcWorker(QObject):
    """Run one DRC pass on a private scene adapter away from the GUI thread."""

    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, document: dict):
        super().__init__()
        self.document = copy.deepcopy(document)

    @Slot()
    def run(self) -> None:
        try:
            adapter = StudioAdapter(self.document)
            report = adapter.run_drc(progress_callback=self.progress.emit)
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(report)


class DrcResultsDialog(QDialog):
    """Modeless physical design-rule report with clickable object pairs."""

    STATUS_TEXT = {
        "clear": "Clear",
        "below_clearance": "Below clearance",
        "intersecting": "Intersecting",
        "invalid": "Incomplete / unsupported",
        "clearance_only": "Surface clear only",
        "precheck_clear": "Centreline precheck clear",
        "mesh_warning": "Mesh warning",
        "mesh_ok": "Mesh healthy",
    }
    STATUS_COLOUR = {
        "clear": "#15803d",
        "below_clearance": "#b45309",
        "intersecting": "#b91c1c",
        "invalid": "#64748b",
        "clearance_only": "#0369a1",
        "precheck_clear": "#0369a1",
        "mesh_warning": "#b45309",
        "mesh_ok": "#15803d",
    }

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        on_select: Callable[[list[str]], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.on_select = on_select
        self.on_settings_changed = on_settings_changed
        self.report: dict | None = None
        self.thread: QThread | None = None
        self.worker: DrcWorker | None = None
        self._live_counts: dict[str, int] = {}
        self._live_result_count = 0
        self._run_started_at: float | None = None
        self._progress_message = "Ready — press Check design to run DRC"
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(100)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed_progress)
        self.setWindowTitle("Design Rule Check")
        self.resize(1050, 620)
        self.setModal(False)

        root = QVBoxLayout(self)
        heading = QLabel("Physical design-rule check")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        settings = adapter.get_drc_settings()
        settings_box = QGroupBox("Clearance rules")
        settings_layout = QHBoxLayout(settings_box)
        default_column = QVBoxLayout()
        default_column.addWidget(QLabel("Default coil-to-exclusion clearance"))
        self.default_clearance = _spin(
            settings["default_clearance_mm"], 0.0, 1e6, 3, " mm"
        )
        default_column.addWidget(self.default_clearance)
        settings_layout.addLayout(default_column)
        coil_column = QVBoxLayout()
        coil_column.addWidget(QLabel("Coil-to-coil clearance"))
        self.coil_clearance = _spin(
            settings["coil_to_coil_clearance_mm"], 0.0, 1e6, 3, " mm"
        )
        coil_column.addWidget(self.coil_clearance)
        settings_layout.addLayout(coil_column)
        mesh_column = QVBoxLayout()
        mesh_column.addWidget(QLabel("Imported-mesh tolerance"))
        self.mesh_tolerance = _spin(
            settings["mesh_tolerance_mm"], 0.001, 100.0, 3, " mm"
        )
        self.mesh_tolerance.setToolTip(
            "Adaptive surface-distance accuracy for STL/OBJ checking. This is not added "
            "to the required physical clearance."
        )
        mesh_column.addWidget(self.mesh_tolerance)
        settings_layout.addLayout(mesh_column)
        settings_layout.addStretch(1)
        self.run_button = QPushButton("Check design")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumHeight(42)
        self.run_button.clicked.connect(self.run_checks)
        settings_layout.addWidget(self.run_button)
        root.addWidget(settings_box)

        note = QLabel(
            "Per-zone and per-region overrides take precedence over inherited clearance. Hidden "
            "active regions remain active. Region-aware watertight hosts use the complete shell for "
            "penetration/containment and named triangle patches for local clearance. Broad-phase "
            "boxes only reject obviously clear pairs; close verdicts use the physical winding, "
            "barrel, and flange solids. If physical construction is unavailable, Check design uses the "
            "zero-thickness magnetic current path as a one-way precheck: failures are definitive, while "
            "precheck-clear results do not certify the eventual winding envelope."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        self.summary = QLabel("Run Check design to inspect the current construction geometry.")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.results = QTreeWidget()
        self.results.setHeaderLabels(
            ["Status", "First object", "Second object", "Clearance / overlap", "Rule"]
        )
        self.results.setAlternatingRowColors(True)
        self.results.setRootIsDecorated(False)
        self.results.itemSelectionChanged.connect(self._select_current_result)
        root.addWidget(self.results, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("drcProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready — press Check design to run DRC")
        root.addWidget(self.progress_bar)

        self.footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.footer.rejected.connect(self.reject)
        root.addWidget(self.footer)

    def _drc_elapsed_seconds(self) -> float:
        if self._run_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._run_started_at)

    def _refresh_elapsed_progress(self) -> None:
        if self._run_started_at is None:
            return
        self.progress_bar.setFormat(
            f"{self._progress_message}  •  elapsed {_format_elapsed_time(self._drc_elapsed_seconds())}"
        )

    def _set_drc_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, int(percent))))
        self._progress_message = str(message)
        if self._run_started_at is None:
            self.progress_bar.setFormat(self._progress_message)
        else:
            self._refresh_elapsed_progress()

    def _finish_drc_progress(self, message: str, *, value: int) -> None:
        self._progress_message = str(message)
        self._set_drc_progress(value, message)
        self._elapsed_timer.stop()
        self._refresh_elapsed_progress()
        self._run_started_at = None

    def _set_running(self, running: bool) -> None:
        self.default_clearance.setEnabled(not running)
        self.coil_clearance.setEnabled(not running)
        self.mesh_tolerance.setEnabled(not running)
        self.run_button.setEnabled(not running)
        close_button = self.footer.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setEnabled(not running)

    def run_checks(self) -> None:
        if self.thread is not None:
            return
        new_settings = {
            "default_clearance_mm": self.default_clearance.value(),
            "coil_to_coil_clearance_mm": self.coil_clearance.value(),
            "mesh_tolerance_mm": self.mesh_tolerance.value(),
        }
        try:
            if new_settings != self.adapter.get_drc_settings():
                self.adapter.set_drc_settings(new_settings)
                if self.on_settings_changed is not None:
                    self.on_settings_changed()
            document = self.adapter.document()
        except Exception as error:  # noqa: BLE001 - GUI error boundary
            QMessageBox.warning(self, "Design Rule Check", str(error))
            return

        self.report = None
        self.results.clear()
        self._live_counts = {}
        self._live_result_count = 0
        self.summary.setProperty("statusTone", "")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)
        self.summary.setText("Running design-rule check…")
        self._run_started_at = time.monotonic()
        self._progress_message = "Preparing design-rule check…"
        self._elapsed_timer.start()
        self._set_drc_progress(0, self._progress_message)
        self._set_running(True)

        self.thread = QThread(self)
        self.worker = DrcWorker(document)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(dict)
    def _progress(self, update: dict) -> None:
        percent = max(0, min(100, int(update.get("percent", 0))))
        self._set_drc_progress(
            percent, str(update.get("message", "Checking design…"))
        )

        result = update.get("result")
        if isinstance(result, dict):
            self._append_result_item(result)
            status = str(result.get("status", ""))
            self._live_counts[status] = self._live_counts.get(status, 0) + 1
            self._live_result_count += 1
            self._update_running_summary()

    @Slot(dict)
    def _completed(self, report: dict) -> None:
        self.report = report
        # The worker ran on a private adapter; transfer its complete report to the
        # live adapter so AI Help can reuse it immediately when the scene is unchanged.
        self.adapter.cache_drc_report(report)
        self._finish_drc_progress("Design-rule check complete", value=100)
        self._populate_report()
        self.refresh_staleness()

    @Slot(str)
    def _failed(self, message: str) -> None:
        self._finish_drc_progress("Design-rule check failed", value=0)
        self.summary.setText("Design-rule check failed.")
        QMessageBox.warning(self, "Design Rule Check", message)

    @Slot()
    def _thread_finished(self) -> None:
        # DRC owns the elapsed-time timer; do not reference the optimizer dialog's
        # progress timer here.  An exception in this slot prevents the worker/thread
        # references from being cleared after QThread.finished.
        self._elapsed_timer.stop()
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        if self._run_started_at is not None and not self._elapsed_timer.isActive():
            self._run_started_at = None
        self._set_running(False)

    def reject(self) -> None:
        if self.thread is not None:
            return
        super().reject()

    def _update_running_summary(self) -> None:
        """Show a compact rolling verdict while rows stream in from the DRC worker."""
        problems = (
            self._live_counts.get("intersecting", 0)
            + self._live_counts.get("below_clearance", 0)
            + self._live_counts.get("invalid", 0)
        )
        cautions = (
            self._live_counts.get("clearance_only", 0)
            + self._live_counts.get("precheck_clear", 0)
            + self._live_counts.get("mesh_warning", 0)
        )
        self.summary.setText(
            f"Running… {self._live_result_count} result"
            + ("" if self._live_result_count == 1 else "s")
            + f" reported • {problems} problem"
            + ("" if problems == 1 else "s")
            + f" • {cautions} caution/precheck"
            + ("" if cautions == 1 else "s")
        )

    def _append_result_item(self, result: dict) -> QTreeWidgetItem:
        """Append one completed DRC verdict without waiting for the full report."""
        status = result["status"]
        distance = result["signed_clearance_mm"]
        if distance is None:
            clearance_text = result.get("reason", "—")
        elif status == "intersecting":
            clearance_text = f"Overlap ≈ {abs(distance):,.5g} mm"
        else:
            if result.get("conservative_lower_bound"):
                prefix = "≥ "
            elif result.get("approximate"):
                prefix = "≈ "
            else:
                prefix = ""
            clearance_text = f"{prefix}{max(0.0, distance):,.5g} mm"
            required = float(result.get("required_clearance_mm", 0.0))
            if required > 0.0:
                clearance_text += f"  (required {required:,.5g} mm)"
        item = QTreeWidgetItem(
            [
                self.STATUS_TEXT[status],
                result["object_a_label"],
                result.get("object_b_label") or "—",
                clearance_text,
                result["rule"],
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, result)
        colour = QColor(self.STATUS_COLOUR[status])
        for column in range(5):
            item.setForeground(column, QBrush(colour))
            if result.get("reason"):
                item.setToolTip(column, str(result["reason"]))
        self.results.addTopLevelItem(item)
        return item

    def _populate_report(self) -> None:
        self.results.clear()
        self.summary.setProperty("statusTone", "")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)
        if not self.report:
            return
        counts = self.report["counts"]
        problem_count = (
            counts["intersecting"] + counts["below_clearance"] + counts["invalid"]
        )
        caution_count = counts["clearance_only"] + counts.get("precheck_clear", 0) + counts["mesh_warning"]
        if problem_count or caution_count:
            self.summary.setText(
                f"{counts['intersecting']} intersecting • {counts['below_clearance']} below "
                f"clearance • {counts['clearance_only']} surface-only • "
                f"{counts.get('precheck_clear', 0)} centreline-precheck clear • "
                f"{counts['mesh_warning']} mesh warnings • {counts['invalid']} incomplete/unsupported • "
                f"{counts['clear']} clear"
            )
        elif counts["clear"]:
            self.summary.setText(f"All {counts['clear']} checked object pairs are clear.")
        elif counts["mesh_ok"]:
            self.summary.setText(
                f"{counts['mesh_ok']} exclusion mesh"
                + ("es are" if counts["mesh_ok"] != 1 else " is")
                + " healthy; no applicable coil pair was found."
            )
        else:
            self.summary.setText(
                "No applicable pairs were found. Add an exclusion zone or active regional DRC rule. "
                "Coils without physical construction use their zero-thickness current path as a precheck."
            )

        for result in self.report["results"]:
            self._append_result_item(result)
        for column in range(5):
            self.results.resizeColumnToContents(column)

    def _select_current_result(self) -> None:
        if self.on_select is None:
            return
        item = self.results.currentItem()
        if item is None:
            return
        result = item.data(0, Qt.ItemDataRole.UserRole) or {}
        object_ids = [
            value
            for value in (result.get("object_a_id"), result.get("object_b_id"))
            if value
        ]
        if object_ids:
            self.on_select(object_ids)

    def refresh_staleness(self) -> None:
        if not self.report or not self.adapter.drc_report_is_stale(self.report):
            return
        self.summary.setText("Results stale — the scene or a clearance rule changed. Run Check design again.")
        self.summary.setProperty("statusTone", "warning")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)


class OptimizerWorker(QObject):
    """Create a private adapter and run one optimizer job off the GUI thread."""

    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        document: dict,
        settings: dict,
        *,
        coil_modeling_method: str = "auto",
        solver_memory_budget_mb: int | None = None,
    ):
        super().__init__()
        self.document = copy.deepcopy(document)
        self.settings = copy.deepcopy(settings)
        self.coil_modeling_method = str(coil_modeling_method)
        self.solver_memory_budget_mb = solver_memory_budget_mb
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            adapter = StudioAdapter(self.document)
            adapter.set_coil_modeling_method(self.coil_modeling_method)
            if self.solver_memory_budget_mb is not None:
                adapter.set_solver_memory_budget_mb(self.solver_memory_budget_mb)
            result = adapter.run_optimizer_tuning(
                self.settings,
                progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
                pause_event=self.pause_event,
            )
        except OptimizationCancelled:
            self.cancelled.emit()
        except StudioOperationError as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(
                f"Unexpected optimizer error ({type(error).__name__}): {error}"
            )
        else:
            self.completed.emit(result)


class OptimizerBenchmarkWorker(QObject):
    """Measure optimizer candidate throughput without touching an open scene."""

    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        *,
        coil_modeling_method: str,
        solver_memory_budget_mb: int,
        max_workers: int,
    ):
        super().__init__()
        self.coil_modeling_method = str(coil_modeling_method)
        self.solver_memory_budget_mb = int(solver_memory_budget_mb)
        self.max_workers = int(max_workers)
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            adapter = StudioAdapter()
            adapter.set_coil_modeling_method(self.coil_modeling_method)
            adapter.set_solver_memory_budget_mb(self.solver_memory_budget_mb)
            result = adapter.benchmark_optimizer_workers(
                max_workers=self.max_workers,
                progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
            )
        except OptimizationCancelled:
            self.cancelled.emit()
        except StudioOperationError as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001 - background task boundary
            self.failed.emit(
                f"Unexpected optimizer benchmark error ({type(error).__name__}): {error}"
            )
        else:
            self.completed.emit(result)


class OptimizerWorkerBenchmarkDialog(QDialog):
    """Cancellable benchmark and recommendation for optimizer concurrency."""

    def __init__(
        self,
        *,
        coil_modeling_method: str,
        solver_memory_budget_mb: int,
        current_workers: int,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.coil_modeling_method = str(coil_modeling_method)
        self.solver_memory_budget_mb = int(solver_memory_budget_mb)
        self.current_workers = normalize_optimizer_worker_count(current_workers)
        self.reported_logical_cpus = reported_logical_cpu_count()
        self.available_workers = available_optimizer_workers()
        self.max_workers = min(16, MAX_OPTIMIZER_BENCHMARK_WORKERS)
        self.recommended_workers: int | None = None
        self.result: dict[str, Any] | None = None
        self.thread: QThread | None = None
        self.worker: OptimizerBenchmarkWorker | None = None
        self._close_after_thread = False

        self.setWindowTitle("Benchmark optimizer workers")
        self.resize(670, 480)
        layout = QVBoxLayout(self)
        heading = QLabel("Benchmark optimizer candidate workers")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        benchmark_range = (
            f"This system reports {self.reported_logical_cpus} logical CPU"
            f"{'s' if self.reported_logical_cpus != 1 else ''}; this is informational only. "
            f"Choose any benchmark ceiling from 1 through {MAX_OPTIMIZER_BENCHMARK_WORKERS} workers. "
            "Counts above the reported logical CPU count intentionally test process oversubscription. "
        )
        explanation = QLabel(
            benchmark_range
            + "Workbench times the same frozen packaged-scene field workload using a warmed, "
            "persistent process pool. Open scenes are not read or changed. "
            "The recommendation favors the smallest count within 8% of the fastest result, "
            "which avoids spending extra memory for negligible throughput."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("hint")
        layout.addWidget(explanation)

        setup_row = QHBoxLayout()
        setup_row.addWidget(QLabel("Test up to"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(
            MIN_OPTIMIZER_WORKERS, MAX_OPTIMIZER_BENCHMARK_WORKERS
        )
        self.max_workers_spin.setValue(self.max_workers)
        self.max_workers_spin.setSuffix(" workers")
        self.max_workers_spin.setToolTip(
            "Highest worker count included in this benchmark. This is not clamped to "
            "the reported CPU count; values above it deliberately test oversubscription."
        )
        setup_row.addWidget(self.max_workers_spin)
        self.start_button = QPushButton("Start benchmark")
        self.start_button.clicked.connect(self._start)
        setup_row.addWidget(self.start_button)
        setup_row.addStretch(1)
        layout.addLayout(setup_row)

        self.results = QTreeWidget()
        self.results.setRootIsDecorated(False)
        self.results.setAlternatingRowColors(True)
        self.results.setHeaderLabels(
            ["Workers", "Elapsed", "Candidates/s", "Speedup"]
        )
        self.results.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results, 1)

        self.status = QLabel(
            f"Choose the highest worker count to test (1–{MAX_OPTIMIZER_BENCHMARK_WORKERS}), "
            "then start the benchmark."
        )
        self.status.setWordWrap(True)
        self.status.setObjectName("hint")
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.max_workers)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        layout.addWidget(self.progress_bar)

        controls = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel benchmark")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        self.use_button = QPushButton("Use recommendation")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self.accept)
        controls.addWidget(self.use_button)
        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(True)
        self.close_button.clicked.connect(self.reject)
        controls.addWidget(self.close_button)
        layout.addLayout(controls)

    def _start(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        self.max_workers = int(self.max_workers_spin.value())
        self.results.clear()
        self.result = None
        self.recommended_workers = None
        self.use_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.max_workers_spin.setEnabled(False)
        self.status.setText(
            f"Preparing benchmark for worker counts 1 through {self.max_workers}…"
        )
        self.progress_bar.setRange(0, self.max_workers)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting…")
        self.thread = QThread(self)
        self.worker = OptimizerBenchmarkWorker(
            coil_modeling_method=self.coil_modeling_method,
            solver_memory_budget_mb=self.solver_memory_budget_mb,
            max_workers=self.max_workers,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._update_progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(dict)
    def _update_progress(self, update: dict) -> None:
        row = dict(update.get("result", {}))
        if row:
            item = QTreeWidgetItem(
                [
                    str(int(row.get("workers", 1))),
                    f"{float(row.get('elapsed_seconds', 0.0)):.3f} s",
                    f"{float(row.get('tasks_per_second', 0.0)):.3f}",
                    f"{float(row.get('speedup', 0.0)):.2f}×",
                ]
            )
            for column in range(4):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignCenter)
            self.results.addTopLevelItem(item)
        completed = int(update.get("completed", 0))
        total = int(update.get("total", self.max_workers))
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat(f"{completed} of {total} worker counts tested")
        self.status.setText(f"Testing worker counts… {completed} of {total} complete.")

    @Slot(dict)
    def _completed(self, result: dict) -> None:
        self.result = copy.deepcopy(result)
        self.recommended_workers = int(result["recommended_workers"])
        self.status.setText(
            f"Recommended: {self.recommended_workers} worker"
            f"{'s' if self.recommended_workers != 1 else ''}. The benchmark tested "
            f"{int(result['task_count'])} independent candidates per worker count using "
            f"{int(result['observer_count'])} observer points per candidate."
        )
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setFormat(
            f"Recommended: {self.recommended_workers} worker"
            f"{'s' if self.recommended_workers != 1 else ''}"
        )
        self.use_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.max_workers_spin.setEnabled(True)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.status.setText(f"Benchmark failed: {message}")
        self.progress_bar.setFormat("Benchmark failed")
        self.close_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.max_workers_spin.setEnabled(True)

    @Slot()
    def _cancelled(self) -> None:
        self.status.setText("Benchmark cancelled. No preference was changed.")
        self.progress_bar.setFormat("Cancelled")
        self.close_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.max_workers_spin.setEnabled(True)

    def _request_cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling after the active field evaluation returns…")

    @Slot()
    def _thread_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        if self._close_after_thread:
            QTimer.singleShot(0, self.reject)

    def reject(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self._close_after_thread = True
            self._request_cancel()
            return
        super().reject()


class OptimizerDialog(QDialog):
    """Wizard-style optimizer that formulates an engineering question before searching."""

    PAGE_START = 0
    PAGE_PARAMETERS = 1
    PAGE_TARGET = 2
    PAGE_GOAL = 3
    PAGE_FREEDOM = 4
    PAGE_LIMITS = 5
    PAGE_REVIEW = 6
    PAGE_RESULTS = 7

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        suggested_target_id: str | None = None,
        on_apply: Callable[[dict, bool], None] | None = None,
        on_append: Callable[[dict], None] | None = None,
        on_open_new_tab: Callable[[dict, dict], None] | None = None,
        on_save_snapshot: Callable[[dict, str, dict], None] | None = None,
        preferences: QSettings | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.source_document = adapter.document()
        self.preview_adapter = StudioAdapter(self.source_document)
        self.coil_modeling_method = str(adapter.coil_modeling_method)
        self.solver_memory_budget_mb = int(adapter.solver_memory_budget_mb)
        self.preview_adapter.set_coil_modeling_method(self.coil_modeling_method)
        self.preview_adapter.set_solver_memory_budget_mb(self.solver_memory_budget_mb)
        self.optimizer_worker_count = int(adapter.optimizer_worker_count)
        self.on_apply = on_apply
        self.on_append = on_append
        self.on_open_new_tab = on_open_new_tab
        self.on_save_snapshot = on_save_snapshot
        self._registry_records_by_id: dict[str, dict[str, Any]] = {}
        self._registry_views: dict[str, list[str]] = {}
        self._materialized_candidate_cache: dict[str, dict[str, Any]] = {}
        self._registry_source_transform_cache: dict[str, dict[str, Any]] = {}
        # MainWindow supplies its application QSettings so Freedom-page numeric
        # entries can remember the last values the user actually used.  Direct
        # programmatic/test construction may omit it, in which case persistence
        # is intentionally disabled rather than leaking state between callers.
        self._freedom_preferences = preferences
        self.run_result: dict | None = None
        self.thread: QThread | None = None
        self.worker: OptimizerWorker | None = None
        self._close_after_thread = False
        self._optimizer_run_started_at: float | None = None
        self._optimizer_elapsed_override: float | None = None
        self._optimizer_last_progress: dict[str, Any] = {}
        self._optimizer_active_settings: dict[str, Any] = {}
        self._optimizer_stage_history: list[dict[str, Any]] = []
        self._optimizer_trace_plan: list[dict[str, Any]] = []
        self._optimizer_trace_nodes: dict[str, dict[str, Any]] = {}
        self._optimizer_trace_active_leaf: str | None = None
        # Determinate optimizer progress is derived from the execution-plan tree.
        # Keep a monotonic floor because adaptive branches can discover a larger
        # denominator after they have started; newly discovered work must consume
        # the still-unfilled part of the bar rather than rewinding it.
        self._optimizer_progress_fraction = 0.0
        self._optimizer_paused = False
        self._optimizer_cancel_requested = False
        self._optimizer_progress_timer = QTimer(self)
        self._optimizer_progress_timer.setInterval(200)
        self._optimizer_progress_timer.timeout.connect(self._refresh_optimizer_progress_display)
        self.search_scope = ""
        self.requested_freedoms: set[str] = set()
        self._coil_records = {
            str(item["id"]): item for item in self.preview_adapter.optimizer_existing_coils()
        }
        self._target_records = {
            str(item["id"]): item for item in self.preview_adapter.optimizer_targets()
        }
        self._snapshot_records = {
            str(item.get("id", "")): item
            for item in self.preview_adapter.list_snapshots()
            if str(item.get("id", "")) and str(item.get("status", "")) != "corrupt"
        }

        self.setWindowTitle("Optimizer — Design Wizard")
        self.resize(1180, 820)
        # Give the detached optimizer an explicit native resize affordance in
        # addition to the normal title-bar maximize/restore controls.
        self.setSizeGripEnabled(True)

        root = QVBoxLayout(self)
        self.dialog_heading = QLabel("Optimizer — formulate the design question")
        self.dialog_heading.setObjectName("dialogHeading")
        root.addWidget(self.dialog_heading)
        self.dialog_intro = QLabel(
            "The optimizer is organized around four things: TARGET — where the field matters; "
            "GOAL — what better means; FREEDOM — what may change; and LIMITS — what must never "
            "be violated. Workbench chooses the search mechanics after the engineering question is explicit."
        )
        self.dialog_intro.setWordWrap(True)
        self.dialog_intro.setObjectName("hint")
        root.addWidget(self.dialog_intro)

        self.step_label = QLabel("")
        self.step_label.setObjectName("sectionHeading")
        root.addWidget(self.step_label)

        self.pages = QStackedWidget()
        # QStackedWidget normally contributes the largest size hint of *all*
        # wizard pages to the top-level minimum size. That made a completed
        # Results page inherit the tall Freedom/Review requirements and could
        # leave the native window impossible to drag above the taskbar. The
        # form pages already own scroll areas, so allow the stack to shrink to
        # the page currently being shown.
        self.pages.setMinimumSize(0, 0)
        self.pages.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        root.addWidget(self.pages, 1)
        self._build_start_page()
        self._build_parameter_page()
        self._build_target_page(suggested_target_id)
        self._build_goal_page(suggested_target_id)
        self._build_freedom_page()
        self._build_limits_page(suggested_target_id)
        self._build_review_page()
        self._build_results_page()
        self.pages.currentChanged.connect(self._page_changed)

        # Keep the legacy progress labels as an internal/text-test surface, but do
        # not show them in the dialog.  Search Trace now owns this telemetry and
        # repeating the same stage/count/best text immediately above the progress
        # bar wastes the vertical space the live console needs.
        self.progress_status = QLabel("Ready")
        self.progress_status.setWordWrap(True)
        self.progress_status.setObjectName("hint")
        self.progress_status.setVisible(False)
        root.addWidget(self.progress_status)
        self.progress_best = QLabel("")
        self.progress_best.setWordWrap(True)
        self.progress_best.setObjectName("hint")
        self.progress_best.setVisible(False)
        root.addWidget(self.progress_best)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        progress_row.addWidget(self.progress_bar, 1)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        progress_row.addWidget(self.pause_button)
        self.cancel_button = QPushButton("Cancel search")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_search)
        progress_row.addWidget(self.cancel_button)
        root.addLayout(progress_row)

        navigation = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._previous_page)
        navigation.addWidget(self.back_button)
        navigation.addStretch(1)
        self.freedom_defaults_button = QPushButton("Defaults")
        self.freedom_defaults_button.setToolTip(
            "Restore the Freedom page's scene-derived numeric bounds and increments. "
            "Checkboxes and relationship selections are left unchanged."
        )
        self.freedom_defaults_button.clicked.connect(self._restore_freedom_input_defaults)
        navigation.addWidget(self.freedom_defaults_button)
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("primaryButton")
        self.next_button.setMinimumHeight(40)
        self.next_button.clicked.connect(self._next_page)
        navigation.addWidget(self.next_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        navigation.addWidget(self.close_button)
        root.addLayout(navigation)

        self.pages.setCurrentIndex(self.PAGE_START)
        self._update_navigation()

    @staticmethod
    def _row_widget(*widgets: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return container

    @staticmethod
    def _integer(value: int, minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(int(value))
        widget.setSuffix(suffix)
        widget.setGroupSeparatorShown(True)
        widget.setMinimumWidth(92)
        return widget

    def _scroll_page(self, title: str, description: str) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        title_label = QLabel(title)
        title_label.setObjectName("dialogHeading")
        layout.addWidget(title_label)
        note = QLabel(description)
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return scroll, layout

    @staticmethod
    def _install_wheel_filter(scroll: QScrollArea, root: QWidget) -> EditorWheelScrollFilter:
        wheel_filter = EditorWheelScrollFilter(scroll)
        for editor_type in (QDoubleSpinBox, QSpinBox, QComboBox):
            for widget in root.findChildren(editor_type):
                widget.installEventFilter(wheel_filter)
        return wheel_filter

    @staticmethod
    def _choice_tree(entries: tuple[tuple[str, str, str], ...], checked: set[str]) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setHeaderLabels(["Change", "Parameter", "What it means"])
        tree.setRootIsDecorated(False)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for key, label, note in entries:
            item = QTreeWidgetItem(["", label, note])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if key in checked else Qt.CheckState.Unchecked)
            tree.addTopLevelItem(item)
        for column in range(tree.columnCount()):
            tree.resizeColumnToContents(column)
        return tree

    @staticmethod
    def _choice_tree_values(tree: QTreeWidget) -> set[str]:
        values: set[str] = set()
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                value = item.data(0, Qt.ItemDataRole.UserRole)
                if value is not None:
                    values.add(str(value))
        return values

    def _simple_choice_changed(self, changed: QTreeWidgetItem, column: int) -> None:
        if column != 0 or changed.checkState(0) != Qt.CheckState.Checked:
            return
        self.simple_choices.blockSignals(True)
        try:
            for index in range(self.simple_choices.topLevelItemCount()):
                item = self.simple_choices.topLevelItem(index)
                if item is not changed:
                    item.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            self.simple_choices.blockSignals(False)

    def _build_start_page(self) -> None:
        scroll, layout = self._scroll_page(
            "Choose an optimizer setup",
            "Start with the simplest route that describes your question. The wizard reveals only the choices needed for that route.",
        )
        self.start_scroll = scroll

        simple = QGroupBox("Simple")
        simple_layout = QVBoxLayout(simple)
        simple_note = QLabel(
            "Change one kind of parameter in the existing scene. The next page asks which one; everything else stays fixed."
        )
        simple_note.setWordWrap(True)
        simple_note.setObjectName("hint")
        simple_layout.addWidget(simple_note)
        simple_button = QPushButton("Simple — change one parameter")
        simple_button.setMinimumHeight(44)
        simple_button.clicked.connect(lambda: self._choose_scope("simple"))
        simple_layout.addWidget(simple_button)
        layout.addWidget(simple)

        complex_box = QGroupBox("Complex")
        complex_layout = QVBoxLayout(complex_box)
        complex_note = QLabel(
            "Change several kinds of parameters in the existing scene. The next page lets you choose which families may vary."
        )
        complex_note.setWordWrap(True)
        complex_note.setObjectName("hint")
        complex_layout.addWidget(complex_note)
        complex_button = QPushButton("Complex — change multiple parameters")
        complex_button.setMinimumHeight(44)
        complex_button.clicked.connect(lambda: self._choose_scope("complex"))
        complex_layout.addWidget(complex_button)
        layout.addWidget(complex_box)

        full = QGroupBox("Skip the wizard")
        full_layout = QVBoxLayout(full)
        full_note = QLabel(
            "Go straight to the full existing-scene optimizer. No parameter family is preselected and no supported Freedom controls are hidden."
        )
        full_note.setWordWrap(True)
        full_note.setObjectName("hint")
        full_layout.addWidget(full_note)
        full_button = QPushButton("Skip wizard — Full control")
        full_button.setMinimumHeight(44)
        full_button.clicked.connect(lambda: self._choose_scope("full"))
        full_layout.addWidget(full_button)
        layout.addWidget(full)
        layout.addStretch(1)

    def _build_parameter_page(self) -> None:
        scroll, layout = self._scroll_page(
            "Choose what may change",
            "This setup step only selects parameter families. Bounds, coil links, and the exact freedoms are defined later on the Freedom page.",
        )
        self.parameter_scroll = scroll
        self.parameter_mode_note = QLabel("")
        self.parameter_mode_note.setWordWrap(True)
        self.parameter_mode_note.setObjectName("sectionHeading")
        layout.addWidget(self.parameter_mode_note)

        simple_entries = (
            ("current", "Current", "Change drive current; all other parameter families stay fixed."),
            ("turns", "Turns", "Change turn count; all other parameter families stay fixed."),
            ("dimensions", "Dimensions", "Change one compatible winding or coil-path dimension selected later."),
            ("position", "Position", "Change one position freedom such as X/Y/Z translation or pair spacing."),
        )
        self.simple_setup_box = QGroupBox("Simple — choose one parameter family")
        simple_layout = QVBoxLayout(self.simple_setup_box)
        simple_note = QLabel(
            "Choose exactly one. The Freedom page will hide every unrelated family and ask only for the links/bounds needed for this choice."
        )
        simple_note.setWordWrap(True)
        simple_note.setObjectName("hint")
        simple_layout.addWidget(simple_note)
        self.simple_choices = self._choice_tree(simple_entries, set())
        self.simple_choices.itemChanged.connect(self._simple_choice_changed)
        self.simple_choices.setMaximumHeight(190)
        simple_layout.addWidget(self.simple_choices)
        layout.addWidget(self.simple_setup_box)

        complex_entries = simple_entries + (
            ("orientation", "Orientation", "Allow rigid assembly rotation and/or explicit in-place coil rotation to change."),
        )
        self.complex_setup_box = QGroupBox("Complex — choose parameter families")
        complex_layout = QVBoxLayout(self.complex_setup_box)
        complex_note = QLabel(
            "Check every family that may change. The Freedom page will expose only those sections and their relationship/bounds controls."
        )
        complex_note.setWordWrap(True)
        complex_note.setObjectName("hint")
        complex_layout.addWidget(complex_note)
        self.complex_choices = self._choice_tree(complex_entries, set())
        self.complex_choices.setMaximumHeight(220)
        complex_layout.addWidget(self.complex_choices)
        layout.addWidget(self.complex_setup_box)
        layout.addStretch(1)
        self._set_parameter_page_mode("simple")

    def _set_parameter_page_mode(self, scope: str) -> None:
        simple = scope == "simple"
        self.simple_setup_box.setVisible(simple)
        self.complex_setup_box.setVisible(not simple)
        if simple:
            self.parameter_mode_note.setText(
                "SIMPLE SETUP — choose the one parameter family this search is allowed to change."
            )
        else:
            self.parameter_mode_note.setText(
                "COMPLEX SETUP — choose every parameter family this search is allowed to change."
            )

    def _build_target_page(self, suggested_target_id: str | None) -> None:
        scroll, layout = self._scroll_page(
            "TARGET — where should the design work?",
            "Select one or more Measurement objects. User-defined sample arrays stay exact; generated "
            "Measurement volumes use Standard sampling. Checked targets are combined into one sample set with equal weight per sample point.",
        )
        self.target_scroll = scroll
        self.target_tree = QTreeWidget()
        self.target_tree.setHeaderLabels(["Use", "Measurement target", "Samples", "Stored target"])
        self.target_tree.setRootIsDecorated(False)
        self.target_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for target in self._target_records.values():
            sample_text = (
                f"{int(target.get('defined_point_count', 0))} defined points"
                if target.get("sampling_mode") == "user_defined"
                else str(target.get("summary", "generated samples"))
            )
            item = QTreeWidgetItem(
                [
                    "",
                    str(target.get("label", target["id"])),
                    sample_text,
                    f"{float(target.get('target_uT', 100.0)):.5g} µT ± "
                    f"{float(target.get('tolerance_pct', 5.0)):.4g}%",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, target["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = target["id"] == suggested_target_id if suggested_target_id else True
            item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.target_tree.addTopLevelItem(item)
        for column in range(self.target_tree.columnCount()):
            self.target_tree.resizeColumnToContents(column)
        layout.addWidget(self.target_tree, 1)

        hint = QLabel(
            "Example: three 96-well Measurement objects with 96 defined points each become one 288-point target. "
            "The optimizer scores the actual points, not the surrounding display boxes."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout.addWidget(hint)

    def _build_goal_page(self, suggested_target_id: str | None) -> None:
        scroll, layout = self._scroll_page(
            "GOAL — what does better mean?",
            "First choose what the optimizer should match. Then choose the engineering criterion that decides which candidate is better. "
            "Workbench explains each criterion instead of exposing raw objective-function names.",
        )
        self.goal_scroll = scroll

        reference_box = QGroupBox("1. What should the optimizer match?")
        reference_form = QFormLayout(reference_box)
        self.goal_reference_mode = QComboBox()
        self.goal_reference_mode.addItem("A target intensity / field specification", "target")
        self.goal_reference_mode.addItem("A stored field snapshot", "snapshot")
        reference_form.addRow("Reference", self.goal_reference_mode)
        layout.addWidget(reference_box)

        reference = self._target_records.get(str(suggested_target_id)) if suggested_target_id else None
        if reference is None and self._target_records:
            reference = next(iter(self._target_records.values()))
        default_target = float(reference.get("target_uT", 100.0)) if reference else 100.0
        default_tolerance = float(reference.get("tolerance_pct", 5.0)) if reference else 5.0

        self.goal_target_box = QGroupBox("Target intensity")
        target_form = QFormLayout(self.goal_target_box)
        self.goal_target_uT = _spin(default_target, 1e-6, 1e9, 4, " µT")
        self.goal_tolerance = _spin(default_tolerance, 0.0, 100.0, 2, " %")
        target_form.addRow("Requested intensity", self.goal_target_uT)
        target_form.addRow("Acceptable band", self.goal_tolerance)
        target_note = QLabel(
            "The band is used by coverage goals and is also reported for every candidate. Other goals may optimize uniformity, "
            "direction, or field strength while still showing how the result relates to this requested intensity."
        )
        target_note.setWordWrap(True)
        target_note.setObjectName("hint")
        target_form.addRow(target_note)
        layout.addWidget(self.goal_target_box)

        self.goal_snapshot_box = QGroupBox("Snapshot reference")
        snapshot_form = QFormLayout(self.goal_snapshot_box)
        self.goal_snapshot = QComboBox()
        snapshots = list(self._snapshot_records.values())
        if snapshots:
            for snapshot in snapshots:
                name = str(snapshot.get("name", snapshot.get("id", "Snapshot")))
                label = str(snapshot.get("source_label", ""))
                status = str(snapshot.get("status", ""))
                suffix = " • ".join(value for value in (label, status) if value)
                self.goal_snapshot.addItem(f"{name}{' — ' + suffix if suffix else ''}", str(snapshot.get("id", "")))
        else:
            self.goal_snapshot.addItem("No stored field snapshots in this scene", "")
        snapshot_form.addRow("Reference snapshot", self.goal_snapshot)
        self.goal_snapshot_magnitude_tolerance = _spin(5.0, 0.0, 100.0, 2, " %")
        self.goal_snapshot_direction_tolerance = _spin(3.0, 0.0, 180.0, 2, "°")
        self.goal_snapshot_tolerance_row = self._row_widget(
            QLabel("Magnitude"), self.goal_snapshot_magnitude_tolerance,
            QLabel("Direction"), self.goal_snapshot_direction_tolerance,
        )
        snapshot_form.addRow("Point-match tolerances", self.goal_snapshot_tolerance_row)
        snapshot_note = QLabel(
            "Snapshot matching is point-for-point. Choose exactly one equivalent Measurement target. Workbench uses the snapshot's stored local sample grid and maps it through the current target's transform, so imported snapshots may come from another scene or world orientation as long as the measurement shape/dimensions are compatible."
        )
        snapshot_note.setWordWrap(True)
        snapshot_note.setObjectName("hint")
        snapshot_form.addRow(snapshot_note)
        layout.addWidget(self.goal_snapshot_box)

        objective_box = QGroupBox("2. How should better be judged?")
        objective_form = QFormLayout(objective_box)
        self.goal_objective = QComboBox()
        objective_form.addRow("Optimize for", self.goal_objective)
        self.goal_snapshot_alignment = QComboBox()
        self.goal_snapshot_alignment.addItem("Keep target-frame orientation fixed", "fixed")
        self.goal_snapshot_alignment.addItem("Exposure registration — free rigid orientation", "exposure")
        self.goal_snapshot_alignment_row = self._row_widget(self.goal_snapshot_alignment)
        self.goal_snapshot_alignment_label = QLabel("Orientation")
        objective_form.addRow(self.goal_snapshot_alignment_label, self.goal_snapshot_alignment_row)
        self.goal_snapshot_alignment_note = QLabel(
            "Fixed orientation compares the exposure in the target's real local axes. Exposure registration treats "
            "absolute orientation as irrelevant: for scoring, Workbench may rigidly rotate the optimizer-participating "
            "exposure sources about the target physical centre, so the spatial field pattern and B-vectors rotate together. "
            "The candidate's real world pose and DRC geometry are not changed, and fixed/background scene sources remain fixed."
        )
        self.goal_snapshot_alignment_note.setWordWrap(True)
        self.goal_snapshot_alignment_note.setObjectName("hint")
        objective_form.addRow(self.goal_snapshot_alignment_note)
        self.goal_objective_explanation = QLabel("")
        self.goal_objective_explanation.setWordWrap(True)
        self.goal_objective_explanation.setObjectName("hint")
        objective_form.addRow(self.goal_objective_explanation)
        layout.addWidget(objective_box)

        self.goal_reference_mode.currentIndexChanged.connect(self._goal_reference_mode_changed)
        self.goal_objective.currentIndexChanged.connect(self._goal_objective_changed)
        self.goal_snapshot_alignment.currentIndexChanged.connect(self._goal_objective_changed)
        self._goal_reference_mode_changed()
        layout.addStretch(1)
        self._goal_wheel_filter = self._install_wheel_filter(scroll, scroll.widget())

    @staticmethod
    def _goal_objective_entries(reference_mode: str) -> tuple[tuple[str, str, str], ...]:
        if str(reference_mode) == "snapshot":
            return (
                (
                    "Match the full field vector at every sample",
                    "snapshot_vector_match",
                    "Minimizes RMS point-by-point difference in Bx, By and Bz together. Use this for the closest overall reproduction of a reference exposure.",
                ),
                (
                    "Match field magnitude at every sample",
                    "snapshot_magnitude_match",
                    "Minimizes RMS point-by-point difference in |B| and ignores which way the field points.",
                ),
                (
                    "Match field direction at every sample",
                    "snapshot_direction_match",
                    "Minimizes the mean angular difference from the reference vectors; field magnitude is secondary.",
                ),
                (
                    "Match the most sample points within tolerances",
                    "snapshot_coverage",
                    "Maximizes the percentage of sample points that simultaneously satisfy the magnitude and direction tolerances above.",
                ),
            )
        return (
            (
                "Put the most samples inside the target band",
                "coverage",
                "Maximizes the fraction of samples that pass the requested intensity ± tolerance. Inside the band, 191 µT and 199 µT are equally acceptable for a 200 µT ±5% target; once coverage ties, the flatter field wins.",
            ),
            (
                "Make field intensity as uniform as possible",
                "uniformity",
                "Minimizes the P95–P5 spread of |B| relative to its mean. Absolute intensity is secondary unless constrained by the target statistics.",
            ),
            (
                "Match the requested intensity as closely as possible",
                "target_match",
                "Minimizes RMS error of |B| from the requested intensity across every sample. Unlike target-band coverage, every amount of error counts: 199 µT is better than 191 µT even if both happen to be inside the allowed band.",
            ),
            (
                "Make field direction as consistent as possible",
                "directional",
                "Maximizes agreement of the field-vector direction across the target, regardless of small intensity differences.",
            ),
            (
                "Make the average field as strong as possible",
                "mean_intensity",
                "Maximizes mean |B| across the target. Use an electrical/current limit if unlimited field strength is not physically meaningful.",
            ),
            (
                "Raise the weakest sampled field as high as possible",
                "minimum_intensity",
                "Maximizes the minimum |B| anywhere in the sampled target — useful when the weakest-exposed location is the limiting case.",
            ),
            (
                "Make the strongest sampled point as high as possible",
                "peak_intensity",
                "Maximizes the maximum |B| found in the target. This is a peak-field goal, not a uniformity goal.",
            ),
        )

    def _goal_reference_mode_changed(self, *args) -> None:
        mode = str(self.goal_reference_mode.currentData() or "target")
        self.goal_target_box.setVisible(mode == "target")
        self.goal_snapshot_box.setVisible(mode == "snapshot")
        self.goal_snapshot_alignment_label.setVisible(mode == "snapshot")
        self.goal_snapshot_alignment_row.setVisible(mode == "snapshot")
        self.goal_snapshot_alignment_note.setVisible(mode == "snapshot")
        previous = str(self.goal_objective.currentData() or "")
        self.goal_objective.blockSignals(True)
        self.goal_objective.clear()
        entries = self._goal_objective_entries(mode)
        for item_index, (label, value, note) in enumerate(entries):
            if mode == "target" and item_index == 4:
                self.goal_objective.insertSeparator(self.goal_objective.count())
            self.goal_objective.addItem(label, value)
            self.goal_objective.setItemData(self.goal_objective.count() - 1, note, Qt.ItemDataRole.ToolTipRole)
        index = self.goal_objective.findData(previous)
        self.goal_objective.setCurrentIndex(index if index >= 0 else 0)
        self.goal_objective.blockSignals(False)
        self._goal_objective_changed()

    def _goal_objective_changed(self, *args) -> None:
        mode = str(self.goal_reference_mode.currentData() or "target")
        value = str(self.goal_objective.currentData() or "")
        explanation = next(
            (note for _label, key, note in self._goal_objective_entries(mode) if key == value),
            "Choose the criterion that should decide which feasible design is better.",
        )
        self.goal_objective_explanation.setText(explanation)
        if hasattr(self, "goal_snapshot_tolerance_row"):
            self.goal_snapshot_tolerance_row.setVisible(
                mode == "snapshot" and value == "snapshot_coverage"
            )

    def _build_freedom_page(self) -> None:
        scroll, layout = self._scroll_page(
            "FREEDOM — define exactly what may change",
            "The first page decides which controls are relevant. Here you define which coils participate, how their properties are linked, "
            "and the numerical bounds. Unselected or unlinked properties stay fixed in the source scene.",
        )
        self.freedom_scroll = scroll

        self.freedom_mode_summary = QLabel("")
        self.freedom_mode_summary.setWordWrap(True)
        self.freedom_mode_summary.setObjectName("sectionHeading")
        layout.addWidget(self.freedom_mode_summary)

        increment_note = QLabel(
            "Each unlocked freedom also has an Increment. Candidate values are snapped to the source scene value plus whole multiples of that increment, while the min/max bounds remain hard limits. This sets the practical search/manufacturing resolution and finalists will not report in-between values. Numeric bounds and increments remember the last values you used; selection checkboxes are never remembered. Use Defaults below to restore the scene-derived numeric values."
        )
        increment_note.setWordWrap(True)
        increment_note.setObjectName("hint")
        layout.addWidget(increment_note)

        coils_box = QGroupBox("Coils and relationship groups")
        coils_layout = QVBoxLayout(coils_box)
        relationship_note = QLabel(
            "A link group means those coils share that optimizer freedom. Example: Current Group A can contain coils 1, 3 and 7, while "
            "Position Group A can independently contain coils 1, 2 and 3. 'Fixed' leaves that property unchanged; 'Independent' gives that coil its own search variable."
        )
        relationship_note.setWordWrap(True)
        relationship_note.setObjectName("hint")
        coils_layout.addWidget(relationship_note)

        relationship_presets = QHBoxLayout()
        self.link_all_button = QPushButton("Link all")
        self.unlink_all_button = QPushButton("Unlink all")
        self.link_by_group_button = QPushButton("Link by group")
        self.link_all_button.setToolTip(
            "Put every participating coil into Group A for every relationship column."
        )
        self.unlink_all_button.setToolTip(
            "Make every participating coil Independent for every relationship column."
        )
        self.link_by_group_button.setToolTip(
            "Mirror the source scene: coils sharing the same immediate scene group are linked; "
            "root-level coils and singleton groups remain Independent."
        )
        self.link_all_button.clicked.connect(
            lambda: self._apply_relationship_preset("all")
        )
        self.unlink_all_button.clicked.connect(
            lambda: self._apply_relationship_preset("independent")
        )
        self.link_by_group_button.clicked.connect(
            lambda: self._apply_relationship_preset("scene_groups")
        )
        relationship_presets.addWidget(self.link_all_button)
        relationship_presets.addWidget(self.unlink_all_button)
        relationship_presets.addWidget(self.link_by_group_button)
        relationship_presets.addStretch(1)
        coils_layout.addLayout(relationship_presets)

        self.freedom_coils = QTreeWidget()
        self.freedom_coils.setHeaderLabels(
            ["Use", "Coil", "Shape", "Current link", "Position link", "Orientation link", "Geometry / turns link"]
        )
        self.freedom_coils.setRootIsDecorated(False)
        self.freedom_coils.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.freedom_link_widgets: dict[str, dict[str, QComboBox]] = {
            "current": {}, "position": {}, "orientation": {}, "geometry": {}
        }
        coils = list(self._coil_records.values())
        link_columns = {"current": 3, "position": 4, "orientation": 5, "geometry": 6}
        for index, coil in enumerate(coils):
            item = QTreeWidgetItem(["", str(coil["label"]), str(coil["shape_label"]), "", "", "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, coil["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if index < 2 else Qt.CheckState.Unchecked)
            self.freedom_coils.addTopLevelItem(item)
            for kind, column in link_columns.items():
                combo = QComboBox()
                combo.addItem("Fixed", "")
                combo.addItem("Independent", "__independent__")
                for letter in "ABCDEFGH":
                    combo.addItem(f"Group {letter}", letter)
                group_a_index = combo.findData("A")
                combo.setCurrentIndex(group_a_index if index < 2 else 0)
                combo.currentIndexChanged.connect(self._freedom_selection_changed)
                # QTreeWidget does not reliably grow rows to the size hint of
                # embedded widgets on every Qt/platform/theme combination.
                # Give link selectors a little vertical breathing room and
                # explicitly advertise that height on the row so text is not
                # clipped (notably on KDE/Linux with compact item metrics).
                row_height = max(28, combo.sizeHint().height() + 4)
                combo.setMinimumHeight(row_height - 2)
                item.setSizeHint(0, QSize(0, row_height))
                self.freedom_link_widgets[kind][str(coil["id"])] = combo
                self.freedom_coils.setItemWidget(item, column, combo)
        for column in range(self.freedom_coils.columnCount()):
            self.freedom_coils.resizeColumnToContents(column)
        self.freedom_coils.itemChanged.connect(self._freedom_selection_changed)
        coils_layout.addWidget(self.freedom_coils)
        layout.addWidget(coils_box)
        # Start in the same relationship mode the guided wizard will restore:
        # mirror scene groups rather than silently bundling the first two coils.
        self._apply_relationship_preset("scene_groups")

        self.current_box = QGroupBox("Current freedom")
        current_form = QFormLayout(self.current_box)
        self.current_active = QCheckBox("Allow current to change")
        self.current_active.setChecked(True)
        current_form.addRow("Freedom", self.current_active)
        self.current_link_mode = QComboBox()
        self.current_link_mode.addItem("Same absolute current; preserve each coil's starting sign", "same_magnitude")
        self.current_link_mode.addItem("Preserve existing signed current ratios", "scale_existing")
        current_form.addRow("Linked-current rule", self.current_link_mode)
        reference_current = self._selected_reference_current_mA()
        current_max_default = max(100.0, 2.5 * reference_current) if reference_current > 0 else 100.0
        current_min_default = 0.25 * reference_current if reference_current > 0 else 0.0
        self.current_min_mA = _spin(current_min_default, 0.0, 1e9, 3, " mA")
        self.current_max_mA = _spin(current_max_default, 0.0, 1e9, 3, " mA")
        self.current_range = self._row_widget(self.current_min_mA, self.current_max_mA)
        current_form.addRow("Current bounds: min / max", self.current_range)
        self.current_increment_mA = _spin(0.1, 0.001, 1e9, 3, " mA")
        self.current_increment_mA.setSingleStep(0.1)
        current_form.addRow("Current increment", self.current_increment_mA)
        current_note = QLabel(
            "Only coils in the active Current link group are varied. Other scene coils still contribute to the field, but their currents remain fixed."
        )
        current_note.setWordWrap(True)
        current_note.setObjectName("hint")
        current_form.addRow(current_note)
        layout.addWidget(self.current_box)

        self.turns_box = QGroupBox("Turn-count freedom")
        turns_form = QFormLayout(self.turns_box)
        self.turns_active = QCheckBox("Allow linked turn count to change")
        turns_form.addRow("Freedom", self.turns_active)
        existing_turns = [int(coil.get("turns", 1)) for coil in coils[:2]] or [100]
        default_turns = max(existing_turns)
        self.turns_min = self._integer(max(1, int(round(default_turns * 0.6))), 1, 100000, " turns")
        self.turns_max = self._integer(max(2, int(round(default_turns * 1.4))), 1, 100000, " turns")
        self.turns_range = self._row_widget(self.turns_min, self.turns_max)
        turns_form.addRow("Turns bounds: min / max", self.turns_range)
        self.turns_increment = self._integer(1, 1, 100000, " turns")
        turns_form.addRow("Turn increment", self.turns_increment)
        turns_note = QLabel(
            "All coils in the active Geometry / turns link group receive the same integer turn count. DRC is rerun because winding construction may change."
        )
        turns_note.setWordWrap(True)
        turns_note.setObjectName("hint")
        turns_form.addRow(turns_note)
        layout.addWidget(self.turns_box)

        self.dimension_box = QGroupBox("Dimension freedom")
        dimension_layout = QVBoxLayout(self.dimension_box)
        self.dimension_active = QCheckBox("Allow dimensions to change")
        dimension_layout.addWidget(self.dimension_active)
        dimension_note = QLabel(
            "Choose the physical dimensions the optimizer may vary. Geometry / turns link groups decide which coils share a value. "
            "Shape-specific rows only apply to compatible coils; unrelated coil types are left fixed."
        )
        dimension_note.setWordWrap(True)
        dimension_note.setObjectName("hint")
        dimension_layout.addWidget(dimension_note)
        self.dimension_compatibility_note = QLabel("")
        self.dimension_compatibility_note.setWordWrap(True)
        self.dimension_compatibility_note.setObjectName("hint")
        dimension_layout.addWidget(self.dimension_compatibility_note)

        dimension_grid = QGridLayout()
        dimension_grid.addWidget(QLabel("Vary"), 0, 0)
        dimension_grid.addWidget(QLabel("Dimension"), 0, 1)
        dimension_grid.addWidget(QLabel("Minimum"), 0, 2)
        dimension_grid.addWidget(QLabel("Maximum"), 0, 3)
        dimension_grid.addWidget(QLabel("Increment"), 0, 4)
        self.dimension_controls: dict[str, dict[str, QWidget]] = {}

        def add_dimension_row(row: int, key: str, label: str, base: float, *, minimum: float = 0.001) -> None:
            check = QCheckBox()
            label_widget = QLabel(label)
            low = _spin(max(minimum, 0.70 * base), minimum, 1e6, 2, " mm")
            high = _spin(max(minimum, 1.30 * base), minimum, 1e6, 2, " mm")
            increment = _spin(0.5, 0.001, 1e6, 3, " mm")
            increment.setSingleStep(0.5)
            dimension_grid.addWidget(check, row, 0, Qt.AlignmentFlag.AlignCenter)
            dimension_grid.addWidget(label_widget, row, 1)
            dimension_grid.addWidget(low, row, 2)
            dimension_grid.addWidget(high, row, 3)
            dimension_grid.addWidget(increment, row, 4)
            self.dimension_controls[key] = {
                "check": check,
                "label": label_widget,
                "min": low,
                "max": high,
                "step": increment,
            }
            check.toggled.connect(self._freedom_selection_changed)

        first_axial = next((float(c.get("axial_winding_length_mm") or 0.0) for c in coils if float(c.get("axial_winding_length_mm") or 0.0) > 0.0), 65.0)
        first_circle = next((float(c.get("circular_diameter_mm") or 0.0) for c in coils if float(c.get("circular_diameter_mm") or 0.0) > 0.0), 100.0)
        first_racetrack = next((coil for coil in coils if coil.get("shape") == "racetrack"), None)
        first_rt_end = float(first_racetrack.get("racetrack_end_diameter_mm", 100.0)) if first_racetrack else 100.0
        first_rt_straight = float(first_racetrack.get("racetrack_straight_length_mm", 100.0)) if first_racetrack else 100.0
        first_square_length = next((float(c.get("square_length_mm") or 0.0) for c in coils if float(c.get("square_length_mm") or 0.0) > 0.0), 100.0)
        first_square_width = next((float(c.get("square_width_mm") or 0.0) for c in coils if float(c.get("square_width_mm") or 0.0) > 0.0), 100.0)
        add_dimension_row(1, "axial_winding_length", "Axial winding length", first_axial)
        add_dimension_row(2, "circular_diameter", "Circular coil diameter", first_circle)
        add_dimension_row(3, "racetrack_end_diameter", "Racetrack corner / end diameter", first_rt_end)
        add_dimension_row(4, "racetrack_straight_length", "Racetrack straight length", first_rt_straight, minimum=0.0)
        add_dimension_row(5, "square_length", "Square / rectangular coil X length", first_square_length)
        add_dimension_row(6, "square_width", "Square / rectangular coil Y width", first_square_width)
        dimension_layout.addLayout(dimension_grid)
        dimension_backend_note = QLabel(
            "Each selected dimension becomes a bounded variable for every compatible Geometry / turns link group. Complex and Full searches may combine several dimensions with current, position, turns, and orientation in one run."
        )
        dimension_backend_note.setWordWrap(True)
        dimension_backend_note.setObjectName("hint")
        dimension_layout.addWidget(dimension_backend_note)
        layout.addWidget(self.dimension_box)

        self.position_box = QGroupBox("Position freedom")
        position_layout = QVBoxLayout(self.position_box)
        self.position_active = QCheckBox("Allow position to change")
        position_layout.addWidget(self.position_active)
        position_note = QLabel(
            "Position link groups define which coils move together. Axis rows are shared translation offsets from the current scene, so linked coils preserve their relative spacing. "
            "The pair-spacing helper instead moves exactly two linked coils symmetrically about their current midpoint."
        )
        position_note.setWordWrap(True)
        position_note.setObjectName("hint")
        position_layout.addWidget(position_note)
        position_grid = QGridLayout()
        position_grid.addWidget(QLabel("Vary"), 0, 0)
        position_grid.addWidget(QLabel("Position freedom"), 0, 1)
        position_grid.addWidget(QLabel("Minimum"), 0, 2)
        position_grid.addWidget(QLabel("Maximum"), 0, 3)
        position_grid.addWidget(QLabel("Increment"), 0, 4)
        self.position_controls: dict[str, dict[str, QWidget]] = {}

        def add_position_row(row: int, key: str, label: str, low_value: float, high_value: float, *, suffix: str = " mm") -> None:
            check = QCheckBox()
            low = _spin(low_value, -1e6, 1e6, 2, suffix)
            high = _spin(high_value, -1e6, 1e6, 2, suffix)
            increment = _spin(0.5, 0.001, 1e6, 3, suffix)
            increment.setSingleStep(0.5)
            position_grid.addWidget(check, row, 0, Qt.AlignmentFlag.AlignCenter)
            position_grid.addWidget(QLabel(label), row, 1)
            position_grid.addWidget(low, row, 2)
            position_grid.addWidget(high, row, 3)
            position_grid.addWidget(increment, row, 4)
            self.position_controls[key] = {"check": check, "min": low, "max": high, "step": increment}
            check.toggled.connect(self._freedom_selection_changed)

        add_position_row(1, "x", "X-axis translation ΔX", -25.0, 25.0)
        add_position_row(2, "y", "Y-axis translation ΔY", -25.0, 25.0)
        add_position_row(3, "z", "Z-axis translation ΔZ", -25.0, 25.0)
        current_spacing = self._selected_pair_spacing_mm(default_if_unavailable=100.0)
        add_position_row(4, "pair_spacing", "Two-coil pair centre spacing", max(0.1, 0.70 * current_spacing), 1.30 * current_spacing)
        position_layout.addLayout(position_grid)
        position_backend_note = QLabel(
            "XYZ rows apply a shared translation offset to the active Position link group. Pair spacing is a two-coil helper that preserves the current midpoint while varying separation."
        )
        position_backend_note.setWordWrap(True)
        position_backend_note.setObjectName("hint")
        position_layout.addWidget(position_backend_note)
        layout.addWidget(self.position_box)

        self.orientation_box = QGroupBox("Orientation freedom")
        orientation_layout = QVBoxLayout(self.orientation_box)
        self.orientation_active = QCheckBox("Allow orientation to change")
        orientation_layout.addWidget(self.orientation_active)
        orientation_note = QLabel(
            "Orientation link groups are shared by two explicitly different freedoms. Rigid assembly rotation moves every linked coil centre around the group's midpoint and rotates every coil plane with it. In-place coil rotation leaves every centre fixed and only tilts/spins the coil itself. Bounds are source-relative world-axis offsets; multiple axes compose in X → Y → Z order."
        )
        orientation_note.setWordWrap(True)
        orientation_note.setObjectName("hint")
        orientation_layout.addWidget(orientation_note)

        self.assembly_orientation_active = QCheckBox(
            "Rigid assembly rotation — move linked coil centres and planes together"
        )
        orientation_layout.addWidget(self.assembly_orientation_active)
        assembly_grid = QGridLayout()
        assembly_grid.addWidget(QLabel("Vary"), 0, 0)
        assembly_grid.addWidget(QLabel("Assembly rotation"), 0, 1)
        assembly_grid.addWidget(QLabel("Minimum"), 0, 2)
        assembly_grid.addWidget(QLabel("Maximum"), 0, 3)
        assembly_grid.addWidget(QLabel("Increment"), 0, 4)
        # Keep the historic attribute name for the primary orientation controls;
        # wizard-created variables now use explicit assembly_orientation_* kinds.
        self.orientation_controls: dict[str, dict[str, QWidget]] = {}
        for row, (key, label) in enumerate((("x", "Rigid rotate about X (ΔRx)"), ("y", "Rigid rotate about Y (ΔRy)"), ("z", "Rigid rotate about Z (ΔRz)")), 1):
            check = QCheckBox()
            low = _spin(-30.0, -360.0, 360.0, 1, "°")
            high = _spin(30.0, -360.0, 360.0, 1, "°")
            increment = _spin(1.0, 0.1, 360.0, 2, "°")
            increment.setSingleStep(0.5)
            assembly_grid.addWidget(check, row, 0, Qt.AlignmentFlag.AlignCenter)
            assembly_grid.addWidget(QLabel(label), row, 1)
            assembly_grid.addWidget(low, row, 2)
            assembly_grid.addWidget(high, row, 3)
            assembly_grid.addWidget(increment, row, 4)
            self.orientation_controls[key] = {"check": check, "min": low, "max": high, "step": increment}
            check.toggled.connect(self._freedom_selection_changed)
        orientation_layout.addLayout(assembly_grid)
        assembly_note = QLabel(
            "For a linked pair this is a true rigid pair pose: rotating the pair also rotates its centre-to-centre axis. Pair spacing is then applied along that rotated axis, while shared Position offsets translate the resulting assembly."
        )
        assembly_note.setWordWrap(True)
        assembly_note.setObjectName("hint")
        orientation_layout.addWidget(assembly_note)

        self.coil_orientation_active = QCheckBox(
            "In-place coil rotation — keep centres fixed"
        )
        orientation_layout.addWidget(self.coil_orientation_active)
        coil_orientation_grid = QGridLayout()
        coil_orientation_grid.addWidget(QLabel("Vary"), 0, 0)
        coil_orientation_grid.addWidget(QLabel("In-place coil rotation"), 0, 1)
        coil_orientation_grid.addWidget(QLabel("Minimum"), 0, 2)
        coil_orientation_grid.addWidget(QLabel("Maximum"), 0, 3)
        coil_orientation_grid.addWidget(QLabel("Increment"), 0, 4)
        self.coil_orientation_controls: dict[str, dict[str, QWidget]] = {}
        for row, (key, label) in enumerate((("x", "Tilt/spin about X (ΔRx)"), ("y", "Tilt/spin about Y (ΔRy)"), ("z", "Tilt/spin about Z (ΔRz)")), 1):
            check = QCheckBox()
            low = _spin(-30.0, -360.0, 360.0, 1, "°")
            high = _spin(30.0, -360.0, 360.0, 1, "°")
            increment = _spin(1.0, 0.1, 360.0, 2, "°")
            increment.setSingleStep(0.5)
            coil_orientation_grid.addWidget(check, row, 0, Qt.AlignmentFlag.AlignCenter)
            coil_orientation_grid.addWidget(QLabel(label), row, 1)
            coil_orientation_grid.addWidget(low, row, 2)
            coil_orientation_grid.addWidget(high, row, 3)
            coil_orientation_grid.addWidget(increment, row, 4)
            self.coil_orientation_controls[key] = {"check": check, "min": low, "max": high, "step": increment}
            check.toggled.connect(self._freedom_selection_changed)
        orientation_layout.addLayout(coil_orientation_grid)
        coil_orientation_note = QLabel(
            "Use in-place rotation only when changing the individual coil planes without moving their centres is physically intended. It is independent of rigid assembly rotation and may be combined with it deliberately."
        )
        coil_orientation_note.setWordWrap(True)
        coil_orientation_note.setObjectName("hint")
        orientation_layout.addWidget(coil_orientation_note)
        layout.addWidget(self.orientation_box)

        for freedom_toggle in (
            self.current_active,
            self.turns_active,
            self.dimension_active,
            self.position_active,
            self.orientation_active,
            self.assembly_orientation_active,
            self.coil_orientation_active,
        ):
            freedom_toggle.toggled.connect(self._freedom_selection_changed)

        self.freedom_summary = QLabel("")
        self.freedom_summary.setWordWrap(True)
        self.freedom_summary.setObjectName("hint")
        layout.addWidget(self.freedom_summary)
        layout.addStretch(1)
        self._freedom_wheel_filter = self._install_wheel_filter(scroll, scroll.widget())
        # Capture the scene-derived values before loading user history.  The
        # Defaults button always returns to these values for this source scene,
        # while only numeric entries (never checkboxes/link selections) persist.
        self._freedom_default_values = {
            key: widget.value() for key, widget in self._freedom_input_widgets().items()
        }
        self._load_freedom_input_history()
        self._apply_scope_to_freedom()
        self._freedom_selection_changed()

    _FREEDOM_HISTORY_PREFIX = "optimizer/freedomLastValues/v1"

    def _freedom_input_widgets(self) -> dict[str, QSpinBox | QDoubleSpinBox]:
        """Return only numeric Freedom inputs that may persist between runs.

        Selection/family/row checkboxes and relationship combo boxes are
        deliberately excluded so each optimization still requires an explicit
        statement of what is allowed to change.
        """
        widgets: dict[str, QSpinBox | QDoubleSpinBox] = {
            "current/min": self.current_min_mA,
            "current/max": self.current_max_mA,
            "current/increment": self.current_increment_mA,
            "turns/min": self.turns_min,
            "turns/max": self.turns_max,
            "turns/increment": self.turns_increment,
        }
        groups = (
            ("dimension", self.dimension_controls),
            ("position", self.position_controls),
            ("assembly_orientation", self.orientation_controls),
            ("coil_orientation", self.coil_orientation_controls),
        )
        for prefix, controls in groups:
            for key, row in controls.items():
                for field, widget_key in (("min", "min"), ("max", "max"), ("increment", "step")):
                    widget = row.get(widget_key)
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        widgets[f"{prefix}/{key}/{field}"] = widget
        return widgets

    def _load_freedom_input_history(self) -> None:
        settings = self._freedom_preferences
        if settings is None:
            return
        for key, widget in self._freedom_input_widgets().items():
            setting_key = f"{self._FREEDOM_HISTORY_PREFIX}/{key}"
            if not settings.contains(setting_key):
                continue
            raw = settings.value(setting_key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(int(round(value)))
            else:
                widget.setValue(value)

    def _save_freedom_input_history(self) -> None:
        settings = self._freedom_preferences
        if settings is None:
            return
        for key, widget in self._freedom_input_widgets().items():
            settings.setValue(
                f"{self._FREEDOM_HISTORY_PREFIX}/{key}",
                widget.value(),
            )
        settings.sync()

    def _restore_freedom_input_defaults(self) -> None:
        defaults = getattr(self, "_freedom_default_values", {})
        for key, widget in self._freedom_input_widgets().items():
            if key not in defaults:
                continue
            value = defaults[key]
            if isinstance(widget, QSpinBox):
                widget.setValue(int(round(float(value))))
            else:
                widget.setValue(float(value))
        self._freedom_selection_changed()

    def _build_limits_page(self, suggested_target_id: str | None) -> None:
        scroll, layout = self._scroll_page(
            "LIMITS — what must never be violated?",
            "Hard limits reject a candidate; they are not soft preferences. Keep the normal Workbench DRC enabled "
            "unless you deliberately want a field-only exploration.",
        )
        self.limits_scroll = scroll
        defaults = self.preview_adapter.optimizer_default_settings(suggested_target_id)

        drc_box = QGroupBox("Physical feasibility")
        drc_form = QFormLayout(drc_box)
        self.use_drc = QCheckBox("Reject candidates that fail the scene's current Check Design rules")
        self.use_drc.setChecked(True)
        drc_form.addRow("Design Rule Check", self.use_drc)
        drc_note = QLabel(
            "This is the ordinary scene DRC: coil-to-coil clearance, exclusions, named surface regions, imported "
            "meshes, and physical winding/bobbin envelopes where construction is enabled."
        )
        drc_note.setWordWrap(True)
        drc_note.setObjectName("hint")
        drc_form.addRow(drc_note)
        layout.addWidget(drc_box)

        electrical_box = QGroupBox("Electrical feasibility")
        electrical_form = QFormLayout(electrical_box)
        self.enforce_electrical = QCheckBox("Enforce electrical hard limits")
        self.enforce_electrical.setChecked(False)
        self.enforce_electrical.toggled.connect(self._limits_toggled)
        electrical_form.addRow("Electrical limits", self.enforce_electrical)
        self.max_current = _spin(defaults["max_current_a"] * 1000.0, 0.001, 1e9, 3, " mA")
        self.max_voltage = _spin(defaults["max_voltage_v"], 0.001, 1e9, 3, " V")
        self.max_power_coil = _spin(defaults["max_power_per_coil_w"], 0.001, 1e9, 3, " W")
        self.max_power_total = _spin(defaults["max_total_power_w"], 0.001, 1e9, 3, " W")
        self.max_current_density = _spin(
            defaults["max_current_density_a_mm2"], 0.001, 1e9, 3, " A/mm²"
        )
        self.electrical_rows = self._row_widget(
            self.max_current,
            self.max_voltage,
            self.max_power_coil,
            self.max_power_total,
            self.max_current_density,
        )
        electrical_form.addRow("I / V / W-coil / W-total / J", self.electrical_rows)
        layout.addWidget(electrical_box)

        convergence_box = QGroupBox("Automatic search convergence")
        convergence_form = QFormLayout(convergence_box)
        self.search_effort = QSlider(Qt.Orientation.Horizontal)
        self.search_effort.setRange(0, 2)
        self.search_effort.setSingleStep(1)
        self.search_effort.setPageStep(1)
        self.search_effort.setTickInterval(1)
        self.search_effort.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.search_effort.setValue(1)
        self.search_effort_label = QLabel()
        self.search_effort_label.setWordWrap(True)
        self.search_effort_label.setObjectName("hint")
        self.search_effort.valueChanged.connect(self._search_effort_changed)
        effort_widget = QWidget()
        effort_layout = QVBoxLayout(effort_widget)
        effort_layout.setContentsMargins(0, 0, 0, 0)
        effort_layout.addWidget(self.search_effort)
        effort_labels = QHBoxLayout()
        effort_labels.addWidget(QLabel("Fast"))
        effort_labels.addStretch(1)
        effort_labels.addWidget(QLabel("Balanced"))
        effort_labels.addStretch(1)
        effort_labels.addWidget(QLabel("Full"))
        effort_layout.addLayout(effort_labels)
        effort_layout.addWidget(self.search_effort_label)
        convergence_form.addRow("Search effort", effort_widget)
        convergence_note = QLabel(
            "All effort levels keep broad exploration in the physics-informed simplified map. Higher effort spends "
            "more work on cheap relation/alignment/symmetry/Sobol coverage. Expensive segmented windings can receive a "
            "reduced-target physical promotion screen before only the strongest survivors reach the authoritative coil model; "
            "it does not buy a larger exploratory full-model search. Pause and Cancel remain available."
        )
        convergence_note.setWordWrap(True)
        convergence_note.setObjectName("hint")
        convergence_form.addRow(convergence_note)
        layout.addWidget(convergence_box)
        layout.addStretch(1)
        self._limits_wheel_filter = self._install_wheel_filter(scroll, scroll.widget())
        self._search_effort_changed(self.search_effort.value())
        self._limits_toggled()

    def _search_effort_changed(self, value: int) -> None:
        descriptions = {
            0: "Fast — compact cheap map, optional reduced-target physical promotion, then the smallest progressive authoritative vet.",
            1: "Balanced — broader cheap map, physical promotion where useful, then a moderate progressive finalist vet.",
            2: "Full — densest cheap map and skeptic coverage, then the broadest promotion/final vet; expensive physics still stays at the end.",
        }
        self.search_effort_label.setText(descriptions.get(int(value), descriptions[1]))

    def _build_review_page(self) -> None:
        scroll, layout = self._scroll_page(
            "Review the engineering question",
            "Workbench will run only what is written below. If the summary sounds wrong, go Back and change the "
            "target, goal, freedom, or limits before spending computation time.",
        )
        self.review_scroll = scroll
        self.review_text = QPlainTextEdit()
        self.review_text.setReadOnly(True)
        self.review_text.setMinimumHeight(360)
        layout.addWidget(self.review_text, 1)
        review_note = QLabel(
            "The source scene is never edited during the search. On Results, a finalist can either be opened as "
            "a separate unsaved scene tab or applied back to the source scene as one undoable edit."
        )
        review_note.setWordWrap(True)
        review_note.setObjectName("hint")
        layout.addWidget(review_note)

    @staticmethod
    def _optimizer_comparison_format(value: Any, unit: str = "", digits: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(number):
            return "—"
        return f"{number:,.{digits}g}{(' ' + unit) if unit else ''}"

    @staticmethod
    def _optimizer_comparison_quality_colour(
        value_pct: Any, *, higher_is_better: bool
    ) -> str:
        """Use the same red → amber → green scale as Snapshot Comparison."""
        try:
            value = float(value_pct)
        except (TypeError, ValueError):
            value = 50.0
        if not math.isfinite(value):
            value = 50.0
        value = min(100.0, max(0.0, value))
        quality = value if higher_is_better else 100.0 - value
        red = (220, 38, 38)
        amber = (245, 158, 11)
        green = (22, 163, 74)
        if quality <= 50.0:
            fraction = quality / 50.0
            start, stop = red, amber
        else:
            fraction = (quality - 50.0) / 50.0
            start, stop = amber, green
        rgb = tuple(
            int(round(a + (b - a) * fraction)) for a, b in zip(start, stop)
        )
        return "#%02x%02x%02x" % rgb

    def _optimizer_comparison_card(
        self, title: str, value: str, detail: str, *, colour: str | None = None,
        prominent: bool = False,
    ) -> QGroupBox:
        """Create the same compact metric-card language used by Compare."""
        card = QGroupBox(title)
        layout = QVBoxLayout(card)
        number = _selectable_label(value, word_wrap=True)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setObjectName("resultValue")
        layout.addWidget(number)
        note = _selectable_label(detail, word_wrap=True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setObjectName("hint")
        layout.addWidget(note)
        if prominent:
            card.setMinimumHeight(132)
        if colour:
            width = 3 if prominent else 2
            card.setStyleSheet(f"QGroupBox {{ border: {width}px solid {colour}; }}")
            if prominent:
                number.setStyleSheet(
                    f"font-size: 24pt; font-weight: 700; color: {colour};"
                )
        return card

    def _optimizer_comparison_metric_card(
        self,
        title: str,
        display_value: str,
        detail: str,
        quality_value_pct: Any,
        *,
        higher_is_better: bool,
    ) -> QGroupBox:
        colour = self._optimizer_comparison_quality_colour(
            quality_value_pct, higher_is_better=higher_is_better
        )
        return self._optimizer_comparison_card(
            title, display_value, detail, colour=colour
        )

    @staticmethod
    def _optimizer_match_overall_score(stats: dict[str, Any]) -> float:
        def clipped(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            return min(100.0, max(0.0, number))

        def inverse_error(value: Any, *, absolute: bool = False) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            if absolute:
                number = abs(number)
            return 100.0 - min(100.0, max(0.0, number))

        return (
            inverse_error(stats.get("mean_intensity_bias_pct"), absolute=True)
            + inverse_error(stats.get("relative_rms_vector_difference_pct"))
            + clipped(stats.get("match_coverage_pct"))
        ) / 3.0

    @staticmethod
    def _optimizer_benchmark_overall_score(entry: dict[str, Any]) -> float:
        def clipped(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            return min(100.0, max(0.0, number))

        return (
            clipped(entry.get("target_coverage_pct"))
            + (100.0 - clipped(entry.get("uniformity_spread_pct")))
            + clipped(entry.get("directional_consistency_pct"))
        ) / 3.0

    def _optimizer_comparison_columns(self, columns: list[QWidget]) -> QWidget:
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for column in columns:
            column.setMinimumWidth(285)
            column.setMaximumWidth(340)
            layout.addWidget(column)
        layout.addStretch(1)
        return content

    @staticmethod
    def _replace_optimizer_scroll_content(scroll: QScrollArea, widget: QWidget) -> None:
        previous = scroll.takeWidget()
        if previous is not None:
            previous.deleteLater()
        scroll.setWidget(widget)

    def _set_optimizer_comparison_placeholder(self, title: str, detail: str) -> None:
        def page() -> QWidget:
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.addStretch(1)
            heading = QLabel(title)
            heading.setObjectName("sectionHeading")
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(heading)
            note = QLabel(detail)
            note.setWordWrap(True)
            note.setObjectName("hint")
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(note)
            layout.addStretch(1)
            return content

        if hasattr(self, "comparison_overview"):
            self._replace_optimizer_scroll_content(self.comparison_overview, page())
        if hasattr(self, "comparison_statistics"):
            self._replace_optimizer_scroll_content(self.comparison_statistics, page())

    def _optimizer_comparison_current_summary(
        self, evaluation: dict[str, Any], candidate: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        currents = dict(evaluation.get("currents_a", {}) or {})
        if not currents and isinstance(candidate, dict):
            currents = dict(candidate.get("currents_a", {}) or {})
        if not currents:
            return "—", "No solved current was retained at this fidelity"
        values_mA = [1000.0 * float(value) for value in currents.values()]
        rounded = {round(value, 9) for value in values_mA}
        if len(rounded) == 1:
            value = values_mA[0]
            detail = f"Shared by {len(values_mA)} coil" + ("s" if len(values_mA) != 1 else "")
            return f"{value:.6g} mA", detail
        text = ", ".join(f"{value:.5g}" for value in values_mA[:4])
        if len(values_mA) > 4:
            text += f", +{len(values_mA) - 4} more"
        return "Independent currents", text + " mA"

    def _populate_optimizer_comparison_summary(
        self,
        *,
        solution_id: str | None,
        evaluation: dict[str, Any],
        record: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
    ) -> None:
        """Render Compare-style results without materializing a candidate scene.

        Registry browsing should be cheap.  This view therefore consumes the
        candidate's already-retained highest trustworthy fidelity metrics.  A full
        replay/field calculation is still available explicitly through Compare,
        Save snapshot, or Open in new tab.
        """
        if self.run_result is None:
            self._set_optimizer_comparison_placeholder(
                "No completed result", "Run an optimization to inspect comparison results."
            )
            return
        metrics = dict(evaluation.get("metrics", {}) or {})
        if not metrics and isinstance(candidate, dict):
            metrics = dict(candidate.get("metrics", {}) or {})
        if not metrics:
            self._set_optimizer_comparison_placeholder(
                "Comparison metrics unavailable",
                "This retained solution has no usable field metrics at its current fidelity.",
            )
            return

        label = (
            self._registry_solution_label(solution_id)
            if solution_id
            else str((candidate or {}).get("id", "Optimizer solution"))
        )
        fidelity = str(evaluation.get("fidelity", "") or (candidate or {}).get("registry_fidelity", "") or "retained")
        sample_count = int(
            evaluation.get("sample_count")
            or metrics.get("valid_sample_count")
            or metrics.get("sample_count")
            or self.run_result.get("final_sample_count", 0)
            or 0
        )
        current_value, current_detail = self._optimizer_comparison_current_summary(
            evaluation, candidate
        )
        drc = dict((record or {}).get("drc", {}) or {})
        clearance = drc.get("minimum_clearance_mm", metrics.get("minimum_clearance_mm"))
        drc_status = str(drc.get("status", "clear" if clearance is not None else "unknown"))
        snapshot_mode = str(self.run_result.get("reference_mode", "target")) == "snapshot"

        overview_columns: list[QWidget] = []
        statistics_rows: list[tuple[str, str]] = [
            ("Fidelity", fidelity.replace("_", " ").title()),
            ("Samples", f"{sample_count:,}" if sample_count else "—"),
            ("DRC status", drc_status.capitalize()),
            ("Minimum DRC clearance", self._optimizer_comparison_format(clearance, "mm")),
        ]

        if snapshot_mode:
            reference = dict(self.run_result.get("reference_snapshot", {}) or {})
            mag_tol = float(reference.get("magnitude_tolerance_pct", 5.0) or 5.0)
            dir_tol = float(reference.get("direction_tolerance_deg", 3.0) or 3.0)
            reference_mean = metrics.get("snapshot_reference_mean_uT")
            if reference_mean is None:
                reference_mean = (reference.get("stats", {}) or {}).get("mean_uT")
            reference_rms = math.nan
            reference_uniformity = math.nan
            # Reading the frozen reference snapshot is cheap (no field solve) and
            # lets this card use the exact same relative-RMS denominator as the
            # normal Snapshot Comparison window: RMS reference magnitude, not the
            # mean field.
            try:
                reference_id = str(reference.get("id", ""))
                if reference_id:
                    reference_result = self.preview_adapter.snapshot_result(reference_id)
                    reference_vectors_uT = (
                        self.preview_adapter._vectors_in_measurement_frame(reference_result) * 1.0e6
                    )
                    finite_reference = np.isfinite(reference_vectors_uT).all(axis=1)
                    if np.any(finite_reference):
                        reference_magnitudes = np.linalg.norm(
                            reference_vectors_uT[finite_reference], axis=1
                        )
                        if reference_mean is None:
                            reference_mean = float(np.mean(reference_magnitudes))
                        reference_rms = float(
                            np.sqrt(np.mean(reference_magnitudes**2))
                        )
                        reference_uniformity = float(
                            StudioAdapter._uniformity_spread(reference_magnitudes)
                        )
            except Exception:  # noqa: BLE001 - lightweight display fallback only
                pass
            candidate_mean = metrics.get("snapshot_candidate_mean_uT", metrics.get("mean_uT"))
            try:
                reference_mean_number = float(reference_mean)
                candidate_mean_number = float(candidate_mean)
                bias_pct = (
                    100.0 * (candidate_mean_number - reference_mean_number) / reference_mean_number
                    if math.isfinite(reference_mean_number) and abs(reference_mean_number) > 1e-15
                    else math.nan
                )
            except (TypeError, ValueError):
                reference_mean_number = math.nan
                candidate_mean_number = math.nan
                bias_pct = math.nan
            vector_rms = metrics.get("snapshot_vector_rms_uT")
            try:
                relative_vector_rms = (
                    100.0 * float(vector_rms) / abs(reference_rms)
                    if math.isfinite(reference_rms) and abs(reference_rms) > 1e-15
                    else math.nan
                )
            except (TypeError, ValueError):
                relative_vector_rms = math.nan
            coverage = metrics.get("snapshot_match_coverage_pct", math.nan)
            match_stats = {
                "mean_intensity_bias_pct": bias_pct,
                "relative_rms_vector_difference_pct": relative_vector_rms,
                "match_coverage_pct": coverage,
            }

            reference_box = QGroupBox(str(reference.get("name", reference.get("id", "Reference exposure"))))
            reference_layout = QVBoxLayout(reference_box)
            reference_layout.addWidget(QLabel("REFERENCE EXPOSURE"))
            source = QLabel(str(reference.get("source_label", reference.get("label", "Stored snapshot"))))
            source.setWordWrap(True)
            source.setObjectName("hint")
            reference_layout.addWidget(source)
            reference_layout.addWidget(
                self._optimizer_comparison_card(
                    "Mean intensity",
                    self._optimizer_comparison_format(reference_mean_number, "µT"),
                    "Reference mean intensity",
                    colour="#16a34a",
                )
            )
            reference_layout.addWidget(
                self._optimizer_comparison_card(
                    "Match tolerances",
                    f"±{self._optimizer_comparison_format(mag_tol, '%')}",
                    f"Direction ≤ {self._optimizer_comparison_format(dir_tol, '°')}",
                )
            )
            reference_layout.addStretch(1)
            overview_columns.append(reference_box)

            candidate_box = QGroupBox(label)
            candidate_layout = QVBoxLayout(candidate_box)
            source = QLabel(
                f"Optimizer {fidelity.replace('_', ' ')} metrics • no 3-D scene rebuild"
            )
            source.setWordWrap(True)
            source.setObjectName("hint")
            candidate_layout.addWidget(source)
            bias_prefix = "+" if math.isfinite(bias_pct) and bias_pct > 0 else ""
            candidate_layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Mean intensity",
                    self._optimizer_comparison_format(candidate_mean_number, "µT"),
                    f"{bias_prefix}{self._optimizer_comparison_format(bias_pct, '%')} vs reference",
                    abs(bias_pct) if math.isfinite(bias_pct) else math.nan,
                    higher_is_better=False,
                )
            )
            candidate_layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Relative vector RMS error",
                    self._optimizer_comparison_format(relative_vector_rms, "%"),
                    f"{self._optimizer_comparison_format(vector_rms, 'µT')} vector RMS",
                    relative_vector_rms,
                    higher_is_better=False,
                )
            )
            candidate_layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Points matching",
                    self._optimizer_comparison_format(coverage, "%"),
                    f"Magnitude ±{self._optimizer_comparison_format(mag_tol, '%')} and direction ≤ {self._optimizer_comparison_format(dir_tol, '°')}",
                    coverage,
                    higher_is_better=True,
                )
            )
            score = self._optimizer_match_overall_score(match_stats)
            score_colour = self._optimizer_comparison_quality_colour(score, higher_is_better=True)
            candidate_layout.addWidget(
                self._optimizer_comparison_card(
                    "Overall score",
                    self._optimizer_comparison_format(score, "%"),
                    "Average of mean-intensity match, inverse vector RMS error, and points matching",
                    colour=score_colour,
                    prominent=True,
                )
            )
            candidate_layout.addWidget(
                self._optimizer_comparison_card("Current", current_value, current_detail)
            )
            candidate_layout.addStretch(1)
            overview_columns.append(candidate_box)

            statistics_rows.extend(
                [
                    ("Reference mean", self._optimizer_comparison_format(reference_mean_number, "µT")),
                    ("Reference uniformity", self._optimizer_comparison_format(reference_uniformity, "%")),
                    ("Candidate mean", self._optimizer_comparison_format(candidate_mean_number, "µT")),
                    ("Mean intensity bias", self._optimizer_comparison_format(bias_pct, "%")),
                    ("Relative vector RMS error", self._optimizer_comparison_format(relative_vector_rms, "%")),
                    ("RMS vector difference", self._optimizer_comparison_format(vector_rms, "µT")),
                    ("RMS magnitude difference", self._optimizer_comparison_format(metrics.get("snapshot_magnitude_rms_uT"), "µT")),
                    ("Matching points", self._optimizer_comparison_format(coverage, "%")),
                    ("Mean angular difference", self._optimizer_comparison_format(metrics.get("snapshot_mean_angle_deg"), "°")),
                    ("P95 angular difference", self._optimizer_comparison_format(metrics.get("snapshot_p95_angle_deg"), "°")),
                    ("Candidate uniformity", self._optimizer_comparison_format(metrics.get("uniformity_spread_pct"), "%")),
                    ("Directional consistency", self._optimizer_comparison_format(metrics.get("directional_consistency_pct"), "%")),
                ]
            )
            alignment_mode = str(reference.get("alignment_mode", "fixed") or "fixed")
            if alignment_mode == "exposure":
                statistics_rows.extend(
                    [
                        ("Exposure registration RX", self._optimizer_comparison_format(metrics.get("snapshot_alignment_rx_deg"), "°")),
                        ("Exposure registration RY", self._optimizer_comparison_format(metrics.get("snapshot_alignment_ry_deg"), "°")),
                        ("Exposure registration RZ", self._optimizer_comparison_format(metrics.get("snapshot_alignment_rz_deg"), "°")),
                    ]
                )
        else:
            target = dict(self.run_result.get("target", {}) or {})
            target_uT = target.get("target_uT", metrics.get("target_uT"))
            tolerance_pct = target.get("tolerance_pct", metrics.get("tolerance_pct"))
            entry = {
                "target_coverage_pct": metrics.get("in_band_pct", math.nan),
                "uniformity_spread_pct": metrics.get("uniformity_spread_pct", math.nan),
                "directional_consistency_pct": metrics.get("directional_consistency_pct", math.nan),
                "mean_uT": metrics.get("mean_uT", math.nan),
            }
            column = QGroupBox(label)
            layout = QVBoxLayout(column)
            source = QLabel(
                f"Optimizer {fidelity.replace('_', ' ')} metrics • no 3-D scene rebuild"
            )
            source.setWordWrap(True)
            source.setObjectName("hint")
            layout.addWidget(source)
            layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Target coverage",
                    self._optimizer_comparison_format(entry["target_coverage_pct"], "%"),
                    f"{self._optimizer_comparison_format(target_uT, 'µT')} ± {self._optimizer_comparison_format(tolerance_pct, '%')}",
                    entry["target_coverage_pct"],
                    higher_is_better=True,
                )
            )
            layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Uniformity spread",
                    self._optimizer_comparison_format(entry["uniformity_spread_pct"], "%"),
                    "100 × (P95 − P5) / mean; lower is better",
                    entry["uniformity_spread_pct"],
                    higher_is_better=False,
                )
            )
            layout.addWidget(
                self._optimizer_comparison_metric_card(
                    "Directional consistency",
                    self._optimizer_comparison_format(entry["directional_consistency_pct"], "%"),
                    f"Mean field {self._optimizer_comparison_format(entry['mean_uT'], 'µT')}",
                    entry["directional_consistency_pct"],
                    higher_is_better=True,
                )
            )
            score = self._optimizer_benchmark_overall_score(entry)
            score_colour = self._optimizer_comparison_quality_colour(score, higher_is_better=True)
            layout.addSpacing(12)
            layout.addWidget(
                self._optimizer_comparison_card(
                    "Overall score",
                    self._optimizer_comparison_format(score, "%"),
                    "Average of target coverage, inverse uniformity spread, and directional consistency",
                    colour=score_colour,
                    prominent=True,
                )
            )
            layout.addWidget(self._optimizer_comparison_card("Current", current_value, current_detail))
            layout.addStretch(1)
            overview_columns.append(column)
            statistics_rows.extend(
                [
                    ("Mean", self._optimizer_comparison_format(metrics.get("mean_uT"), "µT")),
                    ("Median", self._optimizer_comparison_format(metrics.get("median_uT"), "µT")),
                    ("Minimum", self._optimizer_comparison_format(metrics.get("min_uT"), "µT")),
                    ("Maximum", self._optimizer_comparison_format(metrics.get("max_uT"), "µT")),
                    ("Target coverage", self._optimizer_comparison_format(metrics.get("in_band_pct"), "%")),
                    ("Below target", self._optimizer_comparison_format(metrics.get("below_target_pct"), "%")),
                    ("Above target", self._optimizer_comparison_format(metrics.get("above_target_pct"), "%")),
                    ("RMS target error", self._optimizer_comparison_format(metrics.get("rms_target_error_uT"), "µT")),
                    ("Uniformity spread", self._optimizer_comparison_format(metrics.get("uniformity_spread_pct"), "%")),
                    ("Directional consistency", self._optimizer_comparison_format(metrics.get("directional_consistency_pct"), "%")),
                ]
            )

        overview = self._optimizer_comparison_columns(overview_columns)
        self._replace_optimizer_scroll_content(self.comparison_overview, overview)

        statistics = QWidget()
        statistics_layout = QVBoxLayout(statistics)
        statistics_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        statistics_layout.addWidget(_copyable_statistics_group(label, statistics_rows))
        statistics_layout.addWidget(
            self._optimizer_comparison_card("Current", current_value, current_detail)
        )
        note = QLabel(
            "These are the optimizer registry's highest trustworthy retained metrics. "
            "Use Compare selected… for a fresh full-field comparison with difference plots, "
            "or Open in new tab to inspect the physical 3-D scene."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        statistics_layout.addWidget(note)
        statistics_layout.addStretch(1)
        self._replace_optimizer_scroll_content(self.comparison_statistics, statistics)
        _make_result_text_selectable(self.comparison_overview)
        _make_result_text_selectable(self.comparison_statistics)

    def _build_results_page(self) -> None:
        # Completed optimization results are an explorer over the persistent flat
        # candidate registry.  The left list selects a semantic view; the right
        # side shows those same canonical solutions and previews the selected one.
        page = QWidget()
        self.results_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)
        self.results_heading = QLabel("RESULTS — Balanced search")
        self.results_heading.setObjectName("dialogHeading")
        layout.addWidget(self.results_heading)
        self.results_target_info = QLabel("Target: choose a Measurement target before starting the search.")
        self.results_target_info.setWordWrap(True)
        self.results_target_info.setObjectName("hint")
        layout.addWidget(self.results_target_info)
        self.result_summary = QLabel("Run the reviewed optimization plan to create candidates.")
        self.result_summary.setWordWrap(True)
        layout.addWidget(self.result_summary)

        self.result_view_panel = QWidget()
        view_layout = QVBoxLayout(self.result_view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(4)
        view_heading = QLabel("Result views")
        view_heading.setObjectName("sectionHeading")
        view_layout.addWidget(view_heading)
        self.result_view_list = QListWidget()
        self.result_view_list.setMinimumWidth(175)
        self.result_view_list.setMaximumWidth(245)
        self.result_view_list.currentItemChanged.connect(self._result_registry_view_changed)
        view_layout.addWidget(self.result_view_list, 1)
        view_note = QLabel("Views are different rankings of the same retained solution registry.")
        view_note.setWordWrap(True)
        view_note.setObjectName("hint")
        view_layout.addWidget(view_note)

        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(
            [
                "Solution ID",
                "Match / coverage",
                "RMS error",
                "Angle / direction",
                "Mean |B|",
                "DRC clearance",
                "Fidelity",
            ]
        )
        self.result_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.result_tree.setRootIsDecorated(True)
        self.result_tree.setIndentation(18)
        self.result_tree.setMinimumHeight(112)
        self.result_tree.itemSelectionChanged.connect(self._result_selection_changed)
        self.result_tree.itemExpanded.connect(self._result_tree_item_expanded)

        self.candidate_detail = QLabel("")
        self.candidate_detail.setWordWrap(True)
        self.candidate_detail.setObjectName("hint")

        self.result_views = QTabWidget()
        # Candidate browsing is intentionally data-only.  Rebuilding a complete
        # Workbench scene/Plotly 3-D figure for every registry selection made the
        # completed result browser feel much heavier than the optimizer itself.
        # These two lightweight pages mirror the normal Field Snapshot Comparison
        # overview/statistics presentation from the metrics already retained by the
        # candidate registry.  Opening an actual 3-D scene is an explicit action via
        # "Open in new tab".
        self.comparison_overview = QScrollArea()
        self.comparison_overview.setWidgetResizable(True)
        self.comparison_statistics = QScrollArea()
        self.comparison_statistics.setWidgetResizable(True)
        # Compatibility aliases for older UI harnesses that only assert which
        # result tab is current.  They are no longer PlotView instances.
        self.candidate_plot = self.comparison_overview
        self.slice_plot = self.comparison_statistics
        self.sweep_plot = PlotView(image_filename="optimizer-search-trace")
        self._set_optimizer_comparison_placeholder(
            "No solution selected",
            "Select a solution from a result view to inspect its comparison metrics.",
        )
        self.result_views.addTab(self.comparison_overview, "Comparison")
        self.result_views.addTab(self.comparison_statistics, "Full statistics")
        self.result_views.addTab(self.sweep_plot, "Search history")
        self.sweep_plot.show_message(
            "Search history",
            "The live optimizer stage, candidate counts, and best-so-far result will appear here while a search runs.",
        )

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addWidget(self.candidate_detail)
        preview_layout.addWidget(self.result_views, 1)

        self.result_splitter = QSplitter(Qt.Orientation.Vertical)
        self.result_splitter.setChildrenCollapsible(False)
        self.result_splitter.setHandleWidth(6)
        self.result_splitter.addWidget(self.result_tree)
        self.result_splitter.addWidget(preview_container)
        self.result_splitter.setStretchFactor(0, 0)
        self.result_splitter.setStretchFactor(1, 1)
        self.result_splitter.setSizes([160, 470])

        self.result_browser_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.result_browser_splitter.setChildrenCollapsible(False)
        self.result_browser_splitter.setHandleWidth(6)
        self.result_browser_splitter.addWidget(self.result_view_panel)
        self.result_browser_splitter.addWidget(self.result_splitter)
        self.result_browser_splitter.setStretchFactor(0, 0)
        self.result_browser_splitter.setStretchFactor(1, 1)
        self.result_browser_splitter.setSizes([205, 920])
        layout.addWidget(self.result_browser_splitter, 2)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("Copy report")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_report)
        actions.addWidget(self.copy_button)
        self.compare_button = QPushButton("Compare selected…")
        self.compare_button.setToolTip(
            "Run a fresh full-field Field Snapshot Comparison for the selected optimizer solution(s), including detailed difference plots."
        )
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._compare_selected)
        actions.addWidget(self.compare_button)
        self.save_snapshot_button = QPushButton("Save snapshot")
        self.save_snapshot_button.setToolTip(
            "Analyze this candidate and save its frozen exposure into the optimizer source scene without keeping the candidate model."
        )
        self.save_snapshot_button.setEnabled(False)
        self.save_snapshot_button.clicked.connect(self._save_selected_snapshot)
        actions.addWidget(self.save_snapshot_button)
        actions.addStretch(1)
        self.open_tab_button = QPushButton("Open in new tab")
        self.open_tab_button.setEnabled(False)
        self.open_tab_button.clicked.connect(self._open_selected_new_tab)
        actions.addWidget(self.open_tab_button)
        self.append_button = QPushButton("Append to source")
        self.append_button.setToolTip(
            "Copy only participating coils whose geometry or design changed; current-only/unchanged source coils are not duplicated."
        )
        self.append_button.setEnabled(False)
        self.append_button.clicked.connect(self._append_selected)
        actions.addWidget(self.append_button)
        self.apply_button = QPushButton("Apply to source")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_selected)
        actions.addWidget(self.apply_button)
        layout.addLayout(actions)
        self._set_optimizer_result_tab_mode(searching=True)
        self.pages.addWidget(page)

    @staticmethod
    def _checked_tree_ids(tree: QTreeWidget) -> list[str]:
        result: list[str] = []
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                value = item.data(0, Qt.ItemDataRole.UserRole)
                if value is not None:
                    result.append(str(value))
        return result

    def _linked_groups(self, kind: str) -> dict[str, list[str]]:
        selected = set(self._checked_tree_ids(self.freedom_coils)) if hasattr(self, "freedom_coils") else set()
        groups: dict[str, list[str]] = {}
        for object_id, combo in getattr(self, "freedom_link_widgets", {}).get(kind, {}).items():
            if object_id not in selected:
                continue
            group = str(combo.currentData() or "")
            if group == "__independent__":
                group = f"I:{object_id}"
            if group:
                groups.setdefault(group, []).append(object_id)
        return groups

    def _single_link_group_ids(self, kind: str) -> list[str]:
        groups = self._linked_groups(kind)
        if len(groups) != 1:
            return []
        return next(iter(groups.values()))

    def _relationship_scene_group_assignments(self, coil_ids: list[str]) -> dict[str, str]:
        """Return optimizer link assignments that mirror immediate scene groups.

        Only scene groups containing at least two participating coils consume a
        shared optimizer group letter.  A root-level coil, a coil whose scene
        group contributes only one participating member, or a group beyond the
        eight A-H slots remains Independent.
        """
        ordered_ids = [str(value) for value in coil_ids if str(value) in self._coil_records]
        scene_groups: dict[str, list[str]] = {}
        scene_group_order: list[str] = []
        for object_id in ordered_ids:
            group_path = self._coil_records.get(object_id, {}).get("group_path", [])
            if not isinstance(group_path, list) or not group_path:
                continue
            immediate = group_path[-1]
            if not isinstance(immediate, dict):
                continue
            group_id = str(immediate.get("id", "")).strip()
            if not group_id:
                continue
            if group_id not in scene_groups:
                scene_groups[group_id] = []
                scene_group_order.append(group_id)
            scene_groups[group_id].append(object_id)

        assignments = {object_id: "__independent__" for object_id in ordered_ids}
        letters = iter("ABCDEFGH")
        for group_id in scene_group_order:
            members = scene_groups.get(group_id, [])
            if len(members) < 2:
                continue
            letter = next(letters, None)
            if letter is None:
                break
            for object_id in members:
                assignments[object_id] = letter
        return assignments

    def _apply_relationship_preset(self, preset: str) -> None:
        """Bulk-edit relationship columns without changing participating coils."""
        if not hasattr(self, "freedom_coils") or not hasattr(self, "freedom_link_widgets"):
            return
        participating = self._checked_tree_ids(self.freedom_coils)
        participating_set = set(participating)
        preset = str(preset).strip().lower()
        if preset == "all":
            assignments = {object_id: "A" for object_id in participating}
        elif preset == "independent":
            assignments = {object_id: "__independent__" for object_id in participating}
        elif preset == "scene_groups":
            assignments = self._relationship_scene_group_assignments(participating)
        else:
            return

        for kind_widgets in self.freedom_link_widgets.values():
            for object_id, combo in kind_widgets.items():
                desired = assignments.get(object_id, "") if object_id in participating_set else ""
                index = combo.findData(desired)
                if index < 0:
                    index = 0
                combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(index)
                finally:
                    combo.blockSignals(False)
        self._freedom_selection_changed()

    def _optimizer_group_label(self, group: str, coil_ids: list[str]) -> str:
        if str(group).startswith("I:") and coil_ids:
            object_id = str(coil_ids[0])
            return f"Independent {self._coil_records.get(object_id, {}).get('label', object_id)}"
        return f"Group {group}"

    def _selected_pair_spacing_mm(self, *, default_if_unavailable: float = 100.0) -> float:
        ids = self._single_link_group_ids("position") if hasattr(self, "freedom_link_widgets") else []
        if not ids:
            ids = self._checked_tree_ids(self.freedom_coils) if hasattr(self, "freedom_coils") else list(self._coil_records)[:2]
        if len(ids) != 2:
            return float(default_if_unavailable)
        first = np.asarray(self._coil_records[ids[0]]["position_mm"], dtype=float)
        second = np.asarray(self._coil_records[ids[1]]["position_mm"], dtype=float)
        spacing = float(np.linalg.norm(second - first))
        return spacing if math.isfinite(spacing) and spacing > 0 else float(default_if_unavailable)

    def _selected_reference_current_mA(self) -> float:
        ids = self._single_link_group_ids("current") if hasattr(self, "freedom_link_widgets") else []
        if not ids:
            ids = self._checked_tree_ids(self.freedom_coils) if hasattr(self, "freedom_coils") else list(self._coil_records)[:2]
        currents = [abs(float(self._coil_records[value]["drive_current_mA"])) for value in ids if value in self._coil_records]
        return max(currents, default=0.0)

    def _choose_scope(self, scope: str) -> None:
        scope = str(scope).strip().lower()
        if scope in {"simple", "complex"}:
            self.search_scope = scope
            self.requested_freedoms = set()
            self._prepare_guided_freedom_defaults()
            self._set_parameter_page_mode(scope)
            self.pages.setCurrentIndex(self.PAGE_PARAMETERS)
            return
        if scope != "full":
            return

        self.search_scope = "full"
        self.requested_freedoms = {"current", "turns", "dimensions", "position", "orientation"}
        self._clear_full_freedom_presets()
        self._apply_scope_to_freedom()
        self.pages.setCurrentIndex(self.PAGE_TARGET)

    def _commit_parameter_setup(self) -> str | None:
        if self.search_scope == "simple":
            choices = self._choice_tree_values(self.simple_choices)
            if len(choices) != 1:
                return "Simple setup needs exactly one checked parameter family."
            self.requested_freedoms = set(choices)
        elif self.search_scope == "complex":
            choices = self._choice_tree_values(self.complex_choices)
            if not choices:
                return "Complex setup needs at least one checked parameter family."
            self.requested_freedoms = set(choices)
        else:
            return "Choose Simple, Complex, or Skip wizard before continuing."
        self._apply_scope_to_freedom()
        return None

    def _prepare_guided_freedom_defaults(self) -> None:
        # Guided Simple/Complex routes start from the first two coils and mirror
        # the source scene's immediate Collection grouping.  Users can override
        # that relationship in one click with Link all / Unlink all / Link by group.
        for index in range(self.freedom_coils.topLevelItemCount()):
            item = self.freedom_coils.topLevelItem(index)
            use = index < 2
            item.setCheckState(0, Qt.CheckState.Checked if use else Qt.CheckState.Unchecked)
        self._apply_relationship_preset("scene_groups")
        for controls in (
            self.dimension_controls,
            self.position_controls,
            self.orientation_controls,
            self.coil_orientation_controls,
        ):
            for row in controls.values():
                check = row.get("check")
                if isinstance(check, QCheckBox):
                    check.setChecked(False)
        self.assembly_orientation_active.setChecked(False)
        self.coil_orientation_active.setChecked(False)

    def _clear_full_freedom_presets(self) -> None:
        # Full mode is the blank specification route: reveal everything, but do not
        # silently decide which coils, link groups, parameter families, or rows are free.
        for index in range(self.freedom_coils.topLevelItemCount()):
            item = self.freedom_coils.topLevelItem(index)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            object_id = str(item.data(0, Qt.ItemDataRole.UserRole))
            for kind in ("current", "position", "orientation", "geometry"):
                combo = self.freedom_link_widgets[kind].get(object_id)
                if combo is not None:
                    combo.setCurrentIndex(0)
        for toggle in (
            self.current_active,
            self.turns_active,
            self.dimension_active,
            self.position_active,
            self.orientation_active,
            self.assembly_orientation_active,
            self.coil_orientation_active,
        ):
            toggle.setChecked(False)
        for controls in (
            self.dimension_controls,
            self.position_controls,
            self.orientation_controls,
            self.coil_orientation_controls,
        ):
            for row in controls.values():
                check = row.get("check")
                if isinstance(check, QCheckBox):
                    check.setChecked(False)

    def _apply_scope_to_freedom(self) -> None:
        if not hasattr(self, "freedom_coils"):
            return
        active = set(self.requested_freedoms)
        full = self.search_scope == "full"
        self.current_box.setVisible(full or "current" in active)
        self.turns_box.setVisible(full or "turns" in active)
        self.dimension_box.setVisible(full or "dimensions" in active)
        self.position_box.setVisible(full or "position" in active)
        self.orientation_box.setVisible(full or "orientation" in active)
        self.freedom_coils.setColumnHidden(3, not (full or "current" in active))
        self.freedom_coils.setColumnHidden(4, not (full or "position" in active))
        self.freedom_coils.setColumnHidden(5, not (full or "orientation" in active))
        self.freedom_coils.setColumnHidden(6, not (full or ({"turns", "dimensions"} & active)))
        toggle_map = {
            "current": self.current_active,
            "turns": self.turns_active,
            "dimensions": self.dimension_active,
            "position": self.position_active,
            "orientation": self.orientation_active,
        }
        for key, widget in toggle_map.items():
            if full:
                widget.setEnabled(True)
            else:
                # Guided Simple/Complex chooses which families are exposed and
                # preselects them, but the Freedom page is the final authority.
                # Users may uncheck a wizard-selected family if they change their
                # mind before running the optimizer.
                widget.setChecked(key in active)
                widget.setEnabled(True)
        if not full and "orientation" in active:
            # A linked Orientation group now means the physically expected rigid
            # assembly pose by default. In-place coil tilt is a separate opt-in.
            if not self.assembly_orientation_active.isChecked() and not self.coil_orientation_active.isChecked():
                self.assembly_orientation_active.setChecked(True)
            self.assembly_orientation_active.setEnabled(True)
            self.coil_orientation_active.setEnabled(True)
        elif not full:
            self.assembly_orientation_active.setChecked(False)
            self.coil_orientation_active.setChecked(False)
            self.assembly_orientation_active.setEnabled(False)
            self.coil_orientation_active.setEnabled(False)
        labels = {
            "simple": "SIMPLE — only the chosen parameter family is exposed below.",
            "complex": "COMPLEX — only the parameter families selected on Start are exposed below.",
            "full": "FULL — all supported freedoms for the existing scene coils are exposed; nothing is preselected.",
        }
        self.freedom_mode_summary.setText(labels.get(self.search_scope, ""))
        self._freedom_selection_changed()

    def _selected_freedom_coil_records(self) -> list[dict[str, Any]]:
        if not hasattr(self, "freedom_coils"):
            return []
        return [
            self._coil_records[object_id]
            for object_id in self._checked_tree_ids(self.freedom_coils)
            if object_id in self._coil_records
        ]

    @staticmethod
    def _dimension_key_applies_to_coils(key: str, coils: list[dict[str, Any]]) -> bool:
        if not coils:
            return False
        if key == "axial_winding_length":
            return True
        if key == "circular_diameter":
            return any(str(coil.get("native_type", "")) == "current.Circle" for coil in coils)
        if key in {"racetrack_end_diameter", "racetrack_straight_length"}:
            return any(str(coil.get("shape", "")) == "racetrack" for coil in coils)
        if key in {"square_length", "square_width"}:
            return any(
                str(coil.get("native_type", "")) == "current.Polyline"
                and str(coil.get("shape", "")) != "racetrack"
                for coil in coils
            )
        return True

    def _update_dimension_control_visibility(self) -> None:
        if not hasattr(self, "dimension_controls"):
            return
        full = self.search_scope == "full"
        coils = self._selected_freedom_coil_records()
        for key, row in self.dimension_controls.items():
            visible = full or self._dimension_key_applies_to_coils(key, coils)
            for widget_key in ("check", "label", "min", "max", "step"):
                widget = row.get(widget_key)
                if isinstance(widget, QWidget):
                    widget.setVisible(visible)
            if not visible:
                check = row.get("check")
                if isinstance(check, QCheckBox) and check.isChecked():
                    check.blockSignals(True)
                    try:
                        check.setChecked(False)
                    finally:
                        check.blockSignals(False)

        if hasattr(self, "dimension_compatibility_note"):
            if full:
                self.dimension_compatibility_note.setText(
                    "Full mode shows the complete dimension inventory. Shape-specific rows only affect compatible linked coils."
                )
            elif not coils:
                self.dimension_compatibility_note.setText(
                    "Select at least one participating coil above to show compatible dimension choices."
                )
            else:
                shape_labels = sorted({str(coil.get("shape_label", "Coil")) for coil in coils})
                self.dimension_compatibility_note.setText(
                    "Showing dimensions that apply to the selected coil type(s): "
                    + ", ".join(shape_labels)
                    + "."
                )

    def _freedom_selection_changed(self, *args) -> None:
        if not hasattr(self, "freedom_summary"):
            return
        self._update_dimension_control_visibility()
        # Keep the Full page visually honest: a family toggle controls whether its
        # detailed rows matter, and bounds only become editable for checked rows.
        orientation_family_enabled = self.orientation_active.isChecked()
        self.assembly_orientation_active.setEnabled(orientation_family_enabled)
        self.coil_orientation_active.setEnabled(orientation_family_enabled)
        for controls, enabled in (
            (getattr(self, "dimension_controls", {}), self.dimension_active.isChecked()),
            (getattr(self, "position_controls", {}), self.position_active.isChecked()),
            (
                getattr(self, "orientation_controls", {}),
                orientation_family_enabled and self.assembly_orientation_active.isChecked(),
            ),
            (
                getattr(self, "coil_orientation_controls", {}),
                orientation_family_enabled and self.coil_orientation_active.isChecked(),
            ),
        ):
            for row in controls.values():
                check = row.get("check")
                low = row.get("min")
                high = row.get("max")
                step = row.get("step")
                if isinstance(check, QCheckBox):
                    check.setEnabled(enabled)
                    row_enabled = enabled and check.isChecked()
                    if isinstance(low, QWidget):
                        low.setEnabled(row_enabled)
                    if isinstance(high, QWidget):
                        high.setEnabled(row_enabled)
                    if isinstance(step, QWidget):
                        step.setEnabled(row_enabled)
        self.current_link_mode.setEnabled(self.current_active.isChecked())
        self.current_range.setEnabled(self.current_active.isChecked())
        self.current_increment_mA.setEnabled(self.current_active.isChecked())
        self.turns_range.setEnabled(self.turns_active.isChecked())
        self.turns_increment.setEnabled(self.turns_active.isChecked())

        ids = self._checked_tree_ids(self.freedom_coils)
        parts = [f"{len(ids)} optimizer-participating coil(s)"]
        active = self._active_freedoms() if hasattr(self, "current_active") else set()
        enabled_links = {
            "current": "current" in active,
            "position": "position" in active,
            "orientation": "orientation" in active,
            "geometry": bool({"turns", "dimensions"} & active),
        }
        for kind, label in (("current", "current"), ("position", "position"), ("orientation", "orientation"), ("geometry", "geometry/turns")):
            if not enabled_links[kind]:
                continue
            groups = self._linked_groups(kind)
            if groups:
                description = ", ".join(
                    f"{self._optimizer_group_label(name, values)}: {len(values)} coil(s)"
                    for name, values in sorted(groups.items())
                )
                parts.append(f"{label} links [{description}]")
        if "dimensions" in active:
            chosen = self._checked_control_keys(self.dimension_controls)
            if chosen:
                parts.append("dimensions [" + ", ".join(chosen) + "]")
        if "position" in active:
            chosen = self._checked_control_keys(self.position_controls)
            if chosen:
                parts.append("position [" + ", ".join(chosen) + "]")
        if "orientation" in active:
            assembly_chosen = self._checked_control_keys(self.orientation_controls)
            coil_chosen = self._checked_control_keys(self.coil_orientation_controls)
            if self.assembly_orientation_active.isChecked() and assembly_chosen:
                parts.append("assembly rotation [" + ", ".join(assembly_chosen) + "]")
            if self.coil_orientation_active.isChecked() and coil_chosen:
                parts.append("in-place coil rotation [" + ", ".join(coil_chosen) + "]")
        spacing = self._selected_pair_spacing_mm(default_if_unavailable=math.nan)
        if math.isfinite(spacing):
            parts.append(f"current pair spacing {spacing:.6g} mm")
        self.freedom_summary.setText(" • ".join(parts))

    def _limits_toggled(self, *args) -> None:
        self.electrical_rows.setEnabled(self.enforce_electrical.isChecked())

    def _page_changed(self, index: int) -> None:
        if index == self.PAGE_REVIEW:
            self._refresh_review()
        # Results already contains its own heading and status. Hiding the wizard
        # introduction buys useful vertical space for Search Trace/preview; going
        # Back restores the normal formulation guidance.
        self._set_optimizer_results_chrome(compact=index == self.PAGE_RESULTS)
        self._update_navigation()

    def _update_navigation(self) -> None:
        index = self.pages.currentIndex()
        # Results already has finalist actions and an Adjust setup button. The
        # disabled 40 px "Run again" button only made the bottom navigation row
        # taller, which was particularly painful on short Windows displays.
        self.next_button.setVisible(index != self.PAGE_RESULTS)
        self.freedom_defaults_button.setVisible(index == self.PAGE_FREEDOM)
        self.freedom_defaults_button.setEnabled(index == self.PAGE_FREEDOM and self.worker is None)
        names = {
            self.PAGE_START: "START — choose setup",
            self.PAGE_PARAMETERS: (
                "SIMPLE SETUP — choose one parameter"
                if self.search_scope == "simple"
                else "COMPLEX SETUP — choose parameters"
            ),
            self.PAGE_TARGET: "1 of 5 — TARGET",
            self.PAGE_GOAL: "2 of 5 — GOAL",
            self.PAGE_FREEDOM: "3 of 5 — FREEDOM",
            self.PAGE_LIMITS: "4 of 5 — LIMITS",
            self.PAGE_REVIEW: "5 of 5 — REVIEW",
            self.PAGE_RESULTS: "RESULTS",
        }
        self.step_label.setText(names.get(index, "Optimizer"))
        self.back_button.setEnabled(index > self.PAGE_START and self.worker is None)
        self.back_button.setText("Adjust setup" if index == self.PAGE_RESULTS else "Back")
        if index == self.PAGE_START:
            self.next_button.setText("Choose Simple, Complex, or Skip wizard above")
            self.next_button.setEnabled(False)
        elif index == self.PAGE_PARAMETERS:
            self.next_button.setText("Next: Target")
            self.next_button.setEnabled(self.worker is None)
        elif index == self.PAGE_REVIEW:
            self.next_button.setText("Run optimization")
            self.next_button.setEnabled(self.worker is None)
        elif index == self.PAGE_RESULTS:
            self.next_button.setText("Run again")
            self.next_button.setEnabled(False)
        else:
            next_name = {
                self.PAGE_TARGET: "Next: Goal",
                self.PAGE_GOAL: "Next: Freedom",
                self.PAGE_FREEDOM: "Next: Limits",
                self.PAGE_LIMITS: "Next: Review",
            }.get(index, "Next")
            self.next_button.setText(next_name)
            self.next_button.setEnabled(self.worker is None)

    def _previous_page(self) -> None:
        if self.worker is not None:
            return
        index = self.pages.currentIndex()
        if index == self.PAGE_RESULTS:
            self.pages.setCurrentIndex(self.PAGE_REVIEW)
        elif index == self.PAGE_TARGET and self.search_scope == "full":
            self.pages.setCurrentIndex(self.PAGE_START)
        elif index > self.PAGE_START:
            self.pages.setCurrentIndex(index - 1)

    def _next_page(self) -> None:
        if self.worker is not None:
            return
        index = self.pages.currentIndex()
        if index == self.PAGE_REVIEW:
            self._start_search()
            return
        error = self._validate_page(index)
        if error:
            QMessageBox.information(self, "Optimizer setup", error)
            return
        if index == self.PAGE_PARAMETERS:
            error = self._commit_parameter_setup()
            if error:
                QMessageBox.information(self, "Optimizer setup", error)
                return
            self.pages.setCurrentIndex(self.PAGE_TARGET)
            return
        if self.PAGE_TARGET <= index < self.PAGE_REVIEW:
            if index == self.PAGE_FREEDOM:
                self._save_freedom_input_history()
            self.pages.setCurrentIndex(index + 1)

    @staticmethod
    def _checked_control_keys(controls: dict[str, dict[str, QWidget]]) -> list[str]:
        return [
            key for key, row in controls.items()
            if isinstance(row.get("check"), QCheckBox) and row["check"].isChecked()
        ]

    @staticmethod
    def _control_bounds(controls: dict[str, dict[str, QWidget]], key: str) -> tuple[float, float]:
        row = controls[key]
        return float(row["min"].value()), float(row["max"].value())

    @staticmethod
    def _control_increment(controls: dict[str, dict[str, QWidget]], key: str) -> float:
        row = controls[key]
        widget = row.get("step")
        return float(widget.value()) if widget is not None else 0.0

    def _selected_dimension_key(self) -> str | None:
        values = self._checked_control_keys(getattr(self, "dimension_controls", {}))
        return values[0] if len(values) == 1 else None

    def _selected_position_key(self) -> str | None:
        values = self._checked_control_keys(getattr(self, "position_controls", {}))
        return values[0] if len(values) == 1 else None

    def _selected_orientation_key(self) -> str | None:
        values = self._checked_control_keys(getattr(self, "orientation_controls", {}))
        return values[0] if len(values) == 1 else None

    def _selected_coil_orientation_key(self) -> str | None:
        values = self._checked_control_keys(getattr(self, "coil_orientation_controls", {}))
        return values[0] if len(values) == 1 else None

    def _active_freedoms(self) -> set[str]:
        result: set[str] = set()
        if self.current_active.isChecked():
            result.add("current")
        if self.turns_active.isChecked():
            result.add("turns")
        if self.dimension_active.isChecked():
            result.add("dimensions")
        if self.position_active.isChecked():
            result.add("position")
        if self.orientation_active.isChecked():
            result.add("orientation")
        return result

    def _compatible_dimension_ids(self, key: str, coil_ids: list[str]) -> list[str]:
        result: list[str] = []
        for object_id in coil_ids:
            record = self._coil_records.get(str(object_id), {})
            native_type = str(record.get("native_type", ""))
            shape = str(record.get("shape", ""))
            if key == "axial_winding_length":
                result.append(str(object_id))
            elif key == "circular_diameter" and native_type == "current.Circle":
                result.append(str(object_id))
            elif key in {"racetrack_end_diameter", "racetrack_straight_length"} and shape == "racetrack":
                result.append(str(object_id))
            elif key in {"square_length", "square_width"} and native_type == "current.Polyline" and shape != "racetrack":
                result.append(str(object_id))
        return result

    def _optimizer_variable_specs(self) -> list[dict[str, Any]]:
        """Compile the visible Freedom specification into scalar bounded variables.

        The link table answers *which coils share a value*.  The checked rows
        answer *which values may change*.  Complex and Full mode may therefore
        emit any number of variables; Simple mode validates that this list has
        exactly one element.
        """
        active = self._active_freedoms()
        variables: list[dict[str, Any]] = []

        def add(*, kind: str, group_kind: str, group: str, coil_ids: list[str],
                low: float, high: float, label: str, unit: str, base: float,
                step: float | None = None, integer: bool = False,
                extra: dict[str, Any] | None = None) -> None:
            payload: dict[str, Any] = {
                "id": f"{kind}:{group_kind}:{group}",
                "kind": kind,
                "group_kind": group_kind,
                "group": group,
                "coil_ids": list(map(str, coil_ids)),
                "min": float(low),
                "max": float(high),
                "base": float(base),
                "label": label,
                "unit": unit,
                "integer": bool(integer),
            }
            if step is not None:
                payload["step"] = float(step)
                payload["step_origin"] = float(base)
            if extra:
                payload.update(extra)
            variables.append(payload)

        if "current" in active:
            for group, ids in sorted(self._linked_groups("current").items()):
                currents = [float(self._coil_records[value].get("drive_current_mA", 0.0)) for value in ids]
                base = max((abs(value) for value in currents), default=0.0)
                add(
                    kind="current", group_kind="current", group=group, coil_ids=ids,
                    low=self.current_min_mA.value(), high=self.current_max_mA.value(),
                    base=base, label=f"Current {self._optimizer_group_label(group, ids)}", unit="mA",
                    step=self.current_increment_mA.value(),
                    extra={"current_link_mode": str(self.current_link_mode.currentData())},
                )

        geometry_groups = self._linked_groups("geometry")
        if "turns" in active:
            for group, ids in sorted(geometry_groups.items()):
                base = float(self._coil_records[ids[0]].get("turns", self.turns_min.value()))
                add(
                    kind="turns", group_kind="geometry", group=group, coil_ids=ids,
                    low=self.turns_min.value(), high=self.turns_max.value(), base=base,
                    label=f"Turns {self._optimizer_group_label(group, ids)}", unit="turns",
                    step=float(self.turns_increment.value()), integer=True,
                )

        if "dimensions" in active:
            labels = {
                "axial_winding_length": "Axial winding length",
                "circular_diameter": "Circular diameter",
                "racetrack_end_diameter": "Racetrack corner diameter",
                "racetrack_straight_length": "Racetrack straight length",
                "square_length": "Rectangular X length",
                "square_width": "Rectangular Y width",
            }
            record_keys = {
                "axial_winding_length": "axial_winding_length_mm",
                "circular_diameter": "circular_diameter_mm",
                "racetrack_end_diameter": "racetrack_end_diameter_mm",
                "racetrack_straight_length": "racetrack_straight_length_mm",
                "square_length": "square_length_mm",
                "square_width": "square_width_mm",
            }
            for key in self._checked_control_keys(self.dimension_controls):
                low, high = self._control_bounds(self.dimension_controls, key)
                for group, ids in sorted(geometry_groups.items()):
                    compatible = self._compatible_dimension_ids(key, ids)
                    if not compatible:
                        continue
                    base = float(self._coil_records[compatible[0]].get(record_keys[key]) or low)
                    add(
                        kind=key, group_kind="geometry", group=group, coil_ids=compatible,
                        low=low, high=high, base=base,
                        label=f"{labels[key]} {self._optimizer_group_label(group, compatible)}", unit="mm",
                        step=self._control_increment(self.dimension_controls, key),
                    )

        position_groups = self._linked_groups("position")
        if "position" in active:
            for key in self._checked_control_keys(self.position_controls):
                low, high = self._control_bounds(self.position_controls, key)
                for group, ids in sorted(position_groups.items()):
                    if key == "pair_spacing":
                        if len(ids) != 2:
                            continue
                        first = np.asarray(self._coil_records[ids[0]]["position_mm"], dtype=float)
                        second = np.asarray(self._coil_records[ids[1]]["position_mm"], dtype=float)
                        base = float(np.linalg.norm(second - first))
                        label = f"Pair spacing {self._optimizer_group_label(group, ids)}"
                        kind = "pair_spacing"
                    else:
                        base = 0.0
                        label = f"Δ{key.upper()} Position {self._optimizer_group_label(group, ids)}"
                        kind = f"position_{key}"
                    add(
                        kind=kind, group_kind="position", group=group, coil_ids=ids,
                        low=low, high=high, base=base, label=label, unit="mm",
                        step=self._control_increment(self.position_controls, key),
                    )

        orientation_groups = self._linked_groups("orientation")
        if "orientation" in active:
            if self.assembly_orientation_active.isChecked():
                for key in self._checked_control_keys(self.orientation_controls):
                    low, high = self._control_bounds(self.orientation_controls, key)
                    for group, ids in sorted(orientation_groups.items()):
                        add(
                            kind=f"assembly_orientation_{key}", group_kind="orientation", group=group,
                            coil_ids=ids, low=low, high=high, base=0.0,
                            label=f"ΔR{key.upper()} Assembly rotation {self._optimizer_group_label(group, ids)}", unit="°",
                            step=self._control_increment(self.orientation_controls, key),
                        )
            if self.coil_orientation_active.isChecked():
                for key in self._checked_control_keys(self.coil_orientation_controls):
                    low, high = self._control_bounds(self.coil_orientation_controls, key)
                    for group, ids in sorted(orientation_groups.items()):
                        add(
                            kind=f"coil_orientation_{key}", group_kind="orientation", group=group,
                            coil_ids=ids, low=low, high=high, base=0.0,
                            label=f"ΔR{key.upper()} In-place coil rotation {self._optimizer_group_label(group, ids)}", unit="°",
                            step=self._control_increment(self.coil_orientation_controls, key),
                        )
        return variables

    def _validate_page(self, index: int) -> str | None:
        if index == self.PAGE_PARAMETERS:
            if self.search_scope == "simple" and len(self._choice_tree_values(self.simple_choices)) != 1:
                return "Choose exactly one parameter family for Simple setup."
            if self.search_scope == "complex" and not self._choice_tree_values(self.complex_choices):
                return "Choose at least one parameter family for Complex setup."

        if index == self.PAGE_TARGET and not self._checked_tree_ids(self.target_tree):
            return "Select at least one Measurement target before continuing."
        if index == self.PAGE_GOAL:
            mode = str(self.goal_reference_mode.currentData() or "target")
            if mode == "snapshot":
                target_ids = self._checked_tree_ids(self.target_tree)
                if len(target_ids) != 1:
                    return "Snapshot matching currently needs exactly one Measurement target so the stored snapshot grid can be registered onto one current-scene measurement frame."
                snapshot_id = str(self.goal_snapshot.currentData() or "")
                if not snapshot_id:
                    return "Choose a stored field snapshot to match."
                try:
                    self.preview_adapter.optimizer_snapshot_reference(target_ids[0], snapshot_id)
                except StudioOperationError as error:
                    return str(error)
            elif self.goal_target_uT.value() <= 0:
                return "Target intensity must be greater than zero."
        if index != self.PAGE_FREEDOM:
            return None

        coil_ids = self._checked_tree_ids(self.freedom_coils)
        if not coil_ids:
            return "Select at least one coil that participates in the optimization."
        active = self._active_freedoms()
        if not active:
            return "Unlock at least one design freedom before continuing."

        if "current" in active:
            groups = self._linked_groups("current")
            if not groups:
                return "Current freedom needs at least one Current link group."
            if self.current_min_mA.value() > self.current_max_mA.value():
                return "Current minimum must be less than or equal to the maximum."
            if self.current_increment_mA.value() <= 0:
                return "Current increment must be greater than zero."
            if (
                self.current_max_mA.value() > self.current_min_mA.value()
                and self.current_increment_mA.value() > self.current_max_mA.value() - self.current_min_mA.value()
            ):
                return "Current increment must not exceed the current search range."
            if self.current_link_mode.currentData() == "scale_existing":
                for group, ids in groups.items():
                    if not any(abs(float(self._coil_records[value].get("drive_current_mA", 0.0))) > 1e-12 for value in ids):
                        return f"Current Group {group} cannot preserve existing ratios because every starting current is zero."

        geometry_groups = self._linked_groups("geometry")
        if "turns" in active or "dimensions" in active:
            if not geometry_groups:
                return "Turns/dimension freedom needs at least one Geometry / turns link group."
        if "turns" in active and self.turns_min.value() > self.turns_max.value():
            return "Turn-count minimum must be less than or equal to the maximum."
        if "turns" in active and self.turns_max.value() > self.turns_min.value():
            if self.turns_increment.value() > self.turns_max.value() - self.turns_min.value():
                return "Turn increment must not exceed the turn-count search range."

        if "dimensions" in active:
            selected = self._checked_control_keys(self.dimension_controls)
            if not selected:
                return "Choose at least one Dimension freedom."
            for key in selected:
                low, high = self._control_bounds(self.dimension_controls, key)
                if low > high:
                    return "Dimension minimum must be less than or equal to the maximum."
                increment = self._control_increment(self.dimension_controls, key)
                if increment <= 0:
                    return "Dimension increment must be greater than zero."
                if high > low and increment > high - low:
                    return "Dimension increment must not exceed its search range."
                if not any(self._compatible_dimension_ids(key, ids) for ids in geometry_groups.values()):
                    return f"No selected Geometry / turns link group contains a coil compatible with {key.replace('_', ' ')}."

        if "position" in active:
            selected = self._checked_control_keys(self.position_controls)
            if not selected:
                return "Choose at least one Position freedom."
            groups = self._linked_groups("position")
            if not groups:
                return "Position freedom needs at least one Position link group."
            for key in selected:
                low, high = self._control_bounds(self.position_controls, key)
                if low > high:
                    return "Position minimum must be less than or equal to the maximum."
                increment = self._control_increment(self.position_controls, key)
                if increment <= 0:
                    return "Position increment must be greater than zero."
                if high > low and increment > high - low:
                    return "Position increment must not exceed its search range."
                if key == "pair_spacing":
                    if low <= 0:
                        return "Pair centre spacing must stay positive."
                    if not any(len(ids) == 2 for ids in groups.values()):
                        return "Pair-spacing freedom needs at least one Position link group containing exactly two coils."

        if "orientation" in active:
            if not self.assembly_orientation_active.isChecked() and not self.coil_orientation_active.isChecked():
                return "Choose rigid Assembly rotation and/or In-place coil rotation."
            groups = self._linked_groups("orientation")
            if not groups:
                return "Orientation freedom needs at least one Orientation link group."
            orientation_sets = []
            if self.assembly_orientation_active.isChecked():
                orientation_sets.append(("Assembly rotation", self.orientation_controls))
            if self.coil_orientation_active.isChecked():
                orientation_sets.append(("In-place coil rotation", self.coil_orientation_controls))
            for label, controls in orientation_sets:
                selected = self._checked_control_keys(controls)
                if not selected:
                    return f"Choose at least one axis for {label}."
                for key in selected:
                    low, high = self._control_bounds(controls, key)
                    if low > high:
                        return f"{label} minimum must be less than or equal to the maximum."
                    increment = self._control_increment(controls, key)
                    if increment <= 0:
                        return f"{label} increment must be greater than zero."
                    if high > low and increment > high - low:
                        return f"{label} increment must not exceed its search range."

        variables = self._optimizer_variable_specs()
        if not variables:
            return "The current Freedom/link configuration does not produce an executable search variable."
        if self.search_scope == "simple" and len(variables) != 1:
            return (
                f"Simple mode is intentionally one scalar parameter, but the current link setup creates {len(variables)} variables. "
                "Use one link group here or switch to Complex."
            )
        return None

    def _wizard_settings(self) -> dict[str, Any]:
        target_ids = self._checked_tree_ids(self.target_tree)
        participating = self._checked_tree_ids(self.freedom_coils)
        active = self._active_freedoms()
        variables = self._optimizer_variable_specs()
        mode = variables[0]["kind"] if len(variables) == 1 else "multi"
        return {
            "workflow_mode": "tuning",
            "wizard_scope": self.search_scope,
            "requested_freedoms": sorted(active),
            "tuning_mode": mode,
            "reference_mode": str(self.goal_reference_mode.currentData() or "target"),
            "objective": self.goal_objective.currentData(),
            "reference_snapshot_id": str(self.goal_snapshot.currentData() or ""),
            "snapshot_magnitude_tolerance_pct": self.goal_snapshot_magnitude_tolerance.value(),
            "snapshot_direction_tolerance_deg": self.goal_snapshot_direction_tolerance.value(),
            "snapshot_alignment_mode": str(self.goal_snapshot_alignment.currentData() or "fixed"),
            "coil_ids": participating,
            "variables": copy.deepcopy(variables),
            "target_object_ids": target_ids,
            "target_uT": self.goal_target_uT.value(),
            "tolerance_pct": self.goal_tolerance.value(),
            "use_drc": self.use_drc.isChecked(),
            "enforce_electrical_limits": self.enforce_electrical.isChecked(),
            "max_current_a": self.max_current.value() / 1000.0,
            "max_voltage_v": self.max_voltage.value(),
            "max_power_per_coil_w": self.max_power_coil.value(),
            "max_total_power_w": self.max_power_total.value(),
            "max_current_density_a_mm2": self.max_current_density.value(),
            # Six is enough to preserve genuinely different engineering answers
            # without turning the completed Results page into a long candidate dump.
            # The complete search trace remains available for diagnostics.
            "finalist_count": 6,
            "automatic_convergence": True,
            "random_seed": 20260816,
            # Current is a cheap linear inner solve; keep a dense deterministic
            # fallback grid for unusual multi-current cases.
            "current_steps": 161,
            "worker_count": self.optimizer_worker_count,
            "search_effort": ("fast", "balanced", "full")[self.search_effort.value()],
        }

    def _refresh_review(self) -> None:
        target_ids = self._checked_tree_ids(self.target_tree)
        participating = self._checked_tree_ids(self.freedom_coils)
        active = self._active_freedoms()
        target_names = [
            str(self._target_records[value].get("label", value))
            for value in target_ids
            if value in self._target_records
        ]
        coil_names = [
            str(self._coil_records[value].get("label", value))
            for value in participating
            if value in self._coil_records
        ]
        sample_fragments: list[str] = []
        for value in target_ids:
            target = self._target_records.get(value, {})
            if target.get("sampling_mode") == "user_defined":
                sample_fragments.append(
                    f"{target.get('label', value)}: {int(target.get('defined_point_count', 0))} defined points"
                )
            else:
                sample_fragments.append(f"{target.get('label', value)}: Standard generated sampling")

        variables = self._optimizer_variable_specs()
        freedom_lines: list[str] = [f"- Wizard scope: {self.search_scope.title()}."]
        for variable in variables:
            integer_note = " integer" if variable.get("integer") else ""
            increment = variable.get("step")
            increment_note = (
                f"; increment {float(increment):.6g} {variable.get('unit', '')}"
                if increment is not None else ""
            )
            freedom_lines.append(
                f"- {variable['label']}: {float(variable['min']):.6g} to {float(variable['max']):.6g} "
                f"{variable.get('unit', '')}{integer_note}{increment_note}; coils [{', '.join(str(self._coil_records[value].get('label', value)) for value in variable['coil_ids'])}]."
            )
        if len(freedom_lines) == 1:
            freedom_lines.append("- No design variable is currently unlocked.")

        relation_lines: list[str] = []
        enabled_links = {
            "current": "current" in active,
            "position": "position" in active,
            "orientation": "orientation" in active,
            "geometry": bool({"turns", "dimensions"} & active),
        }
        for kind, label in (("current", "Current"), ("position", "Position"), ("orientation", "Orientation"), ("geometry", "Geometry/turns")):
            if not enabled_links[kind]:
                continue
            groups = self._linked_groups(kind)
            if not groups:
                continue
            relation_labels = [label]
            if kind == "orientation":
                relation_labels = []
                if self.assembly_orientation_active.isChecked():
                    relation_labels.append("Assembly rotation")
                if self.coil_orientation_active.isChecked():
                    relation_labels.append("In-place coil rotation")
            for group, ids in sorted(groups.items()):
                names = [str(self._coil_records[value].get("label", value)) for value in ids]
                group_text = self._optimizer_group_label(group, ids)
                for relation_label in relation_labels:
                    relation_lines.append(
                        f"- {relation_label} {group_text}: {', '.join(names)}"
                    )

        background_lines: list[str] = []
        # The optimizer wizard evaluates and reports the frozen source document
        # captured when the dialog opens.  Use that same preview adapter for
        # fixed background fields rather than reaching back into the live scene.
        # This also keeps review generation deterministic if the source window is
        # edited while a detached optimizer dialog remains open.
        review_adapter = self.preview_adapter
        for item in review_adapter.list_objects():
            if not item.get("workbench_background_field"):
                continue
            object_id = str(item.get("id", ""))
            if not object_id:
                continue
            try:
                properties = review_adapter.background_field_properties(object_id)
                active = review_adapter._object_effectively_visible(object_id)
            except Exception:
                continue
            vector = [float(value) for value in properties.get("vector_uT", [0.0, 0.0, 0.0])]
            magnitude = math.sqrt(sum(value * value for value in vector))
            source = properties.get("source", {}) if isinstance(properties.get("source"), dict) else {}
            source_text = "manual vector"
            if str(properties.get("mode", "manual")) == "wmm2025":
                source_text = (
                    f"WMM2025 at {float(source.get('latitude_deg', 0.0)):.6f}°, "
                    f"{float(source.get('longitude_deg', 0.0)):.6f}°, "
                    f"{float(source.get('altitude_m', 0.0)):.1f} m on {source.get('date', '—')}"
                )
            state = "active" if active else "hidden / not contributing"
            background_lines.append(
                f"- {properties.get('label', object_id)}: B = "
                f"({vector[0]:.6g}, {vector[1]:.6g}, {vector[2]:.6g}) µT; "
                f"|B| = {magnitude:.6g} µT; {source_text}; {state}."
            )

        limit_lines = [
            "- Workbench DRC is a hard gate." if self.use_drc.isChecked()
            else "- DRC is disabled for this exploration."
        ]
        limit_lines.append(
            f"- Independent candidate batches may use {self.optimizer_worker_count} concurrent "
            f"worker{'s' if self.optimizer_worker_count != 1 else ''}; dependent refinement steps remain ordered."
        )
        if self.enforce_electrical.isChecked():
            limit_lines.append(
                f"- Electrical hard limits: {self.max_current.value():.6g} mA/coil, "
                f"{self.max_voltage.value():.6g} V/coil, {self.max_power_coil.value():.6g} W/coil, "
                f"{self.max_power_total.value():.6g} W total, {self.max_current_density.value():.6g} A/mm²."
            )
        else:
            limit_lines.append("- Electrical hard limits are not enabled.")
        effort = ("Fast", "Balanced", "Full")[self.search_effort.value()]
        limit_lines.append(f"- Search effort: {effort}.")

        has_current_variable = any(str(item.get("kind")) == "current" for item in variables)
        structural_variable_count = sum(str(item.get("kind")) != "current" for item in variables)
        if has_current_variable and structural_variable_count:
            search_detail = (
                "- Search convergence: automatic. Structural variables are searched and refined until convergence; "
                "allowed current group(s) are solved inside each geometry/placement candidate using the linear field response."
            )
        elif structural_variable_count == 0 and has_current_variable:
            search_detail = (
                "- Search convergence: automatic. Current freedoms are solved directly from the linear field response."
            )
        else:
            search_detail = (
                f"- Search convergence: automatic across {max(1, structural_variable_count)} structural variable(s), "
                "with boundary checks and local refinement continued until the result stabilizes."
            )

        reference_mode = str(self.goal_reference_mode.currentData() or "target")
        if reference_mode == "snapshot":
            snapshot_id = str(self.goal_snapshot.currentData() or "")
            snapshot = self._snapshot_records.get(snapshot_id, {})
            alignment_mode = str(self.goal_snapshot_alignment.currentData() or "fixed")
            alignment_text = (
                "Exposure registration: absolute orientation is free. For scoring, optimizer-participating exposure sources may be rigidly registered about the target physical centre so spatial field structure and vectors rotate together; fixed/background sources and DRC world geometry remain fixed."
                if alignment_mode == "exposure"
                else "World/target orientation is hard: snapshot vectors are compared point-for-point in the Measurement object's local coordinate frame."
            )
            goal_lines = [
                f"Reference: stored snapshot {snapshot.get('name', snapshot_id or 'not selected')}.",
                f"Primary match criterion: {self.goal_objective.currentText()}.",
                f"Point-match tolerances: magnitude ±{self.goal_snapshot_magnitude_tolerance.value():.6g}% and direction {self.goal_snapshot_direction_tolerance.value():.6g}°.",
                alignment_text,
            ]
        else:
            goal_lines = [
                "Reference: requested target intensity / field specification.",
                f"Primary objective: {self.goal_objective.currentText()}.",
                f"Reference field statistics use {self.goal_target_uT.value():.6g} µT ± {self.goal_tolerance.value():.6g}%.",
            ]

        review = [
            "OPTIMIZATION PLAN",
            "",
            "TARGET",
            f"Use {', '.join(target_names) if target_names else 'no selected Measurement target'}.",
            *[f"- {line}" for line in sample_fragments],
            "",
            "GOAL",
            *goal_lines,
            "",
            "FREEDOM",
            f"Optimizer-participating existing coils: {', '.join(coil_names) if coil_names else 'none'}.",
            *freedom_lines,
            "",
            "RELATIONSHIPS",
            *(relation_lines or ["- No linked relationship groups are active."]),
            "Everything not listed as free remains fixed in the source scene.",
            *(["", "BACKGROUND FIELDS", *background_lines] if background_lines else []),
            "",
            "LIMITS",
            *limit_lines,
            search_detail,
        ]
        self.review_text.setPlainText("\n".join(review))

    @staticmethod
    def _optimizer_effort_display_name(value: Any) -> str:
        effort = str(value or "balanced").strip().lower()
        return {"fast": "Fast", "balanced": "Balanced", "full": "Full"}.get(effort, "Balanced")

    def _refresh_results_context(
        self,
        *,
        settings: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Keep Results headed by search effort and the actual target sampling in use."""
        settings = settings or {}
        progress = progress or {}
        result = result or {}
        effective_settings = result.get("settings", {}) if result else settings
        effort = self._optimizer_effort_display_name(
            effective_settings.get("search_effort", settings.get("search_effort", "balanced"))
        )
        if hasattr(self, "results_heading"):
            self.results_heading.setText(f"RESULTS — {effort} search")

        target_ids = list(
            effective_settings.get("target_object_ids", settings.get("target_object_ids", [])) or []
        )
        selected_records = [
            self._target_records[value]
            for value in target_ids
            if value in self._target_records
        ]
        result_records = list(result.get("targets", []) or [])
        progress_records = list(progress.get("target_samples", []) or [])
        records = result_records or progress_records or selected_records

        labels = [str(item.get("label", item.get("id", "target"))) for item in records]
        if not labels:
            labels = [
                str(self._target_records[value].get("label", value))
                for value in target_ids
                if value in self._target_records
            ]
        label_text = " + ".join(labels) if labels else "No selected Measurement target"
        prefix = "Target" if len(labels) <= 1 else "Targets"

        full_count = None
        if result:
            full_count = result.get("final_sample_count")
        if full_count is None:
            full_count = progress.get("target_sample_count")
        if full_count is None and records and all(item.get("sample_count") is not None for item in records):
            try:
                full_count = sum(int(item.get("sample_count", 0)) for item in records)
            except (TypeError, ValueError):
                full_count = None

        scout_count = progress.get("scout_sample_count")
        if scout_count is None and result:
            fidelity = result.get("multi_fidelity", {}) or {}
            scout_count = fidelity.get("scout_sample_count")

        sample_parts: list[str] = []
        if full_count is not None:
            try:
                sample_parts.append(f"{int(full_count):,} full sample{'s' if int(full_count) != 1 else ''}")
            except (TypeError, ValueError):
                pass
        else:
            known_defined = [
                int(item.get("defined_point_count", 0) or 0)
                for item in records
                if str(item.get("sampling_mode", "")) == "user_defined"
            ]
            if records and len(known_defined) == len(records):
                sample_parts.append(f"{sum(known_defined):,} defined samples")
            else:
                sample_parts.append("sample count preparing…")

        if scout_count is not None:
            try:
                scout_value = int(scout_count)
                full_value = int(full_count) if full_count is not None else None
                if scout_value > 0 and scout_value != full_value:
                    sample_parts.append(f"{scout_value:,} scout samples")
            except (TypeError, ValueError):
                pass

        modes = {str(item.get("sampling_mode", "generated")) for item in records}
        if modes == {"user_defined"}:
            sample_parts.append("user-defined sampling")
        elif modes:
            sample_parts.append("Standard generated sampling" if modes <= {"generated", "standard"} else "mixed sampling")

        reference_mode = str(effective_settings.get("reference_mode", "target") or "target")
        if reference_mode == "snapshot":
            snapshot_id = str(effective_settings.get("reference_snapshot_id", "") or "")
            snapshot = self._snapshot_records.get(snapshot_id, {})
            if snapshot:
                sample_parts.append(f"snapshot: {snapshot.get('name', snapshot_id)}")
        else:
            target_uT = effective_settings.get("target_uT")
            tolerance = effective_settings.get("tolerance_pct")
            if target_uT is not None and tolerance is not None:
                try:
                    sample_parts.append(f"{float(target_uT):.6g} µT ± {float(tolerance):.6g}%")
                except (TypeError, ValueError):
                    pass

        if hasattr(self, "results_target_info"):
            suffix = " • ".join(sample_parts)
            self.results_target_info.setText(f"{prefix}: {label_text}" + (f" • {suffix}" if suffix else ""))

    def _set_optimizer_result_tab_mode(self, *, searching: bool, has_finalist: bool = False) -> None:
        """Give Search Trace the Results page while searching, then reveal finalists."""
        if not hasattr(self, "result_views"):
            return

        self.result_views.setTabVisible(0, not searching)
        self.result_views.setTabVisible(1, not searching)
        self.result_views.setTabVisible(2, True)
        self.result_views.setCurrentWidget(
            self.sweep_plot if searching or not has_finalist else self.candidate_plot
        )

        # A finalist table, summary, and finalist actions have nothing useful to
        # show during the search. Hiding them gives the live console a stable,
        # tall viewport while telemetry is updating.
        if hasattr(self, "result_summary"):
            self.result_summary.setVisible(not searching)
        if hasattr(self, "result_tree"):
            self.result_tree.setVisible(not searching and has_finalist)
        if hasattr(self, "result_view_panel"):
            self.result_view_panel.setVisible(not searching and has_finalist)
        if hasattr(self, "candidate_detail"):
            self.candidate_detail.setVisible(not searching and has_finalist)
        for name in (
            "copy_button", "compare_button", "save_snapshot_button",
            "open_tab_button", "append_button", "apply_button",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(not searching)

        # Keep the live Search Trace large enough to show the complete status board
        # without chasing a scrollbar while telemetry updates.  After completion
        # the result tabs still retain enough height for a useful finalist preview.
        self.result_views.setMinimumHeight(500 if searching else 220)

    def _set_optimizer_results_chrome(self, *, compact: bool) -> None:
        """Hide wizard-only explanatory chrome while Results owns the window."""
        for widget in (self.dialog_heading, self.dialog_intro, self.step_label):
            widget.setVisible(not compact)

    def _resize_result_tree_to_content(self, finalist_count: int) -> None:
        """Choose a readable default splitter position without locking it there."""
        count = max(1, min(int(finalist_count), 6))
        header_height = max(22, self.result_tree.header().sizeHint().height())
        row_height = self.result_tree.sizeHintForRow(0) if self.result_tree.topLevelItemCount() else 24
        if row_height <= 0:
            row_height = 24
        # Allow for the horizontal scrollbar: long optimizer setting summaries
        # almost always need one, and omitting it was what squeezed a one-row
        # finalist table into an unreadable sliver.
        target = header_height + count * row_height + 32
        target = max(96, min(190, target))
        self.result_tree.setMinimumHeight(96)
        self.result_tree.setMaximumHeight(16777215)
        splitter = getattr(self, "result_splitter", None)
        if splitter is not None:
            total = max(splitter.height(), sum(splitter.sizes()), target + 220)
            splitter.setSizes([target, max(220, total - target)])

    def _fit_completed_results_to_screen(self) -> None:
        """Clamp a completed optimizer to a comfortable resizable screen fit."""
        # A zero-delay fit is queued when results arrive.  Tests, application
        # shutdown, or a quick user close can delete the native QWidget before
        # that callback is dispatched while the Python wrapper is still alive.
        # Querying screen() on that stale wrapper raises from shiboken, so make
        # the cosmetic resize explicitly safe to abandon.
        try:
            screen = self.screen()
        except RuntimeError:
            return
        if screen is None:
            return
        available = screen.availableGeometry()
        # A compact completed result should fit well clear of the taskbar even on
        # shorter displays. Do not enlarge an already smaller user-sized window.
        target_height = min(760, max(560, available.height() - 48))
        target_width = min(self.width(), max(760, available.width() - 32))
        if self.height() > target_height or self.width() > target_width:
            self.resize(min(self.width(), target_width), min(self.height(), target_height))

    def _initialize_optimizer_trace_plan(self, settings: dict[str, Any]) -> None:
        """Build the live execution-plan tree shown in Search Trace.

        The optimizer has conditional branches, so not every leaf can know its exact
        candidate count in advance.  The broad stage budgets *are* known, however, and
        the tree intentionally shows unknown conditional work as ``?`` until that branch
        is entered instead of pretending the search is a flat progress bar.
        """
        variables = list(settings.get("variables", []) or [])
        structural = [item for item in variables if str(item.get("kind", "")) != "current"]
        currents = [item for item in variables if str(item.get("kind", "")) == "current"]
        structural_count = len(structural)
        effort = str(settings.get("search_effort", "balanced") or "balanced").lower()
        exposure_registration = str(
            settings.get("snapshot_alignment_mode", "fixed") or "fixed"
        ).strip().lower() == "exposure"

        requested_finalists = max(1, int(settings.get("finalist_count", 6) or 6))
        if effort == "fast":
            default_cap = min(8, max(6, requested_finalists))
        elif effort == "balanced":
            default_cap = min(14, max(10, requested_finalists + 4))
        else:
            default_cap = min(24, max(14, 2 * requested_finalists + 2))
        exact_budget = int(settings.get("max_evaluations", settings.get("evaluation_budget", default_cap)) or default_cap)

        def node(
            node_id: str,
            label: str,
            *,
            total: int | None = None,
            total_text: str | None = None,
            total_is_cap: bool = False,
            scope: str = "outer",
            countless: bool = False,
            children: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            return {
                "id": node_id,
                "label": label,
                "total": total,
                "total_text": total_text,
                "total_is_cap": bool(total_is_cap),
                "scope": scope,
                "countless": bool(countless),
                "children": children or [],
                "status": "pending",
                "count": 0,
                "segment_count": 0,
                "base_candidate": None,
                "last_candidate": None,
                "elapsed_accum": 0.0,
                "segment_started": None,
            }

        prepare = node(
            "prepare",
            "Stage 1 — Freeze and validate the problem",
            scope="none",
            countless=True,
            children=[
                node("prepare.freeze", "Frozen scene + target samples", scope="none", countless=True),
                node(
                    "prepare.anchor",
                    "Geometry-only source feasibility anchor" if structural_count else "Source-scene feasibility anchor",
                    total=1,
                    scope="outer",
                ),
            ],
        )

        if structural_count == 0:
            plan = [
                prepare,
                node(
                    "current",
                    "Stage 2 — Direct linear current solve",
                    total_text="direct",
                    children=[
                        node("current.solve", "Bounded current solve + increment snap", total_text="direct"),
                        node("current.verify", "Electrical / DRC verification", total_text="as needed"),
                    ],
                ),
            ]
        else:
            # Every structural search now uses the same cheap-first policy. Physical-only
            # winding width and amplitude-redundant turns may be omitted from the magnetic
            # reconnaissance and reintroduced in the geometry/finalist reality checks.
            scout_structural_count = sum(
                1
                for item in structural
                if str(item.get("kind", "")) != "axial_winding_length"
                and not (currents and str(item.get("kind", "")) == "turns")
            )
            if scout_structural_count <= 1:
                scout_budget = {"fast": 96, "balanced": 128, "full": 192}[effort]
            elif scout_structural_count == 2:
                scout_budget = {"fast": 320, "balanced": 640, "full": 1280}[effort]
            elif effort == "fast":
                scout_budget = min(896, max(704, 48 * max(4, scout_structural_count)))
            elif effort == "balanced":
                scout_budget = min(3072, max(2048, 160 * max(4, scout_structural_count)))
            else:
                scout_budget = min(12288, max(4096, 768 * max(4, scout_structural_count)))

            landmark_limit = {"fast": 6, "balanced": 10, "full": 24}[effort]
            validation_cap = {
                "fast": max(4, requested_finalists),
                "balanced": max(8, requested_finalists + 2),
                "full": max(12, requested_finalists + 4),
            }[effort]
            validation_cap = min(exact_budget, validation_cap)

            if scout_structural_count <= 1:
                map_children = [
                    node("map.cache", "Build + validate cheap Bx/By/Bz representation", scope="scout", countless=True),
                    node("map.sweep", "Full-range cheap 1-D sweep / convergence", scope="scout", total_text=f"≤{scout_budget}"),
                    node("map.fallback", "Plain Centreline fallback map", scope="scout", total_text="only if cache/map fails"),
                ]
            else:
                map_children = [
                    node("map.cache", "Build + validate coarse Bx/By/Bz stamps", scope="scout", countless=True),
                    node("map.canonical", "Canonical assembly + explicit bounds", scope="scout"),
                    node(
                        "map.align",
                        "Target centering / physical pose priors"
                        if exposure_registration
                        else "Target alignment / plane straightening",
                        scope="scout",
                    ),
                    node("map.relations", "Known relations + boundary insurance", scope="scout"),
                    node("map.polish", "Fast-funnel coordinate polish", scope="scout", total_text="if recognized pair"),
                    node("map.symmetry", "Symmetry + collective-bound insurance", scope="scout"),
                    node("map.sobol", "Global Sobol skeptic map", scope="scout"),
                    node(
                        "map.fine", "Fine-cache landmark recheck", scope="scout",
                        total_text="if coarse ambiguous" if effort in {"fast", "balanced"} else None,
                    ),
                    node("map.adaptive", "Adaptive Sobol basin subdivision", scope="scout"),
                    node("map.fallback", "Plain Centreline fallback map", scope="scout", total_text="only if cache/map fails"),
                ]

            mapping = node(
                "map",
                (
                    "Stage 2 — Cheap physics map + inner exposure registration"
                    if exposure_registration
                    else "Stage 2 — Cheap physics map — Centreline/vector cache"
                ),
                # ``scout_budget`` is a safety ceiling, not an exact amount of work.
                # Fast funnels commonly converge after only a small fraction of it, so
                # present it as a cap while the stage is active and collapse to the
                # observed count when the stage completes.
                total=scout_budget,
                total_is_cap=True,
                scope="scout",
                children=map_children,
            )
            reality = node(
                "reality",
                "Stage 3 — Geometry reality + finalist promotion",
                total_text=f"optional middle fidelity + ≤{exact_budget} full-model settings",
                children=[
                    node("reality.drc", "Geometry-only DRC feasibility map", total=landmark_limit),
                    node("reality.promote", "Reduced-target physical promotion", total_text="if source cost warrants"),
                    node("reality.validate", "Progressive authoritative full-model validation", total=validation_cap),
                ],
            )
            plan = [prepare, mapping, reality]

        self._optimizer_trace_plan = plan
        self._optimizer_trace_nodes = {}

        def register(item: dict[str, Any], parent_id: str | None = None) -> None:
            item["parent_id"] = parent_id
            self._optimizer_trace_nodes[str(item["id"])] = item
            for child in item.get("children", []):
                register(child, str(item["id"]))

        for item in plan:
            register(item)
        self._optimizer_trace_active_leaf = None
        self._optimizer_progress_fraction = 0.0

        # Progress weights describe the *planned search*, not the number of raw
        # magnetic samples.  A Centreline scout evaluation and a full-model
        # validation do not have comparable cost, and conditional leaves may not
        # run at all.  Top-level stage weights therefore reflect the configured
        # effort mode, while leaf weights make long known branches (Sobol/fine
        # cache, coordinate convergence, etc.) fill smoothly from their own
        # completed/total telemetry.  Skipped branches resolve their reserved
        # share when the optimizer moves past them.
        if "map" in self._optimizer_trace_nodes:
            # Cheap stamp-native reconnaissance is now genuinely cheap, while the
            # geometry/DRC and authoritative reality stage can dominate wall time by
            # an order of magnitude.  Weight progress by expected computational work,
            # not raw candidate count, so a one-minute map no longer reports ~88% just
            # before several minutes of DRC boundary work.
            if effort == "fast":
                stage_weights = {"prepare": 0.03, "map": 0.27, "reality": 0.70}
            elif effort == "full":
                stage_weights = {"prepare": 0.02, "map": 0.43, "reality": 0.55}
            else:
                stage_weights = {"prepare": 0.03, "map": 0.35, "reality": 0.62}
        elif "current" in self._optimizer_trace_nodes:
            stage_weights = {"prepare": 0.10, "current": 0.90}
        else:
            stage_weights = {"prepare": 0.10, "reality": 0.90}

        child_weights = {
            "prepare.freeze": 0.35,
            "prepare.anchor": 0.65,
            "map.cache": 0.05,
            "map.sweep": 0.95,
            "map.canonical": 0.02,
            "map.align": 0.01,
            "map.relations": 0.08,
            "map.symmetry": 0.18,
            "map.sobol": 0.44,
            "map.fine": 0.18,
            "map.adaptive": 0.04,
            "map.fallback": 0.03,
            "reality.drc": 0.60,
            "reality.promote": 0.15,
            "reality.validate": 0.25,
            "current.solve": 0.80,
            "current.verify": 0.20,
        }
        if effort == "fast" and "map" in self._optimizer_trace_nodes:
            child_weights.update({
                "map.cache": 0.08,
                "map.canonical": 0.04,
                "map.align": 0.02,
                "map.relations": 0.16,
                "map.symmetry": 0.22,
                "map.sobol": 0.34,
                "map.fine": 0.05,
                "map.adaptive": 0.09,
            })
        elif effort == "balanced" and "map" in self._optimizer_trace_nodes:
            child_weights.update({
                "map.cache": 0.07,
                "map.canonical": 0.03,
                "map.align": 0.01,
                "map.relations": 0.10,
                "map.symmetry": 0.18,
                "map.sobol": 0.34,
                "map.fine": 0.12,
                "map.adaptive": 0.15,
            })
        for node_id, weight in stage_weights.items():
            trace_node = self._optimizer_trace_nodes.get(node_id)
            if trace_node is not None:
                trace_node["progress_weight"] = float(weight)
        for node_id, weight in child_weights.items():
            trace_node = self._optimizer_trace_nodes.get(node_id)
            if trace_node is not None:
                trace_node["progress_weight"] = float(weight)

    def _optimizer_trace_leaf_for_phase(self, phase: str) -> str | None:
        text = " ".join(str(phase or "").lower().split())
        if not text:
            return None
        if "optimization complete" in text or text in {"cancelled", "optimization failed"}:
            return None
        if "centreline fallback map" in text:
            return "map.fallback" if "map.fallback" in self._optimizer_trace_nodes else "map.cache"
        if "centreline map" in text:
            if "coarse 1-d search" in text or "converging 1-d optimum" in text:
                return "map.sweep" if "map.sweep" in self._optimizer_trace_nodes else "map.cache"
            if "fast funnel" in text:
                if "target orientation" in text or "target placement" in text:
                    return "map.align"
                if "known relationships" in text or "relation refinement" in text:
                    return "map.relations"
                if "final coordinate polish" in text:
                    return "map.polish" if "map.polish" in self._optimizer_trace_nodes else "map.adaptive"
            if "stage a canonical assembly" in text:
                return "map.canonical"
            if "target alignment" in text or "straightening" in text:
                return "map.align"
            if "known-relation atlas" in text:
                return "map.relations"
            if "symmetry-breaking atlas" in text or "assembly-mode trajectories" in text:
                return "map.symmetry"
            if "global sobol skeptic" in text:
                return "map.sobol"
            if "fine-cache landmark recheck" in text:
                return "map.fine"
            if "adaptive basin subdivision" in text:
                return "map.adaptive"
            return "map.cache"
        if "preparing centreline scout" in text:
            return "map.cache"
        if "locating physical drc boundaries" in text or "drc-clear rescue anchor" in text:
            return "reality.drc"
        if "finalist promotion" in text or "reduced-target screen" in text:
            return "reality.promote" if "reality.promote" in self._optimizer_trace_nodes else "reality.validate"
        if "local full-model basin adaptation" in text:
            return "reality.validate" if "reality.validate" in self._optimizer_trace_nodes else None
        if "full-model validation" in text or "finalist vet" in text:
            return "reality.validate"
        if "preparing frozen scene and target samples" in text:
            return "prepare.freeze"
        if "checking source-scene starting point" in text:
            return "prepare.anchor"
        if "solving current" in text:
            return "current.solve" if "current.solve" in self._optimizer_trace_nodes else "exact.focus"
        if "coarse 1-d search" in text or "converging 1-d optimum" in text:
            return "map.sweep" if "map.sweep" in self._optimizer_trace_nodes else "exact.focus"
        if "screening individual freedoms" in text or "screening collective assembly modes" in text or "screening coupled freedoms" in text:
            return "exact.screen"
        if "focused active-set search" in text or "converging coordinates" in text or "local multistart refinement" in text:
            return "exact.focus"
        if "challenging frozen freedoms" in text or "perfect-coverage" in text or "global verification" in text:
            return "exact.challenge"
        if "verifying explicit search boundaries" in text or "conditionally re-polishing promising bounds" in text or "converging search boundaries" in text or ("refining" in text and "boundary" in text):
            if "reality.validate" in self._optimizer_trace_nodes:
                return "reality.validate"
            return "exact.bounds" if "exact.bounds" in self._optimizer_trace_nodes else None
        if "normalizing frozen freedoms for finalists" in text or "canonical simplification" in text:
            return "exact.curate"
        if "global coarse search" in text or "global search" in text or "sobol landscape" in text:
            return "exact.global"
        if "evaluating process candidate" in text:
            return "exact.focus"
        return "exact.focus" if "exact.focus" in self._optimizer_trace_nodes else None

    def _optimizer_trace_finalize_node(self, node: dict[str, Any], now: float) -> None:
        if node.get("status") != "active":
            return
        started = node.get("segment_started")
        if started is not None:
            node["elapsed_accum"] = float(node.get("elapsed_accum", 0.0)) + max(0.0, now - float(started))
        base = node.get("base_candidate")
        last = node.get("last_candidate")
        if base is not None and last is not None and not node.get("countless"):
            node["count"] = int(node.get("count", 0)) + max(0, int(last) - int(base))
        node["segment_started"] = None
        node["base_candidate"] = None
        node["last_candidate"] = None
        node["status"] = "complete"

    def _update_optimizer_trace_plan(self, update: dict[str, Any]) -> None:
        if not self._optimizer_trace_nodes:
            return
        representation_label = str(update.get("field_representation_label") or "").strip()
        if representation_label:
            cache_node = self._optimizer_trace_nodes.get("map.cache")
            if cache_node is not None:
                cache_node["label"] = representation_label
        leaf_id = self._optimizer_trace_leaf_for_phase(str(update.get("phase_label") or ""))
        if leaf_id is None or leaf_id not in self._optimizer_trace_nodes:
            return
        now = self._optimizer_elapsed_seconds()
        old_id = self._optimizer_trace_active_leaf
        old_node = self._optimizer_trace_nodes.get(old_id or "")
        new_node = self._optimizer_trace_nodes[leaf_id]

        if old_node is not None and old_id != leaf_id:
            old_parent_id = old_node.get("parent_id")
            self._optimizer_trace_finalize_node(old_node, now)
            new_parent_id = new_node.get("parent_id")
            if old_parent_id and old_parent_id != new_parent_id:
                old_parent = self._optimizer_trace_nodes.get(str(old_parent_id))
                if old_parent is not None and old_parent.get("status") == "active":
                    self._optimizer_trace_finalize_node(old_parent, now)

        if new_node.get("status") != "active":
            new_node["status"] = "active"
            new_node["segment_started"] = now
            if not new_node.get("countless"):
                new_node["base_candidate"] = int(update.get("candidate", 0) or 0)
        new_node["last_candidate"] = int(update.get("candidate", 0) or 0)
        phase_label = str(update.get("phase_label") or "")
        previous_phase = str(new_node.get("trace_total_phase") or "")
        same_leaf_new_segment = bool(
            old_id == leaf_id and previous_phase and previous_phase != phase_label
        )
        if update.get("trace_completed") is not None:
            incoming_completed = max(0, int(update.get("trace_completed", 0) or 0))
            if same_leaf_new_segment:
                new_node["absolute_count"] = (
                    max(0, int(new_node.get("absolute_count", 0) or 0))
                    + incoming_completed
                )
            else:
                new_node["absolute_count"] = incoming_completed
        if update.get("trace_total") is not None:
            incoming_total = max(0, int(update.get("trace_total", 0) or 0))
            if same_leaf_new_segment:
                new_node["total"] = max(0, int(new_node.get("total", 0) or 0)) + incoming_total
            else:
                new_node["total"] = incoming_total
            new_node["total_text"] = None
            new_node["trace_total_phase"] = phase_label
        self._optimizer_trace_active_leaf = leaf_id

        parent_id = new_node.get("parent_id")
        if parent_id:
            parent = self._optimizer_trace_nodes.get(str(parent_id))
            if parent is not None:
                if parent.get("status") != "active":
                    parent["status"] = "active"
                    parent["segment_started"] = now
                    if not parent.get("countless"):
                        parent["base_candidate"] = int(update.get("candidate", 0) or 0)
                parent["last_candidate"] = int(update.get("candidate", 0) or 0)
                # The scout's candidate counter is its own complete map progress and
                # therefore more useful than summing child deltas.
                if str(parent.get("scope")) == "scout" and "centreline map" in str(update.get("phase_label", "")).lower():
                    parent["absolute_count"] = int(update.get("candidate", 0) or 0)

    def _complete_optimizer_trace_plan(self, result: dict[str, Any] | None = None, *, cancelled: bool = False) -> None:
        if not self._optimizer_trace_nodes:
            return
        now = self._optimizer_elapsed_seconds()
        active = self._optimizer_trace_nodes.get(self._optimizer_trace_active_leaf or "")
        if active is not None:
            self._optimizer_trace_finalize_node(active, now)
            parent = self._optimizer_trace_nodes.get(str(active.get("parent_id") or ""))
            if parent is not None and parent.get("status") == "active":
                self._optimizer_trace_finalize_node(parent, now)
        self._optimizer_trace_active_leaf = None

        if result is not None:
            curate = self._optimizer_trace_nodes.get("exact.curate")
            if curate is not None:
                curate["status"] = "complete"
                curate["count"] = int(result.get("distinct_finalist_count", len(result.get("finalists", []) or [])) or 0)
                curate["elapsed_accum"] = max(float(curate.get("elapsed_accum", 0.0)), 0.0)
            mapping = self._optimizer_trace_nodes.get("map")
            fidelity = result.get("multi_fidelity", {}) or {}
            if mapping is not None:
                scout_count = int((fidelity or {}).get("scout_evaluations", 0) or 0)
                if scout_count:
                    mapping["absolute_count"] = scout_count
                    mapping["status"] = "complete"
                completed_counts = {
                    "map.canonical": int(fidelity.get("scout_map_canonical_evaluations", 0) or 0),
                    "map.align": (
                        int(fidelity.get("scout_fast_engineering_orientation_evaluations", 0) or 0)
                        + int(fidelity.get("scout_fast_engineering_placement_evaluations", 0) or 0)
                    ),
                    "map.relations": int(fidelity.get("scout_map_relation_evaluations", 0) or 0),
                    "map.polish": int(fidelity.get("scout_fast_engineering_final_polish_evaluations", 0) or 0),
                    "map.symmetry": int(fidelity.get("scout_map_symmetry_evaluations", 0) or 0),
                    "map.sobol": int(fidelity.get("scout_map_global_evaluations", 0) or 0),
                    "map.fine": int(fidelity.get("scout_map_fine_recheck_evaluations", 0) or 0),
                    "map.adaptive": int(fidelity.get("scout_map_adaptive_evaluations", 0) or 0),
                    "reality.drc": int(fidelity.get("drc_probe_count", 0) or 0),
                    "reality.promote": int(fidelity.get("intermediate_candidate_count", 0) or 0),
                    "reality.validate": int(fidelity.get("physical_validation_evaluations", 0) or 0),
                    "reality.adapt": int(fidelity.get("physical_refinement_evaluations", 0) or 0),
                }
                for node_id, completed_count in completed_counts.items():
                    trace_node = self._optimizer_trace_nodes.get(node_id)
                    if trace_node is None or completed_count <= 0:
                        continue
                    if self._optimizer_trace_node_count(trace_node) > 0:
                        continue
                    trace_node["count"] = completed_count
                    trace_node["segment_count"] = 0
                    trace_node["base_candidate"] = None
                    trace_node["last_candidate"] = None
                    trace_node["status"] = "complete"
                    # Conditional work can exceed its initial presentation estimate
                    # (for example DRC bisection probes).  Once the run is complete,
                    # the observed count is the authoritative denominator.
                    if trace_node.get("total") is None or completed_count > int(trace_node.get("total") or 0):
                        trace_node["total"] = completed_count
                        trace_node["total_text"] = None

        for node in self._optimizer_trace_nodes.values():
            if node.get("status") == "pending":
                node["status"] = "skipped" if not cancelled else "pending"

    @staticmethod
    def _optimizer_stage_delta(current: int, baseline: int) -> int:
        return max(0, int(current) - int(baseline))

    def _record_optimizer_stage(self, update: dict[str, Any]) -> None:
        phase = str(update.get("phase_label") or "Preparing search")
        candidate = int(update.get("candidate", 0) or 0)
        feasible = int(update.get("feasible", 0) or 0)
        rejected = int(update.get("rejected", 0) or 0)
        if self._optimizer_stage_history and self._optimizer_stage_history[-1]["phase"] == phase:
            row = self._optimizer_stage_history[-1]
            row["candidate"] = candidate
            row["feasible"] = feasible
            row["rejected"] = rejected
            return

        if self._optimizer_stage_history:
            previous = self._optimizer_stage_history[-1]
            previous_candidate = int(previous.get("candidate", 0) or 0)
            previous_feasible = int(previous.get("feasible", 0) or 0)
            previous_rejected = int(previous.get("rejected", 0) or 0)
            base_candidate = previous_candidate if candidate >= previous_candidate else 0
            base_feasible = previous_feasible if feasible >= previous_feasible else 0
            base_rejected = previous_rejected if rejected >= previous_rejected else 0
        else:
            base_candidate = base_feasible = base_rejected = 0
        self._optimizer_stage_history.append({
            "phase": phase,
            "base_candidate": base_candidate,
            "base_feasible": base_feasible,
            "base_rejected": base_rejected,
            "candidate": candidate,
            "feasible": feasible,
            "rejected": rejected,
        })
        if len(self._optimizer_stage_history) > 24:
            self._optimizer_stage_history = self._optimizer_stage_history[-24:]

    def _optimizer_trace_node_count(self, node: dict[str, Any]) -> int:
        absolute = max(0, int(node.get("absolute_count", 0) or 0)) if "absolute_count" in node else 0
        count = max(0, int(node.get("count", 0) or 0))
        if node.get("status") == "active" and not node.get("countless"):
            base = node.get("base_candidate")
            last = node.get("last_candidate")
            if base is not None and last is not None:
                count += max(0, int(last) - int(base))
        count = max(count, absolute)
        if node.get("children") and not node.get("countless") and str(node.get("scope")) != "scout":
            child_sum = sum(self._optimizer_trace_node_count(child) for child in node.get("children", []))
            count = max(count, child_sum)
        return count

    def _optimizer_trace_node_elapsed(self, node: dict[str, Any]) -> float:
        elapsed = max(0.0, float(node.get("elapsed_accum", 0.0) or 0.0))
        started = node.get("segment_started")
        if node.get("status") == "active" and started is not None:
            elapsed += max(0.0, self._optimizer_elapsed_seconds() - float(started))
        if node.get("children") and elapsed <= 0.0:
            elapsed = sum(self._optimizer_trace_node_elapsed(child) for child in node.get("children", []))
        return elapsed

    def _optimizer_trace_node_progress_fraction(self, node: dict[str, Any]) -> float:
        """Return deterministic completion for one execution-plan node.

        Known work uses completed/total directly.  Unknown conditional work keeps
        its reserved share at zero while pending/active and resolves to one only
        when the branch completes or is skipped.  This deliberately avoids using
        elapsed time as fake progress.
        """
        status = str(node.get("status", "pending"))
        if status in {"complete", "skipped"}:
            return 1.0

        # When a node has an authoritative numeric workload/cap, prefer that
        # counter over child visitation state.  Progressive/revisited work must
        # follow the authoritative counter rather than treating "visited once"
        # as done, which could make a branch appear complete far too early.
        total = node.get("total")
        if total is not None:
            try:
                denominator = max(0, int(total))
            except (TypeError, ValueError):
                denominator = 0
            if denominator > 0:
                completed = self._optimizer_trace_node_count(node)
                fraction = max(0.0, min(1.0, float(completed) / float(denominator)))
                # An active branch cannot consume its final sliver until the
                # optimizer explicitly leaves/completes it.
                if status == "active":
                    fraction = min(fraction, 0.995)
                return fraction

        children = list(node.get("children", []) or [])
        if children:
            weights = [max(0.0, float(child.get("progress_weight", 1.0) or 0.0)) for child in children]
            weight_sum = sum(weights)
            if weight_sum <= 0.0:
                weights = [1.0] * len(children)
                weight_sum = float(len(children))
            return max(0.0, min(1.0, sum(
                weight * self._optimizer_trace_node_progress_fraction(child)
                for child, weight in zip(children, weights)
            ) / weight_sum))

        # Countless and conditionally-sized leaves have no honest denominator.
        # Do not manufacture time-based progress; completion/skipping above will
        # resolve their reserved share deterministically.
        return 0.0

    def _optimizer_trace_plan_progress_fraction(self) -> float:
        if not self._optimizer_trace_plan:
            return 0.0
        weights = [max(0.0, float(stage.get("progress_weight", 1.0) or 0.0)) for stage in self._optimizer_trace_plan]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            weights = [1.0] * len(self._optimizer_trace_plan)
            weight_sum = float(len(self._optimizer_trace_plan))
        return max(0.0, min(1.0, sum(
            weight * self._optimizer_trace_node_progress_fraction(stage)
            for stage, weight in zip(self._optimizer_trace_plan, weights)
        ) / weight_sum))

    def _refresh_optimizer_determinate_progress_bar(self, *, force_complete: bool = False) -> None:
        """Update the global bar from the execution-plan tree, monotonically."""
        raw_fraction = 1.0 if force_complete else self._optimizer_trace_plan_progress_fraction()
        if not force_complete:
            # Reserve the terminal 1% for the actual completion signal.  Adaptive
            # plans can revisit leaves, and the bar must never claim 100% while
            # the worker is still doing authoritative cleanup.
            raw_fraction = min(float(raw_fraction), 0.99)
        self._optimizer_progress_fraction = max(
            float(self._optimizer_progress_fraction or 0.0),
            max(0.0, min(1.0, float(raw_fraction))),
        )
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(int(round(1000.0 * self._optimizer_progress_fraction)))

        if force_complete:
            label = "Optimization complete"
        elif self._optimizer_cancel_requested:
            label = "Cancelling safely"
        elif self._optimizer_paused:
            label = "Paused"
        else:
            label = str(self._optimizer_last_progress.get("phase_label") or "Preparing search")
            detail = str(self._optimizer_last_progress.get("work_detail") or "").strip()
            if detail:
                label = f"{label} • {detail}"
        self.progress_bar.setFormat(f"%p% • {label}")

    def _optimizer_trace_count_text(self, node: dict[str, Any]) -> str:
        """Format one Search Trace work counter using a truthful live denominator.

        While work is active, retain the backend's useful plan/cap denominator (never
        allowing it to fall below work already observed).  Adaptive optimizer rows can
        finish with a different amount of work than was originally planned, so once a
        numeric row is complete its observed count becomes the final denominator.  A
        row that actually performed 12 probes therefore settles at ``12/12`` rather
        than either the stale ``12/4`` or the denominator-free ``12``.
        """
        status = str(node.get("status", "pending"))
        if node.get("countless"):
            return "done" if status == "complete" else "—" if status == "skipped" else "…"

        current = self._optimizer_trace_node_count(node)
        if status == "skipped":
            return "—"

        total = node.get("total")
        total_text = node.get("total_text")
        if total is not None:
            if status == "complete":
                # Completion turns an adaptive/provisional plan into an observed total.
                # Keep the ratio visible, but make it self-consistent: 14/14, 48/48,
                # 125/125, etc.  A safety-cap marker no longer applies after completion.
                return f"{current:,}/{current:,}" if current else "0/0"
            # Repeated/adaptive sub-passes can legitimately discover more unique work
            # than an earlier provisional denominator.  Never show an impossible
            # active ratio such as 14/2; until the backend supplies a newer total, the
            # work already observed is the minimum truthful denominator.
            denominator = max(current, max(0, int(total)))
            cap = "≤" if node.get("total_is_cap") else ""
            return f"{current:,}/{cap}{denominator:,}"
        if total_text:
            escaped = html.escape(str(total_text))
            return f"{current:,} • {escaped}" if current else escaped
        if status == "complete":
            return f"{current:,}" if current else "done"
        if status == "active":
            return f"{current:,}" if current else "…"
        return "—"

    def _optimizer_trace_time_text(self, node: dict[str, Any]) -> str:
        status = str(node.get("status", "pending"))
        if status in {"pending", "skipped"}:
            return "…" if status == "pending" else "—"
        return html.escape(_format_elapsed_time(self._optimizer_trace_node_elapsed(node)))

    def _refresh_optimizer_search_trace(self) -> None:
        if not hasattr(self, "sweep_plot"):
            return
        update = self._optimizer_last_progress
        phase = (
            "Cancelling safely"
            if self._optimizer_cancel_requested
            else "Paused"
            if self._optimizer_paused
            else str(update.get("phase_label") or "Preparing search")
        )
        evaluated = int(update.get("candidate", 0) or 0)
        feasible = int(update.get("feasible", 0) or 0)
        rejected = int(update.get("rejected", 0) or 0)
        workers = int(update.get("worker_count", self.optimizer_worker_count) or 1)
        elapsed = _format_elapsed_time(self._optimizer_elapsed_seconds())
        solution_stats = self._optimizer_solution_stats_text(update, elapsed=elapsed)
        best = self._optimizer_short_setting(update.get("best_setting"), limit=170)
        if current_theme() == "dark":
            plot_colours = plot_theme_colors("dark")
            trace_text = plot_colours["text"]
            trace_muted = plot_colours["reference"]
            trace_subtle = plot_colours["reference"]
            trace_border = plot_colours["major_grid"]
        else:
            trace_text = "#334155"
            trace_muted = "#64748b"
            trace_subtle = "#475569"
            trace_border = "#e2e8f0"

        metric_parts: list[str] = []
        metric_specs = (
            ("best_snapshot_match_coverage_pct", "match", "%"),
            ("best_snapshot_vector_rms_uT", "vector RMS", " µT"),
            ("best_snapshot_magnitude_rms_uT", "magnitude RMS", " µT"),
            ("best_snapshot_mean_angle_deg", "angle", "°"),
            ("best_coverage_pct", "coverage", "%"),
            ("best_uniformity_spread_pct", "uniformity", "%"),
            ("best_mean_uT", "mean", " µT"),
            ("best_rms_target_error_uT", "RMS", " µT"),
        )
        snapshot_mode = update.get("best_snapshot_match_coverage_pct") is not None
        for key, label, suffix in metric_specs:
            value = update.get(key)
            if value is None:
                continue
            if snapshot_mode and key in {"best_coverage_pct", "best_rms_target_error_uT"}:
                continue
            metric_parts.append(f"{label} {float(value):.5g}{suffix}")

        exposure_live_text = ""
        exposure_best_text = ""
        if bool(update.get("exposure_registration_enabled", False)):
            exposure_candidates = int(update.get("exposure_registration_candidates", 0) or 0)
            exposure_evaluations = int(update.get("exposure_registration_evaluations", 0) or 0)
            exposure_basis_builds = int(update.get("exposure_registration_basis_builds", 0) or 0)
            exposure_cache_hits = int(update.get("exposure_registration_cache_hits", 0) or 0)
            exposure_kabsch = int(update.get("exposure_registration_kabsch_seeds", 0) or 0)
            exposure_inherited = int(update.get("exposure_registration_inherited_seeds", 0) or 0)
            exposure_coarse = int(update.get("exposure_registration_coarse_candidates", 0) or 0)
            exposure_standard = int(update.get("exposure_registration_standard_candidates", 0) or 0)
            exposure_refined = int(update.get("exposure_registration_refined_candidates", 0) or 0)
            exposure_seconds = float(update.get("exposure_registration_seconds", 0.0) or 0.0)
            exposure_live_text = (
                f"Exposure registration • {exposure_candidates:,} structural solve{'s' if exposure_candidates != 1 else ''} • "
                f"{exposure_evaluations:,} pose evaluation{'s' if exposure_evaluations != 1 else ''} • "
                f"{exposure_basis_builds:,} extra field-basis build{'s' if exposure_basis_builds != 1 else ''} • "
                f"{exposure_cache_hits:,} registration-cache hit{'s' if exposure_cache_hits != 1 else ''} • "
                f"{exposure_kabsch:,} Kabsch / {exposure_inherited:,} inherited seed{'s' if exposure_inherited != 1 else ''} • "
                f"effort {exposure_coarse:,} coarse / {exposure_standard:,} standard / {exposure_refined:,} refined • "
                f"{_format_elapsed_time(exposure_seconds)} inner-solve time"
            )
            rx = update.get("best_snapshot_alignment_rx_deg")
            ry = update.get("best_snapshot_alignment_ry_deg")
            rz = update.get("best_snapshot_alignment_rz_deg")
            if rx is not None and ry is not None and rz is not None:
                exposure_best_text = (
                    f"exposure registration RX {float(rx):.4g}° • "
                    f"RY {float(ry):.4g}° • RZ {float(rz):.4g}°"
                )

        icon = {"pending": "○", "active": "▶", "complete": "✓", "skipped": "–"}
        rows: list[str] = []
        for stage in self._optimizer_trace_plan:
            stage_status = str(stage.get("status", "pending"))
            rows.append(
                "<tr style='font-weight:600'>"
                f"<td style='padding:3px 6px 2px 0;white-space:nowrap'>{icon.get(stage_status, '○')}</td>"
                f"<td style='padding:3px 8px 2px 0'>{html.escape(str(stage.get('label', 'Stage')))}</td>"
                f"<td style='padding:3px 12px 2px 8px;text-align:right;white-space:nowrap'>{self._optimizer_trace_count_text(stage)}</td>"
                f"<td style='padding:3px 0 2px 8px;text-align:right;white-space:nowrap'>{self._optimizer_trace_time_text(stage)}</td>"
                "</tr>"
            )
            for child in stage.get("children", []):
                child_status = str(child.get("status", "pending"))
                rows.append(
                    "<tr>"
                    f"<td style='padding:1px 6px 1px 0;white-space:nowrap;color:{trace_muted}'>{icon.get(child_status, '○')}</td>"
                    f"<td style='padding:1px 8px 1px 14px;color:{trace_subtle}'>↳ {html.escape(str(child.get('label', 'Step')))}</td>"
                    f"<td style='padding:1px 12px 1px 8px;text-align:right;white-space:nowrap;color:{trace_subtle}'>{self._optimizer_trace_count_text(child)}</td>"
                    f"<td style='padding:1px 0 1px 8px;text-align:right;white-space:nowrap;color:{trace_muted}'>{self._optimizer_trace_time_text(child)}</td>"
                    "</tr>"
                )

        best_html = "Waiting for the first feasible candidate…"
        if best:
            best_html = f"<b>{html.escape(best)}</b>"
            if metric_parts:
                best_html += f" <span style='color:{trace_muted}'>• {html.escape(' • '.join(metric_parts))}</span>"
            if exposure_best_text:
                best_html += f"<br><span style='color:{trace_muted}'>{html.escape(exposure_best_text)}</span>"

        exposure_live_html = ""
        if exposure_live_text:
            exposure_live_html = (
                f"<div style='margin-top:7px;padding-top:6px;border-top:1px solid {trace_border};"
                f"font-size:11px;color:{trace_subtle}'>"
                f"<span style='color:{trace_muted};text-transform:uppercase;letter-spacing:.05em'>Inner solve</span> &nbsp; "
                f"{html.escape(exposure_live_text)}</div>"
            )

        body = f"""
        <div style='font-family:sans-serif;color:{trace_text};padding:8px 14px 7px 14px'>
          <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px'>
            <div><span style='font-size:12px;color:{trace_muted};text-transform:uppercase;letter-spacing:.08em'>Search plan</span>
            <b style='margin-left:9px'>{html.escape(phase)}</b></div>
            <div style='font-size:12px;color:{trace_muted}'>{html.escape(solution_stats)}</div>
          </div>
          <table style='border-collapse:collapse;width:100%;font-size:11px;line-height:1.08'>
            <tr style='color:{trace_muted};border-bottom:1px solid {trace_border}'>
              <th style='width:18px'></th><th style='text-align:left;padding:3px 8px 4px 0'>Execution tree</th>
              <th style='text-align:right;padding:3px 12px 4px 8px'>Work</th><th style='text-align:right;padding:3px 0 4px 8px'>Time</th>
            </tr>
            {''.join(rows)}
          </table>
          {exposure_live_html}
          <div style='margin-top:7px;padding-top:6px;border-top:1px solid {trace_border};font-size:11px;color:{trace_subtle}'>
            <span style='color:{trace_muted};text-transform:uppercase;letter-spacing:.05em'>Best so far</span> &nbsp; {best_html}
          </div>
        </div>
        """
        self.sweep_plot.message.setHtml(body)
        self.sweep_plot.setCurrentWidget(self.sweep_plot.message)

    def apply_workbench_theme(self, _theme: str) -> None:
        """Refresh the optimizer's HTML execution trace without changing tabs."""

        if (
            hasattr(self, "sweep_plot")
            and self.sweep_plot.currentWidget() is self.sweep_plot.message
        ):
            self._refresh_optimizer_search_trace()

    def _optimizer_elapsed_seconds(self) -> float:
        if self._optimizer_elapsed_override is not None:
            return max(0.0, float(self._optimizer_elapsed_override))
        if self._optimizer_run_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._optimizer_run_started_at)

    @staticmethod
    def _optimizer_short_setting(value: Any, limit: int = 180) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    @staticmethod
    def _optimizer_solution_stats_text(update: dict[str, Any], *, elapsed: str) -> str:
        """Format durable master-registry progress instead of stage-local counters."""
        registry_keys = (
            "solutions_discovered",
            "solutions_active",
            "solutions_retired",
            "solutions_blocked",
            "solutions_authoritative",
        )
        if any(key in update for key in registry_keys):
            discovered = int(update.get("solutions_discovered", 0) or 0)
            active = int(update.get("solutions_active", 0) or 0)
            retired = int(update.get("solutions_retired", 0) or 0)
            blocked = int(update.get("solutions_blocked", 0) or 0)
            authoritative = int(update.get("solutions_authoritative", 0) or 0)
            return (
                f"{discovered:,} solutions discovered • {active:,} active • "
                f"{retired:,} retired • {blocked:,} blocked • "
                f"{authoritative:,} authoritative • {elapsed}"
            )

        # Compatibility for legacy/programmatic progress sources that predate the
        # flat candidate registry. Normal Optimizer 2.2 runs always use the branch
        # above.
        evaluated = int(update.get("candidate", 0) or 0)
        feasible = int(update.get("feasible", 0) or 0)
        rejected = int(update.get("rejected", 0) or 0)
        return (
            f"{evaluated:,} evaluated • {feasible:,} feasible • "
            f"{rejected:,} rejected • {elapsed}"
        )

    def _refresh_optimizer_progress_display(self) -> None:
        update = self._optimizer_last_progress
        elapsed = _format_elapsed_time(self._optimizer_elapsed_seconds())
        if self._optimizer_cancel_requested:
            phase = "Cancelling safely"
        elif self._optimizer_paused:
            phase = "Paused"
        else:
            phase = str(update.get("phase_label") or "Preparing search")
            detail = str(update.get("work_detail") or "").strip()
            if detail:
                phase = f"{phase} • {detail}"

        solution_stats = self._optimizer_solution_stats_text(update, elapsed=elapsed)
        self.progress_status.setText(f"{phase} • {solution_stats}")
        self._refresh_optimizer_determinate_progress_bar()

        best = self._optimizer_short_setting(update.get("best_setting"))
        if not best:
            latest_rejection = self._optimizer_short_setting(update.get("last_rejection"), limit=220)
            if latest_rejection:
                self.progress_best.setText(
                    "No feasible candidate yet. Latest rejection: " + latest_rejection
                )
            else:
                self.progress_best.setText("Waiting for the first feasible candidate…")
            self._refresh_optimizer_search_trace()
            return
        metrics: list[str] = []
        coverage = update.get("best_coverage_pct")
        uniformity = update.get("best_uniformity_spread_pct")
        mean_uT = update.get("best_mean_uT")
        rms_uT = update.get("best_rms_target_error_uT")
        direction = update.get("best_directional_consistency_pct")
        snapshot_match = update.get("best_snapshot_match_coverage_pct")
        snapshot_vector_rms = update.get("best_snapshot_vector_rms_uT")
        snapshot_magnitude_rms = update.get("best_snapshot_magnitude_rms_uT")
        snapshot_angle = update.get("best_snapshot_mean_angle_deg")
        alignment_rx = update.get("best_snapshot_alignment_rx_deg")
        alignment_ry = update.get("best_snapshot_alignment_ry_deg")
        alignment_rz = update.get("best_snapshot_alignment_rz_deg")
        if snapshot_match is not None:
            metrics.append(f"snapshot match {float(snapshot_match):.5g}%")
        if snapshot_vector_rms is not None:
            metrics.append(f"vector RMS {float(snapshot_vector_rms):.5g} µT")
        if snapshot_magnitude_rms is not None:
            metrics.append(f"magnitude RMS {float(snapshot_magnitude_rms):.5g} µT")
        if snapshot_angle is not None:
            metrics.append(f"angle {float(snapshot_angle):.5g}°")
        if (
            bool(update.get("exposure_registration_enabled", False))
            and alignment_rx is not None
            and alignment_ry is not None
            and alignment_rz is not None
        ):
            metrics.append(
                "registration "
                f"({float(alignment_rx):.4g}°, {float(alignment_ry):.4g}°, {float(alignment_rz):.4g}°)"
            )
        if coverage is not None and snapshot_match is None:
            metrics.append(f"coverage {float(coverage):.5g}%")
        if uniformity is not None:
            metrics.append(f"uniformity {float(uniformity):.5g}%")
        if mean_uT is not None:
            metrics.append(f"mean {float(mean_uT):.6g} µT")
        if rms_uT is not None and snapshot_match is None:
            metrics.append(f"RMS {float(rms_uT):.5g} µT")
        if direction is not None and snapshot_match is None:
            metrics.append(f"direction {float(direction):.5g}%")
        suffix = " • " + " • ".join(metrics) if metrics else ""
        self.progress_best.setText(f"Best so far: {best}{suffix}")
        self._refresh_optimizer_search_trace()

    def _set_running(self, running: bool) -> None:
        self.pause_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        self.back_button.setEnabled(not running and self.pages.currentIndex() > self.PAGE_START)
        self.next_button.setEnabled(not running and self.pages.currentIndex() != self.PAGE_RESULTS)
        for index in range(self.PAGE_START, self.PAGE_REVIEW + 1):
            widget = self.pages.widget(index)
            if widget is not None:
                widget.setEnabled(not running)
        if not running:
            self._update_navigation()

    def _start_search(self) -> None:
        for page in (self.PAGE_TARGET, self.PAGE_FREEDOM):
            error = self._validate_page(page)
            if error:
                QMessageBox.information(self, "Optimizer setup", error)
                self.pages.setCurrentIndex(page)
                return
        if self.thread is not None:
            return
        self._refresh_review()
        settings = self._wizard_settings()
        settings["optimization_plan"] = self.review_text.toPlainText().strip()
        self._optimizer_active_settings = copy.deepcopy(settings)
        self._refresh_results_context(settings=settings)
        self._initialize_optimizer_trace_plan(settings)
        self.run_result = None
        # Every run owns a fresh registry/result-browser state.  This matters when
        # the same optimizer window is adjusted and run again after completion.
        self._registry_records_by_id.clear()
        self._registry_views.clear()
        self._materialized_candidate_cache.clear()
        self._registry_source_transform_cache.clear()
        self.result_view_list.clear()
        self.result_tree.clear()
        self._optimizer_stage_history = []
        self._set_optimizer_result_tab_mode(searching=True)
        self.result_summary.setText("Evaluating the reviewed optimization plan…")
        self._optimizer_run_started_at = time.monotonic()
        self._optimizer_elapsed_override = None
        self._optimizer_last_progress = {
            "phase_label": "Preparing frozen scene and target samples",
            "candidate": 0,
            "feasible": 0,
            "rejected": 0,
            "solutions_discovered": 0,
            "solutions_active": 0,
            "solutions_retired": 0,
            "solutions_blocked": 0,
            "solutions_authoritative": 0,
        }
        self._optimizer_paused = False
        self._optimizer_cancel_requested = False
        self._record_optimizer_stage(self._optimizer_last_progress)
        self._update_optimizer_trace_plan(self._optimizer_last_progress)
        self._optimizer_progress_timer.start()
        self._optimizer_progress_fraction = 0.0
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% • Preparing search")
        self._refresh_optimizer_progress_display()
        self.pages.setCurrentIndex(self.PAGE_RESULTS)
        self._set_running(True)

        self.thread = QThread(self)
        self.worker = OptimizerWorker(
            self.source_document,
            settings,
            coil_modeling_method=self.coil_modeling_method,
            solver_memory_budget_mb=self.solver_memory_budget_mb,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(dict)
    def _progress(self, update: dict) -> None:
        if not self._optimizer_trace_plan:
            # Primarily useful for tests/programmatic progress injection; normal
            # searches initialize the plan before the worker starts.
            self._initialize_optimizer_trace_plan(self._wizard_settings())
        # The global bar follows the persistent execution plan.  Known branch
        # denominators fill continuously; conditional work keeps a reserved
        # slice until it is completed or skipped.
        merged_update = dict(update)
        for key in (
            "solutions_discovered",
            "solutions_active",
            "solutions_retired",
            "solutions_blocked",
            "solutions_authoritative",
        ):
            if key not in merged_update and key in self._optimizer_last_progress:
                merged_update[key] = self._optimizer_last_progress[key]
        self._optimizer_last_progress = merged_update
        self._refresh_results_context(
            settings=self._optimizer_active_settings, progress=self._optimizer_last_progress
        )
        self._record_optimizer_stage(self._optimizer_last_progress)
        self._update_optimizer_trace_plan(self._optimizer_last_progress)
        self._refresh_optimizer_progress_display()

    @Slot(dict)
    def _completed(self, result: dict) -> None:
        self.run_result = result
        # Normal runs already moved to Results before starting the worker, but
        # completion is also a public Qt slot and is exercised directly by UI
        # harnesses.  Keep its postcondition self-contained: Results owns the
        # window and wizard-only explanatory chrome must be hidden.
        if self.pages.currentIndex() != self.PAGE_RESULTS:
            self.pages.setCurrentIndex(self.PAGE_RESULTS)
        else:
            self._set_optimizer_results_chrome(compact=True)
            self._update_navigation()
        self._refresh_results_context(result=result)
        self._optimizer_elapsed_override = float(result.get("elapsed_seconds", self._optimizer_elapsed_seconds()))
        self._optimizer_progress_timer.stop()
        self._optimizer_paused = False
        self._optimizer_cancel_requested = False
        self._refresh_optimizer_determinate_progress_bar(force_complete=True)
        self.progress_bar.setFormat(
            f"100% • Optimization complete — {_format_elapsed_time(self._optimizer_elapsed_seconds())}"
        )
        finalists = result.get("finalists", [])
        registry_has_solutions = bool((result.get("candidate_registry", {}) or {}).get("candidates", []))
        self._set_optimizer_result_tab_mode(
            searching=False, has_finalist=bool(finalists) or registry_has_solutions
        )
        snapshot_mode = str(result.get("reference_mode", "target")) == "snapshot"
        if snapshot_mode:
            reference = result.get("reference_snapshot", {}) or {}
            reference_name = str(reference.get("name", reference.get("id", "reference snapshot")))
            self.compare_button.setToolTip(
                f"Compare the selected optimizer solution(s) against {reference_name} in the normal Field Snapshot Comparison."
            )
        else:
            target = result.get("target", {}) or {}
            try:
                target_text = f"{float(target.get('target_uT')):.6g} µT ± {float(target.get('tolerance_pct')):.6g}%"
            except (TypeError, ValueError):
                target_text = "the optimization field specification"
            self.compare_button.setToolTip(
                f"Benchmark the selected optimizer solution(s) against {target_text} in the normal Field Snapshot Comparison."
            )
        if snapshot_mode:
            self.result_tree.setHeaderLabels(
                [
                    "Rank",
                    "Setting",
                    "Resulting currents",
                    "Match",
                    "Uniformity",
                    "Angle error",
                    "Mean",
                    "Snapshot RMS",
                    "DRC clearance",
                ]
            )
        else:
            self.result_tree.setHeaderLabels(
                [
                    "Rank",
                    "Setting",
                    "Resulting currents",
                    "Coverage",
                    "Uniformity",
                    "Direction",
                    "Mean",
                    "RMS error",
                    "DRC clearance",
                ]
            )
        self.result_tree.clear()
        for rank, candidate in enumerate(finalists, 1):
            metrics = candidate["metrics"]
            clearance = metrics.get("minimum_clearance_mm")
            currents = ", ".join(
                f"{1000.0 * float(value):.5g} mA"
                for value in candidate.get("currents_a", {}).values()
            ) or "—"
            setting = str(candidate.get("setting_summary", "")).strip()
            if not setting:
                if candidate.get("tuning_mode") == "current":
                    setting = currents
                else:
                    setting = (
                        f"{float(candidate.get('display_parameter_value', 0.0)):.6g} "
                        f"{candidate.get('parameter_unit', '')}"
                    ).strip()
            if snapshot_mode:
                objective = str(result.get("objective", "snapshot_vector_match"))
                snapshot_rms = (
                    float(metrics.get("snapshot_magnitude_rms_uT", math.nan))
                    if objective == "snapshot_magnitude_match"
                    else float(metrics.get("snapshot_vector_rms_uT", math.nan))
                )
                row_values = [
                    str(rank),
                    setting,
                    currents,
                    f"{float(metrics.get('snapshot_match_coverage_pct', math.nan)):.5g}%",
                    f"{float(metrics['uniformity_spread_pct']):.5g}%",
                    f"{float(metrics.get('snapshot_mean_angle_deg', math.nan)):.5g}°",
                    f"{float(metrics['mean_uT']):.6g} µT",
                    f"{snapshot_rms:.6g} µT",
                    "—" if clearance is None else f"{float(clearance):.5g} mm",
                ]
            else:
                row_values = [
                    str(rank),
                    setting,
                    currents,
                    f"{float(metrics['in_band_pct']):.5g}%",
                    f"{float(metrics['uniformity_spread_pct']):.5g}%",
                    f"{float(metrics.get('directional_consistency_pct', math.nan)):.5g}%",
                    f"{float(metrics['mean_uT']):.6g} µT",
                    f"{float(metrics['rms_target_error_uT']):.6g} µT",
                    "—" if clearance is None else f"{float(clearance):.5g} mm",
                ]
            item = QTreeWidgetItem(row_values)
            item.setData(0, Qt.ItemDataRole.UserRole, candidate)
            family_variants = list(candidate.get("field_equivalent_variants", []) or [])
            if family_variants:
                family_text = (
                    f"{1 + len(family_variants)} field-equivalent turn/current realizations. "
                    "Copy report for the electrical alternatives."
                )
                item.setToolTip(1, family_text)
                item.setToolTip(2, family_text)
            self.result_tree.addTopLevelItem(item)
        for column in range(self.result_tree.columnCount()):
            self.result_tree.resizeColumnToContents(column)
        self._resize_result_tree_to_content(len(finalists))
        if finalists:
            best = finalists[0]
            metrics = best["metrics"]
            parameter_count = int(result.get("parameter_count", len(result.get("variables", [])) or 1))
            registry_payload = result.get("candidate_registry", {}) or {}
            registry_solution_count = int(
                registry_payload.get("solution_count", registry_payload.get("candidate_count", 0)) or 0
            )
            lifecycle_counts = registry_payload.get("lifecycle_counts", {}) or {}
            authoritative_count = int(
                lifecycle_counts.get(
                    "solutions_authoritative", result.get("generated_candidate_count", 0)
                )
                or 0
            )
            if registry_solution_count:
                search_description = (
                    f"{authoritative_count} authoritative settings • "
                    f"{registry_solution_count:,} solutions explored across "
                    f"{parameter_count} free parameter{'s' if parameter_count != 1 else ''}"
                )
            else:
                search_description = (
                    f"{result.get('generated_candidate_count', 0)} tested settings across "
                    f"{parameter_count} free parameter{'s' if parameter_count != 1 else ''}"
                )
            reduction = result.get("search_reduction", {}) or {}
            if reduction.get("enabled"):
                initial_structural = int(reduction.get("initial_structural_count", 0) or 0)
                active_structural = int(reduction.get("active_structural_count", 0) or 0)
                search_description += (
                    f" • structural screening {initial_structural}→{active_structural} active"
                )
            distinct_count = int(result.get("distinct_finalist_count", len(finalists)) or 0)
            feasible_count = int(result.get("feasible_candidate_count", len(finalists)) or 0)
            finalist_text = (
                f"{distinct_count} distinct finalist{'s' if distinct_count != 1 else ''} from "
                f"{feasible_count} feasible settings"
            )
            finalist_filter = result.get("finalist_filter", {}) or {}
            if bool(finalist_filter.get("limit_reached")):
                finalist_limit = int(finalist_filter.get("finalist_limit", distinct_count) or distinct_count)
                finalist_text += f" (top {min(distinct_count, finalist_limit)} retained)"
            convergence = result.get("convergence", {}) or {}
            convergence_text = (
                "converged" if convergence.get("converged") else "convergence incomplete"
            )
            if snapshot_mode:
                self.result_summary.setText(
                    f"Best of {search_description} • {finalist_text} • {result['final_sample_count']} corresponding samples • "
                    f"{result['rejected_candidate_count']} rejected by active gates • {convergence_text} • snapshot match "
                    f"{float(metrics.get('snapshot_match_coverage_pct', math.nan)):.5g}% • vector RMS "
                    f"{float(metrics.get('snapshot_vector_rms_uT', math.nan)):.5g} µT • mean angle "
                    f"{float(metrics.get('snapshot_mean_angle_deg', math.nan)):.5g}° • {result['elapsed_seconds']:.3g} s"
                )
            else:
                self.result_summary.setText(
                    f"Best of {search_description} • {finalist_text} • {result['final_sample_count']} samples across "
                    f"{len(result.get('targets', []))} target(s) • {result['rejected_candidate_count']} rejected by "
                    f"active gates • {convergence_text} • coverage {float(metrics['in_band_pct']):.5g}% • uniformity spread "
                    f"{float(metrics['uniformity_spread_pct']):.5g}% • {result['elapsed_seconds']:.3g} s"
                )
        else:
            diagnostics = result.get("failure_diagnostics", {}) or {}
            reasons = list(diagnostics.get("rejection_reasons", []) or [])
            if reasons:
                top_reason = reasons[0]
                self.result_summary.setText(
                    f"No feasible finalist was found after {int(result.get('generated_candidate_count', 0) or 0)} tested settings. "
                    f"Most common rejection: {int(top_reason.get('count', 0))} × {top_reason.get('reason', 'Candidate rejected.')} "
                    "Copy report for the full rejection diagnostics."
                )
            else:
                self.result_summary.setText(
                    "No feasible finalist was found. Copy report for the optimization plan and available diagnostics."
                )
        registry_payload = result.get("candidate_registry", {}) or {}
        registry_records = list(registry_payload.get("candidates", []) or [])
        if registry_records:
            self._load_registry_results(result)
        else:
            # Legacy/programmatic result payloads can still contain finalists but no
            # master registry. Keep their old finalist table usable without showing
            # an empty Result views sidebar or leaking a previous run's registry.
            self._registry_records_by_id.clear()
            self._registry_views.clear()
            self._materialized_candidate_cache.clear()
            self._registry_source_transform_cache.clear()
            self.result_view_list.clear()
            self.result_view_panel.hide()

        diagnostics = result.get("failure_diagnostics", {}) or {}
        lifecycle_counts = registry_payload.get("lifecycle_counts", {}) or {}
        self._optimizer_last_progress = {
            "phase_label": "Optimization complete",
            "candidate": int(result.get("generated_candidate_count", 0) or 0),
            "feasible": int(result.get("feasible_candidate_count", 0) or 0),
            "rejected": int(result.get("rejected_candidate_count", 0) or 0),
            "solutions_discovered": int(lifecycle_counts.get("solutions_discovered", registry_payload.get("solution_count", 0)) or 0),
            "solutions_active": int(lifecycle_counts.get("solutions_active", 0) or 0),
            "solutions_retired": int(lifecycle_counts.get("solutions_retired", 0) or 0),
            "solutions_blocked": int(lifecycle_counts.get("solutions_blocked", 0) or 0),
            "solutions_authoritative": int(lifecycle_counts.get("solutions_authoritative", 0) or 0),
            "last_rejection": diagnostics.get("last_rejection"),
            "rejection_reason_counts": {
                str(item.get("reason", "Candidate rejected.")): int(item.get("count", 0))
                for item in diagnostics.get("rejection_reasons", []) or []
            },
        }
        if finalists:
            best_metrics = finalists[0].get("metrics", {})
            self._optimizer_last_progress.update({
                "best_setting": finalists[0].get("setting_summary"),
                "best_coverage_pct": best_metrics.get("in_band_pct"),
                "best_uniformity_spread_pct": best_metrics.get("uniformity_spread_pct"),
                "best_mean_uT": best_metrics.get("mean_uT"),
                "best_rms_target_error_uT": best_metrics.get("rms_target_error_uT"),
                "best_directional_consistency_pct": best_metrics.get("directional_consistency_pct"),
                "best_snapshot_match_coverage_pct": best_metrics.get("snapshot_match_coverage_pct"),
                "best_snapshot_vector_rms_uT": best_metrics.get("snapshot_vector_rms_uT"),
                "best_snapshot_magnitude_rms_uT": best_metrics.get("snapshot_magnitude_rms_uT"),
                "best_snapshot_mean_angle_deg": best_metrics.get("snapshot_mean_angle_deg"),
            })
        self._record_optimizer_stage(self._optimizer_last_progress)
        self._complete_optimizer_trace_plan(result)
        self._refresh_optimizer_progress_display()
        if self.result_tree.topLevelItemCount():
            self.result_tree.setCurrentItem(self.result_tree.topLevelItem(0))
        self._result_selection_changed()
        self.result_views.setCurrentWidget(
            self.candidate_plot if (finalists or registry_has_solutions) else self.sweep_plot
        )
        # Shrink a formerly tall wizard to a taskbar-safe result window. Window
        # flags are intentionally left alone here: changing native flags on an
        # already-visible Qt window can itself hide/recreate the Win32 surface.
        QTimer.singleShot(0, self._fit_completed_results_to_screen)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self._optimizer_elapsed_override = self._optimizer_elapsed_seconds()
        self._optimizer_progress_timer.stop()
        self._optimizer_paused = False
        self._optimizer_cancel_requested = False
        self._refresh_optimizer_determinate_progress_bar()
        self.progress_bar.setFormat("%p% • Optimization failed")
        self.result_summary.setText("The optimization did not complete; the frozen source and live scenes are unchanged.")
        self.progress_status.setText(
            f"Optimization failed • elapsed {_format_elapsed_time(self._optimizer_elapsed_seconds())}"
        )
        self.progress_best.setText("")
        self._optimizer_last_progress["phase_label"] = "Optimization failed"
        self._record_optimizer_stage(self._optimizer_last_progress)
        self._complete_optimizer_trace_plan(cancelled=True)
        self._set_optimizer_result_tab_mode(searching=True)
        self._refresh_optimizer_search_trace()
        QMessageBox.warning(self, "Optimizer", message)

    @Slot()
    def _cancelled(self) -> None:
        self._optimizer_elapsed_override = self._optimizer_elapsed_seconds()
        self._optimizer_progress_timer.stop()
        self._optimizer_paused = False
        self._optimizer_cancel_requested = False
        self._refresh_optimizer_determinate_progress_bar()
        self.progress_bar.setFormat("%p% • Cancelled — source scene unchanged")
        self.result_summary.setText("Search cancelled. The source scene was not modified.")
        self.progress_status.setText(
            f"Cancelled • elapsed {_format_elapsed_time(self._optimizer_elapsed_seconds())}"
        )
        self.progress_best.setText("")
        self._optimizer_last_progress["phase_label"] = "Cancelled"
        self._record_optimizer_stage(self._optimizer_last_progress)
        self._complete_optimizer_trace_plan(cancelled=True)
        self._set_optimizer_result_tab_mode(searching=True)
        self._refresh_optimizer_search_trace()

    @Slot()
    def _thread_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.pause_button.setText("Pause")
        self._set_running(False)
        if self._close_after_thread:
            self._close_after_thread = False
            self.close()

    def _toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self.worker.pause_event.is_set():
            self.worker.pause_event.clear()
            self._optimizer_paused = False
            self.pause_button.setText("Pause")
            self._refresh_optimizer_determinate_progress_bar()
        else:
            self.worker.pause_event.set()
            self._optimizer_paused = True
            self.pause_button.setText("Resume")
            self._refresh_optimizer_determinate_progress_bar()
        self._refresh_optimizer_progress_display()

    def _cancel_search(self) -> None:
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.worker.pause_event.clear()
            self._optimizer_paused = False
            self._optimizer_cancel_requested = True
            self._refresh_optimizer_determinate_progress_bar()
            self._refresh_optimizer_progress_display()

    def reject(self) -> None:
        if self.thread is not None:
            self._close_after_thread = True
            self._cancel_search()
            return
        self.close()

    @staticmethod
    def _registry_solution_label(solution_id: str) -> str:
        solution_id = str(solution_id)
        if solution_id.startswith("candidate_"):
            tail = solution_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                return f"Solution {int(tail):,}"
        return solution_id

    def _registry_preferred_evaluation(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return the same preferred observation used by backend ranking/materialization."""
        if self.run_result is None:
            return None
        try:
            return self.preview_adapter._optimizer_registry_preferred_evaluation(
                self.run_result, record
            )
        except StudioOperationError:
            return None

    def _registry_display_evaluation(
        self, solution_id: str, record: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return a display observation, with finalist payload as a safe fallback.

        The completed result already carries full authoritative finalist candidates.
        The registry is the normal source of comparison metrics, but the Finalists
        view should never become blank merely because a registry observation is
        missing/malformed or was emitted by an older result payload.  Falling back
        to the finalist candidate also keeps the initial result selection usable
        while preserving the registry as the canonical lifecycle store.
        """
        evaluation = self._registry_preferred_evaluation(record)
        if evaluation is not None or self.run_result is None:
            return evaluation
        for finalist in list(self.run_result.get("finalists", []) or []):
            if str(finalist.get("registry_candidate_id", "")) != str(solution_id):
                continue
            metrics = finalist.get("metrics")
            if not isinstance(metrics, dict):
                continue
            return {
                "stage": "authoritative finalist",
                "fidelity": "authoritative",
                "fidelity_rank": 40,
                "model": str(finalist.get("model", "auto") or "auto"),
                "sample_count": int(
                    finalist.get("sample_count")
                    or self.run_result.get("final_sample_count", 0)
                    or 0
                ),
                "feasible": True,
                "metrics": metrics,
                "currents_a": dict(finalist.get("currents_a", {}) or {}),
            }
        return None

    @staticmethod
    def _registry_solution_compact_label(solution_id: str) -> str:
        """Return the compact scientific-style label used in the result explorer."""
        solution_id = str(solution_id)
        if solution_id.startswith("candidate_"):
            tail = solution_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                return f"Sⁿ #{int(tail):,}"
        return solution_id

    @staticmethod
    def _result_metric_text(value: Any, *, suffix: str = "", precision: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(number):
            return "—"
        return f"{number:.{precision}g}{suffix}"

    def _configure_registry_result_headers(self) -> None:
        if self.run_result is None:
            return
        snapshot_mode = str(self.run_result.get("reference_mode", "target")) == "snapshot"
        if snapshot_mode:
            labels = [
                "Solution ID",
                "Match %",
                "Vector RMS",
                "Magnitude RMS",
                "Mean angle",
                "Mean |B|",
                "DRC clearance",
                "Fidelity",
            ]
        else:
            labels = [
                "Solution ID",
                "Coverage %",
                "RMS error",
                "Uniformity",
                "Direction",
                "Mean |B|",
                "DRC clearance",
                "Fidelity",
            ]
        # QTreeWidget.setHeaderLabels() does not reliably shrink an existing
        # column count. The legacy finalist table used nine columns, while the
        # registry explorer uses eight; reset explicitly so a stale duplicate
        # "DRC clearance" column cannot survive completion.
        self.result_tree.setColumnCount(len(labels))
        self.result_tree.setHeaderLabels(labels)
        header = self.result_tree.header()
        header.setStretchLastSection(False)
        for column in range(len(labels)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        # Comparison metrics should dominate the horizontal space; structural
        # details now live under each expandable solution instead of in a very
        # wide one-line Setting column.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

    def _load_registry_results(self, result: dict[str, Any]) -> None:
        payload = result.get("candidate_registry", {}) or {}
        records = list(payload.get("candidates", []) or [])
        self._registry_records_by_id = {
            str(record.get("id", "")): record
            for record in records if str(record.get("id", ""))
        }
        self._registry_views = {
            str(key): [str(value) for value in list(values or [])]
            for key, values in dict(payload.get("views", {}) or {}).items()
        }
        self._materialized_candidate_cache.clear()
        self._registry_source_transform_cache.clear()
        self.result_view_list.blockSignals(True)
        self.result_view_list.clear()
        specs = (
            ("finalists", "★ Finalists"),
            ("best_feasible", "✓ Best feasible"),
            ("best_overall", "◎ Best overall"),
            ("drc_boundary", "◇ DRC boundary"),
            ("source_baseline", "⌂ Source baseline"),
            ("strong_basins", "⌁ Strong basins"),
        )
        for key, label in specs:
            ids = [value for value in self._registry_views.get(key, []) if value in self._registry_records_by_id]
            item = QListWidgetItem(f"{label}   {len(ids)}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.result_view_list.addItem(item)
        all_item = QListWidgetItem(f"All solutions   {len(self._registry_records_by_id)}")
        all_item.setData(Qt.ItemDataRole.UserRole, "all")
        self.result_view_list.addItem(all_item)
        self.result_view_list.blockSignals(False)
        self._configure_registry_result_headers()
        # Prefer finalists, otherwise the strongest feasible/overall view with data.
        preferred_keys = ("finalists", "best_feasible", "best_overall", "source_baseline", "all")
        selected_row = 0
        for key in preferred_keys:
            for index in range(self.result_view_list.count()):
                item = self.result_view_list.item(index)
                if str(item.data(Qt.ItemDataRole.UserRole)) != key:
                    continue
                ids = self._registry_view_solution_ids(key)
                if ids:
                    selected_row = index
                    break
            else:
                continue
            break
        if self.result_view_list.count():
            # Do not depend on QListWidget emitting currentItemChanged here. On
            # some Qt/Windows runs the freshly rebuilt list can retain an internal
            # row state and the initial Finalists view never receives the signal,
            # leaving the tree empty and the plot on "Preparing the viewer…".
            # Select silently, then populate the chosen registry view explicitly.
            self.result_view_list.blockSignals(True)
            self.result_view_list.setCurrentRow(selected_row)
            current = self.result_view_list.currentItem()
            self.result_view_list.blockSignals(False)
            if current is not None:
                self._result_registry_view_changed(current)
        else:
            self.result_tree.clear()
            self._result_selection_changed()

    def _registry_view_solution_ids(self, key: str) -> list[str]:
        if str(key) == "all":
            return list(self._registry_records_by_id)
        return [
            value for value in self._registry_views.get(str(key), [])
            if value in self._registry_records_by_id
        ]

    def _result_registry_view_changed(self, current, _previous=None) -> None:
        if current is None:
            return
        key = str(current.data(Qt.ItemDataRole.UserRole) or "")
        self._populate_registry_result_tree(self._registry_view_solution_ids(key))

    def _populate_registry_result_tree(self, solution_ids: list[str]) -> None:
        if self.run_result is None:
            return
        snapshot_mode = str(self.run_result.get("reference_mode", "target")) == "snapshot"
        self._configure_registry_result_headers()
        self.result_tree.blockSignals(True)
        self.result_tree.clear()
        for solution_id in solution_ids:
            record = self._registry_records_by_id.get(str(solution_id))
            if record is None:
                continue
            evaluation = self._registry_display_evaluation(str(solution_id), record)
            # Keep the canonical solution visible even if an old/partial registry
            # payload has no field observation. Metrics will display as em dashes,
            # while a matching finalist can still materialize/preview normally.
            if evaluation is None:
                evaluation = {"fidelity": "unknown", "metrics": {}}
            metrics = evaluation.get("metrics", {}) or {}
            clearance = (record.get("drc", {}) or {}).get("minimum_clearance_mm")
            fidelity = str(evaluation.get("fidelity", "scout")).replace("_", " ").title()
            if evaluation.get("sample_count"):
                fidelity += f" · {int(evaluation['sample_count']):,}"
            if snapshot_mode:
                row_values = [
                    self._registry_solution_compact_label(solution_id),
                    self._result_metric_text(metrics.get("snapshot_match_coverage_pct"), suffix="%", precision=5),
                    self._result_metric_text(metrics.get("snapshot_vector_rms_uT"), suffix=" µT"),
                    self._result_metric_text(metrics.get("snapshot_magnitude_rms_uT"), suffix=" µT"),
                    self._result_metric_text(metrics.get("snapshot_mean_angle_deg"), suffix="°", precision=5),
                    self._result_metric_text(metrics.get("mean_uT"), suffix=" µT"),
                    "—" if clearance is None else self._result_metric_text(clearance, suffix=" mm", precision=5),
                    fidelity,
                ]
            else:
                row_values = [
                    self._registry_solution_compact_label(solution_id),
                    self._result_metric_text(metrics.get("in_band_pct"), suffix="%", precision=5),
                    self._result_metric_text(metrics.get("rms_target_error_uT"), suffix=" µT"),
                    self._result_metric_text(metrics.get("uniformity_spread_pct"), suffix="%", precision=5),
                    self._result_metric_text(metrics.get("directional_consistency_pct"), suffix="%", precision=5),
                    self._result_metric_text(metrics.get("mean_uT"), suffix=" µT"),
                    "—" if clearance is None else self._result_metric_text(clearance, suffix=" mm", precision=5),
                    fidelity,
                ]
            item = QTreeWidgetItem(row_values)
            item.setData(0, Qt.ItemDataRole.UserRole, str(solution_id))
            item.setData(0, Qt.ItemDataRole.UserRole + 1, "solution")
            lifecycle = record.get("lifecycle", {}) or {}
            retired = lifecycle.get("retired_reason") or record.get("retired_reason")
            tooltip_bits = []
            if retired:
                tooltip_bits.append(f"Retired from further compute: {retired}")
            origins = list(dict.fromkeys(
                str(event.get("origin", "")).replace("_", " ")
                for event in list(record.get("provenance", []) or [])
                if str(event.get("origin", "")).strip()
            ))
            if origins:
                tooltip_bits.append(f"Origins: {', '.join(origins[:6])}")
            if tooltip_bits:
                item.setToolTip(0, "\n".join(tooltip_bits))
            # One placeholder child gives every solution a disclosure arrow without
            # eagerly materializing thousands of retained registry solutions.
            placeholder = QTreeWidgetItem(["Current and geometry…"] + [""] * (self.result_tree.columnCount() - 1))
            placeholder.setData(0, Qt.ItemDataRole.UserRole + 1, "details_placeholder")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.addChild(placeholder)
            self.result_tree.addTopLevelItem(item)
        self.result_tree.blockSignals(False)
        for column in range(self.result_tree.columnCount()):
            self.result_tree.resizeColumnToContents(column)
        self._resize_result_tree_to_content(self.result_tree.topLevelItemCount())
        if self.result_tree.topLevelItemCount():
            self.result_tree.setCurrentItem(self.result_tree.topLevelItem(0))
        else:
            # Never leave a stale candidate/preparing message from the previous
            # result view when the selected semantic view has no displayable row.
            self._result_selection_changed()

    def _result_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item is None:
            return
        marker = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if marker != "solution":
            return
        solution_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(solution_id, str) or solution_id not in self._registry_records_by_id:
            return
        if item.childCount() != 1 or item.child(0).data(0, Qt.ItemDataRole.UserRole + 1) != "details_placeholder":
            return
        self._populate_registry_solution_details(item, solution_id)

    def _add_result_detail_item(
        self,
        parent: QTreeWidgetItem,
        text: str,
        *,
        marker: str = "detail",
    ) -> QTreeWidgetItem:
        child = QTreeWidgetItem([str(text)] + [""] * (self.result_tree.columnCount() - 1))
        child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        child.setData(0, Qt.ItemDataRole.UserRole + 1, marker)
        parent.addChild(child)
        return child

    @staticmethod
    def _format_xyz(values: Any, *, scale: float = 1.0, unit: str = "") -> str:
        try:
            array = np.asarray(values, dtype=float).reshape(3) * float(scale)
        except (TypeError, ValueError):
            return "—"
        suffix = f" {unit}" if unit else ""
        return f"({array[0]:.5g}, {array[1]:.5g}, {array[2]:.5g}){suffix}"

    @staticmethod
    def _optimizer_detail_update_text(name: str, value: Any) -> str:
        labels = {
            "diameter": ("Diameter", 1000.0, "mm"),
            "diameter_m": ("Diameter", 1000.0, "mm"),
            "turns": ("Turns", 1.0, ""),
            "axial_winding_length_m": ("Axial winding length", 1000.0, "mm"),
            "winding_axial_width_mm": ("Axial winding length", 1.0, "mm"),
            "straight_length_m": ("Straight length", 1000.0, "mm"),
            "side_length_m": ("Side length", 1000.0, "mm"),
            "width_m": ("Width", 1000.0, "mm"),
        }
        label, scale, unit = labels.get(str(name), (str(name).replace("_", " ").title(), 1.0, ""))
        try:
            number = float(value) * float(scale)
        except (TypeError, ValueError):
            return f"{label}: {value}"
        suffix = f" {unit}" if unit else ""
        return f"{label}: {number:.6g}{suffix}"

    def _populate_registry_solution_details(self, root: QTreeWidgetItem, solution_id: str) -> None:
        root.takeChildren()
        record = self._registry_records_by_id.get(solution_id, {})
        evaluation = self._registry_preferred_evaluation(record) or {}
        try:
            candidate = self._materialized_candidate_for_solution(solution_id)
        except Exception as error:  # noqa: BLE001 - optional result-detail expansion boundary
            self._add_result_detail_item(root, f"Unable to materialize details: {error}")
            return

        coil_ids = list(map(str, candidate.get("coil_ids", []) or []))
        coil_labels = list(map(str, candidate.get("coil_labels", []) or []))
        label_by_id = {
            object_id: (coil_labels[index] if index < len(coil_labels) else object_id)
            for index, object_id in enumerate(coil_ids)
        }

        currents = dict(evaluation.get("currents_a", {}) or candidate.get("currents_a", {}) or {})
        if currents:
            current_rows = [
                (label_by_id.get(str(object_id), str(object_id)), 1000.0 * float(current))
                for object_id, current in currents.items()
            ]
            rounded = {round(value, 12) for _label, value in current_rows}
            if len(rounded) == 1:
                labels = ", ".join(label for label, _value in current_rows)
                current_text = f"Current: {current_rows[0][1]:.6g} mA ({labels})"
            else:
                current_text = "Current: " + "; ".join(
                    f"{label} {value:.6g} mA" for label, value in current_rows
                )
        else:
            current_text = "Current: —"
        self._add_result_detail_item(root, current_text)

        geometry_root = self._add_result_detail_item(root, "Geometry")
        variables = {
            str(variable.get("id", "")): variable
            for variable in list(self.run_result.get("variables", []) or [])
            if isinstance(variable, dict) and str(variable.get("kind", "")) != "current"
        } if self.run_result is not None else {}
        structural_values = dict(record.get("structural_values", {}) or {})
        if structural_values:
            freedom_root = self._add_result_detail_item(geometry_root, "Optimized freedoms")
            grouped: dict[str, list[tuple[dict[str, Any], Any]]] = {}
            for variable_id, value in structural_values.items():
                variable = variables.get(str(variable_id), {})
                group_label = str(variable.get("group_kind", "Geometry")).replace("_", " ").title()
                if str(variable.get("kind", "")).startswith("assembly_orientation_"):
                    group_label = "Assembly rotation"
                elif str(variable.get("kind", "")).startswith("coil_orientation_"):
                    group_label = "In-place coil rotation"
                elif str(variable.get("kind", "")).startswith("position_") or str(variable.get("kind", "")) == "pair_spacing":
                    group_label = "Position / spacing"
                grouped.setdefault(group_label, []).append((variable, value))
            for group_label, members in grouped.items():
                group_item = self._add_result_detail_item(freedom_root, group_label)
                for variable, value in members:
                    label = str(variable.get("label", variable.get("kind", "Setting")))
                    unit = str(variable.get("unit", "") or "")
                    base = variable.get("base")
                    value_text = self._result_metric_text(value, suffix=(f" {unit}" if unit else ""), precision=6)
                    detail = f"{label}: {value_text}"
                    try:
                        delta = float(value) - float(base)
                    except (TypeError, ValueError):
                        delta = math.nan
                    # For delta-style variables, the value itself already expresses
                    # the change. For absolute spacing/dimensions, showing the source
                    # value and change makes the nested detail much easier to read.
                    kind = str(variable.get("kind", ""))
                    if math.isfinite(delta) and kind in {
                        "pair_spacing", "circular_diameter", "racetrack_end_diameter",
                        "racetrack_straight_length", "square_length", "square_width",
                        "axial_winding_length", "turns",
                    }:
                        base_text = self._result_metric_text(base, suffix=(f" {unit}" if unit else ""), precision=6)
                        delta_text = self._result_metric_text(delta, suffix=(f" {unit}" if unit else ""), precision=6)
                        detail += f" (source {base_text}; Δ {delta_text})"
                    self._add_result_detail_item(group_item, detail)

        transforms = dict(candidate.get("transforms", {}) or {})
        physical_updates = dict(candidate.get("physical_updates", {}) or {})
        parameter_updates = dict(candidate.get("parameter_updates", {}) or {})
        coils_root = self._add_result_detail_item(geometry_root, "Coils")
        source_document = self.run_result.get("source_document") if self.run_result is not None else None
        source_adapter = None
        for object_id in coil_ids:
            coil_root = self._add_result_detail_item(coils_root, label_by_id.get(object_id, object_id))
            transform = transforms.get(object_id)
            if isinstance(transform, dict):
                candidate_position = np.asarray(transform.get("position_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                candidate_euler = np.asarray(transform.get("euler_deg", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                source_transform = self._registry_source_transform_cache.get(object_id)
                if source_transform is None and isinstance(source_document, dict):
                    try:
                        if source_adapter is None:
                            source_adapter = StudioAdapter(copy.deepcopy(source_document))
                        raw = source_adapter.get_transform(object_id)
                        source_transform = {
                            "position_m": list(map(float, raw.get("position", [0.0, 0.0, 0.0]))),
                            "euler_deg": list(map(float, raw.get("euler", [0.0, 0.0, 0.0]))),
                        }
                        self._registry_source_transform_cache[object_id] = source_transform
                    except Exception:  # noqa: BLE001 - optional display-only metadata
                        source_transform = None
                position_root = self._add_result_detail_item(coil_root, "Position")
                self._add_result_detail_item(
                    position_root,
                    f"Location: {self._format_xyz(candidate_position, scale=1000.0, unit='mm')}",
                )
                if isinstance(source_transform, dict):
                    source_position = np.asarray(source_transform.get("position_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                    self._add_result_detail_item(
                        position_root,
                        f"Delta: {self._format_xyz(candidate_position - source_position, scale=1000.0, unit='mm')}",
                    )
                rotation_root = self._add_result_detail_item(coil_root, "Rotation")
                self._add_result_detail_item(
                    rotation_root,
                    f"Euler XYZ: {self._format_xyz(candidate_euler, unit='°')}",
                )
                if isinstance(source_transform, dict):
                    source_euler = source_transform.get("euler_deg", [0.0, 0.0, 0.0])
                    self._add_result_detail_item(
                        rotation_root,
                        f"Source Euler XYZ: {self._format_xyz(source_euler, unit='°')}",
                    )
            updates = physical_updates.get(object_id)
            parameter_update = parameter_updates.get(object_id)
            if isinstance(updates, dict) or isinstance(parameter_update, dict):
                design_root = self._add_result_detail_item(coil_root, "Design changes")
                if isinstance(updates, dict):
                    for name, value in updates.items():
                        if name == "drive_current_a":
                            continue
                        self._add_result_detail_item(
                            design_root, self._optimizer_detail_update_text(str(name), value)
                        )
                if isinstance(parameter_update, dict) and parameter_update.get("name"):
                    self._add_result_detail_item(
                        design_root,
                        self._optimizer_detail_update_text(
                            str(parameter_update.get("name")), parameter_update.get("value")
                        ),
                    )

        # Span nested descriptive rows across the comparison-metric columns so
        # their hierarchy reads like an outline rather than a mostly-empty table.
        def span_children(parent: QTreeWidgetItem) -> None:
            for index in range(parent.childCount()):
                child = parent.child(index)
                self.result_tree.setFirstColumnSpanned(index, parent, True)
                span_children(child)
        span_children(root)

    def _selected_solution_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.result_tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(value, str) and value in self._registry_records_by_id:
                ids.append(value)
        return list(dict.fromkeys(ids))

    def _current_solution_id(self) -> str | None:
        item = self.result_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return str(value) if isinstance(value, str) and value in self._registry_records_by_id else None

    def _materialized_candidate_for_solution(self, solution_id: str) -> dict[str, Any]:
        solution_id = str(solution_id)
        cached = self._materialized_candidate_cache.get(solution_id)
        if cached is not None:
            return copy.deepcopy(cached)
        if self.run_result is None:
            raise StudioOperationError("No completed optimizer result is available.")
        # Finalists still carry their authoritative field arrays; use them directly
        # so their 3D/slice preview remains instant.
        for finalist in list(self.run_result.get("finalists", []) or []):
            if str(finalist.get("registry_candidate_id", "")) == solution_id:
                candidate = copy.deepcopy(finalist)
                candidate["registry_candidate_id"] = solution_id
                self._materialized_candidate_cache[solution_id] = copy.deepcopy(candidate)
                return candidate
        candidate = self.preview_adapter.optimizer_materialize_registry_solution(
            self.run_result, solution_id
        )
        self._materialized_candidate_cache[solution_id] = copy.deepcopy(candidate)
        return candidate

    def _current_candidate(self) -> dict | None:
        solution_id = self._current_solution_id()
        if solution_id is not None:
            try:
                return self._materialized_candidate_for_solution(solution_id)
            except Exception:
                return None
        item = self.result_tree.currentItem()
        value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        return value if isinstance(value, dict) else None

    def _result_selection_changed(self) -> None:
        solution_id = self._current_solution_id()
        selected_count = (
            len(self._selected_solution_ids())
            if self._registry_records_by_id
            else len(self.result_tree.selectedItems())
        )

        # Selection must stay lightweight.  Do not materialize a candidate
        # Workbench document here: that operation used to rebuild the complete 3-D
        # preview every time the user moved through the registry.  Materialization
        # is now deferred until an explicit scene/action button is pressed.
        candidate: dict[str, Any] | None = None
        record: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        if solution_id is not None:
            record = self._registry_records_by_id.get(solution_id, {})
            evaluation = self._registry_display_evaluation(solution_id, record) or {}
        else:
            item = self.result_tree.currentItem()
            value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(value, dict):
                candidate = value
                evaluation = {
                    "fidelity": str(candidate.get("registry_fidelity", "authoritative")),
                    "sample_count": int(
                        candidate.get("sample_count")
                        or (self.run_result or {}).get("final_sample_count", 0)
                        or 0
                    ),
                    "metrics": dict(candidate.get("metrics", {}) or {}),
                    "currents_a": dict(candidate.get("currents_a", {}) or {}),
                    "feasible": bool(candidate.get("feasible", True)),
                }

        has_candidate = solution_id is not None or candidate is not None
        self.copy_button.setEnabled(self.run_result is not None)
        self.open_tab_button.setEnabled(has_candidate and self.on_open_new_tab is not None)
        self.apply_button.setEnabled(has_candidate and self.on_apply is not None)
        self.append_button.setEnabled(has_candidate and self.on_append is not None)
        self.save_snapshot_button.setEnabled(
            solution_id is not None and self.on_save_snapshot is not None
        )
        # The embedded Comparison page covers the current solution.  The button
        # remains useful for a fresh full-field calculation and multi-selection.
        self.compare_button.setEnabled(solution_id is not None and selected_count >= 1)

        if not has_candidate or self.run_result is None:
            self.candidate_detail.setText("")
            self._set_optimizer_comparison_placeholder(
                "No solution selected",
                "Select a solution from the current result view to inspect its comparison metrics.",
            )
            return

        try:
            self._populate_optimizer_comparison_summary(
                solution_id=solution_id,
                evaluation=evaluation,
                record=record,
                candidate=candidate,
            )
            if solution_id is not None:
                drc = record.get("drc", {}) or {}
                origins = list(
                    dict.fromkeys(
                        str(event.get("origin", "")).replace("_", " ")
                        for event in list(record.get("provenance", []) or [])
                        if str(event.get("origin", "")).strip()
                    )
                )
                self.candidate_detail.setText(
                    f"{self._registry_solution_label(solution_id)} • "
                    f"{str(evaluation.get('fidelity', 'scout')).replace('_', ' ').title()}"
                    f" • DRC {str(drc.get('status', 'unknown'))}"
                    + (f" • origins: {', '.join(origins[:4])}" if origins else "")
                )
            else:
                self.candidate_detail.setText(
                    f"{str((candidate or {}).get('id', 'Optimizer solution'))} • retained result"
                )
            self._refresh_optimizer_search_trace()
        except Exception as error:  # noqa: BLE001 - result presentation boundary
            self._set_optimizer_comparison_placeholder(
                "Unable to show optimizer comparison", str(error)
            )

    def _copy_report(self) -> None:
        candidate = self._current_candidate()
        if self.run_result is None:
            return
        plan = str(
            self.run_result.get("settings", {}).get("optimization_plan")
            or self.review_text.toPlainText()
            or ""
        ).strip()
        QApplication.clipboard().setText(
            self.preview_adapter.optimizer_run_report(
                self.run_result,
                selected_candidate=candidate,
                optimization_plan=plan,
                top_n=3,
            )
        )
        self.copy_button.setText("Copied")

    def _analyze_registry_solution_for_snapshot(
        self, solution_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.run_result is None:
            raise StudioOperationError("No completed optimizer result is available.")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return self.preview_adapter.optimizer_registry_solution_analysis(
                self.run_result, solution_id
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _compare_selected(self) -> None:
        if self.run_result is None:
            return
        solution_ids = self._selected_solution_ids()
        if not solution_ids:
            current = self._current_solution_id()
            solution_ids = [current] if current else []
        if not solution_ids:
            return
        source_document = self.run_result.get("source_document")
        if not isinstance(source_document, dict):
            QMessageBox.warning(self, "Compare optimizer solutions", "The frozen optimizer source scene is unavailable.")
            return
        comparison_adapter = StudioAdapter(copy.deepcopy(source_document))
        comparison_adapter.set_coil_modeling_method(self.coil_modeling_method)
        created_ids: list[str] = []
        consistency_warnings: list[str] = []
        try:
            for solution_id in solution_ids:
                _candidate, analysis, candidate_document = self._analyze_registry_solution_for_snapshot(solution_id)
                warning = str(_candidate.get("comparison_consistency_warning", "") or "").strip()
                if warning:
                    consistency_warnings.append(
                        f"{self._registry_solution_label(solution_id)}: {warning}"
                    )
                candidate_scene = StudioAdapter(candidate_document)._snapshot_field_scene()
                snapshot_id = comparison_adapter.create_imported_snapshot(
                    analysis,
                    f"Optimizer {self._registry_solution_label(solution_id)}",
                    field_scene=candidate_scene,
                    origin_label="Optimizer comparison",
                )
                created_ids.append(snapshot_id)
        except Exception as error:  # noqa: BLE001 - on-demand comparison boundary
            QMessageBox.warning(self, "Compare optimizer solutions", str(error))
            return

        if consistency_warnings:
            QMessageBox.warning(
                self,
                "Optimizer comparison consistency",
                "\n\n".join(consistency_warnings),
            )

        snapshot_mode = str(self.run_result.get("reference_mode", "target")) == "snapshot"
        try:
            if snapshot_mode:
                reference = self.run_result.get("reference_snapshot", {}) or {}
                reference_id = str(reference.get("id", ""))
                if not reference_id:
                    raise StudioOperationError("The optimizer reference snapshot is unavailable.")
                dialog = SnapshotComparisonDialog(
                    comparison_adapter,
                    [reference_id, *created_ids],
                    parent=self,
                    initial_mode="match",
                    initial_reference_id=reference_id,
                    initial_match_magnitude_tolerance_pct=float(reference.get("magnitude_tolerance_pct", 5.0)),
                    initial_match_direction_tolerance_deg=float(reference.get("direction_tolerance_deg", 3.0)),
                    # Exposure-mode optimizer analyses are already materialized in
                    # their solved rigid registration.  Do not apply Snapshot
                    # Comparison's separate vector-only alignment a second time.
                    initial_match_alignment_mode=(
                        "fixed"
                        if str(reference.get("alignment_mode", "fixed") or "fixed") == "exposure"
                        else str(reference.get("alignment_mode", "fixed") or "fixed")
                    ),
                )
            else:
                target = self.run_result.get("target", {}) or {}
                dialog = SnapshotComparisonDialog(
                    comparison_adapter,
                    created_ids,
                    parent=self,
                    initial_mode="benchmark",
                    initial_benchmark_target_uT=float(target.get("target_uT", 200.0)),
                    initial_benchmark_tolerance_pct=float(target.get("tolerance_pct", 5.0)),
                )
            dialog.exec()
        except Exception as error:  # noqa: BLE001 - comparison UI boundary
            QMessageBox.warning(self, "Compare optimizer solutions", str(error))

    def _save_selected_snapshot(self) -> None:
        solution_id = self._current_solution_id()
        if solution_id is None or self.on_save_snapshot is None:
            return
        default_name = f"Optimizer {self._registry_solution_label(solution_id)}"
        name, accepted = QInputDialog.getText(
            self, "Save optimizer snapshot", "Snapshot name:", text=default_name
        )
        if not accepted:
            return
        try:
            _candidate, analysis, candidate_document = self._analyze_registry_solution_for_snapshot(solution_id)
            candidate_scene = StudioAdapter(candidate_document)._snapshot_field_scene()
            self.on_save_snapshot(analysis, str(name), candidate_scene)
        except Exception as error:  # noqa: BLE001 - snapshot integration boundary
            QMessageBox.warning(self, "Save optimizer snapshot", str(error))

    def _open_selected_new_tab(self) -> None:
        candidate = self._current_candidate()
        if candidate is None or self.on_open_new_tab is None or self.run_result is None:
            return
        source_document = self.run_result.get("source_document")
        if not isinstance(source_document, dict):
            QMessageBox.warning(
                self,
                "Open optimizer solution",
                "The optimizer source scene is unavailable, so this solution cannot be opened as a new tab.",
            )
            return
        self.on_open_new_tab(candidate, source_document)

    def _append_selected(self) -> None:
        candidate = self._current_candidate()
        if candidate is None or self.on_append is None:
            return
        answer = QMessageBox.question(
            self,
            "Append to source scene",
            "Append copies of only the participating coils whose geometry or design changed in this optimizer solution? "
            "Unchanged/current-only source coils are skipped and the existing source coils are preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.on_append(candidate)

    def _apply_selected(self) -> None:
        candidate = self._current_candidate()
        if candidate is None or self.on_apply is None:
            return
        answer = QMessageBox.question(
            self,
            "Apply to source scene",
            "Apply this optimizer solution back to the source scene's participating coils? Other scene objects are preserved and "
            "the operation is undoable.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.on_apply(candidate, True)

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        # The worker object may survive for a short queued-event interval after
        # ``completed`` has already populated Results. At that point the native X
        # must behave like an ordinary Close button, not ask about a search that
        # visibly finished. Let the thread wind down, then close automatically.
        search_visibly_complete = self.run_result is not None or str(
            self._optimizer_last_progress.get("phase_label", "")
        ) in {"Optimization failed", "Cancelled"}
        if self.worker is not None and not search_visibly_complete:
            answer = QMessageBox.question(
                self,
                "Optimizer running",
                "Cancel the running search and close this optimizer window? The source scene is unchanged.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_after_thread = True
            self._cancel_search()
            event.ignore()
            return
        if self.thread is not None and self.thread.isRunning():
            self._close_after_thread = True
            self.thread.quit()
            event.ignore()
            return
        # Comparison/statistics pages are ordinary Qt widgets; only Search
        # History owns a PlotView/WebEngine surface that needs explicit cleanup.
        self.sweep_plot.cleanup()
        event.accept()

class _PlaybackCoilSelectionMixin:
    """Shared map-window selector for coils active in static maps and waveform playback."""

    def _init_playback_coil_selector(self) -> None:
        self._playback_selected_coil_ids: set[str] = set()
        self._playback_known_coil_ids: set[str] = set()
        self._playback_selection_initialized = False
        self.playback_coils_menu = QMenu(self)
        self.playback_coils_menu.aboutToShow.connect(self._refresh_playback_coils_menu)
        self.playback_coils_button = QPushButton("Coils…")
        self.playback_coils_button.setMenu(self.playback_coils_menu)
        self.playback_coils_button.setToolTip(
            "Choose which scene coils contribute to this map window and its waveform playback. "
            "This selection is independent of the source scene's enabled state."
        )
        self._refresh_playback_coils_menu()

    def _refresh_playback_coils_menu(self) -> None:
        choices = drive_coil_catalog(self.adapter)
        current_ids = {str(choice["id"]) for choice in choices}
        if not self._playback_selection_initialized:
            self._playback_selected_coil_ids = {
                str(choice["id"])
                for choice in choices
                if bool(choice.get("enabled", True))
            }
            self._playback_selection_initialized = True
        else:
            # The map window owns this selection after it opens. Scene enable/disable
            # changes (and newly added coils) must not silently rewrite the user's
            # local Active coils choices.
            self._playback_selected_coil_ids.intersection_update(current_ids)
        self._playback_known_coil_ids = current_ids

        self.playback_coils_menu.clear()
        scene_defaults = self.playback_coils_menu.addAction("Use scene enabled coils")
        scene_defaults.triggered.connect(self._reset_playback_coils_to_scene)
        select_all = self.playback_coils_menu.addAction("Select all")
        select_all.triggered.connect(self._select_all_playback_coils)
        clear_all = self.playback_coils_menu.addAction("Clear all")
        clear_all.triggered.connect(self._clear_all_playback_coils)
        self.playback_coils_menu.addSeparator()
        if not choices:
            empty = self.playback_coils_menu.addAction("No coils in scene")
            empty.setEnabled(False)
        else:
            for choice in choices:
                object_id = str(choice["id"])
                label = str(choice.get("label") or object_id)
                if not bool(choice.get("enabled", True)):
                    label += " (scene disabled)"
                action = self.playback_coils_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(object_id in self._playback_selected_coil_ids)
                action.setData(object_id)
                action.toggled.connect(
                    lambda checked, selected_id=object_id: self._set_playback_coil_selected(
                        selected_id, checked
                    )
                )
        self._update_playback_coils_button_text(len(current_ids))

    def _update_playback_coils_button_text(self, total: int | None = None) -> None:
        if total is None:
            total = len(self._playback_known_coil_ids)
        selected = len(self._playback_selected_coil_ids & self._playback_known_coil_ids)
        if total:
            self.playback_coils_button.setText(f"Coils {selected}/{total}…")
        else:
            self.playback_coils_button.setText("Coils 0…")

    def _set_playback_coil_selected(self, object_id: str, checked: bool) -> None:
        if checked:
            self._playback_selected_coil_ids.add(str(object_id))
        else:
            self._playback_selected_coil_ids.discard(str(object_id))
        self._update_playback_coils_button_text()

    def _select_all_playback_coils(self, *_args) -> None:
        self._refresh_playback_coils_menu()
        self._playback_selected_coil_ids = set(self._playback_known_coil_ids)
        self._refresh_playback_coils_menu()

    def _clear_all_playback_coils(self, *_args) -> None:
        self._playback_selected_coil_ids.clear()
        self._refresh_playback_coils_menu()

    def _reset_playback_coils_to_scene(self, *_args) -> None:
        choices = drive_coil_catalog(self.adapter)
        self._playback_known_coil_ids = {str(choice["id"]) for choice in choices}
        self._playback_selected_coil_ids = {
            str(choice["id"]) for choice in choices if bool(choice.get("enabled", True))
        }
        self._refresh_playback_coils_menu()

    def selected_active_coil_ids(self) -> list[str]:
        self._refresh_playback_coils_menu()
        return [
            str(choice["id"])
            for choice in drive_coil_catalog(self.adapter)
            if str(choice["id"]) in self._playback_selected_coil_ids
        ]

    def selected_playback_coil_ids(self) -> list[str]:
        """Compatibility alias retained for waveform callers/tests from 0.13.0.121."""
        return self.selected_active_coil_ids()


class _CameraTimelineMixin:
    """Shared camera/playback controls for 2D/3D animated maps and video export."""

    _camera_view_mode = "3d"

    def _init_camera_timeline_controls(self, *, view_mode: str) -> None:
        self._camera_view_mode = str(view_mode).lower()
        self._camera_timeline: dict[str, Any] | None = None
        self._fixed_camera_user_edited = False
        self._fixed_camera_controls_updating = False
        self._timeline_edit_index: int | None = None
        self._timeline_edit_is_new = False
        self._timeline_edit_template: dict[str, Any] | None = None
        self._timeline_edit_original_timeline: dict[str, Any] | None = None

        self.camera_section = CollapsibleSection("Camera", expanded=False)
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(5)

        self.camera_mode_group = QButtonGroup(self)
        self.camera_mode_group.setExclusive(True)
        self.camera_interactive = QRadioButton("Interactive (fixed export)")
        self.camera_timeline_mode = QRadioButton("Scripted")
        self.camera_mode_group.addButton(self.camera_interactive)
        self.camera_mode_group.addButton(self.camera_timeline_mode)
        self.camera_interactive.setChecked(True)
        camera_layout.addWidget(self.camera_interactive)
        camera_layout.addWidget(self.camera_timeline_mode)

        # The same compact camera/view editors are reused for fixed output and
        # for authoring one timeline step at a time.
        self.fixed_camera_widget = QWidget()
        fixed_camera_layout = QGridLayout(self.fixed_camera_widget)
        fixed_camera_layout.setContentsMargins(8, 2, 0, 4)
        fixed_camera_layout.setHorizontalSpacing(6)
        fixed_camera_layout.setVerticalSpacing(3)

        if self._camera_view_mode == "3d":
            default_camera = self._camera_default()
            position = default_camera.get("position", [0.0, 0.0, 1.0])
            target = default_camera.get("target", [0.0, 0.0, 0.0])

            fixed_camera_layout.addWidget(QLabel("Camera position"), 0, 0, 1, 3)
            self.camera_position_x = _spin(float(position[0]), -1e9, 1e9, 3, " mm")
            self.camera_position_y = _spin(float(position[1]), -1e9, 1e9, 3, " mm")
            self.camera_position_z = _spin(float(position[2]), -1e9, 1e9, 3, " mm")
            self.camera_position = [
                self.camera_position_x,
                self.camera_position_y,
                self.camera_position_z,
            ]
            for column, (axis, editor) in enumerate(zip("XYZ", self.camera_position)):
                fixed_camera_layout.addWidget(QLabel(axis), 1, column)
                fixed_camera_layout.addWidget(editor, 2, column)

            fixed_camera_layout.addWidget(QLabel("Look-at target (orientation)"), 3, 0, 1, 3)
            self.camera_target_x = _spin(float(target[0]), -1e9, 1e9, 3, " mm")
            self.camera_target_y = _spin(float(target[1]), -1e9, 1e9, 3, " mm")
            self.camera_target_z = _spin(float(target[2]), -1e9, 1e9, 3, " mm")
            self.camera_target = [
                self.camera_target_x,
                self.camera_target_y,
                self.camera_target_z,
            ]
            for column, (axis, editor) in enumerate(zip("XYZ", self.camera_target)):
                fixed_camera_layout.addWidget(QLabel(axis), 4, column)
                fixed_camera_layout.addWidget(editor, 5, column)

            self.preview_fixed_camera_button = QPushButton("Set viewer")
            self.preview_fixed_camera_button.setToolTip(
                "Set the Selector viewer from these camera values"
            )
            self.preview_fixed_camera_button.clicked.connect(self._preview_fixed_camera)
            fixed_camera_layout.addWidget(self.preview_fixed_camera_button, 6, 0, 1, 3)
            for editor in (*self.camera_position, *self.camera_target):
                editor.setMinimumWidth(0)
                editor.setSizePolicy(
                    QSizePolicy.Policy.Expanding, editor.sizePolicy().verticalPolicy()
                )
                editor.valueChanged.connect(self._fixed_camera_control_changed)
        else:
            default_camera = self._camera_default()
            target = default_camera.get("target", [0.0, 0.0, 0.0])
            self.camera_view_x = _spin(float(target[0]), -1e9, 1e9, 3, " mm")
            self.camera_view_y = _spin(float(target[1]), -1e9, 1e9, 3, " mm")
            self.camera_zoom_2d = _spin(
                float(default_camera.get("zoom_2d", 1.0)), 0.01, 100.0, 4, "×"
            )
            fixed_camera_layout.addWidget(QLabel("View centre X"), 0, 0)
            fixed_camera_layout.addWidget(self.camera_view_x, 0, 1)
            fixed_camera_layout.addWidget(QLabel("View centre Y"), 1, 0)
            fixed_camera_layout.addWidget(self.camera_view_y, 1, 1)
            fixed_camera_layout.addWidget(QLabel("Zoom"), 2, 0)
            fixed_camera_layout.addWidget(self.camera_zoom_2d, 2, 1)
            for editor in (self.camera_view_x, self.camera_view_y, self.camera_zoom_2d):
                editor.valueChanged.connect(self._fixed_camera_control_changed)
            self.preview_fixed_camera_button = QPushButton("Set viewer")
            self.preview_fixed_camera_button.setToolTip(
                "Set the active 2D map viewer from these camera values"
            )
            self.preview_fixed_camera_button.clicked.connect(self._preview_fixed_camera)
            fixed_camera_layout.addWidget(self.preview_fixed_camera_button, 3, 0, 1, 2)
        # Interactive mode takes its fixed-export camera directly from the live
        # viewer. Numeric camera editors are an authoring aid for Scripted mode.
        self.fixed_camera_widget.setVisible(False)
        camera_layout.addWidget(self.fixed_camera_widget)

        # Playback controls belong to the map/camera setup because both animated
        # playback and exported video consume them. Sampling density is chosen
        # automatically by WaveformPlaybackDialog.
        playback_grid = QGridLayout()
        playback_grid.setContentsMargins(8, 2, 0, 3)
        self.camera_target_fps_label = QLabel("Target FPS")
        self.camera_target_fps = _spin(60.0, 1.0, 240.0, 1, " fps")
        self.camera_target_fps.setSingleStep(5.0)
        self.camera_target_fps.setToolTip("Target animation/video frame rate")
        self.camera_playback_speed_label = QLabel("Playback speed")
        self.camera_playback_speed = _spin(
            1.0,
            0.000001,
            1_000_000.0,
            9,
            "×",
            spinbox_type=PlaybackSpeedSpinBox,
        )
        self.camera_playback_speed.setSingleStep(0.25)
        self.camera_playback_speed.setToolTip(
            "Playback speed relative to real time. Arrows use 1, 0.75, 0.5, 0.25, "
            "then successive halves down to 0.000001×; typed values remain valid. "
            "Timeline mode stores this per step instead."
        )
        playback_grid.addWidget(self.camera_target_fps_label, 0, 0)
        playback_grid.addWidget(self.camera_target_fps, 0, 1)
        self.camera_script_loop = QCheckBox("Loop waveform/sequencer to fill timeline")
        self.camera_script_loop.setChecked(False)
        self.camera_script_loop.setToolTip(
            "Repeat the waveform and sequencer until the scripted camera timeline ends"
        )
        playback_grid.addWidget(self.camera_script_loop, 1, 0, 1, 2)
        playback_grid.addWidget(self.camera_playback_speed_label, 2, 0)
        playback_grid.addWidget(self.camera_playback_speed, 2, 1)
        camera_layout.addLayout(playback_grid)

        # Inline timeline editor. There is deliberately no separate timeline
        # window: camera steps are authored against the live Selector preview.
        self.camera_timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(self.camera_timeline_widget)
        timeline_layout.setContentsMargins(8, 2, 0, 4)
        timeline_layout.setSpacing(4)
        self.camera_timeline_summary = QLabel("—")
        self.camera_timeline_summary.setObjectName("hint")
        timeline_layout.addWidget(self.camera_timeline_summary)

        self.camera_timeline_table = QTableWidget(0, 4)
        self.camera_timeline_table.setHorizontalHeaderLabels(["Step", "Time", "Speed", "Camera / view"])
        self.camera_timeline_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.camera_timeline_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.camera_timeline_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.camera_timeline_table.setAlternatingRowColors(True)
        self.camera_timeline_table.verticalHeader().setVisible(False)
        self.camera_timeline_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.camera_timeline_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.camera_timeline_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.camera_timeline_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.camera_timeline_table.setMinimumHeight(112)
        self.camera_timeline_table.setMaximumHeight(170)
        self.camera_timeline_table.itemSelectionChanged.connect(self._camera_timeline_selection_changed)
        self.camera_timeline_table.cellClicked.connect(self._camera_timeline_row_clicked)
        timeline_layout.addWidget(self.camera_timeline_table)

        timeline_buttons = QHBoxLayout()
        self.camera_timeline_add_button = QPushButton("Add")
        self.camera_timeline_edit_button = QPushButton("Edit")
        self.camera_timeline_remove_button = QPushButton("Remove")
        self.camera_timeline_add_button.clicked.connect(self._camera_timeline_add)
        self.camera_timeline_edit_button.clicked.connect(self._camera_timeline_edit)
        self.camera_timeline_remove_button.clicked.connect(self._camera_timeline_remove)
        timeline_buttons.addWidget(self.camera_timeline_add_button)
        timeline_buttons.addWidget(self.camera_timeline_edit_button)
        timeline_buttons.addWidget(self.camera_timeline_remove_button)
        timeline_buttons.addStretch(1)
        timeline_layout.addLayout(timeline_buttons)

        self.camera_timeline_edit_widget = QWidget()
        timeline_edit_layout = QGridLayout(self.camera_timeline_edit_widget)
        timeline_edit_layout.setContentsMargins(0, 2, 0, 0)
        self.camera_timeline_time_label = QLabel("Step time")
        self.camera_timeline_time = _spin(0.0, 0.0, 1.0e9, 6, " s")
        self.camera_timeline_time.setProperty("persistUiState", False)
        self.camera_timeline_speed_label = QLabel("Step playback speed")
        self.camera_timeline_speed = _spin(
            1.0,
            0.000001,
            100.0,
            9,
            "×",
            spinbox_type=PlaybackSpeedSpinBox,
        )
        self.camera_timeline_speed.setSingleStep(0.25)
        self.camera_timeline_speed.setToolTip(
            "Arrows use 1, 0.75, 0.5, 0.25, then successive halves down to "
            "0.000001×; typed values remain valid."
        )
        self.camera_timeline_speed.setProperty("persistUiState", False)
        self.camera_timeline_accept_button = QPushButton("Accept")
        self.camera_timeline_cancel_button = QPushButton("Cancel")
        self.camera_timeline_accept_button.setObjectName("primaryButton")
        self.camera_timeline_accept_button.clicked.connect(self._camera_timeline_accept)
        self.camera_timeline_cancel_button.clicked.connect(self._camera_timeline_cancel)
        timeline_edit_layout.addWidget(self.camera_timeline_time_label, 0, 0)
        timeline_edit_layout.addWidget(self.camera_timeline_time, 0, 1)
        timeline_edit_layout.addWidget(self.camera_timeline_speed_label, 1, 0)
        timeline_edit_layout.addWidget(self.camera_timeline_speed, 1, 1)
        timeline_edit_layout.addWidget(self.camera_timeline_accept_button, 2, 0)
        timeline_edit_layout.addWidget(self.camera_timeline_cancel_button, 2, 1)
        self.camera_timeline_edit_widget.setVisible(False)
        timeline_layout.addWidget(self.camera_timeline_edit_widget)

        self.camera_timeline_widget.setVisible(False)
        self.camera_script_loop.setVisible(False)
        camera_layout.addWidget(self.camera_timeline_widget)
        self.camera_section.setContentLayout(camera_layout)

        self.camera_timeline_mode.toggled.connect(self._camera_mode_changed)
        self.camera_script_loop.toggled.connect(self._camera_script_loop_changed)
        if self._camera_view_mode == "3d":
            self.camera_section.header.toggled.connect(
                lambda expanded: self._capture_selector_camera(mark_user=False)
                if expanded and not self._fixed_camera_user_edited
                else None
            )

    def _camera_mode_changed(self, timeline_enabled: bool) -> None:
        enabled = bool(timeline_enabled)
        self.fixed_camera_widget.setVisible(enabled)
        self.camera_playback_speed_label.setVisible(not enabled)
        self.camera_playback_speed.setVisible(not enabled)
        self.camera_script_loop.setVisible(enabled)
        self.camera_timeline_widget.setVisible(enabled)
        if enabled:
            timeline_was_new = not isinstance(self._camera_timeline, dict)
            self._ensure_camera_timeline()
            self._refresh_camera_timeline_table()
            if self._camera_view_mode == "2d":
                def sync_2d_camera() -> None:
                    def received(camera: dict[str, Any]) -> None:
                        if timeline_was_new and isinstance(self._camera_timeline, dict):
                            points = self._camera_timeline.get("points")
                            if isinstance(points, list) and points and isinstance(points[0], dict):
                                points[0]["camera"] = copy.deepcopy(camera)
                                self._refresh_camera_timeline_table(select_row=0)
                    self._capture_2d_view_camera(mark_user=False, callback=received)
                QTimer.singleShot(0, sync_2d_camera)
        else:
            self._camera_timeline_cancel()

    def _camera_script_loop_changed(self, checked: bool) -> None:
        if isinstance(self._camera_timeline, dict):
            self._camera_timeline["loop_waveform"] = bool(checked)
        if getattr(self, "camera_timeline_mode", None) is not None and self.camera_timeline_mode.isChecked():
            self._refresh_camera_timeline_table()

    def _fixed_camera_control_changed(self, *_args) -> None:
        if not self._fixed_camera_controls_updating:
            self._fixed_camera_user_edited = True

    def _fixed_camera_from_controls(self) -> dict[str, Any]:
        if self._camera_view_mode == "3d" and hasattr(self, "camera_position"):
            camera = {
                "position": [editor.value() for editor in self.camera_position],
                "target": [editor.value() for editor in self.camera_target],
                "fov_deg": DEFAULT_CAMERA_FOV_DEG,
            }
            try:
                return validate_video_camera(camera)
            except ValueError:
                return self._camera_default()

        default = self._camera_default()
        if self._camera_view_mode == "2d" and hasattr(self, "camera_view_x"):
            target = list(default.get("target", [0.0, 0.0, 0.0]))
            position = list(default.get("position", [0.0, 0.0, 1.0]))
            first_axis, second_axis, _normal_axis = self._camera_2d_plane_axes()
            next_first = self.camera_view_x.value()
            next_second = self.camera_view_y.value()
            delta_first = next_first - float(target[first_axis])
            delta_second = next_second - float(target[second_axis])
            target[first_axis] = next_first
            target[second_axis] = next_second
            position[first_axis] += delta_first
            position[second_axis] += delta_second
            camera = {
                "position": position,
                "target": target,
                "fov_deg": DEFAULT_CAMERA_FOV_DEG,
                "zoom_2d": self.camera_zoom_2d.value(),
            }
            try:
                return validate_video_camera(camera) | {"zoom_2d": self.camera_zoom_2d.value()}
            except ValueError:
                return default
        return default

    def _set_fixed_camera_controls(
        self, camera: dict[str, Any], *, mark_user: bool = False
    ) -> None:
        try:
            normalized = validate_video_camera(camera)
        except ValueError:
            normalized = self._camera_default()
        self._fixed_camera_controls_updating = True
        try:
            if self._camera_view_mode == "3d" and hasattr(self, "camera_position"):
                for editor, value in zip(self.camera_position, normalized["position"]):
                    editor.setValue(float(value))
                for editor, value in zip(self.camera_target, normalized["target"]):
                    editor.setValue(float(value))
            elif self._camera_view_mode == "2d" and hasattr(self, "camera_view_x"):
                target = normalized.get("target", [0.0, 0.0, 0.0])
                first_axis, second_axis, _normal_axis = self._camera_2d_plane_axes()
                self.camera_view_x.setValue(float(target[first_axis]))
                self.camera_view_y.setValue(float(target[second_axis]))
                try:
                    zoom = float(camera.get("zoom_2d", 1.0))
                except (TypeError, ValueError):
                    zoom = 1.0
                self.camera_zoom_2d.setValue(max(0.01, min(100.0, zoom)))
        finally:
            self._fixed_camera_controls_updating = False
        self._fixed_camera_user_edited = bool(mark_user)

    def _camera_2d_plane_axes(self) -> tuple[int, int, int]:
        """Return world axes used by the 2D map's horizontal, vertical, normal coordinates."""
        plane_widget = getattr(self, "plane", None)
        plane = str(plane_widget.currentText()).strip().lower() if plane_widget is not None else "xy"
        return {
            "xy": (0, 1, 2),
            "xz": (0, 2, 1),
            "yz": (1, 2, 0),
        }.get(plane, (0, 1, 2))

    def _camera_from_2d_plot_view(self, view: dict[str, Any]) -> dict[str, Any]:
        """Translate live Plotly x/y ranges into the 2D camera centre + zoom model."""
        fallback = self._fixed_camera_from_controls()
        if not isinstance(view, dict):
            return fallback
        x_range = view.get("xaxis.range")
        y_range = view.get("yaxis.range")
        if not (
            isinstance(x_range, list)
            and len(x_range) >= 2
            and isinstance(y_range, list)
            and len(y_range) >= 2
        ):
            return fallback
        try:
            x0, x1 = float(x_range[0]), float(x_range[1])
            y0, y1 = float(y_range[0]), float(y_range[1])
        except (TypeError, ValueError):
            return fallback
        if not all(math.isfinite(value) for value in (x0, x1, y0, y1)):
            return fallback

        first_axis, second_axis, _normal_axis = self._camera_2d_plane_axes()
        target = list(fallback.get("target", [0.0, 0.0, 0.0]))
        position = list(fallback.get("position", [0.0, 0.0, 1.0]))
        next_first = (x0 + x1) * 0.5
        next_second = (y0 + y1) * 0.5
        delta_first = next_first - float(target[first_axis])
        delta_second = next_second - float(target[second_axis])
        target[first_axis] = next_first
        target[second_axis] = next_second
        position[first_axis] += delta_first
        position[second_axis] += delta_second

        span = max(1.0e-9, abs(float(getattr(self, "span").value())))
        widths = [abs(x1 - x0), abs(y1 - y0)]
        finite_widths = [value for value in widths if math.isfinite(value) and value > 1.0e-12]
        zoom = min(finite_widths) / span if finite_widths else float(fallback.get("zoom_2d", 1.0))
        zoom = max(0.01, min(100.0, zoom))
        try:
            normalized = validate_video_camera(
                {
                    "position": position,
                    "target": target,
                    "fov_deg": DEFAULT_CAMERA_FOV_DEG,
                }
            )
        except ValueError:
            return fallback
        normalized["zoom_2d"] = zoom
        return normalized

    def _plotly_view_from_2d_camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        """Translate a 2D camera into Plotly ranges for Set viewer."""
        first_axis, second_axis, _normal_axis = self._camera_2d_plane_axes()
        target = camera.get("target", [0.0, 0.0, 0.0])
        try:
            centre_first = float(target[first_axis])
            centre_second = float(target[second_axis])
            zoom = float(camera.get("zoom_2d", 1.0))
        except (TypeError, ValueError, IndexError):
            centre_first = 0.0
            centre_second = 0.0
            zoom = 1.0
        span = max(1.0e-9, abs(float(getattr(self, "span").value())))
        half = span * max(0.01, min(100.0, zoom)) * 0.5
        return {
            "kind": "2d",
            "updates": {
                "xaxis.range": [centre_first - half, centre_first + half],
                "xaxis.autorange": False,
                "yaxis.range": [centre_second - half, centre_second + half],
                "yaxis.autorange": False,
            },
        }

    def _is_2d_camera_viewer(self, widget: Any) -> bool:
        if isinstance(widget, PlotView):
            return widget is not getattr(self, "plot", None)
        if isinstance(widget, WaveformGLView):
            payload = getattr(widget, "_payload", {})
            return str(payload.get("view_mode", "")).lower() == "2d"
        return False

    def _current_2d_camera_viewer(self) -> Any | None:
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return None
        current = tabs.currentWidget()
        if self._is_2d_camera_viewer(current):
            return current
        for index in range(tabs.count() - 1, -1, -1):
            candidate = tabs.widget(index)
            if self._is_2d_camera_viewer(candidate):
                return candidate
        return None

    def _connect_result_camera_viewer(self, viewer: Any) -> None:
        """Connect generated 2D result viewers to Scripted camera authoring."""
        if self._camera_view_mode != "2d" or not self._is_2d_camera_viewer(viewer):
            return
        if bool(getattr(viewer, "_field_workbench_camera_sync_connected", False)):
            return
        setattr(viewer, "_field_workbench_camera_sync_connected", True)
        if isinstance(viewer, PlotView):
            viewer.live_2d_view_changed.connect(
                lambda view, viewer=viewer: self._2d_plot_view_changed(viewer, view)
            )
        elif isinstance(viewer, WaveformGLView):
            viewer.cameraChanged.connect(
                lambda camera, viewer=viewer: self._2d_webgl_camera_changed(viewer, camera)
            )

    def _2d_plot_view_changed(self, viewer: PlotView, view: dict[str, Any]) -> None:
        if (
            self._camera_view_mode != "2d"
            or not self.camera_timeline_mode.isChecked()
            or getattr(self, "tabs", None) is None
            or self.tabs.currentWidget() is not viewer
        ):
            return
        camera = self._camera_from_2d_plot_view(view)
        self._set_fixed_camera_controls(camera, mark_user=True)

    def _2d_webgl_camera_changed(self, viewer: WaveformGLView, camera: dict[str, Any]) -> None:
        if (
            self._camera_view_mode != "2d"
            or not self.camera_timeline_mode.isChecked()
            or getattr(self, "tabs", None) is None
            or self.tabs.currentWidget() is not viewer
            or not isinstance(camera, dict)
        ):
            return
        self._set_fixed_camera_controls(camera, mark_user=True)

    def _capture_2d_view_camera(
        self,
        *,
        mark_user: bool,
        callback=None,
        viewer: Any | None = None,
    ) -> None:
        """Capture the active/generated 2D map camera, falling back to authored controls."""
        fallback = self._fixed_camera_from_controls()
        active = viewer if self._is_2d_camera_viewer(viewer) else self._current_2d_camera_viewer()
        if active is None:
            if mark_user:
                self._camera_status("2D camera preview unavailable — render a map first")
            if callable(callback):
                callback(fallback)
            return

        def received(value: Any) -> None:
            camera = fallback
            if isinstance(active, PlotView) and isinstance(value, dict):
                camera = self._camera_from_2d_plot_view(value)
            elif isinstance(active, WaveformGLView) and isinstance(value, dict):
                try:
                    normalized = validate_video_camera(value)
                    normalized["zoom_2d"] = max(
                        0.01, min(100.0, float(value.get("zoom_2d", 1.0)))
                    )
                    camera = normalized
                except (TypeError, ValueError):
                    camera = fallback
            self._set_fixed_camera_controls(camera, mark_user=mark_user)
            if mark_user:
                self._camera_status("Camera controls updated from 2D viewer")
            if callable(callback):
                callback(camera)

        if isinstance(active, PlotView):
            active.request_2d_view(received)
        elif isinstance(active, WaveformGLView):
            active.request_camera(received)
        else:
            received(None)

    def _show_camera_selector(self) -> None:
        tabs = getattr(self, "tabs", None)
        selector_index = getattr(self, "_selector_tab_index", None)
        if tabs is not None and isinstance(selector_index, int):
            tabs.setCurrentIndex(selector_index)

    def _camera_status(self, message: str) -> None:
        status = getattr(self, "calculation_status", None)
        if status is not None and hasattr(status, "showMessage"):
            status.showMessage(message)

    def _preview_fixed_camera(self, *_args) -> None:
        """Set the live 2D/3D authoring viewer from the numeric camera controls."""
        camera = self._fixed_camera_from_controls()
        if self._camera_view_mode == "2d":
            viewer = self._current_2d_camera_viewer()
            if viewer is None:
                self._camera_status("Camera Set unavailable — render a 2D map first")
                return
            tabs = getattr(self, "tabs", None)
            if tabs is not None:
                index = tabs.indexOf(viewer)
                if index >= 0:
                    tabs.setCurrentIndex(index)
            if isinstance(viewer, WaveformGLView):
                viewer.set_camera(camera)
                self._fixed_camera_user_edited = True
                self._camera_status("2D viewer set from controls")
                return
            if isinstance(viewer, PlotView):
                viewer.apply_view_state(self._plotly_view_from_2d_camera(camera))
                self._fixed_camera_user_edited = True
                self._camera_status("2D viewer set from controls")
                return
            self._camera_status("Camera Set unavailable — 2D viewer is not ready")
            return

        if self._camera_view_mode != "3d":
            return
        self._show_camera_selector()
        plot = getattr(self, "plot", None)
        request = getattr(plot, "request_3d_view", None)
        apply_view = getattr(plot, "apply_view_state", None)
        if not callable(request) or not callable(apply_view):
            self._camera_status("Camera Set unavailable — Selector viewer is not ready")
            return

        def received(view: Any) -> None:
            if not isinstance(view, dict):
                self._camera_status("Camera Set unavailable — could not read Selector view")
                return
            try:
                state = video_camera_to_plotly_view(camera, view)
            except ValueError:
                self._camera_status("Camera Set unavailable — invalid camera values")
                return
            if not isinstance(state, dict):
                self._camera_status("Camera Set unavailable — Selector view is not a 3D scene")
                return
            apply_view(state)
            self._fixed_camera_user_edited = True
            web = getattr(plot, "web", None)
            if web is not None:
                QTimer.singleShot(0, lambda: web.update())
            self._camera_status("Camera set from controls")

        request(received)

    def _selector_camera_changed(self, view: dict[str, Any]) -> None:
        """Keep the 3D camera editors synchronized with the live Selector view."""
        if self._camera_view_mode != "3d" or not isinstance(view, dict):
            return
        camera = self._camera_from_selector_view(view)
        self._set_fixed_camera_controls(camera, mark_user=True)

    def _capture_selector_camera(self, *, mark_user: bool) -> None:
        if self._camera_view_mode != "3d":
            return
        if mark_user:
            self._show_camera_selector()
        request = getattr(getattr(self, "plot", None), "request_3d_view", None)
        if not callable(request):
            if mark_user:
                self._camera_status("Camera Get unavailable — Selector viewer is not ready")
            return

        def received(view: Any) -> None:
            if not isinstance(view, dict):
                if mark_user:
                    self._camera_status("Camera Get unavailable — could not read Selector view")
                return
            camera = self._camera_from_selector_view(view)
            self._set_fixed_camera_controls(camera, mark_user=mark_user)
            if mark_user:
                self._camera_status("Camera controls updated from Selector")

        request(received)

    def _camera_default(self) -> dict[str, Any]:
        if self._camera_view_mode == "2d":
            plane = str(getattr(self, "plane").currentText()).lower()
            offset = float(getattr(self, "offset").value())
            centre = [0.0, 0.0, 0.0]
            centre[{"xy": 2, "xz": 1, "yz": 0}.get(plane, 2)] = offset
            return default_2d_camera(
                centre_mm=centre,
                span_mm=float(getattr(self, "span").value()),
            )
        payload = {
            "centre_mm": [editor.value() for editor in getattr(self, "coordinates")],
            "span_mm": float(getattr(self, "span").value()),
        }
        return default_video_camera(payload)

    def _camera_from_selector_view(self, view: Any) -> dict[str, Any]:
        fallback = self._fixed_camera_from_controls()
        if self._camera_view_mode != "3d":
            return fallback
        return plotly_view_to_video_camera(view if isinstance(view, dict) else None, fallback)

    def _request_action_camera(self, callback) -> None:
        if self.camera_timeline_mode.isChecked():
            self._ensure_camera_timeline()
            if isinstance(self._camera_timeline, dict):
                points = self._camera_timeline.get("points")
                if isinstance(points, list) and points and isinstance(points[0], dict):
                    camera = points[0].get("camera")
                    if isinstance(camera, dict):
                        callback(dict(camera))
                        return
        if self._camera_view_mode == "2d":
            # Interactive/fixed 2D export follows the live map view when one is
            # available. The numeric fields are hidden in this mode, so they are
            # only a fallback when no 2D result has been rendered yet.
            self._capture_2d_view_camera(mark_user=False, callback=callback)
            return
        if self._camera_view_mode != "3d":
            callback(self._fixed_camera_from_controls())
            return
        if self._fixed_camera_user_edited:
            callback(self._fixed_camera_from_controls())
            return
        request = getattr(getattr(self, "plot", None), "request_3d_view", None)
        if not callable(request):
            callback(self._fixed_camera_from_controls())
            return

        def received(view: Any) -> None:
            camera = self._camera_from_selector_view(view)
            self._set_fixed_camera_controls(camera, mark_user=False)
            callback(camera)

        request(received)

    def _timeline_duration_s(self) -> float:
        duration_s, _old_multiplier = remembered_waveform_timing(self._camera_view_mode)
        return max(1.0e-6, float(duration_s))

    def _ensure_camera_timeline(self) -> dict[str, Any]:
        duration_s = self._timeline_duration_s()
        default_camera = self._fixed_camera_from_controls()
        is_new = not isinstance(self._camera_timeline, dict)
        timeline = normalize_camera_timeline(
            self._camera_timeline,
            duration_s=duration_s,
            default_camera=default_camera,
            view_mode=self._camera_view_mode,
        )
        timeline["enabled"] = True
        timeline["scripted"] = True
        if is_new and timeline.get("points"):
            timeline["points"][0]["camera"] = copy.deepcopy(default_camera)
            timeline["points"][0]["playback_speed"] = float(self.camera_playback_speed.value())
        points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
        authored_end = max(
            (float(point.get("time_s", 0.0)) for point in points if isinstance(point, dict)),
            default=0.0,
        )
        timeline["loop_waveform"] = bool(self.camera_script_loop.isChecked())
        # The authored endpoint remains the last camera point, but when looping is
        # off the effective output extends until the waveform has completed once.
        timeline["duration_s"] = scripted_timeline_duration_s(
            timeline, fallback_s=duration_s
        )
        self._camera_timeline = timeline
        return timeline

    def _camera_timeline_summary_text(self, point: dict[str, Any]) -> str:
        camera = point.get("camera") if isinstance(point.get("camera"), dict) else {}
        if self._camera_view_mode == "2d":
            target = camera.get("target", [0.0, 0.0, 0.0])
            first_axis, second_axis, _normal_axis = self._camera_2d_plane_axes()
            zoom = float(camera.get("zoom_2d", 1.0))
            return (
                f"Centre {float(target[first_axis]):g}, "
                f"{float(target[second_axis]):g} mm • {zoom:g}×"
            )
        position = camera.get("position", [0.0, 0.0, 0.0])
        return f"{float(position[0]):g}, {float(position[1]):g}, {float(position[2]):g} mm"

    def _selected_camera_timeline_row(self) -> int:
        rows = self.camera_timeline_table.selectionModel().selectedRows()
        if rows:
            return max(0, rows[0].row())
        return 0

    def _refresh_camera_timeline_table(self, *, select_row: int | None = None) -> None:
        timeline = self._ensure_camera_timeline()
        points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
        selected = self._selected_camera_timeline_row() if select_row is None else int(select_row)
        selected = max(0, min(max(0, len(points) - 1), selected))
        self.camera_timeline_table.blockSignals(True)
        try:
            self.camera_timeline_table.setRowCount(len(points))
            for row, point in enumerate(points):
                values = (
                    str(row + 1),
                    f"{float(point.get('time_s', 0.0)):g} s",
                    f"{float(point.get('playback_speed', 1.0)):g}×",
                    self._camera_timeline_summary_text(point),
                )
                for column, value in enumerate(values):
                    self.camera_timeline_table.setItem(row, column, QTableWidgetItem(value))
            if points:
                self.camera_timeline_table.selectRow(selected)
        finally:
            self.camera_timeline_table.blockSignals(False)
        duration = self._timeline_duration_s()
        scripted_duration = max(0.0, float(timeline.get("duration_s", duration)))
        authored_end = max(
            (float(point.get("time_s", 0.0)) for point in points if isinstance(point, dict)),
            default=0.0,
        )
        fill = " • waveform/sequencer loop to fill" if self.camera_script_loop.isChecked() else ""
        hold = (
            " • final camera held to waveform end"
            if not self.camera_script_loop.isChecked() and scripted_duration > authored_end + 1.0e-9
            else ""
        )
        self.camera_timeline_summary.setText(
            f"Waveform {duration:g} s • output {scripted_duration:g} s{fill}{hold}"
        )
        self._camera_timeline_selection_changed()

    def _camera_timeline_selection_changed(self) -> None:
        if not hasattr(self, "camera_timeline_remove_button"):
            return
        row = self._selected_camera_timeline_row()
        has_rows = self.camera_timeline_table.rowCount() > 0
        editing = self.camera_timeline_edit_widget.isVisible()
        self.camera_timeline_edit_button.setEnabled(has_rows and not editing)
        self.camera_timeline_add_button.setEnabled(has_rows and not editing)
        self.camera_timeline_remove_button.setEnabled(has_rows and row > 0 and not editing)

    def _camera_timeline_row_clicked(self, row: int, _column: int) -> None:
        """Preview a saved timeline step when its table row is clicked."""
        if self.camera_timeline_edit_widget.isVisible():
            return
        timeline = self._ensure_camera_timeline()
        points = timeline.get("points") if isinstance(timeline.get("points"), list) else []
        if row < 0 or row >= len(points) or not isinstance(points[row], dict):
            return
        camera = points[row].get("camera")
        if not isinstance(camera, dict):
            return
        self._set_fixed_camera_controls(camera, mark_user=True)
        QTimer.singleShot(0, self._preview_fixed_camera)
        self._camera_status(f"Timeline step {row + 1} previewed")

    def _camera_timeline_add(self, *_args) -> None:
        timeline = self._ensure_camera_timeline()
        points = timeline.get("points", [])
        if not points:
            return
        row = self._selected_camera_timeline_row()
        row = max(0, min(len(points) - 1, row))
        current = points[row]
        duration = self._timeline_duration_s()
        current_time = float(current.get("time_s", 0.0))
        if row + 1 < len(points):
            next_time = float(points[row + 1].get("time_s", current_time))
            if next_time > current_time + 1.0e-9:
                guessed_time = (current_time + next_time) * 0.5
            else:
                guessed_time = current_time + max(duration * 0.1, 1.0)
        elif current_time < duration - 1.0e-9:
            guessed_time = (current_time + duration) * 0.5
        else:
            # Once the authored timeline reaches/exceeds the waveform, keep
            # extending naturally instead of pinning new points to its end.
            guessed_time = current_time + max(duration * 0.1, 1.0)
        template = copy.deepcopy(current)
        template["time_s"] = guessed_time

        # Insert the step immediately so Add feels like direct manipulation of
        # the timeline table. Cancel restores the timeline snapshot from before
        # this insertion; Accept edits this already-visible row in place.
        original_timeline = copy.deepcopy(timeline)
        points.append(copy.deepcopy(template))
        normalized = normalize_camera_timeline(
            timeline,
            duration_s=self._timeline_duration_s(),
            default_camera=self._fixed_camera_from_controls(),
            view_mode=self._camera_view_mode,
        )
        normalized["enabled"] = True
        normalized["scripted"] = True
        normalized["loop_waveform"] = bool(self.camera_script_loop.isChecked())
        normalized["duration_s"] = scripted_timeline_duration_s(
            normalized, fallback_s=self._timeline_duration_s()
        )
        self._camera_timeline = normalized
        new_row = min(
            range(len(normalized["points"])),
            key=lambda idx: abs(float(normalized["points"][idx]["time_s"]) - guessed_time),
        )
        self._refresh_camera_timeline_table(select_row=new_row)
        self._begin_camera_timeline_edit(
            new_row,
            copy.deepcopy(normalized["points"][new_row]),
            is_new=True,
            original_timeline=original_timeline,
        )

    def _camera_timeline_edit(self, *_args) -> None:
        timeline = self._ensure_camera_timeline()
        points = timeline.get("points", [])
        if not points:
            return
        row = max(0, min(len(points) - 1, self._selected_camera_timeline_row()))
        self._begin_camera_timeline_edit(row, copy.deepcopy(points[row]))

    def _begin_camera_timeline_edit(
        self,
        index: int,
        point: dict[str, Any],
        *,
        is_new: bool = False,
        original_timeline: dict[str, Any] | None = None,
    ) -> None:
        self._timeline_edit_index = int(index)
        self._timeline_edit_is_new = bool(is_new)
        self._timeline_edit_template = copy.deepcopy(point)
        self._timeline_edit_original_timeline = (
            copy.deepcopy(original_timeline) if isinstance(original_timeline, dict) else None
        )
        # Camera authoring can extend beyond the current excitation duration.
        # Later cues are preserved and simply ignored by shorter playback.
        self.camera_timeline_time.setMaximum(1.0e9)
        self.camera_timeline_time.setValue(float(point.get("time_s", 0.0)))
        self.camera_timeline_time.setEnabled(index != 0)
        self.camera_timeline_speed.setValue(float(point.get("playback_speed", 1.0)))
        camera = point.get("camera") if isinstance(point.get("camera"), dict) else self._fixed_camera_from_controls()
        self._set_fixed_camera_controls(camera, mark_user=True)
        self.camera_timeline_edit_widget.setVisible(True)
        self.camera_timeline_table.setEnabled(False)
        self._camera_timeline_selection_changed()
        QTimer.singleShot(0, self._preview_fixed_camera)
        if self._camera_view_mode == "3d":
            self._camera_status(
                "Timeline step active — move the Selector camera, then click Accept"
            )
        else:
            self._camera_status(
                "Timeline step active — move the 2D map viewer, then click Accept"
            )

    def _camera_timeline_remove(self, *_args) -> None:
        timeline = self._ensure_camera_timeline()
        points = timeline.get("points", [])
        row = self._selected_camera_timeline_row()
        if row <= 0 or row >= len(points):
            return
        points.pop(row)
        self._camera_timeline = normalize_camera_timeline(
            timeline,
            duration_s=self._timeline_duration_s(),
            default_camera=self._fixed_camera_from_controls(),
            view_mode=self._camera_view_mode,
        )
        self._refresh_camera_timeline_table(select_row=max(0, row - 1))

    def _camera_timeline_accept(self, *_args) -> None:
        if self._timeline_edit_template is None:
            return

        def commit(camera: dict[str, Any]) -> None:
            timeline = self._ensure_camera_timeline()
            points = timeline.get("points", [])
            point = {
                "time_s": float(self.camera_timeline_time.value()),
                "camera": copy.deepcopy(camera),
                "playback_speed": float(self.camera_timeline_speed.value()),
            }
            if self._timeline_edit_index is None or not points:
                return
            index = max(0, min(len(points) - 1, int(self._timeline_edit_index)))
            if index == 0:
                point["time_s"] = 0.0
            points[index] = point
            requested_time = point["time_s"]
            normalized = normalize_camera_timeline(
                timeline,
                duration_s=self._timeline_duration_s(),
                default_camera=camera,
                view_mode=self._camera_view_mode,
            )
            normalized["enabled"] = True
            self._camera_timeline = normalized
            nearest = min(
                range(len(normalized["points"])),
                key=lambda idx: abs(float(normalized["points"][idx]["time_s"]) - requested_time),
            )
            self._set_fixed_camera_controls(normalized["points"][nearest]["camera"], mark_user=True)
            self._finish_camera_timeline_edit()
            self._refresh_camera_timeline_table(select_row=nearest)
            self._camera_status(f"Timeline step {nearest + 1} saved")

        if self._camera_view_mode == "2d":
            # Match the 3D authoring workflow: Accept captures the live map view
            # after a pan/zoom, while typed values remain the fallback.
            self._capture_2d_view_camera(mark_user=True, callback=commit)
            return
        if self._camera_view_mode != "3d":
            commit(self._fixed_camera_from_controls())
            return

        # Accept captures the *live* Selector view. This makes the authoring
        # workflow Edit -> move/orbit/zoom -> Accept, without requiring Get.
        request = getattr(getattr(self, "plot", None), "request_3d_view", None)
        if not callable(request):
            commit(self._fixed_camera_from_controls())
            return

        def received(view: Any) -> None:
            if isinstance(view, dict):
                camera = self._camera_from_selector_view(view)
                self._set_fixed_camera_controls(camera, mark_user=True)
                commit(camera)
            else:
                commit(self._fixed_camera_from_controls())

        request(received)

    def _finish_camera_timeline_edit(self) -> None:
        self._timeline_edit_index = None
        self._timeline_edit_is_new = False
        self._timeline_edit_template = None
        self._timeline_edit_original_timeline = None
        self.camera_timeline_edit_widget.setVisible(False)
        self.camera_timeline_table.setEnabled(True)
        self._camera_timeline_selection_changed()

    def _camera_timeline_cancel(self, *_args) -> None:
        if not hasattr(self, "camera_timeline_edit_widget"):
            return
        restore = (
            copy.deepcopy(self._timeline_edit_original_timeline)
            if self._timeline_edit_is_new and isinstance(self._timeline_edit_original_timeline, dict)
            else None
        )
        self._finish_camera_timeline_edit()
        if restore is not None:
            self._camera_timeline = restore
            self._refresh_camera_timeline_table()
            self._camera_status("New timeline step cancelled")

    def _camera_timeline_for_action(self, initial_camera: dict[str, Any]) -> dict[str, Any] | None:
        if not self.camera_timeline_mode.isChecked():
            return None
        timeline = normalize_camera_timeline(
            self._camera_timeline,
            duration_s=self._timeline_duration_s(),
            default_camera=initial_camera,
            view_mode=self._camera_view_mode,
        )
        timeline["enabled"] = True
        timeline["scripted"] = True
        timeline["loop_waveform"] = bool(self.camera_script_loop.isChecked())
        timeline["duration_s"] = scripted_timeline_duration_s(
            timeline, fallback_s=self._timeline_duration_s()
        )
        self._camera_timeline = copy.deepcopy(timeline)
        return timeline

    def _camera_playback_options_for_action(self) -> tuple[float, float]:
        fps = max(1.0, float(self.camera_target_fps.value()))
        # Timeline point speeds are absolute multipliers, so the shared base is
        # 1x in timeline mode. Interactive/fixed uses the single global speed.
        speed = 1.0 if self.camera_timeline_mode.isChecked() else max(
            1.0e-6, float(self.camera_playback_speed.value())
        )
        return fps, speed

    def _camera_persistent_state(self) -> dict[str, Any]:
        return {"camera_timeline": copy.deepcopy(self._camera_timeline)}

    def _restore_camera_persistent_state(self, state: dict[str, Any]) -> None:
        timeline = state.get("camera_timeline") if isinstance(state, dict) else None
        self._camera_timeline = copy.deepcopy(timeline) if isinstance(timeline, dict) else None
        if isinstance(self._camera_timeline, dict) and hasattr(self, "camera_script_loop"):
            self.camera_script_loop.setChecked(bool(self._camera_timeline.get("loop_waveform", False)))
        if getattr(self, "camera_timeline_mode", None) is not None and self.camera_timeline_mode.isChecked():
            self._refresh_camera_timeline_table()


class _FieldRenderSignalRelay(QObject):
    """Marshal render-worker notifications onto the Qt GUI thread.

    The shared render controller is a plain Python mixin, so its inherited
    ``@Slot`` methods are not guaranteed to participate in PySide's QObject
    meta-object dispatch.  Connecting a worker QThread directly to those
    methods can therefore execute result handling in the worker thread.  That
    is harmless for bookkeeping but fatal when the completed 2D path creates a
    PlotView/QWebEngine surface (Qt aborts if its OpenGL context is touched from
    the wrong thread).  This QObject relay has unambiguous GUI-thread affinity
    and is the sole receiver of cross-thread worker notifications.
    """

    def __init__(self, owner: QWidget):
        super().__init__(owner)
        self._owner = owner

    @Slot(int, dict)
    def progress(self, job_id: int, update: dict[str, Any]) -> None:
        self._owner._render_job_progress(job_id, update)

    @Slot(int, dict)
    def completed(self, job_id: int, result: dict[str, Any]) -> None:
        self._owner._render_job_completed(job_id, result)

    @Slot(int, str)
    def failed(self, job_id: int, message: str) -> None:
        self._owner._render_job_failed(job_id, message)

    @Slot(int)
    def cancelled(self, job_id: int) -> None:
        self._owner._render_job_cancelled(job_id)

    @Slot()
    def thread_finished(self) -> None:
        thread = self.sender()
        if isinstance(thread, QThread):
            self._owner._render_job_thread_finished(thread)


class _BackgroundFieldRenderMixin:
    """Shared concurrent render-job controller for 2D and 3D map windows."""

    def _render_signal_relay(self) -> _FieldRenderSignalRelay:
        relay = getattr(self, "_field_render_signal_relay", None)
        if not isinstance(relay, _FieldRenderSignalRelay):
            relay = _FieldRenderSignalRelay(self)
            self._field_render_signal_relay = relay
        return relay

    def _render_preferences_snapshot(self) -> dict[str, Any]:
        return {
            "coil_modeling_method": str(self.adapter.coil_modeling_method),
            "solver_memory_budget_mb": int(self.adapter.solver_memory_budget_mb),
            "worker_count": normalize_optimizer_worker_count(
                self.adapter.optimizer_worker_count
            ),
            "solver_diagnostics_enabled": bool(
                self.adapter.solver_diagnostics_enabled
            ),
            "fluxline_minimum_field_cutoff_enabled": bool(
                self.adapter.fluxline_minimum_field_cutoff_enabled
            ),
            "fluxline_minimum_field_cutoff_percent": float(
                self.adapter.fluxline_minimum_field_cutoff_percent
            ),
            "fluxline_conductor_cutoff_enabled": bool(
                self.adapter.fluxline_conductor_cutoff_enabled
            ),
            "fluxline_conductor_cutoff_mm": float(
                self.adapter.fluxline_conductor_cutoff_mm
            ),
        }

    def _active_render_job_count(self) -> int:
        return sum(
            1
            for job in self._render_jobs.values()
            if not job.get("terminal", False) and not job.get("discard", False)
        )

    def _start_render_job(
        self,
        *,
        render_kind: str,
        request: dict[str, Any],
        job_label: str,
        tooltip: str,
    ) -> None:
        """Create a progress tab immediately and run its frozen request off-thread."""

        try:
            if self._render_document_snapshot is None:
                self._render_document_snapshot = self.adapter.document()
            document = self._render_document_snapshot
            preferences = self._render_preferences_snapshot()
            configure_field_render_worker_limit(preferences["worker_count"])
        except Exception as error:  # noqa: BLE001 - GUI snapshot boundary
            QMessageBox.warning(self, f"Unable to start {job_label}", str(error))
            return

        self._render_job_serial += 1
        job_id = self._render_job_serial
        page = _FieldRenderJobPage(
            job_label,
            render_kind=render_kind,
            request=request,
        )
        index = self.tabs.addTab(page, job_label)
        self.tabs.setTabToolTip(index, f"{tooltip} — calculating in background")
        self.tabs.setCurrentIndex(index)

        thread = QThread()
        thread.setObjectName(f"FieldRenderJob-{job_id}-{render_kind}")
        worker = _FieldRenderWorker(
            job_id=job_id,
            document=document,
            preferences=preferences,
            render_kind=render_kind,
            request=request,
        )
        worker.moveToThread(thread)
        thread._field_render_worker = worker  # type: ignore[attr-defined]
        _ensure_field_render_shutdown_hook()
        _ACTIVE_FIELD_RENDER_THREADS.add(thread)
        self._render_jobs[job_id] = {
            "page": page,
            "thread": thread,
            "worker": worker,
            "label": job_label,
            "tooltip": tooltip,
            "render_kind": render_kind,
            "terminal": False,
            "discard": False,
            "cancel_requested": False,
        }

        page.action_requested.connect(
            lambda job_id=job_id, page=page: self._render_job_page_action(
                job_id, page
            )
        )
        relay = self._render_signal_relay()
        thread.started.connect(worker.run)
        # _BackgroundFieldRenderMixin is intentionally not a QObject subclass.
        # Route every worker->GUI notification through a real QObject with GUI
        # affinity so PlotView/QWebEngine creation can never run on the worker
        # QThread. Explicit QueuedConnection makes that contract independent of
        # PySide's AutoConnection treatment of inherited Python mixin methods.
        worker.progress.connect(relay.progress, Qt.ConnectionType.QueuedConnection)
        worker.completed.connect(relay.completed, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(relay.failed, Qt.ConnectionType.QueuedConnection)
        worker.cancelled.connect(relay.cancelled, Qt.ConnectionType.QueuedConnection)
        for terminal_signal in (worker.completed, worker.failed, worker.cancelled):
            terminal_signal.connect(worker.deleteLater)
            terminal_signal.connect(
                thread.quit, Qt.ConnectionType.DirectConnection
            )
        thread.finished.connect(
            relay.thread_finished, Qt.ConnectionType.QueuedConnection
        )
        QTimer.singleShot(_FIELD_RENDER_BROWSER_RELEASE_DELAY_MS, thread.start)
        active = self._active_render_job_count()
        self.calculation_status.showMessage(
            f"{job_label} submitted • {active} render job"
            f"{'s' if active != 1 else ''} running or queued"
        )

    def _render_job_page_action(
        self, job_id: int, page: _FieldRenderJobPage
    ) -> None:
        if page.terminal:
            index = self.tabs.indexOf(page)
            if index >= 0:
                self._close_map_tab(index)
            return
        job = self._render_jobs.get(job_id)
        if job is None or job.get("terminal", False):
            return
        worker = job.get("worker")
        if isinstance(worker, _FieldRenderWorker):
            worker.request_cancellation()
        job["cancel_requested"] = True
        page.set_cancelling()
        self.calculation_status.showMessage(f"Cancelling {job.get('label', 'render')}…")

    @Slot(int, dict)
    def _render_job_progress(self, job_id: int, update: dict[str, Any]) -> None:
        job = self._render_jobs.get(job_id)
        if (
            job is None
            or job.get("discard", False)
            or job.get("cancel_requested", False)
        ):
            return
        page = job.get("page")
        if isinstance(page, _FieldRenderJobPage):
            page.update_progress(update)

    @Slot(int, dict)
    def _render_job_completed(self, job_id: int, result: dict[str, Any]) -> None:
        job = self._render_jobs.get(job_id)
        if job is None:
            _cleanup_field_render_staging(result)
            return
        job["terminal"] = True
        if job.get("discard", False):
            _cleanup_field_render_staging(result)
            return
        page = job.get("page")
        if not isinstance(page, _FieldRenderJobPage):
            _cleanup_field_render_staging(result)
            return
        if job.get("cancel_requested", False):
            _cleanup_field_render_staging(result)
            page.set_cancelled()
            self.calculation_status.showMessage(
                f"{job.get('label', 'Render')} cancelled"
            )
            return
        index = self.tabs.indexOf(page)
        if index < 0:
            job["discard"] = True
            _cleanup_field_render_staging(result)
            return
        page.set_rendering()
        try:
            isolated_result = _load_field_render_result(result)
        except Exception as error:  # noqa: BLE001 - isolated result boundary
            self._render_job_failed(
                job_id,
                f"Unable to open the isolated render result: {error}",
            )
            return

        # View-specific consumers can inspect/consume auxiliary data from the
        # isolated process before the common result page is constructed. Brain
        # View uses this to update its LPBA40 statistics from the exact static
        # field solve while still sharing the ordinary 3D render controller.
        result_hook = getattr(self, "_render_job_result_loaded", None)
        if callable(result_hook):
            try:
                result_hook(job_id, job, isolated_result)
            except Exception as error:  # noqa: BLE001 - view hook boundary
                self._render_job_failed(job_id, str(error))
                return

        was_current = self.tabs.currentWidget() is page
        report = isolated_result.get("solver_report")
        title = str(job.get("label", "Map"))
        elapsed = float(result.get("elapsed_seconds", page.elapsed_seconds()))
        render_kind = str(job.get("render_kind", ""))
        try:
            if render_kind == "2d_static":
                figure = isolated_result.get("figure")
                if not isinstance(figure, dict):
                    raise StudioOperationError(
                        "The render worker returned no 2D map figure."
                    )
                plot = PlotView(
                    image_filename=f"2d-field-map-{job_id}"
                )
                plot.set_figure(figure)
                connector = getattr(self, "_connect_result_camera_viewer", None)
                if callable(connector):
                    connector(plot)
                page.set_ready()
                self.tabs.removeTab(index)
                new_index = self.tabs.insertTab(index, plot, title)
                self.tabs.setTabToolTip(new_index, str(job.get("tooltip", title)))
                page.deleteLater()
                if was_current:
                    self.tabs.setCurrentIndex(new_index)
                self.calculation_status.showMessage(
                    f"{title} ready • Solver: {_coil_solver_summary(report)} • "
                    f"elapsed {_format_elapsed_time(elapsed)}"
                )
                return

            payload = isolated_result.get("payload")
            if not isinstance(payload, dict):
                raise StudioOperationError(
                    "The render worker returned no WebGL payload."
                )
            plot = WaveformGLView(payload, start_hibernated=not was_current)
            connector = getattr(self, "_connect_result_camera_viewer", None)
            if callable(connector):
                connector(plot)
            tab_widget: QWidget = plot
            wrapper = getattr(self, "_wrap_render_result_view", None)
            if callable(wrapper):
                wrapped = wrapper(plot, job, isolated_result)
                if isinstance(wrapped, QWidget):
                    tab_widget = wrapped
            plot.ready.connect(
                lambda ok, plot=plot, tab_widget=tab_widget, title=title, report=report, elapsed=elapsed: (
                    self._render_gpu_view_ready(plot, title, report, elapsed, ok, tab_widget)
                )
            )
            page.set_ready()
            self.tabs.removeTab(index)
            new_index = self.tabs.insertTab(index, tab_widget, title)
            self.tabs.setTabToolTip(new_index, str(job.get("tooltip", title)))
            page.deleteLater()
            if was_current:
                self.tabs.setCurrentIndex(new_index)
            self._render_tab_changed(self.tabs.currentIndex())
            self.calculation_status.showMessage(
                f"{title} calculated • loading GPU viewer… • "
                f"elapsed {_format_elapsed_time(elapsed)}"
            )
        except Exception as error:  # noqa: BLE001 - result-view boundary
            self._render_job_failed(job_id, str(error))

    def _render_gpu_view_ready(
        self,
        plot: WaveformGLView,
        title: str,
        report: Any,
        elapsed: float,
        ok: bool,
        tab_widget: QWidget | None = None,
    ) -> None:
        owner = tab_widget if isinstance(tab_widget, QWidget) else plot
        if self.tabs.indexOf(owner) < 0:
            return
        if ok:
            self.calculation_status.showMessage(
                f"{title} ready • Solver: {_coil_solver_summary(report)} • "
                f"elapsed {_format_elapsed_time(elapsed)}"
            )
        else:
            self.calculation_status.showMessage(
                f"{title} calculated • GPU viewer waiting to retry"
            )

    @Slot(int, str)
    def _render_job_failed(self, job_id: int, message: str) -> None:
        job = self._render_jobs.get(job_id)
        if job is None:
            return
        job["terminal"] = True
        if job.get("discard", False):
            return
        page = job.get("page")
        if isinstance(page, _FieldRenderJobPage):
            page.set_failed(message)
        self.calculation_status.showMessage(
            f"{job.get('label', 'Render')} failed after "
            f"{_format_elapsed_time(page.elapsed_seconds() if isinstance(page, _FieldRenderJobPage) else 0.0)}"
        )

    @Slot(int)
    def _render_job_cancelled(self, job_id: int) -> None:
        job = self._render_jobs.get(job_id)
        if job is None:
            return
        job["terminal"] = True
        if job.get("discard", False):
            return
        page = job.get("page")
        if isinstance(page, _FieldRenderJobPage):
            page.set_cancelled()
        self.calculation_status.showMessage(f"{job.get('label', 'Render')} cancelled")

    def _render_job_thread_finished(self, thread: QThread) -> None:
        """Release one finished worker after its GUI-thread relay fires."""

        if not isinstance(thread, QThread):
            return
        matching_job_id = next(
            (
                job_id
                for job_id, job in self._render_jobs.items()
                if job.get("thread") is thread
            ),
            None,
        )
        if matching_job_id is not None:
            self._render_jobs.pop(matching_job_id, None)
        if not self._render_jobs:
            self._render_document_snapshot = None
        thread._field_render_worker = None  # type: ignore[attr-defined]
        _ACTIVE_FIELD_RENDER_THREADS.discard(thread)
        thread.deleteLater()

    def _cancel_all_render_jobs(self) -> None:
        for job in self._render_jobs.values():
            job["discard"] = True
            worker = job.get("worker")
            if isinstance(worker, _FieldRenderWorker):
                worker.request_cancellation()


class FieldMapDialog(_BackgroundFieldRenderMixin, _CameraTimelineMixin, _PlaybackCoilSelectionMixin, QDialog):
    """Planar magnitude/component field map with a tabbed embedded 3D plane selector."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        # A map window is a self-contained workspace.  Rebuild its solver model
        # from a deep launch-time snapshot so later source-scene edits, scene-tab
        # switches, and source-window closure cannot alter an open viewer.
        self.adapter = adapter.detached_copy()
        self._selector_scene_figure: dict[str, Any] | None = None
        self._selector_scene_dirty = True
        self._map_tab_counter = 0
        self._waveform_tab_counter = 0
        self._render_job_serial = 0
        self._render_jobs: dict[int, dict[str, Any]] = {}
        self._render_document_snapshot: dict[str, Any] | None = None
        self._active_render_view: WaveformGLView | None = None
        self.setWindowTitle("2D Field Map")
        self.resize(1280, 760)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(300)
        control_column = QWidget()
        control_column.setMinimumWidth(300)
        control_column_layout = QVBoxLayout(control_column)
        control_column_layout.setContentsMargins(0, 0, 0, 0)
        control_column_layout.setSpacing(0)
        control_column_layout.addWidget(control_scroll, 1)
        control_panel = QWidget()
        control_panel.setMinimumWidth(0)
        control_panel.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        control_root = QVBoxLayout(control_panel)
        control_root.setContentsMargins(8, 8, 8, 8)
        control_scroll.setWidget(control_panel)
        viewer_panel = QWidget()
        viewer_root = QVBoxLayout(viewer_panel)
        viewer_root.setContentsMargins(0, 0, 0, 0)
        self.plane = QComboBox()
        self.plane.addItems(["xy", "xz", "yz"])
        self.plane.setCurrentText("xy")
        self.offset = _spin(0.0, -10000, 10000, 2, " mm")
        self.span = _spin(300.0, 1, 100000, 1, " mm")
        self.resolution = QSpinBox()
        self.resolution.setRange(10, 150)
        self.resolution.setValue(55)
        self.field = QComboBox()
        self.field.addItems(["B", "H"])
        self.component = QComboBox()
        self.component.addItem("Magnitude |B|", "magnitude")
        self.component.addItem("X magnitude |Bx|", "abs_x")
        self.component.addItem("Y magnitude |By|", "abs_y")
        self.component.addItem("Z magnitude |Bz|", "abs_z")
        self.component.insertSeparator(self.component.count())
        self.component.addItem("X signed Bx", "x")
        self.component.addItem("Y signed By", "y")
        self.component.addItem("Z signed Bz", "z")
        self.display_mode = QComboBox()
        self.display_mode.addItem("Heatmap", "heatmap")
        self.display_mode.addItem("Fluxlines", "flux")
        self.display_mode.addItem("Heatmap + fluxlines", "both")
        self.display_mode.setCurrentIndex(2)
        self.flux_density = QSpinBox()
        self.flux_density.setRange(2, 64)
        self.flux_density.setValue(9)
        self.flux_density.setToolTip(
            "Approximate number of fluxline seed cells per axis for the 2D streamline overlay"
        )
        self.logarithmic = QCheckBox("Log scale")
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(True)
        self.show_grid.setToolTip(
            "Show the planar coordinate grid in static and animated 2D maps"
        )
        self.object_outlines = QCheckBox("Object outlines")
        self.object_outlines.setChecked(True)
        self.object_outlines.setToolTip(
            "Overlay cross-section outlines of visible solid scene objects where the selected map plane cuts them"
        )
        self._outline_selected_object_ids: set[str] = set()
        self._outline_known_object_ids: set[str] = set()
        self._outline_selection_initialized = False
        self.object_outline_menu = QMenu(self)
        self.object_outline_menu.aboutToShow.connect(self._refresh_object_outline_menu)
        self.object_outline_objects_button = QPushButton("Objects…")
        self.object_outline_objects_button.setMenu(self.object_outline_menu)
        self.object_outline_objects_button.setToolTip(
            "Choose which solid scene objects are drawn in the Selector preview and as "
            "cross-section outlines in this map. Clicking an object in the preview legend "
            "updates this checklist. A new map window starts from the scene's current visibility."
        )
        self.object_outlines.toggled.connect(self.object_outline_objects_button.setEnabled)
        self._refresh_object_outline_menu()
        self._init_playback_coil_selector()

        self.map_plane_section = CollapsibleSection("Map plane", expanded=False)
        map_plane_layout = QFormLayout()
        map_plane_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        map_plane_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        map_plane_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        map_plane_layout.addRow("Plane", self.plane)
        map_plane_layout.addRow("Offset", self.offset)
        map_plane_layout.addRow("View width", self.span)
        self.map_plane_section.setContentLayout(map_plane_layout)

        self.sample_config_section = CollapsibleSection("Sample config", expanded=False)
        sample_config_layout = QFormLayout()
        sample_config_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        sample_config_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        sample_config_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        sample_config_layout.addRow("Samples / axis", self.resolution)
        sample_config_layout.addRow("Field", self.field)
        sample_config_layout.addRow("Show", self.component)
        sample_config_layout.addRow("Flux density", self.flux_density)
        self.sample_config_section.setContentLayout(sample_config_layout)

        field_view_row = QWidget()
        field_view_row_layout = QFormLayout(field_view_row)
        field_view_row_layout.setContentsMargins(0, 2, 0, 2)
        field_view_row_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        field_view_row_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        field_view_row_layout.addRow("Field view", self.display_mode)

        self.show_contours = QCheckBox("Show contours")
        self.show_contours.setChecked(False)
        self.show_contours.setToolTip(
            "Overlay topographic-style contour lines that follow equal field intensity values"
        )
        self.contour_count = QSpinBox()
        self.contour_count.setRange(1, 50)
        self.contour_count.setValue(10)
        self.contour_count.setToolTip("Approximate number of contour bands to draw")
        self.contour_labels = QCheckBox("Labels")
        self.contour_labels.setToolTip("Show field values on contour lines")
        field_view_options_layout = QGridLayout()
        field_view_options_layout.addWidget(self.show_contours, 0, 0, 1, 2)
        field_view_options_layout.addWidget(QLabel("Contour count"), 1, 0)
        field_view_options_layout.addWidget(self.contour_count, 1, 1)
        field_view_options_layout.addWidget(self.contour_labels, 2, 0, 1, 2)
        field_view_options_layout.setColumnStretch(1, 1)
        self.field_view_options_section = CollapsibleSection(
            "Field view options", expanded=True
        )
        self.field_view_options_section.setContentLayout(field_view_options_layout)

        scene_options_layout = QVBoxLayout()
        scene_options_layout.setContentsMargins(0, 0, 0, 0)
        scene_options_layout.setSpacing(3)
        scene_options_layout.addWidget(self.object_outlines)
        scene_options_layout.addWidget(self.object_outline_objects_button)
        active_coils_label = QLabel("Active coils")
        active_coils_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        scene_options_layout.addWidget(active_coils_label)
        scene_options_layout.addWidget(self.playback_coils_button)
        self.scene_options_section = CollapsibleSection("Scene options", expanded=False)
        self.scene_options_section.setContentLayout(scene_options_layout)

        self.colour_minimum_enabled = QCheckBox("Lower limit")
        self.colour_minimum_enabled.setMinimumWidth(
            self.colour_minimum_enabled.sizeHint().width() + 8
        )
        self.colour_minimum = _spin(0.0, -1e12, 1e12, 3, " µT")
        self.colour_minimum.setEnabled(False)
        self.colour_maximum_enabled = QCheckBox("Upper limit")
        self.colour_maximum_enabled.setMinimumWidth(
            self.colour_maximum_enabled.sizeHint().width() + 8
        )
        self.colour_maximum = _spin(1000.0, -1e12, 1e12, 3, " µT")
        self.colour_maximum.setEnabled(False)

        display_options_layout = QGridLayout()
        display_options_layout.setColumnStretch(1, 1)
        display_options_layout.addWidget(self.logarithmic, 0, 0, 1, 2)
        display_options_layout.addWidget(self.show_grid, 1, 0, 1, 2)
        colour_scale_title = QLabel("Colour scale")
        colour_scale_title.setStyleSheet("font-weight: 600;")
        display_options_layout.addWidget(colour_scale_title, 2, 0, 1, 2)
        self.colour_limits_hint = QLabel("Leave limits unchecked for automatic range")
        self.colour_limits_hint.setWordWrap(True)
        self.colour_limits_hint.setObjectName("hint")
        display_options_layout.addWidget(self.colour_limits_hint, 3, 0, 1, 2)
        display_options_layout.addWidget(self.colour_minimum_enabled, 4, 0)
        display_options_layout.addWidget(self.colour_minimum, 4, 1)
        display_options_layout.addWidget(self.colour_maximum_enabled, 5, 0)
        display_options_layout.addWidget(self.colour_maximum, 5, 1)
        self.display_options_section = CollapsibleSection("Display Options", expanded=True)
        self.display_options_section.setToolTip(
            "Colour scaling and presentation controls. For B maps the limits use µT; "
            "for H maps they use A/m."
        )
        self.display_options_section.setContentLayout(display_options_layout)

        self._selector_preview_timer = QTimer(self)
        self._selector_preview_timer.setSingleShot(True)
        self._selector_preview_timer.setInterval(SELECTOR_PREVIEW_DEBOUNCE_MS)
        self._selector_preview_timer.timeout.connect(self._render_preview_plane)
        self.object_outlines.toggled.connect(self._preview_scene_options_changed)
        self.plane.currentTextChanged.connect(self.preview_plane)
        self.offset.valueChanged.connect(self.preview_plane)
        self.span.valueChanged.connect(self.preview_plane)

        for widget in (
            self.plane,
            self.offset,
            self.span,
            self.resolution,
            self.field,
            self.component,
            self.display_mode,
            self.flux_density,
            self.contour_count,
            self.object_outline_objects_button,
            self.colour_minimum,
            self.colour_maximum,
        ):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy()
            )

        self._init_camera_timeline_controls(view_mode="2d")
        self.calculate_button = QPushButton("Static map")
        self.calculate_button.setObjectName("primaryButton")
        self.calculate_button.clicked.connect(self.calculate)
        self.waveform_button = QPushButton("Animated map")
        self.waveform_button.setToolTip(
            "Create a quasi-static current/voltage waveform playback using this map plane and field display"
        )
        self.waveform_button.clicked.connect(self.open_waveform_playback)
        self.reset_defaults_button = QPushButton("Reset defaults")
        self.reset_defaults_button.setToolTip(
            "Restore the built-in 2D map settings and replace the remembered settings for this window"
        )
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        self.exit_button = QPushButton("Exit")
        self.exit_button.setToolTip("Close this 2D Field Map workspace")
        self.exit_button.clicked.connect(self.close)

        control_root.addWidget(self.map_plane_section)
        control_root.addWidget(self.sample_config_section)
        control_root.addWidget(field_view_row)
        control_root.addWidget(self.field_view_options_section)
        control_root.addWidget(self.scene_options_section)
        control_root.addWidget(self.display_options_section)
        control_root.addWidget(self.camera_section)
        control_root.addStretch(1)

        # Keep primary actions visible while the setup sections scroll above them.
        action_panel = QWidget()
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(8, 6, 8, 8)
        action_layout.setSpacing(6)
        # Keep an active timeline-step editor pinned with the primary actions.
        # The table itself remains in the scrolling Camera section, but Step
        # time / playback speed / Accept / Cancel cannot be scrolled away.
        action_layout.addWidget(self.camera_timeline_edit_widget)
        action_layout.addWidget(self.calculate_button)
        action_layout.addWidget(self.waveform_button)
        action_layout.addWidget(self.reset_defaults_button)
        action_layout.addWidget(self.exit_button)
        control_column_layout.addWidget(action_panel, 0)

        # Treat the mouse wheel as navigation throughout the setup pane.
        # Spin boxes and closed combo boxes otherwise consume wheel events and
        # silently change parameters while the user is only trying to scroll.
        self._control_wheel_filter = EditorWheelScrollFilter(control_scroll)
        for editor_type in (QDoubleSpinBox, QSpinBox, QComboBox):
            for editor in control_panel.findChildren(editor_type):
                editor.installEventFilter(self._control_wheel_filter)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_map_tab)
        self.plot = PlotView(
            include_3d_return_axes=True,
            synchronize_legend_visibility=True,
            image_filename="2d-field-map-selector",
        )
        self.plot.visibility_requested.connect(
            self._preview_object_visibility_requested
        )
        self._selector_tab_index = self.tabs.addTab(self.plot, "Selector")
        self.tabs.currentChanged.connect(self._render_tab_changed)
        tab_bar = self.tabs.tabBar()
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.LeftSide, None
        )
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.RightSide, None
        )
        viewer_root.addWidget(self.tabs, 1)

        viewer_actions = QWidget()
        viewer_actions_layout = QHBoxLayout(viewer_actions)
        viewer_actions_layout.setContentsMargins(8, 0, 8, 6)
        viewer_actions_layout.addStretch(1)
        self.export_video_button = QPushButton("Export video")
        self.export_video_button.setToolTip(
            "Export the selected rendered animation without recalculating its field"
        )
        self.export_video_button.clicked.connect(self._export_current_animation_video)
        self.export_video_button.setVisible(False)
        viewer_actions_layout.addWidget(self.export_video_button)
        self.fullscreen_button = QPushButton("Fullscreen viewer")
        self.fullscreen_button.setToolTip(
            "Open a separate true-fullscreen interactive field-map viewer (Esc/F11 closes it)"
        )
        self.fullscreen_button.clicked.connect(self._toggle_current_plot_fullscreen)
        viewer_actions_layout.addWidget(self.fullscreen_button)
        viewer_root.addWidget(viewer_actions)
        self.calculation_status = QStatusBar()
        self.calculation_status.setObjectName("fieldMapStatusBar")
        self.calculation_status.setSizeGripEnabled(False)
        self.calculation_status.showMessage("Ready")
        viewer_root.addWidget(self.calculation_status)
        splitter.addWidget(control_column)
        splitter.addWidget(viewer_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 940])
        root.addWidget(splitter, 1)
        self.finished.connect(lambda _result: self._cleanup_plots())
        self.colour_minimum_enabled.toggled.connect(
            self._update_colour_limit_availability
        )
        self.colour_maximum_enabled.toggled.connect(
            self._update_colour_limit_availability
        )
        self.display_mode.currentIndexChanged.connect(
            self._update_colour_limit_availability
        )
        self.component.currentIndexChanged.connect(self._update_colour_limit_availability)
        self.show_contours.toggled.connect(self._update_colour_limit_availability)
        self.field.currentTextChanged.connect(self._update_colour_limit_units)
        self.field.currentTextChanged.connect(
            lambda value: _update_field_component_combo_labels(self.component, value)
        )
        self._update_colour_limit_units()
        _update_field_component_combo_labels(self.component, self.field.currentText())
        self._update_colour_limit_availability()
        self._render_preview_plane()
        self.calculation_status.showMessage(
            "Ready — position the plane in Selector, then click Static map"
        )
        capture_window_default_settings(self)

    def _reset_defaults(self, *_args) -> None:
        if not reset_window_to_default_settings(self):
            return
        self._update_colour_limit_units()
        _update_field_component_combo_labels(self.component, self.field.currentText())
        self._update_colour_limit_availability()
        self._render_preview_plane()
        self.calculation_status.showMessage("2D map settings reset to defaults")

    def _persistent_ui_state(self) -> dict[str, Any]:
        # Scene/coil checklists remain per-window-session; only the authored camera
        # timeline is durable because it is part of the user's playback setup.
        return self._camera_persistent_state()

    def _restore_persistent_ui_state(self, state: dict[str, Any]) -> None:
        self._restore_camera_persistent_state(state)

    def _refresh_object_outline_menu(self) -> None:
        choices = self.adapter.field_map_outline_objects()
        current_ids = {str(choice.get("id", "")) for choice in choices if choice.get("id")}
        if not self._outline_selection_initialized:
            self._outline_selected_object_ids = {
                str(choice["id"])
                for choice in choices
                if choice.get("id") and bool(choice.get("visible", True))
            }
            self._outline_selection_initialized = True
        else:
            # Once opened, the Objects checklist is local to this map window.
            self._outline_selected_object_ids.intersection_update(current_ids)
        self._outline_known_object_ids = current_ids

        self.object_outline_menu.clear()
        select_all = self.object_outline_menu.addAction("Select all")
        select_all.triggered.connect(self._select_all_object_outlines)
        clear_all = self.object_outline_menu.addAction("Clear all")
        clear_all.triggered.connect(self._clear_all_object_outlines)
        self.object_outline_menu.addSeparator()

        if not choices:
            empty = self.object_outline_menu.addAction("No outline-capable objects")
            empty.setEnabled(False)
        else:
            for choice in choices:
                object_id = str(choice["id"])
                label = str(choice.get("label") or object_id)
                if not bool(choice.get("visible", True)):
                    label += " (hidden)"
                action = self.object_outline_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(object_id in self._outline_selected_object_ids)
                action.setData(object_id)
                action.toggled.connect(
                    lambda checked, selected_id=object_id: self._set_object_outline_selected(
                        selected_id, checked
                    )
                )
        self._update_object_outline_button_text(len(current_ids))

    def _update_object_outline_button_text(self, total: int | None = None) -> None:
        if total is None:
            total = len(self._outline_known_object_ids)
        selected = len(self._outline_selected_object_ids & self._outline_known_object_ids)
        if total:
            self.object_outline_objects_button.setText(f"Objects {selected}/{total}…")
        else:
            self.object_outline_objects_button.setText("Objects 0…")

    def _set_object_outline_selected(self, object_id: str, checked: bool) -> None:
        if checked:
            self._outline_selected_object_ids.add(str(object_id))
        else:
            self._outline_selected_object_ids.discard(str(object_id))
        self._update_object_outline_button_text()
        self._preview_scene_options_changed()

    def _select_all_object_outlines(self, *_args) -> None:
        self._outline_selected_object_ids = set(self._outline_known_object_ids)
        self._refresh_object_outline_menu()
        self._preview_scene_options_changed()

    def _clear_all_object_outlines(self, *_args) -> None:
        self._outline_selected_object_ids.clear()
        self._refresh_object_outline_menu()
        self._preview_scene_options_changed()

    def selected_object_outline_ids(self) -> list[str]:
        self._refresh_object_outline_menu()
        return sorted(self._outline_selected_object_ids & self._outline_known_object_ids)

    def _update_colour_limit_units(self, *_args) -> None:
        magnetic_flux_density = self.field.currentText() == "B"
        suffix = " µT" if magnetic_flux_density else " A/m"
        decimals = 3
        step = 1.0
        default_upper = 1000.0
        for spin in (self.colour_minimum, self.colour_maximum):
            spin.setSuffix(suffix)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
        self.colour_minimum.setValue(0.0)
        self.colour_maximum.setValue(default_upper)

    def _update_colour_limit_availability(self, *_args) -> None:
        heatmap_visible = self.display_mode.currentData() in {"heatmap", "both"}
        magnitude = field_component_is_magnitude(self.component.currentData())
        signed = str(self.component.currentData()) in {"x", "y", "z"}
        self.colour_limits_hint.setText(
            "Signed components use a zero-centred ± range; either checked limit sets the mirrored extent."
            if signed
            else "Leave limits unchecked for automatic range"
        )
        if not magnitude:
            self.logarithmic.setChecked(False)
        self.field_view_options_section.setEnabled(heatmap_visible)
        contours_enabled = heatmap_visible and self.show_contours.isChecked()
        self.contour_count.setEnabled(contours_enabled)
        self.contour_labels.setEnabled(contours_enabled)
        self.logarithmic.setEnabled(heatmap_visible and magnitude)
        self.colour_minimum_enabled.setEnabled(heatmap_visible)
        self.colour_maximum_enabled.setEnabled(heatmap_visible)
        self.colour_minimum.setEnabled(
            heatmap_visible and self.colour_minimum_enabled.isChecked()
        )
        self.colour_maximum.setEnabled(
            heatmap_visible and self.colour_maximum_enabled.isChecked()
        )

    def _static_render_map_settings(self) -> dict[str, Any]:
        """Freeze the current 2D static-map controls for one background render."""

        heatmap_visible = self.display_mode.currentData() in {"heatmap", "both"}
        return {
            "plane": self.plane.currentText(),
            "offset_mm": self.offset.value(),
            "span_mm": self.span.value(),
            "resolution": self.resolution.value(),
            "field": self.field.currentText(),
            "component": self.component.currentData(),
            "logarithmic": self.logarithmic.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "display_mode": self.display_mode.currentData(),
            "fluxline_density": self.flux_density.value(),
            "colour_minimum": (
                self.colour_minimum.value()
                if heatmap_visible and self.colour_minimum_enabled.isChecked()
                else None
            ),
            "colour_maximum": (
                self.colour_maximum.value()
                if heatmap_visible and self.colour_maximum_enabled.isChecked()
                else None
            ),
            "show_contours": heatmap_visible and self.show_contours.isChecked(),
            "contour_count": self.contour_count.value(),
            "contour_labels": self.contour_labels.isChecked(),
            "show_object_outlines": self.object_outlines.isChecked(),
            "outline_object_ids": self.selected_object_outline_ids(),
        }

    def calculate(self) -> None:
        """Queue a fixed-current 2D map without blocking the map window."""

        settings = self._static_render_map_settings()
        self._map_tab_counter += 1
        job_label = f"Map {self._map_tab_counter}"
        tooltip = (
            f"{str(settings['plane']).upper()} — {float(settings['span_mm']):g} mm view — "
            f"{self.component.currentText()}"
        )
        self._start_render_job(
            render_kind="2d_static",
            request={
                "map_settings": settings,
                "active_coil_ids": self.selected_active_coil_ids(),
            },
            job_label=job_label,
            tooltip=tooltip,
        )

    def _plane_overlay(self) -> dict[str, Any]:
        return {
            "kind": "plane",
            "plane": self.plane.currentText(),
            "offset_mm": self.offset.value(),
            "span_mm": self.span.value(),
        }

    def preview_plane(self, *_args) -> None:
        """Debounce selector changes so only the settled plane is rendered."""
        self._selector_preview_timer.start()

    def _preview_scene_options_changed(self, *_args) -> None:
        """Invalidate preview geometry after a local Objects setting changes."""
        self._selector_scene_dirty = True
        self.preview_plane()

    def _preview_object_visibility_requested(
        self, object_id: str, visible: bool
    ) -> None:
        """Route preview-legend visibility changes into the Objects checklist."""
        object_id = str(object_id)
        if object_id not in self._outline_known_object_ids:
            return
        self._set_object_outline_selected(object_id, bool(visible))

    def _preview_scene_base_figure(self) -> dict[str, Any]:
        if self._selector_scene_figure is not None and not self._selector_scene_dirty:
            return self._selector_scene_figure
        choices = self.adapter.field_map_outline_objects()
        enabled = self.object_outlines.isChecked()
        selected = (
            self._outline_selected_object_ids & self._outline_known_object_ids
            if enabled
            else set()
        )
        self._selector_scene_figure = self.adapter.field_map_preview_scene(
            choices if enabled else [],
            selected,
        )
        self._selector_scene_dirty = False
        return self._selector_scene_figure

    def _render_preview_plane(self) -> None:
        """Render the current 2D selector state after its controls become idle."""
        structural_update = self._selector_scene_dirty
        figure = self.adapter.compose_figure(
            self._preview_scene_base_figure(),
            analysis_overlay=self._plane_overlay(),
            include_surface_regions=False,
        )
        if structural_update:
            self.plot.request_clean_rebuild_on_next_figure()
        # Keep latest-wins coalescing as a second guard in case a previous
        # WebEngine/Plotly render is still active when the settled state arrives.
        self.plot.set_figure(figure, coalesce=True)

    def _waveform_map_settings(self) -> dict[str, Any]:
        """Snapshot the current 2D selector/display settings for waveform playback."""
        heatmap_visible = self.display_mode.currentData() in {"heatmap", "both"}
        return {
            "plane": self.plane.currentText(),
            "offset_mm": self.offset.value(),
            "span_mm": self.span.value(),
            "resolution": self.resolution.value(),
            "field": self.field.currentText(),
            "component": self.component.currentData(),
            "logarithmic": self.logarithmic.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "show_contours": heatmap_visible and self.show_contours.isChecked(),
            "contour_count": self.contour_count.value(),
            "contour_labels": self.contour_labels.isChecked(),
            "show_object_outlines": self.object_outlines.isChecked(),
            "outline_object_ids": self.selected_object_outline_ids(),
            "drive_coil_ids": self.selected_active_coil_ids(),
            "colour_minimum": (
                self.colour_minimum.value()
                if heatmap_visible and self.colour_minimum_enabled.isChecked()
                else None
            ),
            "colour_maximum": (
                self.colour_maximum.value()
                if heatmap_visible and self.colour_maximum_enabled.isChecked()
                else None
            ),
        }

    def _open_waveform_action(self, primary_action: str) -> None:
        def launch(initial_camera: dict[str, Any]) -> None:
            camera_timeline = self._camera_timeline_for_action(initial_camera)
            target_fps, playback_speed = self._camera_playback_options_for_action()
            dialog = WaveformPlaybackDialog(
                self.adapter,
                map_kind="2d",
                map_settings=self._waveform_map_settings(),
                camera_timeline=camera_timeline,
                initial_camera=initial_camera,
                playback_fps=target_fps,
                playback_speed=playback_speed,
                primary_action=primary_action,
                defer_payload_build=primary_action == "playback",
                parent=self,
            )
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                if primary_action == "export" or dialog.result_gpu_payload is None:
                    if (
                        primary_action == "playback"
                        and isinstance(dialog.result_gpu_request, dict)
                    ):
                        self._waveform_tab_counter += 1
                        job_label = f"Waveform {self._waveform_tab_counter}"
                        self._start_render_job(
                            render_kind="waveform",
                            request=dialog.result_gpu_request,
                            job_label=job_label,
                            tooltip=dialog.result_label + " — interactive playback",
                        )
                        return
                    self.calculation_status.showMessage("Video export finished")
                    return
                self._waveform_tab_counter += 1
                plot = WaveformGLView(dialog.result_gpu_payload)
                self._connect_result_camera_viewer(plot)
                index = self.tabs.addTab(plot, f"Waveform {self._waveform_tab_counter}")
                self.tabs.setTabToolTip(index, dialog.result_label + " — interactive playback")
                self.tabs.setCurrentIndex(index)
                plot.refresh_viewport()
                self.calculation_status.showMessage("Animated map ready")
            finally:
                dialog.result_figure = None
                dialog.result_gpu_payload = None
                dialog.result_gpu_request = None
                dialog.deleteLater()

        self._request_action_camera(launch)

    def open_waveform_playback(self) -> None:
        self._open_waveform_action("playback")

    def _current_video_export_view(self) -> WaveformGLView | None:
        widget = self.tabs.currentWidget()
        if isinstance(widget, WaveformGLView) and widget.can_export_video():
            return widget
        return None

    def _update_video_export_button(self) -> None:
        self.export_video_button.setVisible(self._current_video_export_view() is not None)

    def _export_current_animation_video(self, *_args) -> None:
        """Run the original video exporter from a frozen rendered result.

        The pre-0.13.0.290 exporter was reliable because its export preview was
        the only WebGL surface involved.  Keep the new rendered-tab workflow,
        but retire that tab's renderer completely before opening the unchanged
        exporter, then recreate the rendered tab from the same frozen payload.
        """
        view = self._current_video_export_view()
        if view is None:
            return
        view.stop_playback()

        def launch(camera: dict[str, Any] | None) -> None:
            try:
                payload = view.export_payload()
                if isinstance(camera, dict):
                    payload["initial_camera"] = copy.deepcopy(camera)
            except Exception as error:  # noqa: BLE001 - rendered-tab payload boundary
                QMessageBox.warning(self, "Unable to export video", str(error))
                return

            tab_index = self.tabs.indexOf(view)
            if tab_index < 0:
                return
            tab_text = self.tabs.tabText(tab_index)
            tab_tooltip = self.tabs.tabToolTip(tab_index)
            placeholder = QLabel("Video export in progress…")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Fully destroy the rendered tab's Chromium/WebGL surface.  The old
            # exporter then runs in exactly the single-renderer situation it had
            # before Export video moved beside Fullscreen viewer.
            self.tabs.removeTab(tab_index)
            self.tabs.insertTab(tab_index, placeholder, tab_text)
            self.tabs.setTabToolTip(tab_index, tab_tooltip)
            self.tabs.setCurrentIndex(tab_index)
            view.cleanup()
            view.deleteLater()

            def run_original_exporter() -> None:
                dialog = WaveformVideoExportDialog(payload, parent=self)
                centre_window_on_parent(dialog, self)
                try:
                    if (
                        dialog.exec() == QDialog.DialogCode.Accepted
                        and dialog.output_path is not None
                    ):
                        self.calculation_status.showMessage(
                            f"Video exported to {dialog.output_path}"
                        )
                finally:
                    dialog.deleteLater()
                    restored = WaveformGLView(
                        payload,
                        initial_camera=(camera if isinstance(camera, dict) else None),
                    )
                    self._connect_result_camera_viewer(restored)
                    current_index = self.tabs.indexOf(placeholder)
                    if current_index >= 0:
                        self.tabs.removeTab(current_index)
                        restored_index = self.tabs.insertTab(
                            current_index, restored, tab_text
                        )
                        self.tabs.setTabToolTip(restored_index, tab_tooltip)
                        self.tabs.setCurrentIndex(restored_index)
                        restored.refresh_viewport()
                    else:
                        restored.cleanup()
                        restored.deleteLater()
                    placeholder.deleteLater()
                    self._update_video_export_button()

            # cleanup() releases QWebEngineView with deleteLater().  Give Qt one
            # event-loop turn to actually retire it before the export preview is
            # constructed.
            QTimer.singleShot(0, run_original_exporter)

        view.request_camera(launch)

    def open_video_export(self) -> None:
        """Compatibility entry point: export only the selected rendered animation."""
        self._export_current_animation_video()

    @Slot(int)
    def _render_tab_changed(self, current_index: int) -> None:
        """Keep only the selected animated WebGL result resident on the GPU."""

        current_widget = self.tabs.widget(current_index)
        if self._camera_view_mode == "2d" and self._is_2d_camera_viewer(current_widget):
            self._connect_result_camera_viewer(current_widget)
            if self.camera_timeline_mode.isChecked():
                QTimer.singleShot(
                    0,
                    lambda current_widget=current_widget: self._capture_2d_view_camera(
                        mark_user=False, viewer=current_widget
                    ),
                )
        current_view = (
            current_widget if isinstance(current_widget, WaveformGLView) else None
        )
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if not isinstance(widget, WaveformGLView):
                continue
            if widget is current_view:
                widget.set_backgrounded(False)
            else:
                widget.set_backgrounded(True, discard=True)
        self._active_render_view = current_view
        self._update_video_export_button()

    def _close_map_tab(self, index: int) -> None:
        """Close generated plots or cancel a calculating tab."""
        if index == self._selector_tab_index:
            return
        widget = self.tabs.widget(index)
        if isinstance(widget, _FieldRenderJobPage):
            for job in getattr(self, "_render_jobs", {}).values():
                if job.get("page") is not widget:
                    continue
                job["discard"] = True
                worker = job.get("worker")
                if isinstance(worker, _FieldRenderWorker):
                    worker.request_cancellation()
                break
        self.tabs.removeTab(index)
        if widget is self._active_render_view:
            self._active_render_view = None
        if isinstance(widget, (PlotView, WaveformGLView)):
            widget.cleanup()
        if widget is not None:
            widget.deleteLater()
        self._render_tab_changed(self.tabs.currentIndex())

    def _current_plot(self):
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, (PlotView, WaveformGLView)) else self.plot

    def _toggle_current_plot_fullscreen(self, *_args) -> None:
        current = self._current_plot()
        toggle = getattr(current, "toggle_fullscreen", None)
        if callable(toggle):
            toggle()

    def _cleanup_plots(self) -> None:
        if getattr(self, "_plots_cleanup_started", False):
            return
        self._plots_cleanup_started = True
        self._cancel_all_render_jobs()
        self._active_render_view = None
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if isinstance(widget, (PlotView, WaveformGLView)):
                widget.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        # Release browser/GPU resources while this top-level window still owns
        # a native surface, and cancel any outstanding isolated calculations.
        self._cleanup_plots()
        super().closeEvent(event)


class FieldVolumeMapDialog(_BackgroundFieldRenderMixin, _CameraTimelineMixin, _PlaybackCoilSelectionMixin, QDialog):
    """Interactive 3D field map with slice and full-volume representations."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        # Keep the 3D workspace independent for its entire lifetime.  In
        # particular, every result tab and animation derives from this private
        # adapter rather than whichever scene is currently active in Workbench.
        self.adapter = adapter.detached_copy()
        self._selector_scene_figure: dict[str, Any] | None = None
        self._selector_scene_dirty = True
        self._map_tab_counter = 0
        self._waveform_tab_counter = 0
        self._render_job_serial = 0
        self._render_jobs: dict[int, dict[str, Any]] = {}
        self._render_document_snapshot: dict[str, Any] | None = None
        self._active_render_view: WaveformGLView | None = None
        self.setWindowTitle("3D Field Map")
        self.resize(1320, 800)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(310)
        control_column = QWidget()
        control_column.setMinimumWidth(310)
        control_column_layout = QVBoxLayout(control_column)
        control_column_layout.setContentsMargins(0, 0, 0, 0)
        control_column_layout.setSpacing(0)
        control_column_layout.addWidget(control_scroll, 1)
        control_panel = QWidget()
        control_panel.setMinimumWidth(0)
        control_panel.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        control_root = QVBoxLayout(control_panel)
        control_root.setContentsMargins(8, 8, 8, 8)
        control_scroll.setWidget(control_panel)
        viewer_panel = QWidget()
        viewer_root = QVBoxLayout(viewer_panel)
        viewer_root.setContentsMargins(0, 0, 0, 0)
        self.map_volume_section = CollapsibleSection("Map volume", expanded=False)
        region_layout = QGridLayout()
        self.coordinates: list[QDoubleSpinBox] = []
        for row, axis in enumerate("XYZ"):
            spin = _spin(0.0, -10000, 10000, 2, " mm")
            self.coordinates.append(spin)
            region_layout.addWidget(QLabel(f"Centre {axis}"), row, 0)
            region_layout.addWidget(spin, row, 1)
        self.span = _spin(300.0, 1, 100000, 1, " mm")
        region_layout.addWidget(QLabel("Cube width"), 3, 0)
        region_layout.addWidget(self.span, 3, 1)
        region_layout.setColumnStretch(1, 1)
        self.map_volume_section.setContentLayout(region_layout)

        self.resolution = QSpinBox()
        self.resolution.setRange(10, 75)
        self.resolution.setValue(35)
        self.field = QComboBox()
        self.field.addItems(["B", "H"])
        self.component = QComboBox()
        self.component.addItem("Magnitude |B|", "magnitude")
        self.component.addItem("X magnitude |Bx|", "abs_x")
        self.component.addItem("Y magnitude |By|", "abs_y")
        self.component.addItem("Z magnitude |Bz|", "abs_z")
        self.component.insertSeparator(self.component.count())
        self.component.addItem("X signed Bx", "x")
        self.component.addItem("Y signed By", "y")
        self.component.addItem("Z signed Bz", "z")
        self.flux_density = QSpinBox()
        self.flux_density.setRange(2, 12)
        self.flux_density.setValue(5)
        self.flux_density.setToolTip(
            "Approximate number of fluxline seed positions per axis throughout the map volume"
        )
        self.sample_config_section = CollapsibleSection("Sample config", expanded=False)
        sample_config_layout = QFormLayout()
        sample_config_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        sample_config_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        sample_config_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        sample_config_layout.addRow("Samples / axis", self.resolution)
        sample_config_layout.addRow("Field", self.field)
        sample_config_layout.addRow("Show", self.component)
        sample_config_layout.addRow("Flux density", self.flux_density)
        self.sample_config_section.setContentLayout(sample_config_layout)

        view_row = QWidget()
        view_row_layout = QFormLayout(view_row)
        view_row_layout.setContentsMargins(0, 2, 0, 2)
        view_row_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        view_row_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.view_type = QComboBox()
        self.view_type.addItem("Slices", "slices")
        self.view_type.addItem("Full Volume", "volume")
        self.view_type.setCurrentIndex(self.view_type.findData("volume"))
        self.view_type.setToolTip(
            "Slices draws selected planes through the map cube. Full Volume samples the complete X×Y×Z grid."
        )
        view_row_layout.addRow("Field view", self.view_type)

        self.slice_axis = QComboBox()
        self.slice_axis.addItem("Z axis — XY planes", "z")
        self.slice_axis.addItem("Y axis — XZ planes", "y")
        self.slice_axis.addItem("X axis — YZ planes", "x")
        self.slice_count = QSpinBox()
        self.slice_count.setRange(1, 25)
        self.slice_count.setSingleStep(2)
        self.slice_count.setValue(9)
        self.slice_count.setToolTip(
            "An odd value includes the exact central plane; use the plot slider to reveal layers."
        )
        slice_options_layout = QFormLayout()
        slice_options_layout.setContentsMargins(0, 0, 0, 4)
        slice_options_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        slice_options_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        slice_options_layout.addRow("Slice direction", self.slice_axis)
        slice_options_layout.addRow("Slices", self.slice_count)
        self.slice_options_section = CollapsibleSection("Slice options", expanded=True)
        self.slice_options_section.setContentLayout(slice_options_layout)
        # Keep the old attribute as a compatibility alias for callers/tests that
        # accessed it before the section was narrowed to slice-only controls.
        self.field_view_options_section = self.slice_options_section

        self.include_scene = QCheckBox("Show scene geometry")
        self.include_scene.setChecked(True)
        self.include_scene.setToolTip(
            "Include coils, magnets, sensors, and passive geometry for spatial context"
        )
        self._scene_selected_object_ids: set[str] = set()
        self._scene_known_object_ids: set[str] = set()
        self._scene_object_opacity_overrides: dict[str, float] = {}
        self._scene_selection_initialized = False
        self.scene_objects_menu = QMenu(self)
        self.scene_objects_menu.aboutToShow.connect(self._refresh_scene_objects_menu)
        self.scene_objects_button = QPushButton("Objects…")
        self.scene_objects_button.setMenu(self.scene_objects_menu)
        self.scene_objects_button.setToolTip(
            "Choose which scene objects are drawn in the Selector preview and for spatial "
            "context in this map, and adjust their viewer-only opacity. Clicking an object in "
            "the preview legend updates this checklist. A new map window starts from the "
            "scene's current visibility; field sources are controlled separately by Active coils."
        )
        self.include_scene.toggled.connect(self.scene_objects_button.setEnabled)
        self._refresh_scene_objects_menu()
        self._init_playback_coil_selector()
        scene_options_layout = QVBoxLayout()
        scene_options_layout.setContentsMargins(0, 0, 0, 0)
        scene_options_layout.setSpacing(3)
        scene_options_layout.addWidget(self.include_scene)
        scene_options_layout.addWidget(self.scene_objects_button)
        active_coils_label = QLabel("Active coils")
        active_coils_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        scene_options_layout.addWidget(active_coils_label)
        scene_options_layout.addWidget(self.playback_coils_button)
        self.scene_options_section = CollapsibleSection("Scene options", expanded=False)
        self.scene_options_section.setContentLayout(scene_options_layout)

        self.display_mode = QComboBox()
        self.display_mode.addItem("Field", "slices")
        self.display_mode.addItem("Fluxlines", "flux")
        self.display_mode.addItem("Field + fluxlines", "both")
        self.display_mode.setCurrentIndex(2)
        self.logarithmic = QCheckBox("Log scale")
        self.logarithmic.setToolTip("Use a log₁₀ display scale for |B| or an absolute axis-component magnitude")
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(True)
        self.show_grid.setToolTip(
            "Show the shared 3D axes, grid planes, tick labels, and map-volume frame "
            "in the Selector preview, rendered maps, waveform playback, and exported video."
        )
        self.intensity_opacity_enabled = QCheckBox("Intensity-linked opacity")
        self.intensity_opacity_enabled.setChecked(True)
        self.intensity_opacity_enabled.setToolTip(
            "Map absolute field strength onto opacity as well as colour. Signed components fade near zero and become more opaque in either direction."
        )
        self.volume_opacity_lower = _spin(15.0, 0.0, 100.0, 1, " %")
        self.volume_opacity_upper = _spin(40.0, 0.0, 100.0, 1, " %")
        self.volume_opacity_lower_level = _spin(0.0, 0.0, 100.0, 1, " %")
        self.volume_opacity_upper_level = _spin(100.0, 0.0, 100.0, 1, " %")
        self.volume_opacity_lower.setToolTip(
            "Opacity used at and below the lower intensity level. For signed components intensity means absolute field strength."
        )
        self.volume_opacity_upper.setToolTip(
            "Opacity used at and above the upper intensity level. Signed positive and negative extrema use the same strength opacity."
        )
        self.volume_opacity_lower_level.setToolTip(
            "Field intensity, as a percentage of the displayed scale, where the lower opacity endpoint is reached. Values below this stay at the lower opacity."
        )
        self.volume_opacity_upper_level.setToolTip(
            "Field intensity, as a percentage of the displayed scale, where the upper opacity endpoint is reached. Values above this stay at the upper opacity."
        )
        self.colour_minimum_enabled = QCheckBox("Lower limit")
        self.colour_minimum_enabled.setMinimumWidth(
            self.colour_minimum_enabled.sizeHint().width() + 8
        )
        self.colour_minimum = _spin(0.0, -1e12, 1e12, 3, " µT")
        self.colour_minimum.setEnabled(False)
        self.colour_maximum_enabled = QCheckBox("Upper limit")
        self.colour_maximum_enabled.setMinimumWidth(
            self.colour_maximum_enabled.sizeHint().width() + 8
        )
        self.colour_maximum = _spin(1000.0, -1e12, 1e12, 3, " µT")
        self.colour_maximum.setEnabled(False)
        display_options_layout = QGridLayout()
        display_options_layout.setColumnStretch(1, 1)
        display_options_layout.addWidget(QLabel("Display"), 0, 0)
        display_options_layout.addWidget(self.display_mode, 0, 1, 1, 3)
        display_options_layout.addWidget(self.logarithmic, 1, 0, 1, 4)
        display_options_layout.addWidget(self.show_grid, 2, 0, 1, 4)
        display_options_layout.addWidget(self.intensity_opacity_enabled, 3, 0, 1, 4)
        self.volume_opacity_lower_label = QLabel("Lower opacity")
        self.volume_opacity_upper_label = QLabel("Upper opacity")
        self.volume_opacity_lower_level_label = QLabel("at level")
        self.volume_opacity_upper_level_label = QLabel("at level")
        display_options_layout.addWidget(self.volume_opacity_lower_label, 4, 0)
        display_options_layout.addWidget(self.volume_opacity_lower, 4, 1)
        display_options_layout.addWidget(self.volume_opacity_lower_level_label, 4, 2)
        display_options_layout.addWidget(self.volume_opacity_lower_level, 4, 3)
        display_options_layout.addWidget(self.volume_opacity_upper_label, 5, 0)
        display_options_layout.addWidget(self.volume_opacity_upper, 5, 1)
        display_options_layout.addWidget(self.volume_opacity_upper_level_label, 5, 2)
        display_options_layout.addWidget(self.volume_opacity_upper_level, 5, 3)
        opacity_note = QLabel(
            "Opacity is clamped outside the two intensity endpoints and interpolated between them. Levels are percentages of the displayed field scale; signed components use absolute strength."
        )
        opacity_note.setWordWrap(True)
        opacity_note.setObjectName("hint")
        display_options_layout.addWidget(opacity_note, 6, 0, 1, 4)
        colour_scale_title = QLabel("Colour scale")
        colour_scale_title.setStyleSheet("font-weight: 600;")
        display_options_layout.addWidget(colour_scale_title, 7, 0, 1, 4)
        self.colour_limits_hint = QLabel("Leave limits unchecked for automatic range")
        self.colour_limits_hint.setWordWrap(True)
        self.colour_limits_hint.setObjectName("hint")
        display_options_layout.addWidget(self.colour_limits_hint, 8, 0, 1, 4)
        display_options_layout.addWidget(self.colour_minimum_enabled, 9, 0)
        display_options_layout.addWidget(self.colour_minimum, 9, 1, 1, 3)
        display_options_layout.addWidget(self.colour_maximum_enabled, 10, 0)
        display_options_layout.addWidget(self.colour_maximum, 10, 1, 1, 3)
        self.display_options_section = CollapsibleSection("Display Options", expanded=True)
        self.display_options_section.setToolTip(
            "Field visibility, logarithmic colour mapping, intensity-linked opacity, and the scene-wide colour scale."
        )
        self.display_options_section.setContentLayout(display_options_layout)

        for widget in (
            *self.coordinates,
            self.span,
            self.view_type,
            self.slice_axis,
            self.slice_count,
            self.volume_opacity_lower,
            self.volume_opacity_upper,
            self.volume_opacity_lower_level,
            self.volume_opacity_upper_level,
            self.resolution,
            self.field,
            self.component,
            self.display_mode,
            self.flux_density,
            self.scene_objects_button,
            self.colour_minimum,
            self.colour_maximum,
        ):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy()
            )

        self._init_camera_timeline_controls(view_mode="3d")
        self.calculate_button = QPushButton("Static map")
        self.calculate_button.setObjectName("primaryButton")
        self.calculate_button.clicked.connect(self.calculate)
        self.waveform_button = QPushButton("Animated map")
        self.waveform_button.setToolTip(
            "Create quasi-static current/voltage playback using the selected field view and display options"
        )
        self.waveform_button.clicked.connect(self.open_waveform_playback)
        self.reset_defaults_button = QPushButton("Reset defaults")
        self.reset_defaults_button.setToolTip(
            "Restore the built-in 3D map settings and replace the remembered settings for this window"
        )
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        self.exit_button = QPushButton("Exit")
        self.exit_button.setToolTip("Close this 3D Field Map workspace")
        self.exit_button.clicked.connect(self.close)

        control_root.addWidget(self.map_volume_section)
        control_root.addWidget(self.sample_config_section)
        control_root.addWidget(view_row)
        control_root.addWidget(self.field_view_options_section)
        control_root.addWidget(self.scene_options_section)
        control_root.addWidget(self.display_options_section)
        control_root.addWidget(self.camera_section)
        control_root.addStretch(1)

        # Keep primary actions visible while the setup sections scroll above them.
        action_panel = QWidget()
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(8, 6, 8, 8)
        action_layout.setSpacing(6)
        # Keep an active timeline-step editor pinned with the primary actions.
        # The table itself remains in the scrolling Camera section, but Step
        # time / playback speed / Accept / Cancel cannot be scrolled away.
        action_layout.addWidget(self.camera_timeline_edit_widget)
        action_layout.addWidget(self.calculate_button)
        action_layout.addWidget(self.waveform_button)
        action_layout.addWidget(self.reset_defaults_button)
        action_layout.addWidget(self.exit_button)
        control_column_layout.addWidget(action_panel, 0)

        # Treat the mouse wheel as navigation throughout the setup pane.
        # Spin boxes and closed combo boxes otherwise consume wheel events and
        # silently change parameters while the user is only trying to scroll.
        self._control_wheel_filter = EditorWheelScrollFilter(control_scroll)
        for editor_type in (QDoubleSpinBox, QSpinBox, QComboBox):
            for editor in control_panel.findChildren(editor_type):
                editor.installEventFilter(self._control_wheel_filter)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_map_tab)
        self.plot = PlotView(
            include_3d_return_axes=True,
            synchronize_legend_visibility=True,
            image_filename="3d-field-map-selector",
        )
        self.plot.visibility_requested.connect(
            self._preview_object_visibility_requested
        )
        self.plot.live_3d_view_changed.connect(self._selector_camera_changed)
        self._selector_tab_index = self.tabs.addTab(self.plot, "Selector")
        self.tabs.currentChanged.connect(self._render_tab_changed)
        tab_bar = self.tabs.tabBar()
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.LeftSide, None
        )
        tab_bar.setTabButton(
            self._selector_tab_index, QTabBar.ButtonPosition.RightSide, None
        )
        viewer_root.addWidget(self.tabs, 1)

        viewer_actions = QWidget()
        viewer_actions_layout = QHBoxLayout(viewer_actions)
        viewer_actions_layout.setContentsMargins(8, 0, 8, 6)
        viewer_actions_layout.addStretch(1)
        self.export_video_button = QPushButton("Export video")
        self.export_video_button.setToolTip(
            "Export the selected rendered animation without recalculating its field"
        )
        self.export_video_button.clicked.connect(self._export_current_animation_video)
        self.export_video_button.setVisible(False)
        viewer_actions_layout.addWidget(self.export_video_button)
        self.fullscreen_button = QPushButton("Fullscreen viewer")
        self.fullscreen_button.setToolTip(
            "Open a separate true-fullscreen interactive 3D field-map viewer (Esc/F11 closes it)"
        )
        self.fullscreen_button.clicked.connect(self._toggle_current_plot_fullscreen)
        viewer_actions_layout.addWidget(self.fullscreen_button)
        viewer_root.addWidget(viewer_actions)
        self.calculation_status = QStatusBar()
        self.calculation_status.setObjectName("fieldVolumeMapStatusBar")
        self.calculation_status.setSizeGripEnabled(False)
        self.calculation_status.showMessage("Ready")
        viewer_root.addWidget(self.calculation_status)
        splitter.addWidget(control_column)
        splitter.addWidget(viewer_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 960])
        root.addWidget(splitter, 1)
        self.finished.connect(lambda _code: self._cleanup_plots())
        self.component.currentIndexChanged.connect(self._update_log_availability)
        self.colour_minimum_enabled.toggled.connect(
            self._update_colour_limit_availability
        )
        self.colour_maximum_enabled.toggled.connect(
            self._update_colour_limit_availability
        )
        self.display_mode.currentIndexChanged.connect(
            self._update_colour_limit_availability
        )
        self.intensity_opacity_enabled.toggled.connect(
            self._update_colour_limit_availability
        )
        self.volume_opacity_lower_level.valueChanged.connect(
            self._lower_opacity_level_changed
        )
        self.volume_opacity_upper_level.valueChanged.connect(
            self._upper_opacity_level_changed
        )
        self.view_type.currentIndexChanged.connect(self._update_view_type_controls)
        self.view_type.currentIndexChanged.connect(self.preview_slice_volume)
        self.field.currentTextChanged.connect(self._update_colour_limit_units)
        self.field.currentTextChanged.connect(
            lambda value: _update_field_component_combo_labels(self.component, value)
        )
        self._selector_preview_timer = QTimer(self)
        self._selector_preview_timer.setSingleShot(True)
        self._selector_preview_timer.setInterval(SELECTOR_PREVIEW_DEBOUNCE_MS)
        self._selector_preview_timer.timeout.connect(self._render_preview_slice_volume)
        self.include_scene.toggled.connect(self._preview_scene_options_changed)
        for spin in self.coordinates:
            spin.valueChanged.connect(self.preview_slice_volume)
        self.span.valueChanged.connect(self.preview_slice_volume)
        self.slice_axis.currentIndexChanged.connect(self.preview_slice_volume)
        self.slice_count.valueChanged.connect(self.preview_slice_volume)
        self.show_grid.toggled.connect(self.preview_slice_volume)
        self._update_view_type_controls()
        self._update_log_availability()
        self._update_colour_limit_units()
        _update_field_component_combo_labels(self.component, self.field.currentText())
        self._update_colour_limit_availability()
        self._render_preview_slice_volume()
        self.calculation_status.showMessage(
            "Ready — position the map volume in Selector, then click Static map"
        )
        capture_window_default_settings(self)

    def _reset_defaults(self, *_args) -> None:
        if not reset_window_to_default_settings(self):
            return
        self._fixed_camera_user_edited = False
        self._update_view_type_controls()
        self._update_log_availability()
        self._update_colour_limit_units()
        _update_field_component_combo_labels(self.component, self.field.currentText())
        self._update_colour_limit_availability()
        self._render_preview_slice_volume()
        self.calculation_status.showMessage("3D map settings reset to defaults")

    def _persistent_ui_state(self) -> dict[str, Any]:
        # Scene/coil checklists remain per-window-session; the camera timeline is
        # retained with this viewer's ordinary map settings.
        return self._camera_persistent_state()

    def _restore_persistent_ui_state(self, state: dict[str, Any]) -> None:
        self._restore_camera_persistent_state(state)

    def _refresh_scene_objects_menu(self) -> None:
        choices = self.adapter.field_volume_map_scene_objects()
        current_ids = {str(choice.get("id", "")) for choice in choices if choice.get("id")}
        if not self._scene_selection_initialized:
            self._scene_selected_object_ids = {
                str(choice["id"])
                for choice in choices
                if choice.get("id") and bool(choice.get("visible", True))
            }
            self._scene_selection_initialized = True
        else:
            # Keep the map window's geometry choices stable after opening.
            self._scene_selected_object_ids.intersection_update(current_ids)
        self._scene_known_object_ids = current_ids
        # Opacity differs from visibility: untouched entries continue to inherit the
        # live scene value for as long as this viewer remains open. Only ids the user
        # edits are retained as viewer-local overrides.
        self._scene_object_opacity_overrides = {
            object_id: float(value)
            for object_id, value in self._scene_object_opacity_overrides.items()
            if object_id in current_ids
        }
        self.scene_objects_menu.clear()
        select_all = self.scene_objects_menu.addAction("Select all")
        select_all.triggered.connect(self._select_all_scene_objects)
        clear_all = self.scene_objects_menu.addAction("Clear all")
        clear_all.triggered.connect(self._clear_all_scene_objects)
        self.scene_objects_menu.addSeparator()
        if not choices:
            empty = self.scene_objects_menu.addAction("No scene objects")
            empty.setEnabled(False)
        else:
            header = QWidget(self.scene_objects_menu)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(8, 0, 8, 0)
            object_heading = QLabel("Object")
            object_heading.setStyleSheet("font-weight: 600;")
            opacity_heading = QLabel("Opacity")
            opacity_heading.setStyleSheet("font-weight: 600;")
            opacity_heading.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(object_heading, 1)
            header_layout.addWidget(opacity_heading)
            header_action = QWidgetAction(self.scene_objects_menu)
            header_action.setDefaultWidget(header)
            self.scene_objects_menu.addAction(header_action)
        for choice in choices:
            object_id = str(choice["id"])
            label = str(choice.get("label") or object_id)
            if not bool(choice.get("visible", True)):
                label += " (hidden)"

            row = QWidget(self.scene_objects_menu)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 1, 8, 1)
            row_layout.setSpacing(8)
            selected = QCheckBox(label)
            selected.setChecked(object_id in self._scene_selected_object_ids)
            selected.setToolTip("Show or hide this object in this 3D viewer only.")
            selected.toggled.connect(
                lambda checked, selected_id=object_id: self._set_scene_object_selected(selected_id, checked)
            )
            opacity = QSpinBox()
            opacity.setRange(0, 100)
            opacity.setSuffix(" %")
            inherited_opacity = max(0.0, min(1.0, float(choice.get("opacity", 1.0))))
            opacity_value = self._scene_object_opacity_overrides.get(object_id, inherited_opacity)
            opacity.setValue(round(opacity_value * 100.0))
            opacity.setKeyboardTracking(False)
            opacity.setFixedWidth(78)
            opacity.setToolTip(
                "Starts from this object's current scene opacity. Changing it creates a "
                "viewer-only override and does not modify the source scene."
            )
            opacity.valueChanged.connect(
                lambda value, selected_id=object_id: self._set_scene_object_opacity(selected_id, value / 100.0)
            )
            row_layout.addWidget(selected, 1)
            row_layout.addWidget(opacity)
            row_action = QWidgetAction(self.scene_objects_menu)
            row_action.setDefaultWidget(row)
            self.scene_objects_menu.addAction(row_action)
        self._update_scene_objects_button_text(len(current_ids))

    def _update_scene_objects_button_text(self, total: int | None = None) -> None:
        total = len(self._scene_known_object_ids) if total is None else total
        selected = len(self._scene_selected_object_ids & self._scene_known_object_ids)
        self.scene_objects_button.setText(f"Objects {selected}/{total}…" if total else "Objects 0…")

    def _set_scene_object_selected(self, object_id: str, checked: bool) -> None:
        if checked:
            self._scene_selected_object_ids.add(str(object_id))
        else:
            self._scene_selected_object_ids.discard(str(object_id))
        self._update_scene_objects_button_text()
        self._preview_scene_options_changed()

    def _set_scene_object_opacity(self, object_id: str, opacity: float) -> None:
        value = max(0.0, min(1.0, float(opacity)))
        self._scene_object_opacity_overrides[str(object_id)] = value
        self._preview_scene_options_changed()

    def scene_object_opacity_overrides(self) -> dict[str, float]:
        self._refresh_scene_objects_menu()
        return {
            object_id: float(value)
            for object_id, value in sorted(self._scene_object_opacity_overrides.items())
            if object_id in self._scene_known_object_ids
        }

    def scene_object_opacity_values(self) -> dict[str, float]:
        """Return the values currently shown in the Objects opacity column."""
        choices = {
            str(choice.get("id", "")): max(0.0, min(1.0, float(choice.get("opacity", 1.0))))
            for choice in self.adapter.field_volume_map_scene_objects()
            if choice.get("id")
        }
        return {
            object_id: float(self._scene_object_opacity_overrides.get(object_id, inherited))
            for object_id, inherited in sorted(choices.items())
        }

    def _select_all_scene_objects(self, *_args) -> None:
        self._scene_selected_object_ids = set(self._scene_known_object_ids)
        self._refresh_scene_objects_menu()
        self._preview_scene_options_changed()

    def _clear_all_scene_objects(self, *_args) -> None:
        self._scene_selected_object_ids.clear()
        self._refresh_scene_objects_menu()
        self._preview_scene_options_changed()

    def selected_scene_object_ids(self) -> list[str]:
        self._refresh_scene_objects_menu()
        return sorted(self._scene_selected_object_ids & self._scene_known_object_ids)

    def _update_view_type_controls(self, *_args) -> None:
        """Show slice-only controls only when the Slices field view is active."""
        volume = self.view_type.currentData() == "volume"
        self.slice_options_section.setVisible(not volume)
        self.resolution.setToolTip(
            "Samples along each X/Y/Z axis of the complete volume grid (N³ field points)."
            if volume
            else "Samples along each in-plane axis of every field slice."
        )
        self._update_colour_limit_availability()

    def _update_colour_limit_units(self, *_args) -> None:
        magnetic_flux_density = self.field.currentText() == "B"
        suffix = " µT" if magnetic_flux_density else " A/m"
        step = 10.0 if magnetic_flux_density else 1.0
        for spin in (self.colour_minimum, self.colour_maximum):
            spin.setSuffix(suffix)
            spin.setDecimals(3)
            spin.setSingleStep(step)
        self.colour_minimum.setValue(0.0)
        self.colour_maximum.setValue(1000.0)

    def _lower_opacity_level_changed(self, value: float) -> None:
        if float(value) > self.volume_opacity_upper_level.value():
            self.volume_opacity_upper_level.setValue(float(value))

    def _upper_opacity_level_changed(self, value: float) -> None:
        if float(value) < self.volume_opacity_lower_level.value():
            self.volume_opacity_lower_level.setValue(float(value))

    def _update_colour_limit_availability(self, *_args) -> None:
        field_visible = self.display_mode.currentData() in {"slices", "both"}
        self.intensity_opacity_enabled.setEnabled(field_visible)
        opacity_editable = field_visible and self.intensity_opacity_enabled.isChecked()
        self.volume_opacity_lower_label.setEnabled(opacity_editable)
        self.volume_opacity_upper_label.setEnabled(opacity_editable)
        self.volume_opacity_lower_level_label.setEnabled(opacity_editable)
        self.volume_opacity_upper_level_label.setEnabled(opacity_editable)
        self.volume_opacity_lower.setEnabled(opacity_editable)
        self.volume_opacity_upper.setEnabled(opacity_editable)
        self.volume_opacity_lower_level.setEnabled(opacity_editable)
        self.volume_opacity_upper_level.setEnabled(opacity_editable)
        self.colour_minimum_enabled.setEnabled(field_visible)
        self.colour_maximum_enabled.setEnabled(field_visible)
        self.colour_minimum.setEnabled(
            field_visible and self.colour_minimum_enabled.isChecked()
        )
        self.colour_maximum.setEnabled(
            field_visible and self.colour_maximum_enabled.isChecked()
        )
        self._update_log_availability()

    def _update_log_availability(self, *_args) -> None:
        field_visible = self.display_mode.currentData() in {"slices", "both"}
        magnitude = field_component_is_magnitude(self.component.currentData())
        signed = str(self.component.currentData()) in {"x", "y", "z"}
        self.colour_limits_hint.setText(
            "Signed components use a zero-centred ± range; either checked limit sets the mirrored extent."
            if signed
            else "Leave limits unchecked for automatic range"
        )
        if not magnitude:
            self.logarithmic.setChecked(False)
        self.logarithmic.setEnabled(field_visible and magnitude)

    def calculate(self) -> None:
        """Create a fixed-current 3D map with the shared WebGL renderer."""
        if not self.calculate_button.isEnabled():
            return
        self._request_action_camera(self._calculate_static_webgl)

    def _calculate_static_webgl(self, initial_camera: dict[str, Any]) -> None:
        settings = self._waveform_map_settings()
        self._map_tab_counter += 1
        job_label = f"Map {self._map_tab_counter}"
        if str(settings.get("view_type")) == "volume":
            tooltip = (
                f"Full Volume — {int(settings['resolution'])}³ samples — "
                f"{float(settings['span_mm']):g} mm cube — WebGL"
            )
        else:
            tooltip = (
                f"{str(settings.get('slice_axis', 'z')).upper()} slices — "
                f"{int(settings['slice_count'])} layers — "
                f"{float(settings['span_mm']):g} mm cube — WebGL"
            )
        self._start_render_job(
            render_kind="static",
            request={
                "map_settings": settings,
                "active_coil_ids": self.selected_active_coil_ids(),
                "initial_camera": copy.deepcopy(initial_camera),
            },
            job_label=job_label,
            tooltip=tooltip,
        )

    def _slice_volume_overlay(self) -> dict[str, Any]:
        base = {
            "centre_mm": [spin.value() for spin in self.coordinates],
            "span_mm": self.span.value(),
        }
        if self.view_type.currentData() == "volume":
            return {"kind": "volume_box", **base}
        return {
            "kind": "slice_stack",
            **base,
            "slice_axis": self.slice_axis.currentData(),
            "slice_count": self.slice_count.value(),
        }

    def preview_slice_volume(self, *_args) -> None:
        """Debounce selector changes so only the settled slice volume is rendered."""
        self._selector_preview_timer.start()

    def _preview_scene_options_changed(self, *_args) -> None:
        """Invalidate preview geometry after a local Scene Objects setting changes."""
        self._selector_scene_dirty = True
        self.preview_slice_volume()

    def _preview_object_visibility_requested(
        self, object_id: str, visible: bool
    ) -> None:
        """Route preview-legend visibility changes into the Objects checklist."""
        object_id = str(object_id)
        if object_id not in self._scene_known_object_ids:
            return
        self._set_scene_object_selected(object_id, bool(visible))

    def _preview_scene_base_figure(self) -> dict[str, Any]:
        if self._selector_scene_figure is not None and not self._selector_scene_dirty:
            return self._selector_scene_figure
        choices = self.adapter.field_volume_map_scene_objects()
        enabled = self.include_scene.isChecked()
        selected = (
            self._scene_selected_object_ids & self._scene_known_object_ids
            if enabled
            else set()
        )
        self._selector_scene_figure = self.adapter.field_map_preview_scene(
            choices if enabled else [],
            selected,
            object_opacities=(
                self.scene_object_opacity_values() if enabled else None
            ),
        )
        self._selector_scene_dirty = False
        return self._selector_scene_figure

    @staticmethod
    def _apply_selector_grid_visibility(
        figure: dict[str, Any], *, show_grid: bool
    ) -> dict[str, Any]:
        """Make the Selector obey the same scene-grid visibility choice as output."""
        if show_grid:
            return figure
        # ``compose_figure`` deliberately keeps the cached base layout by reference,
        # so copy only the lightweight layout before applying a viewer-local axis
        # override. Otherwise switching Show grid off once would also mutate the
        # cached scene used when it is turned back on. Keep the potentially large
        # scene geometry/data list shared.
        updated = dict(figure)
        updated["layout"] = copy.deepcopy(figure.get("layout", {}))
        scene = updated["layout"].setdefault("scene", {})
        hidden_axis = {
            "title": {"text": "", "font": {"color": "rgba(0,0,0,0)"}},
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "showbackground": False,
            "showline": False,
            "showticklabels": False,
            "ticks": "",
            "showspikes": False,
            "backgroundcolor": "rgba(0,0,0,0)",
            "gridcolor": "rgba(0,0,0,0)",
            "zerolinecolor": "rgba(0,0,0,0)",
            "showaxeslabels": False,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"color": "rgba(0,0,0,0)"},
        }
        scene.update(
            {
                "xaxis": copy.deepcopy(hidden_axis),
                "yaxis": copy.deepcopy(hidden_axis),
                "zaxis": copy.deepcopy(hidden_axis),
                "annotations": [],
            }
        )
        return updated

    def _render_preview_slice_volume(self) -> None:
        """Render the current 3D selector state after its controls become idle."""
        structural_update = self._selector_scene_dirty
        show_grid = self.show_grid.isChecked()
        overlay = self._slice_volume_overlay()
        if not show_grid:
            # Keep the derived map-volume boundary visible even when the scene
            # grid is hidden, so the Selector still shows what spatial region
            # will be rendered. Only the outer volume box remains; slice-stack
            # guides stay hidden with the grid.
            overlay = {
                "kind": "volume_box",
                "centre_mm": overlay.get("centre_mm", [spin.value() for spin in self.coordinates]),
                "span_mm": overlay.get("span_mm", self.span.value()),
            }
        figure = self.adapter.compose_figure(
            self._preview_scene_base_figure(),
            analysis_overlay=overlay,
            include_surface_regions=False,
        )
        figure = self._apply_selector_grid_visibility(figure, show_grid=show_grid)
        if structural_update:
            self.plot.request_clean_rebuild_on_next_figure()
        # Retain latest-wins coalescing in case an older WebEngine/Plotly update
        # is still active when this final debounced selector state is submitted.
        self.plot.set_figure(figure, coalesce=True)

    def _add_webgl_map_tab(self, payload: dict[str, Any]) -> None:
        """Add one fixed-current WebGL result tab and make it current."""
        self._map_tab_counter += 1
        plot = WaveformGLView(payload)
        index = self.tabs.addTab(plot, f"Map {self._map_tab_counter}")
        if self.view_type.currentData() == "volume":
            tooltip = (
                f"Full Volume — {self.resolution.value()}³ samples — "
                f"{self.span.value():g} mm cube — WebGL"
            )
        else:
            tooltip = (
                f"{self.slice_axis.currentText()} — {self.slice_count.value()} slices — "
                f"{self.span.value():g} mm cube — WebGL"
            )
        self.tabs.setTabToolTip(index, tooltip)
        self.tabs.setCurrentIndex(index)
        plot.refresh_viewport()

    def _waveform_map_settings(self) -> dict[str, Any]:
        """Snapshot the current 3D selector/display settings for waveform playback."""
        field_visible = self.display_mode.currentData() in {"slices", "both"}
        return {
            "centre_mm": [spin.value() for spin in self.coordinates],
            "span_mm": self.span.value(),
            "slice_axis": self.slice_axis.currentData(),
            "slice_count": self.slice_count.value(),
            "view_type": self.view_type.currentData(),
            "waveform_render_mode": self.view_type.currentData(),
            "intensity_opacity_enabled": self.intensity_opacity_enabled.isChecked(),
            "volume_opacity_lower": self.volume_opacity_lower.value() / 100.0,
            "volume_opacity_upper": self.volume_opacity_upper.value() / 100.0,
            "volume_opacity_lower_level": self.volume_opacity_lower_level.value() / 100.0,
            "volume_opacity_upper_level": self.volume_opacity_upper_level.value() / 100.0,
            "resolution": self.resolution.value(),
            "field": self.field.currentText(),
            "component": self.component.currentData(),
            "logarithmic": self.logarithmic.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "include_scene": self.include_scene.isChecked(),
            "scene_object_ids": self.selected_scene_object_ids(),
            # Snapshot the exact opacity values currently represented by Scene
            # options so GPU waveform geometry matches the static 3D viewer,
            # including both inherited scene opacity and local viewer overrides.
            "scene_object_opacities": self.scene_object_opacity_values(),
            "drive_coil_ids": self.selected_active_coil_ids(),
            "display_mode": self.display_mode.currentData(),
            "fluxline_density": self.flux_density.value(),
            "colour_minimum": (
                self.colour_minimum.value()
                if field_visible and self.colour_minimum_enabled.isChecked()
                else None
            ),
            "colour_maximum": (
                self.colour_maximum.value()
                if field_visible and self.colour_maximum_enabled.isChecked()
                else None
            ),
        }

    @staticmethod
    def _animation_validation_colour_lines(
        *,
        component: str,
        colour_minimum: float | None,
        colour_maximum: float | None,
    ) -> list[str]:
        signed = str(component).strip().lower() in {"x", "y", "z"}
        scale = FIELD_SIGNED_COLOURSCALE if signed else FIELD_HEAT_COLOURSCALE
        lines = [
            "COLOUR SCALE SANITY",
            f"Shared static/GPU colour scale: {'signed blue-white-red' if signed else 'magnitude blue-to-red'}",
        ]
        if colour_minimum is None or colour_maximum is None or not colour_maximum > colour_minimum:
            lines.append(
                "Current colour range is not fully fixed by manual limits; normalized stop colours are:"
            )
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                rgb = _field_colour_at_fraction(scale, fraction)
                lines.append(
                    f"  {fraction:5.1%} -> rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
                )
            return lines
        low = float(colour_minimum)
        high = float(colour_maximum)
        lines.append(f"Current fixed colour range: {low:.9g} to {high:.9g}")
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = low + fraction * (high - low)
            rgb = _field_colour_at_fraction(scale, fraction)
            lines.append(
                f"  {value:.9g} -> {fraction:5.1%} -> rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
            )
        lines.append(
            "This confirms the shared normalization/colour table only; it does not read pixels back from the WebGL shader."
        )
        return lines

    def _show_animation_validation_report(self, report: dict[str, Any]) -> None:
        unit = str(report.get("unit", ""))
        state_lines: list[str] = []
        labels_by_id = {
            str(item["id"]): str(item.get("label") or item["id"])
            for item in drive_coil_catalog(self.adapter)
        }
        drive_ids = [str(value) for value in report.get("drive_coil_ids", [])]
        coil_text = ", ".join(labels_by_id.get(value, value) for value in drive_ids)
        for state in report.get("states", []):
            currents_ma = ", ".join(
                f"{1000.0 * float(value):.6g}" for value in state.get("currents_a", [])
            )
            state_lines.extend(
                [
                    f"{state.get('label', 'State')}: currents [{currents_ma}] mA",
                    f"  peak field              {float(state.get('peak_field', 0.0)):.9g} {unit}",
                    f"  max vector error        {float(state.get('max_vector_error', 0.0)):.9g} {unit}",
                    f"  RMS vector error        {float(state.get('rms_vector_error', 0.0)):.9g} {unit}",
                    f"  max magnitude error     {float(state.get('max_magnitude_error', 0.0)):.9g} {unit}",
                    f"  RMS magnitude error     {float(state.get('rms_magnitude_error', 0.0)):.9g} {unit}",
                    f"  max relative error      {float(state.get('max_relative_error_pct', 0.0)):.9g} %",
                    f"  RMS relative error      {float(state.get('rms_relative_error_pct', 0.0)):.9g} %",
                    "",
                ]
            )

        manual_min = (
            self.colour_minimum.value() if self.colour_minimum_enabled.isChecked() else None
        )
        manual_max = (
            self.colour_maximum.value() if self.colour_maximum_enabled.isChecked() else None
        )
        colour_lines = self._animation_validation_colour_lines(
            component=str(self.component.currentData()),
            colour_minimum=manual_min,
            colour_maximum=manual_max,
        )
        verdict = "PASS" if bool(report.get("passed")) else "CHECK FAILED"
        lines = [
            "FIELD WORKBENCH — ANIMATION FIELD VALIDATION",
            "",
            f"RESULT: {verdict}",
            f"Map lattice: {int(report.get('resolution', 0))}³ = {int(report.get('point_count', 0)):,} points",
            f"Map span: {float(report.get('span_mm', 0.0)):.9g} mm",
            f"Driven coils: {coil_text or '—'}",
            f"Field: {report.get('field', 'B')}",
            "",
            "BASIS RECONSTRUCTION VS FRESH DIRECT SOLVES",
            *state_lines,
            f"Worst vector error: {float(report.get('max_vector_error', 0.0)):.9g} {unit}",
            f"Worst relative error: {float(report.get('max_relative_error_pct', 0.0)):.9g} %",
            f"Validation absolute tolerance: {float(report.get('absolute_tolerance', 0.0)):.9g} {unit}",
            "",
            *colour_lines,
            "",
            "INTERPRETATION",
            "PASS means the field basis used by animation reproduces fresh direct solver results on the exact Full Volume sample lattice within the stated validation tolerance.",
            "This test deliberately does NOT validate volume transparency, ray marching, geometry occlusion, or final rendered pixels.",
        ]
        report_text = "\n".join(lines)
        dialog = QDialog(self)
        enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Animation field validation")
        dialog.resize(920, 720)
        layout = QVBoxLayout(dialog)
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setPlainText(report_text)
        layout.addWidget(output, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = buttons.addButton("Copy report", QDialogButtonBox.ButtonRole.ActionRole)
        copy_button.clicked.connect(lambda *_args: QApplication.clipboard().setText(report_text))
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        dialog.exec()

    def validate_animation_field(self) -> None:
        drive_ids = self.selected_active_coil_ids()
        if not drive_ids:
            QMessageBox.information(
                self,
                "Animation field validation",
                "Select at least one Active coil before running the validation.",
            )
            return
        controlled_ids = [str(choice["id"]) for choice in drive_coil_catalog(self.adapter)]
        progress: _FieldMapProgressDialog | None = None
        started = time.perf_counter()
        try:
            progress = _show_field_map_progress(
                self,
                "Validating the animation field basis against fresh direct solves…",
            )
            progress.setWindowTitle("Validating animation field")
            report = validate_animation_field_basis(
                self.adapter,
                coil_ids=drive_ids,
                controlled_coil_ids=controlled_ids,
                centre_mm=[spin.value() for spin in self.coordinates],
                span_mm=self.span.value(),
                resolution=self.resolution.value(),
                field=self.field.currentText(),
                progress_callback=lambda update: _update_field_map_progress(progress, update),
            )
            self.calculation_status.showMessage(
                f"Animation field validation {'passed' if report.get('passed') else 'needs review'} • "
                f"max error {float(report.get('max_vector_error', 0.0)):.4g} {report.get('unit', '')} • "
                f"elapsed {_format_elapsed_time(time.perf_counter() - started)}"
            )
            self._show_animation_validation_report(report)
        except FieldCalculationCancelled:
            self.calculation_status.showMessage("Animation field validation cancelled")
        except Exception as error:  # noqa: BLE001 - diagnostic GUI boundary
            QMessageBox.warning(self, "Animation field validation", str(error))
            self.calculation_status.showMessage("Animation field validation failed")
        finally:
            _finish_field_map_progress(progress)

    def _show_gpu_volume_validation_report(self, report: dict[str, Any]) -> None:
        if not bool(report.get("ok")):
            QMessageBox.warning(
                self,
                "GPU volume validation",
                str(report.get("error") or "The GPU validation did not return a result."),
            )
            self.calculation_status.showMessage("GPU volume validation failed")
            return

        unit = str(report.get("unit", ""))
        verdict = "PASS" if bool(report.get("passed")) else "CHECK FAILED"
        currents = ", ".join(
            f"{1000.0 * float(value):.6g}" for value in report.get("currents_a", [])
        )
        probe_lines: list[str] = []
        for index, probe in enumerate(report.get("probes", []), start=1):
            position = ", ".join(f"{float(value):.4g}" for value in probe.get("position_mm", []))
            cpu_rgb = ", ".join(str(int(value)) for value in probe.get("cpu_rgb", []))
            gpu_rgb = ", ".join(str(int(value)) for value in probe.get("gpu_rgb", []))
            probe_lines.extend(
                [
                    f"Probe {index}: grid {probe.get('grid', [])}; position [{position}] mm",
                    f"  source lattice scalar   {float(probe.get('cpu_value', 0.0)):.9g} {unit}",
                    f"  GPU texture scalar      {float(probe.get('gpu_value', 0.0)):.9g} {unit}",
                    f"  absolute error          {float(probe.get('error', 0.0)):.9g} {unit}",
                    f"  colour fraction CPU/GPU {float(probe.get('cpu_colour_fraction', 0.0)):.6f} / {float(probe.get('gpu_colour_fraction', 0.0)):.6f}",
                    f"  expected RGB CPU/GPU    rgb({cpu_rgb}) / rgb({gpu_rgb})",
                    "",
                ]
            )

        lines = [
            "FIELD WORKBENCH — GPU FULL VOLUME VALIDATION",
            "",
            f"RESULT: {verdict}",
            f"Map lattice: {int(report.get('resolution', 0))}³ = {int(report.get('point_count', 0)):,} GPU samples",
            f"Field/component: {report.get('field', '')} / {report.get('component', '')}",
            f"Live drive currents: [{currents}] mA",
            f"3D texture filter: {report.get('texture_filter', 'unknown')}",
            f"Colour range: {float(report.get('colour_min', 0.0)):.9g} to {float(report.get('colour_max', 0.0)):.9g} {unit}",
            "",
            "SOURCE LATTICE VS ACTUAL WEBGL TEXTURE SAMPLING",
            f"Peak source field          {float(report.get('peak_field', 0.0)):.9g} {unit}",
            f"Maximum scalar error       {float(report.get('max_abs_error', 0.0)):.9g} {unit}",
            f"RMS scalar error           {float(report.get('rms_abs_error', 0.0)):.9g} {unit}",
            f"Maximum relative error     {float(report.get('max_relative_error_pct', 0.0)):.9g} %",
            f"RMS relative error         {float(report.get('rms_relative_error_pct', 0.0)):.9g} %",
            f"Max colour-position error  {100.0 * float(report.get('max_colour_fraction_error', 0.0)):.6g} % of scale",
            f"Validation abs tolerance   {float(report.get('absolute_tolerance', 0.0)):.9g} {unit}",
            "",
            *probe_lines,
            "INTERPRETATION",
            "PASS means the Full Volume WebGL path samples its uploaded fixed/basis textures, applies the live coil currents, converts the selected component, and reaches the same scalar/colour position as the source animation lattice within tolerance.",
            "This still deliberately excludes transparency, ray-selection/surface choice, scene occlusion, and final framebuffer compositing.",
        ]
        report_text = "\n".join(lines)
        self.calculation_status.showMessage(
            f"GPU volume validation {'passed' if report.get('passed') else 'needs review'} • "
            f"max scalar error {float(report.get('max_abs_error', 0.0)):.4g} {unit}"
        )
        dialog = QDialog(self)
        enable_standard_window_controls(dialog)
        dialog.setWindowTitle("GPU volume validation")
        dialog.resize(920, 720)
        layout = QVBoxLayout(dialog)
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setPlainText(report_text)
        layout.addWidget(output, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = buttons.addButton("Copy report", QDialogButtonBox.ButtonRole.ActionRole)
        copy_button.clicked.connect(lambda *_args: QApplication.clipboard().setText(report_text))
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        dialog.exec()

    def validate_gpu_volume(self) -> None:
        widget = self.tabs.currentWidget()
        if not isinstance(widget, WaveformGLView):
            QMessageBox.information(
                self,
                "GPU volume validation",
                "Open a Full Volume Animated map and leave its Waveform tab selected before running this test.",
            )
            return
        if str(getattr(widget, "_payload", {}).get("render_mode", "")) != "volume":
            QMessageBox.information(
                self,
                "GPU volume validation",
                "The selected Waveform tab is not a Full Volume animation.",
            )
            return
        self.calculation_status.showMessage("Reading the live Full Volume GPU texture pipeline…")
        widget.validate_gpu_volume(self._show_gpu_volume_validation_report)

    def _open_waveform_action(self, primary_action: str) -> None:
        def launch(initial_camera: dict[str, Any]) -> None:
            camera_timeline = self._camera_timeline_for_action(initial_camera)
            target_fps, playback_speed = self._camera_playback_options_for_action()
            dialog = WaveformPlaybackDialog(
                self.adapter,
                map_kind="3d",
                map_settings=self._waveform_map_settings(),
                camera_timeline=camera_timeline,
                initial_camera=initial_camera,
                playback_fps=target_fps,
                playback_speed=playback_speed,
                primary_action=primary_action,
                defer_payload_build=primary_action == "playback",
                parent=self,
            )
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                if primary_action == "export" or dialog.result_gpu_payload is None:
                    if (
                        primary_action == "playback"
                        and isinstance(dialog.result_gpu_request, dict)
                    ):
                        self._waveform_tab_counter += 1
                        job_label = f"Waveform {self._waveform_tab_counter}"
                        self._start_render_job(
                            render_kind="waveform",
                            request=dialog.result_gpu_request,
                            job_label=job_label,
                            tooltip=dialog.result_label + " — interactive playback",
                        )
                        return
                    self.calculation_status.showMessage("Video export finished")
                    return
                self._waveform_tab_counter += 1
                plot = WaveformGLView(dialog.result_gpu_payload)
                self._connect_result_camera_viewer(plot)
                index = self.tabs.addTab(plot, f"Waveform {self._waveform_tab_counter}")
                self.tabs.setTabToolTip(index, dialog.result_label + " — interactive playback")
                self.tabs.setCurrentIndex(index)
                # WebEngine can size its canvas while the new tab is still hidden.
                # Explicitly wake/resize after activation so the animated viewer fits
                # the tab immediately instead of waiting for a mouse interaction.
                plot.refresh_viewport()
                self.calculation_status.showMessage("Animated map ready")
            finally:
                # A parented modal dialog otherwise remains alive until the whole
                # 3D window closes, retaining its trace/request arrays after every
                # successive animation setup.
                dialog.result_figure = None
                dialog.result_gpu_payload = None
                dialog.result_gpu_request = None
                dialog.deleteLater()

        self._request_action_camera(launch)

    def open_waveform_playback(self) -> None:
        self._open_waveform_action("playback")

    def _current_video_export_view(self) -> WaveformGLView | None:
        widget = self.tabs.currentWidget()
        if isinstance(widget, WaveformGLView) and widget.can_export_video():
            return widget
        return None

    def _update_video_export_button(self) -> None:
        self.export_video_button.setVisible(self._current_video_export_view() is not None)

    def _export_current_animation_video(self, *_args) -> None:
        """Run the original video exporter from a frozen rendered result.

        The pre-0.13.0.290 exporter was reliable because its export preview was
        the only WebGL surface involved.  Keep the new rendered-tab workflow,
        but retire that tab's renderer completely before opening the unchanged
        exporter, then recreate the rendered tab from the same frozen payload.
        """
        view = self._current_video_export_view()
        if view is None:
            return
        view.stop_playback()

        def launch(camera: dict[str, Any] | None) -> None:
            try:
                payload = view.export_payload()
                if isinstance(camera, dict):
                    payload["initial_camera"] = copy.deepcopy(camera)
            except Exception as error:  # noqa: BLE001 - rendered-tab payload boundary
                QMessageBox.warning(self, "Unable to export video", str(error))
                return

            tab_index = self.tabs.indexOf(view)
            if tab_index < 0:
                return
            tab_text = self.tabs.tabText(tab_index)
            tab_tooltip = self.tabs.tabToolTip(tab_index)
            placeholder = QLabel("Video export in progress…")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Fully destroy the rendered tab's Chromium/WebGL surface.  The old
            # exporter then runs in exactly the single-renderer situation it had
            # before Export video moved beside Fullscreen viewer.
            self.tabs.removeTab(tab_index)
            self.tabs.insertTab(tab_index, placeholder, tab_text)
            self.tabs.setTabToolTip(tab_index, tab_tooltip)
            self.tabs.setCurrentIndex(tab_index)
            view.cleanup()
            view.deleteLater()

            def run_original_exporter() -> None:
                dialog = WaveformVideoExportDialog(payload, parent=self)
                centre_window_on_parent(dialog, self)
                try:
                    if (
                        dialog.exec() == QDialog.DialogCode.Accepted
                        and dialog.output_path is not None
                    ):
                        self.calculation_status.showMessage(
                            f"Video exported to {dialog.output_path}"
                        )
                finally:
                    dialog.deleteLater()
                    restored = WaveformGLView(
                        payload,
                        initial_camera=(camera if isinstance(camera, dict) else None),
                    )
                    self._connect_result_camera_viewer(restored)
                    current_index = self.tabs.indexOf(placeholder)
                    if current_index >= 0:
                        self.tabs.removeTab(current_index)
                        restored_index = self.tabs.insertTab(
                            current_index, restored, tab_text
                        )
                        self.tabs.setTabToolTip(restored_index, tab_tooltip)
                        self.tabs.setCurrentIndex(restored_index)
                        restored.refresh_viewport()
                    else:
                        restored.cleanup()
                        restored.deleteLater()
                    placeholder.deleteLater()
                    self._update_video_export_button()

            # cleanup() releases QWebEngineView with deleteLater().  Give Qt one
            # event-loop turn to actually retire it before the export preview is
            # constructed.
            QTimer.singleShot(0, run_original_exporter)

        view.request_camera(launch)

    def open_video_export(self) -> None:
        """Compatibility entry point: export only the selected rendered animation."""
        self._export_current_animation_video()

    @Slot(int)
    def _render_tab_changed(self, current_index: int) -> None:
        """Keep only the selected WebGL result resident in RAM and on the GPU."""

        current_widget = self.tabs.widget(current_index)
        current_view = (
            current_widget if isinstance(current_widget, WaveformGLView) else None
        )
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if not isinstance(widget, WaveformGLView):
                continue
            if widget is current_view:
                widget.set_backgrounded(False)
            else:
                # Release the hidden Chromium surface. Selecting the tab builds a
                # fresh surface from its existing local HTML; no field solve repeats.
                widget.set_backgrounded(True, discard=True)
        self._active_render_view = current_view
        self._update_video_export_button()

    def _close_map_tab(self, index: int) -> None:
        """Close generated plots while keeping the Selector tab permanent."""
        if index == self._selector_tab_index:
            return
        widget = self.tabs.widget(index)
        if isinstance(widget, _FieldRenderJobPage):
            for job in getattr(self, "_render_jobs", {}).values():
                if job.get("page") is not widget:
                    continue
                job["discard"] = True
                worker = job.get("worker")
                if isinstance(worker, _FieldRenderWorker):
                    worker.request_cancellation()
                break
        self.tabs.removeTab(index)
        if widget is self._active_render_view:
            self._active_render_view = None
        if isinstance(widget, (PlotView, WaveformGLView)):
            widget.cleanup()
        if widget is not None:
            widget.deleteLater()
        self._render_tab_changed(self.tabs.currentIndex())

    def _current_plot(self):
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, (PlotView, WaveformGLView)) else self.plot

    def _toggle_current_plot_fullscreen(self, *_args) -> None:
        current = self._current_plot()
        toggle = getattr(current, "toggle_fullscreen", None)
        if callable(toggle):
            toggle()

    def _cleanup_plots(self) -> None:
        if getattr(self, "_plots_cleanup_started", False):
            return
        self._plots_cleanup_started = True
        cancel_jobs = getattr(self, "_cancel_all_render_jobs", None)
        if callable(cancel_jobs):
            cancel_jobs()
        self._active_render_view = None
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if isinstance(widget, (PlotView, WaveformGLView)):
                widget.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        # Dispose active WebGL pages before Qt destroys the dialog's native
        # surface. This gives the renderer a live GL context for final cleanup.
        self._cleanup_plots()
        super().closeEvent(event)


class MeasurementResultsDialog(QDialog):
    """Single-volume statistics and distribution for one analyzed geometry object."""

    def __init__(
        self,
        adapter: StudioAdapter,
        result: dict,
        parent=None,
        on_preview: Callable[[dict], None] | None = None,
        on_snapshot_saved: Callable[[str], None] | None = None,
        allow_snapshot: bool = True,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.result = result
        self.on_preview = on_preview
        self.on_snapshot_saved = on_snapshot_saved
        snapshot_name = result.get("snapshot_name")
        self.setWindowTitle(
            f"Snapshot — {snapshot_name}" if snapshot_name else f"Measurement — {result['label']}"
        )
        self.resize(920, 760)

        root = QVBoxLayout(self)
        heading_text = f"Field inside {result['label']}"
        if snapshot_name:
            heading_text = f"{snapshot_name} — {heading_text}"
        heading = QLabel(heading_text)
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        sampling_mode = str(
            result.get(
                "sampling_mode",
                result.get("settings", {}).get("sampling_mode", "generated"),
            )
        )
        method_text = (
            "User-defined equal-weight sample points in the object's local coordinate frame. "
            if sampling_mode == "user_defined"
            else "Equal-volume voxel-centre samples in the object's local coordinate frame. "
        )
        method = QLabel(
            method_text
            + "Field magnitude statistics use |B|; direction statistics use the complete B vector."
        )
        method.setWordWrap(True)
        method.setObjectName("hint")
        root.addWidget(method)
        snapshot_status = result.get("snapshot_status")
        if snapshot_status and snapshot_status != "current":
            warning = QLabel(
                "This stored result is stale: the source scene or measurement definition has changed."
                if snapshot_status == "stale"
                else "The original measurement object is no longer available; the stored result is still intact."
            )
            warning.setWordWrap(True)
            warning.setObjectName("warningPanelCompact")
            root.addWidget(warning)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_overview_tab()
        self._build_statistics_tab()
        cross_section_name = f"{snapshot_name or result['label']}-cross-sections"
        self.cross_sections = PlotView(image_filename=cross_section_name)
        self.cross_sections.set_figure(adapter.measurement_cross_section_figure(result))
        self.tabs.addTab(self.cross_sections, "Cross sections")
        histogram_name = f"{snapshot_name or result['label']}-distribution"
        self.histogram = PlotView(image_filename=histogram_name)
        self.histogram.set_figure(adapter.measurement_histogram_figure(result))
        self.tabs.addTab(self.histogram, "Distribution")
        self.finished.connect(
            lambda _code: (self.cross_sections.cleanup(), self.histogram.cleanup())
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        if allow_snapshot and not snapshot_name:
            self.save_snapshot_button = buttons.addButton(
                "Save snapshot…", QDialogButtonBox.ButtonRole.ActionRole
            )
            self.save_snapshot_button.setObjectName("primaryButton")
            self.save_snapshot_button.clicked.connect(self.save_snapshot)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if self.on_preview is not None:
            self.on_preview(adapter.measurement_overlay(result))
        _make_result_text_selectable(self)

    def save_snapshot(self) -> None:
        suggested = self.adapter.suggested_snapshot_name(self.result)
        name, accepted = QInputDialog.getText(
            self,
            "Save field snapshot",
            "Snapshot name",
            text=suggested,
        )
        if not accepted:
            return
        try:
            snapshot_id = self.adapter.create_snapshot(self.result, name)
        except Exception as error:  # noqa: BLE001 - saved-result boundary
            QMessageBox.warning(self, "Save field snapshot", str(error))
            return
        if self.on_snapshot_saved is not None:
            self.on_snapshot_saved(snapshot_id)
        QMessageBox.information(
            self,
            "Field snapshot saved",
            f"Saved ‘{name.strip()}’ with the exact sample coordinates and B-vector grid.",
        )

    @staticmethod
    def _format(value: float, unit: str = "", digits: int = 6) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(value):
            return "—"
        return f"{value:,.{digits}g}{(' ' + unit) if unit else ''}"

    def _card(self, title: str, value: str, detail: str) -> QGroupBox:
        card = QGroupBox(title)
        layout = QVBoxLayout(card)
        number = _selectable_label(value)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setObjectName("resultValue")
        layout.addWidget(number)
        note = _selectable_label(detail, word_wrap=True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setObjectName("hint")
        layout.addWidget(note)
        return card

    def _build_overview_tab(self) -> None:
        stats = self.result["stats"]
        page = QWidget()
        layout = QVBoxLayout(page)
        cards = QGridLayout()
        cards.addWidget(
            self._card(
                "Mean field",
                self._format(stats["mean_uT"], "µT"),
                f"Median {self._format(stats['median_uT'], 'µT')}",
            ),
            0,
            0,
        )
        cards.addWidget(
            self._card(
                "Uniformity spread",
                self._format(stats["uniformity_spread_pct"], "%"),
                "100 × (P95 − P5) / mean; lower is more uniform",
            ),
            0,
            1,
        )
        cards.addWidget(
            self._card(
                "Within target band",
                self._format(stats["in_band_pct"], "%"),
                f"{self._format(stats['target_uT'], 'µT')} ± "
                f"{self._format(stats['tolerance_pct'], '%')}",
            ),
            1,
            0,
        )
        cards.addWidget(
            self._card(
                "Directional consistency",
                self._format(stats["directional_consistency_pct"], "%"),
                "100% means every sampled B vector points the same way",
            ),
            1,
            1,
        )
        layout.addLayout(cards)

        target = QGroupBox("Target coverage")
        form = QFormLayout(target)
        form.addRow("Target band", _selectable_label(
            f"{self._format(stats['target_low_uT'], 'µT')} to "
            f"{self._format(stats['target_high_uT'], 'µT')}"
        ))
        form.addRow("Below band", _selectable_label(self._format(stats["below_target_pct"], "%")))
        form.addRow("Inside band", _selectable_label(self._format(stats["in_band_pct"], "%")))
        form.addRow("Above band", _selectable_label(self._format(stats["above_target_pct"], "%")))
        form.addRow("RMS error from target", _selectable_label(self._format(stats["rms_target_error_uT"], "µT")))
        layout.addWidget(target)

        preview = QGroupBox("Object preview")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 10, 8, 8)
        self.object_preview = MeasurementPreviewWidget(self.result)
        preview_layout.addWidget(self.object_preview)
        layout.addWidget(preview, 1)
        self.tabs.addTab(page, "Overview")

    def _build_statistics_tab(self) -> None:
        stats = self.result["stats"]
        content = QWidget()
        layout = QVBoxLayout(content)

        magnitude_rows = [
            (label, self._format(stats[key], unit))
            for label, key, unit in (
                ("Mean", "mean_uT", "µT"),
                ("Median", "median_uT", "µT"),
                ("Minimum", "min_uT", "µT"),
                ("Maximum", "max_uT", "µT"),
                ("Standard deviation", "std_uT", "µT"),
                ("Coefficient of variation", "coefficient_variation_pct", "%"),
                ("P5", "p5_uT", "µT"),
                ("P25", "p25_uT", "µT"),
                ("P75", "p75_uT", "µT"),
                ("P95", "p95_uT", "µT"),
                ("P5–P95 uniformity spread", "uniformity_spread_pct", "%"),
            )
        ]
        layout.addWidget(_copyable_statistics_group("Field magnitude |B|", magnitude_rows))

        mean_vector = stats["mean_vector_uT"]
        mean_direction = stats["mean_direction"]
        direction_rows = [
            ("Mean B vector", ", ".join(self._format(value, "µT") for value in mean_vector)),
            ("Mean unit direction", ", ".join(self._format(value, digits=5) for value in mean_direction)),
            ("Directional consistency", self._format(stats["directional_consistency_pct"], "%")),
            ("Median angular deviation", self._format(stats["median_angular_deviation_deg"], "°")),
            ("P95 angular deviation", self._format(stats["p95_angular_deviation_deg"], "°")),
        ]
        layout.addWidget(_copyable_statistics_group("Vector direction", direction_rows))

        spacing = stats["sample_spacing_mm"]
        sampling_mode = str(
            self.result.get(
                "sampling_mode",
                self.result.get("settings", {}).get("sampling_mode", "generated"),
            )
        )
        sampling_rows = [
            ("Shape", str(self.result.get("shape_label", self.result["shape"])).capitalize()),
            (
                "Sampling",
                "User-defined local points"
                if sampling_mode == "user_defined"
                else str(self.result["quality"]).capitalize(),
            ),
        ]
        if sampling_mode != "user_defined":
            sampling_rows.extend(
                [
                    ("Samples per bounding axis", str(stats["samples_per_axis"])),
                    (
                        "Voxel spacing (local X, Y, Z)",
                        ", ".join(self._format(value, "mm", 5) for value in spacing),
                    ),
                ]
            )
        sampling_rows.extend(
            [
                ("Requested samples", f"{stats['sample_count']:,}"),
                ("Finite samples used", f"{stats['valid_sample_count']:,}"),
                ("Invalid samples excluded", f"{stats['invalid_sample_count']:,}"),
                ("Geometric volume", self._format(stats["volume_cm3"], "cm³")),
            ]
        )
        layout.addWidget(_copyable_statistics_group("Sampling record", sampling_rows))
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Full statistics")

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        self.cross_sections.cleanup()
        self.histogram.cleanup()
        super().closeEvent(event)


class CurrentNormalizationDialog(QDialog):
    """Arrange candidate coils into visual shared-current fit groups."""

    FIXED_TEXT = "Fixed (not normalized)"
    INDEPENDENT_TEXT = "Normalize independently"
    SHARED_TEXT = "Normalize together"

    def __init__(
        self,
        adapter: StudioAdapter,
        candidate_ids: list[str],
        *,
        current_mode: str = "common",
        selected_coils: dict[str, list[str]] | None = None,
        selected_assignments: dict[str, dict[str, str]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.candidate_ids = list(candidate_ids)
        self.setWindowTitle("Configure current normalization")
        self.resize(900, 620)

        self._candidate_names: dict[str, str] = {
            str(item.get("id")): str(item.get("name", item.get("id", "Snapshot")))
            for item in adapter.list_snapshots()
        }
        self._coils: dict[str, dict[str, dict]] = {}
        self._coil_order: dict[str, list[str]] = {}
        self._groups: dict[str, dict[str, dict]] = {}
        self._ungrouped_modes: dict[str, dict[str, str]] = {}
        self._group_serial = 0

        self._candidate_items: dict[str, QTreeWidgetItem] = {}
        self._coil_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self._group_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self._coil_combos: dict[tuple[str, str], QComboBox] = {}
        self._group_combos: dict[tuple[str, str], QComboBox] = {}

        root = QVBoxLayout(self)
        heading = QLabel("Candidate-coil current normalization")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            "Current fitting minimizes pointwise field-magnitude error; field direction remains "
            "a separate comparison result. Each candidate snapshot keeps its own coil subtree. "
            "Existing scene groups are "
            "inherited automatically when they contain two or more adjustable coils; otherwise "
            "coils begin ungrouped. Grouped coils share one current dropdown, while ungrouped "
            "coils have their own Fixed/Normalize independently dropdown. Saved field-scale "
            "factors and coil polarity remain unchanged."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick setting for every group and ungrouped coil"))
        self.all_relationship = QComboBox()
        self.all_relationship.addItem(self.FIXED_TEXT, "fixed")
        self.all_relationship.addItem("Normalize", "normalize")
        self.all_relationship.setCurrentIndex(1)
        self.all_relationship.setMinimumWidth(230)
        preset_row.addWidget(self.all_relationship)
        apply_all = QPushButton("Apply")
        apply_all.clicked.connect(self._apply_all_relationship)
        preset_row.addWidget(apply_all)
        preset_row.addStretch(1)
        root.addLayout(preset_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ["Candidate / current group / coil", "Saved current", "Normalization"]
        )
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setColumnWidth(0, 430)
        self.tree.setColumnWidth(1, 150)
        self.tree.setColumnWidth(2, 260)
        self.tree.itemSelectionChanged.connect(self._update_action_buttons)
        self.tree.itemDoubleClicked.connect(self._tree_item_double_clicked)
        root.addWidget(self.tree, 1)

        action_row = QHBoxLayout()
        self.group_button = QPushButton("Group selected coils")
        self.group_button.clicked.connect(self._group_selected)
        self.ungroup_button = QPushButton("Ungroup")
        self.ungroup_button.clicked.connect(self._ungroup_selected)
        self.rename_group_button = QPushButton("Rename group")
        self.rename_group_button.clicked.connect(self._rename_selected_group)
        action_row.addWidget(self.group_button)
        action_row.addWidget(self.ungroup_button)
        action_row.addWidget(self.rename_group_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        self._load_state(
            current_mode=current_mode,
            selected_coils=selected_coils or {},
            selected_assignments=selected_assignments or {},
        )
        self._rebuild_tree()

        legend = QLabel(
            "Scene groups are only a starting arrangement; normalization groups can still be "
            "edited here without changing the scene. Select two or more coil rows and choose "
            "Group selected coils, or select a group/member and choose Ungroup. Double-click a "
            "group name to rename it."
        )
        legend.setWordWrap(True)
        legend.setObjectName("hint")
        root.addWidget(legend)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_configured)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _normalized_relationship(value: str) -> str:
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {"", "fixed", "off", "none"}:
            return "fixed"
        if lowered in {"independent", "normalize independently"}:
            return "independent"
        return text

    def _new_group_id(self) -> str:
        self._group_serial += 1
        return f"fit_group_{self._group_serial}"

    def _next_group_name(self, candidate_id: str) -> str:
        existing = {
            str(group.get("name", "")).casefold()
            for group in self._groups.get(candidate_id, {}).values()
        }
        index = 1
        while f"current group {index}".casefold() in existing:
            index += 1
        return f"Current group {index}"

    def _load_state(
        self,
        *,
        current_mode: str,
        selected_coils: dict[str, list[str]],
        selected_assignments: dict[str, dict[str, str]],
    ) -> None:
        for candidate_id in self.candidate_ids:
            coil_records = self.adapter.snapshot_normalization_coils(candidate_id)
            records = {str(coil["id"]): dict(coil) for coil in coil_records}
            order = [str(coil["id"]) for coil in coil_records]
            self._coils[candidate_id] = records
            self._coil_order[candidate_id] = order
            self._groups[candidate_id] = {}
            self._ungrouped_modes[candidate_id] = {}

            assignments = {
                str(key): self._normalized_relationship(value)
                for key, value in selected_assignments.get(candidate_id, {}).items()
            }
            legacy_selected = set(map(str, selected_coils.get(candidate_id, [])))

            if assignments:
                grouped_by_name: dict[str, list[str]] = {}
                group_display_names: dict[str, str] = {}
                for coil_id in order:
                    relationship = assignments.get(coil_id, "fixed")
                    if relationship in {"fixed", "independent"}:
                        self._ungrouped_modes[candidate_id][coil_id] = relationship
                    else:
                        key = relationship.casefold()
                        grouped_by_name.setdefault(key, []).append(coil_id)
                        group_display_names.setdefault(key, relationship)
                for key, coil_ids in grouped_by_name.items():
                    if len(coil_ids) == 1:
                        self._ungrouped_modes[candidate_id][coil_ids[0]] = "independent"
                        continue
                    group_id = self._new_group_id()
                    self._groups[candidate_id][group_id] = {
                        "name": group_display_names[key],
                        "coil_ids": list(coil_ids),
                        "mode": "shared",
                    }
                continue

            if candidate_id in selected_coils:
                if str(current_mode).strip().lower() == "independent":
                    for coil_id in order:
                        self._ungrouped_modes[candidate_id][coil_id] = (
                            "independent" if coil_id in legacy_selected else "fixed"
                        )
                else:
                    selected = [coil_id for coil_id in order if coil_id in legacy_selected]
                    if selected:
                        group_id = self._new_group_id()
                        self._groups[candidate_id][group_id] = {
                            "name": "Group 1",
                            "coil_ids": selected,
                            "mode": "shared",
                        }
                    for coil_id in order:
                        if coil_id not in legacy_selected:
                            self._ungrouped_modes[candidate_id][coil_id] = "fixed"
                continue

            # Fresh configuration mirrors the source scene: coils sharing the same
            # immediate Collection parent inherit one shared-current fit group and its
            # display name.  Root-level coils (and singleton scene groups) remain
            # independently controllable rather than being bundled automatically.
            scene_groups: dict[str, dict[str, Any]] = {}
            for coil_id in order:
                group_path = records.get(coil_id, {}).get("group_path", [])
                if not isinstance(group_path, list) or not group_path:
                    continue
                immediate = group_path[-1]
                if not isinstance(immediate, dict):
                    continue
                scene_group_id = str(immediate.get("id", "")).strip()
                if not scene_group_id:
                    continue
                entry = scene_groups.setdefault(
                    scene_group_id,
                    {
                        "name": str(immediate.get("label") or scene_group_id),
                        "coil_ids": [],
                    },
                )
                entry["coil_ids"].append(coil_id)

            grouped_ids: set[str] = set()
            for scene_group in scene_groups.values():
                coil_ids = [
                    coil_id for coil_id in order
                    if coil_id in set(map(str, scene_group.get("coil_ids", [])))
                ]
                if len(coil_ids) < 2:
                    continue
                requested_name = str(scene_group.get("name") or "Current group")
                name = requested_name
                existing_names = {
                    str(group.get("name", "")).casefold()
                    for group in self._groups[candidate_id].values()
                }
                suffix = 2
                while name.casefold() in existing_names:
                    name = f"{requested_name} ({suffix})"
                    suffix += 1
                group_id = self._new_group_id()
                self._groups[candidate_id][group_id] = {
                    "name": name,
                    "coil_ids": coil_ids,
                    "mode": "shared",
                }
                grouped_ids.update(coil_ids)

            for coil_id in order:
                if coil_id not in grouped_ids:
                    self._ungrouped_modes[candidate_id][coil_id] = "independent"

    def _mode_combo(self, mode: str, *, grouped: bool) -> QComboBox:
        combo = QComboBox()
        combo.addItem(self.FIXED_TEXT, "fixed")
        combo.addItem(self.SHARED_TEXT if grouped else self.INDEPENDENT_TEXT,
                      "shared" if grouped else "independent")
        target = "shared" if grouped and mode != "fixed" else mode
        index = combo.findData(target)
        combo.setCurrentIndex(max(0, index))
        combo.setMinimumWidth(230)
        return combo

    def _saved_current_summary(self, candidate_id: str, coil_ids: list[str]) -> str:
        currents = [
            float(self._coils[candidate_id][coil_id].get("current_a", 0.0))
            for coil_id in coil_ids
            if coil_id in self._coils[candidate_id]
        ]
        if not currents:
            return "—"
        if np.allclose(currents, currents[0], rtol=1e-9, atol=1e-12):
            return f"{currents[0]:.7g} A each"
        return "varies"

    def _rebuild_tree(self, selected_keys: set[tuple] | None = None) -> None:
        selected_keys = selected_keys or set()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._candidate_items.clear()
        self._coil_items.clear()
        self._group_items.clear()
        self._coil_combos.clear()
        self._group_combos.clear()

        for candidate_id in self.candidate_ids:
            candidate_item = QTreeWidgetItem(
                [self._candidate_names.get(candidate_id, candidate_id), "", ""]
            )
            candidate_item.setData(0, Qt.ItemDataRole.UserRole, ("candidate", candidate_id))
            self.tree.addTopLevelItem(candidate_item)
            self._candidate_items[candidate_id] = candidate_item

            records = self._coils.get(candidate_id, {})
            if not records:
                unavailable = QTreeWidgetItem(
                    ["No replayable coil-current data", "—", "Save a fresh snapshot"]
                )
                unavailable.setForeground(2, QBrush(QColor("#b91c1c")))
                candidate_item.addChild(unavailable)
                candidate_item.setExpanded(True)
                continue

            grouped_coils: set[str] = set()
            for group_id, group in self._groups.get(candidate_id, {}).items():
                coil_ids = [
                    coil_id for coil_id in group.get("coil_ids", []) if coil_id in records
                ]
                if not coil_ids:
                    continue
                grouped_coils.update(coil_ids)
                group_item = QTreeWidgetItem(
                    [str(group.get("name", "Current group")),
                     self._saved_current_summary(candidate_id, coil_ids), ""]
                )
                key = ("fit_group", candidate_id, group_id)
                group_item.setData(0, Qt.ItemDataRole.UserRole, key)
                candidate_item.addChild(group_item)
                self._group_items[(candidate_id, group_id)] = group_item
                combo = self._mode_combo(str(group.get("mode", "shared")), grouped=True)
                combo.currentIndexChanged.connect(
                    lambda _index, cid=candidate_id, gid=group_id, widget=combo:
                    self._set_group_mode(cid, gid, str(widget.currentData()))
                )
                self.tree.setItemWidget(group_item, 2, combo)
                self._group_combos[(candidate_id, group_id)] = combo

                for coil_id in coil_ids:
                    coil = records[coil_id]
                    child = QTreeWidgetItem(
                        [str(coil.get("label", coil_id)),
                         f"{float(coil.get('current_a', 0.0)):.7g} A",
                         "Shares group current"]
                    )
                    coil_key = ("coil", candidate_id, coil_id)
                    child.setData(0, Qt.ItemDataRole.UserRole, coil_key)
                    group_item.addChild(child)
                    self._coil_items[(candidate_id, coil_id)] = child
                    if coil_key in selected_keys:
                        child.setSelected(True)
                group_item.setExpanded(True)
                if key in selected_keys:
                    group_item.setSelected(True)

            for coil_id in self._coil_order.get(candidate_id, []):
                if coil_id in grouped_coils or coil_id not in records:
                    continue
                coil = records[coil_id]
                item = QTreeWidgetItem(
                    [str(coil.get("label", coil_id)),
                     f"{float(coil.get('current_a', 0.0)):.7g} A", ""]
                )
                key = ("coil", candidate_id, coil_id)
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                candidate_item.addChild(item)
                self._coil_items[(candidate_id, coil_id)] = item
                mode = self._ungrouped_modes[candidate_id].get(coil_id, "fixed")
                combo = self._mode_combo(mode, grouped=False)
                combo.currentIndexChanged.connect(
                    lambda _index, cid=candidate_id, coil=coil_id, widget=combo:
                    self._set_ungrouped_mode(cid, coil, str(widget.currentData()))
                )
                self.tree.setItemWidget(item, 2, combo)
                self._coil_combos[(candidate_id, coil_id)] = combo
                if key in selected_keys:
                    item.setSelected(True)

            candidate_item.setExpanded(True)
            candidate_key = ("candidate", candidate_id)
            if candidate_key in selected_keys:
                candidate_item.setSelected(True)

        self.tree.blockSignals(False)
        self._update_action_buttons()

    def _set_group_mode(self, candidate_id: str, group_id: str, mode: str) -> None:
        group = self._groups.get(candidate_id, {}).get(group_id)
        if group is not None:
            group["mode"] = "fixed" if mode == "fixed" else "shared"

    def _set_ungrouped_mode(self, candidate_id: str, coil_id: str, mode: str) -> None:
        self._ungrouped_modes.setdefault(candidate_id, {})[coil_id] = (
            "fixed" if mode == "fixed" else "independent"
        )

    def _apply_all_relationship(self) -> None:
        normalize = str(self.all_relationship.currentData()) == "normalize"
        for candidate_id in self.candidate_ids:
            for group in self._groups.get(candidate_id, {}).values():
                group["mode"] = "shared" if normalize else "fixed"
            for coil_id in list(self._ungrouped_modes.get(candidate_id, {})):
                self._ungrouped_modes[candidate_id][coil_id] = (
                    "independent" if normalize else "fixed"
                )
        selected = {
            tuple(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.tree.selectedItems()
            if isinstance(item.data(0, Qt.ItemDataRole.UserRole), (tuple, list))
        }
        self._rebuild_tree(selected)

    def _group_for_coil(self, candidate_id: str, coil_id: str) -> str | None:
        for group_id, group in self._groups.get(candidate_id, {}).items():
            if coil_id in group.get("coil_ids", []):
                return group_id
        return None

    def _selected_coil_refs(self) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in self.tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(value, (tuple, list)) or not value:
                continue
            if value[0] == "coil":
                ref = (str(value[1]), str(value[2]))
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
            elif value[0] == "fit_group":
                candidate_id, group_id = str(value[1]), str(value[2])
                group = self._groups.get(candidate_id, {}).get(group_id, {})
                for coil_id in group.get("coil_ids", []):
                    ref = (candidate_id, str(coil_id))
                    if ref not in seen:
                        refs.append(ref)
                        seen.add(ref)
        return refs

    def _selected_group_refs(self) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        for item in self.tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if (
                isinstance(value, (tuple, list))
                and len(value) >= 3
                and value[0] == "fit_group"
            ):
                refs.append((str(value[1]), str(value[2])))
        return refs

    def _remove_coil_from_group(self, candidate_id: str, coil_id: str) -> str:
        group_id = self._group_for_coil(candidate_id, coil_id)
        if group_id is None:
            return self._ungrouped_modes.get(candidate_id, {}).get(coil_id, "fixed")
        group = self._groups[candidate_id][group_id]
        mode = "fixed" if group.get("mode") == "fixed" else "independent"
        group["coil_ids"] = [value for value in group.get("coil_ids", []) if value != coil_id]
        if not group["coil_ids"]:
            del self._groups[candidate_id][group_id]
        return mode

    def _collapse_singleton_groups(self, candidate_id: str) -> None:
        for group_id, group in list(self._groups.get(candidate_id, {}).items()):
            coil_ids = list(group.get("coil_ids", []))
            if len(coil_ids) != 1:
                continue
            coil_id = str(coil_ids[0])
            self._ungrouped_modes[candidate_id][coil_id] = (
                "fixed" if group.get("mode") == "fixed" else "independent"
            )
            del self._groups[candidate_id][group_id]

    def _create_group(
        self,
        candidate_id: str,
        coil_ids: list[str],
        *,
        name: str | None = None,
    ) -> str:
        ordered = [
            coil_id for coil_id in self._coil_order.get(candidate_id, []) if coil_id in coil_ids
        ]
        if len(ordered) < 2:
            raise ValueError("Select at least two coils from one candidate.")
        for coil_id in ordered:
            self._remove_coil_from_group(candidate_id, coil_id)
            self._ungrouped_modes[candidate_id].pop(coil_id, None)
        self._collapse_singleton_groups(candidate_id)
        group_id = self._new_group_id()
        self._groups[candidate_id][group_id] = {
            "name": name or self._next_group_name(candidate_id),
            "coil_ids": ordered,
            "mode": "shared",
        }
        return group_id

    def _group_selected(self) -> None:
        refs = self._selected_coil_refs()
        candidate_ids = {candidate_id for candidate_id, _coil_id in refs}
        if len(refs) < 2 or len(candidate_ids) != 1:
            QMessageBox.information(
                self,
                "Group coils",
                "Select at least two coils belonging to the same candidate snapshot.",
            )
            return
        candidate_id = next(iter(candidate_ids))
        group_id = self._create_group(candidate_id, [coil_id for _cid, coil_id in refs])
        self._rebuild_tree({("fit_group", candidate_id, group_id)})

    def _ungroup_selected(self) -> None:
        selected_groups = set(self._selected_group_refs())
        selected_coils = self._selected_coil_refs()
        to_ungroup: dict[str, set[str]] = {}
        for candidate_id, group_id in selected_groups:
            group = self._groups.get(candidate_id, {}).get(group_id, {})
            to_ungroup.setdefault(candidate_id, set()).update(group.get("coil_ids", []))
        for candidate_id, coil_id in selected_coils:
            if self._group_for_coil(candidate_id, coil_id) is not None:
                to_ungroup.setdefault(candidate_id, set()).add(coil_id)
        if not any(to_ungroup.values()):
            return
        selected_keys: set[tuple] = set()
        for candidate_id, coil_ids in to_ungroup.items():
            for coil_id in coil_ids:
                mode = self._remove_coil_from_group(candidate_id, coil_id)
                self._ungrouped_modes[candidate_id][coil_id] = mode
                selected_keys.add(("coil", candidate_id, coil_id))
            self._collapse_singleton_groups(candidate_id)
        self._rebuild_tree(selected_keys)

    def _rename_selected_group(self) -> None:
        refs = self._selected_group_refs()
        if len(refs) != 1:
            return
        candidate_id, group_id = refs[0]
        group = self._groups.get(candidate_id, {}).get(group_id)
        if group is None:
            return
        value, accepted = QInputDialog.getText(
            self,
            "Rename current group",
            "Group name",
            text=str(group.get("name", "Current group")),
        )
        if not accepted:
            return
        name = str(value).strip()
        if not name:
            return
        if name.casefold() in {"fixed", "off", "none", "independent"}:
            QMessageBox.warning(
                self,
                "Rename current group",
                "That name is reserved for a normalization relationship.",
            )
            return
        duplicate = any(
            other_id != group_id
            and str(other.get("name", "")).casefold() == name.casefold()
            for other_id, other in self._groups.get(candidate_id, {}).items()
        )
        if duplicate:
            QMessageBox.warning(
                self,
                "Rename current group",
                "Current-group names must be unique within a candidate.",
            )
            return
        group["name"] = name
        self._rebuild_tree({("fit_group", candidate_id, group_id)})

    def _tree_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if column == 0 and isinstance(value, (tuple, list)) and value and value[0] == "fit_group":
            self.tree.clearSelection()
            item.setSelected(True)
            self._rename_selected_group()

    def _update_action_buttons(self) -> None:
        refs = self._selected_coil_refs()
        candidate_ids = {candidate_id for candidate_id, _coil_id in refs}
        self.group_button.setEnabled(len(refs) >= 2 and len(candidate_ids) == 1)
        grouped_selection = any(
            self._group_for_coil(candidate_id, coil_id) is not None
            for candidate_id, coil_id in refs
        ) or bool(self._selected_group_refs())
        self.ungroup_button.setEnabled(grouped_selection)
        self.rename_group_button.setEnabled(len(self._selected_group_refs()) == 1)

    def _accept_configured(self) -> None:
        assignments = self.selected_assignments()
        missing = [
            candidate_id
            for candidate_id in self.candidate_ids
            if not any(
                value != "fixed"
                for value in assignments.get(candidate_id, {}).values()
            )
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Configure current normalization",
                "Enable normalization for at least one group or ungrouped coil for every "
                "candidate, or turn current normalization off.",
            )
            return
        self.accept()

    def selected_assignments(self) -> dict[str, dict[str, str]]:
        selected: dict[str, dict[str, str]] = {
            candidate_id: {} for candidate_id in self.candidate_ids
        }
        for candidate_id in self.candidate_ids:
            for group in self._groups.get(candidate_id, {}).values():
                relationship = (
                    "fixed" if group.get("mode") == "fixed" else str(group.get("name"))
                )
                for coil_id in group.get("coil_ids", []):
                    selected[candidate_id][str(coil_id)] = relationship
            for coil_id in self._coil_order.get(candidate_id, []):
                if coil_id not in selected[candidate_id]:
                    selected[candidate_id][coil_id] = self._ungrouped_modes.get(
                        candidate_id, {}
                    ).get(coil_id, "fixed")
        return selected

    def selected_coils(self) -> dict[str, list[str]]:
        """Compatibility helper returning every non-fixed coil."""
        return {
            candidate_id: [
                coil_id
                for coil_id, relationship in assignments.items()
                if relationship != "fixed"
            ]
            for candidate_id, assignments in self.selected_assignments().items()
        }

    def current_mode(self) -> str:
        """Visual fit groups map onto the grouped current-normalization backend."""
        return "grouped"

class SnapshotComparisonDialog(QDialog):
    """Benchmark many snapshots or match candidates to one reference exposure."""

    def __init__(
        self,
        adapter: StudioAdapter,
        snapshot_ids: list[str] | str,
        snapshot_b_id: str | None = None,
        parent=None,
        *,
        initial_mode: str | None = None,
        initial_reference_id: str | None = None,
        initial_benchmark_target_uT: float | None = None,
        initial_benchmark_tolerance_pct: float | None = None,
        initial_match_magnitude_tolerance_pct: float | None = None,
        initial_match_direction_tolerance_deg: float | None = None,
        initial_match_alignment_mode: str | None = None,
    ):
        # Preserve the old two-id constructor for external callers and saved tests.
        if parent is None and snapshot_b_id is not None and not isinstance(snapshot_b_id, str):
            parent = snapshot_b_id
            snapshot_b_id = None
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        if isinstance(snapshot_ids, str):
            ids = [snapshot_ids]
            if snapshot_b_id is not None:
                ids.append(str(snapshot_b_id))
        else:
            ids = list(snapshot_ids)
        self.snapshot_ids = list(dict.fromkeys(map(str, ids)))
        if len(self.snapshot_ids) < 1:
            raise ValueError("Snapshot comparison requires at least one snapshot.")
        self._match_mode_available = len(self.snapshot_ids) >= 2
        self._plot_widgets: list[PlotView] = []
        self._normalization_assignments: dict[str, dict[str, str]] = {}
        self.comparison: dict | None = None
        self.setWindowTitle("Compare field snapshots")
        self.resize(1320, 860)

        root = QVBoxLayout(self)
        heading = QLabel("Field snapshot comparison")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Question"))
        self.comparison_mode = QComboBox()
        self.comparison_mode.addItem("Performance benchmark — against a shared target", "benchmark")
        if self._match_mode_available:
            self.comparison_mode.addItem("Exposure matching — against a reference snapshot", "match")
        self.comparison_mode.setMinimumWidth(440)
        self.comparison_mode.currentIndexChanged.connect(self._comparison_mode_changed)
        mode_row.addWidget(self.comparison_mode)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        self.benchmark_controls = QWidget()
        benchmark_row = QHBoxLayout(self.benchmark_controls)
        benchmark_row.setContentsMargins(0, 0, 0, 0)
        benchmark_row.addWidget(QLabel("Shared target"))
        first_stats = adapter.snapshot_result(self.snapshot_ids[0]).get("stats", {})
        try:
            initial_target = float(first_stats.get("target_uT", 200.0))
        except (TypeError, ValueError):
            initial_target = 200.0
        if not math.isfinite(initial_target) or initial_target <= 0:
            initial_target = 200.0
        try:
            initial_tolerance = float(first_stats.get("tolerance_pct", 5.0))
        except (TypeError, ValueError):
            initial_tolerance = 5.0
        if not math.isfinite(initial_tolerance) or initial_tolerance < 0:
            initial_tolerance = 5.0
        self.benchmark_target = _spin(
            initial_target, 1e-9, 1e9, 6, " µT"
        )
        self.benchmark_target.valueChanged.connect(self.refresh_comparison)
        benchmark_row.addWidget(self.benchmark_target)
        benchmark_row.addWidget(QLabel("Tolerance"))
        self.benchmark_tolerance = _spin(
            initial_tolerance, 0.0, 1000.0, 4, " %"
        )
        self.benchmark_tolerance.valueChanged.connect(self.refresh_comparison)
        benchmark_row.addWidget(self.benchmark_tolerance)
        benchmark_row.addStretch(1)
        root.addWidget(self.benchmark_controls)

        self.match_controls = QWidget()
        match_row = QHBoxLayout(self.match_controls)
        match_row.setContentsMargins(0, 0, 0, 0)
        match_row.addWidget(QLabel("Reference"))
        self.reference = QComboBox()
        summaries = {str(item.get("id")): item for item in adapter.list_snapshots()}
        for snapshot_id in self.snapshot_ids:
            self.reference.addItem(
                str(summaries.get(snapshot_id, {}).get("name", snapshot_id)), snapshot_id
            )
        self.reference.currentIndexChanged.connect(self._reference_changed)
        self.reference.setMinimumWidth(230)
        match_row.addWidget(self.reference)
        match_row.addWidget(QLabel("Magnitude ±"))
        self.match_magnitude_tolerance = _spin(5.0, 0.0, 1000.0, 3, " %")
        self.match_magnitude_tolerance.valueChanged.connect(self.refresh_comparison)
        match_row.addWidget(self.match_magnitude_tolerance)
        match_row.addWidget(QLabel("Direction ≤"))
        self.match_direction_tolerance = _spin(3.0, 0.0, 180.0, 3, "°")
        self.match_direction_tolerance.valueChanged.connect(self.refresh_comparison)
        match_row.addWidget(self.match_direction_tolerance)
        match_row.addWidget(QLabel("Orientation"))
        self.match_alignment = QComboBox()
        self.match_alignment.addItem("Fixed", "fixed")
        self.match_alignment.addItem("Exposure", "exposure")
        self.match_alignment.currentIndexChanged.connect(self.refresh_comparison)
        match_row.addWidget(self.match_alignment)
        self.normalize_current = QCheckBox("Normalize current")
        self.normalize_current.toggled.connect(self._normalization_toggled)
        match_row.addWidget(self.normalize_current)
        self.configure_currents = QPushButton("Choose coils…")
        self.configure_currents.setEnabled(False)
        self.configure_currents.clicked.connect(self.configure_current_normalization)
        match_row.addWidget(self.configure_currents)
        match_row.addStretch(1)
        root.addWidget(self.match_controls)

        self.method = QLabel()
        self.method.setWordWrap(True)
        self.method.setObjectName("hint")
        root.addWidget(self.method)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setObjectName("warningPanelCompact")
        self.warning.hide()
        root.addWidget(self.warning)

        self.detail_candidate_row = QWidget()
        detail_row = QHBoxLayout(self.detail_candidate_row)
        detail_row.setContentsMargins(0, 0, 0, 0)
        detail_row.addWidget(QLabel("Detailed candidate"))
        self.detail_candidate = QComboBox()
        self.detail_candidate.currentIndexChanged.connect(self._rebuild_tabs_from_cached)
        self.detail_candidate.setMinimumWidth(260)
        detail_row.addWidget(self.detail_candidate)
        detail_row.addStretch(1)
        root.addWidget(self.detail_candidate_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if initial_benchmark_target_uT is not None:
            self.benchmark_target.setValue(float(initial_benchmark_target_uT))
        if initial_benchmark_tolerance_pct is not None:
            self.benchmark_tolerance.setValue(float(initial_benchmark_tolerance_pct))
        if initial_match_magnitude_tolerance_pct is not None:
            self.match_magnitude_tolerance.setValue(float(initial_match_magnitude_tolerance_pct))
        if initial_match_direction_tolerance_deg is not None:
            self.match_direction_tolerance.setValue(float(initial_match_direction_tolerance_deg))
        if initial_match_alignment_mode is not None:
            alignment_index = self.match_alignment.findData(str(initial_match_alignment_mode))
            if alignment_index >= 0:
                self.match_alignment.setCurrentIndex(alignment_index)
        if initial_reference_id is not None:
            reference_index = self.reference.findData(str(initial_reference_id))
            if reference_index >= 0:
                self.reference.setCurrentIndex(reference_index)
        requested_mode = str(initial_mode or "").strip().lower()
        mode_index = self.comparison_mode.findData(requested_mode) if requested_mode else -1
        if mode_index >= 0:
            self.comparison_mode.setCurrentIndex(mode_index)
        self._comparison_mode_changed()

    @staticmethod
    def _format(value, unit: str = "", digits: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(number):
            return "—"
        return f"{number:,.{digits}g}{(' ' + unit) if unit else ''}"

    def _card(self, title: str, value: str, detail: str) -> QGroupBox:
        card = QGroupBox(title)
        layout = QVBoxLayout(card)
        number = _selectable_label(value, word_wrap=True)
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setObjectName("resultValue")
        layout.addWidget(number)
        note = _selectable_label(detail, word_wrap=True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setObjectName("hint")
        layout.addWidget(note)
        return card

    @staticmethod
    def _benchmark_quality_colour(value_pct: float, *, higher_is_better: bool) -> str:
        """Return a red -> amber -> green border colour for a 0-100% metric.

        Performance overview cards use the metric's natural 0-100 scale.  Coverage
        and directional consistency improve as the value rises, while uniformity
        spread improves as it falls, so the latter simply reverses the scale.
        Values outside 0-100 are clipped for colour purposes only.
        """
        try:
            value = float(value_pct)
        except (TypeError, ValueError):
            value = 50.0
        if not math.isfinite(value):
            value = 50.0
        value = min(100.0, max(0.0, value))
        quality = value if higher_is_better else 100.0 - value

        red = (220, 38, 38)
        amber = (245, 158, 11)
        green = (22, 163, 74)
        if quality <= 50.0:
            fraction = quality / 50.0
            start, stop = red, amber
        else:
            fraction = (quality - 50.0) / 50.0
            start, stop = amber, green
        rgb = tuple(
            int(round(a + (b - a) * fraction))
            for a, b in zip(start, stop)
        )
        return "#%02x%02x%02x" % rgb

    def _benchmark_card(
        self,
        title: str,
        value_pct: float,
        detail: str,
        *,
        higher_is_better: bool,
    ) -> QGroupBox:
        """Create one performance headline card with a quality-coloured border."""
        card = self._card(title, self._format(value_pct, "%"), detail)
        colour = self._benchmark_quality_colour(
            value_pct, higher_is_better=higher_is_better
        )
        # Keep the normal Workbench card styling and only strengthen/recolour its
        # border.  This remains local to performance-overview headline cards.
        card.setStyleSheet(f"QGroupBox {{ border: 2px solid {colour}; }}")
        return card

    @staticmethod
    def _benchmark_overall_score(entry: dict) -> float:
        """Return the simple 0-100 average used by the benchmark overview.

        Coverage and directional consistency already use a higher-is-better
        0-100 scale.  Uniformity spread is inverted so zero spread contributes
        100 points and a spread of 100% or more contributes zero points.
        Components are clipped only for this presentation score; the underlying
        benchmark statistics remain unchanged.
        """
        def clipped(value: object) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            return min(100.0, max(0.0, number))

        coverage = clipped(entry.get("target_coverage_pct"))
        uniformity = 100.0 - clipped(entry.get("uniformity_spread_pct"))
        direction = clipped(entry.get("directional_consistency_pct"))
        return (coverage + uniformity + direction) / 3.0

    def _benchmark_overall_card(self, entry: dict) -> QGroupBox:
        """Create the larger aggregate benchmark score card."""
        score = self._benchmark_overall_score(entry)
        colour = self._benchmark_quality_colour(score, higher_is_better=True)
        card = self._card(
            "Overall score",
            self._format(score, "%"),
            "Average of target coverage, inverse uniformity spread, and "
            "directional consistency",
        )
        card.setMinimumHeight(132)
        card.setStyleSheet(f"QGroupBox {{ border: 3px solid {colour}; }}")
        value_label = card.findChild(QLabel, "resultValue")
        if value_label is not None:
            value_label.setStyleSheet(
                f"font-size: 24pt; font-weight: 700; color: {colour};"
            )
        return card

    @staticmethod
    def _match_overall_score(stats: dict) -> float:
        """Return a simple 0-100 exposure-match score for the overview.

        Mean-intensity bias and relative vector RMS error are both error metrics,
        so their absolute/clipped percentages are inverted.  Point-match coverage
        is already a higher-is-better percentage.  This presentation score does not
        alter any of the underlying comparison statistics.
        """
        def clipped(value: object) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            return min(100.0, max(0.0, number))

        def inverse_error(value: object, *, absolute: bool = False) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(number):
                return 0.0
            if absolute:
                number = abs(number)
            return 100.0 - min(100.0, max(0.0, number))

        intensity_match = inverse_error(
            stats.get("mean_intensity_bias_pct"), absolute=True
        )
        vector_match = inverse_error(
            stats.get("relative_rms_vector_difference_pct")
        )
        point_match = clipped(stats.get("match_coverage_pct"))
        return (intensity_match + vector_match + point_match) / 3.0

    def _match_metric_card(
        self,
        title: str,
        display_value: str,
        detail: str,
        quality_value_pct: float,
        *,
        higher_is_better: bool,
    ) -> QGroupBox:
        """Create one exposure-match headline card with a quality border."""
        card = self._card(title, display_value, detail)
        colour = self._benchmark_quality_colour(
            quality_value_pct, higher_is_better=higher_is_better
        )
        card.setStyleSheet(f"QGroupBox {{ border: 2px solid {colour}; }}")
        return card

    def _match_overall_card(self, stats: dict) -> QGroupBox:
        """Create the larger aggregate exposure-match score card."""
        score = self._match_overall_score(stats)
        colour = self._benchmark_quality_colour(score, higher_is_better=True)
        card = self._card(
            "Overall score",
            self._format(score, "%"),
            "Average of mean-intensity match, inverse vector RMS error, and "
            "points matching",
        )
        card.setMinimumHeight(132)
        card.setStyleSheet(f"QGroupBox {{ border: 3px solid {colour}; }}")
        value_label = card.findChild(QLabel, "resultValue")
        if value_label is not None:
            value_label.setStyleSheet(
                f"font-size: 24pt; font-weight: 700; color: {colour};"
            )
        return card

    def _clear_tabs(self) -> None:
        for plot in self._plot_widgets:
            plot.cleanup()
        self._plot_widgets.clear()
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()

    def _comparison_mode_changed(self) -> None:
        benchmark = str(self.comparison_mode.currentData()) == "benchmark"
        self.benchmark_controls.setVisible(benchmark)
        self.match_controls.setVisible(not benchmark)
        self.detail_candidate_row.setVisible(not benchmark)
        self.refresh_comparison()

    def _reference_changed(self) -> None:
        self._normalization_assignments = {}
        self.refresh_comparison()

    def _normalization_toggled(self, enabled: bool) -> None:
        self.configure_currents.setEnabled(bool(enabled))
        self.refresh_comparison()

    def _candidate_ids(self) -> list[str]:
        reference_id = str(self.reference.currentData() or self.snapshot_ids[0])
        return [snapshot_id for snapshot_id in self.snapshot_ids if snapshot_id != reference_id]

    def configure_current_normalization(self) -> None:
        dialog = CurrentNormalizationDialog(
            self.adapter,
            self._candidate_ids(),
            selected_assignments=self._normalization_assignments,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._normalization_assignments = dialog.selected_assignments()
        self.refresh_comparison()

    def refresh_comparison(self) -> None:
        mode = str(self.comparison_mode.currentData() or "benchmark")
        try:
            if mode == "benchmark":
                comparison = self.adapter.benchmark_snapshots(
                    self.snapshot_ids,
                    target_uT=self.benchmark_target.value(),
                    tolerance_pct=self.benchmark_tolerance.value(),
                )
            else:
                reference_id = str(self.reference.currentData() or self.snapshot_ids[0])
                comparison = self.adapter.match_snapshot_exposures(
                    reference_id,
                    self._candidate_ids(),
                    magnitude_tolerance_pct=self.match_magnitude_tolerance.value(),
                    direction_tolerance_deg=self.match_direction_tolerance.value(),
                    alignment_mode=str(self.match_alignment.currentData() or "fixed"),
                    normalize_current=self.normalize_current.isChecked(),
                    normalization_assignments=self._normalization_assignments,
                )
        except Exception as error:  # noqa: BLE001 - stored project boundary
            self.comparison = None
            self._clear_tabs()
            self.warning.setText(str(error))
            self.warning.show()
            self.method.setText(
                "Adjust the comparison settings or choose compatible snapshots. Stored snapshot "
                "values have not been changed."
            )
            return
        self.comparison = comparison
        self.warning.hide()
        if mode == "benchmark":
            self.method.setText(
                "Each snapshot is judged independently against the same target and tolerance. "
                "Sample grids may differ because no point-to-point correspondence is assumed."
            )
        else:
            normalization_text = (
                " Candidate coil currents are fitted to minimize pointwise field-magnitude error "
                "using the configured fixed, independent, and shared-current groups. Direction is "
                "not used in the current fit; saved field-scale factors and coil polarity are "
                "preserved."
                if comparison["normalize_current"]
                else " Candidate currents are used exactly as captured."
            )
            alignment_text = (
                " A single best-fit rigid 3-D rotation of each candidate vector field is removed before scoring."
                if str(comparison.get("alignment_mode", "fixed")) == "exposure"
                else " Absolute target-frame orientation is preserved."
            )
            self.method.setText(
                "Candidates are compared point-by-point to the reference in the measurement "
                "objects' local coordinate frame." + alignment_text + normalization_text
            )
        self._populate_detail_candidate(comparison)
        self._rebuild_tabs(comparison)

    def _populate_detail_candidate(self, comparison: dict) -> None:
        if comparison.get("mode") != "match":
            return
        previous = str(self.detail_candidate.currentData() or "")
        self.detail_candidate.blockSignals(True)
        self.detail_candidate.clear()
        for candidate in comparison["candidates"]:
            self.detail_candidate.addItem(candidate["name"], candidate["snapshot_id"])
        index = self.detail_candidate.findData(previous)
        self.detail_candidate.setCurrentIndex(index if index >= 0 else 0)
        self.detail_candidate.blockSignals(False)

    def _rebuild_tabs_from_cached(self) -> None:
        if self.comparison is not None:
            self._rebuild_tabs(self.comparison)

    def _rebuild_tabs(self, comparison: dict) -> None:
        self._clear_tabs()
        if comparison["mode"] == "benchmark":
            self._build_benchmark_overview(comparison)
            self._build_benchmark_statistics(comparison)
            self._add_plot_tab(
                "Distributions", self._benchmark_distribution_figure(comparison)
            )
        else:
            self._build_match_overview(comparison)
            self._build_match_statistics(comparison)
            candidate = self._selected_candidate(comparison)
            if candidate is not None:
                pair = candidate["comparison"]
                self._add_plot_tab(
                    "Central slices", self.adapter.comparison_slice_figure(pair)
                )
                self._add_plot_tab(
                    "3D magnitude difference",
                    self.adapter.comparison_difference_figure(pair),
                )
                self._add_plot_tab(
                    "Direction difference",
                    self.adapter.comparison_direction_figure(pair),
                )
                self._add_plot_tab(
                    "Distributions", self.adapter.comparison_distribution_figure(pair)
                )
        _make_result_text_selectable(self)

    def _selected_candidate(self, comparison: dict) -> dict | None:
        selected_id = str(self.detail_candidate.currentData() or "")
        return next(
            (
                item
                for item in comparison.get("candidates", [])
                if str(item.get("snapshot_id")) == selected_id
            ),
            comparison.get("candidates", [None])[0]
            if comparison.get("candidates")
            else None,
        )

    def _horizontal_columns_page(self, columns: list[QWidget]) -> QScrollArea:
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for column in columns:
            column.setMinimumWidth(285)
            column.setMaximumWidth(340)
            layout.addWidget(column)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_benchmark_overview(self, comparison: dict) -> None:
        columns: list[QWidget] = []
        for entry in comparison["entries"]:
            column = QGroupBox(entry["name"])
            layout = QVBoxLayout(column)
            source = QLabel(entry["source_label"])
            source.setWordWrap(True)
            source.setObjectName("hint")
            layout.addWidget(source)
            layout.addWidget(
                self._benchmark_card(
                    "Target coverage",
                    entry["target_coverage_pct"],
                    f"{self._format(comparison['target_uT'], 'µT')} ± "
                    f"{self._format(comparison['tolerance_pct'], '%')}",
                    higher_is_better=True,
                )
            )
            layout.addWidget(
                self._benchmark_card(
                    "Uniformity spread",
                    entry["uniformity_spread_pct"],
                    "100 × (P95 − P5) / mean; lower is better",
                    higher_is_better=False,
                )
            )
            layout.addWidget(
                self._benchmark_card(
                    "Directional consistency",
                    entry["directional_consistency_pct"],
                    f"Mean field {self._format(entry['mean_uT'], 'µT')}",
                    higher_is_better=True,
                )
            )
            layout.addSpacing(12)
            layout.addWidget(self._benchmark_overall_card(entry))
            layout.addStretch(1)
            columns.append(column)
        self.tabs.addTab(self._horizontal_columns_page(columns), "Overview")

    def _build_benchmark_statistics(self, comparison: dict) -> None:
        columns: list[QWidget] = []
        for entry in comparison["entries"]:
            column = QWidget()
            layout = QVBoxLayout(column)
            layout.addWidget(
                _copyable_statistics_group(
                    entry["name"],
                    [
                        ("Status", entry["status"].capitalize()),
                        ("Samples", f"{entry['sample_count']:,}"),
                        ("Mean", self._format(entry["mean_uT"], "µT")),
                        ("Median", self._format(entry["median_uT"], "µT")),
                        ("Minimum", self._format(entry["minimum_uT"], "µT")),
                        ("Maximum", self._format(entry["maximum_uT"], "µT")),
                        ("Target coverage", self._format(entry["target_coverage_pct"], "%")),
                        ("Below target", self._format(entry["below_pct"], "%")),
                        ("Above target", self._format(entry["above_pct"], "%")),
                        ("RMS target error", self._format(entry["rms_error_uT"], "µT")),
                        ("Uniformity spread", self._format(entry["uniformity_spread_pct"], "%")),
                        ("Directional consistency", self._format(entry["directional_consistency_pct"], "%")),
                    ],
                )
            )
            layout.addStretch(1)
            columns.append(column)
        self.tabs.addTab(self._horizontal_columns_page(columns), "Full statistics")

    @staticmethod
    def _benchmark_distribution_figure(comparison: dict) -> dict:
        traces = []
        for entry in comparison["entries"]:
            traces.append(
                {
                    "type": "histogram",
                    "x": np.asarray(entry["magnitudes_uT"], dtype=float).tolist(),
                    "name": entry["name"],
                    "opacity": 0.55,
                    "nbinsx": 36,
                }
            )
        return {
            "data": traces,
            "layout": {
                "template": "plotly_white",
                "barmode": "overlay",
                "margin": {"l": 60, "r": 25, "t": 40, "b": 55},
                "title": {"text": "Field-magnitude distributions", "x": 0.5},
                "xaxis": {"title": {"text": "|B| (µT)"}},
                "yaxis": {"title": {"text": "Sample count"}},
                "shapes": [
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": comparison["target_low_uT"],
                        "x1": comparison["target_high_uT"],
                        "y0": 0,
                        "y1": 1,
                        "fillcolor": "rgba(22,163,74,0.12)",
                        "line": {"width": 0},
                        "layer": "below",
                    }
                ],
                "uirevision": "snapshot-benchmark-distributions",
            },
        }

    def _normalization_summary(self, candidate: dict) -> tuple[str, str]:
        normalization = candidate.get("normalization")
        if not isinstance(normalization, dict):
            return "As captured", "Saved snapshot coil currents unchanged"
        labels = normalization.get("coil_labels", [])
        original = normalization.get("original_currents_a", [])
        fitted = normalization.get("fitted_currents_a", [])
        fit_groups = normalization.get("fit_groups", [])
        if isinstance(fit_groups, list) and fit_groups:
            if len(fit_groups) == 1:
                group = fit_groups[0]
                headline = self._format(
                    group.get("fitted_current_magnitude_a"), "A"
                )
                member_count = len(group.get("coil_ids", []))
                detail = str(group.get("label", "Shared current"))
                if member_count > 1:
                    detail += f" shared by {member_count} coils"
                return headline, detail
            pairs = [
                f"{group.get('label', 'Group')}: "
                f"{float(group.get('fitted_current_magnitude_a', 0.0)):.5g} A"
                for group in fit_groups
            ]
            headline = f"{len(fit_groups)} fitted current groups"
            detail = " • ".join(pairs[:3])
            if len(pairs) > 3:
                detail += f" • +{len(pairs) - 3} more"
            return headline, detail
        if normalization.get("current_mode") == "common" and fitted:
            headline = self._format(abs(float(fitted[0])), "A")
            detail = "Common current magnitude for " + str(len(fitted)) + " selected coil"
            if len(fitted) != 1:
                detail += "s"
            return headline, detail
        pairs = [
            f"{label}: {float(value):.5g} A"
            for label, value in zip(labels, fitted)
        ]
        headline = "Independent currents"
        detail = " • ".join(pairs[:3])
        if len(pairs) > 3:
            detail += f" • +{len(pairs) - 3} more"
        if original and not detail:
            detail = "Fitted from saved coil currents"
        return headline, detail

    def _build_match_overview(self, comparison: dict) -> None:
        columns: list[QWidget] = []
        reference_mean = float(comparison.get("reference_mean_uT", math.nan))
        if not math.isfinite(reference_mean) and comparison.get("candidates"):
            reference_mean = float(
                comparison["candidates"][0].get("stats", {}).get("mean_reference_uT", math.nan)
            )

        reference = QGroupBox(comparison["reference_name"])
        reference_layout = QVBoxLayout(reference)
        reference_layout.addWidget(QLabel("REFERENCE EXPOSURE"))
        reference_source = QLabel(comparison["reference_source_label"])
        reference_source.setWordWrap(True)
        reference_source.setObjectName("hint")
        reference_layout.addWidget(reference_source)
        reference_mean_card = self._card(
            "Mean intensity",
            self._format(reference_mean, "µT"),
            "Reference mean intensity",
        )
        reference_mean_card.setStyleSheet(
            "QGroupBox { border: 2px solid #16a34a; }"
        )
        reference_layout.addWidget(reference_mean_card)
        reference_layout.addWidget(
            self._card(
                "Match tolerances",
                f"±{self._format(comparison['magnitude_tolerance_pct'], '%')}",
                f"Direction ≤ {self._format(comparison['direction_tolerance_deg'], '°')}",
            )
        )
        reference_layout.addStretch(1)
        columns.append(reference)

        for candidate in comparison["candidates"]:
            stats = candidate["stats"]
            current_value, current_detail = self._normalization_summary(candidate)
            column = QGroupBox(candidate["name"])
            layout = QVBoxLayout(column)
            source = QLabel(candidate["source_label"])
            source.setWordWrap(True)
            source.setObjectName("hint")
            layout.addWidget(source)

            bias = float(stats["mean_intensity_bias_pct"])
            bias_prefix = "+" if math.isfinite(bias) and bias > 0 else ""
            layout.addWidget(
                self._match_metric_card(
                    "Mean intensity",
                    self._format(stats["mean_candidate_uT"], "µT"),
                    f"{bias_prefix}{self._format(bias, '%')} vs reference",
                    abs(bias),
                    higher_is_better=False,
                )
            )
            layout.addWidget(
                self._match_metric_card(
                    "Relative vector RMS error",
                    self._format(stats["relative_rms_vector_difference_pct"], "%"),
                    f"{self._format(stats['rms_vector_difference_uT'], 'µT')} vector RMS",
                    stats["relative_rms_vector_difference_pct"],
                    higher_is_better=False,
                )
            )
            layout.addWidget(
                self._match_metric_card(
                    "Points matching",
                    self._format(stats["match_coverage_pct"], "%"),
                    f"Magnitude ±{self._format(comparison['magnitude_tolerance_pct'], '%')} "
                    f"and direction ≤ {self._format(comparison['direction_tolerance_deg'], '°')}",
                    stats["match_coverage_pct"],
                    higher_is_better=True,
                )
            )
            layout.addWidget(self._match_overall_card(stats))
            layout.addWidget(
                self._card("Current", current_value, current_detail)
            )
            layout.addStretch(1)
            columns.append(column)
        self.tabs.addTab(self._horizontal_columns_page(columns), "Overview")

    def _build_match_statistics(self, comparison: dict) -> None:
        columns: list[QWidget] = []
        for candidate in comparison["candidates"]:
            stats = candidate["stats"]
            column = QWidget()
            layout = QVBoxLayout(column)
            rows = [
                ("Status", candidate["status"].capitalize()),
                ("Corresponding samples", f"{stats['sample_count']:,}"),
                ("Reference mean", self._format(stats["mean_reference_uT"], "µT")),
                ("Candidate mean", self._format(stats["mean_candidate_uT"], "µT")),
                ("Mean intensity bias", self._format(stats["mean_intensity_bias_pct"], "%")),
                ("Relative vector RMS error", self._format(stats["relative_rms_vector_difference_pct"], "%")),
                ("RMS vector difference", self._format(stats["rms_vector_difference_uT"], "µT")),
                ("RMS magnitude difference", self._format(stats["rms_magnitude_difference_uT"], "µT")),
                ("Mean absolute magnitude difference", self._format(stats["mean_absolute_magnitude_difference_uT"], "µT")),
                ("Maximum absolute magnitude difference", self._format(stats["maximum_absolute_magnitude_difference_uT"], "µT")),
                ("Matching points", self._format(stats["match_coverage_pct"], "%")),
                ("Median angular difference", self._format(stats["median_angular_difference_deg"], "°")),
                ("P95 angular difference", self._format(stats["p95_angular_difference_deg"], "°")),
                ("Maximum angular difference", self._format(stats["maximum_angular_difference_deg"], "°")),
                ("Reference uniformity", self._format(stats["uniformity_reference_pct"], "%")),
                ("Candidate uniformity", self._format(stats["uniformity_candidate_pct"], "%")),
            ]
            normalization = candidate.get("normalization")
            if isinstance(normalization, dict):
                mode = str(normalization.get("current_mode", "grouped")).capitalize()
                rows.append(("Current relationship", mode))
                for group in normalization.get("fit_groups", []):
                    members = ", ".join(map(str, group.get("coil_labels", [])))
                    rows.append(
                        (
                            f"Fit group: {group.get('label', 'Group')}",
                            f"{float(group.get('fitted_current_magnitude_a', 0.0)):.7g} A"
                            + (f" — {members}" if members else ""),
                        )
                    )
                for label, before, after in zip(
                    normalization.get("coil_labels", []),
                    normalization.get("original_currents_a", []),
                    normalization.get("fitted_currents_a", []),
                ):
                    rows.append(
                        (
                            f"{label} current",
                            f"{float(before):.7g} A saved → {float(after):.7g} A fitted",
                        )
                    )
            layout.addWidget(_copyable_statistics_group(candidate["name"], rows))
            layout.addStretch(1)
            columns.append(column)
        self.tabs.addTab(self._horizontal_columns_page(columns), "Full statistics")

    def _add_plot_tab(self, title: str, figure: dict) -> None:
        plot = PlotView(image_filename=f"snapshot-comparison-{title}")
        plot.set_figure(figure)
        self._plot_widgets.append(plot)
        self.tabs.addTab(plot, title)

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        for plot in self._plot_widgets:
            plot.cleanup()
        self._plot_widgets.clear()
        super().closeEvent(event)


class SnapshotManagerDialog(QDialog):
    """Inspect and manage field snapshots stored inside the current project."""

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        on_preview: Callable[[dict], None] | None = None,
        on_changed: Callable[[], None] | None = None,
        on_open_result: Callable[[dict], None] | None = None,
        on_open_comparison: Callable[[list[str]], None] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.on_preview = on_preview
        self.on_changed = on_changed
        self.on_open_result = on_open_result
        self.on_open_comparison = on_open_comparison
        self._independent_windows: dict[int, QDialog] = {}
        self.setWindowTitle("Field snapshots")
        self.resize(1100, 540)

        root = QVBoxLayout(self)
        heading = QLabel("Stored field snapshots")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        note = QLabel(
            "Snapshots preserve the exact local/world sample coordinates, Bx/By/Bz values, "
            "settings, and statistics. Stale snapshots are retained for later comparison. "
            "Import snapshots from another saved scene to compare models without merging them. "
            "View, Compare, and Delete apply to every selected snapshot. Comparison supports "
            "any selection of two or more snapshots."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        root.addWidget(note)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            [
                "Name",
                "Measurement volume",
                "Origin",
                "Status",
                "Saved (UTC)",
                "Samples",
                "Mean |B|",
            ]
        )
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(self._view_double_clicked)
        self.tree.setColumnWidth(0, 210)
        self.tree.setColumnWidth(1, 170)
        self.tree.setColumnWidth(2, 170)
        self.tree.setColumnWidth(3, 105)
        self.tree.setColumnWidth(4, 155)
        root.addWidget(self.tree, 1)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("refreshSnapshotsButton")
        self.refresh_button.setToolTip("Reload snapshots stored in the current scene")
        self.refresh_button.clicked.connect(
            lambda _checked=False: self.refresh(preserve_selection=True)
        )
        self.import_button = QPushButton("Import from scene…")
        self.import_button.setToolTip(
            "Copy one or more stored field snapshots from another .magpy.json scene"
        )
        self.import_button.clicked.connect(self.import_from_scene)
        self.view_button = QPushButton("View results")
        self.view_button.setObjectName("primaryButton")
        self.view_button.setToolTip(
            "Open one independent result window for every selected snapshot"
        )
        self.view_button.clicked.connect(self.view_selected)
        self.compare_button = QPushButton("Compare selected…")
        self.compare_button.setObjectName("accentButton")
        self.compare_button.clicked.connect(self.compare_selected)
        self.rename_button = QPushButton("Rename…")
        self.rename_button.clicked.connect(self.rename_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.setToolTip(
            "Delete every selected snapshot as one undoable operation"
        )
        self.delete_button.clicked.connect(self.delete_selected)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.import_button)
        actions.addWidget(self.view_button)
        actions.addWidget(self.compare_button)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.delete_button)
        actions.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        root.addLayout(actions)
        self._last_catalog_signature: tuple | None = None
        self.refresh()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.refresh_if_changed)
        self.refresh_timer.start()

    def _show_independent_window(self, dialog: QDialog) -> None:
        """Keep a normal top-level result alive without blocking this manager."""
        dialog.setParent(None, Qt.WindowType.Window)
        enable_standard_window_controls(dialog)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        key = id(dialog)
        self._independent_windows[key] = dialog
        dialog.destroyed.connect(
            lambda _object=None, window_key=key: self._independent_windows.pop(
                window_key, None
            )
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def _catalog_signature(snapshots: list[dict]) -> tuple:
        """Return a compact signature for refresh-on-change polling."""
        signature = []
        for snapshot in snapshots:
            stats = snapshot.get("stats", {})
            origin = snapshot.get("imported_from")
            signature.append(
                (
                    str(snapshot.get("id", "")),
                    str(snapshot.get("name", "")),
                    str(snapshot.get("source_label", "")),
                    str(snapshot.get("source_object_id", "")),
                    str(snapshot.get("status", "")),
                    str(snapshot.get("created_at", "")),
                    int(snapshot.get("sample_count", 0)),
                    str(origin.get("file_name", "")) if isinstance(origin, dict) else "",
                    stats.get("mean_uT") if isinstance(stats, dict) else None,
                )
            )
        return tuple(signature)

    def refresh_if_changed(self) -> bool:
        """Refresh without disturbing selection when the stored catalog changed."""
        snapshots = self.adapter.list_snapshots()
        signature = self._catalog_signature(snapshots)
        if signature == self._last_catalog_signature:
            return False
        self.refresh(preserve_selection=True, snapshots=snapshots)
        return True

    def refresh(
        self,
        select_id: str | None = None,
        *,
        preserve_selection: bool = False,
        snapshots: list[dict] | None = None,
    ) -> None:
        previous_selected = (
            set(self.selected_snapshot_ids()) if preserve_selection and select_id is None else set()
        )
        previous_current = (
            self.selected_snapshot_id() if preserve_selection and select_id is None else None
        )
        snapshots = self.adapter.list_snapshots() if snapshots is None else snapshots
        self.tree.clear()
        status_labels = {
            "current": "Current",
            "stale": "Stale",
            "imported": "Imported",
            "missing": "Source missing",
            "corrupt": "Corrupt",
        }
        status_colours = {
            "current": QColor("#15803d"),
            "stale": QColor("#c2410c"),
            "imported": QColor("#0369a1"),
            "missing": QColor("#b91c1c"),
            "corrupt": QColor("#b91c1c"),
        }
        selected_item = None
        selected_items: list[QTreeWidgetItem] = []
        current_target = str(select_id or previous_current or "")
        for snapshot in snapshots:
            stats = snapshot.get("stats", {})
            created = str(snapshot.get("created_at", "")).replace("T", " ").replace("Z", "")
            if "." in created:
                created = created.split(".", 1)[0]
            mean = stats.get("mean_uT")
            mean_text = f"{float(mean):.6g} µT" if isinstance(mean, (int, float)) else "—"
            status = str(snapshot.get("status", "corrupt"))
            origin = snapshot.get("imported_from")
            origin_text = (
                str(origin.get("file_name", "Imported scene"))
                if isinstance(origin, dict)
                else "Current scene"
            )
            item = QTreeWidgetItem(
                [
                    str(snapshot.get("name", snapshot.get("id", "Snapshot"))),
                    str(snapshot.get("source_label", snapshot.get("source_object_id", "—"))),
                    origin_text,
                    status_labels.get(status, status.capitalize()),
                    created or "—",
                    f"{int(snapshot.get('sample_count', 0)):,}",
                    mean_text,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(snapshot.get("id", "")))
            item.setForeground(3, QBrush(status_colours.get(status, QColor("#475569"))))
            if isinstance(origin, dict):
                source_file = str(origin.get("file_name", "another scene"))
                item.setToolTip(2, f"Frozen result imported from {source_file}")
                item.setToolTip(3, f"Frozen result imported from {source_file}")
            self.tree.addTopLevelItem(item)
            snapshot_id = str(snapshot.get("id", ""))
            if snapshot_id == current_target:
                selected_item = item
            if snapshot_id in previous_selected:
                selected_items.append(item)
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
        elif self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        for item in selected_items:
            item.setSelected(True)
        self._last_catalog_signature = self._catalog_signature(snapshots)
        self._update_buttons()

    def selected_snapshot_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def selected_snapshot_ids(self) -> list[str]:
        values = []
        for item in self.tree.selectedItems():
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if value:
                values.append(str(value))
        return list(dict.fromkeys(values))

    def _update_buttons(self) -> None:
        selected = self.selected_snapshot_ids()
        single = len(selected) == 1
        any_selected = bool(selected)
        self.view_button.setEnabled(any_selected)
        self.compare_button.setEnabled(len(selected) >= 2)
        self.rename_button.setEnabled(single)
        self.delete_button.setEnabled(any_selected)

    def import_from_scene(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import field snapshots from scene",
            "",
            "Magpylib scene (*.magpy.json);;JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            catalog = self.adapter.snapshot_catalog_from_file(filename)
        except Exception as error:  # noqa: BLE001 - external project boundary
            QMessageBox.warning(self, "Import field snapshots", str(error))
            return

        available = [item for item in catalog if item.get("importable")]
        if not available:
            details = next(
                (str(item.get("error")) for item in catalog if item.get("error")),
                "No compatible snapshot data was found.",
            )
            QMessageBox.warning(self, "Import field snapshots", details)
            return

        if len(available) == 1:
            selected_ids = [str(available[0]["id"])]
        else:
            choices = [f"All snapshots ({len(available)})"] + [
                f"{item['name']} — {item['source_label']} ({int(item['sample_count']):,} samples)"
                for item in available
            ]
            choice, accepted = QInputDialog.getItem(
                self,
                "Import field snapshots",
                "Choose a snapshot, or import all snapshots from this scene:",
                choices,
                0,
                False,
            )
            if not accepted:
                return
            choice_index = choices.index(choice)
            selected_ids = (
                [str(item["id"]) for item in available]
                if choice_index == 0
                else [str(available[choice_index - 1]["id"])]
            )

        try:
            imported_ids = self.adapter.import_snapshots_from_file(filename, selected_ids)
        except Exception as error:  # noqa: BLE001 - external project boundary
            QMessageBox.warning(self, "Import field snapshots", str(error))
            return
        if self.on_changed is not None:
            self.on_changed()
        self.refresh(imported_ids[-1] if imported_ids else None)
        skipped = len(catalog) - len(available)
        message = f"Imported {len(imported_ids)} field snapshot"
        if len(imported_ids) != 1:
            message += "s"
        if skipped:
            message += f". {skipped} corrupt or unsupported snapshot was skipped"
        QMessageBox.information(self, "Import field snapshots", message + ".")

    def compare_selected(self) -> None:
        selected = self.selected_snapshot_ids()
        if len(selected) < 2:
            return
        if self.on_open_comparison is not None:
            self.on_open_comparison(selected)
            return
        parent = self.parentWidget() or self
        self._show_independent_window(
            SnapshotComparisonDialog(self.adapter, selected, parent=parent)
        )

    def _view_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Open only the row that was double-clicked, even during multi-selection."""
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if value:
            self._view_snapshot_ids([str(value)])

    def _view_snapshot_ids(self, snapshot_ids: list[str]) -> None:
        """Open one independent result window for every supplied snapshot."""
        errors: list[tuple[str, str]] = []
        catalog = {
            str(item.get("id", "")): str(item.get("name", item.get("id", "Snapshot")))
            for item in self.adapter.list_snapshots()
        }
        for snapshot_id in snapshot_ids:
            try:
                result = self.adapter.snapshot_result(snapshot_id)
                if self.on_open_result is not None:
                    self.on_open_result(result)
                else:
                    parent = self.parentWidget() or self
                    self._show_independent_window(
                        MeasurementResultsDialog(
                            self.adapter,
                            result,
                            parent,
                            on_preview=self.on_preview,
                            allow_snapshot=False,
                        )
                    )
            except Exception as error:  # noqa: BLE001 - stored project boundary
                errors.append((catalog.get(snapshot_id, snapshot_id), str(error)))

        if errors:
            shown = errors[:8]
            details = "\n".join(f"• {name}: {message}" for name, message in shown)
            if len(errors) > len(shown):
                details += f"\n• …and {len(errors) - len(shown)} more"
            QMessageBox.warning(
                self,
                "Open field snapshots",
                f"Could not open {len(errors)} selected snapshot"
                f"{'s' if len(errors) != 1 else ''}:\n{details}",
            )

    def view_selected(self) -> None:
        self._view_snapshot_ids(self.selected_snapshot_ids())

    def rename_selected(self) -> None:
        snapshot_id = self.selected_snapshot_id()
        item = self.tree.currentItem()
        if snapshot_id is None or item is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Rename field snapshot",
            "Snapshot name",
            text=item.text(0),
        )
        if not accepted:
            return
        try:
            self.adapter.rename_snapshot(snapshot_id, name)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Rename field snapshot", str(error))
            return
        if self.on_changed is not None:
            self.on_changed()
        self.refresh(snapshot_id)

    def delete_selected(self) -> None:
        snapshot_ids = self.selected_snapshot_ids()
        if not snapshot_ids:
            return
        selected_names = [item.text(0) for item in self.tree.selectedItems()]
        if len(snapshot_ids) == 1:
            prompt = (
                f"Delete ‘{selected_names[0]}’? "
                "This can be restored with Undo after closing this window."
            )
            title = "Delete field snapshot"
        else:
            prompt = (
                f"Delete {len(snapshot_ids)} selected field snapshots? "
                "The complete deletion can be restored with one Undo after closing this window."
            )
            title = "Delete field snapshots"
        answer = QMessageBox.question(
            self,
            title,
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.adapter.delete_snapshots(snapshot_ids)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, title, str(error))
            return
        if self.on_changed is not None:
            self.on_changed()
        self.refresh()


class SceneNotesDialog(QDialog):
    """Small modeless editor for free-form notes stored with one scene."""

    apply_requested = Signal()

    def __init__(self, notes: str = "", parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Scene notes")
        self.resize(680, 480)
        self.setMinimumSize(480, 320)

        root = QVBoxLayout(self)
        heading = QLabel("Scene notes")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        hint = QLabel(
            "Jot down design choices, measurements, build details, or anything else "
            "that should travel with this scene. Notes are saved inside the .magpy.json file."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("sceneNotesEditor")
        self.editor.setPlaceholderText("Notes about this scene or design…")
        self.editor.setPlainText(str(notes))
        root.addWidget(self.editor, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply_requested.emit
        )
        buttons.rejected.connect(self.close)
        root.addWidget(buttons)

    def notes(self) -> str:
        return self.editor.toPlainText()

    def set_notes(self, notes: str) -> None:
        previous = self.editor.blockSignals(True)
        try:
            self.editor.setPlainText(str(notes))
        finally:
            self.editor.blockSignals(previous)


class MeasurementCalibrationDialog(QDialog):
    """Capture real B-field measurements and fit one linked-group scale factor."""

    COL_INDEX = 0
    COL_X = 1
    COL_Y = 2
    COL_Z = 3
    COL_PATH = 4
    COL_MEASURED = 5
    COL_MODEL = 6
    COL_CORRECTED = 7
    COL_RESIDUAL = 8

    def __init__(
        self,
        adapter: StudioAdapter,
        parent=None,
        *,
        initial_coil_ids: list[str] | None = None,
    ):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.initial_coil_ids = set(str(value) for value in (initial_coil_ids or []))
        self.current_dataset_id: str | None = None
        self.current_samples: list[dict[str, Any]] = []
        self.fit_result: dict[str, Any] | None = None
        self._loading = False

        self.setWindowTitle("Measurement calibration")
        self.resize(1180, 780)
        root = QVBoxLayout(self)

        heading = QLabel("Real measurement calibration")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)
        intro = QLabel(
            "Phase 1 fits one absolute field-scale correction to a user-selected coil or linked "
            "coil group. Coordinates come from a Point or linear axis sensor, while measured values can be "
            "typed, pasted, or imported. The fit always evaluates the selected group at 1.000×, "
            "so an existing correction is never compounded."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        dataset_box = QGroupBox("Dataset")
        dataset_form = QFormLayout(dataset_box)
        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_selector = QComboBox()
        self.dataset_selector.setObjectName("measurementDatasetSelector")
        dataset_row_layout.addWidget(self.dataset_selector, 1)
        self.delete_dataset_button = QPushButton("Delete")
        self.delete_dataset_button.setObjectName("deleteMeasurementDatasetButton")
        self.delete_dataset_button.clicked.connect(self.delete_dataset)
        dataset_row_layout.addWidget(self.delete_dataset_button)
        dataset_form.addRow("Saved dataset", dataset_row)

        self.dataset_name = QLineEdit()
        self.dataset_name.setObjectName("measurementDatasetName")
        self.dataset_name.setPlaceholderText("e.g. 4D box axial scan")
        dataset_form.addRow("Name", self.dataset_name)

        self.sensor_selector = QComboBox()
        self.sensor_selector.setObjectName("measurementSensorSelector")
        self.sensor_selector.setToolTip(
            "The sensor supplies the world XYZ coordinates and local measurement axes. "
            "Those coordinates/orientations are copied into the dataset when it is created."
        )
        self.sensor_selector.currentIndexChanged.connect(self._sensor_changed)
        dataset_form.addRow("Coordinate sensor", self.sensor_selector)

        self.output_selector = QComboBox()
        self.output_selector.setObjectName("measurementOutputSelector")
        self.output_selector.addItem("Magnitude |B|", "B")
        self.output_selector.addItem("Local Bx", "Bx")
        self.output_selector.addItem("Local By", "By")
        self.output_selector.addItem("Local Bz", "Bz")
        self.output_selector.currentIndexChanged.connect(self._invalidate_fit_if_input_changed)
        dataset_form.addRow("Measured quantity", self.output_selector)
        convention = QLabel(
            "Enter magnetic flux density in µT using the same RMS / peak / DC convention as the "
            "coil drive values in the scene. Phase 1 fits scale only: the selected group is isolated "
            "by subtracting the same modeled scene with that group switched off."
        )
        convention.setWordWrap(True)
        convention.setObjectName("hint")
        dataset_form.addRow(convention)

        group_box = QGroupBox("Calibration coil group")
        group_layout = QVBoxLayout(group_box)
        self.coil_tree = QTreeWidget()
        self.coil_tree.setObjectName("measurementCalibrationCoilTree")
        self.coil_tree.setHeaderLabels(["Use / coil", "Current correction"])
        self.coil_tree.setRootIsDecorated(False)
        self.coil_tree.setAlternatingRowColors(True)
        self.coil_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.coil_tree.itemChanged.connect(self._coil_selection_changed)
        group_layout.addWidget(self.coil_tree)
        group_hint = QLabel(
            "Every checked coil is treated as one linked magnetic group and receives the same fitted "
            "absolute correction factor when Apply calibration is pressed."
        )
        group_hint.setWordWrap(True)
        group_hint.setObjectName("hint")
        group_layout.addWidget(group_hint)

        table_box = QGroupBox("Measurement samples")
        table_layout = QVBoxLayout(table_box)
        action_row = QHBoxLayout()
        self.add_row_button = QPushButton("Add row")
        self.add_row_button.setObjectName("addMeasurementRowButton")
        self.add_row_button.clicked.connect(self.add_measurement_row)
        action_row.addWidget(self.add_row_button)
        self.remove_rows_button = QPushButton("Remove selected rows")
        self.remove_rows_button.setObjectName("removeMeasurementRowsButton")
        self.remove_rows_button.clicked.connect(self.remove_selected_measurement_rows)
        action_row.addWidget(self.remove_rows_button)
        self.paste_button = QPushButton("Paste measurements")
        self.paste_button.setObjectName("pasteMeasurementValuesButton")
        self.paste_button.clicked.connect(self.paste_measurements)
        action_row.addWidget(self.paste_button)
        self.import_button = QPushButton("Import CSV / text…")
        self.import_button.setObjectName("importMeasurementValuesButton")
        self.import_button.clicked.connect(self.import_measurements)
        action_row.addWidget(self.import_button)
        self.clear_button = QPushButton("Clear measured values")
        self.clear_button.clicked.connect(self.clear_measurements)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)
        table_layout.addLayout(action_row)
        table_hint = QLabel(
            "X/Y/Z and Path cells are editable. Added rows inherit the nearest row's local "
            "measurement orientation; edit their coordinates to place them in the model. Spreadsheet paste "
            "treats every copied numeric cell as one measurement, transposes horizontal rows "
            "into entries, and automatically adds table rows when needed."
        )
        table_hint.setWordWrap(True)
        table_hint.setObjectName("hint")
        table_layout.addWidget(table_hint)

        self.table = QTableWidget(0, 9)
        self.table.setObjectName("measurementCalibrationTable")
        self.table.setHorizontalHeaderLabels(
            [
                "#",
                "X (mm)",
                "Y (mm)",
                "Z (mm)",
                "Path (mm)",
                "Measured (µT)",
                "Model @ 1× (µT)",
                "Corrected (µT)",
                "Residual (µT)",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.itemChanged.connect(self._table_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_MEASURED, QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table, 1)

        fit_box = QGroupBox("Scale-only fit")
        fit_layout = QHBoxLayout(fit_box)
        self.fit_summary = QLabel("No fit calculated yet.")
        self.fit_summary.setObjectName("measurementFitSummary")
        self.fit_summary.setWordWrap(True)
        self.fit_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fit_layout.addWidget(self.fit_summary, 1)
        self.fit_button = QPushButton("Fit calibration")
        self.fit_button.setObjectName("fitMeasurementCalibrationButton")
        self.fit_button.setProperty("workbenchAction", "fitMeasurementCalibration")
        self.fit_button.clicked.connect(self.fit_calibration)
        fit_layout.addWidget(self.fit_button)
        self.save_button = QPushButton("Save dataset")
        self.save_button.setObjectName("saveMeasurementDatasetButton")
        self.save_button.clicked.connect(self.save_dataset)
        fit_layout.addWidget(self.save_button)
        self.apply_button = QPushButton("Apply calibration")
        self.apply_button.setObjectName("applyMeasurementCalibrationButton")
        self.apply_button.setProperty("workbenchAction", "applyMeasurementCalibration")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_calibration)
        fit_layout.addWidget(self.apply_button)

        # Keep the four working sections in a vertical splitter so dense measurement
        # datasets can borrow room from the setup/fit sections without resizing the
        # whole dialog.  Measurement samples intentionally receive the largest
        # initial share because that is normally the section users spend time in.
        self.section_splitter = QSplitter(Qt.Orientation.Vertical)
        self.section_splitter.setObjectName("measurementCalibrationSectionSplitter")
        self.section_splitter.setChildrenCollapsible(False)
        self.section_splitter.setHandleWidth(7)
        self.section_splitter.addWidget(dataset_box)
        self.section_splitter.addWidget(group_box)
        self.section_splitter.addWidget(table_box)
        self.section_splitter.addWidget(fit_box)
        self.section_splitter.setStretchFactor(0, 0)
        self.section_splitter.setStretchFactor(1, 1)
        self.section_splitter.setStretchFactor(2, 5)
        self.section_splitter.setStretchFactor(3, 0)
        self.section_splitter.setSizes([165, 145, 390, 75])
        root.addWidget(self.section_splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._populate_coils()
        self._populate_sensor_selector()
        self._populate_dataset_selector()
        if self.dataset_selector.count():
            self.dataset_selector.setCurrentIndex(0)
        self.dataset_selector.currentIndexChanged.connect(self._dataset_changed)
        self._start_new_dataset()

    @staticmethod
    def _readonly_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _editable_number_item(text: str, *, tooltip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    @staticmethod
    def _format_number(value: Any, digits: int = 7) -> str:
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        return f"{number:.{digits}g}"

    def _populate_coils(self) -> None:
        self._loading = True
        try:
            self.coil_tree.clear()
            candidates = []
            for obj in self.adapter.list_objects():
                if obj.get("derived"):
                    continue
                object_id = str(obj.get("id", ""))
                if not object_id:
                    continue
                try:
                    if not self.adapter.is_coil(object_id):
                        continue
                    props = self.adapter.coil_physical_properties(object_id)
                except Exception:
                    continue
                candidates.append((obj, props))
            only_one = len(candidates) == 1 and not self.initial_coil_ids
            for obj, props in candidates:
                object_id = str(obj["id"])
                label = str(obj.get("label") or object_id)
                item = QTreeWidgetItem(
                    [label, f"{float(props.get('field_scale_factor', 1.0)):.6g} ×"]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, object_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = object_id in self.initial_coil_ids or only_one
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                )
                if not bool(props.get("enabled", True)):
                    item.setToolTip(0, "This coil is currently magnetically disabled in the scene.")
                self.coil_tree.addTopLevelItem(item)
            self.coil_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.coil_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        finally:
            self._loading = False

    def _populate_sensor_selector(self) -> None:
        self._loading = True
        try:
            self.sensor_selector.clear()
            for sensor in self.adapter.measurement_sensor_catalog():
                self.sensor_selector.addItem(sensor["label"], sensor["id"])
        finally:
            self._loading = False

    def _populate_dataset_selector(self, selected_id: str | None = None) -> None:
        self._loading = True
        try:
            self.dataset_selector.clear()
            self.dataset_selector.addItem("New dataset…", None)
            for dataset in self.adapter.list_measurement_datasets():
                self.dataset_selector.addItem(
                    str(dataset.get("label") or dataset.get("id")),
                    str(dataset.get("id")),
                )
            if selected_id is not None:
                index = self.dataset_selector.findData(str(selected_id))
                if index >= 0:
                    self.dataset_selector.setCurrentIndex(index)
        finally:
            self._loading = False
        self.delete_dataset_button.setEnabled(self.dataset_selector.currentData() is not None)

    def _start_new_dataset(self) -> None:
        self._loading = True
        try:
            # Remove any temporary "saved coordinates" entry inserted for a
            # dataset whose original sensor has since been deleted. New datasets
            # must always start from a live scene sensor.
            live_sensors = self.adapter.measurement_sensor_catalog()
            live_ids = {str(item["id"]) for item in live_sensors}
            for index in range(self.sensor_selector.count() - 1, -1, -1):
                if str(self.sensor_selector.itemData(index) or "") not in live_ids:
                    self.sensor_selector.removeItem(index)
            for sensor in live_sensors:
                if self.sensor_selector.findData(sensor["id"]) < 0:
                    self.sensor_selector.addItem(sensor["label"], sensor["id"])
            self.current_dataset_id = None
            self.fit_result = None
            self.dataset_name.clear()
            self.output_selector.setCurrentIndex(max(0, self.output_selector.findData("B")))
            if self.sensor_selector.count() > 0:
                self.sensor_selector.setCurrentIndex(0)
                sensor_id = str(self.sensor_selector.currentData())
                self.current_samples = self.adapter.measurement_sensor_samples(sensor_id)
                sensor_label = self.sensor_selector.currentText()
                self.dataset_name.setText(f"{sensor_label} measurements")
            else:
                self.current_samples = []
                self.dataset_name.setPlaceholderText("Add a Point or linear axis sensor to the scene first")
            self._populate_table(self.current_samples)
            self.fit_summary.setText(
                "No fit calculated yet. Enter measured µT values, select the coil group, then press Fit calibration."
            )
            self.apply_button.setEnabled(False)
            self.delete_dataset_button.setEnabled(False)
        finally:
            self._loading = False
        self._update_fit_enabled()

    def _dataset_changed(self, _index: int) -> None:
        if self._loading:
            return
        dataset_id = self.dataset_selector.currentData()
        if dataset_id is None:
            self._start_new_dataset()
            return
        try:
            dataset = self.adapter.measurement_dataset(str(dataset_id))
        except Exception as error:
            QMessageBox.warning(self, "Measurement dataset", str(error))
            self._populate_dataset_selector()
            return
        self._load_dataset(dataset)

    def _load_dataset(self, dataset: dict[str, Any]) -> None:
        self._loading = True
        try:
            self.current_dataset_id = str(dataset.get("id", "")) or None
            self.dataset_name.setText(str(dataset.get("label", "")))
            output_index = self.output_selector.findData(str(dataset.get("output", "B")))
            self.output_selector.setCurrentIndex(max(0, output_index))

            sensor_id = str(dataset.get("sensor_id", ""))
            sensor_index = self.sensor_selector.findData(sensor_id)
            if sensor_index < 0 and sensor_id:
                self.sensor_selector.insertItem(
                    0,
                    f"{dataset.get('sensor_label') or sensor_id} (saved coordinates)",
                    sensor_id,
                )
                sensor_index = 0
            if sensor_index >= 0:
                self.sensor_selector.setCurrentIndex(sensor_index)

            selected = set(str(value) for value in dataset.get("coil_ids", []))
            for index in range(self.coil_tree.topLevelItemCount()):
                item = self.coil_tree.topLevelItem(index)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if str(item.data(0, Qt.ItemDataRole.UserRole)) in selected
                    else Qt.CheckState.Unchecked,
                )
            self.current_samples = copy.deepcopy(dataset.get("samples", []))
            self._populate_table(self.current_samples)
            self.fit_result = copy.deepcopy(dataset.get("fit"))
            self._show_fit_summary(self.fit_result, applied=dataset.get("applied"))
            factor = None if not self.fit_result else self.fit_result.get("factor")
            try:
                valid_factor = factor is not None and 1e-6 <= float(factor) <= 1e6
            except (TypeError, ValueError):
                valid_factor = False
            self.apply_button.setEnabled(bool(valid_factor))
            self.delete_dataset_button.setEnabled(True)
        finally:
            self._loading = False
        self._update_fit_enabled()

    def _sensor_changed(self, _index: int) -> None:
        if self._loading:
            return
        sensor_id = self.sensor_selector.currentData()
        if sensor_id is None:
            self.current_samples = []
        else:
            try:
                self.current_samples = self.adapter.measurement_sensor_samples(str(sensor_id))
            except Exception as error:
                QMessageBox.warning(self, "Measurement coordinates", str(error))
                return
        self._populate_table(self.current_samples)
        if self.current_dataset_id is None and not self.dataset_name.text().strip():
            self.dataset_name.setText(f"{self.sensor_selector.currentText()} measurements")
        self._invalidate_fit(clear_columns=True)
        self._update_fit_enabled()

    def _populate_table(self, samples: list[dict[str, Any]]) -> None:
        self._loading = True
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(samples))
            for row, sample in enumerate(samples):
                position = sample.get("position_m", [0.0, 0.0, 0.0])
                try:
                    xyz_mm = [1000.0 * float(value) for value in position]
                except (TypeError, ValueError):
                    xyz_mm = [0.0, 0.0, 0.0]
                path = sample.get("path_value_m")
                path_mm = None if path is None else 1000.0 * float(path)
                self.table.setItem(row, self.COL_INDEX, self._readonly_item(str(row)))
                coordinate_tip = (
                    "World coordinate in mm. Editing it changes this measurement row only; "
                    "the row keeps the local-axis orientation copied from its source sensor/row."
                )
                self.table.setItem(
                    row,
                    self.COL_X,
                    self._editable_number_item(self._format_number(xyz_mm[0]), tooltip=coordinate_tip),
                )
                self.table.setItem(
                    row,
                    self.COL_Y,
                    self._editable_number_item(self._format_number(xyz_mm[1]), tooltip=coordinate_tip),
                )
                self.table.setItem(
                    row,
                    self.COL_Z,
                    self._editable_number_item(self._format_number(xyz_mm[2]), tooltip=coordinate_tip),
                )
                self.table.setItem(
                    row,
                    self.COL_PATH,
                    self._editable_number_item(
                        self._format_number(path_mm),
                        tooltip="Optional path/display coordinate in mm. Leave blank if not applicable.",
                    ),
                )
                measured = QTableWidgetItem(self._format_number(sample.get("measured_uT")))
                measured.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, self.COL_MEASURED, measured)
                self.table.setItem(
                    row,
                    self.COL_MODEL,
                    self._readonly_item(self._format_number(sample.get("model_uT"))),
                )
                self.table.setItem(
                    row,
                    self.COL_CORRECTED,
                    self._readonly_item(self._format_number(sample.get("corrected_uT"))),
                )
                self.table.setItem(
                    row,
                    self.COL_RESIDUAL,
                    self._readonly_item(self._format_number(sample.get("residual_uT"))),
                )
        finally:
            self.table.setUpdatesEnabled(True)
            self._loading = False

    def _table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        if item.column() not in {self.COL_X, self.COL_Y, self.COL_Z, self.COL_PATH, self.COL_MEASURED}:
            return
        self._invalidate_fit(clear_columns=True)

    @staticmethod
    def _parse_table_number(text: str, *, unit: str = "") -> float:
        cleaned = str(text).strip()
        if unit:
            cleaned = cleaned.replace(unit, "").strip()
        value = float(cleaned)
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    def _samples_from_table(self) -> list[dict[str, Any]]:
        if len(self.current_samples) != self.table.rowCount():
            raise StudioOperationError("Measurement coordinates no longer match the table.")
        samples = copy.deepcopy(self.current_samples)
        axes = ((self.COL_X, "X"), (self.COL_Y, "Y"), (self.COL_Z, "Z"))
        for row, sample in enumerate(samples):
            position_mm: list[float] = []
            for column, axis_name in axes:
                item = self.table.item(row, column)
                text = "" if item is None else item.text().strip()
                if not text:
                    raise StudioOperationError(
                        f"{axis_name} coordinate on row {row + 1} is blank."
                    )
                try:
                    position_mm.append(self._parse_table_number(text, unit="mm"))
                except ValueError as error:
                    raise StudioOperationError(
                        f"{axis_name} coordinate on row {row + 1} is not a valid finite number."
                    ) from error
            sample["position_m"] = [value / 1000.0 for value in position_mm]
            path_item = self.table.item(row, self.COL_PATH)
            path_text = "" if path_item is None else path_item.text().strip()
            if not path_text:
                sample["path_value_m"] = None
            else:
                try:
                    sample["path_value_m"] = self._parse_table_number(path_text, unit="mm") / 1000.0
                except ValueError as error:
                    raise StudioOperationError(
                        f"Path coordinate on row {row + 1} is not a valid finite number."
                    ) from error
            measured_item = self.table.item(row, self.COL_MEASURED)
            measured_text = "" if measured_item is None else measured_item.text().strip()
            if not measured_text:
                sample["measured_uT"] = None
            else:
                try:
                    sample["measured_uT"] = self._parse_table_number(
                        measured_text.replace("uT", "µT"), unit="µT"
                    )
                except ValueError as error:
                    raise StudioOperationError(
                        f"Measured value on row {row + 1} is not a valid finite number."
                    ) from error
            sample["sample_index"] = row
        return samples

    def _manual_sample_template(self, insert_row: int) -> dict[str, Any]:
        source: dict[str, Any] | None = None
        if self.current_samples:
            source_index = max(0, min(len(self.current_samples) - 1, insert_row - 1))
            source = self.current_samples[source_index]
        else:
            sensor_id = self.sensor_selector.currentData()
            if sensor_id is not None:
                try:
                    sensor_samples = self.adapter.measurement_sensor_samples(str(sensor_id))
                except Exception:
                    sensor_samples = []
                if sensor_samples:
                    source = sensor_samples[0]
        sample = copy.deepcopy(source) if source is not None else {
            "position_m": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3, dtype=float).tolist(),
            "path_value_m": None,
        }
        sample["sample_index"] = int(insert_row)
        for name in ("measured_uT", "model_uT", "corrected_uT", "residual_uT"):
            sample[name] = None
        return sample

    def add_measurement_row(self) -> None:
        try:
            self.current_samples = self._samples_from_table()
        except Exception as error:
            QMessageBox.warning(self, "Add measurement row", str(error))
            return
        selected_rows = sorted({index.row() for index in self.table.selectedIndexes()})
        insert_row = (selected_rows[-1] + 1) if selected_rows else self.table.rowCount()
        self.current_samples.insert(insert_row, self._manual_sample_template(insert_row))
        for row, sample in enumerate(self.current_samples):
            sample["sample_index"] = row
        self._populate_table(self.current_samples)
        self._invalidate_fit(clear_columns=True)
        self._update_fit_enabled()
        if 0 <= insert_row < self.table.rowCount():
            self.table.setCurrentCell(insert_row, self.COL_MEASURED)
            self.table.editItem(self.table.item(insert_row, self.COL_MEASURED))

    def remove_selected_measurement_rows(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        if not rows:
            return
        try:
            self.current_samples = self._samples_from_table()
        except Exception as error:
            QMessageBox.warning(self, "Remove measurement rows", str(error))
            return
        for row in rows:
            if 0 <= row < len(self.current_samples):
                del self.current_samples[row]
        for row, sample in enumerate(self.current_samples):
            sample["sample_index"] = row
        self._populate_table(self.current_samples)
        self._invalidate_fit(clear_columns=True)
        self._update_fit_enabled()

    def _coil_selection_changed(self, _item: QTreeWidgetItem, _column: int) -> None:
        if self._loading:
            return
        self._invalidate_fit(clear_columns=True)
        self._update_fit_enabled()

    def _invalidate_fit_if_input_changed(self, *_args) -> None:
        if not self._loading:
            self._invalidate_fit(clear_columns=True)

    def _invalidate_fit(self, *, clear_columns: bool) -> None:
        self.fit_result = None
        self.apply_button.setEnabled(False)
        self.fit_summary.setText("Inputs changed. Re-run Fit calibration before applying a correction.")
        if clear_columns:
            for sample in self.current_samples:
                sample["model_uT"] = None
                sample["corrected_uT"] = None
                sample["residual_uT"] = None
            self._loading = True
            try:
                for row in range(self.table.rowCount()):
                    for column in (self.COL_MODEL, self.COL_CORRECTED, self.COL_RESIDUAL):
                        item = self.table.item(row, column)
                        if item is None:
                            self.table.setItem(row, column, self._readonly_item(""))
                        else:
                            item.setText("")
            finally:
                self._loading = False

    def _selected_coils(self) -> list[str]:
        selected: list[str] = []
        for index in range(self.coil_tree.topLevelItemCount()):
            item = self.coil_tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                selected.append(str(item.data(0, Qt.ItemDataRole.UserRole)))
        return selected

    def _update_fit_enabled(self) -> None:
        has_rows = bool(self.current_samples)
        can_seed_rows = has_rows or self.sensor_selector.count() > 0
        self.fit_button.setEnabled(bool(has_rows and self._selected_coils()))
        self.save_button.setEnabled(has_rows)
        self.add_row_button.setEnabled(can_seed_rows)
        self.remove_rows_button.setEnabled(has_rows)
        self.paste_button.setEnabled(can_seed_rows)
        self.import_button.setEnabled(can_seed_rows)
        self.clear_button.setEnabled(has_rows)

    def _measured_values(self) -> list[float | None]:
        values: list[float | None] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_MEASURED)
            text = "" if item is None else item.text().strip()
            if not text:
                values.append(None)
                continue
            try:
                value = float(text.replace("µT", "").replace("uT", "").strip())
            except ValueError as error:
                raise StudioOperationError(
                    f"Measured value on row {row + 1} is not a valid number."
                ) from error
            if not math.isfinite(value):
                raise StudioOperationError(
                    f"Measured value on row {row + 1} must be finite."
                )
            values.append(value)
        return values

    @staticmethod
    def _parse_measurement_text(text: str, *, spreadsheet_cells: bool = False) -> list[float]:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        if not lines:
            return []

        if spreadsheet_cells:
            # Clipboard data from spreadsheets is usually tab-delimited. Treat every
            # numeric copied cell as one independent measurement. Flattening row-major
            # naturally turns a horizontal selection into vertical measurement entries.
            values: list[float] = []
            for line in lines:
                separator = next((value for value in ("\t", ",", ";") if value in line), None)
                parts = line.split(separator) if separator else line.split()
                for part in parts:
                    token = part.strip().replace("µT", "").replace("uT", "")
                    if not token:
                        continue
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        values.append(value)
            return values

        if len(lines) == 1 and not any(separator in lines[0] for separator in (",", "\t", ";")):
            tokens = lines[0].split()
            values: list[float] = []
            all_numeric = True
            for token in tokens:
                try:
                    values.append(float(token.replace("µT", "").replace("uT", "")))
                except ValueError:
                    all_numeric = False
                    break
            if all_numeric and values:
                return values
        values = []
        for line in lines:
            separator = next((value for value in ("\t", ",", ";") if value in line), None)
            parts = line.split(separator) if separator else line.split()
            numeric: list[float] = []
            for part in parts:
                token = part.strip().replace("µT", "").replace("uT", "")
                if not token:
                    continue
                try:
                    value = float(token)
                except ValueError:
                    continue
                if math.isfinite(value):
                    numeric.append(value)
            if numeric:
                # File/CSV import keeps the Phase-1 convention that a row may
                # contain coordinates followed by the measured value.
                values.append(numeric[-1])
        return values

    def _paste_values(self, values: list[float]) -> None:
        if not values:
            QMessageBox.information(
                self,
                "Measurement values",
                "No numeric measurement values were found.",
            )
            return
        try:
            self.current_samples = self._samples_from_table()
        except Exception as error:
            QMessageBox.warning(self, "Measurement values", str(error))
            return

        selected_rows = sorted({index.row() for index in self.table.selectedIndexes()})
        start_row = selected_rows[0] if selected_rows else 0
        required_rows = start_row + len(values)
        while len(self.current_samples) < required_rows:
            insert_row = len(self.current_samples)
            self.current_samples.append(self._manual_sample_template(insert_row))

        for offset, value in enumerate(values):
            row = start_row + offset
            self.current_samples[row]["measured_uT"] = float(value)
        for row, sample in enumerate(self.current_samples):
            sample["sample_index"] = row

        self._populate_table(self.current_samples)
        self._invalidate_fit(clear_columns=True)
        self._update_fit_enabled()
        if values:
            self.table.setCurrentCell(start_row, self.COL_MEASURED)

    def paste_measurements(self) -> None:
        self._paste_values(
            self._parse_measurement_text(
                QApplication.clipboard().text(),
                spreadsheet_cells=True,
            )
        )

    def import_measurements(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import measured field values",
            "",
            "Data files (*.csv *.txt *.tsv);;All files (*)",
        )
        if not filename:
            return
        try:
            text = Path(filename).read_text(encoding="utf-8-sig")
        except Exception as error:
            QMessageBox.warning(self, "Import measurements", str(error))
            return
        self._paste_values(self._parse_measurement_text(text))

    def clear_measurements(self) -> None:
        self._loading = True
        try:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, self.COL_MEASURED)
                if item is not None:
                    item.setText("")
        finally:
            self._loading = False
        self._invalidate_fit(clear_columns=True)

    def fit_calibration(self) -> None:
        try:
            samples = self._samples_from_table()
            self.current_samples = copy.deepcopy(samples)
            measured = self._measured_values()
            selected_coils = self._selected_coils()
            if not selected_coils:
                raise StudioOperationError("Select at least one coil in the calibration group.")
            if not any(value is not None for value in measured):
                raise StudioOperationError("Enter at least one measured µT value first.")
        except Exception as error:
            QMessageBox.warning(self, "Measurement calibration", str(error))
            return

        progress = SensorPathProgressDialog(self)
        progress.setWindowTitle("Calculating measurement calibration")
        progress.setLabelText("Evaluating the selected coil group at 1.000× correction…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.fit_button.setEnabled(False)
        try:
            self.fit_result = self.adapter.fit_measurement_calibration(
                sensor_id=str(self.sensor_selector.currentData() or ""),
                coil_ids=selected_coils,
                measured_uT=measured,
                output=str(self.output_selector.currentData()),
                samples=copy.deepcopy(samples),
                progress_callback=progress.update_from_solver,
            )
            predicted = self.fit_result.get("predicted_uT", [])
            corrected = self.fit_result.get("corrected_uT", [])
            residual = self.fit_result.get("residual_uT", [])
            self._loading = True
            try:
                for row in range(self.table.rowCount()):
                    self.table.item(row, self.COL_MODEL).setText(
                        self._format_number(predicted[row] if row < len(predicted) else None)
                    )
                    self.table.item(row, self.COL_CORRECTED).setText(
                        self._format_number(corrected[row] if row < len(corrected) else None)
                    )
                    measured_value = measured[row] if row < len(measured) else None
                    self.table.item(row, self.COL_RESIDUAL).setText(
                        self._format_number(
                            residual[row] if measured_value is not None and row < len(residual) else None
                        )
                    )
            finally:
                self._loading = False
            self._show_fit_summary(self.fit_result)
            factor = float(self.fit_result.get("factor", float("nan")))
            self.apply_button.setEnabled(math.isfinite(factor) and 1e-6 <= factor <= 1e6)
        except FieldCalculationCancelled:
            self.fit_result = None
            self.apply_button.setEnabled(False)
            self.fit_summary.setText("Calibration calculation cancelled.")
        except Exception as error:
            self.fit_result = None
            self.apply_button.setEnabled(False)
            self.fit_summary.setText("Calibration fit failed.")
            QMessageBox.warning(self, "Measurement calibration", str(error))
        finally:
            progress.finish()
            QApplication.restoreOverrideCursor()
            self._update_fit_enabled()

    def _show_fit_summary(
        self,
        fit: dict[str, Any] | None,
        *,
        applied: dict[str, Any] | None = None,
    ) -> None:
        if not fit:
            self.fit_summary.setText("No fit calculated yet.")
            return
        try:
            factor = float(fit.get("factor"))
        except (TypeError, ValueError):
            self.fit_summary.setText("Saved fit metadata is incomplete; re-run the fit.")
            return
        percent = (factor - 1.0) * 100.0
        parts = [f"Factor {factor:.7g} × ({percent:+.3f}%)"]
        if fit.get("rms_before_uT") is not None and fit.get("rms_after_uT") is not None:
            parts.append(
                f"RMS {float(fit['rms_before_uT']):.5g} → {float(fit['rms_after_uT']):.5g} µT"
            )
        if fit.get("max_abs_residual_uT") is not None:
            parts.append(f"max |residual| {float(fit['max_abs_residual_uT']):.5g} µT")
        if fit.get("r_squared") is not None:
            parts.append(f"R² {float(fit['r_squared']):.5g}")
        if factor <= 0.0:
            parts.append("negative factor indicates a polarity/sign mismatch and cannot be applied")
        if applied:
            try:
                parts.append(f"applied at {float(applied.get('factor')):.7g} ×")
            except (TypeError, ValueError):
                parts.append("previously applied")
        self.fit_summary.setText("  •  ".join(parts))

    def _fit_metadata(self) -> dict[str, Any] | None:
        if not self.fit_result:
            return None
        names = (
            "mode",
            "factor",
            "sample_count",
            "rms_before_uT",
            "rms_after_uT",
            "max_abs_residual_uT",
            "r_squared",
            "coil_modeling_method",
        )
        return {name: copy.deepcopy(self.fit_result.get(name)) for name in names}

    def _dataset_payload(self) -> dict[str, Any]:
        measured = self._measured_values()
        selected_coils = self._selected_coils()
        samples = self._samples_from_table()
        self.current_samples = copy.deepcopy(samples)
        predicted = [] if not self.fit_result else list(self.fit_result.get("predicted_uT", []))
        corrected = [] if not self.fit_result else list(self.fit_result.get("corrected_uT", []))
        residual = [] if not self.fit_result else list(self.fit_result.get("residual_uT", []))
        for row, sample in enumerate(samples):
            sample["sample_index"] = row
            sample["measured_uT"] = measured[row]
            if predicted:
                sample["model_uT"] = predicted[row] if row < len(predicted) else None
            if corrected:
                sample["corrected_uT"] = corrected[row] if row < len(corrected) else None
            if residual:
                sample["residual_uT"] = (
                    residual[row]
                    if measured[row] is not None and row < len(residual)
                    else None
                )
            elif measured[row] is None:
                sample["residual_uT"] = None
        label = self.dataset_name.text().strip()
        if not label:
            label = f"{self.sensor_selector.currentText() or 'Measurement'} dataset"
        return {
            "id": self.current_dataset_id or "",
            "label": label,
            "sensor_id": str(self.sensor_selector.currentData() or ""),
            "sensor_label": self.sensor_selector.currentText(),
            "output": str(self.output_selector.currentData()),
            "unit": "uT",
            "amplitude_convention": "same_as_scene",
            "coil_ids": selected_coils,
            "samples": samples,
            "fit": self._fit_metadata(),
        }

    def _save_dataset(self) -> str:
        payload = self._dataset_payload()
        dataset_id = self.adapter.save_measurement_dataset(payload)
        self.current_dataset_id = dataset_id
        self._populate_dataset_selector(selected_id=dataset_id)
        self.delete_dataset_button.setEnabled(True)
        return dataset_id

    def save_dataset(self) -> None:
        try:
            dataset_id = self._save_dataset()
        except Exception as error:
            QMessageBox.warning(self, "Save measurement dataset", str(error))
            return
        self.fit_summary.setText(
            (self.fit_summary.text() + "  •  dataset saved").strip(" •")
        )
        self.current_dataset_id = dataset_id

    def delete_dataset(self) -> None:
        dataset_id = self.dataset_selector.currentData()
        if dataset_id is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete measurement dataset",
            "Delete this saved measurement dataset? This does not change any coil correction already applied.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.adapter.delete_measurement_dataset(str(dataset_id))
        except Exception as error:
            QMessageBox.warning(self, "Delete measurement dataset", str(error))
            return
        self._populate_dataset_selector()
        self.dataset_selector.setCurrentIndex(0)
        self._start_new_dataset()

    def apply_calibration(self) -> None:
        if not self.fit_result:
            QMessageBox.information(
                self,
                "Measurement calibration",
                "Run Fit calibration before applying a correction.",
            )
            return
        selected_coils = self._selected_coils()
        try:
            factor = float(self.fit_result.get("factor"))
            if not math.isfinite(factor) or not 1e-6 <= factor <= 1e6:
                raise StudioOperationError(
                    "The fitted factor is not a positive correction that can be applied."
                )
            dataset_id = self._save_dataset()
            self.adapter.apply_measurement_calibration(selected_coils, factor)
            self.adapter.mark_measurement_dataset_applied(
                dataset_id,
                coil_ids=selected_coils,
                factor=factor,
            )
        except Exception as error:
            QMessageBox.warning(self, "Apply measurement calibration", str(error))
            return
        self.accept()


class SensorPlotDialog(QDialog):
    """Plot field along sensors and their paths in an independent result window."""

    def __init__(self, adapter: StudioAdapter, parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.adapter = adapter
        self.export_rows: list[dict] = []
        self.export_metadata: dict = {}
        self.setWindowTitle("Sensor Path Plot")
        self.resize(900, 650)

        root = QVBoxLayout(self)
        heading = QLabel("Field along sensor paths")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        settings_box = QGroupBox("Plot settings")
        settings_layout = QHBoxLayout(settings_box)
        output_column = QVBoxLayout()
        output_column.addWidget(QLabel("Output"))
        self.output = QComboBox()
        self.output.addItems(["B", "Bx", "By", "Bz", "Bxy", "H", "Hx", "Hy", "Hz"])
        output_column.addWidget(self.output)
        output_column.addStretch(1)
        settings_layout.addLayout(output_column)

        sensor_column = QVBoxLayout()
        sensor_column.addWidget(QLabel("Sensors"))
        self.sensor_choices_widget = QWidget()
        self.sensor_choices_widget.setObjectName("sensorPlotSensorChoices")
        self.sensor_choices_layout = QVBoxLayout(self.sensor_choices_widget)
        self.sensor_choices_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_choices_layout.setSpacing(3)
        sensor_column.addWidget(self.sensor_choices_widget)
        sensor_column.addStretch(1)
        settings_layout.addLayout(sensor_column, 1)
        settings_layout.addStretch(1)

        self.sensor_checks: dict[str, QCheckBox] = {}
        self._sensor_choice_signature: tuple[tuple[str, str], ...] | None = None

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setObjectName("primaryButton")
        self.calculate_button.setProperty("workbenchAction", "calculateSensorPathButton")
        self.calculate_button.setMinimumHeight(40)
        self.calculate_button.clicked.connect(self.calculate)
        settings_layout.addWidget(self.calculate_button)

        self.export_csv_button = QPushButton("Export CSV…")
        self.export_csv_button.setObjectName("exportSensorPathCsvButton")
        self.export_csv_button.setMinimumHeight(40)
        self.export_csv_button.setToolTip(
            "Export every series and sample currently shown in the sensor-path plot"
        )
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setEnabled(False)
        settings_layout.addWidget(self.export_csv_button)
        root.addWidget(settings_box)
        self.plot = PlotView(image_filename="sensor-path-field")
        root.addWidget(self.plot, 1)
        self._refresh_sensor_choices()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        root.addWidget(buttons)
        self.calculation_status = QStatusBar()
        self.calculation_status.setObjectName("sensorPathStatusBar")
        self.calculation_status.setSizeGripEnabled(False)
        self.calculation_status.showMessage("Ready")
        root.addWidget(self.calculation_status)


    def _sensor_records(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.adapter.list_objects()
            if item.get("type") == "Sensor" and not item.get("derived")
        ]

    def _selected_sensor_ids(self) -> list[str]:
        return [
            sensor_id
            for sensor_id, checkbox in self.sensor_checks.items()
            if checkbox.isChecked()
        ]

    def _sensor_selection_changed(self) -> None:
        self.calculate_button.setEnabled(bool(self._selected_sensor_ids()))

    def _refresh_sensor_choices(self) -> None:
        records = self._sensor_records()
        signature = tuple(
            (str(item["id"]), str(item.get("label") or item["id"]))
            for item in records
        )
        if signature == self._sensor_choice_signature:
            self._sensor_selection_changed()
            return

        previous = {
            sensor_id: checkbox.isChecked()
            for sensor_id, checkbox in self.sensor_checks.items()
        }
        while self.sensor_choices_layout.count():
            item = self.sensor_choices_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.sensor_checks = {}
        self._sensor_choice_signature = signature

        for sensor_id, label in signature:
            checkbox = QCheckBox(label)
            checkbox.setObjectName("sensorPlotSensorCheckbox")
            checkbox.setProperty("sensorId", sensor_id)
            checkbox.setChecked(previous.get(sensor_id, True))
            checkbox.toggled.connect(self._sensor_selection_changed)
            self.sensor_choices_layout.addWidget(checkbox)
            self.sensor_checks[sensor_id] = checkbox

        if not records:
            empty_label = QLabel("No sensors in scene")
            empty_label.setObjectName("sensorPlotNoSensorsLabel")
            self.sensor_choices_layout.addWidget(empty_label)
            self.plot.show_message(
                "Add a sensor to the scene to create a sensor plot",
                "Point and Linear axis sensors can both be plotted.",
            )
            self.export_rows = []
            self.export_metadata = {}
            self.export_csv_button.setEnabled(False)
        elif not previous and self.plot.currentWidget() is self.plot.message:
            self.plot.show_message(
                "No sensor path calculated yet",
                "Choose the sensors and output, then press Calculate.",
            )
        self._sensor_selection_changed()

    def event(self, event):  # noqa: A003 - Qt API name
        if (
            event.type() == QEvent.Type.WindowActivate
            and hasattr(self, "sensor_checks")
            and hasattr(self, "plot")
        ):
            self._refresh_sensor_choices()
        return super().event(event)

    def calculate(self) -> None:
        self._refresh_sensor_choices()
        selected_sensor_ids = self._selected_sensor_ids()
        if not selected_sensor_ids:
            if not self.sensor_checks:
                self.plot.show_message(
                    "Add a sensor to the scene to create a sensor plot",
                    "Point and Linear axis sensors can both be plotted.",
                )
            else:
                self.plot.show_message(
                    "Select at least one sensor to create a sensor plot",
                    "Check one or more sensors in Plot settings.",
                )
            self.calculation_status.showMessage("Ready")
            return
        calculation_started = time.perf_counter()
        progress = SensorPathProgressDialog(self)
        self.calculate_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.calculation_status.showMessage("Calculating sensor path…")
        try:
            bundle = self.adapter.sensor_plot_export_data(
                self.output.currentText(),
                sensor_ids=selected_sensor_ids,
                progress_callback=progress.update_from_solver,
                progress_label="Sensor path",
            )
            _set_field_map_rendering(progress, "sensor-path plot")
            self.export_rows = bundle["rows"]
            self.export_metadata = bundle["metadata"]
            self.export_csv_button.setEnabled(bool(self.export_rows))
            self.export_csv_button.setText("Export CSV…")
            self.plot.set_figure(bundle["figure"])
            elapsed = time.perf_counter() - calculation_started
            self.calculation_status.showMessage(
                _field_map_status_text(self.adapter, elapsed)
            )
        except FieldCalculationCancelled:
            elapsed = time.perf_counter() - calculation_started
            self.calculation_status.showMessage(
                f"Calculation cancelled after {_format_elapsed_time(elapsed)}"
            )
        except Exception as error:  # noqa: BLE001
            elapsed = time.perf_counter() - calculation_started
            self.export_rows = []
            self.export_metadata = {}
            self.export_csv_button.setEnabled(False)
            self.plot.show_message("Unable to plot the sensor path", str(error))
            self.calculation_status.showMessage(
                f"Calculation failed after {_format_elapsed_time(elapsed)}"
            )
        finally:
            progress.finish()
            self.calculate_button.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def export_csv(self) -> None:
        if not self.export_rows:
            QMessageBox.information(
                self,
                "Export sensor path",
                "Calculate the plot after adding a sensor or sensor path first.",
            )
            return
        output = self.export_metadata.get("output", self.output.currentText())
        suggested = f"sensor-path-{str(output).lower()}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export sensor path data",
            suggested,
            "CSV files (*.csv);;All files (*)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        fieldnames = [
            "series",
            "sample_index",
            "path_value",
            "path_axis_label",
            "output",
            "value",
            "value_axis_label",
            "value_unit",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.export_rows)
        except Exception as error:  # noqa: BLE001 - user-selected export boundary
            QMessageBox.warning(self, "Export sensor path", str(error))
            return
        self.export_csv_button.setText("Exported")

    def closeEvent(self, event):  # noqa: N802 - Qt API name
        self.plot.cleanup()
        super().closeEvent(event)
