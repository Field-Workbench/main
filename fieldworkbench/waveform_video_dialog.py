"""Video export for cached 2D/3D waveform playback payloads."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QSignalBlocker, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .camera_timeline import (
    camera_at_physical_time,
    physical_time_for_video_elapsed,
    projected_video_duration_s,
    scripted_timeline_duration_s,
    waveform_elapsed_for_script_time,
)
from .waveform_gl_view import WaveformGLView
from .video_export import (
    decode_png_data_url,
    default_video_camera,
    ffmpeg_h264_arguments,
    find_ffmpeg_executable,
    unique_directory,
    validate_video_camera,
    orbit_video_camera,
)
from .window_utils import (
    CompactDoubleSpinBox,
    centre_window_on_parent,
    enable_standard_window_controls,
)


class _CollapsibleSection(QWidget):
    """Small local counterpart to the map viewer's disclosure sections."""

    def __init__(self, title: str, *, expanded: bool = True, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        self.header = QToolButton()
        self.header.setText(str(title))
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
        self.content.setVisible(bool(expanded))
        self.content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root.addWidget(self.content)
        self.header.toggled.connect(self._set_expanded)

    def setContentLayout(self, layout: QLayout) -> None:  # noqa: N802 - Qt helper
        self.content.setLayout(layout)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))



class WaveformVideoExportDialog(QDialog):
    """Configure and render a 2D/3D waveform video."""

    RESOLUTIONS = (
        ("Full HD — 1920 × 1080", 1920, 1080),
        ("HD — 1280 × 720", 1280, 720),
    )

    def __init__(self, payload: dict[str, Any], parent=None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        if str(payload.get("view_mode", "")).lower() not in {"2d", "3d"}:
            raise ValueError("Video export requires a 2D or 3D waveform playback payload.")
        self.payload = dict(payload)
        self.output_path: Path | None = None
        payload_camera = self.payload.get("initial_camera")
        try:
            self._default_camera = validate_video_camera(payload_camera) if isinstance(payload_camera, dict) else default_video_camera(self.payload)
            if isinstance(payload_camera, dict) and "zoom_2d" in payload_camera:
                self._default_camera["zoom_2d"] = float(payload_camera.get("zoom_2d", 1.0))
        except ValueError:
            self._default_camera = default_video_camera(self.payload)
        self._camera_timeline = self.payload.get("camera_timeline") if isinstance(self.payload.get("camera_timeline"), dict) else None
        self._preview_ready = False
        self._exporting = False
        self._cancel_requested = False
        self._frame_index = 0
        self._frame_directory: Path | None = None
        self._temporary_frames = False
        self._requested_output: Path | None = None
        self._encoder: QProcess | None = None
        self._progress: QProgressDialog | None = None
        self._encoder_failure_handled = False

        self.setWindowTitle("Export waveform video")
        self.resize(1540, 900)
        self.setMinimumSize(1240, 760)
        root = QVBoxLayout(self)
        heading = QLabel("Export video")
        heading.setObjectName("dialogHeading")
        root.addWidget(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)
        scroll.setMaximumWidth(560)
        options = QWidget()
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(8, 8, 8, 8)
        options_layout.setSpacing(9)
        scroll.setWidget(options)
        splitter.addWidget(scroll)

        self.video_format_section = _CollapsibleSection("Video format", expanded=True)
        format_form = QFormLayout()
        self.format = QComboBox()
        self.format.addItem("MP4 video (H.264)", "mp4")
        self.format.addItem("PNG frame sequence", "png")
        self.resolution = QComboBox()
        for label, width, height in self.RESOLUTIONS:
            self.resolution.addItem(label, (width, height))
        self.frame_rate = QLabel(f"{float(self.payload.get('playback_fps', 60.0)):g} fps")
        source_count = max(0, int(self.payload.get("frame_count", 0)))
        fps_value = max(1.0e-9, float(self.payload.get("playback_fps", 60.0)))
        base_multiplier = max(1.0e-9, float(self.payload.get("real_time_multiplier", 1.0)))
        fallback_physical_duration = (
            max(0.0, (source_count - 1) / fps_value * base_multiplier)
            if source_count > 1
            else 0.0
        )
        try:
            explicit_physical_duration = float(
                self.payload.get("waveform_duration_s", fallback_physical_duration)
            )
        except (TypeError, ValueError):
            explicit_physical_duration = fallback_physical_duration
        self._physical_duration_s = (
            explicit_physical_duration
            if explicit_physical_duration >= 0.0
            else fallback_physical_duration
        )
        self._projected_video_duration_s = projected_video_duration_s(
            self._camera_timeline,
            physical_duration_s=self._physical_duration_s,
            base_real_time_multiplier=base_multiplier,
        )
        self._export_frame_total = max(2, int(round(self._projected_video_duration_s * fps_value)) + 1) if source_count > 1 else source_count
        self.frame_count = QLabel(f"{self._export_frame_total:,} frames • {self._projected_video_duration_s:g} s")
        observation_traces = self.payload.get("observations")
        if not isinstance(observation_traces, list):
            observation_traces = (
                [self.payload["observation"]]
                if isinstance(self.payload.get("observation"), dict)
                else []
            )
        trace_names = [
            str(trace.get("label") or "Observation trace")
            for trace in observation_traces
            if isinstance(trace, dict)
        ]
        for key, fallback in (
            ("excitation", "Base excitation"),
            ("sequencer", "Sequencer"),
        ):
            trace = self.payload.get(key)
            if isinstance(trace, dict):
                trace_names.append(str(trace.get("label") or fallback))
        self.trace_summary = QLabel(", ".join(trace_names) if trace_names else "None selected")
        self.trace_summary.setWordWrap(True)
        self.trace_summary.setToolTip(
            "Enabled animation traces are included beneath the map view in every exported frame."
        )
        camera_points = (
            self._camera_timeline.get("points", [])
            if isinstance(self._camera_timeline, dict)
            else []
        )
        camera_summary_text = (
            f"Scripted — {len(camera_points)} control point{'s' if len(camera_points) != 1 else ''}"
            if camera_points
            else "Interactive/fixed — selector view"
        )
        self.camera_summary = QLabel(camera_summary_text)
        self.camera_summary.setWordWrap(True)
        format_form.addRow("Format", self.format)
        format_form.addRow("Resolution", self.resolution)
        format_form.addRow("Frame rate", self.frame_rate)
        format_form.addRow("Timeline", self.frame_count)
        format_form.addRow("Camera", self.camera_summary)
        format_form.addRow("Traces", self.trace_summary)
        brain_table = self.payload.get("brain_region_video_table")
        if isinstance(brain_table, dict) and brain_table.get("enabled"):
            self.brain_table_summary = QLabel(
                "Included — frozen filters, sorting, columns, and expansion state"
            )
            self.brain_table_summary.setWordWrap(True)
            self.brain_table_summary.setToolTip(
                "Brain Areas values update for every exported frame; the table layout and filters are frozen when Export video is opened."
            )
            format_form.addRow("Brain areas", self.brain_table_summary)

        encoder_row = QWidget()
        encoder_layout = QHBoxLayout(encoder_row)
        encoder_layout.setContentsMargins(0, 0, 0, 0)
        self.encoder_path = QLineEdit(find_ffmpeg_executable() or "")
        self.encoder_path.setPlaceholderText("Choose ffmpeg or ffmpeg.exe…")
        self.browse_encoder_button = QPushButton("Browse…")
        self.browse_encoder_button.clicked.connect(self._browse_encoder)
        encoder_layout.addWidget(self.encoder_path, 1)
        encoder_layout.addWidget(self.browse_encoder_button)
        self.encoder_label = QLabel("FFmpeg encoder")
        format_form.addRow(self.encoder_label, encoder_row)
        if sys.platform == "win32":
            # Packaged Windows builds always use the bundled, tested encoder.
            self.encoder_label.setVisible(False)
            encoder_row.setVisible(False)
        self.format_note = QLabel()
        self.format_note.setWordWrap(True)
        self.format_note.setObjectName("hint")
        format_form.addRow(self.format_note)
        self.video_format_section.setContentLayout(format_form)
        options_layout.addWidget(self.video_format_section)

        self.camera_options_section = _CollapsibleSection("Camera options", expanded=True)
        camera_layout = QVBoxLayout()
        camera_layout.setSpacing(8)
        camera_motion_form = QFormLayout()
        self.camera_motion = QComboBox()
        self.camera_motion.addItem("Fixed", "fixed")
        self.camera_motion.addItem("Orbit", "orbit")
        camera_motion_form.addRow("Camera motion", self.camera_motion)
        self.orbit_count = CompactDoubleSpinBox()
        self.orbit_count.setRange(0.25, 100.0)
        self.orbit_count.setDecimals(2)
        self.orbit_count.setSingleStep(0.25)
        self.orbit_count.setValue(1.0)
        self.orbit_count.setSuffix(" orbits")
        self.orbit_count_label = QLabel("Orbit count")
        camera_motion_form.addRow(self.orbit_count_label, self.orbit_count)
        camera_layout.addLayout(camera_motion_form)

        self.camera_position = self._coordinate_group(
            camera_layout, "Camera position"
        )
        self.camera_target = self._coordinate_group(camera_layout, "Look-at target")

        camera_fov_form = QFormLayout()
        self.camera_fov = CompactDoubleSpinBox()
        self.camera_fov.setRange(10.0, 120.0)
        self.camera_fov.setDecimals(1)
        self.camera_fov.setSuffix("°")
        self.camera_fov.setKeyboardTracking(False)
        camera_fov_form.addRow("Vertical field of view", self.camera_fov)
        up_note = QLabel("Orientation uses the look-at target with +Z as the vertical axis.")
        up_note.setWordWrap(True)
        up_note.setObjectName("hint")
        camera_fov_form.addRow(up_note)
        camera_layout.addLayout(camera_fov_form)
        camera_actions = QHBoxLayout()
        self.home_camera_button = QPushButton("Home camera")
        self.home_camera_button.clicked.connect(self._reset_camera)
        camera_actions.addWidget(self.home_camera_button)
        camera_actions.addStretch(1)
        camera_layout.addLayout(camera_actions)
        self.camera_options_section.setContentLayout(camera_layout)
        self.camera_options_section.setVisible(False)
        options_layout.addWidget(self.camera_options_section)
        options_layout.addStretch(1)

        preview_column = QWidget()
        preview_layout = QVBoxLayout(preview_column)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_heading = QLabel("Timeline preview" if self._camera_timeline else "Camera preview")
        preview_heading.setObjectName("sectionHeading")
        preview_layout.addWidget(preview_heading)
        self.preview = WaveformGLView(
            self.payload,
            preview_mode=True,
            initial_camera=self._default_camera,
        )
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self.preview, 1)
        splitter.addWidget(preview_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 1040])

        self.status = QLabel("Preparing camera preview…")
        self.status.setObjectName("hint")
        root.addWidget(self.status)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("primaryButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._choose_output)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.cancel_button)
        root.addLayout(button_row)

        self.format.currentIndexChanged.connect(self._update_format_controls)
        self.encoder_path.textChanged.connect(self._update_format_controls)
        self.camera_motion.currentIndexChanged.connect(self._update_camera_motion_controls)
        for editor in [*self.camera_position, *self.camera_target, self.camera_fov]:
            editor.valueChanged.connect(self._camera_editor_changed)
        self.preview.ready.connect(self._preview_loaded)
        self.preview.cameraChanged.connect(self._preview_camera_changed)
        self._set_camera_fields(self._default_camera, push=False)
        self._update_camera_motion_controls()
        self._update_format_controls()

    @staticmethod
    def _coordinate_group(
        parent_layout: QVBoxLayout, label: str
    ) -> list[QDoubleSpinBox]:
        group = QWidget()
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QGridLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)

        section_label = QLabel(label)
        section_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(section_label, 0, 0, 1, 3)

        editors: list[QDoubleSpinBox] = []
        for column, axis in enumerate("XYZ"):
            axis_label = QLabel(axis)
            axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            editor = CompactDoubleSpinBox()
            editor.setRange(-1.0e9, 1.0e9)
            editor.setDecimals(3)
            editor.setSuffix(" mm")
            editor.setKeyboardTracking(False)
            editor.setMinimumWidth(120)
            editor.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            layout.addWidget(axis_label, 1, column)
            layout.addWidget(editor, 2, column)
            layout.setColumnStretch(column, 1)
            editors.append(editor)
        parent_layout.addWidget(group)
        return editors

    def _normalized_camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_video_camera(camera)
        if str(self.payload.get("view_mode", "")).lower() == "2d":
            try:
                zoom = float(camera.get("zoom_2d", self._default_camera.get("zoom_2d", 1.0)))
            except (TypeError, ValueError):
                zoom = 1.0
            normalized["zoom_2d"] = max(0.01, min(100.0, zoom))
        return normalized

    def _camera_from_fields(self) -> dict[str, Any]:
        camera = {
            "position": [editor.value() for editor in self.camera_position],
            "target": [editor.value() for editor in self.camera_target],
            "fov_deg": self.camera_fov.value(),
        }
        if "zoom_2d" in self._default_camera:
            camera["zoom_2d"] = self._default_camera.get("zoom_2d", 1.0)
        return self._normalized_camera(camera)

    def _update_camera_motion_controls(self, *_args) -> None:
        orbit = self.camera_motion.currentData() == "orbit"
        self.orbit_count_label.setVisible(orbit)
        self.orbit_count.setVisible(orbit)

    def _camera_for_frame(self, frame_index: int, frame_total: int) -> dict[str, Any]:
        if self._camera_timeline is not None and self._camera_timeline.get("enabled", False):
            camera_time = (
                self._timeline_time_for_export_frame(frame_index)
                if self._camera_timeline.get("scripted", False)
                else self._physical_time_for_export_frame(frame_index)
            )
            return camera_at_physical_time(self._camera_timeline, camera_time)
        return self._normalized_camera(self._export_camera)

    def _timeline_time_for_export_frame(self, frame_index: int) -> float:
        fps = max(1.0e-9, float(self.payload.get("playback_fps", 60.0)))
        video_elapsed = max(0.0, int(frame_index)) / fps
        if isinstance(self._camera_timeline, dict) and self._camera_timeline.get("scripted", False):
            duration = scripted_timeline_duration_s(
                self._camera_timeline, fallback_s=self._physical_duration_s
            )
            base = max(1.0e-9, float(self.payload.get("real_time_multiplier", 1.0)))
            return min(duration, video_elapsed * base)
        return self._physical_time_for_export_frame(frame_index)

    def _physical_time_for_export_frame(self, frame_index: int) -> float:
        fps = max(1.0e-9, float(self.payload.get("playback_fps", 60.0)))
        video_elapsed = max(0.0, int(frame_index)) / fps
        if isinstance(self._camera_timeline, dict) and self._camera_timeline.get("scripted", False):
            script_time = self._timeline_time_for_export_frame(frame_index)
            return waveform_elapsed_for_script_time(
                self._camera_timeline,
                script_elapsed_s=script_time,
                base_real_time_multiplier=1.0,
            )
        return physical_time_for_video_elapsed(
            self._camera_timeline,
            video_elapsed_s=video_elapsed,
            physical_duration_s=self._physical_duration_s,
            base_real_time_multiplier=float(self.payload.get("real_time_multiplier", 1.0)),
        )

    def _source_frame_for_export_frame(self, frame_index: int) -> int:
        source_total = max(1, int(self.payload.get("frame_count", 1)))
        if source_total <= 1 or self._physical_duration_s <= 0.0:
            return 0
        waveform_elapsed = self._physical_time_for_export_frame(frame_index)
        looping = bool(
            isinstance(self._camera_timeline, dict)
            and self._camera_timeline.get("scripted", False)
            and self._camera_timeline.get("loop_waveform", False)
        )
        if looping:
            phase = waveform_elapsed % self._physical_duration_s
        else:
            phase = min(self._physical_duration_s, waveform_elapsed)
        fraction = max(0.0, min(1.0, phase / self._physical_duration_s))
        return max(0, min(source_total - 1, int(round(fraction * (source_total - 1)))))

    def _set_camera_fields(self, camera: dict[str, Any], *, push: bool) -> None:
        try:
            normalized = self._normalized_camera(camera)
        except ValueError:
            return
        editors = [*self.camera_position, *self.camera_target, self.camera_fov]
        blockers = [QSignalBlocker(editor) for editor in editors]
        for editor, value in zip(self.camera_position, normalized["position"], strict=True):
            editor.setValue(value)
        for editor, value in zip(self.camera_target, normalized["target"], strict=True):
            editor.setValue(value)
        self.camera_fov.setValue(normalized["fov_deg"])
        del blockers
        if push and self._preview_ready:
            self.preview.set_camera(normalized)

    def _camera_editor_changed(self, *_args) -> None:
        try:
            camera = self._camera_from_fields()
        except ValueError as error:
            self.status.setText(str(error))
            return
        if self._preview_ready:
            self.preview.set_camera(camera)
            self.status.setText("Camera preview ready")

    def _preview_camera_changed(self, camera: dict[str, Any]) -> None:
        self._set_camera_fields(camera, push=False)

    def _preview_loaded(self, ok: bool) -> None:
        self._preview_ready = bool(ok)
        if ok:
            self.preview.set_camera(self._camera_from_fields())
            self.status.setText("Camera preview ready")
        else:
            self.status.setText("The WebGL2 camera preview could not be initialized.")
        self._update_export_enabled()

    def _reset_camera(self, *_args) -> None:
        self._set_camera_fields(self._default_camera, push=True)
        self.status.setText("Home camera restored")

    def _encoder_program(self) -> str:
        if sys.platform == "win32":
            return find_ffmpeg_executable() or ""
        return self.encoder_path.text().strip()

    def _encoder_is_valid(self) -> bool:
        text = self._encoder_program()
        return bool(text and Path(text).expanduser().is_file())

    def _update_format_controls(self, *_args) -> None:
        mp4 = self.format.currentData() == "mp4"
        show_encoder_controls = mp4 and sys.platform != "win32"
        self.encoder_label.setVisible(show_encoder_controls)
        self.encoder_path.parentWidget().setVisible(show_encoder_controls)
        if mp4:
            if self._encoder_is_valid():
                if sys.platform == "win32":
                    self.format_note.setText(
                        "Exports an H.264 MP4 with the bundled Windows FFmpeg encoder and yuv420p colour for broad editor and player compatibility."
                    )
                else:
                    self.format_note.setText(
                        "Exports an H.264 MP4 with yuv420p colour for broad editor and player compatibility."
                    )
            else:
                if sys.platform == "win32":
                    self.format_note.setText(
                        "This Windows installation is missing its bundled FFmpeg encoder. "
                        "PNG frame sequence remains available without it."
                    )
                else:
                    self.format_note.setText(
                        "MP4 encoding requires FFmpeg. Choose ffmpeg before exporting. "
                        "PNG frame sequence remains available without an encoder."
                    )
        else:
            self.format_note.setText(
                "Writes lossless numbered PNG files into a new folder. This is the dependable fallback and can be imported as an image sequence in Shotcut."
            )
        self._update_export_enabled()

    def _update_export_enabled(self) -> None:
        format_ready = self.format.currentData() != "mp4" or self._encoder_is_valid()
        self.export_button.setEnabled(
            bool(self._preview_ready and format_ready and not self._exporting)
        )

    def _browse_encoder(self, *_args) -> None:
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose FFmpeg executable",
            self.encoder_path.text().strip(),
            "FFmpeg executable (ffmpeg ffmpeg.exe);;All files (*)",
        )
        if filename:
            self.encoder_path.setText(filename)

    def _resolution_values(self) -> tuple[int, int]:
        data = self.resolution.currentData()
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return int(data[0]), int(data[1])
        return 1920, 1080

    def _choose_output(self, *_args) -> None:
        if self._exporting:
            return
        try:
            camera = self._camera_from_fields()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid camera", str(error))
            return

        frame_total = int(self._export_frame_total)
        if frame_total > 10_000:
            answer = QMessageBox.question(
                self,
                "Large video export",
                f"This export contains {frame_total:,} full-resolution frames and may "
                "require substantial time and disk space. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        if self.format.currentData() == "mp4":
            if not self._encoder_is_valid():
                message = (
                    "The bundled Windows FFmpeg encoder is missing from this installation; "
                    "select PNG frame sequence instead."
                    if sys.platform == "win32"
                    else "Choose a valid FFmpeg executable, or select PNG frame sequence."
                )
                QMessageBox.warning(self, "FFmpeg required", message)
                return
            filename, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export waveform video",
                "field_workbench_waveform.mp4",
                "MP4 video (*.mp4)",
            )
            if not filename:
                return
            output = Path(filename)
            if output.suffix.lower() != ".mp4":
                output = output.with_suffix(".mp4")
            self._requested_output = output
        else:
            parent = QFileDialog.getExistingDirectory(
                self,
                "Choose parent folder for PNG frame sequence",
                "",
            )
            if not parent:
                return
            self._requested_output = unique_directory(
                Path(parent) / "FieldWorkbench_waveform_frames"
            )

        self.status.setText("Reading the camera from the preview…")
        self.preview.request_camera(self._begin_export)

    def _begin_export(self, camera: Any) -> None:
        if self._requested_output is None:
            return
        try:
            normalized_camera = self._normalized_camera(
                camera if isinstance(camera, dict) else self._camera_from_fields()
            )
        except ValueError as error:
            QMessageBox.warning(self, "Invalid camera", str(error))
            return
        self._set_camera_fields(normalized_camera, push=False)

        frame_total = int(self._export_frame_total)
        if frame_total < 2:
            QMessageBox.warning(
                self,
                "Unable to export video",
                "The waveform playback does not contain enough frames to export.",
            )
            return
        try:
            if self.format.currentData() == "mp4":
                self._frame_directory = Path(
                    tempfile.mkdtemp(prefix="field-workbench-video-frames-")
                )
                self._temporary_frames = True
            else:
                self._frame_directory = self._requested_output
                self._frame_directory.mkdir(parents=True, exist_ok=False)
                self._temporary_frames = False
        except OSError as error:
            QMessageBox.critical(self, "Unable to export video", str(error))
            return

        self._export_camera = normalized_camera
        self._export_camera_motion = "timeline" if self._camera_timeline else "fixed"
        self._export_orbit_count = 0.0
        self._exporting = True
        self._cancel_requested = False
        self._frame_index = 0
        self._encoder_failure_handled = False
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.format.setEnabled(False)
        self.resolution.setEnabled(False)
        self.camera_motion.setEnabled(False)
        self.orbit_count.setEnabled(False)
        self._progress = QProgressDialog(
            "Rendering video frame 1…",
            "Cancel",
            0,
            frame_total,
            self,
        )
        self._progress.setWindowTitle("Export waveform video")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.canceled.connect(self._cancel_export)
        self._progress.setValue(0)
        self._progress.show()
        centre_window_on_parent(self._progress, self)
        QTimer.singleShot(0, self._capture_next_frame)

    def _capture_next_frame(self) -> None:
        if not self._exporting or self._cancel_requested:
            return
        frame_total = int(self._export_frame_total)
        if self._frame_index >= frame_total:
            self._frames_finished()
            return
        if self._progress is not None:
            self._progress.setLabelText(
                f"Rendering video frame {self._frame_index + 1:,} of {frame_total:,}…"
            )
        width, height = self._resolution_values()
        try:
            frame_camera = self._camera_for_frame(self._frame_index, frame_total)
        except ValueError as error:
            self._fail_export(str(error))
            return
        self.preview.capture_frame(
            frame_index=self._source_frame_for_export_frame(self._frame_index),
            width=width,
            height=height,
            camera=frame_camera,
            callback=self._frame_captured,
            sequence_elapsed_s=self._physical_time_for_export_frame(self._frame_index),
            timeline_elapsed_s=self._timeline_time_for_export_frame(self._frame_index),
        )

    def _frame_captured(self, value: Any) -> None:
        if not self._exporting or self._cancel_requested:
            return
        try:
            png = decode_png_data_url(value)
            if self._frame_directory is None:
                raise OSError("The frame output folder is unavailable.")
            path = self._frame_directory / f"frame_{self._frame_index:06d}.png"
            path.write_bytes(png)
        except (OSError, ValueError) as error:
            self._fail_export(f"Could not render video frame {self._frame_index + 1}:\n{error}")
            return
        self._frame_index += 1
        if self._progress is not None:
            self._progress.setValue(self._frame_index)
        QTimer.singleShot(0, self._capture_next_frame)

    def _frames_finished(self) -> None:
        if self.format.currentData() == "png":
            destination = self._frame_directory
            self._temporary_frames = False
            self.output_path = destination
            self._finish_success(
                f"Exported {self._frame_index:,} PNG frames to:\n{destination}"
            )
            return
        self._start_ffmpeg()

    def _start_ffmpeg(self) -> None:
        if self._frame_directory is None or self._requested_output is None:
            self._fail_export("The rendered frames or MP4 destination are unavailable.")
            return
        if self._progress is not None:
            self._progress.setRange(0, 0)
            self._progress.setLabelText("Encoding H.264 MP4 with FFmpeg…")
        frame_rate = float(self.payload.get("playback_fps", 60.0))
        try:
            arguments = ffmpeg_h264_arguments(
                self._frame_directory / "frame_%06d.png",
                frame_rate=frame_rate,
                output_path=self._requested_output,
            )
        except ValueError as error:
            self._fail_export(str(error), preserve_frames=True)
            return
        self._encoder = QProcess(self)
        self._encoder.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._encoder.finished.connect(self._ffmpeg_finished)
        self._encoder.errorOccurred.connect(self._ffmpeg_process_error)
        self._encoder.setProgram(str(Path(self._encoder_program()).expanduser()))
        self._encoder.setArguments(arguments)
        self._encoder.start()

    def _ffmpeg_process_error(self, error) -> None:
        if not self._exporting or self._encoder_failure_handled:
            return
        self._encoder_failure_handled = True
        self._fail_export(
            f"FFmpeg could not be started ({error}).",
            preserve_frames=True,
        )

    def _ffmpeg_finished(self, exit_code: int, exit_status) -> None:
        if not self._exporting or self._cancel_requested or self._encoder_failure_handled:
            return
        output_text = ""
        if self._encoder is not None:
            output_text = bytes(self._encoder.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            ).strip()
        output_ok = (
            exit_status == QProcess.ExitStatus.NormalExit
            and int(exit_code) == 0
            and self._requested_output is not None
            and self._requested_output.is_file()
            and self._requested_output.stat().st_size > 0
        )
        if not output_ok:
            details = output_text[-1800:] if output_text else f"FFmpeg exited with code {exit_code}."
            self._fail_export(
                "FFmpeg could not create the MP4:\n" + details,
                preserve_frames=True,
            )
            return
        self.output_path = self._requested_output
        self._finish_success(f"Exported H.264 MP4 video to:\n{self.output_path}")

    def _preserve_rendered_frames(self) -> Path | None:
        source = self._frame_directory
        output = self._requested_output
        if source is None or output is None or not source.exists():
            return None
        destination = unique_directory(output.parent / f"{output.stem}_frames")
        try:
            shutil.move(str(source), str(destination))
        except OSError:
            return None
        self._frame_directory = destination
        self._temporary_frames = False
        return destination

    def _finish_success(self, message: str) -> None:
        self._exporting = False
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._temporary_frames and self._frame_directory is not None:
            shutil.rmtree(self._frame_directory, ignore_errors=True)
        self._temporary_frames = False
        QMessageBox.information(self, "Video export complete", message)
        self.accept()

    def _fail_export(self, message: str, *, preserve_frames: bool = False) -> None:
        preserved = self._preserve_rendered_frames() if preserve_frames else None
        self._exporting = False
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._temporary_frames and self._frame_directory is not None:
            shutil.rmtree(self._frame_directory, ignore_errors=True)
        self._temporary_frames = False
        self.cancel_button.setEnabled(True)
        self.format.setEnabled(True)
        self.resolution.setEnabled(True)
        self.camera_motion.setEnabled(True)
        self.orbit_count.setEnabled(True)
        self._update_export_enabled()
        if preserved is not None:
            message += f"\n\nThe rendered PNG frames were retained at:\n{preserved}"
        QMessageBox.critical(self, "Unable to export video", message)
        self.status.setText("Video export failed")

    def _cancel_export(self, *_args) -> None:
        if not self._exporting:
            return
        self._exporting = False
        self._cancel_requested = True
        if self._encoder is not None and self._encoder.state() != QProcess.ProcessState.NotRunning:
            self._encoder.kill()
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._frame_directory is not None:
            shutil.rmtree(self._frame_directory, ignore_errors=True)
        self._temporary_frames = False
        self.cancel_button.setEnabled(True)
        self.format.setEnabled(True)
        self.resolution.setEnabled(True)
        self.camera_motion.setEnabled(True)
        self.orbit_count.setEnabled(True)
        self.status.setText("Video export cancelled")
        self._update_export_enabled()

    def done(self, result: int) -> None:
        if self._exporting and result != int(QDialog.DialogCode.Accepted):
            self._cancel_export()
        self.preview.cleanup()
        super().done(result)
