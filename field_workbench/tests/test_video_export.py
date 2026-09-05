from __future__ import annotations

import base64
import hashlib
import math
import unittest
from unittest.mock import patch
from pathlib import Path

import fieldworkbench.video_export as video_export
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from fieldworkbench.video_export import (
    decode_png_data_url,
    BUNDLED_WINDOWS_FFMPEG,
    default_video_camera,
    ffmpeg_h264_arguments,
    find_ffmpeg_executable,
    orbit_video_camera,
    validate_video_camera,
)


class VideoExportTests(unittest.TestCase):
    def test_default_camera_fits_map_and_scene_bounds(self) -> None:
        camera = default_video_camera(
            {
                "centre_mm": [0.0, 0.0, 0.0],
                "span_mm": 100.0,
                "scene": {
                    "bounds_min": [-200.0, -50.0, -20.0],
                    "bounds_max": [100.0, 150.0, 80.0],
                },
            }
        )
        self.assertEqual(camera["target"], [-50.0, 50.0, 15.0])
        self.assertAlmostEqual(camera["fov_deg"], 45.0)
        distance = math.dist(camera["position"], camera["target"])
        expected_radius = math.sqrt(300.0**2 + 200.0**2 + 130.0**2) / 2.0
        self.assertAlmostEqual(distance, expected_radius * 2.7)

    def test_camera_validation_rejects_coincident_position_and_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            validate_video_camera(
                {
                    "position": [1.0, 2.0, 3.0],
                    "target": [1.0, 2.0, 3.0],
                    "fov_deg": 45.0,
                }
            )

    def test_orbit_camera_rotates_about_look_at_target_and_preserves_height(self) -> None:
        camera = {
            "position": [10.0, 0.0, 7.0],
            "target": [0.0, 0.0, 2.0],
            "fov_deg": 45.0,
        }
        quarter = orbit_video_camera(camera, progress=0.25, revolutions=1.0)
        self.assertAlmostEqual(quarter["position"][0], 0.0, places=9)
        self.assertAlmostEqual(quarter["position"][1], 10.0, places=9)
        self.assertEqual(quarter["position"][2], 7.0)
        self.assertEqual(quarter["target"], [0.0, 0.0, 2.0])
        self.assertEqual(quarter["fov_deg"], 45.0)
        complete = orbit_video_camera(camera, progress=1.0, revolutions=2.0)
        for actual, expected in zip(complete["position"], camera["position"], strict=True):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_png_data_url_decoder_checks_png_signature(self) -> None:
        png = b"\x89PNG\r\n\x1a\nexample"
        encoded = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        self.assertEqual(decode_png_data_url(encoded), png)
        with self.assertRaisesRegex(ValueError, "invalid PNG"):
            decode_png_data_url(
                "data:image/png;base64," + base64.b64encode(b"not png").decode("ascii")
            )

    def test_windows_ffmpeg_bundle_is_pinned_and_resolved_automatically(self) -> None:
        self.assertTrue(BUNDLED_WINDOWS_FFMPEG.is_file())
        self.assertEqual(BUNDLED_WINDOWS_FFMPEG.stat().st_size, 4_056_576)
        digest = hashlib.sha256(BUNDLED_WINDOWS_FFMPEG.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "9e6f299118efd596309c5e1bcd5f4cb9410f825e1efd4ee36dc4279390ae7f47",
        )
        with patch("fieldworkbench.video_export.sys.platform", "win32"):
            self.assertEqual(find_ffmpeg_executable(), str(BUNDLED_WINDOWS_FFMPEG.resolve()))

    def test_windows_ffmpeg_resolver_accepts_pyinstaller_internal_layout(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            meipass = Path(temporary)
            bundled = (
                meipass
                / "fieldworkbench"
                / "assets"
                / "ffmpeg"
                / "windows-x86_64"
                / "ffmpeg.exe"
            )
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"stub")
            with (
                patch("fieldworkbench.video_export.sys.platform", "win32"),
                patch.object(video_export.sys, "_MEIPASS", str(meipass), create=True),
            ):
                self.assertEqual(find_ffmpeg_executable(), str(bundled.resolve()))

    def test_rendered_tab_bridge_keeps_original_exporter_renderer(self) -> None:
        dialog_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_video_dialog.py").read_text(encoding="utf-8")
        gl_source = (PROJECT_ROOT / "fieldworkbench" / "waveform_gl_view.py").read_text(encoding="utf-8")
        map_source = (PROJECT_ROOT / "fieldworkbench" / "dialogs.py").read_text(encoding="utf-8")
        brain_source = (PROJECT_ROOT / "fieldworkbench" / "brain_view.py").read_text(encoding="utf-8")
        self.assertNotIn("render_view: WaveformGLView | None", dialog_source)
        self.assertIn("preserveDrawingBuffer:previewMode", gl_source)
        self.assertNotIn("video_capture_enabled", gl_source)
        self.assertIn("view.cleanup()", map_source)
        self.assertIn("QTimer.singleShot(0, run_original_exporter)", map_source)
        self.assertIn("view.cleanup()", brain_source)
        self.assertIn("QTimer.singleShot(0, run_original_exporter)", brain_source)

    def test_ffmpeg_mp4_arguments_use_editor_friendly_h264(self) -> None:
        arguments = ffmpeg_h264_arguments(
            Path("frames") / "frame_%06d.png",
            frame_rate=60.0,
            output_path=Path("output.mp4"),
        )
        self.assertIn("libx264", arguments)
        self.assertIn("yuv420p", arguments)
        self.assertIn("+faststart", arguments)
        self.assertEqual(arguments[-1], "output.mp4")

    def test_gpu_html_exposes_fixed_camera_and_png_capture_hooks(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "fieldworkbench"
            / "waveform_gl_view.py"
        ).read_text(encoding="utf-8")
        self.assertIn("fieldWorkbenchSetCamera", source)
        self.assertIn("fieldWorkbenchGetCamera", source)
        self.assertIn("fieldWorkbenchCaptureFrame", source)
        self.assertIn("preserveDrawingBuffer:previewMode", source)
        self.assertIn("exportTraceFraction", source)
        self.assertIn("observationCaptureSize", source)
        self.assertIn("x.drawImage(obsCanvas", source)
        self.assertIn("x.drawImage(gridCanvas", source)
        self.assertNotIn("body.export-preview #observation-wrap", source)

    def test_export_dialog_exposes_requested_camera_layout(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "fieldworkbench"
            / "waveform_video_dialog.py"
        ).read_text(encoding="utf-8")
        self.assertIn('_CollapsibleSection("Video format"', source)
        self.assertIn('self.camera_options_section.setVisible(False)', source)
        self.assertIn('addItem("MP4 video (H.264)", "mp4")', source)
        self.assertIn('addItem("PNG frame sequence", "png")', source)
        self.assertIn('addItem("Fixed", "fixed")', source)
        self.assertIn('camera_at_physical_time', source)
        self.assertIn('_source_frame_for_export_frame', source)
        self.assertIn("orbit_video_camera", source)
        self.assertIn("layout.addWidget(section_label, 0, 0, 1, 3)", source)
        self.assertIn("layout.addWidget(axis_label, 1, column)", source)
        self.assertIn("layout.addWidget(editor, 2, column)", source)
        self.assertIn("parent_layout.addWidget(group)", source)
        self.assertIn("layout.setColumnStretch(column, 1)", source)
        self.assertNotIn("form.addRow(section_label)", source)
        self.assertIn('format_form.addRow("Traces", self.trace_summary)', source)
        self.assertIn(
            '"Timeline preview" if self._camera_timeline else "Camera preview"',
            source,
        )
        self.assertIn("preview_layout.addWidget(self.preview, 1)", source)
        self.assertNotIn("_AspectRatioFrame", source)
        self.assertNotIn("Drag to orbit", source)
        self.assertNotIn("fixed view shown here", source)
        self.assertIn('QPushButton("Export")', source)
        self.assertIn('QPushButton("Cancel")', source)


if __name__ == "__main__":
    unittest.main()
