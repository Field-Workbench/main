"""Field Workbench desktop entry point."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


# PyInstaller's spawned optimizer processes re-enter this executable. Divert
# them before importing Qt/WebEngine so multiprocessing can run the requested
# worker target instead of constructing another GUI application.
if __name__ == "__main__":
    multiprocessing.freeze_support()


def _prepare_environment() -> None:
    cache = Path(tempfile.gettempdir()) / "field-workbench-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        if "--no-sandbox" not in flags:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} --no-sandbox".strip()


_prepare_environment()

# Source launches on POSIX use a fork server for crash-isolated 3D renders.
# Start it before importing Qt/WebEngine, while this process is still
# single-threaded. Frozen builds and Windows retain multiprocessing spawn.
from fieldworkbench.process_ipc import (  # noqa: E402
    prime_field_render_process_context,
)

if __name__ == "__main__":
    prime_field_render_process_context()

# A source multiprocessing child re-imports this file as ``__mp_main__``.
# It needs the render target, not another Qt/WebEngine application tree.
if __name__ == "__mp_main__":
    pyi_splash = None
else:
    try:  # Available only in a PyInstaller build created with --splash.
        import pyi_splash  # type: ignore[import-not-found]
    except ImportError:  # Normal source, test, and non-splash launches.
        pyi_splash = None

    from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer  # noqa: E402
    from PySide6.QtWidgets import QApplication, QStyleFactory  # noqa: E402

    from fieldworkbench import __version__  # noqa: E402
    from fieldworkbench.main_window import MainWindow  # noqa: E402
    from fieldworkbench.plot_view import (  # noqa: E402
        _cleanup_plot_views_for_shutdown,
    )
    from fieldworkbench.theme import (  # noqa: E402
        apply_workbench_theme,
        load_theme_preference,
    )
    from fieldworkbench.waveform_gl_view import (  # noqa: E402
        _cleanup_waveform_gl_views_for_shutdown,
    )


def _close_packaged_splash() -> bool:
    """Close PyInstaller's splash if this executable was built with one."""

    if pyi_splash is None:
        return False
    try:
        if not pyi_splash.is_alive():
            return False
        pyi_splash.close()
        return True
    except Exception:  # noqa: BLE001 - optional bootloader integration
        return False


def _request_windows_foreground(window: MainWindow) -> None:
    """Best-effort native fallback for PyInstaller onedir splash focus loss."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(window.winId())
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    except Exception:  # noqa: BLE001 - best-effort platform workaround
        return


def _raise_main_window(window: MainWindow, *, native_fallback: bool = False) -> None:
    """Bring the Workbench forward after the onedir splash closes."""

    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()
    handle = window.windowHandle()
    if handle is not None:
        handle.requestActivate()
    if native_fallback:
        _request_windows_foreground(window)


def _raise_main_window_if_needed(
    window: MainWindow, *, native_fallback: bool = False
) -> None:
    """Retry activation only while the main window is still inactive."""

    if window.isActiveWindow():
        return
    _raise_main_window(window, native_fallback=native_fallback)


def _schedule_post_splash_raise(window: MainWindow) -> None:
    """Recover focus that Windows can lose when an onedir splash closes."""

    QTimer.singleShot(100, lambda: _raise_main_window_if_needed(window))
    if sys.platform == "win32":
        # A second pass catches the documented PyInstaller onedir race where
        # destroying the splash can demote another window in the same process.
        # Keep the native fallback delayed and conditional so normal launches
        # and an already-active Workbench window are left alone.
        QTimer.singleShot(350, lambda: _raise_main_window_if_needed(window, native_fallback=True))


def _cleanup_browser_surfaces_for_shutdown() -> None:
    """Destroy WebGL-backed children while Qt's shared contexts still exist."""

    _cleanup_waveform_gl_views_for_shutdown()
    _cleanup_plot_views_for_shutdown()
    # aboutToQuit is the last reliable event-loop turn. Both cleanup paths use
    # deleteLater() so WebEngine objects keep Qt ownership semantics; explicitly
    # deliver those deferred deletes before QApplication removes shared contexts.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Field Workbench workflow prototype")
    parser.add_argument("scene", nargs="?", help="optional .magpy.json scene to open")
    parser.add_argument("--smoke-test", action="store_true", help="launch briefly and exit with a status code")
    parser.add_argument("--version", action="version", version=f"Field Workbench {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Also cover embedders that import app.main() instead of executing app.py.
    # This remains before QApplication/QtWebEngine creates worker threads.
    prime_field_render_process_context()
    args = build_parser().parse_args(argv)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QCoreApplication.setAttribute(
        Qt.ApplicationAttribute.AA_UseStyleSheetPropagationInWidgetStyles
    )
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Field Workbench")
    application.setOrganizationName("Field Workbench")
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        application.setStyle(fusion)
    # Install the saved palette before any Workbench widgets are built, so
    # native sub-controls never snapshot an unrelated desktop palette.
    apply_workbench_theme(theme=load_theme_preference())

    window = MainWindow()
    if args.scene:
        if not window._load_scene_file(args.scene, replace_initial=True):
            print(f"Could not open {args.scene}", file=sys.stderr)
            return 2
    window.show()
    application.processEvents()
    if _close_packaged_splash():
        # PyInstaller onedir splash windows can briefly retain foreground focus.
        # Raise the real application after the splash has actually disappeared,
        # then retry once on Windows if the first activation request was lost.
        _schedule_post_splash_raise(window)
    application.aboutToQuit.connect(_cleanup_browser_surfaces_for_shutdown)
    if args.smoke_test:
        QTimer.singleShot(2500, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
