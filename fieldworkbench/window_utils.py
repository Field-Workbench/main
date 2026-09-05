"""Shared helpers for consistent native top-level window controls and UI state."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QSplitter,
    QWidget,
)
from shiboken6 import isValid as _shiboken_is_valid

from .numeric_inputs import compact_fixed_decimal_text, stepped_playback_speed


_STANDARD_WINDOW_CONTROL_HINTS = (
    Qt.WindowType.CustomizeWindowHint
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowMinimizeButtonHint
    | Qt.WindowType.WindowMaximizeButtonHint
    | Qt.WindowType.WindowCloseButtonHint
)

# Parentless application workspaces should look like ordinary desktop windows,
# not customized QDialogs.  Keeping this exact flag set free of
# CustomizeWindowHint lets the native window manager supply its normal close
# decoration while still requesting every standard title-bar control.
_APPLICATION_WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowMinimizeButtonHint
    | Qt.WindowType.WindowMaximizeButtonHint
    | Qt.WindowType.WindowCloseButtonHint
)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """A fixed-precision editor that does not display insignificant zeroes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._minimum_display_decimals = 0

    def setMinimumDisplayDecimals(self, decimals: int) -> None:  # noqa: N802 - Qt-style API
        self._minimum_display_decimals = max(0, int(decimals))
        self.update()

    def minimumDisplayDecimals(self) -> int:  # noqa: N802 - Qt-style API
        return int(self._minimum_display_decimals)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt virtual
        return compact_fixed_decimal_text(
            super().textFromValue(value),
            decimal_point=str(self.locale().decimalPoint()),
            minimum_decimals=self._minimum_display_decimals,
        )


class PlaybackSpeedSpinBox(CompactDoubleSpinBox):
    """Playback multiplier editor with ratio-aware arrow stepping."""

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt virtual
        # Commit a manually typed value before choosing the next rung.
        self.interpretText()
        self.setValue(
            stepped_playback_speed(
                self.value(),
                steps,
                minimum=self.minimum(),
                maximum=self.maximum(),
                decimals=self.decimals(),
            )
        )


def _application_uses_persistent_settings() -> bool:
    """Avoid leaking a developer/test QApplication's widgets into user settings."""
    return (
        QCoreApplication.applicationName() == "Field Workbench"
        and QCoreApplication.organizationName() == "Field Workbench"
    )


def _persistent_scope(window: QWidget) -> str:
    explicit = str(getattr(window, "_persistent_settings_key", "") or "").strip()
    if explicit:
        return explicit
    cls = type(window)
    return f"{cls.__module__}.{cls.__qualname__}"


def _qt_object_is_alive(value: QObject) -> bool:
    """Return False after Qt has destroyed a still-referenced Python wrapper."""

    try:
        return bool(_shiboken_is_valid(value))
    except (RuntimeError, TypeError):
        return False


def _iter_public_controls(window: QWidget):
    """Yield stable public attribute names and ordinary setting-like controls."""

    supported = (QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QSplitter, QLineEdit, QAbstractButton)

    def walk(name: str, value: Any):
        if isinstance(value, supported):
            if _qt_object_is_alive(value):
                yield name, value
            return
        # Coordinate triplets and similar editor collections are commonly stored
        # as a public list/tuple rather than as separate named attributes.
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, supported) and _qt_object_is_alive(item):
                    yield f"{name}/{index}", item
            return
        # Dense map sidebars expose CollapsibleSection instances as attributes.
        # Avoid importing dialogs.py here (which would create a circular import)
        # and persist the disclosure header by duck-typing its checkable button.
        try:
            header = getattr(value, "header", None)
            if (
                isinstance(header, QAbstractButton)
                and _qt_object_is_alive(header)
                and header.isCheckable()
            ):
                yield f"{name}/expanded", header
        except RuntimeError:
            # Dynamic panels can retain a Python wrapper for one event-loop turn
            # after Qt has deleted the underlying disclosure button.
            return

    for name, value in vars(window).items():
        if name.startswith("_"):
            continue
        yield from walk(name, value)


def _control_state(widget: QWidget) -> dict[str, Any] | None:
    if not _qt_object_is_alive(widget):
        return None
    try:
        if widget.property("persistUiState") is False:
            return None
        if isinstance(widget, QComboBox):
            return {"kind": "combo", "text": widget.currentText(), "index": widget.currentIndex()}
        if isinstance(widget, QDoubleSpinBox):
            return {"kind": "double", "value": float(widget.value())}
        if isinstance(widget, QSpinBox):
            return {"kind": "spin", "value": int(widget.value())}
        if isinstance(widget, QSlider):
            return {"kind": "slider", "value": int(widget.value())}
        if isinstance(widget, QLineEdit) and not widget.isReadOnly():
            return {"kind": "lineedit", "text": widget.text()}
        if isinstance(widget, QSplitter):
            return {"kind": "splitter", "sizes": [int(value) for value in widget.sizes()]}
        if isinstance(widget, QAbstractButton) and widget.isCheckable():
            return {"kind": "checked", "checked": bool(widget.isChecked())}
    except RuntimeError:
        # A deferred child deletion can race a top-level Hide event during
        # application shutdown. UI persistence is best-effort, so omit only the
        # control whose native Qt object has already gone away.
        return None
    return None


def _restore_control_state(widget: QWidget, state: Any) -> None:
    if not isinstance(state, dict) or not _qt_object_is_alive(widget):
        return
    try:
        if widget.property("persistUiState") is False:
            return
        kind = str(state.get("kind", ""))
        if kind == "combo" and isinstance(widget, QComboBox):
            text = str(state.get("text", ""))
            index = widget.findText(text)
            if index < 0 and text in {"X component", "Y component", "Z component"}:
                # 0.13.0.111 split the old signed X/Y/Z map choices into
                # explicit signed and absolute-magnitude choices. Preserve the
                # old semantics rather than restoring by the now-shifted index.
                prefix = text[0] + " signed "
                index = next(
                    (item for item in range(widget.count()) if widget.itemText(item).startswith(prefix)),
                    -1,
                )
            if index < 0:
                index = int(state.get("index", -1))
            if 0 <= index < widget.count():
                widget.setCurrentIndex(index)
        elif kind == "double" and isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(state.get("value", widget.value())))
        elif kind == "spin" and isinstance(widget, QSpinBox):
            widget.setValue(int(state.get("value", widget.value())))
        elif kind == "slider" and isinstance(widget, QSlider):
            widget.setValue(int(state.get("value", widget.value())))
        elif kind == "lineedit" and isinstance(widget, QLineEdit) and not widget.isReadOnly():
            widget.setText(str(state.get("text", widget.text())))
        elif kind == "splitter" and isinstance(widget, QSplitter):
            sizes = [int(value) for value in state.get("sizes", [])]
            if sizes:
                widget.setSizes(sizes)
        elif kind == "checked" and isinstance(widget, QAbstractButton) and widget.isCheckable():
            widget.setChecked(bool(state.get("checked", widget.isChecked())))
    except (RuntimeError, TypeError, ValueError):
        # Stale settings from an older version should never prevent a window from opening.
        return


class _PersistentWindowStateFilter(QObject):
    """Save ordinary user controls when a Workbench window closes and restore on show."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.window = window
        self._restored = False
        self._saved_since_show = False
        self._base_key_value = f"ui/windows/{_persistent_scope(window)}"

    @property
    def _base_key(self) -> str:
        return self._base_key_value

    def _allow_resave_after_ignored_close(self) -> None:
        """Clear the Close/Hide guard if a window rejected its close request."""

        if not _qt_object_is_alive(self.window):
            return
        try:
            if self.window.isVisible():
                self._saved_since_show = False
        except RuntimeError:
            return

    def _restore(self) -> None:
        if self._restored or not _application_uses_persistent_settings():
            return
        self._restored = True
        settings = QSettings()
        geometry = settings.value(f"{self._base_key}/geometry")
        if geometry is not None:
            try:
                self.window.restoreGeometry(geometry)
            except Exception:  # noqa: BLE001 - stale platform-specific geometry
                pass

        raw = settings.value(f"{self._base_key}/controls", "")
        if raw:
            try:
                stored = json.loads(str(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                stored = {}
            if isinstance(stored, dict):
                for name, widget in _iter_public_controls(self.window):
                    if name in stored:
                        _restore_control_state(widget, stored[name])

        restore_extra = getattr(self.window, "_restore_persistent_ui_state", None)
        if callable(restore_extra):
            extra_raw = settings.value(f"{self._base_key}/extra", "")
            if extra_raw:
                try:
                    extra = json.loads(str(extra_raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    extra = None
                if isinstance(extra, dict):
                    try:
                        restore_extra(extra)
                    except Exception:  # noqa: BLE001 - stale state must not break the window
                        pass

    def _save(self) -> None:
        if not _application_uses_persistent_settings():
            return
        settings = QSettings()
        try:
            settings.setValue(f"{self._base_key}/geometry", self.window.saveGeometry())
        except Exception:  # noqa: BLE001 - best-effort native geometry persistence
            pass

        controls: dict[str, Any] = {}
        for name, widget in _iter_public_controls(self.window):
            state = _control_state(widget)
            if state is not None:
                controls[name] = state
        settings.setValue(
            f"{self._base_key}/controls",
            json.dumps(controls, separators=(",", ":"), ensure_ascii=False),
        )

        save_extra = getattr(self.window, "_persistent_ui_state", None)
        if callable(save_extra):
            try:
                extra = save_extra()
                settings.setValue(
                    f"{self._base_key}/extra",
                    json.dumps(extra, separators=(",", ":"), ensure_ascii=False),
                )
            except Exception:  # noqa: BLE001 - persistence must never block close/accept
                pass
        settings.sync()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        event_type = event.type()
        if event_type == QEvent.Type.Show:
            self._saved_since_show = False
            try:
                self._restore()
            except RuntimeError:
                # A queued Show can arrive while a parent-driven teardown is
                # already invalidating child wrappers. Persistence must never
                # turn that native lifecycle race into a Python override error.
                pass
        elif event_type in {QEvent.Type.Close, QEvent.Type.Hide}:
            # A successful close normally emits Close and then Hide. Save while
            # the controls are intact on the first event and do not traverse the
            # same wrappers again after child destruction has started.
            if not self._saved_since_show:
                self._saved_since_show = True
                try:
                    self._save()
                except RuntimeError:
                    # Dynamic Qt forms can still lose a child between validity
                    # checks. State persistence is intentionally best-effort.
                    pass
                if event_type == QEvent.Type.Close:
                    QTimer.singleShot(0, self._allow_resave_after_ignored_close)
        return super().eventFilter(watched, event)



def centre_window_on_parent(window: QWidget, parent: QWidget | None = None) -> None:
    """Centre one transient window over its owner and keep it on the active screen.

    Call this after the window has been shown (and, for dynamically sized dialogs,
    after one event pass) so ``width()``/``height()`` reflect the native layout.
    If no owner is available, centre on the window's/current application's screen.
    """
    window.adjustSize()
    owner = parent or window.parentWidget()
    screen = owner.screen() if owner is not None else window.screen()
    if screen is None:
        screen = QApplication.primaryScreen()

    if owner is not None and owner.isVisible():
        target = owner.mapToGlobal(owner.rect().center())
        x = target.x() - window.width() // 2
        y = target.y() - window.height() // 2
    elif screen is not None:
        available = screen.availableGeometry()
        x = available.center().x() - window.width() // 2
        y = available.center().y() - window.height() // 2
    else:
        return

    if screen is not None:
        available = screen.availableGeometry()
        x = max(available.left(), min(x, available.right() - window.width() + 1))
        y = max(available.top(), min(y, available.bottom() - window.height() + 1))
    window.move(x, y)


def capture_window_default_settings(window: QWidget) -> dict[str, Any]:
    """Capture the constructor defaults for a persistent user-facing window.

    Call this once near the end of ``__init__`` after dynamic controls have been
    populated but before the first Show event restores persisted user settings.
    The snapshot deliberately excludes native window geometry: Reset defaults is
    about the tool parameters, not moving/resizing the user's window.
    """
    controls: dict[str, Any] = {}
    for name, widget in _iter_public_controls(window):
        state = _control_state(widget)
        if state is not None:
            controls[name] = state

    extra: dict[str, Any] | None = None
    save_extra = getattr(window, "_persistent_ui_state", None)
    if callable(save_extra):
        try:
            candidate = save_extra()
            if isinstance(candidate, dict):
                # JSON round-tripping gives us an independent, persistence-safe
                # snapshot even when the producer returns nested mutable lists.
                extra = json.loads(json.dumps(candidate, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - default capture is best-effort
            extra = None

    snapshot = {"controls": controls, "extra": extra}
    window._default_ui_state_snapshot = snapshot  # type: ignore[attr-defined]
    return snapshot


def reset_window_to_default_settings(window: QWidget) -> bool:
    """Restore one window's captured constructor defaults and persist them.

    Returns ``True`` when a default snapshot was available. The current scene or
    document is never touched; only controls owned by this top-level window are
    restored.
    """
    snapshot = getattr(window, "_default_ui_state_snapshot", None)
    if not isinstance(snapshot, dict):
        return False

    controls = snapshot.get("controls", {})
    if isinstance(controls, dict):
        for name, widget in _iter_public_controls(window):
            if name in controls:
                _restore_control_state(widget, controls[name])

    extra = snapshot.get("extra")
    restore_extra = getattr(window, "_restore_persistent_ui_state", None)
    if isinstance(extra, dict) and callable(restore_extra):
        try:
            restore_extra(json.loads(json.dumps(extra, ensure_ascii=False)))
        except Exception:  # noqa: BLE001 - reset should never strand the dialog
            pass

    # Viewer-backed setup dialogs redraw their selector immediately after this
    # helper returns. PlotView normally preserves the current camera/ranges across
    # those redraws, which can leave restored default geometry completely outside
    # the user's previous zoom. Mark the primary preview for a clean Home fit on
    # that next figure. Duck typing keeps this utility independent of PlotView and
    # harmless for ordinary dialogs such as the waveform editor.
    primary_viewer = getattr(window, "plot", None)
    request_home = getattr(primary_viewer, "request_home_on_next_figure", None)
    if callable(request_home):
        try:
            request_home()
        except Exception:  # noqa: BLE001 - resetting controls must still succeed
            pass

    # Make the reset sticky immediately. This also replaces any older persisted
    # parameter state so reopening the dialog starts from these defaults.
    state_filter = getattr(window, "_persistent_window_state_filter", None)
    save = getattr(state_filter, "_save", None)
    if callable(save):
        save()
    return True

def install_persistent_window_settings(window: QWidget) -> QWidget:
    """Install Workbench-wide geometry/control persistence on one top-level window.

    The helper is intentionally inert unless the running QApplication identifies
    itself as Field Workbench. This keeps unit tests and embedded uses deterministic.
    """
    if getattr(window, "_persistent_window_state_filter", None) is None:
        state_filter = _PersistentWindowStateFilter(window)
        window._persistent_window_state_filter = state_filter  # type: ignore[attr-defined]
        window.installEventFilter(state_filter)
    return window


def enable_standard_window_controls(window: QWidget) -> QWidget:
    """Request normal title-bar controls and remember user-facing window settings.

    Windows treats ``QDialog`` title bars more conservatively than many Linux
    window managers and may normalize individual control hints unless the title
    bar is explicitly customized. Preserve the current window type/modality and
    unrelated hints, opt into customized decorations, and request the same
    title/system/minimize/maximize/close controls on every platform.
    """
    flags = window.windowFlags() | _STANDARD_WINDOW_CONTROL_HINTS
    flags &= ~Qt.WindowType.WindowContextHelpButtonHint
    window.setWindowFlags(flags)
    install_persistent_window_settings(window)
    return window


_APPLICATION_WINDOWS_ATTRIBUTE = "_field_workbench_application_windows"


def application_owned_windows() -> tuple[QWidget, ...]:
    """Return top-level workspaces retained by the QApplication itself."""
    application = QApplication.instance()
    if application is None:
        return ()
    windows = getattr(application, _APPLICATION_WINDOWS_ATTRIBUTE, None)
    if not isinstance(windows, dict):
        return ()
    return tuple(windows.values())


def show_application_window(window: QWidget) -> QWidget:
    """Show a non-modal top-level window whose lifetime is independent of MainWindow.

    Parentless Qt widgets need a strong Python owner even though the window
    manager can display them.  Retaining the widget on QApplication gives it an
    application lifetime without making any MainWindow or scene document its
    owner.  Closing the source document therefore has no effect on the window.
    """
    application = QApplication.instance()
    if application is None:
        raise RuntimeError("A QApplication is required before showing a workspace window.")

    windows = getattr(application, _APPLICATION_WINDOWS_ATTRIBUTE, None)
    if not isinstance(windows, dict):
        windows = {}
        setattr(application, _APPLICATION_WINDOWS_ATTRIBUTE, windows)

    window.setParent(None, Qt.WindowType.Window)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    window.setWindowModality(Qt.WindowModality.NonModal)
    set_modal = getattr(window, "setModal", None)
    if callable(set_modal):
        set_modal(False)
    # Apply the ordinary top-level flags last.  QDialog modality/parent changes
    # can otherwise cause a platform plugin to normalize the native title bar
    # after WindowCloseButtonHint was requested.
    window.setWindowFlags(_APPLICATION_WINDOW_FLAGS)
    install_persistent_window_settings(window)

    key = id(window)
    windows[key] = window

    def release_window(_object=None, *, window_key=key, owner=application) -> None:
        registry = getattr(owner, _APPLICATION_WINDOWS_ATTRIBUTE, None)
        if isinstance(registry, dict):
            registry.pop(window_key, None)

    window.destroyed.connect(release_window)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
