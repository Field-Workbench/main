"""Shared Qt lifecycle and validation behavior for Waveform Editor windows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

from .theme import FontSet
from .widgets import NumericField, NumericSpec


class BaseWindow:
    """Cooperative mixin used by both the main window and every dialog.

    Concrete windows inherit this class before `QMainWindow` or `QDialog` in
    their MRO. Components such as plots and random-step editors use composition.
    """

    def __init__(
        self,
        owner: "BaseWindow | None",
        *,
        title: str,
        modal: bool = False,
        delete_on_close: bool = False,
    ) -> None:
        qt_parent = owner if isinstance(owner, QWidget) else None
        super().__init__(qt_parent)  # type: ignore[misc]
        if not isinstance(self, QWidget):
            raise TypeError("BaseWindow must be combined with a QWidget subclass.")

        self.owner = owner
        self.window = self
        self.master = self
        self._modal = modal
        self._fields: list[NumericField] = []
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, delete_on_close)

        app = QApplication.instance()
        if not isinstance(app, QApplication):
            raise RuntimeError("Create QApplication before constructing a Waveform Editor window.")
        self.fonts = owner.fonts if owner is not None else FontSet.create(app)

        if isinstance(self, QDialog):
            self.setModal(modal)
            if owner is not None:
                self.setWindowModality(Qt.WindowModality.WindowModal)

    def numeric_field(
        self,
        parent: QWidget,
        spec: NumericSpec,
        *,
        on_change: Callable[[], None] | None = None,
        label_position: Literal["top", "left"] = "top",
        fixed_input_width: int | None = None,
    ) -> NumericField:
        field = NumericField(
            parent,
            spec,
            on_change=on_change,
            label_position=label_position,
            fixed_input_width=fixed_input_width,
        )
        self._fields.append(field)
        return field

    def unregister_fields(self, fields: Sequence[NumericField]) -> None:
        self._fields = [field for field in self._fields if field not in fields]

    def fields_valid(self, fields: Sequence[NumericField] | None = None) -> bool:
        selected = self._fields if fields is None else fields
        results = [field.mark_validity() for field in selected]
        return all(results)

    def show_window(self, *, modal: bool | None = None) -> None:
        self.on_show()
        use_modal = self._modal if modal is None else modal
        if isinstance(self, QDialog) and use_modal:
            self.setWindowModality(Qt.WindowModality.WindowModal)
            self.open()
        else:
            self.show()
        app = QApplication.instance()
        if isinstance(app, QApplication) and app.platformName() != "offscreen":
            self.raise_()
            self.activateWindow()
        QTimer.singleShot(0, self._center_on_owner)

    def on_show(self) -> None:
        """Synchronize state immediately before a window appears."""

    def _center_on_owner(self) -> None:
        if self.owner is None or not isinstance(self.owner, QWidget):
            return
        owner_geometry = self.owner.frameGeometry()
        geometry = self.frameGeometry()
        geometry.moveCenter(owner_geometry.center())
        self.move(geometry.topLeft())

    def close_window(self) -> None:
        self.close()

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)  # type: ignore[arg-type]

    def show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)  # type: ignore[arg-type]

    def confirm(self, title: str, message: str) -> bool:
        answer = QMessageBox.question(
            self,  # type: ignore[arg-type]
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes


class BaseDialog(BaseWindow, QDialog):
    """Concrete Qt dialog base retaining the common `BaseWindow` contract."""

    def __init__(
        self,
        owner: BaseWindow,
        *,
        title: str,
        modal: bool = True,
        delete_on_close: bool = False,
    ) -> None:
        super().__init__(
            owner,
            title=title,
            modal=modal,
            delete_on_close=delete_on_close,
        )
