"""Reusable command strip for interactive 3-D :class:`PlotView` surfaces.

The Workbench has several places that present the same small set of scene-view
commands.  Keeping those buttons in one widget prevents Model Editor, Maps and
specialized viewers from drifting into subtly different camera/export behavior.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from .plot_view import PlotView


class SceneViewControls(QFrame):
    """Shared ``SCENE VIEW`` command bar for a 3-D :class:`PlotView`.

    Native Plotly modebar tools (orbit/turntable/zoom/pan/etc.) remain inside the
    plot itself.  This bar owns only Workbench-level commands: export, Home,
    fullscreen, and the engineering camera presets.
    """

    def __init__(
        self,
        plot: PlotView,
        parent=None,
        *,
        export_callback: Callable[[], None] | None = None,
        home_callback: Callable[[], None] | None = None,
        show_export: bool = True,
        show_fullscreen: bool = True,
        show_presets: bool = True,
        label: str = "SCENE VIEW",
    ) -> None:
        super().__init__(parent)
        self.plot = plot
        self.setObjectName("sceneTools")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(5)

        self.scene_tools_label = QLabel(str(label))
        self.scene_tools_label.setObjectName("sceneToolsLabel")
        layout.addWidget(self.scene_tools_label)
        layout.addSpacing(4)

        self.export_image_button = QPushButton("Export image…")
        self.export_image_button.setToolTip("Save the current 3D view as a high-resolution PNG")
        if export_callback is not None:
            self.export_image_button.clicked.connect(export_callback)
        else:
            self.export_image_button.clicked.connect(self.plot.save_png_with_dialog)
        self.export_image_button.setVisible(bool(show_export))
        layout.addWidget(self.export_image_button)

        self.home_view_button = QPushButton("Home")
        self.home_view_button.setToolTip("Fit the whole scene and restore the isometric view")
        if home_callback is not None:
            self.home_view_button.clicked.connect(home_callback)
        else:
            self.home_view_button.clicked.connect(self.plot.home_view)
        layout.addWidget(self.home_view_button)

        self.fullscreen_view_button = QPushButton("Fullscreen")
        self.fullscreen_view_button.setToolTip(
            "Open a separate true-fullscreen interactive viewer (Esc/F11 closes it)"
        )
        self.fullscreen_view_button.clicked.connect(self.plot.toggle_fullscreen)
        self.fullscreen_view_button.setVisible(bool(show_fullscreen))
        layout.addWidget(self.fullscreen_view_button)

        self.view_preset_buttons: dict[str, QPushButton] = {}
        for text, preset in (("Front", "front"), ("Right", "right"), ("Top", "top")):
            button = QPushButton(text)
            button.setToolTip(f"Fit the scene and show the {text.lower()} engineering view")
            button.clicked.connect(
                lambda checked=False, view_name=preset: self.plot.set_camera_preset(view_name)
            )
            button.setVisible(bool(show_presets))
            self.view_preset_buttons[preset] = button
            layout.addWidget(button)

        layout.addStretch(1)
