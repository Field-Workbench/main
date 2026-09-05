"""Shared, application-wide Field Workbench light and dark themes.

Both appearances keep the same Workbench widget geometry, spacing, typography,
and control treatment. Light uses the established pre-0.13.0.254 stylesheet and
palette; Dark uses the consolidated dark stylesheet introduced in 0.13.0.254.
Keeping their colour implementations separate prevents future Dark fixes from
changing Light layout or vice versa.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QWidget


DEFAULT_THEME = "light"
THEME_CHOICES: tuple[tuple[str, str], ...] = (("Light", "light"), ("Dark", "dark"))
_THEME_IDS = frozenset(theme_id for _label, theme_id in THEME_CHOICES)
_THEME_SETTING_KEY = "appearance/theme"
_THEME_PROPERTY = "fieldWorkbenchTheme"


LIGHT_THEME_COLORS = {
    "window": "#f5f7fa",
    "panel": "#ffffff",
    "panel_alternate": "#f8fafc",
    "text": "#1f2937",
    "heading": "#0f172a",
    "muted": "#64748b",
    "disabled": "#94a3b8",
    "border": "#d9e0e8",
    "control_border": "#cbd5e1",
    "selection": "#dbeafe",
    "selection_text": "#0f172a",
    "accent": "#1769c2",
    "accent_dark": "#14599f",
}

# These are the literal dark-theme values used by WaveBuilder 2.0.0.16. Extra
# semantic names keep Field Workbench's more extensive stylesheet readable
# without inventing a second, nearly-identical dark palette.
DARK_THEME_COLORS = {
    "window": "#20252b",
    "panel": "#2a3037",
    "panel_alternate": "#252b31",
    "text": "#e4e8ec",
    "heading": "#d9dee3",
    "muted": "#9ca7b1",
    "disabled": "#7d8791",
    "border": "#48535e",
    "control_border": "#48535e",
    "selection": "#344c5e",
    "selection_text": "#ffffff",
    "accent": "#3d7197",
    "accent_dark": "#344c5e",
    "button": "#303740",
    "hover": "#394550",
    "hover_border": "#6589a4",
    "highlight_hover": "#4a82aa",
    "focus": "#6299be",
    "disabled_bg": "#292f35",
    "disabled_border": "#3b444d",
    "invalid_bg": "#4a2f32",
    "invalid_border": "#b85c62",
    "status_bg": "#252b31",
    "status_text": "#cbd2d8",
    "status_error_bg": "#4a2b2e",
    "status_error_text": "#f0b8bb",
    "toolbar_bg": "#252b31",
    "toolbar_hover": "#35414b",
    "toolbar_hover_border": "#566a79",
    "menu_bg": "#252b31",
    "menu_selected": "#344c5e",
    "warning_text": "#e6ad63",
    "danger_text": "#e58e92",
}

PLOT_THEME_COLORS = {
    "light": {
        "figure": "#ffffff",
        "axes": "#ffffff",
        "text": "#20262d",
        "major_grid": "#c8d0d8",
        "minor_grid": "#e1e6eb",
        "reference": "#6f7a84",
        "sample": "#91a9ba",
        "canvas_outer": "#f7f9fb",
    },
    "dark": {
        "figure": "#aeb3b8",
        "axes": "#aeb3b8",
        "text": "#14181c",
        "major_grid": "#878f97",
        "minor_grid": "#9aa1a8",
        "reference": "#545d66",
        "sample": "#5e7b90",
        "canvas_outer": "#9ea4aa",
    },
}


def _normalized_theme(theme: object) -> str:
    value = str(theme).strip().lower()
    return value if value in _THEME_IDS else DEFAULT_THEME


def load_theme_preference(settings: QSettings | None = None) -> str:
    """Read the saved appearance choice, falling back safely to Light."""

    source = settings if settings is not None else QSettings()
    return _normalized_theme(source.value(_THEME_SETTING_KEY, DEFAULT_THEME))


def save_theme_preference(
    theme: str,
    settings: QSettings | None = None,
) -> None:
    """Persist a validated theme choice for future launches."""

    selected = _normalized_theme(theme)
    if selected != theme:
        raise ValueError(f"Unsupported Field Workbench theme: {theme}")
    destination = settings if settings is not None else QSettings()
    destination.setValue(_THEME_SETTING_KEY, selected)
    destination.sync()


def current_theme(application: QApplication | None = None) -> str:
    """Return the theme currently installed on the QApplication."""

    app = application if application is not None else QApplication.instance()
    if not isinstance(app, QApplication):
        return DEFAULT_THEME
    return _normalized_theme(app.property(_THEME_PROPERTY))


def theme_colors(theme: str | None = None) -> dict[str, str]:
    """Return the application chrome colours for the current or named theme."""

    selected = current_theme() if theme is None else _normalized_theme(theme)
    return DARK_THEME_COLORS if selected == "dark" else LIGHT_THEME_COLORS


def plot_theme_colors(theme: str | None = None) -> dict[str, str]:
    """Return WaveBuilder-compatible high-contrast plot-surface colours."""

    selected = current_theme() if theme is None else _normalized_theme(theme)
    return PLOT_THEME_COLORS[selected]


def themed_icon(source: QIcon, theme: str | None = None) -> QIcon:
    """Return a palette-aware copy of a standard application icon.

    Fusion's built-in file and history icons are dark bitmaps even when the
    application palette is dark.  Keep those originals in Light mode and tint
    their alpha masks with WaveBuilder's light silver foreground in Dark mode.
    """

    selected = current_theme() if theme is None else _normalized_theme(theme)
    if theme is not None and selected != theme:
        raise ValueError(f"Unsupported Field Workbench theme: {theme}")
    if selected != "dark" or source.isNull():
        return QIcon(source)

    def tinted_pixmap(size: int, colour: str):
        pixmap = source.pixmap(size, size)
        if pixmap.isNull():
            return pixmap
        tinted = pixmap.copy()
        painter = QPainter(tinted)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(tinted.rect(), QColor(colour))
        painter.end()
        return tinted

    result = QIcon()
    for size in (16, 20, 24, 32, 48):
        result.addPixmap(
            tinted_pixmap(size, DARK_THEME_COLORS["heading"]),
            QIcon.Mode.Normal,
            QIcon.State.Off,
        )
        result.addPixmap(
            tinted_pixmap(size, DARK_THEME_COLORS["disabled"]),
            QIcon.Mode.Disabled,
            QIcon.State.Off,
        )
    return result if not result.isNull() else QIcon(source)


def build_workbench_light_palette() -> QPalette:
    """Return a complete light palette independent of the system theme."""
    colours = LIGHT_THEME_COLORS
    palette = QPalette()

    role_colours = {
        QPalette.ColorRole.Window: colours["window"],
        QPalette.ColorRole.WindowText: colours["text"],
        QPalette.ColorRole.Base: colours["panel"],
        QPalette.ColorRole.AlternateBase: colours["panel_alternate"],
        QPalette.ColorRole.ToolTipBase: colours["panel"],
        QPalette.ColorRole.ToolTipText: colours["text"],
        QPalette.ColorRole.Text: colours["text"],
        QPalette.ColorRole.Button: colours["panel"],
        QPalette.ColorRole.ButtonText: colours["text"],
        QPalette.ColorRole.BrightText: "#b42318",
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: "#e2e8f0",
        QPalette.ColorRole.Mid: colours["control_border"],
        QPalette.ColorRole.Dark: colours["disabled"],
        QPalette.ColorRole.Shadow: colours["muted"],
        QPalette.ColorRole.Highlight: colours["selection"],
        QPalette.ColorRole.HighlightedText: colours["selection_text"],
        QPalette.ColorRole.Link: colours["accent"],
        QPalette.ColorRole.LinkVisited: "#0f4f94",
        QPalette.ColorRole.PlaceholderText: colours["disabled"],
    }
    for role, colour in role_colours.items():
        palette.setColor(role, QColor(colour))

    # Qt can retain disabled colours from the platform palette unless they are
    # supplied explicitly.  Keep disabled controls light and readable too.
    disabled_roles = {
        QPalette.ColorRole.Window: colours["window"],
        QPalette.ColorRole.WindowText: colours["disabled"],
        QPalette.ColorRole.Base: colours["panel_alternate"],
        QPalette.ColorRole.AlternateBase: colours["panel_alternate"],
        QPalette.ColorRole.Text: colours["disabled"],
        QPalette.ColorRole.Button: colours["panel_alternate"],
        QPalette.ColorRole.ButtonText: colours["disabled"],
        QPalette.ColorRole.Highlight: "#e2e8f0",
        QPalette.ColorRole.HighlightedText: colours["muted"],
        QPalette.ColorRole.PlaceholderText: colours["disabled"],
    }
    for role, colour in disabled_roles.items():
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            role,
            QColor(colour),
        )

    return palette


def build_workbench_dark_palette() -> QPalette:
    """Return Field Workbench's WaveBuilder-compatible dark palette."""

    colours = DARK_THEME_COLORS
    palette = QPalette()
    role_colours = {
        QPalette.ColorRole.Window: colours["window"],
        QPalette.ColorRole.WindowText: colours["text"],
        QPalette.ColorRole.Base: colours["panel"],
        QPalette.ColorRole.AlternateBase: colours["panel_alternate"],
        QPalette.ColorRole.ToolTipBase: "#e8ebee",
        QPalette.ColorRole.ToolTipText: colours["window"],
        QPalette.ColorRole.Text: colours["text"],
        QPalette.ColorRole.Button: colours["button"],
        QPalette.ColorRole.ButtonText: colours["text"],
        QPalette.ColorRole.BrightText: colours["danger_text"],
        QPalette.ColorRole.Light: colours["hover_border"],
        QPalette.ColorRole.Midlight: colours["border"],
        QPalette.ColorRole.Mid: colours["border"],
        QPalette.ColorRole.Dark: colours["disabled_border"],
        QPalette.ColorRole.Shadow: "#14181c",
        QPalette.ColorRole.Highlight: colours["accent"],
        QPalette.ColorRole.HighlightedText: colours["selection_text"],
        QPalette.ColorRole.Link: colours["focus"],
        QPalette.ColorRole.LinkVisited: colours["hover_border"],
        QPalette.ColorRole.PlaceholderText: colours["muted"],
    }
    for role, colour in role_colours.items():
        palette.setColor(role, QColor(colour))

    disabled_roles = {
        QPalette.ColorRole.Window: colours["window"],
        QPalette.ColorRole.WindowText: colours["disabled"],
        QPalette.ColorRole.Base: colours["disabled_bg"],
        QPalette.ColorRole.AlternateBase: colours["disabled_bg"],
        QPalette.ColorRole.Text: colours["disabled"],
        QPalette.ColorRole.Button: colours["disabled_bg"],
        QPalette.ColorRole.ButtonText: colours["disabled"],
        QPalette.ColorRole.Highlight: colours["disabled_border"],
        QPalette.ColorRole.HighlightedText: colours["muted"],
        QPalette.ColorRole.PlaceholderText: colours["disabled"],
    }
    for role, colour in disabled_roles.items():
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(colour))
    return palette


def build_workbench_palette(theme: str = DEFAULT_THEME) -> QPalette:
    """Return a complete palette for a validated Field Workbench theme."""

    selected = _normalized_theme(theme)
    if selected != theme:
        raise ValueError(f"Unsupported Field Workbench theme: {theme}")
    if selected == "dark":
        return build_workbench_dark_palette()
    return build_workbench_light_palette()


LIGHT_WORKBENCH_STYLE_SHEET = """
/* Global surfaces ------------------------------------------------------- */
/* Retain this compact rule for compatibility with existing theme checks. */
QMainWindow, QDialog { background: #f5f7fa; }
QMainWindow, QDialog, QMessageBox, QProgressDialog {
    background-color: #f5f7fa;
    color: #1f2937;
}
QWidget {
    font-size: 10pt;
    color: #1f2937;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QLabel { background-color: transparent; }
QScrollArea, QAbstractScrollArea {
    background-color: #f5f7fa;
    border: none;
}
QScrollArea > QWidget > QWidget { background-color: #f5f7fa; }
QSplitter::handle { background-color: #e2e8f0; }
QSplitter::handle:hover { background-color: #cbd5e1; }

/* Brain View workspaces -----------------------------------------------
   Selector and rendered result pages are inserted into splitter/tab stacks,
   where plain QWidget containers are otherwise transparent. Give all Brain
   View workspace surfaces semantic selectors so the application-wide Light /
   Dark translation paints them consistently, including widgets created after
   the theme was installed. Auto-fill on the corresponding widgets provides a
   palette-backed fallback for native styles that skip QWidget background QSS. */
QWidget[brainWorkspacePage="true"],
QWidget#brainSelectorTab,
QWidget#brainResultTab {
    background-color: #f5f7fa;
    color: #1f2937;
}
QWidget[brainRegionPanel="true"],
QWidget#brainSelectorRegionPanel,
QWidget#brainResultRegionPanel {
    background-color: #ffffff;
    color: #1f2937;
}
QTabWidget#brainViewTabs::pane {
    background-color: #f5f7fa;
    border-color: #d9e0e8;
}

/* Menus and toolbars ---------------------------------------------------- */
QMenuBar {
    background-color: #f5f7fa;
    color: #1f2937;
    border-bottom: 1px solid #d9e0e8;
}
QMenuBar::item {
    background-color: transparent;
    padding: 5px 8px;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background-color: #dbeafe;
    color: #0f172a;
}
QMenu {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    padding: 4px;
}
QMenu::item { padding: 6px 28px 6px 22px; }
QMenu::item:selected { background-color: #dbeafe; color: #0f172a; }
QMenu::item:disabled { color: #94a3b8; }
QMenu::separator {
    height: 1px;
    background-color: #e2e8f0;
    margin: 4px 7px;
}
QToolBar {
    background-color: #ffffff;
    border: none;
    border-bottom: 1px solid #d9e0e8;
    spacing: 5px;
    padding: 5px;
}
QToolBar::separator {
    background-color: #d9e0e8;
    width: 1px;
    margin: 5px 4px;
}
QToolButton {
    background-color: transparent;
    color: #1f2937;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 6px 9px;
}
QToolButton:hover, QToolButton:checked {
    background-color: #eff6ff;
    border-color: #93b9e8;
}
QToolButton:pressed { background-color: #dbeafe; }
QToolTip {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #94a3b8;
    padding: 4px;
}

/* Cards and frames ------------------------------------------------------ */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 7px;
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
    color: #475569;
    background-color: #f5f7fa;
}
QFrame#sceneTools {
    background-color: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 7px;
}

/* Buttons --------------------------------------------------------------- */
QPushButton {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 9px;
}
QPushButton:hover { background-color: #eff6ff; border-color: #79aaf0; }
QPushButton:pressed { background-color: #dbeafe; }
QPushButton:checked { background-color: #dbeafe; border-color: #79aaf0; }
QPushButton:disabled {
    color: #94a3b8;
    background-color: #f8fafc;
    border-color: #e2e8f0;
}
QPushButton#primaryButton {
    background-color: #1769c2;
    color: #ffffff;
    border: 1px solid #14599f;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #0f5cad; }
QPushButton#primaryButton:pressed { background-color: #0b4f96; }
QPushButton#primaryButton:disabled {
    background-color: #b8cce3;
    color: #f8fafc;
    border-color: #a7bdd6;
}
QPushButton#accentButton {
    background-color: #eff6ff;
    color: #0f4f94;
    border: 1px solid #93b9e8;
    font-weight: 600;
}
QPushButton#accentButton:hover { background-color: #dbeafe; border-color: #5794d9; }
QPushButton#dangerButton { color: #b42318; }
QDialogButtonBox { background-color: transparent; }

/* Editors and item views ------------------------------------------------ */
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #5794d9;
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled,
QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #f8fafc;
    color: #94a3b8;
    border-color: #e2e8f0;
}
QComboBox::drop-down {
    background-color: #f8fafc;
    border: none;
    border-left: 1px solid #e2e8f0;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
    outline: 0;
}
QAbstractItemView, QListView, QTreeView, QTableView,
QListWidget, QTreeWidget, QTableWidget {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d9e0e8;
    alternate-background-color: #f8fafc;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
    gridline-color: #e2e8f0;
    outline: 0;
}
QTreeWidget, QListWidget, QTableWidget { border-radius: 6px; }
QAbstractItemView::item:selected { background-color: #dbeafe; color: #0f172a; }
QAbstractItemView::item:hover { background-color: #eff6ff; color: #0f172a; }
QTableWidget#brainRegionFilterTable::item:hover:!selected {
    background-color: transparent;
    color: #1f2937;
}
QTableWidget#brainRegionFilterTable QComboBox:hover,
QTableWidget#brainRegionFilterTable QAbstractSpinBox:hover {
    background-color: #ffffff;
    color: #1f2937;
    border-color: #cbd5e1;
}
QTableWidget#brainRegionFilterTable QComboBox[brainFilterSelected="true"],
QTableWidget#brainRegionFilterTable QAbstractSpinBox[brainFilterSelected="true"] {
    background-color: #dbeafe;
    color: #0f172a;
    border-color: #93c5fd;
}
QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    border: none;
    border-right: 1px solid #d9e0e8;
    border-bottom: 1px solid #d9e0e8;
    padding: 6px;
    font-weight: 600;
}
QGraphicsView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d9e0e8;
}
QCheckBox, QRadioButton { background-color: transparent; color: #1f2937; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: #94a3b8; }
/* Some dark Linux desktop styles leave the native unchecked box white-on-white
   after the Workbench light palette is applied. Draw that state explicitly;
   the platform keeps drawing its normal visible check mark for the checked state. */
QCheckBox::indicator:unchecked {
    width: 14px;
    height: 14px;
    background-color: #ffffff;
    border: 1px solid #64748b;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked:hover {
    background-color: #eff6ff;
    border-color: #1769c2;
}
QCheckBox::indicator:unchecked:disabled {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}
/* Make radio choices unmistakable on Linux themes: always draw the white
   circular control, with a blue centre for the selected choice. */
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    background-color: #ffffff;
    border: 1px solid #64748b;
    border-radius: 7px;
}
QRadioButton::indicator:unchecked:hover {
    background-color: #eff6ff;
    border-color: #1769c2;
}
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #1769c2, stop:0.36 #1769c2, stop:0.39 #ffffff, stop:1 #ffffff);
    border: 1px solid #64748b;
}
QRadioButton::indicator:disabled {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}

/* Tabs, progress, status, and scrollbars ------------------------------- */
QTabWidget::pane {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    top: -1px;
}
QTabBar { background-color: transparent; }
QTabBar::tab {
    background-color: #f8fafc;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    padding: 6px 11px;
    margin-right: 1px;
}
QTabBar::tab:selected { background-color: #ffffff; color: #0f4f94; font-weight: 600; }
QTabBar::tab:hover:!selected { background-color: #eff6ff; }
QProgressBar {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    text-align: center;
    min-height: 20px;
}
QProgressBar::chunk { background-color: #1769c2; border-radius: 4px; }
QStatusBar {
    background-color: #f8fafc;
    color: #475569;
    border-top: 1px solid #d9e0e8;
}
QStatusBar::item { border: none; }
QScrollBar:vertical {
    background-color: #f1f5f9;
    width: 13px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #cbd5e1;
    min-height: 28px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background-color: #94a3b8; }
QScrollBar:horizontal {
    background-color: #f1f5f9;
    height: 13px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #cbd5e1;
    min-width: 28px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background-color: #94a3b8; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* Named Field Workbench elements --------------------------------------- */
QLabel#sectionTitle {
    color: #64748b;
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 6px;
}
QLabel#playbackRuntime {
    color: #0f172a;
    font-size: 15pt;
    font-weight: 700;
    padding: 5px 0 2px 0;
}
QLabel#inspectorHeading, QLabel#dialogHeading {
    font-size: 16pt;
    font-weight: 650;
    color: #0f172a;
}
QLabel#hint { color: #64748b; }
QLabel#resultValue { font-size: 14pt; font-weight: 650; color: #0f4f94; }
QLabel[statusTone="success"] { color: #15803d; font-weight: 600; }
QLabel[statusTone="warning"] { color: #b45309; font-weight: 600; }
QLabel[statusTone="muted"] { color: #475569; font-weight: 600; }
QLabel#warningPanel {
    background-color: #fff7ed;
    color: #7c2d12;
    border: 1px solid #fdba74;
    border-radius: 4px;
    padding: 8px;
}
QLabel#warningPanelCompact {
    background-color: #fff7ed;
    color: #9a3412;
    border: 1px solid #fdba74;
    border-radius: 5px;
    padding: 7px;
}
QLabel#sceneToolsLabel {
    color: #64748b;
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.8px;
}
"""


WORKBENCH_STYLE_SHEET = """
/* Global surfaces ------------------------------------------------------- */
/* Retain this compact rule for compatibility with existing theme checks. */
QMainWindow, QDialog { background: #f5f7fa; }
QMainWindow, QDialog, QMessageBox, QProgressDialog {
    background-color: #f5f7fa;
    color: #1f2937;
}
QWidget {
    font-size: 10pt;
    color: #1f2937;
    background-color: transparent;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QLabel { background-color: transparent; }
QScrollArea, QAbstractScrollArea {
    background-color: #f5f7fa;
    border: none;
}
QScrollArea > QWidget > QWidget { background-color: #f5f7fa; }
QSplitter::handle { background-color: #e2e8f0; }
QSplitter::handle:hover { background-color: #cbd5e1; }

/* Menus and toolbars ---------------------------------------------------- */
QMenuBar {
    background-color: #f5f7fa;
    color: #1f2937;
    border-bottom: 1px solid #d9e0e8;
}
QMenuBar::item {
    background-color: transparent;
    padding: 5px 8px;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background-color: #dbeafe;
    color: #0f172a;
}
QMenu {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    padding: 4px;
}
QMenu::item { padding: 6px 28px 6px 22px; }
QMenu::item:selected { background-color: #dbeafe; color: #0f172a; }
QMenu::item:disabled { color: #94a3b8; }
QMenu::separator {
    height: 1px;
    background-color: #e2e8f0;
    margin: 4px 7px;
}
QToolBar {
    background-color: #ffffff;
    border: none;
    border-bottom: 1px solid #d9e0e8;
    spacing: 5px;
    padding: 5px;
}
QToolBar::separator {
    background-color: #d9e0e8;
    width: 1px;
    margin: 5px 4px;
}
QToolButton {
    background-color: transparent;
    color: #1f2937;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 6px 9px;
}
QToolButton:hover, QToolButton:checked {
    background-color: #eff6ff;
    border-color: #93b9e8;
}
QToolButton:pressed { background-color: #dbeafe; }
QToolTip {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #94a3b8;
    padding: 4px;
}

/* Cards and frames ------------------------------------------------------ */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 7px;
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
    color: #475569;
    background-color: #f5f7fa;
}
QFrame#sceneTools {
    background-color: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 7px;
}

/* Buttons --------------------------------------------------------------- */
QPushButton {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 9px;
}
QPushButton:hover { background-color: #eff6ff; border-color: #79aaf0; }
QPushButton:pressed { background-color: #dbeafe; }
QPushButton:checked { background-color: #dbeafe; border-color: #79aaf0; }
QPushButton:disabled {
    color: #94a3b8;
    background-color: #f8fafc;
    border-color: #e2e8f0;
}
QPushButton#primaryButton {
    background-color: #1769c2;
    color: #ffffff;
    border: 1px solid #14599f;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #0f5cad; }
QPushButton#primaryButton:pressed { background-color: #0b4f96; }
QPushButton#primaryButton:disabled {
    background-color: #b8cce3;
    color: #f8fafc;
    border-color: #a7bdd6;
}
QPushButton#accentButton {
    background-color: #eff6ff;
    color: #0f4f94;
    border: 1px solid #93b9e8;
    font-weight: 600;
}
QPushButton#accentButton:hover { background-color: #dbeafe; border-color: #5794d9; }
QPushButton#dangerButton { color: #b42318; }
QDialogButtonBox { background-color: transparent; }

/* Editors and item views ------------------------------------------------ */
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #5794d9;
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled,
QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #f8fafc;
    color: #94a3b8;
    border-color: #e2e8f0;
}
QComboBox::drop-down {
    background-color: #f8fafc;
    border: none;
    border-left: 1px solid #e2e8f0;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
    outline: 0;
}
QAbstractItemView, QListView, QTreeView, QTableView,
QListWidget, QTreeWidget, QTableWidget {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d9e0e8;
    alternate-background-color: #f8fafc;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
    gridline-color: #e2e8f0;
    outline: 0;
}
QTreeWidget, QListWidget, QTableWidget { border-radius: 6px; }
QAbstractItemView::item:selected { background-color: #dbeafe; color: #0f172a; }
QAbstractItemView::item:hover { background-color: #eff6ff; color: #0f172a; }
QTableWidget#brainRegionFilterTable::item:hover:!selected {
    background-color: transparent;
    color: #1f2937;
}
QTableWidget#brainRegionFilterTable QComboBox:hover,
QTableWidget#brainRegionFilterTable QAbstractSpinBox:hover {
    background-color: #ffffff;
    color: #1f2937;
    border-color: #cbd5e1;
}
QTableWidget#brainRegionFilterTable QComboBox[brainFilterSelected="true"],
QTableWidget#brainRegionFilterTable QAbstractSpinBox[brainFilterSelected="true"] {
    background-color: #dbeafe;
    color: #0f172a;
    border-color: #93c5fd;
}
QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    border: none;
    border-right: 1px solid #d9e0e8;
    border-bottom: 1px solid #d9e0e8;
    padding: 6px;
    font-weight: 600;
}
QGraphicsView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d9e0e8;
}
QCheckBox, QRadioButton { background-color: transparent; color: #1f2937; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: #94a3b8; }
/* Some dark Linux desktop styles leave the native unchecked box white-on-white
   after the Workbench light palette is applied. Draw that state explicitly;
   the platform keeps drawing its normal visible check mark for the checked state. */
QCheckBox::indicator:unchecked {
    width: 14px;
    height: 14px;
    background-color: #ffffff;
    border: 1px solid #64748b;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked:hover {
    background-color: #eff6ff;
    border-color: #1769c2;
}
QCheckBox::indicator:unchecked:disabled {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}
/* Make radio choices unmistakable on Linux themes: always draw the white
   circular control, with a blue centre for the selected choice. */
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    background-color: #ffffff;
    border: 1px solid #64748b;
    border-radius: 7px;
}
QRadioButton::indicator:unchecked:hover {
    background-color: #eff6ff;
    border-color: #1769c2;
}
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #1769c2, stop:0.36 #1769c2, stop:0.39 #ffffff, stop:1 #ffffff);
    border: 1px solid #64748b;
}
QRadioButton::indicator:disabled {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}

/* Tabs, progress, status, and scrollbars ------------------------------- */
QTabWidget::pane {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    top: -1px;
}
QTabBar { background-color: transparent; }
QTabBar::tab {
    background-color: #f8fafc;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    padding: 6px 11px;
    margin-right: 1px;
}
QTabBar::tab:selected { background-color: #ffffff; color: #0f4f94; font-weight: 600; }
QTabBar::tab:hover:!selected { background-color: #eff6ff; }
QProgressBar {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    text-align: center;
    min-height: 20px;
}
QProgressBar::chunk { background-color: #1769c2; border-radius: 4px; }
QStatusBar {
    background-color: #f8fafc;
    color: #475569;
    border-top: 1px solid #d9e0e8;
}
QStatusBar::item { border: none; }
QScrollBar:vertical {
    background-color: #f1f5f9;
    width: 13px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #cbd5e1;
    min-height: 28px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background-color: #94a3b8; }
QScrollBar:horizontal {
    background-color: #f1f5f9;
    height: 13px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #cbd5e1;
    min-width: 28px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background-color: #94a3b8; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* Named Field Workbench elements --------------------------------------- */
QLabel#sectionTitle {
    color: #64748b;
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 6px;
}
QLabel#playbackRuntime {
    color: #0f172a;
    font-size: 15pt;
    font-weight: 700;
    padding: 5px 0 2px 0;
}
QLabel#inspectorHeading, QLabel#dialogHeading {
    font-size: 16pt;
    font-weight: 650;
    color: #0f172a;
}
QLabel#hint { color: #64748b; }
QLabel#resultValue { font-size: 14pt; font-weight: 650; color: #0f4f94; }
QLabel[statusTone="success"] { color: #15803d; font-weight: 600; }
QLabel[statusTone="warning"] { color: #b45309; font-weight: 600; }
QLabel[statusTone="muted"] { color: #475569; font-weight: 600; }
QLabel#warningPanel {
    background-color: #fff7ed;
    color: #7c2d12;
    border: 1px solid #fdba74;
    border-radius: 4px;
    padding: 8px;
}
QLabel#warningPanelCompact {
    background-color: #fff7ed;
    color: #9a3412;
    border: 1px solid #fdba74;
    border-radius: 5px;
    padding: 7px;
}
QLabel#sceneToolsLabel {
    color: #64748b;
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.8px;
}
"""


def _build_dark_style_sheet() -> str:
    """Translate the established light selectors onto WaveBuilder's dark tokens."""

    # Protect only the foreground ``color`` property before translating white
    # surfaces. A plain string replacement also matched the ``color`` suffix in
    # ``background-color`` and was the root cause of white Light-theme panels
    # surviving unchanged in Dark mode.
    sheet = re.sub(
        r"(?<![-\w])color\s*:\s*#ffffff\s*;",
        "color: __FIELD_WORKBENCH_ACCENT_FOREGROUND__;",
        WORKBENCH_STYLE_SHEET,
    )
    replacements = (
        ("#f5f7fa", "#20252b"),
        ("#ffffff", "#2a3037"),
        ("#f8fafc", "#252b31"),
        ("#f1f5f9", "#292f35"),
        ("#eff6ff", "#394550"),
        ("#e2e8f0", "#3b444d"),
        ("#dbeafe", "#344c5e"),
        ("#d9e0e8", "#48535e"),
        ("#cbd5e1", "#48535e"),
        ("#b8cce3", "#292f35"),
        ("#a7bdd6", "#3b444d"),
        ("#94a3b8", "#7d8791"),
        ("#93b9e8", "#566a79"),
        ("#79aaf0", "#6589a4"),
        ("#64748b", "#9ca7b1"),
        ("#5794d9", "#6299be"),
        ("#475569", "#cbd2d8"),
        ("#1f2937", "#e4e8ec"),
        ("#1769c2", "#3d7197"),
        ("#14599f", "#3d7197"),
        ("#0f5cad", "#4a82aa"),
        ("#0f4f94", "#6299be"),
        ("#0f172a", "#e4e8ec"),
        ("#0b4f96", "#344c5e"),
        ("#b42318", "#e58e92"),
    )
    for light, dark in replacements:
        sheet = sheet.replace(light, dark)
    sheet = sheet.replace("__FIELD_WORKBENCH_ACCENT_FOREGROUND__", "#ffffff")
    return sheet + """

/* WaveBuilder dark-theme semantic overrides --------------------------- */
/* Plain QWidget containers are intentionally transparent in Dark mode, but
   QDialog inherits QWidget and would otherwise become transparent too.  Paint
   top-level dialog surfaces explicitly so Preferences and other dialog-style
   subwindows use a stable dark silver-grey instead of the platform's black
   backing surface. */
QDialog, QMessageBox, QProgressDialog {
    background-color: #2f353b;
}
QToolTip {
    background-color: #e8ebee;
    color: #20252b;
    border-color: #48535e;
}
QMenu, QMenuBar { background-color: #252b31; }
QToolBar { background-color: #252b31; }
QToolButton:hover, QToolButton:checked {
    background-color: #35414b;
    border-color: #566a79;
}
QPushButton { background-color: #303740; }
QPushButton:disabled {
    background-color: #292f35;
    color: #7d8791;
    border-color: #3b444d;
}
QPushButton#primaryButton:disabled {
    background-color: #292f35;
    color: #7d8791;
    border-color: #3b444d;
}
QPushButton#accentButton { color: #6299be; }
QGroupBox::title { color: #d9dee3; }
QAbstractItemView::item:selected, QMenu::item:selected,
QMenuBar::item:selected, QMenuBar::item:pressed {
    color: #ffffff;
}
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit,
QComboBox QAbstractItemView, QAbstractItemView, QListView, QTreeView,
QTableView, QListWidget, QTreeWidget, QTableWidget, QGraphicsView,
QTabWidget::pane, QProgressBar {
    selection-color: #ffffff;
}
QScrollBar:vertical, QScrollBar:horizontal { background-color: #2a3037; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #48535e;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #6589a4;
}
QLabel#resultValue { color: #6299be; }
QLabel[statusTone="success"] { color: #86d39b; }
QLabel[statusTone="warning"] { color: #e6ad63; }
QLabel[statusTone="muted"] { color: #9ca7b1; }
QLabel#warningPanel, QLabel#warningPanelCompact {
    background-color: #292f35;
    color: #e6ad63;
    border-color: #48535e;
}

/* Global dark cards/groups --------------------------------------------
   QGroupBox can retain a native light panel on some Linux Qt styles even
   after the application palette changes.  Make the group body and title
   surface explicit for every window instead of chasing individual dialogs. */
QGroupBox {
    background: #252b31;
    background-color: #252b31;
    border-color: #48535e;
}
QGroupBox::title {
    background: #20252b;
    background-color: #20252b;
    color: #d9dee3;
}

/* Global Dark-mode item-view work surfaces ---------------------------
   Tables already use the medium-silver editor surface.  Tree/list views
   need the same explicit treatment too: platform Qt styles can otherwise
   paint their viewport from a native/light Base brush even while the rest
   of the application is dark.  Keeping this class-level rule here means
   new DRC/result/navigation trees inherit the intended palette without
   requiring one-off object-name selectors. */
QTreeView, QTreeWidget, QListView, QListWidget {
    background-color: #9ea4aa;
    alternate-background-color: #8f969c;
    color: #14181c;
    border-color: #56616b;
    selection-background-color: #3d7197;
    selection-color: #ffffff;
}
QTreeView::item:hover:!selected, QTreeWidget::item:hover:!selected,
QListView::item:hover:!selected, QListWidget::item:hover:!selected {
    background-color: #aab0b5;
}
QTreeView::item:selected, QTreeWidget::item:selected,
QListView::item:selected, QListWidget::item:selected {
    background-color: #3d7197;
    color: #ffffff;
}
QTreeView QHeaderView, QTreeWidget QHeaderView,
QListView QHeaderView, QListWidget QHeaderView {
    background-color: #9ea4aa;
}
QTreeView QHeaderView::section, QTreeWidget QHeaderView::section,
QListView QHeaderView::section, QListWidget QHeaderView::section {
    background-color: #303740;
    color: #e4e8ec;
    border-color: #48535e;
}

/* Tabs ---------------------------------------------------------------
   Keep one application-wide tab treatment.  Earlier releases accumulated
   separate Scene, Measurement, 2D/3D, and Brain selectors solely to work
   around native Light surfaces; those windows now inherit these rules. */
QTabWidget::pane {
    background-color: #252b31;
    border-color: #48535e;
}
QTabBar::tab {
    background-color: #303740;
    color: #cbd2d8;
    border-color: #48535e;
    border-bottom: 1px solid #48535e;
}
QTabBar::tab:selected {
    background-color: #344c5e;
    color: #ffffff;
    border-color: #6589a4;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background-color: #394550;
    color: #ffffff;
}

/* Dark mode deliberately uses a medium-silver editor surface with near-black
   text.  Numeric editors remain immediately legible while still sitting well
   below the former white fields in brightness. */
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #9ea4aa;
    color: #14181c;
    border-color: #56616b;
    selection-background-color: #3d7197;
    selection-color: #ffffff;
}
QAbstractSpinBox QLineEdit {
    background-color: transparent;
    color: #14181c;
    border: none;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #6299be;
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled,
QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #8f969c;
    color: #14181c;
    border-color: #56616b;
}
QAbstractSpinBox:disabled QLineEdit {
    background-color: transparent;
    color: #14181c;
}
QComboBox::drop-down {
    background-color: #8f969c;
    border-left-color: #626b74;
}
QComboBox QAbstractItemView {
    background-color: #b7bcc1;
    color: #14181c;
    selection-background-color: #3d7197;
    selection-color: #ffffff;
}

/* Tables share the same medium-silver work surface as Dark-mode editors.
   Keep headers dark so dense data grids remain easy to scan without the
   dazzling white body inherited from the base Light stylesheet. */
QTableView, QTableWidget {
    background-color: #9ea4aa;
    alternate-background-color: #8f969c;
    color: #14181c;
    border-color: #56616b;
    gridline-color: #6f7880;
    selection-background-color: #3d7197;
    selection-color: #ffffff;
}
QTableView::item, QTableWidget::item {
    color: #14181c;
}
QTableView::item:hover:!selected, QTableWidget::item:hover:!selected {
    background-color: #aab0b5;
    color: #14181c;
}
QTableWidget#brainRegionFilterTable::item:hover:!selected {
    background-color: transparent;
    color: #14181c;
}
QTableWidget#brainRegionFilterTable QComboBox:hover,
QTableWidget#brainRegionFilterTable QAbstractSpinBox:hover {
    background-color: #9ea4aa;
    color: #14181c;
    border-color: #56616b;
}
QTableWidget#brainRegionFilterTable QComboBox[brainFilterSelected="true"],
QTableWidget#brainRegionFilterTable QAbstractSpinBox[brainFilterSelected="true"] {
    background-color: #3d7197;
    color: #ffffff;
    border-color: #3d7197;
}
QTableView::item:selected, QTableWidget::item:selected {
    background-color: #3d7197;
    color: #ffffff;
}
/* Give the header view itself the table work-surface colour.  The section
   rules below still paint actual row/column labels dark, while any unused
   header area beyond the final section stays silver instead of reverting to
   the base Light stylesheet's white background. */
QTableView QHeaderView, QTableWidget QHeaderView {
    background-color: #9ea4aa;
}
QTableView QHeaderView::section, QTableWidget QHeaderView::section {
    background-color: #303740;
    color: #e4e8ec;
    border-color: #48535e;
}
QTableCornerButton::section {
    background-color: #303740;
    border-color: #48535e;
}
"""


DARK_WORKBENCH_STYLE_SHEET = _build_dark_style_sheet()


def workbench_style_sheet(theme: str = DEFAULT_THEME) -> str:
    """Return the complete application stylesheet for a validated theme."""

    selected = _normalized_theme(theme)
    if selected != theme:
        raise ValueError(f"Unsupported Field Workbench theme: {theme}")
    return (
        DARK_WORKBENCH_STYLE_SHEET
        if selected == "dark"
        else LIGHT_WORKBENCH_STYLE_SHEET
    )


def apply_workbench_theme(
    widget: QWidget | None = None,
    theme: str | None = None,
) -> str:
    """Install and refresh the selected palette and stylesheet application-wide."""

    application = QApplication.instance()
    if application is None:
        return DEFAULT_THEME

    selected = current_theme(application) if theme is None else _normalized_theme(theme)
    if theme is not None and selected != theme:
        raise ValueError(f"Unsupported Field Workbench theme: {theme}")
    application.setProperty(_THEME_PROPERTY, selected)

    # Native title bars are drawn by the OS/window manager rather than Qt's
    # stylesheet. Ask supported platforms to match the selected application mode.
    try:
        scheme = Qt.ColorScheme.Dark if selected == "dark" else Qt.ColorScheme.Light
        application.styleHints().setColorScheme(scheme)
    except (AttributeError, RuntimeError):
        pass

    palette = build_workbench_palette(selected)
    # Clear the prior theme first so widgets repolish from the target palette,
    # then install the complete stylesheet for that appearance. Light and Dark
    # intentionally have separate QSS sources but retain the same visual sizing
    # and control language.
    application.setStyleSheet("")
    application.setPalette(palette)
    application.setStyleSheet(workbench_style_sheet(selected))

    # A few platform styles snapshot a top-level palette as it is constructed.
    # Applying the same palette to an already-created window closes that gap;
    # future independent windows inherit the application palette automatically.
    if widget is not None:
        widget.setPalette(palette)
        widget.ensurePolished()

    # Reapply and repolish every existing widget.  This closes the remaining
    # dynamic-switch gap for group boxes, input fields, item views, and named
    # frames that inherited explicit Light colours when they were constructed.
    # Browser-backed and hand-painted views then receive their own theme hook.
    for existing_widget in application.allWidgets():
        try:
            existing_widget.setPalette(palette)
            style = existing_widget.style()
            style.unpolish(existing_widget)
            style.polish(existing_widget)
            existing_widget.updateGeometry()
        except RuntimeError:
            continue
        theme_handler = getattr(existing_widget, "apply_workbench_theme", None)
        if callable(theme_handler):
            try:
                theme_handler(selected)
            except RuntimeError:
                pass
        try:
            existing_widget.update()
        except RuntimeError:
            pass
    return selected
