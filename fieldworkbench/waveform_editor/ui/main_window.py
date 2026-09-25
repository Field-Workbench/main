"""Project-aware native Field Workbench Waveform Editor.

Waveform authoring uses the same format-neutral document model that playback and
project resources consume. External encodings are translated to native physical samples for playback.
Profiles whose timing is entered at import time retain their raw source samples
and runtime settings in the authoring recipe, so bundled and user-defined
encodings use the same editable import path. Active waveform imports use inline
controls beneath the persistent viewer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ... import __version__
from ...project_context import ProjectContext
from ...theme import themed_icon
from ...waveform_import_dialog import WaveformImportDialog
from ...waveform_io import WaveformDocument, WaveformSeries, write_waveform_document
from ...waveform_table_io import write_waveform_csv
from ...waveform_recipe import (
    AUTHORING_RECIPE_KEY,
    WaveformRecipeError,
    normalize_authoring_recipe,
    recipe_is_fully_generated,
)
from .base_window import BaseWindow
from .native_dialogs import (
    DacExportDialog,
    NativeRigolExportDialog,
)
from .native_editor import NativeWaveformEditor
from .embedded_tools import EmbeddedGeneratorPanel
from .embedded_imports import EmbeddedRuntimeFormatPanel, EmbeddedWaveformImportPanel
from .widgets import StatusBar, action_button, waveform_button


_RESOURCE_ID_ROLE = Qt.ItemDataRole.UserRole


class WaveformEditorWindow(BaseWindow, QMainWindow):
    """Native waveform editor attached to one project context."""

    projectResourcesChanged = Signal()

    def __init__(self, project_context: ProjectContext) -> None:
        self.project_context = project_context
        self.current_waveform_id: str | None = None
        self._standard_icon_specs: list[tuple[QAction, QStyle.StandardPixmap]] = []
        self._task_mode: str | None = None
        self._task_original_id: str | None = None
        self._working_name: str | None = None
        self._edit_metadata_baseline: dict[str, str] = {}
        self._edit_form_baseline: dict[str, str] = {}
        self._encoding_setup_active = False
        self._encoding_confirmed = False
        self._encoding_edit_snapshot: dict[str, object] | None = None
        self._project_close_prepared = False
        super().__init__(
            None,
            title=f"Waveform Editor — {project_context.display_name()}",
            modal=False,
            delete_on_close=False,
        )
        self.setObjectName("waveformEditorWindow")
        self.resize(1180, 800)
        self.setMinimumSize(940, 640)
        self._build_ui()
        self._create_actions()
        self._create_menus_and_toolbar()
        self.refresh_project_resources()

    # --- construction -------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("waveformEditorCentral")
        self.setCentralWidget(central)
        body = QVBoxLayout(central)
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(8)

        # Persistent workflow strip immediately beneath the menu bar.  It keeps
        # the current editor mode visible while placing authoring actions outside
        # the resizable content panes, so Notes can collapse all the way closed.
        self.workflow_bar = QFrame(central)
        self.workflow_bar.setObjectName("waveformWorkflowBar")
        self.workflow_bar.setFrameShape(QFrame.Shape.Panel)
        self.workflow_bar.setFrameShadow(QFrame.Shadow.Sunken)
        self.workflow_bar.setLineWidth(1)
        workflow_layout = QGridLayout(self.workflow_bar)
        workflow_layout.setContentsMargins(6, 4, 6, 4)
        workflow_layout.setHorizontalSpacing(8)
        workflow_layout.setColumnStretch(0, 1)
        workflow_layout.setColumnStretch(1, 1)
        workflow_layout.setColumnStretch(2, 1)

        self.workflow_mode_label = QLabel("Waveform navigator", self.workflow_bar)
        self.workflow_mode_label.setObjectName("waveformWorkflowMode")
        self.workflow_mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_font = self.workflow_mode_label.font()
        mode_font.setBold(True)
        self.workflow_mode_label.setFont(mode_font)
        workflow_layout.addWidget(self.workflow_mode_label, 0, 1)

        self.toggle_notes_button = QPushButton("Show / hide notes", self.workflow_bar)
        self.toggle_notes_button.setObjectName("waveformToggleNotesButton")
        self.toggle_notes_button.clicked.connect(self._toggle_notes_panel)
        workflow_layout.addWidget(
            self.toggle_notes_button, 0, 2, Qt.AlignmentFlag.AlignRight
        )
        body.addWidget(self.workflow_bar, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        body.addWidget(splitter, 1)

        # Left: browse mode shows the project catalogue.  Once a waveform-level
        # definition is confirmed, authoring shows that definition here as a
        # read-only encoding summary with an explicit Edit action.
        self.left_stack = QStackedWidget(splitter)
        self.left_stack.setObjectName("waveformEditorLeftStack")
        self.left_stack.setMinimumWidth(255)
        self.left_stack.setMaximumWidth(360)

        self.catalogue_group = QGroupBox("Waveform catalogue", self.left_stack)
        catalogue_layout = QVBoxLayout(self.catalogue_group)
        catalogue_layout.setSpacing(8)

        self.resource_list = QListWidget(self.catalogue_group)
        self.resource_list.setObjectName("waveformProjectResourceList")
        self.resource_list.setToolTip("Select a waveform to preview it.")
        self.resource_list.currentItemChanged.connect(self._catalogue_selection_changed)
        catalogue_layout.addWidget(self.resource_list, 1)

        catalogue_buttons = QGridLayout()
        catalogue_buttons.setHorizontalSpacing(6)
        catalogue_buttons.setVerticalSpacing(6)
        self.create_resource_button = QPushButton("Create new", self.catalogue_group)
        self.create_resource_button.clicked.connect(self.new_waveform)
        self.edit_button = action_button(
            "Edit", self.catalogue_group, self.edit_selected_resource, role="primary"
        )
        self.rename_resource_button = QPushButton("Rename", self.catalogue_group)
        self.rename_resource_button.clicked.connect(self.rename_selected_resource)
        self.duplicate_resource_button = QPushButton("Duplicate", self.catalogue_group)
        self.duplicate_resource_button.clicked.connect(self.duplicate_selected_resource)
        self.import_resource_button = QPushButton("Import…", self.catalogue_group)
        self.import_resource_button.clicked.connect(self.import_native_waveform)
        self.export_button = action_button(
            "Export…", self.catalogue_group, self.export_selected_waveform
        )
        self.delete_resource_button = QPushButton("Delete", self.catalogue_group)
        self.delete_resource_button.clicked.connect(self.delete_selected_resource)
        self.close_button = QPushButton("Close", self.catalogue_group)
        self.close_button.clicked.connect(self.close)
        catalogue_buttons.addWidget(self.create_resource_button, 0, 0)
        catalogue_buttons.addWidget(self.edit_button, 0, 1)
        catalogue_buttons.addWidget(self.rename_resource_button, 1, 0)
        catalogue_buttons.addWidget(self.duplicate_resource_button, 1, 1)
        catalogue_buttons.addWidget(self.import_resource_button, 2, 0)
        catalogue_buttons.addWidget(self.export_button, 2, 1)
        catalogue_buttons.addWidget(self.delete_resource_button, 3, 0, 1, 2)
        catalogue_buttons.addWidget(self.close_button, 4, 0, 1, 2)
        catalogue_layout.addLayout(catalogue_buttons)

        # Mirror the selected-waveform catalogue commands in the list context
        # menu.  These actions intentionally share the same enable/disable state
        # as their visible navigation buttons.
        self.edit_resource_action = QAction("Edit", self.resource_list)
        self.edit_resource_action.triggered.connect(self.edit_selected_resource)
        self.rename_resource_action = QAction("Rename", self.resource_list)
        self.rename_resource_action.triggered.connect(self.rename_selected_resource)
        self.duplicate_resource_action = QAction("Duplicate", self.resource_list)
        self.duplicate_resource_action.triggered.connect(self.duplicate_selected_resource)
        self.export_resource_action = QAction("Export…", self.resource_list)
        self.export_resource_action.triggered.connect(self.export_selected_waveform)
        self.delete_resource_action = QAction("Delete", self.resource_list)
        self.delete_resource_action.setShortcut(QKeySequence("Delete"))
        self.delete_resource_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.delete_resource_action.triggered.connect(self.delete_selected_resource)
        for action in (
            self.edit_resource_action,
            self.rename_resource_action,
            self.duplicate_resource_action,
            self.export_resource_action,
            self.delete_resource_action,
        ):
            self.resource_list.addAction(action)
        self.resource_list.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.left_stack.addWidget(self.catalogue_group)

        # Once a waveform definition has been confirmed, the left column shows
        # it as read-only waveform-level metadata.  Authoring tools inherit this
        # encoding; changing it is an explicit action rather than something each
        # generator/importer can renegotiate.
        self.edit_options_group = QGroupBox("Waveform encoding", self.left_stack)
        edit_options_layout = QVBoxLayout(self.edit_options_group)
        edit_options_layout.setSpacing(8)
        encoding_form = QFormLayout()
        encoding_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        encoding_form.setVerticalSpacing(8)

        def encoding_value(object_name: str) -> QLabel:
            label = QLabel("—", self.edit_options_group)
            label.setObjectName(object_name)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return label

        self.encoding_name = encoding_value("waveformEncodingName")
        self.encoding_quantity = encoding_value("waveformEncodingSignalType")
        self.encoding_units = encoding_value("waveformEncodingUnits")
        self.encoding_slew = encoding_value("waveformEncodingSlew")
        self.encoding_output_resistance = encoding_value("waveformEncodingOutputResistance")
        self.encoding_slew_label = QLabel("Current slew rate", self.edit_options_group)
        self.encoding_output_resistance_label = QLabel("Output resistance", self.edit_options_group)
        self.encoding_output_resistance_label.setToolTip(
            "Series source resistance for a voltage waveform. 0 Ω represents an ideal voltage source."
        )

        encoding_form.addRow("Name", self.encoding_name)
        encoding_form.addRow("Signal type", self.encoding_quantity)
        encoding_form.addRow("Units", self.encoding_units)
        encoding_form.addRow(self.encoding_slew_label, self.encoding_slew)
        encoding_form.addRow(self.encoding_output_resistance_label, self.encoding_output_resistance)
        edit_options_layout.addLayout(encoding_form)

        self.edit_encoding_button = QPushButton("Edit", self.edit_options_group)
        self.edit_encoding_button.setToolTip("Edit the waveform-level signal encoding and electrical source settings.")
        self.edit_encoding_button.clicked.connect(self._begin_encoding_edit)
        self.edit_encoding_button.setMinimumHeight(34)
        self.edit_encoding_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        edit_options_layout.addWidget(self.edit_encoding_button)

        # Chunk management belongs with the authored waveform definition rather
        # than competing with the generator palette for space beneath the plot.
        # This mirrors the Model Editor's object-list workflow: select one
        # authored chunk, then operate on that selection below the list.
        self.chunk_group = QGroupBox("Components", self.edit_options_group)
        chunk_layout = QVBoxLayout(self.chunk_group)
        chunk_layout.setContentsMargins(6, 8, 6, 6)
        chunk_layout.setSpacing(6)
        self.chunk_list = QListWidget(self.chunk_group)
        self.chunk_list.setObjectName("waveformChunkList")
        self.chunk_list.setMinimumHeight(130)
        self.chunk_list.setToolTip(
            "Authored waveform components in playback order. Select a component to edit, clone, move, or delete it."
        )
        self.chunk_list.currentRowChanged.connect(self._chunk_selection_changed)
        self.chunk_list.itemDoubleClicked.connect(lambda *_args: self.edit_selected_chunk())
        chunk_layout.addWidget(self.chunk_list, 1)
        edit_options_layout.addWidget(self.chunk_group, 1)

        self.chunk_actions_group = QGroupBox("Component actions", self.edit_options_group)
        chunk_actions = QGridLayout(self.chunk_actions_group)
        chunk_actions.setContentsMargins(6, 8, 6, 6)
        chunk_actions.setHorizontalSpacing(6)
        chunk_actions.setVerticalSpacing(6)
        self.edit_chunk_button = action_button(
            "Edit", self.chunk_actions_group, self.edit_selected_chunk, role="primary"
        )
        self.clone_chunk_button = QPushButton("Clone", self.chunk_actions_group)
        self.clone_chunk_button.clicked.connect(self.clone_selected_chunk)
        self.move_chunk_up_button = QPushButton("Move up", self.chunk_actions_group)
        self.move_chunk_up_button.setToolTip("Move the selected component one position earlier.")
        self.move_chunk_up_button.clicked.connect(self.move_selected_chunk_up)
        self.move_chunk_down_button = QPushButton("Move down", self.chunk_actions_group)
        self.move_chunk_down_button.setToolTip("Move the selected component one position later.")
        self.move_chunk_down_button.clicked.connect(self.move_selected_chunk_down)
        self.delete_chunk_button = QPushButton("Delete", self.chunk_actions_group)
        self.delete_chunk_button.clicked.connect(self.delete_selected_chunk)
        chunk_actions.addWidget(self.edit_chunk_button, 0, 0)
        chunk_actions.addWidget(self.clone_chunk_button, 0, 1)
        chunk_actions.addWidget(self.move_chunk_up_button, 1, 0)
        chunk_actions.addWidget(self.move_chunk_down_button, 1, 1)
        chunk_actions.addWidget(self.delete_chunk_button, 2, 0, 1, 2)
        edit_options_layout.addWidget(self.chunk_actions_group, 0)

        # Authoring Save/Cancel live at the bottom of the left waveform panel.
        # The top workflow strip is reserved for the current mode and Notes toggle.
        self.authoring_actions = QWidget(self.edit_options_group)
        authoring_actions_layout = QHBoxLayout(self.authoring_actions)
        authoring_actions_layout.setContentsMargins(0, 0, 0, 0)
        authoring_actions_layout.setSpacing(6)
        self.save_waveform_button = action_button(
            "Save waveform", self.authoring_actions, self.save_to_project, role="primary"
        )
        self.cancel_waveform_button = QPushButton("Cancel", self.authoring_actions)
        self.cancel_waveform_button.clicked.connect(self.cancel_task)
        authoring_actions_layout.addWidget(self.save_waveform_button, 1)
        authoring_actions_layout.addWidget(self.cancel_waveform_button, 1)
        edit_options_layout.addWidget(self.authoring_actions, 0)

        self.left_stack.addWidget(self.edit_options_group)
        self.left_stack.setCurrentWidget(self.catalogue_group)
        splitter.addWidget(self.left_stack)

        # Centre/right: the viewer remains persistent. Its lower card is either
        # browse information, the initial/edit waveform-definition form, or the
        # embedded waveform tools. Notes stay on the right with a draggable divider.
        editor_host = QWidget(splitter)
        editor_outer = QHBoxLayout(editor_host)
        editor_outer.setContentsMargins(6, 6, 6, 6)
        editor_outer.setSpacing(0)

        self.editor_side_splitter = QSplitter(Qt.Orientation.Horizontal, editor_host)
        self.editor_side_splitter.setObjectName("waveformEditorNotesSplitter")
        self.editor_side_splitter.setChildrenCollapsible(True)
        editor_outer.addWidget(self.editor_side_splitter, 1)

        main_column = QWidget(self.editor_side_splitter)
        main_column.setMinimumWidth(420)
        main_layout = QVBoxLayout(main_column)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        self.native_editor = NativeWaveformEditor(main_column)
        self.native_editor.documentChanged.connect(self._waveform_edited)
        self.native_editor.segmentEditRequested.connect(self._edit_authored_segment_index)
        self.native_editor.plot.segmentSelected.connect(self._select_chunk_row)
        main_layout.addWidget(self.native_editor, 1)

        self.lower_stack = QStackedWidget(main_column)
        self.lower_stack.setObjectName("waveformEditorLowerStack")

        self.info_group = QGroupBox("Waveform information", self.lower_stack)
        info = QGridLayout(self.info_group)
        info.setColumnStretch(1, 1)
        info.setColumnStretch(3, 1)

        self.info_empty_hint = QLabel(
            "No waveform selected. Create a new waveform or import one; the plot remains available as the workspace.",
            self.info_group,
        )
        self.info_empty_hint.setWordWrap(True)
        self.info_empty_hint.setProperty("role", "muted")
        info.addWidget(self.info_empty_hint, 0, 0, 1, 4)

        def value_label(object_name: str) -> QLabel:
            label = QLabel("—", self.info_group)
            label.setObjectName(object_name)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(True)
            return label

        self.info_name = value_label("waveformInfoName")
        self.info_duration = value_label("waveformInfoDuration")
        self.info_quantity = value_label("waveformInfoSignalType")
        self.info_units = value_label("waveformInfoUnits")
        self.info_peak_to_peak = value_label("waveformInfoPeakToPeak")
        self.info_slew = value_label("waveformInfoSlew")
        self.info_output_resistance = value_label("waveformInfoOutputResistance")
        self.info_origin = value_label("waveformInfoOrigin")

        info.addWidget(QLabel("Name", self.info_group), 1, 0)
        info.addWidget(self.info_name, 1, 1)
        info.addWidget(QLabel("Duration", self.info_group), 1, 2)
        info.addWidget(self.info_duration, 1, 3)
        info.addWidget(QLabel("Signal type", self.info_group), 2, 0)
        info.addWidget(self.info_quantity, 2, 1)
        info.addWidget(QLabel("Peak-to-peak", self.info_group), 2, 2)
        info.addWidget(self.info_peak_to_peak, 2, 3)
        info.addWidget(QLabel("Units", self.info_group), 3, 0)
        info.addWidget(self.info_units, 3, 1)

        self.slew_label = QLabel("Current slew rate", self.info_group)
        info.addWidget(self.slew_label, 4, 0)
        info.addWidget(self.info_slew, 4, 1)

        resistance_tip = (
            "Series source resistance for a voltage waveform. 0 Ω represents an ideal voltage source."
        )
        self.output_resistance_label = QLabel("Output resistance", self.info_group)
        self.output_resistance_label.setToolTip(resistance_tip)
        self.info_output_resistance.setToolTip(resistance_tip)
        info.addWidget(self.output_resistance_label, 5, 0)
        info.addWidget(self.info_output_resistance, 5, 1)

        info.addWidget(QLabel("Source", self.info_group), 6, 0)
        info.addWidget(self.info_origin, 6, 1, 1, 3)
        self.lower_stack.addWidget(self.info_group)

        # Creating a waveform starts here, in the same centre-panel position as
        # the browse-mode information card.  The waveform definition is confirmed
        # before any authoring tools become available.  The same form is reused
        # only when the user explicitly chooses Edit from the left encoding card.
        self.encoding_setup_group = QGroupBox("Waveform information", self.lower_stack)
        setup = QVBoxLayout(self.encoding_setup_group)
        setup.setSpacing(8)
        self.encoding_setup_hint = QLabel(
            "Set the waveform definition before adding waveform components.",
            self.encoding_setup_group,
        )
        self.encoding_setup_hint.setWordWrap(True)
        self.encoding_setup_hint.setProperty("role", "muted")
        setup.addWidget(self.encoding_setup_hint)

        setup_form = QFormLayout()
        setup_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        setup_form.setVerticalSpacing(8)

        self.edit_name = QLineEdit(self.encoding_setup_group)
        self.edit_name.setObjectName("waveformEditName")
        setup_form.addRow("Name", self.edit_name)

        self.edit_quantity = QComboBox(self.encoding_setup_group)
        self.edit_quantity.addItem("Current", "current")
        self.edit_quantity.addItem("Voltage", "voltage")
        self.edit_quantity.currentIndexChanged.connect(self._electrical_type_changed)
        setup_form.addRow("Signal type", self.edit_quantity)

        self.edit_units = QComboBox(self.encoding_setup_group)
        self.edit_units.currentIndexChanged.connect(self._electrical_units_changed)
        setup_form.addRow("Units", self.edit_units)

        self.edit_slew = QDoubleSpinBox(self.encoding_setup_group)
        self.edit_slew.setRange(0.0, 1.0e12)
        self.edit_slew.setDecimals(6)
        self.edit_slew.setKeyboardTracking(False)
        self.edit_slew.setSpecialValueText("No limit")
        self.edit_slew_label = QLabel("Current slew rate", self.encoding_setup_group)
        setup_form.addRow(self.edit_slew_label, self.edit_slew)

        self.edit_output_resistance = QDoubleSpinBox(self.encoding_setup_group)
        self.edit_output_resistance.setObjectName("waveformEditOutputResistance")
        self.edit_output_resistance.setRange(0.0, 1.0e12)
        self.edit_output_resistance.setDecimals(6)
        self.edit_output_resistance.setKeyboardTracking(False)
        self.edit_output_resistance.setSuffix(" Ω")
        self.edit_output_resistance_label = QLabel("Output resistance", self.encoding_setup_group)
        resistance_tip = (
            "Series source resistance for a voltage waveform. 0 Ω represents an ideal voltage source."
        )
        self.edit_output_resistance_label.setToolTip(resistance_tip)
        self.edit_output_resistance.setToolTip(resistance_tip)
        setup_form.addRow(self.edit_output_resistance_label, self.edit_output_resistance)
        setup.addLayout(setup_form)

        setup_actions = QHBoxLayout()
        setup_actions.addStretch(1)
        self.encoding_setup_primary_button = action_button(
            "Create waveform", self.encoding_setup_group, self._confirm_encoding_setup, role="primary"
        )
        self.encoding_setup_cancel_button = QPushButton("Cancel", self.encoding_setup_group)
        self.encoding_setup_cancel_button.clicked.connect(self._cancel_encoding_setup)
        setup_actions.addWidget(self.encoding_setup_primary_button)
        setup_actions.addWidget(self.encoding_setup_cancel_button)
        setup.addLayout(setup_actions)
        self.lower_stack.addWidget(self.encoding_setup_group)

        # During confirmed Create/Edit authoring this occupies the exact lower-card position used by
        # Waveform information in browse mode.  The tool selector and each
        # generator's controls share this one embedded card: no second builder
        # window or second waveform preview is created.
        self.toolbox_group = QGroupBox("Waveform tools", self.lower_stack)
        toolbox_outer = QVBoxLayout(self.toolbox_group)
        toolbox_outer.setContentsMargins(6, 8, 6, 6)
        self.tool_stack = QStackedWidget(self.toolbox_group)
        self.tool_stack.setObjectName("waveformEmbeddedToolStack")
        toolbox_outer.addWidget(self.tool_stack)

        self.tool_selector_page = QWidget(self.tool_stack)
        toolbox = QGridLayout(self.tool_selector_page)
        toolbox.setContentsMargins(0, 0, 0, 0)
        toolbox.setHorizontalSpacing(5)
        toolbox.setVerticalSpacing(5)
        toolbox.setColumnStretch(0, 1)
        toolbox.setColumnStretch(1, 1)
        toolbox.setRowStretch(0, 1)
        toolbox.setRowStretch(1, 1)
        toolbox.setRowStretch(2, 1)

        tool_specs = (
            ("Square wave", self.use_square_tool, "square"),
            ("Sine wave", self.use_sine_tool, "sine"),
            ("DC / pause", self.use_dc_tool, "dc"),
            ("Action potential", self.use_action_potential_tool, "action_potential"),
            ("Free form", self.use_free_form_tool, "free_form"),
            ("Import waveform…", self.import_native_waveform, "import"),
        )
        for index, (label, callback, icon_kind) in enumerate(tool_specs):
            button = waveform_button(label, self.tool_selector_page, callback, icon_kind=icon_kind)
            button.setMinimumSize(130, 82)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            if icon_kind == "import":
                self.import_tool_button = button
                button.setToolTip(
                    "Append one signal from a Field Workbench waveform, text/CSV table, or declarative encoded file, "
                    "supported structured waveform file, or reusable format profile."
                )
            row = index // 2
            col = index % 2
            toolbox.addWidget(button, row, col)
        # Component actions stay on the left; all ways of adding waveform content
        # live together in the Waveform tools workspace beneath the viewer.
        self.tool_stack.addWidget(self.tool_selector_page)
        self.tool_stack.setCurrentWidget(self.tool_selector_page)
        self._active_tool_panel: QWidget | None = None
        self.lower_stack.addWidget(self.toolbox_group)
        self.lower_stack.setCurrentWidget(self.info_group)
        main_layout.addWidget(self.lower_stack, 0)

        self.editor_side_splitter.addWidget(main_column)

        self.notes_panel = QWidget(self.editor_side_splitter)
        self.notes_panel.setObjectName("waveformEditorNotesPanel")
        self.notes_panel.setMinimumWidth(105)
        self.notes_panel.setMaximumWidth(520)
        notes_panel_layout = QVBoxLayout(self.notes_panel)
        notes_panel_layout.setContentsMargins(0, 0, 0, 0)
        notes_panel_layout.setSpacing(6)

        self.notes_group = QGroupBox("Notes", self.notes_panel)
        notes_layout = QVBoxLayout(self.notes_group)
        self.notes_stack = QStackedWidget(self.notes_group)
        self.notes_view = QTextEdit(self.notes_stack)
        self.notes_view.setObjectName("waveformNotesView")
        self.notes_view.setReadOnly(True)
        self.notes_view.setPlaceholderText("No notes recorded.")
        self.notes_stack.addWidget(self.notes_view)
        self.edit_notes = QTextEdit(self.notes_stack)
        self.edit_notes.setObjectName("waveformEditNotes")
        self.edit_notes.setPlaceholderText("Purpose, origin, or other notes…")
        self.notes_stack.addWidget(self.edit_notes)
        notes_layout.addWidget(self.notes_stack, 1)
        notes_panel_layout.addWidget(self.notes_group, 1)

        self._notes_panel_visible = True
        self._notes_panel_width = 230
        self.editor_side_splitter.addWidget(self.notes_panel)
        self.editor_side_splitter.setStretchFactor(0, 1)
        self.editor_side_splitter.setStretchFactor(1, 0)
        self.editor_side_splitter.setSizes([650, self._notes_panel_width])

        splitter.addWidget(editor_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([285, 895])

        self.notes_stack.setCurrentWidget(self.notes_view)
        self._set_workflow_mode("navigator")

        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)

    def _create_action(
        self,
        text: str,
        callback: Callable[[], None],
        *,
        shortcut: str | None = None,
        icon: QStyle.StandardPixmap | None = None,
    ) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(callback)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(themed_icon(self.style().standardIcon(icon)))
            self._standard_icon_specs.append((action, icon))
        return action

    def _create_actions(self) -> None:
        self.new_action = self._create_action("New waveform", self.new_waveform, shortcut="Ctrl+N", icon=QStyle.StandardPixmap.SP_FileIcon)
        self.import_action = self._create_action("Import waveform…", self.import_native_waveform, shortcut="Ctrl+O", icon=QStyle.StandardPixmap.SP_DialogOpenButton)
        self.save_project_action = self._create_action("Save to project", self.save_to_project, shortcut="Ctrl+S", icon=QStyle.StandardPixmap.SP_DialogSaveButton)
        self.export_native_action = self._create_action("Export FWB waveform…", self.export_native_waveform)
        self.export_csv_action = self._create_action("Export CSV…", self.export_csv_waveform)
        self.export_dac_action = self._create_action("Export DAC…", self.export_dac)
        self.export_rigol_action = self._create_action("Export Rigol…", self.open_export)
        self.close_action = self._create_action("Close", self.close, shortcut="Ctrl+Q")
        self.square_action = self._create_action("Square wave…", self.use_square_tool)
        self.sine_action = self._create_action("Sine wave…", self.use_sine_tool)
        self.dc_action = self._create_action("DC / pause…", self.use_dc_tool)
        self.action_potential_action = self._create_action("Action potential…", self.use_action_potential_tool)
        self.free_form_action = self._create_action("Free form…", self.use_free_form_tool)
        self.about_action = self._create_action("About Waveform Editor", self.show_about)
        self.cancel_context_action = self._create_action(
            "Cancel active waveform action", self._cancel_current_context, shortcut="Esc"
        )
        self.cancel_context_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.addAction(self.cancel_context_action)

    def _create_menus_and_toolbar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.import_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_native_action)
        file_menu.addAction(self.export_csv_action)
        file_menu.addAction(self.export_dac_action)
        file_menu.addAction(self.export_rigol_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_action)

        self.waveform_menu = self.menuBar().addMenu("&Waveform")
        self.waveform_menu.addAction(self.square_action)
        self.waveform_menu.addAction(self.sine_action)
        self.waveform_menu.addAction(self.dc_action)
        self.waveform_menu.addAction(self.action_potential_action)
        self.waveform_menu.addAction(self.free_form_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)
        self._sync_action_states()

    # --- browsing / task state -----------------------------------------
    def _set_catalogue_enabled(self, enabled: bool) -> None:
        for widget in (
            self.resource_list,
            self.create_resource_button,
            self.import_resource_button,
            self.edit_button,
            self.duplicate_resource_button,
            self.rename_resource_button,
            self.export_button,
            self.delete_resource_button,
            self.close_button,
        ):
            widget.setEnabled(bool(enabled))

    def _sync_action_states(self) -> None:
        task_active = self._task_mode is not None
        tool_active = bool(getattr(self, "_active_tool_panel", None))
        encoding_ready = bool(task_active and self._encoding_confirmed and not self._encoding_setup_active)

        self.save_project_action.setVisible(task_active)
        self.save_project_action.setEnabled(encoding_ready and not tool_active)
        self.waveform_menu.menuAction().setVisible(True)
        for action in (
            self.square_action,
            self.sine_action,
            self.dc_action,
            self.action_potential_action,
            self.free_form_action,
        ):
            # From browse mode a tool can still enter Edit automatically. During
            # new-waveform definition setup, however, no authoring tool is
            # available until Create waveform confirms the encoding.
            action.setEnabled((not task_active or encoding_ready) and not tool_active)
        self.new_action.setEnabled(not task_active)
        # Catalogue import remains available in browse mode.  Inline encoded import
        # tools become available only after the waveform encoding is confirmed.
        self.import_action.setEnabled((not task_active or encoding_ready) and not tool_active)
        self.cancel_context_action.setEnabled(task_active or tool_active)
        selected = self._selected_resource_id() is not None
        for action in (self.export_native_action, self.export_csv_action, self.export_dac_action, self.export_rigol_action):
            action.setEnabled(selected and not task_active)
        catalogue_selection_enabled = selected and not task_active
        self.edit_button.setEnabled(catalogue_selection_enabled)
        self.rename_resource_button.setEnabled(catalogue_selection_enabled)
        self.duplicate_resource_button.setEnabled(catalogue_selection_enabled)
        self.export_button.setEnabled(catalogue_selection_enabled)
        self.delete_resource_button.setEnabled(catalogue_selection_enabled)
        for action in (
            self.edit_resource_action,
            self.rename_resource_action,
            self.duplicate_resource_action,
            self.export_resource_action,
            self.delete_resource_action,
        ):
            action.setEnabled(catalogue_selection_enabled)
        if hasattr(self, "save_waveform_button"):
            self.save_waveform_button.setEnabled(encoding_ready and not tool_active)
            self.edit_encoding_button.setEnabled(encoding_ready and not tool_active)
            self.edit_quantity.setEnabled(self._encoding_setup_active and not tool_active)
            self.edit_units.setEnabled(self._encoding_setup_active and not tool_active)
            self.edit_name.setEnabled(self._encoding_setup_active and not tool_active)
            self.edit_slew.setEnabled(self._encoding_setup_active and not tool_active)
            self.edit_output_resistance.setEnabled(self._encoding_setup_active and not tool_active)
        if hasattr(self, "chunk_list"):
            recipe = self.native_editor.exact_authoring_recipe() if encoding_ready else None
            segments = list(recipe.get("segments", [])) if recipe else []
            row = self.chunk_list.currentRow()
            selected_segment = segments[row] if 0 <= row < len(segments) else None
            editable_chunk = bool(
                selected_segment
                and str(selected_segment.get("kind") or "").strip().lower() in {"generator", "encoded"}
            )
            has_chunk = selected_segment is not None
            self.edit_chunk_button.setEnabled(editable_chunk and not tool_active)
            self.clone_chunk_button.setEnabled(has_chunk and not tool_active)
            self.move_chunk_up_button.setEnabled(has_chunk and row > 0 and not tool_active)
            self.move_chunk_down_button.setEnabled(
                has_chunk and row < len(segments) - 1 and not tool_active
            )
            self.delete_chunk_button.setEnabled(has_chunk and not tool_active)
        if hasattr(self, "import_tool_button"):
            self.import_tool_button.setEnabled(encoding_ready and not tool_active)

    @staticmethod
    def _signal_type_for_series(series: WaveformSeries | None) -> str:
        if series is None:
            return "current"
        quantity = str(series.quantity or "").strip().lower()
        if quantity in {"current", "voltage"}:
            return quantity
        unit = str(series.unit or "").strip().lower().replace("µ", "u")
        if unit in {"a", "ma"}:
            return "current"
        if unit in {"v", "mv"}:
            return "voltage"
        # The editor intentionally has no arbitrary/unitless electrical mode.
        # An unclassified saved/imported signal is treated as Current when its
        # waveform-level definition is first presented to the user.
        return "current"

    @staticmethod
    def _units_for_signal_type(signal_type: str) -> tuple[str, str]:
        return ("V", "mV") if signal_type == "voltage" else ("mA", "A")

    @staticmethod
    def _value_scale_for_unit(unit: str) -> float:
        return 0.001 if str(unit) in {"mA", "mV"} else 1.0

    def _set_edit_units(self, signal_type: str, preferred: str = "") -> None:
        choices = self._units_for_signal_type(signal_type)
        current = str(preferred or self.edit_units.currentText() or "")
        self.edit_units.blockSignals(True)
        self.edit_units.clear()
        self.edit_units.addItems(list(choices))
        if current in choices:
            self.edit_units.setCurrentText(current)
        else:
            self.edit_units.setCurrentIndex(0)
        self.edit_units.blockSignals(False)
        self._update_electrical_control_labels()

    def _electrical_type_changed(self, *_args) -> None:
        # Signal type/units are only committed when the waveform-definition form
        # is confirmed.  Merely changing a setup control must not silently alter
        # the working waveform because Cancel needs to be a true rollback.
        signal_type = str(self.edit_quantity.currentData() or "current")
        self._set_edit_units(signal_type)
        self._update_electrical_control_labels()

    def _electrical_units_changed(self, *_args) -> None:
        self._update_electrical_control_labels()

    def _update_electrical_control_labels(self) -> None:
        signal_type = str(self.edit_quantity.currentData() or "current")
        unit = self.edit_units.currentText() or self._units_for_signal_type(signal_type)[0]
        self.edit_slew.setSuffix(f" {unit}/ms")
        voltage = signal_type == "voltage"
        self.edit_slew_label.setText("Voltage slew rate" if voltage else "Current slew rate")
        self.edit_output_resistance_label.setVisible(voltage)
        self.edit_output_resistance.setVisible(voltage)

    def _display_slew_from_si(self, slew_per_s: float, unit: str) -> float:
        return float(slew_per_s) * 0.001 / self._value_scale_for_unit(unit)

    def _slew_si_from_edit(self) -> float:
        unit = self.edit_units.currentText()
        return float(self.edit_slew.value()) * self._value_scale_for_unit(unit) / 0.001

    @staticmethod
    def _electrical_metadata(document: WaveformDocument) -> dict:
        electrical = dict(document.metadata).get("electrical")
        return dict(electrical) if isinstance(electrical, dict) else {}

    @staticmethod
    def _peak_to_peak_text(series: WaveformSeries | None, *, unit: str | None = None) -> str:
        if series is None or not len(series.values):
            return "—"
        values = [float(value) for value in series.values]
        magnitude = max(values) - min(values)
        suffix = str(unit if unit is not None else series.unit or "").strip()
        return f"{magnitude:.6g} {suffix}".strip()

    def _toggle_notes_panel(self) -> None:
        """Collapse/restore Notes completely; the workflow-strip button stays reachable."""
        sizes = self.editor_side_splitter.sizes()
        total = max(1, int(sum(sizes)) if sizes else int(self.editor_side_splitter.width()))
        if self._notes_panel_visible:
            if len(sizes) >= 2 and sizes[1] > 0:
                self._notes_panel_width = max(175, int(sizes[1]))
            self._notes_panel_visible = False
            self.notes_panel.setVisible(False)
            self.editor_side_splitter.setSizes([total, 0])
        else:
            self._notes_panel_visible = True
            self.notes_panel.setVisible(True)
            right = min(max(175, self._notes_panel_width), max(175, total - 1))
            self.editor_side_splitter.setSizes([max(1, total - right), right])

    def _populate_editable_info(self) -> None:
        """Populate the waveform-definition form from the current working copy."""
        try:
            document = self.native_editor.document(title=self._current_resource_name(self._working_name or "Waveform"))
        except Exception:
            return
        signal = document.signal
        self.edit_name.setText(self._current_resource_name(self._working_name or document.title or "Waveform"))

        signal_type = self._signal_type_for_series(signal)
        index = self.edit_quantity.findData(signal_type)
        self.edit_quantity.blockSignals(True)
        self.edit_quantity.setCurrentIndex(max(0, index))
        self.edit_quantity.blockSignals(False)
        preferred_unit = str(signal.unit or "")
        self._set_edit_units(signal_type, preferred_unit)

        electrical = self._electrical_metadata(document)
        slew_si = float(electrical.get("slew_rate_per_s", 0.0) or 0.0)
        self.edit_slew.setValue(self._display_slew_from_si(slew_si, self.edit_units.currentText()))
        self.edit_output_resistance.setValue(
            float(electrical.get("output_resistance_ohm", 0.0) or 0.0) if signal_type == "voltage" else 0.0
        )
        self._update_electrical_control_labels()

        metadata = dict(document.metadata)
        self.info_origin.setText(self._source_summary(document))
        notes_text = str(metadata.get("notes", metadata.get("note", "")) or "")
        self.edit_notes.setPlainText(notes_text)
        self._edit_metadata_baseline = {"notes": notes_text.strip()}
        self._edit_form_baseline = self._current_edit_form_state()
        self._update_encoding_summary()
        self._update_working_derived_info()

    def _current_edit_form_state(self) -> dict[str, str]:
        return {
            "name": self.edit_name.text().strip(),
            "quantity": str(self.edit_quantity.currentData() or "current"),
            "units": self.edit_units.currentText().strip(),
            "slew": f"{self.edit_slew.value():.12g}",
            "output_resistance_ohm": (
                f"{self.edit_output_resistance.value():.12g}"
                if self.edit_quantity.currentData() == "voltage" else ""
            ),
            "notes": self.edit_notes.toPlainText().strip(),
        }

    def _capture_encoding_form(self) -> dict[str, object]:
        return {
            "name": self.edit_name.text(),
            "quantity": str(self.edit_quantity.currentData() or "current"),
            "units": self.edit_units.currentText(),
            "slew": float(self.edit_slew.value()),
            "output_resistance_ohm": float(self.edit_output_resistance.value()),
        }

    def _restore_encoding_form(self, state: dict[str, object]) -> None:
        self.edit_name.setText(str(state.get("name") or "Waveform"))
        quantity = str(state.get("quantity") or "current")
        index = self.edit_quantity.findData(quantity)
        self.edit_quantity.blockSignals(True)
        self.edit_quantity.setCurrentIndex(max(0, index))
        self.edit_quantity.blockSignals(False)
        self._set_edit_units(quantity, str(state.get("units") or ""))
        self.edit_slew.setValue(float(state.get("slew", 0.0) or 0.0))
        self.edit_output_resistance.setValue(float(state.get("output_resistance_ohm", 0.0) or 0.0))
        self._update_electrical_control_labels()

    def _update_encoding_summary(self) -> None:
        if not hasattr(self, "encoding_name"):
            return
        name = self.edit_name.text().strip() or self._working_name or "Waveform"
        signal_type = str(self.edit_quantity.currentData() or "current")
        unit = self.edit_units.currentText().strip() or self._units_for_signal_type(signal_type)[0]
        self.encoding_name.setText(name)
        self.encoding_quantity.setText("Voltage" if signal_type == "voltage" else "Current")
        self.encoding_units.setText(unit)
        self.encoding_slew_label.setText("Voltage slew rate" if signal_type == "voltage" else "Current slew rate")
        slew = float(self.edit_slew.value())
        self.encoding_slew.setText("No limit" if slew <= 0.0 else f"{slew:.6g} {unit}/ms")
        voltage = signal_type == "voltage"
        self.encoding_output_resistance_label.setVisible(voltage)
        self.encoding_output_resistance.setVisible(voltage)
        if voltage:
            self.encoding_output_resistance.setText(f"{self.edit_output_resistance.value():.6g} Ω")

    def _working_copy_dirty(self) -> bool:
        if self.native_editor.dirty:
            return True
        return bool(self._edit_form_baseline) and self._current_edit_form_state() != self._edit_form_baseline

    def _update_working_derived_info(self) -> None:
        try:
            document = self.native_editor.document(title=self.edit_name.text().strip() or "Waveform")
        except Exception:
            return
        signal = document.signal
        self.info_duration.setText(f"{signal.duration_s:.6g} s")
        self.info_peak_to_peak.setText(
            self._peak_to_peak_text(signal, unit=self.edit_units.currentText())
        )
        self.info_origin.setText(self._source_summary(document))
        self.info_empty_hint.setVisible(False)

    def _set_workflow_mode(self, mode: str) -> None:
        labels = {
            "navigator": "Waveform navigator",
            "encoding": "Waveform encoding",
            "editor": "Waveform editor",
        }
        self.workflow_mode_label.setText(labels.get(mode, str(mode)))
        authoring = mode == "editor"
        self.authoring_actions.setVisible(authoring)
        # Component selection is an authoring affordance, not a passive-viewer
        # navigation tool.  Keep the modebar aligned with the current workflow.
        self.native_editor.plot.set_component_selection_available(authoring)

    def _show_browse_mode(self) -> None:
        self._close_embedded_tool(restore_preview=True, sync=False)
        self._task_mode = None
        self._task_original_id = None
        self._working_name = None
        self._edit_metadata_baseline = {}
        self._edit_form_baseline = {}
        self._encoding_setup_active = False
        self._encoding_confirmed = False
        self._encoding_edit_snapshot = None
        self.left_stack.setCurrentWidget(self.catalogue_group)
        self.lower_stack.setCurrentWidget(self.info_group)
        self._set_workflow_mode("navigator")
        self.notes_stack.setCurrentWidget(self.notes_view)
        self.native_editor.set_recipe_editable(False)
        self._refresh_chunk_list(select_row=-1)
        self._set_catalogue_enabled(True)
        self._sync_action_states()

    def _show_encoding_setup(self, *, initial: bool) -> None:
        self._encoding_setup_active = True
        self._set_workflow_mode("encoding")
        if initial:
            self.left_stack.setCurrentWidget(self.catalogue_group)
            self.encoding_setup_primary_button.setText("Create waveform")
            self.encoding_setup_hint.setText(
                "Choose the waveform name, signal type, and units before adding waveform components. "
                "Generators and imports will inherit this encoding."
            )
        else:
            self.left_stack.setCurrentWidget(self.edit_options_group)
            self.encoding_setup_primary_button.setText("Apply changes")
            self.encoding_setup_hint.setText(
                "Edit the waveform-level encoding. These settings remain fixed for all waveform tools until changed here again."
            )
        self.lower_stack.setCurrentWidget(self.encoding_setup_group)
        self.notes_stack.setCurrentWidget(self.edit_notes)
        self.native_editor.set_recipe_editable(False)
        self._sync_action_states()

    def _enter_task_mode(self, mode: str, *, original_id: str | None) -> None:
        self._project_close_prepared = False
        self._close_embedded_tool(restore_preview=True, sync=False)
        self._task_mode = str(mode)
        self._task_original_id = original_id
        self.notes_stack.setCurrentWidget(self.edit_notes)
        self._populate_editable_info()
        self._set_catalogue_enabled(False)
        if self._task_mode == "create":
            self._encoding_confirmed = False
            self._encoding_edit_snapshot = None
            self._show_encoding_setup(initial=True)
        else:
            self._encoding_confirmed = True
            self._encoding_setup_active = False
            self._encoding_edit_snapshot = None
            self._update_encoding_summary()
            self._set_workflow_mode("editor")
            self.left_stack.setCurrentWidget(self.edit_options_group)
            self.lower_stack.setCurrentWidget(self.toolbox_group)
            self.tool_stack.setCurrentWidget(self.tool_selector_page)
            self.native_editor.set_recipe_editable(True)
            self._refresh_chunk_list()
            self._sync_action_states()

    def _begin_encoding_edit(self) -> None:
        if self._task_mode is None or not self._encoding_confirmed or self._active_tool_panel is not None:
            return
        self._encoding_edit_snapshot = self._capture_encoding_form()
        self._show_encoding_setup(initial=False)
        self.status_bar.set("Editing waveform encoding; Apply changes or Cancel before returning to waveform tools.")

    def _confirm_encoding_setup(self) -> None:
        if self._task_mode is None or not self._encoding_setup_active:
            return
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.information(self, "Waveform information", "Enter a waveform name.")
            return
        signal_type = str(self.edit_quantity.currentData() or "current")
        unit = self.edit_units.currentText().strip()
        if signal_type not in {"current", "voltage"} or unit not in self._units_for_signal_type(signal_type):
            QMessageBox.information(self, "Waveform information", "Choose matching Current/Voltage units.")
            return
        try:
            self.native_editor.set_current_signal_definition(signal_type, unit)
        except Exception as error:
            QMessageBox.warning(self, "Unable to set waveform encoding", str(error))
            return
        self._working_name = name
        self._encoding_confirmed = True
        self._encoding_setup_active = False
        self._encoding_edit_snapshot = None
        self._update_encoding_summary()
        self._set_workflow_mode("editor")
        self.left_stack.setCurrentWidget(self.edit_options_group)
        self.lower_stack.setCurrentWidget(self.toolbox_group)
        self.tool_stack.setCurrentWidget(self.tool_selector_page)
        self.notes_stack.setCurrentWidget(self.edit_notes)
        self.native_editor.set_recipe_editable(True)
        self._refresh_chunk_list()
        self._update_working_derived_info()
        self._sync_action_states()
        if self._task_mode == "create":
            self.status_bar.set("Waveform created. Add waveform components below, then Save waveform when finished.")
        else:
            self.status_bar.set("Waveform encoding updated. Waveform tools now use the confirmed signal type and units.")

    def _cancel_encoding_setup(self) -> None:
        if self._task_mode is None or not self._encoding_setup_active:
            return
        if not self._encoding_confirmed:
            self.cancel_task()
            return
        if self._encoding_edit_snapshot is not None:
            self._restore_encoding_form(self._encoding_edit_snapshot)
        self._encoding_edit_snapshot = None
        self._encoding_setup_active = False
        self._update_encoding_summary()
        self._set_workflow_mode("editor")
        self.left_stack.setCurrentWidget(self.edit_options_group)
        self.lower_stack.setCurrentWidget(self.toolbox_group)
        self.tool_stack.setCurrentWidget(self.tool_selector_page)
        self.native_editor.set_recipe_editable(True)
        self._sync_action_states()
        self.status_bar.set("Waveform encoding edit cancelled.")

    def _waveform_edited(self) -> None:
        if self._task_mode is not None and self._encoding_confirmed:
            self._refresh_chunk_list()
            self._update_working_derived_info()
            self._sync_action_states()
            self.status_bar.set("Waveform edited; save to keep the changes in the project.")

    def _catalogue_selection_changed(self, *_args) -> None:
        if self._task_mode is not None:
            return
        self.load_selected_resource()

    def _prepare_waveform_tool(self) -> bool:
        """Enter authoring only when a waveform definition is already confirmed."""
        if self._task_mode is not None:
            return bool(self._encoding_confirmed and not self._encoding_setup_active)
        if self._selected_resource_id() is not None:
            self.edit_selected_resource()
        else:
            # A tool invoked with no saved selection starts the new-waveform
            # definition step, but does not bypass it. The user must press
            # Create waveform before the requested authoring tool is available.
            self.new_waveform()
        return bool(self._task_mode is not None and self._encoding_confirmed and not self._encoding_setup_active)

    def _cancel_current_context(self) -> None:
        """Let Escape unwind the innermost waveform-authoring state first."""
        if self._active_tool_panel is not None:
            self._close_embedded_tool(restore_preview=True, sync=True)
            self.status_bar.set("Waveform tool preview cancelled.")
            return
        if self._encoding_setup_active and self._encoding_confirmed:
            self._cancel_encoding_setup()
            return
        if self._task_mode is not None:
            self.cancel_task()

    def _close_embedded_tool(self, *, restore_preview: bool = True, sync: bool = True) -> None:
        panel = getattr(self, "_active_tool_panel", None)
        if panel is not None:
            if hasattr(self, "native_editor"):
                self.native_editor.plot.clear_free_form_editor()
            detach = getattr(panel, "detach_preview_editor", None)
            if callable(detach):
                detach()
            self.tool_stack.setCurrentWidget(self.tool_selector_page)
            self.tool_stack.removeWidget(panel)
            panel.deleteLater()
            self._active_tool_panel = None
        if restore_preview and hasattr(self, "native_editor"):
            editable = bool(self._task_mode is not None and self._encoding_confirmed and not self._encoding_setup_active)
            self.native_editor.restore_working_preview(editable=editable)
        if sync and hasattr(self, "save_project_action"):
            self._sync_action_states()

    def _preview_embedded_generator(self, trace, segment_index: int | None) -> str:
        composed, boundary = self.native_editor.show_generated_preview(
            trace,
            segment_index=segment_index,
        )
        duration = float(composed.time_s[-1] - composed.time_s[0]) if len(composed.time_s) >= 2 else 0.0
        if segment_index is not None:
            return f"Previewing edit of component {segment_index + 1} · resulting waveform {duration:.6g} s."
        if boundary is None:
            return f"Previewing first waveform component · resulting duration {duration:.6g} s."
        return f"Previewing append at {boundary:.6g} s · resulting duration {duration:.6g} s."

    def _apply_embedded_generator(self, trace, segment_index: int | None) -> None:
        shape = str(trace.settings.get("shape", "waveform")).replace("_", " ")
        if segment_index is None:
            self.native_editor.apply_generated_trace(trace, append=True)
            message = f"Applied {shape} component to the working waveform."
        else:
            self.native_editor.replace_generated_recipe_segment(segment_index, trace)
            message = f"Updated generated component {segment_index + 1} ({shape})."
        self._close_embedded_tool(restore_preview=True, sync=False)
        recipe = self.native_editor.exact_authoring_recipe()
        if recipe:
            row = int(segment_index) if segment_index is not None else len(recipe.get("segments", [])) - 1
            self._refresh_chunk_list(select_row=row)
        self._update_working_derived_info()
        self._sync_action_states()
        self.status_bar.set(message)

    def _cancel_embedded_generator(self) -> None:
        if self._active_tool_panel is None:
            return
        self._close_embedded_tool(restore_preview=True, sync=True)
        self.status_bar.set("Waveform tool preview cancelled.")

    def _open_embedded_generator(
        self,
        generator: str,
        *,
        recipe_segment: dict | None = None,
        recipe_segment_index: int | None = None,
    ) -> None:
        if not self._prepare_waveform_tool():
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        panel = EmbeddedGeneratorPanel(
            self,
            generator=generator,
            recipe_segment=recipe_segment,
            recipe_segment_index=recipe_segment_index,
            on_preview=self._preview_embedded_generator,
            on_apply=self._apply_embedded_generator,
            on_cancel=self._cancel_embedded_generator,
        )
        self._active_tool_panel = panel
        self.tool_stack.addWidget(panel)
        self.tool_stack.setCurrentWidget(panel)
        self.native_editor.set_recipe_editable(False)
        self._sync_action_states()
        verb = "Editing" if recipe_segment_index is not None else "Adding"
        self.status_bar.set(f"{verb} {str(generator).replace('_', ' ')} component; adjust controls, then Apply or Cancel.")

    def use_square_tool(self) -> None:
        self._open_embedded_generator("square")

    def use_sine_tool(self) -> None:
        self._open_embedded_generator("sine")

    def use_dc_tool(self) -> None:
        self._open_embedded_generator("dc")

    def use_action_potential_tool(self) -> None:
        self._open_embedded_generator("action_potential")

    def use_free_form_tool(self) -> None:
        self._open_embedded_generator("free_form")

    def _preview_embedded_import(self, document: WaveformDocument, _description: str) -> str:
        composed, boundary = self.native_editor.show_document_preview(document)
        duration = float(composed.time_s[-1] - composed.time_s[0]) if len(composed.time_s) >= 2 else 0.0
        if boundary is None:
            return f"Previewing first imported component · resulting duration {duration:.6g} s."
        return f"Previewing imported append at {boundary:.6g} s · resulting duration {duration:.6g} s."

    def _apply_embedded_import(self, document: WaveformDocument, description: str) -> None:
        try:
            self.native_editor.append_document(document, description=description)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to append waveform", str(error))
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        recipe = self.native_editor.exact_authoring_recipe()
        if recipe:
            self._refresh_chunk_list(select_row=len(recipe.get("segments", [])) - 1)
        self._update_working_derived_info()
        self._sync_action_states()
        self.status_bar.set(f"Applied {description.lower()} to the working waveform.")

    def _cancel_embedded_import(self) -> None:
        if self._active_tool_panel is None:
            return
        self._close_embedded_tool(restore_preview=True, sync=True)
        self.status_bar.set("Waveform import preview cancelled.")

    def _open_embedded_waveform_import(self) -> None:
        if not self._prepare_waveform_tool():
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        panel = EmbeddedWaveformImportPanel(
            self,
            on_preview=self._preview_embedded_import,
            on_apply=self._apply_embedded_import,
            on_cancel=self._cancel_embedded_import,
            on_runtime_profile_source=self._open_embedded_runtime_profile_source,
        )
        self._active_tool_panel = panel
        self.tool_stack.addWidget(panel)
        self.tool_stack.setCurrentWidget(panel)
        self.native_editor.set_recipe_editable(False)
        self._sync_action_states()
        self.status_bar.set("Choose a waveform source; its selected signal will preview as an appended component above.")

    def _open_embedded_runtime_import(self, profile, source: Path) -> None:
        if not self._prepare_waveform_tool():
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        try:
            panel = EmbeddedRuntimeFormatPanel(
                self,
                profile=profile,
                source=source,
                on_preview=self._preview_embedded_encoded,
                on_apply=self._apply_embedded_encoded,
                on_cancel=self._cancel_embedded_import,
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to import encoded waveform", str(error))
            self._open_embedded_waveform_import()
            return
        self._active_tool_panel = panel
        self.tool_stack.addWidget(panel)
        self.tool_stack.setCurrentWidget(panel)
        self.native_editor.set_recipe_editable(False)
        self._sync_action_states()
        self.status_bar.set(
            f"Adjust import-time settings for {profile.name}; the translated append previews live above."
        )

    def _preview_embedded_encoded(self, segment: dict, segment_index: int | None) -> str:
        composed, boundary = self.native_editor.show_recipe_segment_preview(
            segment,
            segment_index=segment_index,
        )
        duration = float(composed.time_s[-1] - composed.time_s[0]) if len(composed.time_s) >= 2 else 0.0
        if segment_index is not None:
            return f"Previewing edit of encoded component {segment_index + 1} · resulting waveform {duration:.6g} s."
        if boundary is None:
            return f"Previewing first encoded component · resulting duration {duration:.6g} s."
        return f"Previewing encoded append at {boundary:.6g} s · resulting duration {duration:.6g} s."

    def _apply_embedded_encoded(self, segment: dict, segment_index: int | None) -> None:
        try:
            if segment_index is None:
                self.native_editor.append_recipe_segment(segment)
                message = "Applied editable encoded component to the working waveform."
            else:
                self.native_editor.replace_recipe_segment(segment_index, segment)
                message = f"Updated editable encoded component {segment_index + 1}."
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to apply encoded component", str(error))
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        recipe = self.native_editor.exact_authoring_recipe()
        if recipe:
            row = int(segment_index) if segment_index is not None else len(recipe.get("segments", [])) - 1
            self._refresh_chunk_list(select_row=row)
        self._update_working_derived_info()
        self._sync_action_states()
        self.status_bar.set(message)

    def _open_embedded_encoded_segment(self, index: int, segment: dict) -> None:
        if not self._prepare_waveform_tool():
            return
        self._close_embedded_tool(restore_preview=True, sync=False)
        try:
            panel = EmbeddedRuntimeFormatPanel(
                self,
                recipe_segment=segment,
                recipe_segment_index=int(index),
                on_preview=self._preview_embedded_encoded,
                on_apply=self._apply_embedded_encoded,
                on_cancel=self._cancel_embedded_import,
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to edit encoded component", str(error))
            return
        self._active_tool_panel = panel
        self.tool_stack.addWidget(panel)
        self.tool_stack.setCurrentWidget(panel)
        self.native_editor.set_recipe_editable(False)
        self._sync_action_states()
        self.status_bar.set(
            f"Editing encoded component {index + 1}; source samples and import-time timing/mapping remain editable."
        )

    # --- project resource library -------------------------------------
    def refresh_project_resources(
        self,
        select_id: str | None = None,
        *,
        preserve_working_copy: bool = True,
    ) -> None:
        """Refresh the catalogue without silently discarding an active edit.

        Project resources can change in another suite window while Waveform Editor
        is open.  Refreshing the navigator must therefore be separate from loading
        a catalogue selection.  When a Create/Edit task is active we update only
        the list by default and leave the working waveform untouched.
        """

        task_active = self._task_mode is not None
        wanted = select_id or self.current_waveform_id or self._task_original_id
        self.resource_list.blockSignals(True)
        self.resource_list.clear()
        try:
            resources = self.project_context.adapter.list_waveforms()
        except Exception:
            resources = []
        selected_item: QListWidgetItem | None = None
        for row in resources:
            item = QListWidgetItem(str(row.get("name") or "Waveform"))
            waveform_id = str(row.get("id") or "")
            item.setData(_RESOURCE_ID_ROLE, waveform_id)
            item.setToolTip(
                f"{str(row.get('quantity', '')).title()} ({row.get('unit', '')}) · "
                f"{float(row.get('duration_s', 0.0) or 0.0):.6g} s"
            )
            self.resource_list.addItem(item)
            if waveform_id == wanted:
                selected_item = item
        if selected_item is not None:
            self.resource_list.setCurrentItem(selected_item)
        elif self.resource_list.count() and not task_active:
            self.resource_list.setCurrentRow(0)
        self.resource_list.blockSignals(False)

        if task_active and preserve_working_copy:
            # The catalogue is disabled while editing, but keeping it current is
            # useful when another suite application creates/renames a waveform.
            # Never call load_selected_resource() here: that would replace the
            # unsaved native-editor document and collapse Create/Edit mode.
            return

        if self.resource_list.currentItem() is not None:
            self.load_selected_resource()
        else:
            self.current_waveform_id = None
            self.native_editor.show_empty_preview()
            self._clear_resource_summary()
            self.setWindowTitle(f"Waveform Editor — {self.project_context.display_name()}")
            self._show_browse_mode()

    def request_open_resource(self, waveform_id: str | None) -> bool:
        """Safely focus a project waveform requested by another suite window.

        Exposure Editor can ask Waveform Editor to open a referenced source.  If
        another waveform is currently being authored, protect that working copy
        instead of treating the cross-window request like a catalogue refresh.
        """

        target = str(waveform_id or "").strip() or None
        if self._active_tool_panel is not None:
            QMessageBox.information(
                self,
                "Waveform tool preview active",
                "Apply or Cancel the active waveform tool preview before opening another project waveform.",
            )
            return False
        if self._task_mode is not None:
            editing_same = bool(
                target
                and self._task_mode == "edit"
                and target == self.current_waveform_id
            )
            if editing_same:
                return True
            if self._working_copy_dirty():
                answer = QMessageBox.warning(
                    self,
                    "Unsaved waveform changes",
                    "Save the current waveform changes before opening another project waveform?",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Save,
                )
                if answer == QMessageBox.StandardButton.Cancel:
                    return False
                if answer == QMessageBox.StandardButton.Save:
                    self.save_to_project()
                    if self._task_mode is not None:
                        # Validation may have kept the edit task open.
                        return False
                else:
                    self._show_browse_mode()
            else:
                self._show_browse_mode()

        self.refresh_project_resources(target, preserve_working_copy=False)
        return target is None or self.current_waveform_id == target

    def _selected_resource_id(self) -> str | None:
        item = self.resource_list.currentItem()
        if item is None:
            return None
        value = str(item.data(_RESOURCE_ID_ROLE) or "").strip()
        return value or None

    @staticmethod
    def _quantity_label(quantity: str) -> str:
        labels = {"current": "Current", "voltage": "Voltage"}
        return labels.get(str(quantity or "").strip().lower(), "Unclassified")

    @staticmethod
    def _metadata_summary(value) -> str:
        if isinstance(value, dict):
            parts: list[str] = []
            for key, item in value.items():
                if item is None:
                    continue
                if isinstance(item, str) and not item.strip():
                    continue
                if isinstance(item, (list, tuple, dict)) and not item:
                    continue
                parts.append(f"{key}: {item}")
            return "; ".join(parts) if parts else "—"
        text = str(value or "").strip()
        return text or "—"

    @classmethod
    def _source_summary(cls, document: WaveformDocument) -> str:
        metadata = dict(document.metadata)
        recipe_value = metadata.get(AUTHORING_RECIPE_KEY)
        if recipe_value is not None:
            try:
                recipe = normalize_authoring_recipe(recipe_value)
            except WaveformRecipeError:
                recipe = None
            if recipe is not None:
                generated = [
                    segment for segment in recipe.get("segments", [])
                    if segment.get("kind") == "generator"
                ]
                detached = recipe.get("integrity") == "detached"
                suffix = " · recipe retained as provenance only" if detached else ""
                if len(generated) == 1 and len(recipe.get("segments", [])) == 1:
                    segment = generated[0]
                    shape = str(segment.get("generator") or "waveform").replace("_", " ").title()
                    parameters = segment.get("parameters", {})
                    parts = [f"Field Workbench authored waveform · {shape}"]
                    frequency = parameters.get("frequency_hz") if isinstance(parameters, dict) else None
                    if frequency is not None:
                        parts.append(f"{float(frequency):.6g} Hz")
                    duty = parameters.get("duty_cycle_percent") if isinstance(parameters, dict) else None
                    if duty is not None:
                        parts.append(f"{float(duty):.6g}% duty")
                    return " · ".join(parts) + suffix
                if generated:
                    return (
                        f"Field Workbench authored waveform · {len(generated)} generated "
                        f"component{'s' if len(generated) != 1 else ''}{suffix}"
                    )
                encoded_segments = [
                    segment for segment in recipe.get("segments", [])
                    if segment.get("kind") == "encoded"
                ]
                if len(encoded_segments) == 1 and len(recipe.get("segments", [])) == 1:
                    segment = encoded_segments[0]
                    source_name = str(segment.get("source_name") or "encoded import").strip() or "encoded import"
                    return (
                        f"Field Workbench authored waveform · editable encoded import · {source_name}"
                        f" · {len(segment.get('raw_values') or ())} points{suffix}"
                    )
                if encoded_segments:
                    return (
                        f"Field Workbench authored waveform · {len(encoded_segments)} editable encoded "
                        f"component{'s' if len(encoded_segments) != 1 else ''}{suffix}"
                    )

        if document.source_path:
            source = f"Imported from {Path(document.source_path).name}"
            settings = metadata.get("import_settings")
            if isinstance(settings, dict):
                if settings.get("format") == "text_csv_table":
                    column = settings.get("signal_header") or settings.get("signal_column")
                    return f"{source} · text/CSV column {column}"
                if settings.get("format") == "format_profile":
                    profile_payload = settings.get("profile")
                    if isinstance(profile_payload, dict):
                        profile_data = profile_payload.get("profile")
                        if isinstance(profile_data, dict) and profile_data.get("name"):
                            return f"{source} · format profile {profile_data['name']}"
            profile = metadata.get("format_profile")
            if profile:
                return f"{source} · format profile {profile}"
            return source
        origin = metadata.get("origin") or metadata.get("source") or metadata.get("created_by")
        return cls._metadata_summary(origin)

    def _clear_resource_summary(self) -> None:
        for label in (
            self.info_name,
            self.info_duration,
            self.info_quantity,
            self.info_units,
            self.info_peak_to_peak,
            self.info_slew,
            self.info_output_resistance,
            self.info_origin,
        ):
            label.setText("—")
        self.slew_label.setText("Current slew rate")
        self.output_resistance_label.setVisible(False)
        self.info_output_resistance.setVisible(False)
        self.edit_output_resistance.setVisible(False)
        self.info_empty_hint.setVisible(True)
        self.notes_view.clear()

    def _update_resource_summary(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            self._clear_resource_summary()
            return
        try:
            row = self.project_context.adapter.waveform_resource(waveform_id)
            document = self.project_context.adapter.waveform_document(waveform_id)
        except Exception as error:  # noqa: BLE001
            self.status_bar.fail(str(error))
            return

        signal = document.signal
        signal_type = self._signal_type_for_series(signal)
        unit = signal.unit
        self.info_name.setText(str(row.get("name") or document.title or "Waveform"))
        self.info_duration.setText(f"{signal.duration_s:.6g} s")

        if str(signal.quantity) not in {"current", "voltage"}:
            self.info_quantity.setText("Unclassified")
        else:
            self.info_quantity.setText("Voltage" if signal_type == "voltage" else "Current")
        self.info_units.setText(unit or "Not specified")
        self.info_peak_to_peak.setText(self._peak_to_peak_text(signal))

        electrical = self._electrical_metadata(document)
        slew = electrical.get("slew_rate_per_s")
        self.slew_label.setText("Voltage slew rate" if signal_type == "voltage" else "Current slew rate")
        if slew is None:
            self.info_slew.setText("Not specified")
        elif float(slew) <= 0.0:
            self.info_slew.setText("No limit")
        else:
            display_unit = unit if unit in self._units_for_signal_type(signal_type) else self._units_for_signal_type(signal_type)[0]
            display_slew = self._display_slew_from_si(float(slew), display_unit)
            self.info_slew.setText(f"{display_slew:.6g} {display_unit}/ms")

        voltage = signal_type == "voltage"
        self.output_resistance_label.setVisible(voltage)
        self.info_output_resistance.setVisible(voltage)
        self.edit_output_resistance.setVisible(False)
        if voltage:
            resistance = float(electrical.get("output_resistance_ohm", 0.0) or 0.0)
            self.info_output_resistance.setText(f"{resistance:.6g} Ω")

        metadata = dict(document.metadata)
        self.info_origin.setText(self._source_summary(document))
        notes = metadata.get("notes", metadata.get("note", ""))
        self.notes_view.setPlainText(str(notes or "").strip())
        self.info_empty_hint.setVisible(False)

    def _current_resource_name(self, fallback: str = "Waveform") -> str:
        if self.current_waveform_id:
            try:
                return str(self.project_context.adapter.waveform_resource(self.current_waveform_id).get("name") or fallback)
            except Exception:
                pass
        return fallback

    def _document_with_info_edits(self, *, title: str) -> WaveformDocument:
        document = self.native_editor.document(title=title)
        current = document.signal
        signal_type = str(self.edit_quantity.currentData() or "current")
        unit = self.edit_units.currentText().strip()
        if signal_type not in {"current", "voltage"}:
            raise ValueError("Waveforms must be either Current or Voltage signals.")
        if unit not in self._units_for_signal_type(signal_type):
            raise ValueError("Choose units that match the selected Current or Voltage signal type.")
        signal = WaveformSeries(
            name=current.name or title or "Waveform",
            time_s=current.time_s,
            values=current.values,
            quantity=signal_type,
            unit=unit,
            metadata=current.metadata,
        )

        metadata = dict(document.metadata)
        electrical = self._electrical_metadata(document)
        electrical["slew_rate_per_s"] = float(self._slew_si_from_edit())
        # A voltage waveform is a source command and may carry its Thevenin-like
        # series output resistance. Current sources intentionally have no output
        # resistance model yet; compliance limits can be added separately later.
        if signal_type == "voltage":
            electrical["output_resistance_ohm"] = float(self.edit_output_resistance.value())
        else:
            electrical.pop("output_resistance_ohm", None)
        metadata["electrical"] = electrical

        notes_text = self.edit_notes.toPlainText().strip()
        if notes_text != self._edit_metadata_baseline.get("notes", ""):
            metadata.pop("notes", None)
            metadata.pop("note", None)
            if notes_text:
                metadata["notes"] = notes_text
        return WaveformDocument(
            [signal],
            title=title,
            metadata=metadata,
            source_format=document.source_format,
            source_path=document.source_path,
        )

    def save_to_project(self) -> None:
        if self._task_mode is None:
            return
        if not self._encoding_confirmed or self._encoding_setup_active:
            self.status_bar.fail("Create or apply the waveform encoding before saving.")
            return
        if self._active_tool_panel is not None:
            self.status_bar.fail("Apply or Cancel the active waveform tool preview before saving.")
            return
        name = self.edit_name.text().strip() or self._working_name or self._current_resource_name("Waveform")
        if not name:
            QMessageBox.information(self, "Save waveform", "Enter a waveform name.")
            return
        try:
            native = self._document_with_info_edits(title=name)
            editor_state = {}
            if self.current_waveform_id:
                try:
                    editor_state = dict(
                        self.project_context.adapter.waveform_resource(self.current_waveform_id).get("editor", {})
                    )
                except Exception:
                    editor_state = {}
            waveform_id = self.project_context.adapter.save_waveform_document(
                native,
                name=name,
                waveform_id=self.current_waveform_id,
                editor=editor_state,
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to save waveform", str(error))
            return
        self.current_waveform_id = waveform_id
        self._show_browse_mode()
        self.refresh_project_resources(waveform_id, preserve_working_copy=False)
        self.projectResourcesChanged.emit()
        self.status_bar.set(f"Saved {self._current_resource_name(name)!r} to the project.")

    def load_selected_resource(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            self.current_waveform_id = None
            self.native_editor.show_empty_preview()
            self._clear_resource_summary()
            self.setWindowTitle(f"Waveform Editor — {self.project_context.display_name()}")
            self._show_browse_mode()
            return
        try:
            row = self.project_context.adapter.waveform_resource(waveform_id)
            native = self.project_context.adapter.waveform_document(waveform_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to preview waveform", str(error))
            return
        self.native_editor.load_document(native)
        self.current_waveform_id = waveform_id
        self._update_resource_summary()
        self._show_browse_mode()
        name = str(row.get("name") or "Waveform")
        self.setWindowTitle(f"Waveform Editor — {self.project_context.display_name()} — {name}")
        self.status_bar.set(f"Previewing {name!r}.")

    def edit_selected_resource(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            self.status_bar.fail("Select a waveform to edit.")
            return
        if self.current_waveform_id != waveform_id:
            self.load_selected_resource()
        self._enter_task_mode("edit", original_id=waveform_id)
        recipe = self.native_editor.exact_generated_recipe()
        if recipe is not None and recipe_is_fully_generated(recipe):
            self.status_bar.set(
                "Editing the selected waveform. Choose a waveform tool below the viewer or reopen a generated component."
            )
        else:
            self.status_bar.set("Editing the selected waveform. Choose a waveform tool below the viewer, then Save waveform or Cancel.")

    @staticmethod
    def _recipe_segment_label(index: int, segment: dict) -> str:
        kind = str(segment.get("kind") or "").strip().lower()
        if kind == "encoded":
            source = str(segment.get("source_name") or "Encoded import").strip() or "Encoded import"
            count = len(segment.get("raw_values") or ())
            interval_ms = float(segment.get("sample_interval_s", 0.0) or 0.0) * 1000.0
            start = float(segment.get("start_s", 0.0) or 0.0)
            duration = float(segment.get("duration_s", 0.0) or 0.0)
            return (
                f"{index + 1}. Encoded · {source} · {start:.6g}–{start + duration:.6g} s"
                f" · {count} points @ {interval_ms:.6g} ms"
            )
        if kind == "sampled":
            description = str(segment.get("description") or "Imported waveform").strip() or "Imported waveform"
            start = float(segment.get("start_s", 0.0) or 0.0)
            duration = float(segment.get("duration_s", 0.0) or 0.0)
            count = len(segment.get("values") or ())
            return (
                f"{index + 1}. Imported · {description} · {start:.6g}–{start + duration:.6g} s"
                f" · {count} samples"
            )
        if kind == "compiled":
            description = str(segment.get("description") or "Compiled waveform").strip() or "Compiled waveform"
            start = float(segment.get("start_s", 0.0) or 0.0)
            duration = float(segment.get("duration_s", 0.0) or 0.0)
            return f"{index + 1}. Compiled · {description} · {start:.6g}–{start + duration:.6g} s"
        generator = str(segment.get("generator") or "waveform").replace("_", " ").title()
        start = float(segment.get("start_s", 0.0) or 0.0)
        duration = float(segment.get("duration_s", 0.0) or 0.0)
        details = ""
        parameters = segment.get("parameters")
        if isinstance(parameters, dict):
            if "frequency_hz" in parameters:
                details = f" · {float(parameters['frequency_hz']):.6g} Hz"
            elif generator.lower().startswith("free form"):
                points = parameters.get("control_points")
                if isinstance(points, (list, tuple)):
                    details = f" · {len(points)} anchors"
        return f"{index + 1}. {generator} · {start:.6g}–{start + duration:.6g} s{details}"

    @staticmethod
    def _chunk_list_label(index: int, segment: dict) -> str:
        """Return a compact object-list style label for the left chunk browser."""

        kind = str(segment.get("kind") or "").strip().lower()
        duration = float(segment.get("duration_s", 0.0) or 0.0)
        if kind == "generator":
            generator = str(segment.get("generator") or "waveform").replace("_", " ").title()
            parameters = segment.get("parameters")
            detail = ""
            if isinstance(parameters, dict) and "frequency_hz" in parameters:
                detail = f" · {float(parameters['frequency_hz']):.6g} Hz"
            return f"{index + 1}  {generator}{detail} · {duration:.6g} s"
        if kind == "encoded":
            source = str(segment.get("source_name") or "Encoded import").strip() or "Encoded import"
            return f"{index + 1}  Encoded · {source} · {duration:.6g} s"
        if kind == "sampled":
            description = str(segment.get("description") or "Imported waveform").strip() or "Imported waveform"
            return f"{index + 1}  Imported · {description} · {duration:.6g} s"
        description = str(segment.get("description") or kind or "Waveform component").strip()
        return f"{index + 1}  {description} · {duration:.6g} s"

    def _refresh_chunk_list(self, select_row: int | None = None) -> None:
        """Refresh the authored-chunk browser while preserving its selection."""

        if not hasattr(self, "chunk_list"):
            return
        previous = self.chunk_list.currentRow() if select_row is None else int(select_row)
        recipe = self.native_editor.exact_authoring_recipe() if self._encoding_confirmed else None
        segments = list(recipe.get("segments", [])) if recipe else []
        self.chunk_list.blockSignals(True)
        self.chunk_list.clear()
        for index, raw in enumerate(segments):
            segment = dict(raw)
            item = QListWidgetItem(self._chunk_list_label(index, segment), self.chunk_list)
            item.setData(_RESOURCE_ID_ROLE, index)
            item.setToolTip(self._recipe_segment_label(index, segment))
        if segments:
            row = previous if 0 <= previous < len(segments) else min(max(previous, 0), len(segments) - 1)
            self.chunk_list.setCurrentRow(row)
        else:
            row = -1
            self.chunk_list.setCurrentRow(-1)
        self.chunk_list.blockSignals(False)
        self.native_editor.plot.set_selected_segment(row)
        if hasattr(self, "edit_chunk_button"):
            self._sync_action_states()

    def _select_chunk_row(self, index: int) -> None:
        """Mirror viewer chunk selection into the left chunk browser."""

        if not hasattr(self, "chunk_list"):
            return
        row = int(index)
        self.chunk_list.blockSignals(True)
        self.chunk_list.setCurrentRow(row if 0 <= row < self.chunk_list.count() else -1)
        self.chunk_list.blockSignals(False)
        self._sync_action_states()

    def _chunk_selection_changed(self, row: int) -> None:
        if hasattr(self, "native_editor"):
            self.native_editor.plot.set_selected_segment(int(row))
        self._sync_action_states()

    def _selected_chunk(self) -> tuple[int, dict] | None:
        if self._task_mode is None or not self._encoding_confirmed:
            return None
        recipe = self.native_editor.exact_authoring_recipe()
        if recipe is None:
            return None
        row = self.chunk_list.currentRow()
        segments = list(recipe.get("segments", []))
        if row < 0 or row >= len(segments):
            return None
        return row, dict(segments[row])

    def edit_selected_chunk(self) -> None:
        """Open the selected editable generated/encoded component under the viewer."""

        selected = self._selected_chunk()
        if selected is None:
            self.status_bar.fail("Select a waveform component to edit.")
            return
        index, segment = selected
        kind = str(segment.get("kind") or "").strip().lower()
        if kind not in {"generator", "encoded"}:
            self.status_bar.fail(
                "This sampled import is lossless and reorderable, but it does not have authored controls to reopen."
            )
            return
        self._open_recipe_segment_builder(index, segment)

    def clone_selected_chunk(self) -> None:
        selected = self._selected_chunk()
        if selected is None or self._active_tool_panel is not None:
            self.status_bar.fail("Select a waveform component to clone.")
            return
        index, _segment = selected
        try:
            new_index = self.native_editor.clone_recipe_segment(index)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to clone waveform component", str(error))
            return
        self._refresh_chunk_list(select_row=new_index)
        self.status_bar.set(f"Cloned waveform component {index + 1} as component {new_index + 1}.")

    def _move_selected_chunk(self, offset: int) -> None:
        selected = self._selected_chunk()
        if selected is None or self._active_tool_panel is not None:
            self.status_bar.fail("Select a waveform component to move.")
            return
        index, _segment = selected
        recipe = self.native_editor.exact_authoring_recipe() or {}
        segments = list(recipe.get("segments", []))
        target = index + int(offset)
        if target < 0 or target >= len(segments):
            return
        self.native_editor.reorder_recipe_segment(index, target)
        self._refresh_chunk_list(select_row=target)
        self._update_working_derived_info()
        self._sync_action_states()
        direction = "up" if target < index else "down"
        self.status_bar.set(
            f"Moved waveform component {index + 1} {direction} to position {target + 1}."
        )

    def move_selected_chunk_up(self) -> None:
        self._move_selected_chunk(-1)

    def move_selected_chunk_down(self) -> None:
        self._move_selected_chunk(1)

    def delete_selected_chunk(self) -> None:
        selected = self._selected_chunk()
        if selected is None or self._active_tool_panel is not None:
            self.status_bar.fail("Select a waveform component to delete.")
            return
        index, segment = selected
        label = self._chunk_list_label(index, segment)
        if QMessageBox.question(
            self,
            "Delete waveform component",
            f"Delete {label}?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.native_editor.delete_recipe_segment(index)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to delete waveform component", str(error))
            return
        next_row = min(index, self.chunk_list.count() - 2)
        self._refresh_chunk_list(select_row=next_row)
        self.status_bar.set(f"Deleted waveform component {index + 1}.")

    def _open_recipe_segment_builder(self, index: int, segment: dict) -> None:
        kind = str(segment.get("kind") or "").strip().lower()
        if kind == "encoded":
            self._open_embedded_encoded_segment(index, segment)
            return
        generator = str(segment.get("generator") or "").strip().lower()
        if generator not in {"square", "sine", "dc", "action_potential", "free_form"}:
            QMessageBox.information(
                self,
                "Edit generated component",
                f"The {generator or 'unknown'} generator does not have an editable builder.",
            )
            return
        self._open_embedded_generator(
            generator,
            recipe_segment=segment,
            recipe_segment_index=int(index),
        )

    def _edit_authored_segment_index(self, index: int) -> None:
        """Open a double-clicked editable authored chunk in the embedded editor."""
        if self._task_mode is None or self._active_tool_panel is not None:
            return
        recipe = self.native_editor.exact_authoring_recipe()
        if recipe is None:
            return
        segments = list(recipe.get("segments", []))
        target = int(index)
        if target < 0 or target >= len(segments):
            return
        self._select_chunk_row(target)
        segment = dict(segments[target])
        if str(segment.get("kind") or "") not in {"generator", "encoded"}:
            self.status_bar.fail("This sampled component does not have editable authoring controls to reopen.")
            return
        self._open_recipe_segment_builder(target, segment)

    def cancel_task(self) -> None:
        if self._task_mode is None:
            return
        if self._working_copy_dirty() and QMessageBox.question(
            self, "Cancel waveform changes", "Discard the unsaved waveform changes?"
        ) != QMessageBox.StandardButton.Yes:
            return
        original_id = self._task_original_id
        self._task_mode = None
        self._task_original_id = None
        if original_id:
            self.refresh_project_resources(original_id)
        elif self.resource_list.count():
            self.refresh_project_resources(self._selected_resource_id())
        else:
            self.current_waveform_id = None
            self.native_editor.show_empty_preview()
            self._clear_resource_summary()
            self._show_browse_mode()
        self.status_bar.set("Changes cancelled.")

    def rename_selected_resource(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            return
        try:
            row = self.project_context.adapter.waveform_resource(waveform_id)
        except Exception as error:
            QMessageBox.warning(self, "Rename waveform", str(error))
            return
        name, ok = QInputDialog.getText(self, "Rename waveform", "Name:", text=str(row.get("name") or "Waveform"))
        if not ok or not name.strip():
            return
        try:
            self.project_context.adapter.rename_waveform(waveform_id, name.strip())
        except Exception as error:
            QMessageBox.warning(self, "Rename waveform", str(error))
            return
        self.refresh_project_resources(waveform_id)
        self.projectResourcesChanged.emit()

    def duplicate_selected_resource(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            return
        try:
            duplicate_id = self.project_context.adapter.duplicate_waveform(waveform_id)
        except Exception as error:
            QMessageBox.warning(self, "Duplicate waveform", str(error))
            return
        self.refresh_project_resources(duplicate_id)
        self.projectResourcesChanged.emit()

    def delete_selected_resource(self) -> None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None:
            return
        item = self.resource_list.currentItem()
        name = item.text() if item is not None else "the selected waveform"
        answer = QMessageBox.question(self, "Delete waveform", f"Delete {name!r} from this project?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.project_context.adapter.delete_waveforms([waveform_id])
        except Exception as error:
            QMessageBox.warning(self, "Delete waveform", str(error))
            return
        if self.current_waveform_id == waveform_id:
            self.current_waveform_id = None
        self.refresh_project_resources()
        self.projectResourcesChanged.emit()

    # --- native document ---------------------------------------------
    def new_waveform(self) -> None:
        if self._task_mode is not None:
            if self._working_copy_dirty() and QMessageBox.question(
                self, "Create waveform", "Discard the current unsaved waveform changes?"
            ) != QMessageBox.StandardButton.Yes:
                return
        original_id = self._selected_resource_id()
        self.native_editor.new_document()
        self.current_waveform_id = None
        self._working_name = "Waveform"
        self._enter_task_mode("create", original_id=original_id)
        self.setWindowTitle(f"Waveform Editor — {self.project_context.display_name()} — New waveform")
        self.status_bar.set("Set the waveform information below the viewer, then choose Create waveform to begin authoring.")

    @staticmethod
    def _import_name(document: WaveformDocument) -> str:
        return str(
            document.title
            or document.signal.name
            or (Path(document.source_path).stem if document.source_path else "Imported waveform")
        ).strip() or "Imported waveform"

    def import_native_waveform(self) -> None:
        # While authoring, Import is a normal waveform tool: source/configuration
        # controls live below the persistent viewer and Apply appends exactly one
        # lossless sampled chunk. Catalogue import intentionally keeps the larger
        # multi-signal/profile-management dialog.
        if self._task_mode is not None:
            self._open_embedded_waveform_import()
            return

        dialog = WaveformImportDialog(self, single_signal=False)
        if dialog.exec() != dialog.DialogCode.Accepted or not dialog.result_documents:
            return
        documents = tuple(dialog.result_documents)
        saved_ids: list[str] = []
        try:
            for document in documents:
                saved_ids.append(
                    self.project_context.adapter.save_waveform_document(
                        document, name=self._import_name(document)
                    )
                )
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to import waveform", str(error))
            return
        select_id = saved_ids[-1] if saved_ids else None
        self.refresh_project_resources(select_id)
        self.projectResourcesChanged.emit()
        count = len(saved_ids)
        self.status_bar.set(
            f"Imported {count} waveform{'s' if count != 1 else ''} into the project catalogue."
        )

    def _open_embedded_runtime_profile_source(self, profile, source: Path) -> None:
        """Switch a runtime-timed declarative encoding into its editable inline controls."""

        self._open_embedded_runtime_import(profile, Path(source))

    def _selected_saved_waveform(self) -> tuple[str, WaveformDocument] | None:
        waveform_id = self._selected_resource_id()
        if waveform_id is None or self._task_mode is not None:
            QMessageBox.information(
                self,
                "Export waveform",
                "Select a saved waveform in the catalogue first. Save or Cancel any active edits before exporting.",
            )
            return None
        try:
            row = self.project_context.adapter.waveform_resource(waveform_id)
            document = self.project_context.adapter.waveform_document(waveform_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to export waveform", str(error))
            return None
        return str(row.get("name") or document.title or "Waveform"), document

    def export_selected_waveform(self) -> None:
        selected = self._selected_saved_waveform()
        if selected is None:
            return
        choice, accepted = QInputDialog.getItem(
            self,
            "Export waveform",
            "Export the selected saved waveform as:",
            ["Field Workbench waveform", "CSV table", "8-bit DAC", "Rigol waveform"],
            0,
            False,
        )
        if not accepted:
            return
        {
            "Field Workbench waveform": self.export_native_waveform,
            "CSV table": self.export_csv_waveform,
            "8-bit DAC": self.export_dac,
            "Rigol waveform": self.open_export,
        }[choice]()

    def export_native_waveform(self) -> None:
        selected = self._selected_saved_waveform()
        if selected is None:
            return
        name, native = selected
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export Field Workbench waveform",
            f"{name}.fwbwave.json",
            "Field Workbench waveform (*.fwbwave *.fwbwave.json);;JSON files (*.json)",
        )
        if not destination:
            return
        path = Path(destination)
        if not (path.name.lower().endswith(".fwbwave") or path.name.lower().endswith(".fwbwave.json")):
            path = path.with_name(path.name + ".fwbwave.json")
        try:
            write_waveform_document(path, native)
        except Exception as error:
            QMessageBox.warning(self, "Unable to export waveform", str(error))
            return
        self.status_bar.set(f"Exported {path.name}.")

    def export_csv_waveform(self) -> None:
        selected = self._selected_saved_waveform()
        if selected is None:
            return
        name, document = selected
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export waveform CSV", f"{name}.csv", "CSV table (*.csv);;All files (*)"
        )
        if not destination:
            return
        path = Path(destination)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(path.suffix + ".csv" if path.suffix else ".csv")
        try:
            write_waveform_csv(path, document)
        except Exception as error:
            QMessageBox.warning(self, "Unable to export waveform CSV", str(error))
            return
        self.status_bar.set(f"Exported {path.name}.")

    def export_dac(self) -> None:
        selected = self._selected_saved_waveform()
        if selected is None:
            return
        _name, document = selected
        DacExportDialog(self, document.signal).exec()

    def open_export(self) -> None:
        selected = self._selected_saved_waveform()
        if selected is None:
            return
        _name, document = selected
        NativeRigolExportDialog(self, document.signal).exec()

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Waveform Editor",
            f"<h2>Field Workbench Waveform Editor</h2>"
            f"<p>Field Workbench {__version__}</p>"
            "<p>The editor works in physical signal terms: Current or Voltage, engineering units, duration, "
            "frequency, magnitude, duty cycle, phase, and other shape controls. Internal waveform representation "
            "is handled automatically.</p>"
            "<p>The left Components list is the authored waveform structure: select a component there to Edit, Clone, or Delete it. "
            "Generated components reopen inline with their original Square, Sine, DC / pause, Action Potential, or Free-form settings rather than reverse-engineering samples. "
            "Runtime-timed encoded components retain their original source samples plus import-time timing/mapping controls and can be reopened the same way.</p>"
            "<p><b>Create new</b> first defines the waveform in the centre information card. After <b>Create waveform</b>, "
            "the confirmed Current/Voltage type and engineering unit move to the read-only left encoding card and every generator/import tool inherits them. "
            "Use the encoding card's <b>Edit</b> button only when the waveform-level definition itself needs to change.</p>"
            "<p>Catalogue import can map several source signals at once. During confirmed Create/Edit, <b>Import waveform…</b> lives with the other Waveform tools beneath the viewer and supports native Field Workbench files, text/CSV, reusable formats, and reusable declarative encodings. Its inline controls temporarily occupy that same tool area and preview one prospective appended component before Apply. "
            "Any encoding tagged for user-entered timing uses the same editable runtime-import component: the source samples and import-time timing/mapping are retained with the waveform. "
            "Device-specific representations and timing appear only when the selected encoding profile actually requires them.</p>"
            "<p>Export always operates on the selected saved catalogue waveform; save or cancel edits before exporting.</p>",
        )

    def apply_workbench_theme(self, theme: str) -> None:
        self.native_editor.apply_workbench_theme(theme)
        for action, standard_pixmap in self._standard_icon_specs:
            action.setIcon(themed_icon(self.style().standardIcon(standard_pixmap)))

    def prepare_project_close(self) -> bool:
        """Commit/discard/cancel an active working copy before project close.

        The project adapter cannot see an unsaved Waveform Editor working copy,
        so Project Manager calls this before deciding whether the project itself
        is dirty.  Saving here intentionally makes the adapter dirty, allowing the
        normal project Save/Discard/Cancel prompt to include the waveform change.
        """

        if self._active_tool_panel is not None:
            answer = QMessageBox.question(
                self,
                "Unapplied waveform tool preview",
                "Discard the active waveform tool preview and continue closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._project_close_prepared = False
                return False
            self._close_embedded_tool(restore_preview=True, sync=True)

        if self._task_mode is None:
            # Nothing private needs resolving. Keep the one-shot close flag
            # clear so a cancelled project close cannot arm a later dirty edit.
            self._project_close_prepared = False
            return True

        original_id = self._task_original_id or self.current_waveform_id
        if not self._working_copy_dirty():
            # The window is reused after close, so a clean Create/Edit session
            # must still be unwound back to the saved catalogue state.
            self._show_browse_mode()
            self.refresh_project_resources(original_id, preserve_working_copy=False)
            self._project_close_prepared = False
            return True
        answer = QMessageBox.warning(
            self,
            "Unsaved waveform changes",
            "Save the waveform changes before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            self._project_close_prepared = False
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.save_to_project()
            if self._task_mode is not None:
                self._project_close_prepared = False
                return False
        else:
            self._show_browse_mode()
            self.refresh_project_resources(original_id, preserve_working_copy=False)
        self._project_close_prepared = True
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._project_close_prepared and not self.prepare_project_close():
            event.ignore()
            return
        self._project_close_prepared = False
        event.accept()
