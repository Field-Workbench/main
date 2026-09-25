"""Single-signal catalogue, exposure binding, and editor integration checks."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QDialogButtonBox, QDoubleSpinBox, QTableWidgetItem
from fieldworkbench.waveform_io import WaveformDocument, WaveformSeries, WaveformIOError, waveform_signal_to_payload, waveform_document_from_payload
from fieldworkbench.studio_adapter import StudioAdapter, StudioOperationError
from fieldworkbench.exposures import new_exposure
from fieldworkbench.project_context import ProjectContext
from fieldworkbench.waveform_editor.ui.native_editor import NativeWaveformEditor
from fieldworkbench.waveform_editor.core.native import GeneratedTrace, dc_trace, sine_wave_trace
from fieldworkbench.waveform_editor.ui.main_window import WaveformEditorWindow
from fieldworkbench.waveform_import_dialog import TabularWaveformImportDialog, WaveformImportDialog
from fieldworkbench.exposure_library import ExposureEditorDialog, ExposureEditorWindow


def signal(quantity='current', unit='mA', slew=2.5, output_resistance=None):
    electrical = {'slew_rate_per_s': slew}
    if output_resistance is not None:
        electrical['output_resistance_ohm'] = output_resistance
    return WaveformDocument([WaveformSeries('Source', [1, 1.01, 1.02], [0, 10, -5], quantity, unit)],
        title='Source', metadata={'notes': 'Keep me', 'electrical': electrical})


def exposure_for(waveform_id, coil_ids=None, connection='series'):
    exposure = new_exposure('Sequence')
    exposure['protocol']['sources'] = [{'id': 'source_1', 'name': 'Source 1', 'waveform_id': waveform_id, 'playback_mode': 'continuous', 'phase_offset_s': 0.0}]
    exposure['protocol']['sequence']['steps'] = [{
        'id': 'step_1',
        'duration_s': 1.0,
        'drive_groups': [{
            'id': 'drive_1',
            'source_id': 'source_1',
                        'coil_ids': list(coil_ids or ['coil_1']),
            'connection': connection,
        }],
    }]
    return exposure


class WaveformCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.adapter = StudioAdapter({'objects': []})

    def test_editor_appends_first_chunk_then_reorders_generated_and_imported_chunks(self):
        editor = NativeWaveformEditor()
        editor.apply_generated_trace(dc_trace(duration_s=0.2, level=5.0), append=True)
        self.assertAlmostEqual(editor.current_signal().time_s[-1], 0.2)
        self.assertEqual(len(editor.exact_authoring_recipe()['segments']), 1)

        imported = WaveformDocument([
            WaveformSeries('DAC', [9.0, 9.1], [-2.0, -2.0], 'current', 'mA')
        ], title='DAC')
        editor.append_document(imported, description='DAC import')
        editor.apply_generated_trace(
            sine_wave_trace(frequency_hz=10.0, duration_s=0.1, peak_to_peak=2.0, offset=0.0),
            append=True,
        )
        recipe = editor.exact_authoring_recipe()
        self.assertEqual([item['kind'] for item in recipe['segments']], ['generator', 'sampled', 'generator'])
        editor.reorder_recipe_segment(2, 0)
        reordered = editor.exact_authoring_recipe()['segments']
        self.assertEqual(reordered[0]['generator'], 'sine')
        self.assertEqual(reordered[1]['generator'], 'dc')
        self.assertEqual(reordered[2]['kind'], 'sampled')
        self.assertAlmostEqual(editor.current_signal().time_s[-1], 0.4)

    def test_signal_payload_roundtrip_and_physics_separation(self):
        doc = signal()
        payload = waveform_signal_to_payload(doc)
        self.assertEqual(payload['version'], 2)
        self.assertNotIn('channels', payload)
        self.assertNotIn('output_resistance_ohm', payload['metadata']['electrical'])
        decoded = waveform_document_from_payload(payload)
        np.testing.assert_array_equal(decoded.signal.values, doc.signal.values)
        self.assertEqual(decoded.metadata['notes'], 'Keep me')

        for legacy_load_key in ('inductance_h', 'resistance_ohm', 'manual_resistance_ohm'):
            invalid = signal()
            invalid.metadata['electrical'][legacy_load_key] = 1.0
            with self.assertRaisesRegex(WaveformIOError, 'Unsupported waveform electrical field'):
                waveform_signal_to_payload(invalid)

        invalid_current = signal()
        invalid_current.metadata['electrical']['output_resistance_ohm'] = 50.0
        with self.assertRaisesRegex(WaveformIOError, 'Unsupported waveform electrical field'):
            waveform_signal_to_payload(invalid_current)

    def test_voltage_source_output_resistance_roundtrip_and_runtime_metadata(self):
        doc = signal('voltage', 'V', output_resistance=50)
        payload = waveform_signal_to_payload(doc)
        self.assertEqual(payload['metadata']['electrical']['output_resistance_ohm'], 50.0)
        self.assertNotIn('output_resistance_ohm', payload['signal']['metadata'])
        decoded = waveform_document_from_payload(payload)
        self.assertEqual(decoded.metadata['electrical']['output_resistance_ohm'], 50.0)

        wid = self.adapter.save_waveform_document(doc)
        _times, _values, runtime = self.adapter._waveform_source_trace(wid)
        self.assertEqual(runtime['output_resistance_ohm'], 50.0)

    def test_multi_signal_save_is_rejected_without_changes(self):
        doc = WaveformDocument([signal().signal, signal('voltage', 'V').signal])
        before = self.adapter.signature()
        with self.assertRaises(StudioOperationError):
            self.adapter.save_waveform_document(doc)
        self.assertEqual(before, self.adapter.signature())
        parts = doc.split_signals()
        ids = [self.adapter.save_waveform_document(part) for part in parts]
        self.assertEqual(len(set(ids)), 2)

    def test_native_file_export_uses_catalogue_signal_record(self):
        from fieldworkbench.waveform_io import NativeWaveformJSON
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'signal.fwbwave'
            NativeWaveformJSON().write(path, signal())
            payload = json.loads(path.read_text())
            self.assertIn('signal', payload)
            self.assertNotIn('channels', payload)
            decoded = NativeWaveformJSON().read(path)
            np.testing.assert_allclose(decoded.signal.time_s, signal().signal.time_s)
            self.assertEqual(decoded.metadata['electrical']['slew_rate_per_s'], 2.5)

    def test_invalid_electrical_source_is_rejected(self):
        for doc in [
            signal('arbitrary',''), signal('voltage','A'), signal(slew=-1), signal(slew=float('nan')),
            signal('voltage', 'V', output_resistance=-1),
            signal('voltage', 'V', output_resistance=float('nan')),
        ]:
            with self.assertRaises(StudioOperationError):
                self.adapter.save_waveform_document(doc)
        self.assertEqual(self.adapter.list_waveforms(), [])

    def test_project_reload_rename_and_reference_resolve(self):
        wid = self.adapter.save_waveform_document(signal())
        eid = self.adapter.save_exposure(exposure_for(wid))
        self.adapter.rename_waveform(wid, 'Renamed')
        adapter = StudioAdapter(self.adapter.document())
        protocol = adapter.exposure(eid)['protocol']
        group = protocol['sequence']['steps'][0]['drive_groups'][0]
        self.assertEqual(group['source_id'], 'source_1')
        self.assertNotIn('waveform_name', group)
        self.assertEqual(adapter.list_exposures()[0]['waveform'], 'Renamed')
        times, values, metadata = adapter._waveform_source_trace(wid)
        np.testing.assert_allclose(times, [0, .01, .02])
        np.testing.assert_allclose(values, [0, .01, -.005])
        self.assertEqual(metadata['waveform_name'], 'Renamed')
        self.assertEqual(metadata['slew_rate_per_s'], 2.5)
        self.assertEqual(adapter.waveform_document(wid).title, 'Renamed')
        with self.assertRaises(StudioOperationError):
            adapter.delete_waveforms([wid])

    def test_edit_duplicate_undo_redo_keep_independent_samples(self):
        wid = self.adapter.save_waveform_document(signal())
        duplicate = self.adapter.duplicate_waveform(wid)
        replacement = signal(); replacement.signal.values[1] = 20
        self.adapter.save_waveform_document(replacement, waveform_id=wid)
        self.assertEqual(self.adapter.waveform_document(duplicate).signal.values[1], 10)
        self.adapter.undo()
        self.assertEqual(self.adapter.waveform_document(wid).signal.values[1], 10)
        self.adapter.redo()
        self.assertEqual(self.adapter.waveform_document(wid).signal.values[1], 20)

    def test_portable_exposure_roundtrip_and_units(self):
        wid = self.adapter.save_waveform_document(signal('voltage', 'mV'))
        eid = self.adapter.save_exposure(exposure_for(wid))
        with tempfile.TemporaryDirectory() as tmp:
            path = self.adapter.export_exposure(eid, Path(tmp)/'exposure.json')
            other = StudioAdapter({'objects': []})
            imported = other.import_exposure_file(path)
        protocol = other.exposure(imported)['protocol']
        imported_id = protocol['sources'][0]['waveform_id']
        self.assertNotEqual(imported_id, wid)
        np.testing.assert_allclose(other._waveform_source_trace(imported_id)[1], [0,.01,-.005])
        self.assertEqual(other.waveform_document(imported_id).signal.quantity, 'voltage')

    def test_missing_drive_group_source_reference_fails_explicitly(self):
        with self.assertRaises(StudioOperationError):
            self.adapter.resolve_exposure_drive_group({
                'waveform_id': 'missing', 'coil_ids': ['coil_1'], 'connection': 'series'
            })

    def test_native_editor_owns_one_signal_and_clears_preview(self):
        editor = NativeWaveformEditor()
        try:
            editor.load_document(signal())
            with self.assertRaises(WaveformIOError):
                editor.load_document(WaveformDocument([signal().signal]*2))
            self.assertFalse(hasattr(editor, 'channel_combo'))
            self.assertFalse(hasattr(editor, 'add_channel'))
            editor.show_empty_preview()
            self.assertIsNone(editor.plot.signal)
        finally:
            editor.deleteLater()

    def test_new_waveform_confirms_encoding_before_tools_and_encoding_edit_can_cancel(self):
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.show(); self.app.processEvents()
            window.new_waveform()
            self.assertEqual(window._task_mode, 'create')
            self.assertFalse(window._encoding_confirmed)
            self.assertTrue(window._encoding_setup_active)
            self.assertIs(window.left_stack.currentWidget(), window.catalogue_group)
            self.assertIs(window.lower_stack.currentWidget(), window.encoding_setup_group)
            self.assertFalse(window.import_action.isEnabled())

            window.edit_name.setText('Voltage test')
            window.edit_quantity.setCurrentIndex(window.edit_quantity.findData('voltage'))
            window.edit_units.setCurrentText('V')
            window._confirm_encoding_setup()
            self.assertTrue(window._encoding_confirmed)
            self.assertFalse(window._encoding_setup_active)
            self.assertIs(window.left_stack.currentWidget(), window.edit_options_group)
            self.assertIs(window.lower_stack.currentWidget(), window.toolbox_group)
            self.assertEqual(window.encoding_quantity.text(), 'Voltage')
            self.assertEqual(window.encoding_units.text(), 'V')
            self.assertEqual(window.native_editor.current_signal().quantity, 'voltage')
            self.assertEqual(window.native_editor.current_signal().unit, 'V')

            window._begin_encoding_edit()
            window.edit_quantity.setCurrentIndex(window.edit_quantity.findData('current'))
            window._cancel_encoding_setup()
            self.assertEqual(window.native_editor.current_signal().quantity, 'voltage')
            self.assertEqual(window.native_editor.current_signal().unit, 'V')
            self.assertEqual(window.encoding_quantity.text(), 'Voltage')
        finally:
            window._show_browse_mode()
            window.close(); window.deleteLater()

    def test_first_import_chunk_cannot_redefine_confirmed_encoding(self):
        editor = NativeWaveformEditor()
        try:
            editor.set_current_signal_definition('voltage', 'V')
            incompatible = WaveformDocument([
                WaveformSeries('Current', [0.0, 0.1], [0.0, 1.0], 'current', 'mA')
            ])
            with self.assertRaisesRegex(WaveformIOError, 'confirmed signal type and units'):
                editor.show_document_preview(incompatible)
            with self.assertRaisesRegex(WaveformIOError, 'confirmed signal type and units'):
                editor.append_document(incompatible)
            self.assertEqual(editor.current_signal().quantity, 'voltage')
            self.assertEqual(editor.current_signal().unit, 'V')
        finally:
            editor.deleteLater()

    def test_editor_browse_edit_import_choice_and_cancel(self):
        wid = self.adapter.save_waveform_document(signal())
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.show(); self.app.processEvents()
            self.assertFalse(hasattr(window.native_editor, 'editor_panel'))
            window.edit_selected_resource()
            self.assertFalse(hasattr(window.native_editor, 'editor_panel'))
            self.assertFalse(hasattr(window, 'edit_data_button'))
            self.assertFalse(hasattr(window, 'toggle_data_editor'))
            window.native_editor.apply_generated_trace(
                GeneratedTrace(
                    (0.0, 0.01, 0.02),
                    (0.0, 20.0, 0.0),
                    {
                        "shape": "sine",
                        "duration_s": 0.02,
                        "frequency_hz": 25.0,
                        "peak_to_peak": 40.0,
                        "offset": 0.0,
                        "phase_deg": 0.0,
                    },
                )
            )
            window._begin_encoding_edit()
            window.edit_name.setText('Edited')
            window._confirm_encoding_setup()
            window.save_to_project()
            self.assertEqual(self.adapter.waveform_document(wid).signal.values[1], 20)
            window.edit_selected_resource()
            window.native_editor.apply_generated_trace(
                GeneratedTrace(
                    (0.0, 0.01, 0.02),
                    (0.0, 99.0, 0.0),
                    {
                        "shape": "sine",
                        "duration_s": 0.02,
                        "frequency_hz": 25.0,
                        "peak_to_peak": 198.0,
                        "offset": 0.0,
                        "phase_deg": 0.0,
                    },
                )
            )
            with patch('fieldworkbench.waveform_editor.ui.main_window.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
                window.cancel_task()
            self.assertEqual(window.native_editor.document().signal.values[1], 20)
            self.assertFalse(hasattr(window, '_select_imported_signal'))
            self.assertEqual(len(self.adapter.list_waveforms()), 1)
            self.assertEqual(window._import_name(signal('voltage', 'V')), 'Source')
        finally:
            window.close(); window.deleteLater()


    def test_phase5_manual_input_maps_columns_to_separate_waveforms(self):
        dialog = TabularWaveformImportDialog()
        try:
            dialog.data_table.setColumnCount(3)
            dialog.data_table.setHorizontalHeaderLabels(['Time (ms)', 'I1', 'V1'])
            rows = ((0, 0, 1), (1, 10, 2), (2, -5, 3))
            for row, values in enumerate(rows):
                for column, value in enumerate(values):
                    dialog.data_table.setItem(row, column, QTableWidgetItem(str(value)))
            dialog._rebuild_column_mapping()
            dialog.time_column_combo.setCurrentIndex(dialog.time_column_combo.findData(0))
            dialog.time_unit_combo.setCurrentText('ms')
            quantity = dialog.signal_mapping.cellWidget(2, 3)
            quantity.setCurrentIndex(quantity.findData('voltage'))
            dialog._accept_import()
            self.assertEqual(len(dialog.result_documents), 2)
            self.assertEqual(dialog.result_documents[0].signal.quantity, 'current')
            self.assertEqual(dialog.result_documents[1].signal.quantity, 'voltage')
            np.testing.assert_allclose(dialog.result_documents[0].signal.time_s, [0, .001, .002])
        finally:
            dialog.close(); dialog.deleteLater()


    def test_catalogue_import_native_reader_can_select_waveforms_from_project(self):
        source = StudioAdapter({'objects': []})
        current_id = source.save_waveform_document(signal(), name='Project current')
        voltage_id = source.save_waveform_document(
            signal('voltage', 'V', output_resistance=50), name='Project voltage'
        )
        self.assertNotEqual(current_id, voltage_id)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.magpy.json'
            source.save_file(path)
            dialog = WaveformImportDialog()
            try:
                dialog._decode_selected_file(str(path))
                self.assertEqual(dialog.signal_list.count(), 2)
                self.assertIn('Project current', dialog.signal_list.item(0).text())
                self.assertIn('Project voltage', dialog.signal_list.item(1).text())
                self.assertIn('Project waveform 1 of 2', dialog.summary.text())
                self.assertTrue(
                    dialog.buttons.button(QDialogButtonBox.StandardButton.Open).isEnabled()
                )

                dialog.signal_list.item(0).setCheckState(Qt.CheckState.Unchecked)
                dialog._accept_import()
                self.assertEqual(len(dialog.result_documents), 1)
                imported = dialog.result_documents[0]
                self.assertEqual(imported.title, 'Project voltage')
                self.assertEqual(imported.source_path, str(path))
                self.assertEqual(imported.signal.quantity, 'voltage')
                self.assertEqual(
                    imported.metadata['electrical']['output_resistance_ohm'], 50.0
                )
            finally:
                dialog.close(); dialog.deleteLater()

    def test_phase5_catalogue_import_saves_every_selected_signal(self):
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        documents = (
            signal('current', 'mA'),
            WaveformDocument([WaveformSeries('Voltage A', [0, .01, .02], [0, 1, 0], 'voltage', 'V')], title='Voltage A'),
        )
        class FakeImportDialog:
            class DialogCode:
                Accepted = 1
            result_documents = documents
            def __init__(self, *_args, **_kwargs): pass
            def exec(self): return 1
        try:
            with patch('fieldworkbench.waveform_editor.ui.main_window.WaveformImportDialog', FakeImportDialog):
                window.import_native_waveform()
            rows = self.adapter.list_waveforms()
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['quantity'] for row in rows}, {'current', 'voltage'})
        finally:
            window.close(); window.deleteLater()

    def test_editor_voltage_output_resistance_is_source_only(self):
        wid = self.adapter.save_waveform_document(signal('voltage', 'V', output_resistance=50))
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.show(); self.app.processEvents()
            self.assertFalse(window.output_resistance_label.isHidden())
            self.assertEqual(window.info_output_resistance.text(), '50 Ω')
            window.edit_selected_resource()
            self.assertEqual(window.encoding_quantity.text(), 'Voltage')
            self.assertEqual(window.encoding_units.text(), 'V')
            window._begin_encoding_edit()
            self.assertFalse(window.edit_output_resistance.isHidden())
            self.assertAlmostEqual(window.edit_output_resistance.value(), 50.0)
            window.edit_output_resistance.setValue(75.0)
            window._confirm_encoding_setup()
            window.save_to_project()
            self.assertEqual(
                self.adapter.waveform_document(wid).metadata['electrical']['output_resistance_ohm'],
                75.0,
            )

            window.edit_selected_resource()
            window._begin_encoding_edit()
            window.edit_quantity.setCurrentIndex(window.edit_quantity.findData('current'))
            self.app.processEvents()
            self.assertTrue(window.edit_output_resistance.isHidden())
            window._confirm_encoding_setup()
            window.save_to_project()
            self.assertNotIn(
                'output_resistance_ohm',
                self.adapter.waveform_document(wid).metadata['electrical'],
            )
        finally:
            window.close(); window.deleteLater()


    def test_adapter_electrical_resolver_combines_source_and_model_loads(self):
        wid = self.adapter.save_waveform_document(signal('voltage', 'V', output_resistance=50))
        loads = {
            'coil_a': {'resistance_ohm': 2.0, 'inductance_h': 0.01, 'resistance_mode': 'calculated', 'inductance_mode': 'estimated'},
            'coil_b': {'resistance_ohm': 3.0, 'inductance_h': 0.02, 'resistance_mode': 'override', 'inductance_mode': 'override'},
        }
        with patch.object(self.adapter, 'coil_electrical_load', side_effect=lambda coil_id: loads[coil_id]):
            resolved = self.adapter.resolve_exposure_drive_group({
                'id': 'drive_test',
                'waveform_id': wid,
                'coil_ids': ['coil_a', 'coil_b'],
                'connection': 'series',
            })
        self.assertEqual(resolved['source_kind'], 'voltage')
        self.assertEqual(resolved['source_output_resistance_ohm'], 50.0)
        self.assertAlmostEqual(resolved['equivalent_resistance_ohm'], 5.0)
        self.assertAlmostEqual(resolved['equivalent_inductance_h'], 0.03)
        self.assertEqual(set(resolved['coil_currents_a']), {'coil_a', 'coil_b'})
        np.testing.assert_allclose(resolved['coil_currents_a']['coil_a'], resolved['coil_currents_a']['coil_b'])

    def test_drive_group_diagnostics_report_load_peaks_and_cautions(self):
        wid = self.adapter.save_waveform_document(signal('voltage', 'V', output_resistance=2.0))
        loads = {
            'coil_a': {'resistance_ohm': 2.0, 'inductance_h': 0.01, 'resistance_mode': 'estimated', 'inductance_mode': 'estimated'},
            'coil_b': {'resistance_ohm': 3.0, 'inductance_h': 0.02, 'resistance_mode': 'override', 'inductance_mode': 'override'},
        }
        with patch.object(self.adapter, 'coil_electrical_load', side_effect=lambda coil_id: loads[coil_id]):
            diagnostic = self.adapter.exposure_drive_group_diagnostics({
                'waveform_id': wid,
                'coil_ids': ['coil_a', 'coil_b'],
                'connection': 'series',
            }, sample_count=129)
        self.assertAlmostEqual(diagnostic['equivalent_resistance_ohm'], 5.0)
        self.assertAlmostEqual(diagnostic['equivalent_inductance_h'], 0.03)
        self.assertGreaterEqual(diagnostic['peak_source_current_a'], 0.0)
        self.assertGreaterEqual(diagnostic['peak_source_voltage_v'], 0.0)
        self.assertEqual(set(diagnostic['branch_peaks']), {'coil_a', 'coil_b'})
        self.assertTrue(any('Mutual inductance' in text for text in diagnostic['warnings']))
        self.assertTrue(any('geometry-estimated' in text for text in diagnostic['warnings']))
        self.assertFalse(any('current or power limit' in text for text in diagnostic['notes']))

    def test_shared_source_diagnostics_apply_output_resistance_once(self):
        wid = self.adapter.save_waveform_document(signal('voltage', 'V', output_resistance=1.0))
        loads = {
            coil_id: {'resistance_ohm': 2.0, 'inductance_h': 0.1, 'resistance_mode': 'calculated', 'inductance_mode': 'override'}
            for coil_id in ('a', 'b', 'c', 'd', 'e', 'f')
        }
        groups = [
            {'id': 'g1', 'coil_ids': ['a', 'b'], 'connection': 'series'},
            {'id': 'g2', 'coil_ids': ['c', 'd'], 'connection': 'series'},
            {'id': 'g3', 'coil_ids': ['e', 'f'], 'connection': 'series'},
        ]
        with patch.object(self.adapter, 'coil_electrical_load', side_effect=lambda coil_id: loads[coil_id]):
            diagnostic = self.adapter.exposure_source_network_diagnostics(
                groups, waveform_id=wid, sample_count=129
            )
        self.assertEqual(diagnostic['network_group_count'], 3)
        self.assertAlmostEqual(diagnostic['equivalent_resistance_ohm'], 4.0 / 3.0)
        self.assertAlmostEqual(diagnostic['equivalent_inductance_h'], 0.2 / 3.0)
        self.assertEqual(diagnostic['source_output_resistance_ohm'], 1.0)
        self.assertTrue(any('parallel routes' in text for text in diagnostic['notes']))

    def test_exposure_editor_uses_drive_groups_and_runtime_metadata_keeps_source_slew(self):
        wid = self.adapter.save_waveform_document(signal())
        dialog = ExposureEditorDialog(self.adapter, exposure_for(wid))
        try:
            self.assertFalse(hasattr(dialog, 'project_waveform_channel'))
            self.assertFalse(hasattr(dialog, 'controller'))
            self.assertTrue(hasattr(dialog, 'sequence_tree'))
            self.assertTrue(hasattr(dialog, 'drive_info_button'))
            self.assertFalse(hasattr(dialog, 'diagnostics'))
            self.assertFalse(dialog.drive_info_button.isEnabled())
            first_step = dialog.sequence_tree.topLevelItem(0)
            self.assertGreater(first_step.childCount(), 0)
            dialog.sequence_tree.setCurrentItem(first_step.child(0))
            dialog._update_buttons()
            self.assertTrue(dialog.drive_info_button.isEnabled())
            self.assertEqual(dialog._sources[0]['waveform_id'], wid)
            _times, _values, metadata = self.adapter._waveform_source_trace(wid)
            self.assertEqual(metadata['slew_rate_per_s'], 2.5)
        finally:
            dialog.close(); dialog.deleteLater()

    def test_exposure_step_duration_edits_inline_and_new_step_copies_last_duration(self):
        wid = self.adapter.save_waveform_document(signal())
        dialog = ExposureEditorDialog(self.adapter, exposure_for(wid))
        try:
            first_step = dialog.sequence_tree.topLevelItem(0)
            duration = dialog.sequence_tree.itemWidget(first_step, 1)
            self.assertIsInstance(duration, QDoubleSpinBox)
            self.assertEqual(duration.suffix(), " s")
            duration.setValue(2.75)
            self.assertAlmostEqual(dialog._steps[0]["duration_s"], 2.75)

            with patch("fieldworkbench.exposure_library.QInputDialog.getDouble") as get_double:
                dialog._add_step()
                get_double.assert_not_called()

            self.assertEqual(len(dialog._steps), 2)
            self.assertAlmostEqual(dialog._steps[1]["duration_s"], 2.75)
            second_step = dialog.sequence_tree.topLevelItem(1)
            second_duration = dialog.sequence_tree.itemWidget(second_step, 1)
            self.assertIsInstance(second_duration, QDoubleSpinBox)
            self.assertAlmostEqual(second_duration.value(), 2.75)
            self.assertFalse(dialog.edit_button.isEnabled())
        finally:
            dialog.close(); dialog.deleteLater()

    def test_exposure_duplicate_step_clones_duration_drive_groups_and_overrides(self):
        wid = self.adapter.save_waveform_document(signal())
        exposure = exposure_for(wid)
        exposure['protocol']['sequence']['steps'][0]['duration_s'] = 0.5
        exposure['protocol']['sequence']['steps'][0]['drive_groups'][0]['load_overrides'] = {
            'coil_1': {'resistance_ohm': 12.5, 'inductance_h': 0.034}
        }
        dialog = ExposureEditorDialog(self.adapter, exposure)
        try:
            first_step = dialog.sequence_tree.topLevelItem(0)
            dialog.sequence_tree.setCurrentItem(first_step)
            dialog._update_buttons()
            self.assertTrue(dialog.duplicate_step_button.isEnabled())

            original_step_id = dialog._steps[0]['id']
            original_group_id = dialog._steps[0]['drive_groups'][0]['id']
            dialog._duplicate_step()

            self.assertEqual(len(dialog._steps), 2)
            original, clone = dialog._steps
            self.assertAlmostEqual(clone['duration_s'], 0.5)
            self.assertNotEqual(clone['id'], original_step_id)
            self.assertNotEqual(clone['drive_groups'][0]['id'], original_group_id)
            self.assertEqual(clone['drive_groups'][0]['source_id'], original['drive_groups'][0]['source_id'])
            self.assertEqual(clone['drive_groups'][0]['coil_ids'], original['drive_groups'][0]['coil_ids'])
            self.assertEqual(clone['drive_groups'][0]['connection'], original['drive_groups'][0]['connection'])
            self.assertEqual(clone['drive_groups'][0]['load_overrides'], original['drive_groups'][0]['load_overrides'])
            self.assertIsNot(clone['drive_groups'][0]['load_overrides'], original['drive_groups'][0]['load_overrides'])
            self.assertEqual(dialog.sequence_tree.currentItem().text(0), 'Step 2')

            clone['drive_groups'][0]['load_overrides']['coil_1']['resistance_ohm'] = 99.0
            self.assertAlmostEqual(
                original['drive_groups'][0]['load_overrides']['coil_1']['resistance_ohm'], 12.5
            )

            dialog.sequence_tree.setCurrentItem(dialog.sequence_tree.topLevelItem(1).child(0))
            dialog._update_buttons()
            self.assertTrue(dialog.duplicate_step_button.isEnabled())

            original_drive = clone['drive_groups'][0]
            original_drive_id = original_drive['id']
            dialog._duplicate_selected()
            self.assertEqual(len(clone['drive_groups']), 2)
            duplicated_drive = clone['drive_groups'][1]
            self.assertNotEqual(duplicated_drive['id'], original_drive_id)
            self.assertEqual(duplicated_drive['source_id'], original_drive['source_id'])
            self.assertEqual(duplicated_drive['coil_ids'], original_drive['coil_ids'])
            self.assertEqual(duplicated_drive['connection'], original_drive['connection'])
            self.assertEqual(duplicated_drive['load_overrides'], original_drive['load_overrides'])
            self.assertIsNot(duplicated_drive['load_overrides'], original_drive['load_overrides'])
            self.assertEqual(dialog.sequence_tree.currentItem().text(0), 'Drive 2')

            duplicated_drive['load_overrides']['coil_1']['inductance_h'] = 0.222
            self.assertAlmostEqual(original_drive['load_overrides']['coil_1']['inductance_h'], 0.034)
        finally:
            dialog.close(); dialog.deleteLater()

    def test_phase6_reference_lifecycle_survives_edit_rename_duplicate_reopen_and_delete(self):
        wid = self.adapter.save_waveform_document(signal(), name='Thomas-1')
        eid = self.adapter.save_exposure(exposure_for(wid))

        # Edit and rename in place: drive groups keep only the stable resource id.
        edited = signal()
        edited.signal.values[1] = 42
        self.adapter.save_waveform_document(edited, waveform_id=wid, name='Thomas revised')
        protocol = self.adapter.exposure(eid)['protocol']
        self.assertEqual(protocol['sources'][0]['waveform_id'], wid)
        self.assertEqual(self.adapter.list_exposures()[0]['waveform'], 'Thomas revised')
        self.assertEqual(self.adapter.waveform_document(wid).signal.values[1], 42)

        # Duplicates are independent resources; duplicating an exposure preserves
        # the intended source reference rather than copying/renaming the waveform.
        copied_waveform = self.adapter.duplicate_waveform(wid)
        copied_exposure = self.adapter.duplicate_exposure(eid)
        self.assertNotEqual(copied_waveform, wid)
        self.assertNotEqual(copied_exposure, eid)
        self.assertEqual(
            self.adapter.exposure(copied_exposure)['protocol']['sources'][0]['waveform_id'],
            wid,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phase6-roundtrip.magpy.json'
            self.adapter.save_file(path)
            reopened = StudioAdapter({'objects': []})
            reopened.load_file(path)
            self.assertEqual(reopened.waveform_document(wid).signal.values[1], 42)
            self.assertEqual(reopened.list_exposures()[0]['waveform'], 'Thomas revised')
            with self.assertRaises(StudioOperationError):
                reopened.delete_waveforms([wid])
            reopened.delete_exposures([eid, copied_exposure])
            reopened.delete_waveforms([wid])
            self.assertEqual([row['id'] for row in reopened.list_waveforms()], [copied_waveform])

    def test_phase6_exposure_cancel_and_waveform_rename_refresh_leave_saved_protocol_untouched(self):
        wid = self.adapter.save_waveform_document(signal(), name='Before')
        eid = self.adapter.save_exposure(exposure_for(wid))
        original = self.adapter.exposure(eid)
        dialog = ExposureEditorDialog(self.adapter, original)
        try:
            dialog._steps[0]['duration_s'] = 9.0
            self.adapter.rename_waveform(wid, 'After')
            dialog.refresh_waveform_resources()
            drive_item = dialog.sequence_tree.topLevelItem(0).child(0)
            self.assertEqual(drive_item.text(2), 'Source 1')
            source_row = dialog.source_table.topLevelItem(0)
            waveform_combo = dialog.source_table.itemWidget(source_row, 1)
            self.assertIn('After', waveform_combo.currentText())
            dialog.reject()
        finally:
            dialog.close(); dialog.deleteLater()
        # Draft duration was never committed, while the stable waveform reference
        # naturally resolves the new resource name.
        saved = self.adapter.exposure(eid)
        self.assertEqual(saved['protocol']['sequence']['steps'][0]['duration_s'], 1.0)
        self.assertEqual(saved['protocol']['sources'][0]['waveform_id'], wid)
        self.assertEqual(self.adapter.list_exposures()[0]['waveform'], 'After')

    def test_phase6_cross_window_open_request_protects_dirty_working_copy(self):
        first = self.adapter.save_waveform_document(signal(), name='First')
        second = self.adapter.save_waveform_document(signal('voltage', 'V'), name='Second')
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.refresh_project_resources(first)
            window.edit_selected_resource()
            window._begin_encoding_edit()
            window.edit_name.setText('Unsaved first')
            with patch(
                'fieldworkbench.waveform_editor.ui.main_window.QMessageBox.warning',
                return_value=QMessageBox.StandardButton.Cancel,
            ):
                self.assertFalse(window.request_open_resource(second))
            self.assertEqual(window._task_mode, 'edit')
            self.assertEqual(window.current_waveform_id, first)
            self.assertEqual(window.edit_name.text(), 'Unsaved first')

            with patch(
                'fieldworkbench.waveform_editor.ui.main_window.QMessageBox.warning',
                return_value=QMessageBox.StandardButton.Discard,
            ):
                self.assertTrue(window.request_open_resource(second))
            self.assertIsNone(window._task_mode)
            self.assertEqual(window.current_waveform_id, second)
        finally:
            window.close(); window.deleteLater()

    def test_phase6_prepare_project_close_can_commit_private_waveform_draft(self):
        wid = self.adapter.save_waveform_document(signal(), name='Before close')
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.refresh_project_resources(wid)
            window.edit_selected_resource()
            window._begin_encoding_edit()
            window.edit_name.setText('Saved on close')
            window._confirm_encoding_setup()
            with patch(
                'fieldworkbench.waveform_editor.ui.main_window.QMessageBox.warning',
                return_value=QMessageBox.StandardButton.Save,
            ):
                self.assertTrue(window.prepare_project_close())
            self.assertIsNone(window._task_mode)
            self.assertEqual(self.adapter.waveform_resource(wid)['name'], 'Saved on close')
        finally:
            window.close(); window.deleteLater()

    def test_phase6_background_resource_refresh_preserves_waveform_working_copy(self):
        wid = self.adapter.save_waveform_document(signal(), name='Working')
        window = WaveformEditorWindow(ProjectContext(self.adapter))
        try:
            window.refresh_project_resources(wid)
            window.edit_selected_resource()
            window._begin_encoding_edit()
            window.edit_name.setText('Still unsaved')
            self.adapter.save_waveform_document(signal('voltage', 'V'), name='Sibling')
            window.refresh_project_resources()
            self.assertEqual(window._task_mode, 'edit')
            self.assertEqual(window.current_waveform_id, wid)
            self.assertEqual(window.edit_name.text(), 'Still unsaved')
        finally:
            # Avoid an interactive dirty-close prompt in the test cleanup.
            window._show_browse_mode()
            window.close(); window.deleteLater()

    def test_phase6_project_close_waits_for_open_exposure_draft(self):
        window = ExposureEditorWindow(ProjectContext(self.adapter))
        draft = ExposureEditorDialog(self.adapter, parent=window.library)
        try:
            window.show()
            draft.show()
            self.app.processEvents()
            with patch('fieldworkbench.exposure_library.QMessageBox.information') as info:
                self.assertFalse(window.prepare_project_close())
                info.assert_called_once()
            draft.reject()
            self.app.processEvents()
            self.assertTrue(window.prepare_project_close())
        finally:
            draft.close(); draft.deleteLater()
            window.close(); window.deleteLater()
