"""Measurement preparation, physical cache identity and multi-target browsing."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
import copy
import time
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from fieldworkbench.studio_adapter import StudioAdapter
from fieldworkbench.waveform_io import WaveformDocument, WaveformSeries
from fieldworkbench.exposure_analysis import (exposure_analysis_targets, _measurement_target_samples,
    prepare_exposure_analysis_payload)
from fieldworkbench.exposure_workspace import ExposureAnalysisWorkspace
from fieldworkbench.region_analysis import analyse_regions_exposure_summary, _region_spatial_rms_values
from fieldworkbench.waveform import field_bases_for_drives
from workflow_fixture import compact_workflow_scene


def scene():
    adapter = StudioAdapter(compact_workflow_scene())
    boxes = []
    for index in range(2):
        box = adapter.add_template('guide_box')
        adapter.apply_operations([
            {'method': 'set_param', 'params': {'object_id': box, 'name': 'dimension', 'value': [.02, .02, .02]}},
            {'method': 'set_geometry_style', 'params': {'object_id': box, 'role': 'measurement'}},
            {'method': 'set_measurement_settings', 'params': {'object_id': box, 'quality': 'preview'}}])
        boxes.append(box)
    document = WaveformDocument([WaveformSeries('Drive', [0., .01, .02, .03, .04],
                                               [0., 1., 0., -1., 0.], quantity='current', unit='A')])
    wave = adapter.save_waveform_document(document, name='Drive')
    exposure_id = adapter.save_exposure({'name': 'Example', 'protocol': {
        'duration_s': .08, 'sources': [{'id': 'source', 'waveform_id': wave}],
        'sequence': {'loop': True, 'steps': [{'duration_s': .04, 'drive_groups': [
            {'source_id': 'source', 'coil_ids': ['coil_a'], 'connection': 'series'}]}]}}})
    return adapter, adapter.exposure(exposure_id), boxes


class MeasurementPreparationTests(unittest.TestCase):
    def test_generated_and_custom_points_match_direct_field_rms_and_reopen(self):
        adapter, exposure, boxes = scene()
        for mode in ('generated', 'user_defined'):
            box = boxes[0]
            if mode == 'user_defined':
                adapter.apply_operations([{'method': 'set_measurement_settings', 'params': {
                    'object_id': box, 'sampling_mode': mode, 'defined_points_mm': [[0, 0, 0], [2, 3, 1]]}}])
            payload = prepare_exposure_analysis_payload(adapter, exposure, target_id=box)
            self.assertEqual(set(payload['regions']), {box})
            context = payload['exposure_analysis_context']
            self.assertEqual(context['sampling_mode'], mode)
            self.assertEqual(context['target_kind'], 'measurement_volume')
            hierarchy, samples = _measurement_target_samples(adapter, box)
            fixed, bases = field_bases_for_drives(adapter, coil_ids=['coil_a'], zero_coil_ids=['coil_a'],
                                                points_m=samples.points_m, field='B')
            currents = np.array([[0.], [.3], [-.7], [1.]])
            region = payload['regions'][box]
            actual = _region_spatial_rms_values(region['constant'], region['linear'], region['quadratic'], currents)
            direct = np.sqrt(np.mean(np.sum((fixed[None, :, :] + currents[:, 0, None, None]*bases[0])**2, axis=2), axis=1))*1e6
            np.testing.assert_allclose(actual, direct, rtol=1e-11)
            cache = {('summary', '__all_regions__'): {'regions': analyse_regions_exposure_summary(payload)}}
            analysis_id = adapter.save_exposure_analysis_resource(payload, analysis_cache=cache)
            reopened = StudioAdapter(adapter.document()).exposure_analysis(analysis_id)
            self.assertEqual(reopened['status'], 'current')
            self.assertEqual(reopened['atlas'], '')
            self.assertEqual(reopened['analysis_cache'][('summary', '__all_regions__')]['regions'][box], cache[('summary', '__all_regions__')]['regions'][box])
            rows = adapter.list_exposure_analyses()
            self.assertFalse(any(row['target_id'] == boxes[1] for row in rows))
            self.assertTrue(any(row['target_id'] == box and row['status'] == 'current' for row in rows))
            self.assertAlmostEqual(context['physical_duration_s'], .08)
            self.assertNotIn('statistics_scope', context)
            self.assertNotIn('statistics_duration_s', context)
            self.assertNotIn('basis_source', context)

    def test_sampling_geometry_placement_invalidate_only_relevant_target(self):
        adapter, exposure, boxes = scene()
        signature = lambda box: adapter.exposure_analysis_dependency_signature(exposure['id'], box)
        baseline = [signature(box) for box in boxes]
        mesh = adapter._mesh(boxes[0])
        mesh['label'] = 'Renamed'
        mesh['measurement']['target_uT'] = 500.
        self.assertEqual(signature(boxes[0]), baseline[0])
        mesh['measurement']['quality'] = 'fine'
        self.assertNotEqual(signature(boxes[0]), baseline[0])
        self.assertEqual(signature(boxes[1]), baseline[1])
        mesh['measurement']['quality'] = 'preview'
        mesh['mesh']['dimension'][0] *= 2
        self.assertNotEqual(signature(boxes[0]), baseline[0])
        mesh['mesh']['dimension'][0] /= 2
        mesh['position'][0] += .01
        self.assertNotEqual(signature(boxes[0]), baseline[0])

    def test_catalogue_excludes_visual_objects_and_includes_brains_once(self):
        adapter, exposure, boxes = scene()
        adapter.add_template('guide_sphere')
        # Catalogue identification does not require atlas loading or point sampling.
        brain = copy.deepcopy(adapter._mesh(boxes[0]))
        brain.update(id='brain_test', label='Brain', builtin_template='sri24_brain')
        adapter.geometry.append(brain)
        with patch.object(adapter, 'measurement_sample_points', side_effect=AssertionError('No sampling on browse')):
            targets = exposure_analysis_targets(adapter)
        self.assertEqual({t['id'] for t in targets}, {*boxes, 'brain_test'})
        self.assertEqual(sum(t['id'] == 'brain_test' for t in targets), 1)


class WorkspaceTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_object_chooser_stays_flat_and_shows_target_details_without_preparing(self):
        from fieldworkbench.brain_analysis import BrainSampleSet, lpba40_region_hierarchy
        adapter, exposure, boxes = scene()
        hierarchy = lpba40_region_hierarchy()
        labels = np.array(hierarchy.label_ids[:2], dtype=np.int32)
        samples = BrainSampleSet(np.array([[0., 0., 0.], [.001, 0., 0.]]), labels,
            np.ones(2), np.arange(2), {int(k): 1 for k in labels}, {int(k): 1 for k in labels}, 1.)
        for index in range(2):
            brain = copy.deepcopy(adapter._mesh(boxes[0]))
            brain.update(id=f'brain_{index}', label=f'Brain {index}', builtin_template='sri24_brain')
            adapter.geometry.append(brain)
            with patch('fieldworkbench.exposure_analysis._brain_target_samples', return_value=(hierarchy, samples)):
                payload = prepare_exposure_analysis_payload(adapter, exposure, target_id=brain['id'])
            adapter.save_exposure_analysis_resource(payload, analysis_cache={})
        workspace = ExposureAnalysisWorkspace(adapter, exposure)
        try:
            self.assertIsNone(workspace.parentWidget())
            self.assertEqual(workspace.windowType(), Qt.WindowType.Window)
            self.assertFalse(workspace.isModal())
            self.assertEqual(len(workspace._items), 4)
            self.assertFalse(hasattr(workspace, 'preview_scope'))
            self.assertFalse(hasattr(workspace, 'preview_duration'))
            self.assertIs(workspace.stack.currentWidget(), workspace.chooser_page)
            with patch('fieldworkbench.exposure_analysis.prepare_exposure_analysis_payload') as prepare:
                workspace.target_tree.setCurrentItem(workspace._items['brain_0'])
                self.assertIsNone(workspace._active_view)
                self.assertEqual(workspace._items['brain_0'].childCount(), 0)
                self.assertIn('Atlas: LPBA40', workspace.target_details.text())
                self.assertIn('Atlas regions:', workspace.target_details.text())
                self.assertEqual(workspace.saved_analysis_tree.topLevelItemCount(), 1)
                self.assertEqual(workspace.saved_analysis_tree.topLevelItem(0).text(1), 'Current')
                workspace.target_tree.setCurrentItem(workspace._items[boxes[0]])
                self.assertIsNone(workspace._active_view)
                self.assertEqual(workspace._items[boxes[0]].childCount(), 0)
                self.assertIn('Type: Measurement volume', workspace.target_details.text())
                self.assertIn('Sample points:', workspace.target_details.text())
                self.assertEqual(workspace.saved_analysis_tree.topLevelItemCount(), 0)
                prepare.assert_not_called()
        finally:
            workspace.close()
            workspace.deleteLater()
            self.app.processEvents()

    def test_saved_analyses_open_explicitly_without_result_footer_navigation(self):
        adapter, exposure, boxes = scene()
        analysis_ids = {}
        for box in boxes:
            payload = prepare_exposure_analysis_payload(adapter, exposure, target_id=box)
            analysis_ids[box] = adapter.save_exposure_analysis_resource(payload, analysis_cache={})
        with patch('fieldworkbench.exposure_analysis.field_bases_for_drives') as fields:
            workspace = ExposureAnalysisWorkspace(adapter, exposure)
            try:
                self.assertIsNone(workspace._active_view)
                self.assertIs(workspace.stack.currentWidget(), workspace.chooser_page)
                self.assertFalse(hasattr(workspace, 'objects_button'))
                workspace.target_tree.setCurrentItem(workspace._items[boxes[0]])
                self.assertEqual(workspace.saved_analysis_tree.topLevelItemCount(), 1)
                self.assertEqual(workspace._selected_saved_analysis_id(), analysis_ids[boxes[0]])
                workspace.open_analysis_button.click()
                first = workspace._active_view
                self.assertIsNotNone(first)
                self.assertTrue(first.region_tree.isHidden())
                self.assertIs(workspace.stack.currentWidget(), first)
                self.assertFalse(hasattr(first, 'footer_back_button'))
                page = first.open_exposure_summary()
                end = time.monotonic() + 10
                while page._analysis_busy and time.monotonic() < end:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertFalse(page._analysis_busy)
                self.assertTrue(first.analysis_cache)
                self.assertFalse(hasattr(page, 'recalculate_button'))
                fields.assert_not_called()
            finally:
                workspace.close()
                workspace.deleteLater()
                self.app.processEvents()

    def test_create_new_analysis_confirms_when_current_cache_exists(self):
        adapter, exposure, boxes = scene()
        payload = prepare_exposure_analysis_payload(adapter, exposure, target_id=boxes[0])
        adapter.save_exposure_analysis_resource(payload, analysis_cache={})
        workspace = ExposureAnalysisWorkspace(adapter, exposure)
        try:
            workspace.target_tree.setCurrentItem(workspace._items[boxes[0]])
            self.assertEqual(workspace.saved_analysis_tree.topLevelItemCount(), 1)
            self.assertEqual(workspace.saved_analysis_tree.topLevelItem(0).text(1), 'Current')

            with patch('fieldworkbench.exposure_workspace.QMessageBox.question',
                       return_value=QMessageBox.StandardButton.No) as question, \
                    patch.object(workspace, '_start_preparation') as start:
                workspace.create_button.click()
                question.assert_called_once()
                self.assertIn('A cached analysis already exists', question.call_args.args[2])
                start.assert_not_called()

            with patch('fieldworkbench.exposure_workspace.QMessageBox.question',
                       return_value=QMessageBox.StandardButton.Yes) as question, \
                    patch.object(workspace, '_start_preparation') as start:
                workspace.create_button.click()
                question.assert_called_once()
                start.assert_called_once_with(boxes[0])
            self.assertIsNone(workspace._active_view)
        finally:
            workspace.close()
            workspace.deleteLater()
            self.app.processEvents()

