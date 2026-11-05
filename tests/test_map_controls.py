"""Controls come from the selected result, never whichever setup is current."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QWidget
from fieldworkbench.map_controls import (AUX_KEY, MapControlsError, attach_controls,
    capture_controls, legacy_controls, recorded_controls, restore_controls)
from fieldworkbench.studio_adapter import StudioAdapter
from fieldworkbench.dialogs import FieldMapDialog, FieldVolumeMapDialog
from fieldworkbench.main_window import UnifiedFieldWorkspaceDialog
from workflow_fixture import compact_workflow_scene


class MapControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.adapter = StudioAdapter(compact_workflow_scene())
        self.pages = []

    def tearDown(self):
        for page in self.pages:
            for plot in page.findChildren(QWidget):
                if hasattr(plot, 'cleanup'):
                    try: plot.cleanup()
                    except RuntimeError: pass
            page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def page(self, volume=False):
        page = (FieldVolumeMapDialog if volume else FieldMapDialog)(self.adapter)
        self.pages.append(page)
        return page

    def test_open_tab_frozen_controls_survive_later_edits_and_do_not_change_scene(self):
        page = self.page()
        page.span.setValue(222)
        page.offset.setValue(12)
        page.colour_minimum_enabled.setChecked(True)
        page.colour_minimum.setValue(2)
        page.colour_maximum_enabled.setChecked(True)
        page.colour_maximum.setValue(37)
        state = capture_controls(page, mode='static', view='2d')
        tab = QWidget(); attach_controls(tab, state)
        state['widgets']['span']['value'] = 999
        page.span.setValue(444); page.offset.setValue(-4)
        page.colour_maximum.setValue(80)
        before = self.adapter.document()
        with mock.patch.object(page, 'calculate') as calculate:
            result = restore_controls(page, recorded_controls(tab), mode='static', view='2d')
        self.assertEqual(page.span.value(), 222)
        self.assertEqual(page.offset.value(), 12)
        self.assertEqual(page.colour_minimum.value(), 2)
        self.assertEqual(page.colour_maximum.value(), 37)
        self.assertEqual(self.adapter.document(), before)
        self.assertFalse(result['skipped']); calculate.assert_not_called()
        tab.deleteLater()

    def test_scene_signature_is_hashed_and_legacy_signature_still_compares(self):
        page = self.page()
        state = capture_controls(page, mode='static', view='2d')
        self.assertRegex(state['source_signature'], r'^sha256:[0-9a-f]{64}$')
        self.assertNotIn(page.adapter.signature(), state['source_signature'])
        report = restore_controls(page, state, mode='static', view='2d')
        self.assertFalse(any('source scene differs' in item for item in report['skipped']))
        state['source_signature'] = page.adapter.signature()
        report = restore_controls(page, state, mode='static', view='2d')
        self.assertFalse(any('source scene differs' in item for item in report['skipped']))
        state['source_signature'] = 'sha256:' + '0' * 64
        report = restore_controls(page, state, mode='static', view='2d')
        self.assertTrue(any('source scene differs' in item for item in report['skipped']))

    def test_invalid_log_limits_rejected_before_any_render_work(self):
        from fieldworkbench import dialogs
        for volume in (False, True):
            page = self.page(volume)
            for lower, upper in ((0, 100), (-1, 100), (None, 0)):
                for render_kind in (('static' if volume else '2d_static'), 'waveform'):
                    with self.subTest(volume=volume, lower=lower, upper=upper, kind=render_kind):
                        count = page.tabs.count()
                        with mock.patch.object(dialogs.QMessageBox, 'warning') as warning, \
                             mock.patch.object(page.adapter, 'document') as document:
                            page._start_render_job(
                                render_kind=render_kind,
                                request={'map_settings': {'logarithmic': True,
                                    'colour_minimum': lower, 'colour_maximum': upper}},
                                job_label='Invalid', tooltip='Invalid')
                        document.assert_not_called()
                        self.assertEqual(page.tabs.count(), count)
                        self.assertFalse(page._render_jobs)
                        warning.assert_called_once()
                        message = warning.call_args.args[2]
                        self.assertIn('above zero', message)
                        self.assertIn('Auto', message)
                        self.assertIn('No calculation was started', message)

    def test_colour_validation_accepts_auto_positive_log_and_signed_linear(self):
        for volume in (False, True):
            page = self.page(volume)
            for logarithmic, lower, upper in ((True, None, None), (True, .1, 100),
                                              (True, None, 100), (False, 0, 100),
                                              (False, -100, 100)):
                with self.subTest(volume=volume, logarithmic=logarithmic, lower=lower):
                    self.assertTrue(page._validate_render_colour_settings({
                        'logarithmic': logarithmic, 'colour_minimum': lower,
                        'colour_maximum': upper}))

    def test_reversed_or_nonfinite_limits_rejected_early(self):
        from fieldworkbench import dialogs
        for volume in (False, True):
            page = self.page(volume)
            for lower, upper in ((100, 10), (10, 10), (float('nan'), 100),
                                 (0, float('inf'))):
                with mock.patch.object(dialogs.QMessageBox, 'warning') as warning:
                    self.assertFalse(page._validate_render_colour_settings({
                        'logarithmic': False, 'colour_minimum': lower,
                        'colour_maximum': upper}))
                warning.assert_called_once()

    def test_restored_invalid_log_combination_is_preserved_but_cannot_calculate(self):
        from fieldworkbench import dialogs
        for volume in (False, True):
            page = self.page(volume)
            page.logarithmic.setChecked(True)
            page.colour_minimum_enabled.setChecked(True)
            page.colour_minimum.setValue(0)
            state = capture_controls(page, mode='static', view='3d' if volume else '2d')
            page.logarithmic.setChecked(False)
            restore_controls(page, state, mode='static', view='3d' if volume else '2d')
            self.assertTrue(page.logarithmic.isChecked())
            self.assertEqual(page.colour_minimum.value(), 0)
            settings = page._waveform_map_settings() if volume else page._static_render_map_settings()
            with mock.patch.object(dialogs.QMessageBox, 'warning'):
                self.assertFalse(page._validate_render_colour_settings(settings))

    def test_legacy_linear_colour_metadata_clears_previous_log_scale(self):
        traces = [
            {'type': 'heatmap', 'x': [0, 1], 'y': [0, 1],
             'colorbar': {'title': {'text': '|B| (µT)'}}, 'zmin': 0, 'zmax': 100},
            {'type': 'volume', 'x': [0, 1], 'y': [0, 1], 'z': [0, 1],
             'meta': {'unit': 'µT', 'component': '|B|'},
             'colorbar': {'title': {'text': '|B| (µT)<br>display-clipped'}},
             'cmin': 0, 'cmax': 100},
        ]
        for trace in traces:
            volume = trace['type'] == 'volume'
            page = self.page(volume)
            page.logarithmic.setChecked(True)
            state = legacy_controls({'data': [trace]}, {}, mode='static',
                                    view='3d' if volume else '2d')
            restore_controls(page, state, mode='static', view='3d' if volume else '2d')
            self.assertFalse(page.logarithmic.isChecked())
            self.assertTrue(page.colour_minimum_enabled.isChecked())
            self.assertEqual(page.colour_minimum.value(), 0)

    def test_volume_scene_selection_and_opacity_are_local(self):
        page = self.page(True)
        page.view_type.setCurrentIndex(page.view_type.findData('volume'))
        page.volume_opacity_lower.setValue(23)
        ids = page.selected_active_coil_ids()
        page._playback_selected_coil_ids = {ids[0]}
        choices = page.selected_scene_object_ids()
        page._scene_selected_object_ids = set(choices[:1])
        state = capture_controls(page, mode='static', view='3d')
        page._playback_selected_coil_ids = set(ids)
        page.volume_opacity_lower.setValue(50)
        before = self.adapter.document()
        report = restore_controls(page, state, mode='static', view='3d')
        self.assertEqual(page.volume_opacity_lower.value(), 23)
        self.assertEqual(page.selected_active_coil_ids(), [ids[0]])
        self.assertEqual(self.adapter.document(), before)
        self.assertFalse(report['skipped'])

    def test_saved_reopened_aux_roundtrip_and_capture_types(self):
        page = self.page()
        state = capture_controls(page, mode='static', view='2d')
        state = json.loads(json.dumps(state))
        map_id = self.adapter.save_map_resource(name='Captured',kind='static',view='2d',
            renderer='plotly',content={'data':[],'layout':{}},aux={AUX_KEY:state,'other':'keep'})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'project.magpy.json'; self.adapter.save_file(path)
            reopened = StudioAdapter(json.loads(path.read_text()))
            self.assertEqual(reopened.saved_map(map_id)['aux'][AUX_KEY],state)
            self.assertEqual(reopened.saved_map(map_id)['aux']['other'],'keep')

    def test_old_volume_recovers_only_explicit_settings(self):
        page=self.page(True); page.volume_opacity_lower.setValue(44)
        page.field.setCurrentText("H")
        coords=[-10+i*20/10 for i in range(11)]
        xyz=[(x,y,z) for z in coords for y in coords for x in coords]
        payload={'data':[{'type':'volume','x':[p[0] for p in xyz],
            'y':[p[1] for p in xyz],'z':[p[2] for p in xyz],
            'meta':{'component':'|B|','unit':'µT'},'cmin':0,'cmax':60}]}
        state=legacy_controls(payload,{},mode='static',view='3d')
        result=restore_controls(page,state,mode='static',view='3d')
        self.assertTrue(result['partial']);self.assertEqual(page.span.value(),20)
        self.assertEqual(page.resolution.value(),11)
        self.assertEqual(page.view_type.currentData(),'volume')
        self.assertEqual(page.colour_maximum.value(),60)
        self.assertEqual(page.field.currentText(),'B')
        self.assertEqual(page.volume_opacity_lower.value(),44)

    def test_legacy_without_metadata_leaves_controls_unchanged(self):
        page=self.page(); before=capture_controls(page,mode='static',view='2d')
        result=restore_controls(page,legacy_controls({'data':[]},{},mode='static',view='2d'),mode='static',view='2d')
        self.assertEqual(result['restored'],0)
        self.assertEqual(capture_controls(page,mode='static',view='2d'),before)

    def test_partial_invalid_and_unknown_version_do_not_clamp_or_write_scene(self):
        page=self.page();state=capture_controls(page,mode='static',view='2d')
        state['widgets']={'span':{'kind':'number','value':float('nan')},
            'offset':{'kind':'number','value':7},'field':{'kind':'choice','data':'bogus'},
            'not_a_control':{'kind':'checked','value':True}}
        state['selections']['active_coil_ids']=['deleted-coil']
        before=self.adapter.document();old_span=page.span.value();old_coils=page.selected_active_coil_ids()
        report=restore_controls(page,state,mode='static',view='2d')
        self.assertTrue(report['partial']);self.assertGreaterEqual(len(report['skipped']),4)
        self.assertEqual(page.offset.value(),7);self.assertEqual(page.span.value(),old_span)
        self.assertEqual(page.selected_active_coil_ids(),old_coils)
        self.assertEqual(self.adapter.document(),before)
        state['version']=2;old=page.offset.value()
        with self.assertRaises(MapControlsError):restore_controls(page,state,mode='static',view='2d')
        self.assertEqual(page.offset.value(),old)

    def test_workspace_button_uses_selected_tab_record_and_survives_save_reload(self):
        maps=UnifiedFieldWorkspaceDialog(mode='static',scene_name='Demo',adapter=self.adapter,brain_object_id=None)
        self.pages.append(maps)
        page=maps._pages['static:2d']; page.span.setValue(250)
        state=capture_controls(page,mode='static',view='2d',workspace=maps)
        map_id=self.adapter.save_map_resource(name='Old setup',kind='static',view='2d',
            renderer='plotly',content={'data':[],'layout':{}},aux={AUX_KEY:state,'other':'keep'})
        self.assertTrue(maps.load_saved_map(map_id));tab=maps.shared_tabs.currentWidget()
        self.assertEqual(recorded_controls(tab),state)
        self.assertEqual(tab._saved_map_aux['other'],'keep')
        page.span.setValue(800)
        self.assertFalse(maps.workspace_restore_controls_button.isHidden())
        maps.workspace_restore_controls_button.click()
        self.assertEqual(page.span.value(),250)
        self.assertIs(maps.shared_tabs.currentWidget(),tab)
        maps.shared_tabs.setCurrentIndex(0)
        self.assertTrue(maps.workspace_restore_controls_button.isHidden())

    def saved_map_workspace(self):
        maps = UnifiedFieldWorkspaceDialog(mode='static', scene_name='Demo',
                                          adapter=self.adapter, brain_object_id=None)
        self.pages.append(maps)
        map_id = self.adapter.save_map_resource(name='Original', kind='static', view='2d',
            renderer='plotly', content={'data': [], 'layout': {}})
        self.assertTrue(maps.load_saved_map(map_id))
        return maps

    def test_save_updates_only_saved_tab_to_actual_unique_resource_name(self):
        from fieldworkbench import main_window
        maps = self.saved_map_workspace()
        first = maps.shared_tabs.currentWidget()
        self.assertTrue(maps.load_saved_map(first._saved_map_resource_id))
        second = maps.shared_tabs.currentWidget()
        del second._saved_map_resource_id
        maps.shared_tabs.setTabText(maps.shared_tabs.indexOf(second), '2D\\Static - 2')
        first_index = maps.shared_tabs.indexOf(first)
        second_index = maps.shared_tabs.indexOf(second)
        original_title = maps.shared_tabs.tabText(first_index)
        with mock.patch.object(main_window.QInputDialog, 'getText', return_value=('  Renamed  ', True)):
            maps._save_current_map()
        self.assertEqual(maps.shared_tabs.tabText(second_index), '2D\\Renamed')
        self.assertEqual(maps.shared_tabs.tabText(first_index), original_title)
        renamed_id = second._saved_map_resource_id
        # Existing Save creates a uniquely named copy, rather than overwriting.
        with mock.patch.object(main_window.QInputDialog, 'getText', return_value=('Renamed', True)) as prompt:
            maps._save_current_map()
        self.assertEqual(prompt.call_args.kwargs['text'], 'Renamed')
        copy_id = second._saved_map_resource_id
        self.assertNotEqual(copy_id, renamed_id)
        actual_name = self.adapter.saved_map(copy_id)['name']
        self.assertNotEqual(actual_name, 'Renamed')
        self.assertEqual(maps.shared_tabs.tabText(second_index), '2D\\' + actual_name)
        self.assertEqual(maps.shared_tabs.tabText(first_index), original_title)
        self.assertEqual(self.adapter.saved_map(renamed_id)['name'], 'Renamed')

    def test_resource_rename_updates_linked_tabs_but_not_other_maps(self):
        maps = self.saved_map_workspace()
        original = maps.shared_tabs.currentWidget()
        self.assertTrue(maps.load_saved_map(original._saved_map_resource_id))
        duplicate = maps.shared_tabs.currentWidget()
        other_id = self.adapter.save_map_resource(name='Other', kind='static', view='2d',
            renderer='plotly', content={'data': [], 'layout': {}})
        self.assertTrue(maps.load_saved_map(other_id))
        other = maps.shared_tabs.currentWidget()
        self.adapter.rename_saved_map(original._saved_map_resource_id, 'Updated name')
        maps._project_resources_changed()
        for tab in (original, duplicate):
            self.assertEqual(maps.shared_tabs.tabText(maps.shared_tabs.indexOf(tab)), '2D\\Updated name')
        self.assertEqual(maps.shared_tabs.tabText(maps.shared_tabs.indexOf(other)), '2D\\Other')
        self.assertIs(maps.shared_tabs.currentWidget(), other)

    def test_cancel_blank_and_failed_save_leave_title_and_resource_link_unchanged(self):
        from fieldworkbench import main_window
        maps = self.saved_map_workspace()
        tab = maps.shared_tabs.currentWidget()
        index = maps.shared_tabs.indexOf(tab)
        original = maps.shared_tabs.tabText(index)
        map_id = tab._saved_map_resource_id
        count = len(self.adapter.list_saved_maps())
        for name, accepted, fail in (('Cancelled', False, False), (' ', True, False),
                                      ('Failure', True, True)):
            with self.subTest(name=name), \
                 mock.patch.object(main_window.QInputDialog, 'getText', return_value=(name, accepted)), \
                 mock.patch.object(main_window.QMessageBox, 'warning'), \
                 mock.patch.object(self.adapter, 'save_map_resource',
                                   side_effect=RuntimeError('simulated failure')) as save:
                maps._save_current_map()
                self.assertEqual(save.call_count, int(fail))
            self.assertEqual(maps.shared_tabs.tabText(index), original)
            self.assertEqual(tab._saved_map_resource_id, map_id)
            self.assertEqual(len(self.adapter.list_saved_maps()), count)

    def test_queued_job_keeps_submission_controls_on_completion(self):
        from fieldworkbench import dialogs
        page=self.page(); page.span.setValue(222)
        with mock.patch.object(dialogs.QTimer,'singleShot'):
            page._start_render_job(render_kind='2d_static',
                request={'map_settings':page._static_render_map_settings(),
                         'active_coil_ids':page.selected_active_coil_ids()},
                job_label='Frozen',tooltip='Frozen setup')
        job_id=page._render_job_serial;job=page._render_jobs[job_id]
        page.span.setValue(444)
        self.assertEqual(job['map_controls']['widgets']['span']['value'],222)
        with mock.patch.object(dialogs,'_load_field_render_result',
                return_value={'figure':{'data':[],'layout':{}}}), \
             mock.patch.object(page,'_render_tab_changed'), \
             mock.patch.object(page,'_connect_result_camera_viewer'):
            page._render_job_completed(job_id,{})
        tab=page.tabs.currentWidget()
        self.assertEqual(recorded_controls(tab)['widgets']['span']['value'],222)
        # No numerical worker was started; dispose the dormant thread fixture.
        job['thread'].deleteLater();page._render_jobs.clear()

    def test_scripted_source_controls_restore_without_changing_definitions(self):
        maps=UnifiedFieldWorkspaceDialog(mode='playback',scene_name='Demo',adapter=self.adapter,brain_object_id=None)
        self.pages.append(maps);page=maps._pages['playback:2d']
        maps.scripted_amplitude.setValue(8)
        maps.scripted_frequency.setValue(12)
        state=capture_controls(page,mode='playback',view='2d',workspace=maps)
        maps.scripted_amplitude.setValue(20);maps.scripted_frequency.setValue(50)
        before=self.adapter.document()
        report=restore_controls(page,state,mode='playback',view='2d',workspace=maps)
        self.assertEqual(maps.scripted_amplitude.value(),8)
        self.assertEqual(maps.scripted_frequency.value(),12)
        self.assertEqual(self.adapter.document(),before)
        self.assertFalse(report['skipped'])

    def test_legacy_2d_grid_and_units_but_not_plane_are_recovered(self):
        page=self.page();page.plane.setCurrentText('xz')
        coordinates=[-10+i*2 for i in range(11)]
        state=legacy_controls({'data':[{'type':'heatmap','x':coordinates,'y':coordinates,
            'zmin':0,'zmax':60,'colorbar':{'title':{'text':'|B| (µT)<br>≥60 saturated'}}}]},
            {},mode='static',view='2d')
        result=restore_controls(page,state,mode='static',view='2d')
        self.assertTrue(result['partial']);self.assertEqual(page.resolution.value(),11)
        self.assertEqual(page.colour_maximum.value(),60)
        self.assertEqual(page.plane.currentText(),'xz')

    def test_inconsistent_bounds_and_huge_integer_are_skipped(self):
        page=self.page();state=capture_controls(page,mode='static',view='2d')
        old=(page.colour_minimum.value(),page.colour_maximum.value())
        state['widgets']['colour_minimum']['value']=10
        state['widgets']['colour_maximum']['value']=5
        state['widgets']['colour_minimum_enabled']['value']=True
        state['widgets']['colour_maximum_enabled']['value']=True
        state['widgets']['resolution']['value']=10**1000
        report=restore_controls(page,state,mode='static',view='2d')
        self.assertEqual((page.colour_minimum.value(),page.colour_maximum.value()),old)
        self.assertGreaterEqual(len(report['skipped']),2)

    def test_brain_display_controls_keep_brain_and_scene_unchanged(self):
        from fieldworkbench.brain_view import BrainViewDialog
        brain_id=self.adapter.add_template('sri24_brain')
        page=BrainViewDialog(self.adapter,brain_id)
        self.pages.append(page)
        page.volume_opacity_lower.setValue(25)
        state=capture_controls(page,mode='static',view='brain')
        page.volume_opacity_lower.setValue(60)
        before=self.adapter.document()
        report=restore_controls(page,state,mode='static',view='brain')
        self.assertEqual(page.volume_opacity_lower.value(),25)
        self.assertEqual(self.adapter.document(),before)
        self.assertFalse(report['skipped'])

    def test_missing_saved_exposure_keeps_source_group_unchanged(self):
        maps=UnifiedFieldWorkspaceDialog(mode='playback',scene_name='Demo',adapter=self.adapter,brain_object_id=None)
        self.pages.append(maps);page=maps._pages['playback:2d']
        state=capture_controls(page,mode='playback',view='2d',workspace=maps)
        state['scripted_widgets']['scripted_source_type']={'kind':'choice','data':'exposure'}
        state['scripted_widgets']['scripted_exposure']={'kind':'choice','data':'deleted'}
        state['scripted_widgets']['scripted_exposure_duration']={'kind':'number','value':5}
        report=restore_controls(page,state,mode='playback',view='2d',workspace=maps)
        self.assertEqual(maps.scripted_source_type.currentData(),'builtin')
        self.assertTrue(any('saved exposure missing' in x for x in report['skipped']))

    def test_saved_exposure_duration_restores_across_different_spin_maxima(self):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QComboBox,QDoubleSpinBox
        page=self.page()
        source=QComboBox();source.addItem('Saved exposure','exposure')
        exposure=QComboBox();exposure.addItem('Long','long');exposure.addItem('Short','short')
        duration=QDoubleSpinBox();duration.setRange(.01,100);duration.setValue(70)
        workspace=SimpleNamespace(scripted_source_type=source,scripted_exposure=exposure,
            scripted_exposure_duration=duration,snapshot_adapter=SimpleNamespace(
                list_exposures=lambda:[{'id':'long','duration_s':100},{'id':'short','duration_s':10}]))
        state=capture_controls(page,mode='playback',view='2d',workspace=workspace)
        exposure.setCurrentIndex(1);duration.setMaximum(10);duration.setValue(8)
        report=restore_controls(page,state,mode='playback',view='2d',workspace=workspace)
        self.assertEqual(exposure.currentData(),'long');self.assertEqual(duration.value(),70)
        self.assertFalse(report['skipped'])
        for widget in (source,exposure,duration):widget.deleteLater()

if __name__=='__main__':unittest.main()
