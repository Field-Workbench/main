"""Snapshot freshness follows resolved model inputs, not result management."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fieldworkbench.studio_adapter import StudioAdapter


class SnapshotFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.adapter = StudioAdapter({"objects": [{"id": "coil", "type": "current.Circle",
            "params": {"diameter": .1, "current": 1.0}}]})
        self.volume = self.adapter.add_template("guide_box")
        self.adapter.apply_operations([{"method": "set_geometry_style", "params": {
            "object_id": self.volume, "role": "measurement"}},
            {"method": "set_measurement_settings", "params": {
                "object_id": self.volume, "quality": "preview"}}])

    def fingerprint(self):
        return self.adapter.measurement_fingerprint(self.volume)

    def test_saved_results_and_noop_history_do_not_stale(self):
        result = self.adapter.analyze_measurement(self.volume)
        sid = self.adapter.create_snapshot(result, "Baseline")
        self.adapter.save_map_resource(name="Map", kind="static", view="2d",
                                       renderer="plotly", content={"data": []})
        self.adapter.create_snapshot(result, "Another snapshot")
        self.adapter.rename_snapshot(sid, "Renamed")
        self.adapter.apply_operations([{"method": "set_transform", "params": {
            "object_id": "coil", "position": [0., 0., 0.]}}])
        self.assertEqual(self.adapter.snapshot_status(sid), "current")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.magpy.json"
            self.adapter.save_file(path)
            reopened = StudioAdapter(json.loads(path.read_text()))
            self.assertEqual(reopened.snapshot_status(sid), "current")

    def test_physical_changes_and_restoration(self):
        original = self.fingerprint()
        for current in (2., 1.):
            self.adapter.apply_operations([{"method": "set_param", "params": {
                "object_id": "coil", "name": "current", "value": current}}])
            self.assertEqual(self.fingerprint() == original, current == 1.)
        self.adapter.apply_operations([{"method": "set_transform", "params": {
            "object_id": "coil", "position": [0., 0., .03]}}])
        self.assertNotEqual(self.fingerprint(), original)

    def test_only_linked_dataset_content_counts(self):
        a = self.adapter
        a.coil_specs['coil'].update(correction_mode='calibration', correction_calibration_id='fit')
        a.measurement_calibrations = [{'id':'fit', 'sources':[{'dataset_id':'data'}], 'fit':{'scale':1.}}]
        a.measurement_datasets = [{'id':'data', 'samples':[{'value':1.}], 'label':'Measured'},
                                  {'id':'unused', 'samples':[{'value':3.}]}]
        baseline = self.fingerprint()
        a.measurement_datasets[1]['samples'][0]['value'] = 4.
        a.measurement_datasets[0]['updated_at'] = 'later'
        self.assertEqual(self.fingerprint(), baseline)
        a.measurement_datasets[0]['samples'][0]['value'] = 2.
        self.assertNotEqual(self.fingerprint(), baseline)
        a.measurement_datasets[0]['samples'][0]['value'] = 1.
        self.assertEqual(self.fingerprint(), baseline)
        a.measurement_calibrations.clear()
        self.assertNotEqual(self.fingerprint(), baseline)

    def test_legacy_fingerprint_remains_readable(self):
        result = self.adapter.analyze_measurement(self.volume)
        sid = self.adapter.create_snapshot(result, 'Legacy')
        self.adapter._snapshot(sid)['input_fingerprint'] = self.adapter.measurement_fingerprint(self.volume, legacy=True)
        self.assertEqual(self.adapter.snapshot_status(sid), 'current')


if __name__ == '__main__':
    unittest.main()
