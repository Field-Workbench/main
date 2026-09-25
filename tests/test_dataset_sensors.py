import copy
import unittest
from unittest import mock
import numpy as np
from scipy.spatial.transform import Rotation
from fieldworkbench.studio_adapter import StudioAdapter, StudioOperationError


class DatasetSensorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = StudioAdapter({'objects': []})
        self.samples = [
            {'position_m': [.01, .02, .03], 'rotation_matrix': Rotation.from_euler('xyz', [23, 41, -12], degrees=True).as_matrix().tolist(), 'measured_uT': -5.},
            {'position_m': [0., .05, 0.], 'rotation_matrix': np.eye(3).tolist(), 'measured_uT': None},
        ]
        self.did = self.adapter.save_measurement_dataset({'label':'Measurements', 'sensor_id':'missing',
            'sensor_label':'Axis 1 (saved coordinates) (saved coordinates)', 'samples':self.samples, 'output':'By'})

    def test_old_labels_are_cleaned_idempotently(self):
        record = self.adapter.measurement_dataset(self.did)
        self.assertEqual(record['sensor_label'], 'Axis 1')
        self.adapter.save_measurement_dataset(record)
        self.assertEqual(self.adapter.measurement_dataset(self.did)['sensor_label'], 'Axis 1')

    def test_points_orientations_dataset_and_single_undo(self):
        before = self.adapter.document()
        count = self.adapter.history()['undo']
        ids = self.adapter.create_measurement_dataset_sensors(self.did)
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.adapter.history()['undo'], count + 1)
        parents = {self.adapter.get_object(oid)['parent'] for oid in ids}
        self.assertEqual(len(parents), 1)
        self.assertNotIn(None, parents)
        for oid, expected in zip(ids, self.samples):
            actual = self.adapter.measurement_sensor_samples(oid)
            self.assertEqual(len(actual), 1)
            np.testing.assert_allclose(actual[0]['position_m'], expected['position_m'], atol=1e-14)
            np.testing.assert_allclose(actual[0]['rotation_matrix'], expected['rotation_matrix'], atol=1e-14)
        self.assertEqual(self.adapter.document()['field_workbench']['measurement_datasets'], before['field_workbench']['measurement_datasets'])
        self.adapter.undo()
        self.assertEqual(self.adapter.document(), before)
        self.adapter.redo()
        self.assertEqual(len(self.adapter.measurement_sensor_catalog()), 2)
        second = self.adapter.create_measurement_dataset_sensors(self.did)
        self.assertFalse(set(ids) & set(second))

    def test_failure_rolls_back(self):
        before = self.adapter.document()
        history = self.adapter.history()
        with mock.patch.object(self.adapter.session, 'set_transform', return_value={'ok':False,'error':'failure'}):
            with self.assertRaises(StudioOperationError):
                self.adapter.create_measurement_dataset_sensors(self.did)
        self.assertEqual(self.adapter.document(), before)
        self.assertEqual(self.adapter.history(), history)

    def test_invalid_orientation_and_empty_dataset_leave_scene_untouched(self):
        for samples in ([], [{'position_m':[0,0,0], 'rotation_matrix':np.zeros((3,3)).tolist()}]):
            self.adapter.measurement_datasets[0]['samples'] = samples
            before = self.adapter.document()
            with self.assertRaises(StudioOperationError):
                self.adapter.create_measurement_dataset_sensors(self.did)
            self.assertEqual(self.adapter.document(), before)


if __name__ == '__main__':
    unittest.main()
