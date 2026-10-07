"""Check point-cloud copy boundaries against the native setter's ownership contract."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class Tensor:
    def __init__(self, array):
        self.array = array
        self.clone = Mock(side_effect=AssertionError('Unexpected redundant clone'))
        self.numpy = Mock(side_effect=self._numpy)

    def _numpy(self, copy=True):
        self.last_numpy = self.array.copy() if copy else self.array
        return self.last_numpy

    @staticmethod
    def from_numpy(array, copy=True):
        if not copy or not array.flags.c_contiguous:
            raise AssertionError('Native from_numpy requires a contiguous copying import')
        return Tensor(array.copy())


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[2]
        cls.host = SimpleNamespace(
            Tensor=Tensor, log=Mock(), get_scene=Mock(),
            ui=SimpleNamespace(Panel=object, PanelSpace=SimpleNamespace(MAIN_PANEL_TAB=1),
                               PanelHeightMode=SimpleNamespace(CONTENT=1)),
        )
        sdk = SimpleNamespace(ScrubFieldController=Mock(), ScrubFieldSpec=SimpleNamespace)
        cls.modules_patch = patch.dict(sys.modules, {'lichtfeld': cls.host, 'lfs_plugins': sdk})
        cls.modules_patch.start()
        package = ModuleType('transfer_test_plugin')
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        cls.Panel = importlib.import_module('transfer_test_plugin.panels.densification').DensificationPanel

    @classmethod
    def tearDownClass(cls):
        cls.modules_patch.stop()

    def setUp(self):
        self.panel = self.Panel.__new__(self.Panel)

    def test_native_tensors_are_passed_to_setter_without_cloning(self):
        points = Tensor(np.ones((2, 3), dtype=np.float32))
        colors = Tensor(np.ones((2, 3), dtype=np.uint8))
        cloud = Mock()
        ticket = self.panel._set_point_cloud_data(cloud, points, colors)
        cloud.set_data_async.assert_called_once_with(points, colors, queue_policy='latest')
        self.assertIs(ticket, cloud.set_data_async.return_value)
        cloud.set_data.assert_not_called()
        points.clone.assert_not_called()
        colors.clone.assert_not_called()

    def test_numpy_transfer_handles_strides_and_preserves_dtypes(self):
        points = np.arange(18, dtype=np.float32).reshape(3, 6)[:, ::2]
        colors = np.arange(9, dtype=np.uint8).reshape(3, 3)
        cloud = Mock()
        self.panel._set_point_cloud_data(cloud, points, colors)
        actual_points, actual_colors = cloud.set_data_async.call_args.args
        np.testing.assert_array_equal(actual_points, points)
        np.testing.assert_array_equal(actual_colors, colors)
        self.assertTrue(actual_points.flags.c_contiguous)
        self.assertIs(actual_colors, colors)
        self.assertEqual(actual_points.dtype, np.float32)
        self.assertEqual(actual_colors.dtype, np.uint8)
        cloud.set_data.assert_not_called()

    def test_float64_arrays_are_prepared_without_tensor_staging(self):
        points = np.ones((2, 3), dtype=np.float64)
        colors = np.full((2, 3), .5, dtype=np.float64)
        cloud = Mock()
        self.panel._set_point_cloud_data(cloud, points, colors)
        actual_points, actual_colors = cloud.set_data_async.call_args.args
        self.assertEqual(actual_points.dtype, np.float32)
        self.assertEqual(actual_colors.dtype, np.float32)
        np.testing.assert_array_equal(actual_colors, colors)

    def test_snapshot_is_independent_without_copying_native_export_twice(self):
        original = np.arange(9, dtype=np.float32).reshape(3, 3)
        for value in (original, Tensor(original)):
            with self.subTest(native=isinstance(value, Tensor)):
                snapshot = self.Panel._coerce_point_cloud_array(value, 'means')
                self.assertFalse(np.shares_memory(snapshot, original))
                np.testing.assert_array_equal(snapshot, original)
                if isinstance(value, Tensor):
                    self.assertIs(snapshot, value.last_numpy)
                    value.numpy.assert_called_once_with(copy=True)

    def test_roi_merge_reads_views_and_owns_merged_arrays(self):
        panel = self.Panel.__new__(self.Panel)
        panel._base_point_cloud_points = np.zeros((1, 3), dtype=np.float32)
        panel._base_point_cloud_colors = np.zeros((1, 3), dtype=np.uint8)
        for native in (False, True):
            with self.subTest(native=native):
                points = np.ones((2, 3), dtype=np.float32)
                colors = np.full((2, 3), 128, dtype=np.uint8)
                inputs = (Tensor(points), Tensor(colors)) if native else (points, colors)
                merged_points, merged_colors = panel._build_roi_merge_arrays(*inputs)
                np.testing.assert_array_equal(merged_points, np.vstack((points, np.zeros((1, 3)))))
                np.testing.assert_array_equal(merged_colors, np.vstack((colors, np.zeros((1, 3)))))
                self.assertFalse(np.shares_memory(merged_points, points))
                self.assertFalse(np.shares_memory(merged_colors, colors))
                if native:
                    for value in inputs:
                        value.numpy.assert_called_once_with(copy=False)

    def test_invalid_snapshot_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'must be 2D'):
            self.Panel._coerce_point_cloud_array(np.zeros(3), 'means')
