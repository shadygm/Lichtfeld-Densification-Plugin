"""Legacy publication shares completion, rollback and ROI lifecycle semantics."""
import importlib
from threading import current_thread, main_thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from async_host import AsyncPanelTestCase, Tensor


class LegacyCloud:
    # Deliberately has no async setter, like the current personal LFS checkout.
    def __init__(self, node):
        self.node = node
        self.means, self.colors = node.payload.means, node.payload.colors
        self.published = Mock()

    def set_data(self, points, colors):
        assert current_thread() is main_thread()
        assert isinstance(points, Tensor) and isinstance(colors, Tensor)
        self.means, self.colors = points, colors
        self.published(points, colors)
        if self.node.scene.on_publish:
            self.node.scene.on_publish()


class LegacyTransferTests(AsyncPanelTestCase):
    def setUp(self):
        super().setUp()
        version = patch.object(self.host, '__version__', 'v0.5.4', create=True)
        version.start()
        self.addCleanup(version.stop)
        self.scene.node.payload = LegacyCloud(self.scene.node)

    def test_publisher_selected_once_and_uses_current_cloud_wrapper(self):
        publishing = importlib.import_module(f'{self.module.__name__}.publishing')
        points, colors = np.ones((2, 3), np.float32), np.ones((2, 3), np.uint8)
        for version, async_supported in (('v0.5.4', False), ('v0.5.5', True)):
            with self.subTest(version=version):
                self.host.__version__ = version
                panel = self.Panel.__new__(self.Panel)
                with patch.object(publishing, 'supports_async_publication',
                                  wraps=publishing.supports_async_publication) as check:
                    for _ in range(3):
                        cloud = Mock()
                        panel._set_point_cloud_data(cloud, points, colors)
                        if async_supported:
                            cloud.set_data_async.assert_called_once_with(points, colors, queue_policy='latest')
                            cloud.set_data.assert_not_called()
                        else:
                            cloud.set_data.assert_called_once()
                            cloud.set_data_async.assert_not_called()
                    check.assert_called_once()

    def test_version_boundary_and_api_availability(self):
        points = np.ones((2, 3), np.float32)
        colors = np.full((2, 3), 128, np.uint8)
        for version, async_available, expected_async in (
            ('0.5.0', True, False), ('0.5.3', True, False),
            ('v0.5.4', True, False), ('v0.5.4-12-gabc-dirty', True, False),
            ('v0.5.5', True, True), ('0.10.0', True, True),
            ('v0.6.0-rc1', True, True), ('v0.5.5', False, False),
            ('unknown', True, True), ('6f0a44031', False, False),
        ):
            with self.subTest(version=version, api=async_available):
                cloud = Mock()
                if not async_available:
                    cloud.set_data_async = None
                self.host.__version__ = version
                panel = self.Panel.__new__(self.Panel)
                ticket = panel._set_point_cloud_data(cloud, points, colors)
                if expected_async:
                    cloud.set_data_async.assert_called_once_with(points, colors, queue_policy='latest')
                    cloud.set_data.assert_not_called()
                    self.assertIs(ticket, cloud.set_data_async.return_value)
                else:
                    if async_available:
                        cloud.set_data_async.assert_not_called()
                    self.assertEqual(ticket.state, 'published')
                    self.assertFalse(ticket.cancel())
                    actual_points, actual_colors = cloud.set_data.call_args.args
                    np.testing.assert_array_equal(actual_points.array, points)
                    np.testing.assert_array_equal(actual_colors.array, colors)
                    self.assertEqual(actual_colors.array.dtype, np.uint8)
                    self.assertFalse(np.shares_memory(actual_points.array, points))
        self.host.__version__ = 'unknown'
        with patch.object(self.host, 'build_info', SimpleNamespace(version='v0.5.4'), create=True):
            cloud = Mock()
            panel = self.Panel.__new__(self.Panel)
            panel._set_point_cloud_data(cloud, points, colors)
            cloud.set_data_async.assert_not_called()
            cloud.set_data.assert_called_once()

    def test_final_publication_keeps_snapshot_during_scene_callbacks(self):
        cloud = self.scene.node.point_cloud()
        base = self.panel._base_point_cloud_points
        def scene_changed():
            self.panel.on_scene_changed(None)
            self.assertIs(self.panel._base_point_cloud_points, base)
            self.host.start_training.assert_not_called()
        self.scene.on_publish = scene_changed
        result, ticket = self.submit_final(train=True)
        self.assertEqual(ticket.state, 'published')
        self.assertIsNone(result.cloud)
        self.assertIsNone(self.panel.last_result)
        np.testing.assert_array_equal(cloud.means.array, np.ones((3, 3)))
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, result)
        self.assertIsNone(self.panel._base_point_cloud_points)
        self.assertFalse(self.panel._is_running())
        self.host.start_training.assert_called_once()

    def test_preview_then_error_restores_original_cloud(self):
        cloud = self.scene.node.point_cloud()
        self.panel._on_cloud_preview(self.result().cloud)
        self.panel.on_update(None)
        np.testing.assert_array_equal(cloud.means.array, np.ones((3, 3)))
        self.panel._on_error(RuntimeError('worker failed'))
        self.panel.on_update(None)
        np.testing.assert_array_equal(cloud.means.array, np.zeros((2, 3)))
        self.panel.on_update(None)
        self.assertEqual(self.panel.last_result.error, 'worker failed')
        self.assertIsNone(self.panel._base_point_cloud_points)
        self.host.start_training.assert_not_called()

    def test_cancel_after_sync_final_restores_without_training(self):
        self.submit_final(train=True)
        self.panel._on_do_cancel(None, None, None)
        self.panel.on_update(None)
        self.panel.on_update(None)
        np.testing.assert_array_equal(self.scene.node.point_cloud().means.array, np.zeros((2, 3)))
        self.assertEqual(self.panel.last_result.error, 'Cancelled')
        self.assertFalse(self.panel._is_running())
        self.host.start_training.assert_not_called()

    def test_sync_submission_failure_restores_previous_preview(self):
        cloud = self.scene.node.point_cloud()
        self.panel._on_cloud_preview(self.result().cloud)
        self.panel.on_update(None)
        self.panel.on_update(None)
        set_data = cloud.set_data
        fail_once = True
        def submit(points, colors):
            nonlocal fail_once
            if fail_once:
                fail_once = False
                raise RuntimeError('legacy upload failed')
            return set_data(points, colors)
        with patch.object(cloud, 'set_data', side_effect=submit):
            self.submit_final(train=True)
        self.panel.on_update(None)
        self.assertEqual(self.panel.last_result.error, 'legacy upload failed')
        np.testing.assert_array_equal(cloud.means.array, np.zeros((2, 3)))
        self.assertFalse(self.panel._is_running())
        self.host.start_training.assert_not_called()

    def test_roi_and_empty_arrays_publish_synchronously(self):
        self.panel.config.roi_only_selected = True
        result, _ = self.submit_final()
        cloud = self.scene.node.point_cloud()
        self.assertEqual(result.num_points, 5)
        np.testing.assert_allclose(cloud.colors.array[:3], 128 / 255)
        np.testing.assert_allclose(cloud.colors.array[3:], .25)
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, result)
        self.panel.config.roi_only_selected = False
        result = self.result(np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
        self.panel._on_complete(result)
        self.panel.on_update(None)
        self.panel.on_update(None)
        self.assertEqual(cloud.means.array.shape, (0, 3))
        self.assertIs(self.panel.last_result, result)

    def test_legacy_tensor_inputs_do_not_require_an_extra_import(self):
        points = Tensor(np.ones((2, 3), np.float32))
        colors = Tensor(np.ones((2, 3), np.float32))
        cloud = Mock()
        with patch.object(Tensor, 'from_numpy', side_effect=AssertionError('extra import')):
            self.panel._set_point_cloud_data(cloud, points, colors)
        cloud.set_data.assert_called_once_with(points, colors)
