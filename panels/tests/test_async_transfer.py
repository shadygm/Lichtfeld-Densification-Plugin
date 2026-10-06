"""Publication, cancellation and rollback obey the asynchronous API contract."""
import gc
from threading import Thread, current_thread, main_thread
from types import SimpleNamespace
from unittest.mock import Mock, patch
import weakref

import numpy as np

from async_host import AsyncPanelTestCase, Scene


class AsyncTransferTests(AsyncPanelTestCase):
    def test_training_and_completion_wait_for_publication(self):
        old_cloud = self.scene.node.point_cloud()
        result, ticket = self.submit_final(train=True)
        self.assertIsNone(result.cloud)
        self.assertIsNone(self.panel.last_result)
        self.assertTrue(self.panel._is_running())
        self.assertEqual(self.panel._display_stage(), 'Uploading')
        self.assertEqual(self.panel._display_progress(), 99)
        self.assertIs(self.scene.node.point_cloud(), old_cloud)
        ticket.state = 'uploading'
        self.panel.on_update(None)
        self.host.start_training.assert_not_called()
        self.scene.on_publish = lambda: self.panel.on_scene_changed(None)
        base = self.panel._base_point_cloud_points
        ticket.publish()
        self.assertIs(self.panel._base_point_cloud_points, base)
        self.assertIsNot(self.scene.node.point_cloud(), old_cloud)
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, result)
        self.assertFalse(self.panel._is_running())
        self.host.start_training.assert_called_once()
        self.panel.on_update(None)
        self.host.start_training.assert_called_once()

    def test_latest_final_supersedes_pending_preview(self):
        points, colors = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8)
        self.panel._apply_dense_point_cloud(self.scene.node, self.scene.node.point_cloud(), points, colors)
        preview = self.panel._cloud_update.ticket
        result, final = self.submit_final()
        self.assertEqual(preview.state, 'superseded')
        preview.publish()
        self.assertIsNone(self.panel.last_result)
        final.publish()
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, result)
        self.assertEqual(len(self.scene.node.point_cloud().means.array), 3)

    def test_failed_or_superseded_final_never_starts_training(self):
        for state in ('failed', 'superseded', 'cancelled'):
            with self.subTest(state=state):
                self.panel._cancel_requested = False
                result, ticket = self.submit_final(train=True)
                ticket.state, ticket.error = state, 'native failure' if state == 'failed' else ''
                self.panel.on_update(None)
                self.assertFalse(self.panel.last_result.success)
                self.assertIn('native failure' if state == 'failed' else state, self.panel.last_result.error)
                self.assertFalse(self.panel._is_running())
                self.assertIsNone(result.cloud)
                self.host.start_training.assert_not_called()

    def test_worker_error_queues_restore_and_preserves_snapshot_until_publication(self):
        self.panel._active_run_roi_only_selected = True
        self.panel._apply_dense_point_cloud(self.scene.node, self.scene.node.point_cloud(),
                                            np.ones((3, 3), np.float32), np.zeros((3, 3), np.uint8))
        preview = self.panel._cloud_update.ticket
        preview.publish()  # Published between panel ticks.
        original_scene = self.host.get_scene.return_value
        calls = []

        def get_scene():
            calls.append(current_thread())
            self.assertIs(current_thread(), main_thread())
            return original_scene

        with patch.object(self.host, 'get_scene', side_effect=get_scene):
            worker = Thread(target=self.panel._on_error, args=(RuntimeError('worker failed'),))
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(calls, [])
            self.panel.on_update(None)
            self.assertFalse(self.panel.last_result.success)
            self.assertEqual(self.panel.last_result.error, 'worker failed')
            restore = self.panel._cloud_update.ticket
            self.assertEqual(self.panel._cloud_update.kind, 'restore')
            self.assertTrue(self.panel._is_running())
            self.assertTrue(self.panel._active_run_roi_only_selected)
            self.scene.on_publish = lambda: self.panel.on_scene_changed(None)
            restore.publish()
            self.assertIsNotNone(self.panel._base_point_cloud_points)
            self.panel.on_update(None)
            self.assertFalse(self.panel._preview_override_active)
            self.assertIsNone(self.panel._active_run_roi_only_selected)
            np.testing.assert_array_equal(self.scene.node.point_cloud().means.array, np.zeros((2, 3)))

    def test_cancel_pending_final_restores_instead_of_training(self):
        result, final = self.submit_final(train=True)
        self.panel._on_do_cancel(None, None, None)
        self.panel.on_update(None)
        self.assertEqual(final.state, 'cancelled')
        self.assertEqual(self.panel.last_result.error, 'Cancelled')
        self.assertEqual(self.panel._cloud_update.kind, 'restore')
        self.panel._cloud_update.ticket.publish()
        self.panel.on_update(None)
        self.assertFalse(self.panel._is_running())
        self.host.start_training.assert_not_called()
        late_result = self.result()
        self.panel._on_complete(late_result)
        self.assertIsNone(late_result.cloud)
        self.assertIsNone(self.panel._pending_import)

    def test_scene_replacement_does_not_paint_a_different_node(self):
        _, ticket = self.submit_final(train=True)
        replacement = Scene()
        replacement.node.uuid = 'replacement-uuid'
        replacement.nodes = {replacement.node.uuid: replacement.node}
        self.host.get_scene.return_value = replacement
        self.panel.on_scene_changed(None)
        self.panel.on_update(None)
        self.assertEqual(ticket.state, 'cancelled')
        self.assertIsNone(self.panel._resolve_target_point_cloud_node(replacement))
        self.assertEqual(replacement.node.payload.submissions, [])
        self.assertFalse(self.panel.last_result.success)
        self.host.start_training.assert_not_called()

    def test_target_rename_is_resolved_by_uuid(self):
        result, ticket = self.submit_final()
        self.scene.node.name = 'renamed'
        self.panel.on_scene_changed(None)
        self.assertIsNone(self.panel._pending_error)
        ticket.publish()
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, result)

    def test_roi_colors_preserve_normalized_base_and_dense_colors(self):
        self.panel.config.roi_only_selected = True
        result, ticket = self.submit_final()
        self.assertEqual(result.num_points, 5)
        self.assertEqual(ticket.colors.dtype, np.float32)
        np.testing.assert_allclose(ticket.colors[:3], 128 / 255)
        np.testing.assert_allclose(ticket.colors[3:], .25)
        self.assertTrue(ticket.colors.flags.c_contiguous)
        self.assertEqual(ticket.points.dtype, np.float32)
        ticket.publish()
        self.panel.on_update(None)
        self.assertEqual(len(self.scene.node.point_cloud().means.array), 5)

    def test_async_api_retains_owners_after_result_release_and_cancellation(self):
        result = self.result()
        points, colors = result.cloud.points, result.cloud.colors
        owner = weakref.ref(points)
        self.panel._on_complete(result)
        self.panel.on_update(None)
        ticket = self.panel._cloud_update.ticket
        self.assertIs(ticket.points, points)
        self.assertIs(ticket.colors, colors)
        del points, colors
        self.assertIsNone(result.cloud)
        self.panel._cancel_cloud_update()
        gc.collect()
        self.assertFalse(ticket.inputs_released)
        self.assertIsNotNone(owner())
        ticket.release_inputs()
        gc.collect()
        self.assertIsNone(owner())

    def test_unmount_cancels_uploads_and_ignores_late_completions(self):
        _, ticket = self.submit_final(train=True)
        self.panel.job = SimpleNamespace(cancel=Mock(), is_running=lambda: False)
        self.panel.on_unmount(Mock())
        self.assertEqual(ticket.state, 'cancelled')
        self.panel.job.cancel.assert_called_once()
        result = self.result()
        self.panel._on_complete(result)
        self.assertIsNone(result.cloud)
        self.assertIsNone(self.panel._pending_import)
        self.host.start_training.assert_not_called()

    def test_empty_cloud_publishes(self):
        result = self.result(np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
        self.panel._on_complete(result)
        self.panel.on_update(None)
        self.panel._cloud_update.ticket.publish()
        self.panel.on_update(None)
        self.assertTrue(self.panel.last_result.success)
        self.assertEqual(self.scene.node.point_cloud().means.array.shape, (0, 3))
