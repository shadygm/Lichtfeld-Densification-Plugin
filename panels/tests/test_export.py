"""Validate UI cloud handoff and optional export without launching LichtFeld."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from threading import current_thread, main_thread
import unittest
from unittest.mock import Mock, patch

import numpy as np


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # Load once before patching host modules; its native types cannot be reimported.

        root = Path(__file__).parents[2]
        cls.host = SimpleNamespace(log=Mock())
        cls.host_patch = patch.dict(sys.modules, {'lichtfeld': cls.host})
        cls.host_patch.start()
        # Namespace packages let us import the actual adapter/job without UI registration.
        for name, path in (
            ('export_test_plugin', root),
            ('export_test_plugin.panels', root / 'panels'),
            ('export_test_plugin.panels.densification', root / 'panels/densification'),
        ):
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        cls.adapter = importlib.import_module('export_test_plugin.densify')
        cls.pipeline = importlib.import_module('export_test_plugin.core.pipeline')
        cls.jobs = importlib.import_module('export_test_plugin.panels.densification.job')
        cls.writers = importlib.import_module('export_test_plugin.core.reconstruction.writers')

    @classmethod
    def tearDownClass(cls):
        cls.host_patch.stop()
        for name in list(sys.modules):
            if name == 'export_test_plugin' or name.startswith('export_test_plugin.'):
                del sys.modules[name]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'sparse/0'
        self.config = self.adapter.DensePipelineConfig(
            output_path=str(self.output), num_refs=1.0, nns_per_ref=1, min_track_length=2,
        )
        K, R = np.eye(3), np.eye(3)
        self.records = [self.adapter.CameraRecord(
            uid=uid, image_path=f'{uid}.png', width=100, height=100,
            K=K, R=R, t=np.array([-uid, 0., 0.]), P=np.zeros((3, 4)),
            C=np.array([uid, 0., 0.]),
        ) for uid in (1, 2)]
        self.raw = SimpleNamespace(
            xyz=np.array([[1., 0., 5.], [2., 0., 5.], [3., 0., 5.]], dtype=np.float64),
            rgb=np.array([[0., 0., 0.], [.5, 0., 1.], [1., 1., 1.]]),
            err=np.array([.1, .2, .3]),
            tracks=[[(1, 0., 0.)], [(1, 1., 1.), (2, 2., 2.)],
                    [(1, 3., 3.), (2, 4., 4.)]],
        )
        self.enterContext(patch.object(self.adapter, 'extract_cameras_from_lfs', return_value=self.records))
        self.run_pipeline = Mock(return_value=self.raw)
        self.enterContext(patch.dict(self.pipeline.__dict__, run_dense_pipeline=self.run_pipeline))

    def test_default_skips_writer_and_returns_filtered_cloud(self):
        progress = Mock()
        with patch.object(self.adapter, 'write_sparse_model_bin') as write:
            code, cloud = self.adapter.dense_init_from_lfs([], self.config, progress)
        self.assertEqual(code, 0)
        write.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIsNone(cloud.output_path)
        np.testing.assert_array_equal(cloud.points, self.raw.xyz[1:])
        np.testing.assert_array_equal(cloud.colors, [[128, 0, 255], [255, 255, 255]])
        self.assertEqual(cloud.points.dtype, np.float32)
        self.assertEqual(cloud.colors.dtype, np.uint8)
        self.assertFalse(any('Writing' in call.args[1] for call in progress.call_args_list))

    def test_cpu_preview_callback_reaches_core(self):
        callback = Mock()
        code, _ = self.adapter.dense_init_from_lfs([], self.config, on_cloud_preview=callback)
        self.assertEqual(code, 0)
        self.assertIs(self.run_pipeline.call_args.kwargs['on_cloud_preview'], callback)

    def test_opt_in_export_matches_direct_cloud_after_filter_and_cap(self):
        self.config.max_points = 1
        _, direct = self.adapter.dense_init_from_lfs([], self.config)
        code, exported = self.adapter.dense_init_from_lfs([], self.config, write_colmap=True)
        self.assertEqual(code, 0)
        self.assertEqual(exported.output_path, str(self.output))
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {'cameras.bin', 'images.bin', 'points3D.bin'})
        xyz, rgb, lengths = self.writers.read_points3D_bin_point_cloud(str(self.output / 'points3D.bin'))
        np.testing.assert_array_equal(xyz, direct.points)
        np.testing.assert_array_equal(rgb, direct.colors)
        np.testing.assert_array_equal(exported.points, direct.points)
        np.testing.assert_array_equal(lengths, [2])

    def test_cancellation_and_empty_filter_never_export(self):
        with patch.object(self.adapter, 'write_sparse_model_bin') as write:
            code, message = self.adapter.dense_init_from_lfs(
                [], self.config, cancel_requested=lambda: True, write_colmap=True,
            )
            self.assertEqual((code, message), (2, 'Cancelled'))
            self.config.min_track_length = 3
            code, message = self.adapter.dense_init_from_lfs([], self.config, write_colmap=True)
            self.assertEqual(code, 1)
            self.assertIn('No points remain', message)
        write.assert_not_called()

    def test_job_counts_in_memory_points_and_reports_finalizing(self):
        complete, progress = Mock(), Mock()
        job = self.jobs.DensifyJob(self.config, camera_nodes=[object(), object()],
                                  on_complete=complete, on_progress=progress)
        with patch.object(self.adapter, 'write_sparse_model_bin') as write:
            job._run()
        self.assertEqual(job.stage, self.jobs.DensifyStage.DONE)
        self.assertEqual(job.result.num_points, 2)
        self.assertIsNone(job.result.output_path)
        self.assertIsNotNone(job.result.cloud)
        self.assertIsNone(job.camera_nodes)
        complete.assert_called_once_with(job.result)
        self.assertIn('Finalizing', [call.args[0] for call in progress.call_args_list])
        self.assertNotIn('Writing', [call.args[0] for call in progress.call_args_list])
        write.assert_not_called()

    def test_cancelled_job_never_completes_or_exports(self):
        complete = Mock()
        job = self.jobs.DensifyJob(self.config, camera_nodes=[object(), object()],
                                  on_complete=complete, write_colmap=True)
        self.run_pipeline.side_effect = lambda *args, **kwargs: (job.cancel(), self.raw)[1]
        with patch.object(self.adapter, 'write_sparse_model_bin') as write:
            job._run()
        self.assertEqual(job.stage, self.jobs.DensifyStage.CANCELLED)
        self.assertFalse(job.result.success)
        self.assertIsNone(job.camera_nodes)
        complete.assert_not_called()
        write.assert_not_called()

    def test_postprocessing_reuses_observations_and_keeps_alignment(self):
        xyz, rgb, err, tracks = self.adapter._apply_track_filter(
            self.raw.xyz, self.raw.rgb, self.raw.err, self.raw.tracks, 2,
        )
        self.assertIs(tracks[0], self.raw.tracks[1])
        self.assertIs(tracks[1], self.raw.tracks[2])
        np.testing.assert_array_equal(err, [.2, .3])
        no_op = self.adapter._apply_track_filter(xyz, rgb, err, tracks, 1)
        for original, returned in zip((xyz, rgb, err, tracks), no_op):
            self.assertIs(original, returned)
        no_cap = self.adapter._apply_point_cap(xyz, rgb, err, tracks, 0, 0)
        self.assertIs(no_cap[3], tracks)
        capped = self.adapter._apply_point_cap(xyz, rgb, err, tracks, 1, 0)
        np.testing.assert_array_equal(capped[0], self.raw.xyz[2:])
        np.testing.assert_array_equal(capped[1], self.raw.rgb[2:])
        np.testing.assert_array_equal(capped[2], self.raw.err[2:])
        self.assertIs(capped[3][0], self.raw.tracks[2])
        self.assertEqual([len(track) for track in self.raw.tracks], [1, 2, 2])

    def test_allocator_cleanup_runs_on_worker_after_success_error_and_cancel(self):
        for outcome in ('success', 'error', 'cancel'):
            with self.subTest(outcome=outcome):
                self.run_pipeline.side_effect = RuntimeError('pipeline failed') if outcome == 'error' else None
                job = self.jobs.DensifyJob(self.config, camera_nodes=[object(), object()])
                if outcome == 'cancel':
                    job.cancel()
                calls = []

                def release():
                    self.assertIsNot(current_thread(), main_thread())
                    self.assertIsNone(job.camera_nodes)
                    calls.append(current_thread())

                with patch.object(self.jobs, 'release_free_memory', side_effect=release):
                    job.start()
                    job.wait(timeout=5)
                self.assertFalse(job._thread.is_alive())
                self.assertEqual(len(calls), 1)
                expected = {'success': 'DONE', 'error': 'ERROR', 'cancel': 'CANCELLED'}[outcome]
                self.assertEqual(job.stage, getattr(self.jobs.DensifyStage, expected))

    def test_distance_filter_preserves_track_then_error_priority(self):
        xyz = np.array([[.1, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [1.1, 0., 0.]])
        rgb = np.arange(12).reshape(4, 3)
        tracks = [[(1, 0., 0.)], [(1, 0., 0.), (2, 0., 0.)],
                  [(1, 0., 0.), (2, 0., 0.)], [(1, 0., 0.)]]
        err = np.array([.01, .3, .2, .4])
        points, colors, errors, kept = self.adapter._voxel_select_track_preserving(
            xyz, rgb, err, tracks, 1.0,
        )
        np.testing.assert_array_equal(points, xyz[[2, 3]])
        np.testing.assert_array_equal(colors, rgb[[2, 3]])
        np.testing.assert_array_equal(errors, err[[2, 3]])
        self.assertIs(kept[0], tracks[2])
        self.assertIs(kept[1], tracks[3])
