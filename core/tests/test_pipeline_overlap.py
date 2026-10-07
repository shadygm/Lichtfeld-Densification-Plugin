"""The pipeline overlaps work but drains results and previews in order."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
from threading import Event, current_thread, main_thread
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.pipeline.config import DensePipelineConfig
from core.pipeline import PipelineCancelled, run_dense_pipeline
from core.pipeline.types import _TriangulatedReference
from core.previews.files import TemporaryCloudPreviews
from core.reconstruction.tracks import ObservationTracks


class PipelineOverlapTests(unittest.TestCase):
    def test_overlap_order_and_main_thread_previews(self):
        self.run_pipeline(cancel=False)

    def test_cancellation_waits_for_worker_and_releases_runtime(self):
        self.run_pipeline(cancel=True)

    def test_pipeline_passes_mps_to_matcher(self):
        self.run_pipeline(cancel=False, mps=True)

    def test_ui_pipeline_keeps_lengths_without_observation_buffers(self):
        self.run_pipeline(cancel=False, retain=False)

    def test_file_previews_are_removed_on_success_cancel_and_failure(self):
        for cancel, fail in ((False, False), (True, False), (False, True)):
            with self.subTest(cancel=cancel, fail=fail):
                self.run_pipeline(cancel=cancel, files=True, fail=fail)

    def test_triangulation_errors_fail_mid_run_and_final_drain(self):
        for ref in (1, 2):
            with self.subTest(ref=ref):
                self.run_pipeline(cancel=False, files=True, tri_fail=ref)

    def test_skipped_packages_do_not_count_as_matching_work(self):
        self.run_pipeline(cancel=False, skip=True)

    def run_pipeline(self, cancel, mps=False, files=False, fail=False, retain=True, tri_fail=None, skip=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / 'final.ply'
        output.write_bytes(b'keep explicit export')
        temporary_roots, preview_paths = [], []

        def create_previews(*args):
            owned = TemporaryCloudPreviews(*args)
            if owned.base:
                temporary_roots.append(Path(owned.base).parent)
            return owned

        def consume_preview(path):
            preview_paths.append(Path(path))
            self.assertTrue(Path(path).read_bytes().startswith(b'ply\n'))

        cancelled = Event()
        next_match = Event()
        packages = [SimpleNamespace(ref_id=uid) for uid in (1, 2)]
        if skip:
            packages[0] = None
        def packs():
            yield from packages
            if fail:
                raise RuntimeError('loader failed')

        loader = MagicMock()
        loader.__iter__.return_value = packs()
        matcher = SimpleNamespace(w_resized=10, h_resized=10, sample_thresh=.9, close=MagicMock())
        seen = []
        progress = MagicMock()

        def match(packed, pair_counter, **kwargs):
            if packed.ref_id == 2:
                next_match.set()
            return SimpleNamespace(packed=packed), pair_counter + 1

        def triangulate(matched, context, **kwargs):
            self.assertIsNot(current_thread(), main_thread())
            self.assertEqual(context.retain_observations, retain)
            self.assertTrue(next_match.wait(timeout=2))
            if cancel:
                cancelled.set()
            uid = matched.packed.ref_id
            if uid == tri_fail:
                raise ValueError('unexpected geometry failure')
            return _TriangulatedReference(
                np.full((1, 3), uid, dtype=np.float32), np.zeros((1, 3)),
                np.zeros(1), (ObservationTracks.from_rows([[(uid, 0., 0.)]]) if retain
                              else ObservationTracks(np.array([1], np.uint32))), {}, {},
            )

        def preview(points, **kwargs):
            self.assertIs(current_thread(), main_thread())
            seen.append(points.pairs_processed)

        with ExitStack() as stack:
            mocks = {}
            for target, replacement in {
                'core.pipeline.runner._build_camera_lookup': SimpleNamespace(img_ids=[1, 2], by_id={}, distorted_ids=set()),
                'core.pipeline.runner._build_pack_loader': loader,
                'core.matching.roma.RomaMatcher': matcher,
                'core.matching.roma.has_cached_romav2_weights': True,
                'core.pipeline.runner.torch.cuda.is_available': False,
                'core.pipeline.runner.torch.backends.mps.is_available': mps,
            }.items():
                mocks[target] = stack.enter_context(patch(target, return_value=replacement))
            stack.enter_context(patch('core.pipeline.runner._collect_reference_matches', side_effect=match))
            stack.enter_context(patch('core.pipeline.runner._triangulate_ref', side_effect=triangulate))
            stack.enter_context(patch('core.pipeline.runner.TemporaryCloudPreviews', side_effect=create_previews))
            if skip:
                stack.enter_context(patch('core.pipeline.control.time.perf_counter', side_effect=[100., 110.]))
            if not files:
                stack.enter_context(patch('core.pipeline.runner._emit_intermediate_preview', side_effect=preview))
            def run():
                return run_dense_pipeline(
                    [], [0, 1], np.array([[1], [0]]),
                    DensePipelineConfig(output_path=str(output), viz_interval=1),
                    cancel_requested=cancelled.is_set,
                    on_sequential_viz=consume_preview if files else None,
                    retain_observations=retain,
                    progress_callback=progress,
                )
            if cancel:
                with self.assertRaises(PipelineCancelled):
                    run()
            elif fail:
                with self.assertRaisesRegex(RuntimeError, 'loader failed'):
                    run()
            elif tri_fail:
                with self.assertRaisesRegex(RuntimeError, f'Triangulation failed for ref {tri_fail}') as caught:
                    run()
                self.assertIsInstance(caught.exception.__cause__, ValueError)
            else:
                result = run()
                np.testing.assert_array_equal(result.xyz[:, 0], [2] if skip else [1, 2])
                if not files:
                    self.assertEqual(seen, [1] if skip else [1, 2])
                self.assertEqual(result.pairs_processed, 1 if skip else 2)
                self.assertEqual(result.tracks.has_observations, retain)
                if skip:
                    self.assertIn('Matching 1/2 (1 skipped)', [call.args[1] for call in progress.call_args_list])
                    self.assertIn('Matching 2/2 (1 skipped) | 0.1 it/s',
                                  [call.args[1] for call in progress.call_args_list])
            self.assertEqual(mocks['core.matching.roma.RomaMatcher'].call_args.kwargs['device'],
                             'mps' if mps else 'cpu')
        loader.close.assert_called_once_with(wait=True)
        matcher.close.assert_called_once()
        if files:
            self.assertEqual(len(temporary_roots), 1)
            self.assertFalse(temporary_roots[0].exists())
            self.assertTrue(all(not path.exists() for path in preview_paths))
            if not cancel and tri_fail != 1:
                self.assertGreater(len(preview_paths), 0)
        self.assertEqual(output.read_bytes(), b'keep explicit export')
