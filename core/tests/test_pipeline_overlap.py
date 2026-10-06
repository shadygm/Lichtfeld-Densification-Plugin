"""The pipeline overlaps work but drains results and previews in order."""
from contextlib import ExitStack
from types import SimpleNamespace
from threading import Event, current_thread, main_thread
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.config import DensePipelineConfig
from core.pipeline import PipelineCancelled, run_dense_pipeline
from core.stages.types import _TriangulatedReference


class PipelineOverlapTests(unittest.TestCase):
    def test_overlap_order_and_main_thread_previews(self):
        self.run_pipeline(cancel=False)

    def test_cancellation_waits_for_worker_and_releases_runtime(self):
        self.run_pipeline(cancel=True)

    def run_pipeline(self, cancel):
        cancelled = Event()
        next_match = Event()
        packages = [SimpleNamespace(ref_id=uid) for uid in (1, 2)]
        loader = MagicMock()
        loader.__iter__.return_value = iter(packages)
        matcher = SimpleNamespace(w_resized=10, h_resized=10, sample_thresh=.9, close=MagicMock())
        seen = []

        def match(packed, pair_counter, **kwargs):
            if packed.ref_id == 2:
                next_match.set()
            return SimpleNamespace(packed=packed), pair_counter + 1

        def triangulate(matched, context, **kwargs):
            self.assertIsNot(current_thread(), main_thread())
            self.assertTrue(next_match.wait(timeout=2))
            if cancel:
                cancelled.set()
            uid = matched.packed.ref_id
            return _TriangulatedReference(
                np.full((1, 3), uid, dtype=np.float32), np.zeros((1, 3)),
                np.zeros(1), [[(uid, 0., 0.)]], {}, {},
            )

        def preview(points, **kwargs):
            self.assertIs(current_thread(), main_thread())
            seen.append(points.pairs_processed)

        with ExitStack() as stack:
            for target, replacement in {
                'core.pipeline._build_camera_lookup': SimpleNamespace(img_ids=[1, 2], by_id={}, distorted_ids=set()),
                'core.pipeline._build_pack_loader': loader,
                'core.matcher.RomaMatcher': matcher,
                'core.matcher.has_cached_romav2_weights': True,
                'core.pipeline.torch.cuda.is_available': False,
            }.items():
                stack.enter_context(patch(target, return_value=replacement))
            stack.enter_context(patch('core.pipeline._collect_reference_matches', side_effect=match))
            stack.enter_context(patch('core.pipeline._triangulate_ref', side_effect=triangulate))
            stack.enter_context(patch('core.pipeline._emit_intermediate_preview', side_effect=preview))
            def run():
                return run_dense_pipeline(
                    [], [0, 1], np.array([[1], [0]]), DensePipelineConfig(output_path=''),
                    cancel_requested=cancelled.is_set,
                )
            if cancel:
                with self.assertRaises(PipelineCancelled):
                    run()
            else:
                result = run()
                np.testing.assert_array_equal(result.xyz[:, 0], [1, 2])
                self.assertEqual(seen, [1, 2])
                self.assertEqual(result.pairs_processed, 2)
        loader.close.assert_called_once_with(wait=True)
        matcher.close.assert_called_once()
