"""Known stereo geometry through the shared triangulation path."""
import unittest

import numpy as np
import torch

from core.cameras.models import CameraRecord
from core.pipeline.config import DensePipelineConfig
from core.pipeline.types import _MatchedReference, _PackedReferenceBatch, _TriangulationContext
from core.pipeline.preparation import _build_camera_lookup
from core.pipeline.triangulation import _triangulate_ref


class TriangulationTests(unittest.TestCase):
    def test_stereo_tracks_and_optional_debug(self):
        size = 10
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)
        records = []
        for uid, center_x in ((1, 0), (2, 1)):
            R = np.eye(3, dtype=np.float32)
            t = np.array([[-center_x], [0], [0]], dtype=np.float32)
            records.append(CameraRecord(
                uid=uid, image_path='', width=100, height=100, K=K, R=R, t=t,
                P=K @ np.concatenate([R, t], axis=1), C=np.array([center_x, 0, 0]),
            ))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        # Baseline 1, focal 100, depth 5 -> disparity 20 pixels in full camera coordinates.
        disparity = 2 * 20 / ((size - 1) * (100 / size))
        warp = torch.stack([xx, yy, xx - disparity, yy], dim=-1)
        packed = _PackedReferenceBatch(
            ref_id=1, ref_path='', imA_np=np.full((size, size, 3), 128, dtype=np.uint8),
            maskA_np=None, wA_cam=100, hA_cam=100,
            nn_ids=[2], nn_masks=[None], nn_arrays=[],
        )
        matched = _MatchedReference(packed, [warp], [torch.ones(size, size)], {}, {})
        cameras = _build_camera_lookup(records)
        self.assertIs(cameras.by_id[1], records[0])
        ctx = _TriangulationContext(cameras, DensePipelineConfig(output_path='', matches_per_ref=8), 0.9, size, size)
        for debug, retain in ((False, True), (True, True), (False, False)):
            ctx = _TriangulationContext(cameras, ctx.config, .9, size, size, retain_observations=retain)
            np.random.seed(0)
            result = _triangulate_ref(matched, ctx, collect_debug_matches=debug)
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result.xyz[:, 2], 5, atol=1e-5)
            self.assertLess(float(result.err.max()), 1e-4)
            np.testing.assert_array_equal(result.tracks.lengths, np.full(len(result.xyz), 2))
            self.assertEqual(result.tracks.has_observations, retain)
            if retain:
                np.testing.assert_array_equal(result.tracks.camera_ids.reshape(-1, 2),
                                               np.tile([1, 2], (len(result.xyz), 1)))
            self.assertEqual(bool(result.debug_matches_by_nbr), debug)


if __name__ == '__main__':
    unittest.main()
