"""Numerically known residuals, observation weighting and invalid projections."""
import unittest

import numpy as np
import pycolmap

from core.camera_models import CameraRecord
from core.metrics import compute_reprojection_metrics


def camera(model=None):
    K = np.eye(3, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)
    return CameraRecord(1, '', 100, 100, K, R, t, np.eye(3, 4), np.zeros(3), colmap_camera=model)


class MetricsTests(unittest.TestCase):
    def test_known_residuals_and_observation_weighting(self):
        xyz = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        result = compute_reprojection_metrics(xyz, [[(1, 0, 0), (1, 3, 4)], [(1, 0, 0)]], [camera()], 4)
        self.assertAlmostEqual(result['observations']['mean'], 5 / 3)
        self.assertAlmostEqual(result['observations']['rmse'], np.sqrt(25 / 3))
        self.assertEqual(result['observations']['median'], 0)
        self.assertEqual(result['per_point_max']['mean'], 2.5)
        self.assertEqual(result['observations']['fraction_over_threshold_or_invalid'], 1 / 3)

    def test_distorted_camera_uses_full_projection(self):
        model = pycolmap.Camera(model='SIMPLE_RADIAL_FISHEYE', width=100, height=100, params=[100, 50, 50, 0.1])
        xyz = np.array([[0.5, 0.3, 1]], dtype=np.float32)
        uv = model.img_from_cam(xyz.astype(np.float64))[0] + [3, 4]
        result = compute_reprojection_metrics(xyz, [[(1, *uv)]], [camera(model)], 6)
        self.assertAlmostEqual(result['observations']['mean'], 5, places=4)

    def test_invalid_and_unobserved_points_are_counted(self):
        xyz = np.array([[0, 0, -1], [0, 0, 1]], dtype=np.float32)
        result = compute_reprojection_metrics(xyz, [[(1, 0, 0)], []], [camera()], 1)
        self.assertEqual(result['observations']['invalid_count'], 1)
        self.assertIsNone(result['observations']['mean'])
        self.assertEqual(result['per_point_max']['invalid_count'], 2)
        self.assertEqual(result['tracks']['unobserved_points'], 1)
        self.assertEqual(result['observations']['fraction_over_threshold_or_invalid'], 1)

    def test_missing_track_or_camera_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_reprojection_metrics(np.zeros((1, 3)), [], [camera()], 1)
        with self.assertRaises(ValueError):
            compute_reprojection_metrics(np.zeros((1, 3)), [[(2, 0, 0)]], [camera()], 1)


if __name__ == '__main__':
    unittest.main()
