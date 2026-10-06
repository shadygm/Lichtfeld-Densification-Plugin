"""Projection metadata and small-batch triangulation regression checks."""
import unittest

import numpy as np
import pycolmap

from core.cameras.geometry import K_from_camera, dlt_triangulate_batch


class GeometryTests(unittest.TestCase):
    def test_single_and_empty_triangulation(self):
        P1 = np.eye(3, 4, dtype=np.float32)
        P2 = P1.copy()
        P2[0, 3] = -1
        uv1 = np.array([[0.2, 0.1]], dtype=np.float32)
        uv2 = np.array([[0, 0.1]], dtype=np.float32)
        np.testing.assert_allclose(dlt_triangulate_batch(P1, P2, uv1, uv2), [[1, 0.5, 5, 1]], atol=1e-6)
        self.assertEqual(dlt_triangulate_batch(P1, P2, uv1[:0], uv2[:0]).shape, (0, 4))

    def test_intrinsics_for_single_and_dual_focal_models(self):
        models = {
            'PINHOLE': [700, 710, 500, 400],
            'SIMPLE_PINHOLE': [700, 500, 400],
            'SIMPLE_RADIAL': [700, 500, 400, 0.01],
            'SIMPLE_RADIAL_FISHEYE': [700, 500, 400, 0.01],
            'RADIAL_FISHEYE': [700, 500, 400, 0.01, 0.02],
            'OPENCV': [700, 710, 500, 400, 0.01, 0.02, 0, 0],
            'OPENCV_FISHEYE': [700, 710, 500, 400, 0.01, 0.02, 0, 0],
        }
        for model, params in models.items():
            with self.subTest(model=model):
                camera = pycolmap.Camera(model=model, width=1000, height=800, params=params)
                dual_focal = model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE')
                expected = [[700, 0, 500], [0, 710 if dual_focal else 700, 400], [0, 0, 1]]
                np.testing.assert_array_equal(K_from_camera(camera), expected)


if __name__ == '__main__':
    unittest.main()
