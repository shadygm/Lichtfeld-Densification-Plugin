"""Binary PLY layout must remain compatible with existing readers."""
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import pycolmap

from core.cameras.models import CameraRecord
from core.reconstruction.tracks import ObservationTracks
from core.reconstruction.writers import write_ply, write_sparse_model_bin


class PlyTests(unittest.TestCase):
    def test_packed_vertices_and_empty_cloud(self):
        for count in (0, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                xyz = np.array([[.1, -2., 3.5], [1000., 0., -.25]], dtype=np.float64)[:count]
                rgb = np.array([[0, 128, 255], [17, 32, 63]], dtype=np.uint8)[:count]
                path = Path(directory) / 'cloud.ply'
                write_ply(str(path), xyz, rgb)
                header, body = path.read_bytes().split(b'end_header\n', 1)
                self.assertIn(f'element vertex {count}\n'.encode(), header)
                expected = b''.join(struct.pack('<fffBBB', *point, *color)
                                    for point, color in zip(xyz, rgb))
                self.assertEqual(body, expected)
                self.assertEqual(len(body), 15 * count)


class OutputPathTests(unittest.TestCase):
    def test_relative_and_nested_output(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name in ('OUT.ply', 'nested/cloud.ply'):
                    with self.subTest(name=name):
                        write_ply(name, np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8))
                        self.assertIn(b'element vertex 1\n', Path(name).read_bytes())
            finally:
                os.chdir(previous)


class SparseWriterTests(unittest.TestCase):
    def test_numeric_export_round_trip_preserves_observation_indices_and_precision(self):
        records = [CameraRecord(uid, f'{uid}.png', 100, 100, np.eye(3), np.eye(3),
                                np.zeros(3), np.eye(3, 4), np.zeros(3)) for uid in (7, 3)]
        exact = 1.123456789123456
        tracks = ObservationTracks.from_rows([
            [(7, exact, 2.), (3, 3., 4.), (7, 99., 99.), (123, 5., 6.)], [],
            [(3, 9., 10.), (7, 11., 12.)],
        ])
        with tempfile.TemporaryDirectory() as directory:
            write_sparse_model_bin(directory, records, np.array([[1., 2., 3.]] * 3),
                                   np.full((3, 3), 128, np.uint8), np.array([.1, .2, .3]), tracks)
            reconstruction = pycolmap.Reconstruction(directory)
        self.assertEqual(reconstruction.num_points3D(), 3)
        np.testing.assert_array_equal(reconstruction.images[7].points2D[0].xy, [exact, 2.])
        np.testing.assert_array_equal(reconstruction.images[7].points2D[1].xy, [11., 12.])
        self.assertEqual(reconstruction.images[3].points2D[1].point3D_id, 3)
        first = reconstruction.points3D[1].track.elements
        self.assertEqual([(row.image_id, row.point2D_idx) for row in first], [(7, 0), (3, 0)])
        self.assertEqual(reconstruction.points3D[2].track.length(), 0)
        third = reconstruction.points3D[3].track.elements
        self.assertEqual([(row.image_id, row.point2D_idx) for row in third], [(3, 1), (7, 1)])

    def test_empty_export_and_lengths_only_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            write_sparse_model_bin(directory, [], np.empty((0, 3)), np.empty((0, 3), np.uint8),
                                   None, ObservationTracks.from_rows([]))
            self.assertEqual(Path(directory, 'points3D.bin').read_bytes(), struct.pack('<Q', 0))
            with self.assertRaisesRegex(ValueError, 'lengths only'):
                write_sparse_model_bin(directory, [], np.zeros((1, 3)), np.zeros((1, 3), np.uint8),
                                       None, ObservationTracks(np.array([2], np.uint32)))
