"""Binary PLY layout must remain compatible with existing readers."""
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from core.reconstruction.writers import write_ply


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
