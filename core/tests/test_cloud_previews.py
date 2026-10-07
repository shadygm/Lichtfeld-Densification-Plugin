"""Live CPU snapshots preserve PLY preview values without writing a file."""
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.pipeline.preview import _emit_intermediate_preview
from core.pipeline.types import PipelineCancelled


class CloudPreviewTests(unittest.TestCase):
    def setUp(self):
        self.points = SimpleNamespace(
            xyz_parts=[np.array([[1., 2., 3.]], np.float32), np.array([[4., 5., 6.]], np.float32)],
            rgb_parts=[np.array([[0., .5, 1.]], np.float32), np.array([[1., 0., .25]], np.float32)],
            pairs_processed=3,
        )

    def test_cpu_snapshot_matches_legacy_ply_and_avoids_file_io(self):
        callback, files = Mock(), Mock()
        with patch('core.pipeline.preview.write_ply') as writer:
            _emit_intermediate_preview(files, None, 3, self.points, None, on_cloud_preview=callback)
            writer.assert_not_called()
        files.assert_not_called()
        cloud = callback.call_args.args[0]
        self.assertEqual(cloud.points.dtype, np.float32)
        self.assertEqual(cloud.colors.dtype, np.uint8)
        self.assertTrue(cloud.points.flags.c_contiguous)
        self.assertTrue(cloud.colors.flags.c_contiguous)
        with tempfile.TemporaryDirectory() as directory:
            _emit_intermediate_preview(files, str(Path(directory) / 'preview'), 3, self.points, None)
            body = Path(files.call_args.args[0]).read_bytes().split(b'end_header\n', 1)[1]
        expected = b''.join(struct.pack('<fffBBB', *point, *color)
                            for point, color in zip(cloud.points, cloud.colors))
        self.assertEqual(body, expected)

    def test_submitted_snapshot_does_not_alias_accumulator_or_later_previews(self):
        callback = Mock()
        _emit_intermediate_preview(None, None, 3, self.points, None, on_cloud_preview=callback)
        first = callback.call_args.args[0]
        self.points.xyz_parts[0][:] = 99
        self.points.rgb_parts[0][:] = 1
        self.points.pairs_processed = 6
        _emit_intermediate_preview(None, None, 3, self.points, None, on_cloud_preview=callback)
        second = callback.call_args.args[0]
        np.testing.assert_array_equal(first.points[0], [1, 2, 3])
        np.testing.assert_array_equal(first.colors[0], [0, 128, 255])
        self.assertFalse(np.shares_memory(first.points, second.points))
        self.assertFalse(np.shares_memory(first.colors, second.colors))

    def test_cancellation_stops_snapshot_publication(self):
        callback = Mock()
        with self.assertRaises(PipelineCancelled):
            _emit_intermediate_preview(None, None, 3, self.points, lambda: True,
                                       on_cloud_preview=callback)
        callback.assert_not_called()
