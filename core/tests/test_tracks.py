"""Packed tracks retain observation order, precision and point alignment."""
import unittest
import numpy as np

from core.reconstruction.tracks import ObservationTracks, ObservationTrackBuilder


class TrackTests(unittest.TestCase):
    def test_bulk_builder_keeps_reference_then_neighbor_order_and_precision(self):
        builder = ObservationTrackBuilder(3, 3)
        pixels = np.array([[1.123456789123, 2.], [3., 4.], [5., 6.]])
        builder.append(np.arange(3), 2**40, pixels)
        builder.append(np.array([0, 2]), 9, pixels[[0, 2]] + .25)
        tracks = builder.finish()
        np.testing.assert_array_equal(tracks.lengths, [2, 1, 2])
        np.testing.assert_array_equal(tracks.offsets, [0, 2, 3, 5])
        np.testing.assert_array_equal(tracks.camera_ids, [2**40, 9, 2**40, 2**40, 9])
        np.testing.assert_array_equal(tracks.pixels, pixels[[0, 0, 1, 2, 2]] + [[0], [.25], [0], [0], [.25]])
        builder.append(np.array([1]), 3, np.zeros((1, 2)))
        np.testing.assert_array_equal(tracks.lengths, [2, 1, 2])

    def test_lengths_only_builder_allocates_no_observation_buffers(self):
        builder = ObservationTrackBuilder(3, 4, retain_observations=False)
        builder.append(np.arange(3), 1, np.zeros((3, 2)))
        builder.append(np.array([0, 2]), 2, np.ones((2, 2)))
        tracks = builder.finish()
        self.assertFalse(tracks.has_observations)
        self.assertIsNone(builder.camera_ids)
        self.assertIsNone(builder.pixels)
        self.assertEqual(tracks.lengths.nbytes, 12)
        np.testing.assert_array_equal(tracks.select([2, 1]).lengths, [2, 1])
        with self.assertRaisesRegex(ValueError, 'lengths only'):
            tracks.require_observations()

    def test_selection_preserves_empty_repeated_and_reordered_tracks(self):
        tracks = ObservationTracks.from_rows([[], [(1, .1, .2)], [(2, .3, .4), (1, .5, .6)]])
        selected = tracks.select([2, 0, 2, 1])
        expected = ObservationTracks.from_rows([[(2, .3, .4), (1, .5, .6)], [],
                                                [(2, .3, .4), (1, .5, .6)], [(1, .1, .2)]])
        for name in ('lengths', 'offsets', 'camera_ids', 'pixels'):
            np.testing.assert_array_equal(getattr(selected, name), getattr(expected, name))
        empty = tracks.select(np.zeros(3, dtype=bool))
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.pixels.shape, (0, 2))
        np.testing.assert_array_equal(tracks.select(np.array([False, True, True])).lengths, [1, 2])

    def test_concatenation_and_camera_groups_preserve_observation_order(self):
        first = ObservationTracks.from_rows([[(3, 1., 2.), (1, 3., 4.)]])
        second = ObservationTracks.from_rows([[], [(3, 5., 6.)]])
        combined = ObservationTracks.concatenate([first, second])
        np.testing.assert_array_equal(combined.lengths, [2, 0, 1])
        np.testing.assert_array_equal(combined.point_indices(), [0, 0, 2])
        groups = list(combined.camera_groups())
        self.assertEqual([camera for camera, _ in groups], [1, 3])
        np.testing.assert_array_equal(groups[1][1], [0, 2])
        with self.assertRaises(ValueError):
            ObservationTracks.concatenate([first, ObservationTracks(np.array([1], np.uint32))])
