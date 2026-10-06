"""Coverage sampling must retain probability draws and certainty ordering."""
import unittest

import numpy as np
import torch

from core.matching.sampling import select_samples_with_coverage


class SamplingTests(unittest.TestCase):
    def test_coverage_selects_best_positive_sample_per_tile(self):
        cert = torch.tensor([[.1, .2, 0, .4], [.6, .5, 0, .3],
                             [.1, .7, .2, .8], [0, .1, .3, .9]])
        weights = (cert / cert.sum()).numpy().ravel()
        np.random.seed(7)
        main = np.random.choice(16, size=8, replace=False, p=weights)
        # Tiles are 2x2: highest certainty indices, in descending order.
        expected = np.unique(np.concatenate((main, [15, 9])))
        np.random.seed(7)
        actual = select_samples_with_coverage(cert, 10, border=0, tiles=2)
        np.testing.assert_array_equal(actual, expected)

    def test_empty_certainty_has_no_samples(self):
        self.assertEqual(select_samples_with_coverage(torch.zeros(4, 4), 10).size, 0)
