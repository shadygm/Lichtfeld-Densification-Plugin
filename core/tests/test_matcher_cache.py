"""Feature reuse must be bounded and tied to the exact prepared image."""
import unittest

import torch

from core.matcher import _CachedFeatures


class FeatureCacheTests(unittest.TestCase):
    def test_reuse_and_eviction(self):
        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                return image * 2

        module = Features()
        cached = _CachedFeatures(module, capacity=2)
        a, b, c = (torch.tensor([value]) for value in range(3))
        first = cached(a)
        cached(b)
        self.assertIs(cached(a), first)
        cached(c)
        cached(b)
        self.assertEqual(module.calls, 4)
        self.assertEqual(len(cached.cache), 2)
        torch.testing.assert_close(cached(b), b * 2)
