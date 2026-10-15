"""Feature reuse must be bounded and tied to the exact prepared image."""
import unittest
import weakref
from collections import OrderedDict
from unittest import mock

import torch

from core.matching.roma import RomaMatcher, _CachedFeatures, _fit_caches_to_free_memory


class FeatureCacheTests(unittest.TestCase):
    def test_close_releases_owned_model_and_cached_tensors(self):
        def matcher_with_cached_features():
            matcher = RomaMatcher.__new__(RomaMatcher)
            model = torch.nn.Module()
            model.f = _CachedFeatures(torch.nn.Linear(3, 3))
            model.refiner_features = _CachedFeatures(torch.nn.Linear(3, 3))
            image = torch.ones(1, 3)
            with torch.inference_mode():
                descriptor = model.f(image)
                fine_features = model.refiner_features(image)
            grid = torch.zeros(2, 2, 2)
            matcher.model = model
            matcher._image_cache = OrderedDict([(1, image)])
            matcher._grid_cache = {(2, 2): grid}
            probes = [weakref.ref(value) for value in (model, image, descriptor, fine_features, grid)]
            return matcher, probes

        matcher, probes = matcher_with_cached_features()
        self.assertTrue(all(probe() is not None for probe in probes))
        matcher.close()
        self.assertTrue(all(probe() is None for probe in probes))
        matcher.close()

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


class CacheCapacityTests(unittest.TestCase):
    DEFAULTS = (256, 256, 24)

    def fit(self, free_gib):
        free = int(free_gib * 2**30)
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(free, free)):
            return _fit_caches_to_free_memory(torch.device("cuda"), 512 * 512, 0, 2, self.DEFAULTS)

    def test_small_gpu_shrinks_caches(self):
        image, descriptor, refiner = self.fit(5.0)
        self.assertLess(image, self.DEFAULTS[0])
        self.assertLess(descriptor, self.DEFAULTS[1])
        self.assertLessEqual(refiner, self.DEFAULTS[2])

    def test_large_gpu_keeps_defaults(self):
        self.assertEqual(self.fit(40.0), self.DEFAULTS)

    def test_exhausted_gpu_keeps_minimum_capacity(self):
        self.assertEqual(self.fit(0.5), (2, 2, 2))
