"""Bidirectional prediction must not change either cached descriptor list."""
import unittest

import torch

from core.matching.model import RoMaV2
from core.matching.model.matcher import _compute_head_preds
from core.matching.roma import _CachedFeatures


class BidirectionalCacheTests(unittest.TestCase):
    def test_neighbor_descriptors_survive_repeated_bidirectional_predictions(self):
        class Features(torch.nn.Module):
            def forward(self, image):
                return [image.permute(0, 2, 3, 1).clone()]

        class Head(torch.nn.Module):
            def forward(self, features, **kwargs):
                return features[-1]

        class Matcher(torch.nn.Module):
            def forward(self, features_a, features_b, img_A, img_B, bidirectional):
                predictions = {}
                for suffix, features in (("AB", features_a), ("BA", features_b)):
                    warp, confidence = _compute_head_preds(
                        f_list_A=features,
                        match_emb_AB=torch.ones_like(features[-1]),
                        f_mv_A=torch.zeros_like(features[-1]),
                        img_A=img_A, img_B=img_B, head=Head(),
                    )
                    predictions[f"warp_{suffix}"] = warp
                    predictions[f"confidence_{suffix}"] = confidence
                return predictions

        previous_precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous_precision)
        torch.set_float32_matmul_precision("highest")
        model = RoMaV2.__new__(RoMaV2)
        torch.nn.Module.__init__(model)
        model.f = _CachedFeatures(Features())
        model.matcher = Matcher()
        model.refiners = torch.nn.ModuleDict()
        model.refiner_features = lambda image: {}
        model.bidirectional = True
        model.anchor_width = model.anchor_height = 512
        model.eval()
        image_a, image_b = torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2)
        with torch.inference_mode():
            features_a, features_b = model.f(image_a), model.f(image_b)
            original_a, original_b = features_a[-1], features_b[-1]
            first = model._forward_from_features(features_a, image_a, image_b)
            second = model._forward_from_features(features_a, image_a, image_b)
        self.assertIs(features_a[-1], original_a)
        self.assertIs(features_b[-1], original_b)
        torch.testing.assert_close(original_a, torch.zeros_like(original_a))
        torch.testing.assert_close(original_b, torch.ones_like(original_b))
        for name in ("warp_AB", "warp_BA", "confidence_AB", "confidence_BA"):
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
