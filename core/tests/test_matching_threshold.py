"""Certainty controls reject low scores without mutating matcher output."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from core.pipeline.config import DensePipelineConfig
from core.pipeline.matching import _collect_reference_matches


class MatchingThresholdTests(TestCase):
    def test_threshold_masks_and_unfiltered_mode(self):
        scores = torch.tensor([[0., .1], [.2, .9]])
        packed = SimpleNamespace(
            ref_id=1, imA_np=np.zeros((2, 2, 3), np.uint8),
            nn_arrays=[np.zeros((2, 2, 3), np.uint8)], nn_ids=[2],
            maskA_np=np.array([[1, 1], [0, 1]]), nn_masks=[None],
        )
        matcher = SimpleNamespace(match_grids_batch=lambda *args, **kw:
                                  [(torch.zeros(2, 2, 4), scores)])
        for no_filter, expected in ((False, [[0., 0.], [0., .9]]),
                                    (True, [[0., .1], [0., .9]])):
            with self.subTest(no_filter=no_filter), patch('torch.cuda.is_available', return_value=False):
                result, count = _collect_reference_matches(
                    packed, matcher, DensePipelineConfig('', no_filter=no_filter), 0, None,
                )
                torch.testing.assert_close(result.cert_list_cpu[0], torch.tensor(expected))
                self.assertEqual(count, 1)
        torch.testing.assert_close(scores, torch.tensor([[0., .1], [.2, .9]]))
