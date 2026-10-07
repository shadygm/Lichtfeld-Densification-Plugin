"""Device selection respects CUDA priority and falls back to MPS, then CPU."""
import runpy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from core.matching.device import get_device


class DeviceTests(unittest.TestCase):
    def test_model_and_pipeline_share_device_selection(self):
        model_device = Path(__file__).parents[1] / 'matching' / 'model' / 'device.py'
        for cuda, mps, expected in [(True, True, 'cuda'), (True, False, 'cuda'),
                                    (False, True, 'mps'), (False, False, 'cpu')]:
            with self.subTest(cuda=cuda, mps=mps), patch.object(
                torch.cuda, 'is_available', return_value=cuda,
            ), patch.object(torch.backends.mps, 'is_available', return_value=mps):
                self.assertEqual(get_device(), expected)
                namespace = runpy.run_path(str(model_device),
                                          run_name='core.matching.model._device_test')
                self.assertEqual(namespace['device'].type, expected)
