"""Backend tuning must not persist into host training or after failures."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.pipeline.config import DensePipelineConfig
from core.pipeline.runner import run_dense_pipeline
from core.runtime.backend import matcher_backend_settings

import numpy as np


class BackendSettingsTests(TestCase):
    def test_restore_success_and_error(self):
        for setting in ('fast', 'high'):
            for benchmark in (False, True):
                for fail in (False, True):
                    with self.subTest(setting=setting, benchmark=benchmark, fail=fail):
                        matmul = SimpleNamespace(allow_tf32=False)
                        cudnn = SimpleNamespace(allow_tf32=False, benchmark=benchmark)
                        with patch('torch.cuda.is_available', return_value=True), \
                             patch('torch.backends.cuda.matmul', matmul), \
                             patch('torch.backends.cudnn', cudnn):
                            try:
                                with matcher_backend_settings(setting):
                                    self.assertTrue(matmul.allow_tf32)
                                    self.assertTrue(cudnn.allow_tf32)
                                    self.assertEqual(cudnn.benchmark, setting != 'fast')
                                    if fail:
                                        raise RuntimeError('matching failed')
                            except RuntimeError:
                                pass
                            self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32, cudnn.benchmark),
                                             (False, False, benchmark))

    def test_pipeline_restores_settings_when_model_setup_fails(self):
        matmul = SimpleNamespace(allow_tf32=False)
        cudnn = SimpleNamespace(allow_tf32=False, benchmark=True)
        cameras = SimpleNamespace(img_ids=[], distorted_ids=set())
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), patch('torch.cuda.ipc_collect'), \
             patch('torch.backends.cuda.matmul', matmul), \
             patch('torch.backends.cudnn', cudnn), \
             patch('core.pipeline.runner._build_camera_lookup', return_value=cameras), \
             patch('core.matching.roma.has_cached_romav2_weights', return_value=True), \
             patch('core.matching.roma.RomaMatcher', side_effect=RuntimeError('model failed')):
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                run_dense_pipeline([], [], np.empty((0, 0)), DensePipelineConfig(''))
        self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32, cudnn.benchmark),
                         (False, False, True))
