"""Prepared images are reused within a job and refreshed for the next job."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from core.pipeline import _CameraLookup, _PackContext, _pack_reference_batch


class PreparedImageTests(unittest.TestCase):
    def test_reuse_mask_application_and_job_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cameras = {}
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, 0] = 255
            Image.fromarray(mask).save(root / 'mask.png')
            for uid in (1, 2):
                path = root / f'{uid}.png'
                Image.fromarray(np.full((8, 8, 3), 23 * uid, dtype=np.uint8)).save(path)
                cameras[uid] = SimpleNamespace(
                    image_path=str(path), width=8, height=8,
                    mask_path=str(root / 'mask.png') if uid == 1 else None,
                )
            lookup = _CameraLookup([1, 2], cameras, set())
            def context():
                return _PackContext(lookup, np.array([[1], [0]]), 1, 8, 8)
            job = context()
            first = _pack_reference_batch(0, job)
            second = _pack_reference_batch(1, job)
            self.assertIs(first.imA_np, second.nn_arrays[0])
            self.assertIs(second.imA_np, first.nn_arrays[0])
            self.assertIs(first.maskA_np, second.nn_masks[0])
            self.assertTrue(np.all(first.imA_np[:, 0] == 23))
            self.assertTrue(np.all(first.imA_np[:, 1:] == 0))
            self.assertEqual(job.load_image.cache_info().currsize, 2)
            Image.fromarray(np.full((8, 8, 3), 99, dtype=np.uint8)).save(root / '2.png')
            self.assertTrue(np.all(context().load_image(2)[0] == 99))
            job.load_image.cache_clear()
            self.assertEqual(job.load_image.cache_info().currsize, 0)
