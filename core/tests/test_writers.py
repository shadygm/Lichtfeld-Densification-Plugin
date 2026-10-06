"""PLY output accepts bare relative names and creates nested parents."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.writers import write_ply


class OutputPathTests(unittest.TestCase):
    def test_relative_and_nested_output(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name in ('OUT.ply', 'nested/cloud.ply'):
                    with self.subTest(name=name):
                        write_ply(name, np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8))
                        self.assertIn(b'element vertex 1\n', Path(name).read_bytes())
            finally:
                os.chdir(previous)
