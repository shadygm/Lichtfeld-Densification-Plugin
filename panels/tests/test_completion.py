"""Exercise the completion callback without importing or launching LichtFeld."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


class CompletionTests(unittest.TestCase):
    def test_completion_with_and_without_base_cloud(self):
        source = Path(__file__).parents[1] / 'densification.py'
        panel_class = next(node for node in ast.parse(source.read_text()).body
                           if isinstance(node, ast.ClassDef) and node.name == 'DensificationPanel')
        callback = next(node for node in panel_class.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_on_complete')
        namespace = {'lf': SimpleNamespace(log=Mock()), 'DensifyResult': SimpleNamespace}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), 'exec'), namespace)
        for roi, base in [(False, None), (True, None), (True, np.zeros((7, 3)))]:
            with self.subTest(roi=roi, base=base is not None):
                panel = SimpleNamespace(
                    _base_point_cloud_points=base, _run_roi_only_selected=lambda: roi,
                    _auto_import=True, _start_training_when_complete=True,
                )
                result = SimpleNamespace(success=True, num_points=10, output_path='/tmp/dense/sparse/0')
                namespace['_on_complete'](panel, result)
                self.assertIs(panel.last_result, result)
                self.assertEqual(result.num_points, 17 if roi and base is not None else 10)
                self.assertEqual(panel._pending_import, result.output_path)
                self.assertTrue(panel._pending_start_training)
