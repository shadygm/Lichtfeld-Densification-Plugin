"""Clamped integer settings remain usable as slice indices."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class IntegerSettingsTests(unittest.TestCase):
    def test_float_bounds_and_existing_float_values(self):
        source = Path(__file__).parents[1] / 'densification.py'
        if not source.is_file():
            source = source.with_suffix('') / 'settings.py'
        panel = next(node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.ClassDef) and any(
                         isinstance(method, ast.FunctionDef) and method.name == '_set_int_config'
                         for method in node.body))
        setter = next(node for node in panel.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_set_int_config')
        namespace = {}
        exec(compile(ast.Module(body=[setter], type_ignores=[]), str(source), 'exec'), namespace)
        for current, value, expected in [(3, 1, 1), (3, 10, 10), (3, -10, 1),
                                         (3, 20, 10), (1., 1, 1), (10., 10, 10)]:
            with self.subTest(current=current, value=value):
                host = SimpleNamespace(config=SimpleNamespace(nns_per_ref=current), _dirty=Mock())
                namespace['_set_int_config'](host, 'nns_per_ref', value, 1., 10.)
                self.assertIs(type(host.config.nns_per_ref), int)
                self.assertEqual(host.config.nns_per_ref, expected)
                self.assertEqual(list(range(20))[:host.config.nns_per_ref], list(range(expected)))
                host._dirty.assert_called_once_with('nns_per_ref')
