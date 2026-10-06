"""Exercise the nested panel package using host stubs, without LichtFeld."""
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


@dataclass
class ScrubSpec:
    min_value: float
    max_value: float
    step: float
    fmt: str
    data_type: type


class PanelStructureTests(unittest.TestCase):
    def test_bindings_settings_completion_and_assets(self):
        class ScrubController:
            def __init__(self, specs, **kwargs):
                self.specs = specs

            def set_spec(self, prop, spec):
                self.specs[prop] = spec

        class Model:
            def __init__(self):
                self.bindings = {}
                self.events = {}

            def bind(self, key, getter, setter):
                self.bindings[key] = (getter, setter)

            def bind_func(self, key, getter):
                self.bindings[key] = (getter, None)

            def bind_event(self, key, callback):
                self.events[key] = callback

            def get_handle(self):
                return Mock()

        host = SimpleNamespace(
            ui=SimpleNamespace(
                Panel=object, PanelSpace=SimpleNamespace(MAIN_PANEL_TAB=1),
                PanelHeightMode=SimpleNamespace(CONTENT=1),
            ),
            log=Mock(), get_scene=lambda: None,
        )
        sdk = SimpleNamespace(ScrubFieldSpec=ScrubSpec, ScrubFieldController=ScrubController)
        root = Path(__file__).parents[2]
        with patch.dict(sys.modules, {'lichtfeld': host, 'lfs_plugins': sdk}):
            spec = importlib.util.spec_from_file_location(
                'panel_structure_plugin', root / '__init__.py',
                submodule_search_locations=[str(root)],
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = package
            spec.loader.exec_module(package)
            module = importlib.import_module('panel_structure_plugin.panels.densification')
            Panel = module.DensificationPanel
            with tempfile.TemporaryDirectory() as directory, patch.object(
                Panel, '_get_cache_dir', staticmethod(lambda: directory),
            ):
                panel = Panel()
                self.assertTrue(Path(panel.template).is_file())
                self.assertTrue(Path(panel.template).with_suffix('.rcss').is_file())
                model = Model()
                panel.on_bind_model(SimpleNamespace(create_data_model=lambda _: model))
                self.assertEqual(model.bindings['num_refs'][0](), '0.80')
                self.assertEqual(set(model.events), {'do_start', 'do_cancel', 'toggle_section', 'num_step'})
                model.bindings['certainty_thresh'][1](-2)
                self.assertEqual(panel.config.certainty_thresh, 0)
                panel._update_scrub_spec('nns_per_ref', max_value=2)
                panel._set_scrub_field_value('nns_per_ref', 9)
                self.assertEqual(panel.config.nns_per_ref, 2)
                panel._set_scrub_field_value('reproj_thresh', 'invalid')
                self.assertEqual(panel.config.reproj_thresh, 0.8)
                panel.config.roi_only_selected = True
                panel._base_point_cloud_points = np.zeros((7, 3))
                panel._start_training_when_complete = True
                result = module.DensifyResult(True, '/tmp/dense/sparse/0', 10)
                panel._on_complete(result)
                self.assertEqual(result.num_points, 17)
                self.assertEqual(panel._pending_import, result.output_path)
                self.assertTrue(panel._pending_start_training)
