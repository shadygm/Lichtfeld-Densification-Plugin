"""Exercise the nested panel package using host stubs, without LichtFeld."""
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from threading import Thread, current_thread, main_thread
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

            def sync_all(self):
                return False

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
            log=Mock(), get_scene=lambda: None, Tensor=type('Tensor', (), {}),
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
                self.assertFalse(model.bindings['write_colmap'][0]())
                self.assertIn('data-checked="write_colmap"', Path(panel.template).read_text())
                model.bindings['write_colmap'][1](True)
                self.assertTrue(panel._write_colmap)
                with patch.object(panel, '_has_training_data', return_value=True), patch.object(
                    panel, '_get_effective_camera_nodes', return_value=[object(), object()],
                ), patch.object(panel, '_capture_base_point_cloud', return_value=True), patch(
                    'panel_structure_plugin.panels.densification.workflow.DensifyJob',
                ) as job:
                    panel._start()
                    self.assertTrue(job.call_args.kwargs['write_colmap'])
                    self.assertIs(job.call_args.kwargs['on_cloud_preview'].__self__, panel)
                    self.assertNotIn('on_sequential_viz', job.call_args.kwargs)
                    job.return_value.start.assert_called_once()
                panel.job = None
                panel._active_run_roi_only_selected = None
                self.assertEqual(set(model.events), {'do_start', 'do_cancel', 'toggle_section', 'num_step'})
                model.bindings['matches_per_ref_str'][1]('30000')
                self.assertEqual(panel.config.matches_per_ref, 30000)
                panel._on_num_step(None, None, ['matches_per_ref', 1])
                self.assertEqual(panel.config.matches_per_ref, 30000)
                panel._on_num_step(None, None, ['matches_per_ref', -1])
                self.assertEqual(panel.config.matches_per_ref, 29500)
                panel._on_num_step(None, None, ['matches_per_ref', 1])
                self.assertEqual(panel.config.matches_per_ref, 30000)
                model.bindings['matches_per_ref_str'][1]('30001')
                self.assertEqual(panel.config.matches_per_ref, 30000)
                model.bindings['certainty_thresh'][1](-2)
                self.assertEqual(panel.config.certainty_thresh, 0)
                panel._update_scrub_spec('nns_per_ref', max_value=2)
                panel._set_scrub_field_value('nns_per_ref', 9)
                self.assertEqual(panel.config.nns_per_ref, 2)
                self.assertIs(type(panel.config.nns_per_ref), int)
                for attr, lower, upper in [('nns_per_ref', 1.0, 2.0),
                                           ('min_track_length', 2.0, 10.0)]:
                    for value, expected in [(-100, int(lower)), (100, int(upper)),
                                             (lower, int(lower))]:
                        panel._set_int_config(attr, value, lower, upper)
                        self.assertEqual(getattr(panel.config, attr), expected)
                        self.assertIs(type(getattr(panel.config, attr)), int)
                        self.assertEqual(len(list(range(getattr(panel.config, attr)))), expected)
                    setattr(panel.config, attr, upper)
                    panel._set_int_config(attr, upper, lower, upper)
                    self.assertIs(type(getattr(panel.config, attr)), int)
                panel._set_scrub_field_value('reproj_thresh', 'invalid')
                self.assertEqual(panel.config.reproj_thresh, 0.8)
                panel.config.roi_only_selected = True
                panel._base_point_cloud_points = np.zeros((7, 3))
                panel._base_point_cloud_colors = np.zeros((7, 3), dtype=np.uint8)
                panel._start_training_when_complete = True
                cloud_module = importlib.import_module('panel_structure_plugin.core.reconstruction.cloud')
                points = np.arange(30, dtype=np.float32).reshape(10, 3)
                colors = np.full((10, 3), 128, dtype=np.uint8)
                result = module.DensifyResult(True, num_points=10,
                                            cloud=cloud_module.DenseCloud(points, colors))
                panel._on_complete(result)
                self.assertEqual(result.num_points, 17)
                self.assertIs(panel._pending_import, result)
                self.assertTrue(panel._pending_start_training)
                scene = Mock()
                target = SimpleNamespace(name='cloud', uuid='cloud-uuid', point_cloud=lambda: object())
                scene.get_node_by_uuid.return_value = target
                operations = Mock()
                ticket = SimpleNamespace(state='queued', error='', cancel=Mock())
                operations.set_data.return_value = ticket
                with patch.object(host, 'get_scene', return_value=scene), patch.object(
                    panel, '_resolve_target_point_cloud_node', return_value=target,
                ), patch.object(panel, '_set_point_cloud_data', operations.set_data), patch.object(
                    panel, '_start_training_after_import', operations.start_training,
                ):
                    panel.on_update(None)
                    self.assertIsNone(panel.last_result)
                    operations.start_training.assert_not_called()
                    ticket.state = 'published'
                    panel.on_update(None)
                self.assertEqual([call[0] for call in operations.mock_calls], ['set_data', 'start_training'])
                imported_points, imported_colors = operations.set_data.call_args.args[1:]
                np.testing.assert_array_equal(imported_points[:10], points)
                np.testing.assert_array_equal(imported_colors[:10], colors)
                self.assertEqual(len(imported_points), 17)
                scene.notify_changed.assert_not_called()
                self.assertIsNone(result.cloud)
                self.assertIsNone(panel._pending_import)
                self.assertEqual(result.num_points, 17)
                failed_import = module.DensifyResult(True, num_points=10,
                                                   cloud=cloud_module.DenseCloud(points, colors))
                panel._on_complete(failed_import)
                with patch.object(panel, '_start_training_after_import') as start_training:
                    panel.on_update(None)  # Host has no scene: importing fails.
                start_training.assert_not_called()
                self.assertIsNone(failed_import.cloud)
                for preview_active in (False, True):
                    with self.subTest(preview_active=preview_active):
                        failed = Panel()
                        base_points = np.zeros((7, 3))
                        failed._base_point_cloud_points = base_points
                        failed._base_point_cloud_colors = np.zeros((7, 3), dtype=np.uint8)
                        failed._active_run_roi_only_selected = True
                        failed._preview_override_active = preview_active
                        failed._pending_import = 'stale-preview.ply'
                        failed._pending_start_training = True

                        restore_ticket = SimpleNamespace(state='queued', error='', cancel=Mock())

                        def set_data(*args):
                            self.assertIs(current_thread(), main_thread())
                            np.testing.assert_array_equal(args[1], base_points)
                            # A synchronous scene notification must keep the snapshot alive.
                            failed.on_scene_changed(None)
                            self.assertIs(failed._base_point_cloud_points, base_points)
                            return restore_ticket

                        with patch.object(host, 'get_scene', return_value=scene) as get_scene, patch.object(
                            failed, '_resolve_target_point_cloud_node', return_value=target,
                        ), patch.object(failed, '_set_point_cloud_data', side_effect=set_data) as setter, patch.object(
                            failed, '_import_output', side_effect=AssertionError('Stale preview imported'),
                        ), patch.object(failed, '_start_training_after_import') as start_training:
                            worker = Thread(target=failed._on_error, args=(RuntimeError('worker failed'),))
                            worker.start()
                            worker.join(timeout=2)
                            self.assertFalse(worker.is_alive())
                            get_scene.assert_not_called()
                            setter.assert_not_called()
                            self.assertIsNone(failed.last_result)
                            self.assertTrue(failed._is_running())
                            failed.on_scene_changed(None)
                            self.assertIs(failed._base_point_cloud_points, base_points)
                            failed.on_update(None)
                            self.assertEqual(setter.call_count, int(preview_active))
                            self.assertFalse(failed.last_result.success)
                            self.assertEqual(failed.last_result.error, 'worker failed')
                            self.assertIsNone(failed._pending_import)
                            self.assertIsNone(failed._pending_error)
                            self.assertFalse(failed._pending_start_training)
                            if preview_active:
                                self.assertTrue(failed._is_running())
                                restore_ticket.state = 'published'
                                failed.on_update(None)
                            self.assertIsNone(failed._active_run_roi_only_selected)
                            self.assertFalse(failed._preview_override_active)
                            failed.on_update(None)
                            self.assertEqual(setter.call_count, int(preview_active))
                            start_training.assert_not_called()
