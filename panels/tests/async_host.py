"""A scene-owned async API stub: submitting is separate from publication."""
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
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


class Tensor:
    def __init__(self, array):
        self.array = array

    def numpy(self, copy=True):
        return self.array.copy() if copy else self.array


class Ticket:
    def __init__(self, node, points, colors):
        self.node = node
        self.points, self.colors = points, colors
        self.state, self.error = 'queued', ''
        self.inputs_released = False

    def cancel(self):
        if self.state in ('queued', 'uploading'):
            self.state = 'cancelled'
            return True
        return False

    def release_inputs(self):
        self.points = self.colors = None
        self.inputs_released = True

    def publish(self):
        if self.state not in ('queued', 'uploading'):
            return
        self.state = 'uploading'
        points = self.points.array if isinstance(self.points, Tensor) else self.points
        colors = self.colors.array if isinstance(self.colors, Tensor) else self.colors
        self.node.payload = Cloud(self.node, points.copy(), colors.copy())
        if self.node.scene.on_publish:
            self.node.scene.on_publish()
        self.state = 'published'
        self.release_inputs()


class Cloud:
    def __init__(self, node, points, colors):
        self.node = node
        self.means, self.colors = Tensor(points), Tensor(colors)
        self.submissions = []

    def set_data_async(self, points, colors, *, queue_policy):
        assert queue_policy == 'latest'
        previous = self.node.latest
        if previous and previous.state in ('queued', 'uploading'):
            previous.state = 'superseded'
        ticket = Ticket(self.node, points, colors)
        self.node.latest = ticket
        self.submissions.append(ticket)
        return ticket

    def set_data(self, *args):
        raise AssertionError('Synchronous point-cloud upload')


class Scene:
    def __init__(self):
        self.on_publish = None
        self.node = SimpleNamespace(uuid='cloud-uuid', name='cloud', type=1,
                                    has_camera=False, latest=None, scene=self)
        self.node.payload = Cloud(self.node, np.zeros((2, 3), np.float32),
                                  np.full((2, 3), .25, np.float32))
        self.node.point_cloud = lambda: self.node.payload
        self.nodes = {self.node.uuid: self.node}

    def get_node_by_uuid(self, uuid):
        return self.nodes.get(uuid)

    def get_node(self, name):
        return next((node for node in self.nodes.values() if node.name == name), None)

    def get_nodes(self):
        return list(self.nodes.values())


class AsyncPanelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = SimpleNamespace(
            Tensor=Tensor, log=Mock(), get_scene=Mock(), start_training=Mock(),
            io=SimpleNamespace(load_point_cloud=Mock(side_effect=AssertionError('Synchronous PLY load'))),
            scene=SimpleNamespace(NodeType=SimpleNamespace(POINTCLOUD=1)),
            ui=SimpleNamespace(Panel=object, PanelSpace=SimpleNamespace(MAIN_PANEL_TAB=1),
                               PanelHeightMode=SimpleNamespace(CONTENT=1)),
        )
        sdk = SimpleNamespace(ScrubFieldController=Mock(), ScrubFieldSpec=ScrubSpec)
        cls.modules_patch = patch.dict(sys.modules, {'lichtfeld': cls.host, 'lfs_plugins': sdk})
        cls.modules_patch.start()
        package = ModuleType(f'async_test_{cls.__name__}')
        package.__path__ = [str(Path(__file__).parents[2])]
        sys.modules[package.__name__] = package
        cls.module = importlib.import_module(f'{package.__name__}.panels.densification')
        cls.Panel = cls.module.DensificationPanel
        cls.cloud_module = importlib.import_module(f'{package.__name__}.core.reconstruction.cloud')

    @classmethod
    def tearDownClass(cls):
        cls.modules_patch.stop()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = patch.object(self.Panel, '_get_cache_dir', staticmethod(lambda: directory.name))
        cache.start()
        self.addCleanup(cache.stop)
        self.scene = Scene()
        self.host.get_scene.reset_mock()
        self.host.get_scene.side_effect = None
        self.host.get_scene.return_value = self.scene
        self.host.start_training.reset_mock()
        self.panel = self.Panel()
        self.panel._sync_scrub_specs = Mock(return_value=False)
        self.panel._scrub_fields.sync_all.return_value = False
        self.assertTrue(self.panel._capture_base_point_cloud())

    def result(self, points=None, colors=None):
        points = np.ones((3, 3), np.float32) if points is None else points
        colors = np.full((len(points), 3), 128, np.uint8) if colors is None else colors
        return self.module.DensifyResult(
            True, num_points=len(points), cloud=self.cloud_module.DenseCloud(points, colors),
        )

    def submit_final(self, *, train=False):
        result = self.result()
        self.panel._start_training_when_complete = train
        self.panel._on_complete(result)
        self.panel.on_update(None)
        return result, self.panel._cloud_update.ticket
