# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense initialization panel state and registration metadata."""

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
import lichtfeld as lf
from lfs_plugins import ScrubFieldController
from ...core.pipeline.config import DensePipelineConfig
from .settings import SCRUB_FIELD_SPECS


from .bindings import DensificationBindings
from .settings import DensificationSettings
from .scene import DensificationScene
from .workflow import DensificationWorkflow
from .transfer import DensificationTransfer


class DensificationPanel(DensificationBindings, DensificationSettings, DensificationScene, DensificationWorkflow, DensificationTransfer, lf.ui.Panel):
    """GUI panel for dense point cloud initialization workflow.

    This panel uses cameras already loaded in LichtFeld Studio.
    Simply load your scene, adjust parameters if desired, and click Start.
    The resulting dense point cloud will be automatically added to the scene.
    """

    id = "densification.main"

    label = "Dense Initialization"

    space = lf.ui.PanelSpace.MAIN_PANEL_TAB

    order = 21

    template = str(Path(__file__).resolve().with_name("densification.rml"))

    height_mode = lf.ui.PanelHeightMode.CONTENT

    update_interval_ms = 100

    update_policy = "interval"

    _ROMA_SETTINGS = ["high", "base", "fast", "turbo"]

    _ROMA_DESCRIPTIONS = {
        "high": "High: High quality, moderate speed (640px bidirectional)",
        "base": "Base: Balanced quality/speed (640px)",
        "fast": "Fast: Good quality, fast (512px) - Recommended",
        "turbo": "Turbo: Fastest, lower quality (320px)",
    }

    _MATCHES_STEP = 500

    _MAX_POINTS_STEP = 10000

    def __init__(self):
        self._handle = None
        self._doc = None

        self.job = None
        self.last_result = None
        self._pending_import = None
        self._pending_error = None
        self._cloud_update = None
        self._cancel_requested = False
        self._pending_start_training = False
        self._auto_import = True
        self._write_colmap = False
        self._start_training_when_complete = False

        self.debug_state = None

        self.config = DensePipelineConfig(output_path=self._get_temp_output_path())
        self._voxel_size_ui = 0.01  # remembered slider value when filter is toggled
        self._scrub_specs = dict(SCRUB_FIELD_SPECS)
        self._scrub_fields = ScrubFieldController(
            specs=self._scrub_specs,
            get_value=self._get_scrub_field_value,
            set_value=self._set_scrub_field_value,
        )

        self._target_point_cloud_uuid: Optional[str] = None
        self._base_point_cloud_points = None
        self._base_point_cloud_colors = None
        self._active_run_roi_only_selected: Optional[bool] = None
        self._preview_override_active = False

        self._collapsed = {"cameras", "filtering", "output"}

        # Track last-known state for dirty detection
        self._last_running = False
        self._last_progress = 0.0
        self._last_status = ""
        self._last_stage = ""
        self._last_has_result = False
        self._last_has_error = False
        self._last_camera_count = -1
        self._last_selected_camera_count = -1
        self._last_effective_camera_count = -1
        self._last_has_masks = False

    @staticmethod
    def _get_cache_dir() -> str:
        base = os.path.expanduser("~/.lichtfeld/cache")
        Path(base).mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def _get_temp_output_path() -> str:
        return os.path.join(DensificationPanel._get_cache_dir(), f"dense_{uuid.uuid4().hex}", "sparse", "0")

    @staticmethod
    def _cleanup_cache(max_age_seconds: float = 3600.0):
        cache_dir = DensificationPanel._get_cache_dir()
        now = time.time()

        for file_name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, file_name)
            try:
                if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
                elif os.path.isdir(path) and now - os.path.getmtime(path) > max_age_seconds:
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
