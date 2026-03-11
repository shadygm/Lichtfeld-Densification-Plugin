# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Panel using RoMa v2.

This panel allows users to densify a sparse point cloud using the cameras
already loaded in LichtFeld Studio. No additional file paths are needed -
simply load your scene, adjust parameters if desired, and click Start.
"""

import os
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List, ClassVar

import lichtfeld as lf

from ..core.config import DensePipelineConfig
from ..core.debug_viz import MatchDebugState
from .debug_matches import DebugMatchesPanel


class DensifyStage(Enum):
    """Pipeline execution stage."""

    IDLE = "Idle"
    LOADING = "Loading"
    MATCHING = "Matching"
    TRIANGULATING = "Triangulating"
    WRITING = "Writing"
    DONE = "Done"
    ERROR = "Error"
    CANCELLED = "Cancelled"


@dataclass
class DensifyResult:
    """Result of dense initialization."""

    success: bool
    output_path: Optional[str] = None
    num_points: int = 0
    elapsed_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class DensifyJob:
    """Background densification job with progress tracking.
    
    Uses cameras from LichtFeld Studio directly - no file paths needed.
    """
    _instances: ClassVar[weakref.WeakSet["DensifyJob"]] = weakref.WeakSet()
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: DensePipelineConfig,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_complete: Optional[Callable[[DensifyResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_sequential_viz: Optional[Callable[[str], None]] = None,
        debug_state: Optional[MatchDebugState] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_sequential_viz = on_sequential_viz
        self.debug_state = debug_state

        self._stage = DensifyStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._cancelled = False
        self._result: Optional[DensifyResult] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        with self._instances_lock:
            self._instances.add(self)

    @classmethod
    def cancel_all(cls, timeout: float = 5.0):
        with cls._instances_lock:
            jobs = list(cls._instances)

        for job in jobs:
            try:
                job.cancel()
            except Exception:
                pass

        deadline = time.time() + max(0.0, float(timeout))
        for job in jobs:
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break
            try:
                job.wait(timeout=remaining)
            except Exception:
                pass

    @property
    def stage(self) -> DensifyStage:
        with self._lock:
            return self._stage

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[DensifyResult]:
        with self._lock:
            return self._result

    def is_running(self) -> bool:
        return self.stage in (
            DensifyStage.LOADING,
            DensifyStage.MATCHING,
            DensifyStage.TRIANGULATING,
            DensifyStage.WRITING,
        )

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._status = "Cancelling..."
            if self.debug_state:
                self.debug_state.release_waiters()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Job already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> Optional[DensifyResult]:
        if self._thread:
            self._thread.join(timeout)
        return self._result

    def _update(self, stage: DensifyStage, progress: float, status: str):
        with self._lock:
            self._stage = stage
            self._progress = progress
            self._status = status

        if self.on_progress:
            self.on_progress(stage.value, progress, status)

    def _run(self):
        import time
        t0 = time.time()

        try:
            def check_cancelled():
                with self._lock:
                    return self._cancelled

            self._update(DensifyStage.LOADING, 5.0, "Loading cameras from scene...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 0.0, "Cancelled")
                return

            # Get cameras from scene graph (each camera is a node with has_camera=True)
            scene = lf.get_scene()
            camera_nodes = [n for n in scene.get_nodes() if n.has_camera]
            if not camera_nodes:
                raise RuntimeError("No cameras found in scene. Please load a dataset first.")
            
            lf.log.info(f"Found {len(camera_nodes)} cameras in scene")

            self._update(DensifyStage.MATCHING, 10.0, "Initializing RoMa v2...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 10.0, "Cancelled")
                return

            # Import and run the LFS-based densify pipeline
            from ..densify import dense_init_from_lfs

            def progress_cb(pct: float, msg: str):
                if check_cancelled():
                    return
                
                stage = DensifyStage.MATCHING
                if pct >= 95.0:
                    stage = DensifyStage.WRITING
                elif pct >= 90.0 or "triangula" in msg.lower():
                    stage = DensifyStage.TRIANGULATING
                
                self._update(stage, pct, msg)

            # Run the dense initialization
            result_code, result_info = dense_init_from_lfs(
                camera_nodes,
                self.config,
                progress_callback=progress_cb,
                on_sequential_viz=self.on_sequential_viz,
                debug_state=self.debug_state,
                cancel_requested=check_cancelled,
            )

            if result_code == 2 or check_cancelled():
                with self._lock:
                    self._result = DensifyResult(success=False, error="Cancelled")
                self._update(DensifyStage.CANCELLED, self._progress, "Cancelled")
                return

            if result_code != 0:
                raise RuntimeError(result_info or "Densification failed")

            elapsed = time.time() - t0
            output_path = result_info  # On success, this is the output path

            # Count points from the output file (read PLY header)
            num_points = 0
            if output_path and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    for line in f:
                        line = line.decode("ascii", errors="ignore").strip()
                        if line.startswith("element vertex"):
                            num_points = int(line.split()[-1])
                            break
                        if line == "end_header":
                            break

            result = DensifyResult(
                success=True,
                output_path=output_path,
                num_points=num_points,
                elapsed_time=elapsed,
            )

            with self._lock:
                self._result = result

            self._update(DensifyStage.DONE, 100.0, "Complete")
            lf.log.info(f"Densification complete: {num_points:,} points in {elapsed:.1f}s")

            if self.on_complete:
                self.on_complete(result)

        except Exception as e:
            lf.log.error(f"Densification error: {e}")
            self._update(DensifyStage.ERROR, self._progress, str(e))
            with self._lock:
                self._result = DensifyResult(success=False, error=str(e))

            if self.on_error:
                self.on_error(e)

        finally:
            if self.debug_state:
                self.debug_state.release_waiters()


class DensificationPanel(lf.ui.Panel):
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
        self._auto_import = True

        self.debug_state = MatchDebugState()
        DebugMatchesPanel.set_debug_state(self.debug_state)
        self._debug_enabled = False
        self._debug_auto_step = True

        self.config = DensePipelineConfig(output_path=self._get_temp_output_path())
        self._voxel_size_ui = 0.01  # remembered slider value when filter is toggled

        self._collapsed = {"cameras", "filtering", "output", "debug"}

        # Track last-known state for dirty detection
        self._last_running = False
        self._last_progress = 0.0
        self._last_status = ""
        self._last_stage = ""
        self._last_has_result = False
        self._last_has_error = False
        self._last_camera_count = -1
        self._last_has_masks = False

    # ── Helpers ──────────────────────────────────────────

    @staticmethod
    def _get_temp_output_path() -> str:
        return os.path.join(tempfile.gettempdir(), "lfs_dense_init.ply")

    def _has_training_data(self) -> bool:
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return cameras is not None and len(cameras) > 0
        except Exception:
            return False

    def _has_masks(self) -> bool:
        try:
            scene = lf.get_scene()
            for n in scene.get_nodes():
                if n.has_camera and getattr(n, "has_mask", False):
                    return True
            return False
        except Exception:
            return False

    def _get_camera_count(self) -> int:
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return len(cameras)
        except Exception:
            return 0

    def _dirty(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for f in fields:
            self._handle.dirty(f)

    # ── Retained lifecycle ───────────────────────────────

    def on_mount(self, doc):
        self._doc = doc
        self._sync_section_states()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("densification")
        if model is None:
            return

        # --- Scene state (read-only) ---
        model.bind_func("has_scene", self._has_training_data)
        model.bind_func("camera_count_text",
                        lambda: f"Scene loaded with {self._get_camera_count()} cameras")

        # --- Quality setting ---
        model.bind("roma_setting",
                    lambda: self.config.roma_setting,
                    self._set_roma_setting)
        model.bind_func("roma_description",
                        lambda: self._ROMA_DESCRIPTIONS.get(self.config.roma_setting, ""))
        model.bind_func("has_masks", self._has_masks)
        model.bind("use_masks",
                    lambda: self.config.use_masks,
                    self._set_use_masks)

        # --- Slider-bound config values ---
        model.bind("num_refs",
                    lambda: f"{self.config.num_refs:.2f}",
                    lambda v: self._set_float_config("num_refs", v, 0.1, 1.0))
        model.bind("nns_per_ref",
                    lambda: str(self.config.nns_per_ref),
                    lambda v: self._set_int_config("nns_per_ref", v, 1, 10))
        model.bind("certainty_thresh",
                    lambda: f"{self.config.certainty_thresh:.2f}",
                    lambda v: self._set_float_config("certainty_thresh", v, 0.0, 1.0))
        model.bind("reproj_thresh",
                    lambda: f"{self.config.reproj_thresh:.1f}",
                    lambda v: self._set_float_config("reproj_thresh", v, 0.1, 5.0))
        model.bind("sampson_thresh",
                    lambda: f"{self.config.sampson_thresh:.1f}",
                    lambda v: self._set_float_config("sampson_thresh", v, 0.0, 10.0))
        model.bind("min_parallax_deg",
                    lambda: f"{self.config.min_parallax_deg:.1f}",
                    lambda v: self._set_float_config("min_parallax_deg", v, 0.0, 5.0))
        model.bind("viz_interval",
                    lambda: str(self.config.viz_interval),
                    lambda v: self._set_int_config("viz_interval", v, 0, 10))

        # --- Number-input config values ---
        model.bind("matches_per_ref_str",
                    lambda: str(self.config.matches_per_ref),
                    lambda v: self._set_int_config("matches_per_ref", v, 1000, 15000))
        model.bind("max_points_str",
                    lambda: str(self.config.max_points),
                    lambda v: self._set_int_config("max_points", v, 0, 10000000))

        # --- Distance filter ---
        model.bind("distance_filter_enabled",
                    lambda: self.config.voxel_size > 0.0,
                    self._set_distance_filter_enabled)
        model.bind("voxel_size",
                    lambda: f"{self._voxel_size_ui:.3f}",
                    lambda v: self._set_voxel_size(v))

        # --- Debug controls ---
        model.bind("debug_enabled",
                    lambda: self._debug_enabled,
                    self._set_debug_enabled)
        model.bind("debug_auto_step",
                    lambda: self._debug_auto_step,
                    self._set_debug_auto_step)

        # --- Job state (read-only) ---
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text",
                        lambda: self.job.stage.value.capitalize() if self.job else "Idle")
        model.bind_func("progress_value",
                        lambda: f"{max(0.0, min(1.0, self.job.progress / 100.0)):.4f}"
                        if self.job else "0")
        model.bind_func("progress_pct",
                        lambda: f"{int(self.job.progress)}%"
                        if self.job else "0%")
        model.bind_func("progress_status",
                        lambda: self.job.status if self.job else "")

        # --- Result state ---
        model.bind_func("show_results",
                        lambda: self.last_result is not None and self.last_result.success)
        model.bind_func("result_points",
                        lambda: f"{self.last_result.num_points:,}"
                        if self.last_result and self.last_result.success else "0")
        model.bind_func("result_time",
                        lambda: f"{self.last_result.elapsed_time:.1f}s"
                        if self.last_result and self.last_result.success else "")
        model.bind_func("show_error",
                        lambda: self.last_result is not None and not self.last_result.success)
        model.bind_func("error_text",
                        lambda: self.last_result.error or "Unknown error"
                        if self.last_result and not self.last_result.success else "")

        # --- Events ---
        model.bind_event("do_start", self._on_do_start)
        model.bind_event("do_cancel", self._on_do_cancel)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("num_step", self._on_num_step)

        self._handle = model.get_handle()

    def on_update(self, doc):
        # Handle pending import on main thread
        if self._pending_import:
            path = self._pending_import
            self._pending_import = None
            lf.log.info(f"Loading dense point cloud: {path}")
            self._import_ply(path)

        dirty = False

        # Track running state changes
        running = self._is_running()
        if running != self._last_running:
            self._last_running = running
            self._dirty("show_idle", "show_running")
            dirty = True

        if running and self.job:
            progress = self.job.progress
            status = self.job.status
            stage = self.job.stage.value
            if (progress != self._last_progress or
                    status != self._last_status or
                    stage != self._last_stage):
                self._last_progress = progress
                self._last_status = status
                self._last_stage = stage
                self._dirty("stage_text", "progress_value", "progress_pct", "progress_status")
                dirty = True

        # Track result changes
        has_result = self.last_result is not None and self.last_result.success
        has_error = self.last_result is not None and not self.last_result.success
        if has_result != self._last_has_result or has_error != self._last_has_error:
            self._last_has_result = has_result
            self._last_has_error = has_error
            self._dirty("show_results", "result_points", "result_time",
                        "show_error", "error_text",
                        "show_idle", "show_running")
            dirty = True

        # Track camera count changes
        cam_count = self._get_camera_count()
        if cam_count != self._last_camera_count:
            self._last_camera_count = cam_count
            self._dirty("has_scene", "camera_count_text")
            dirty = True

        # Track mask availability changes
        has_masks = self._has_masks()
        if has_masks != self._last_has_masks:
            self._last_has_masks = has_masks
            if not has_masks:
                self.config.use_masks = True
            self._dirty("has_masks", "use_masks")
            dirty = True

        return dirty

    def on_scene_changed(self, doc):
        self._last_camera_count = -1
        self._last_has_masks = not self._has_masks()  # force redirty next update
        if self._handle:
            self._dirty("has_scene", "camera_count_text", "has_masks", "use_masks")

    def on_unmount(self, doc):
        doc.remove_data_model("densification")
        self._handle = None
        self._doc = None

    # ── Section toggle ───────────────────────────────────

    def _get_section_elements(self, name):
        if not self._doc:
            return None, None, None
        header = self._doc.get_element_by_id(f"hdr-{name}")
        arrow = self._doc.get_element_by_id(f"arrow-{name}")
        content = self._doc.get_element_by_id(f"sec-{name}")
        return header, arrow, content

    def _sync_section_states(self):
        for name in ("matching", "cameras", "filtering", "output", "debug"):
            header, arrow, content = self._get_section_elements(name)
            if content:
                expanded = name not in self._collapsed
                if expanded:
                    content.set_class("collapsed", False)
                else:
                    content.set_class("collapsed", True)
                if arrow:
                    arrow.set_class("is-expanded", expanded)
                if header:
                    header.set_class("is-expanded", expanded)

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        header, arrow, content = self._get_section_elements(name)
        if content:
            content.set_class("collapsed", not expanding)
        if arrow:
            arrow.set_class("is-expanded", expanding)
        if header:
            header.set_class("is-expanded", expanding)

    # ── Config setters ───────────────────────────────────

    def _set_roma_setting(self, value):
        value = str(value)
        if value in self._ROMA_SETTINGS and value != self.config.roma_setting:
            self.config.roma_setting = value
            self._dirty("roma_setting", "roma_description")

    def _set_use_masks(self, value):
        v = bool(value)
        if v != self.config.use_masks:
            self.config.use_masks = v
            self._dirty("use_masks")

    def _set_float_config(self, attr, value, vmin, vmax):
        try:
            v = max(vmin, min(vmax, float(value)))
        except (TypeError, ValueError):
            return
        if abs(v - getattr(self.config, attr)) < 1e-9:
            return
        setattr(self.config, attr, v)
        self._dirty(attr)

    def _set_int_config(self, attr, value, vmin, vmax):
        try:
            v = max(vmin, min(vmax, int(float(value))))
        except (TypeError, ValueError):
            return
        if v == getattr(self.config, attr):
            return
        setattr(self.config, attr, v)
        self._dirty(attr)

    def _set_debug_enabled(self, value):
        self._debug_enabled = bool(value)
        self.debug_state.set_enabled(self._debug_enabled)
        if not self._debug_enabled:
            self.debug_state.set_auto_step(True)
            self.debug_state.release_waiters()
        # Share state and toggle floating debug panel
        DebugMatchesPanel.set_debug_state(self.debug_state)
        lf.ui.set_panel_enabled(DebugMatchesPanel.id, self._debug_enabled)
        self._dirty("debug_enabled")

    def _set_distance_filter_enabled(self, value):
        enabled = bool(value)
        if enabled:
            self.config.voxel_size = self._voxel_size_ui
        else:
            self.config.voxel_size = 0.0
        self._dirty("distance_filter_enabled", "voxel_size")

    def _set_voxel_size(self, value):
        try:
            v = max(0.001, min(0.1, float(value)))
        except (TypeError, ValueError):
            return
        if abs(v - self._voxel_size_ui) < 1e-6:
            return
        self._voxel_size_ui = v
        self.config.voxel_size = v
        self._dirty("voxel_size")

    def _set_debug_auto_step(self, value):
        self._debug_auto_step = bool(value)
        self.debug_state.set_auto_step(self._debug_auto_step)
        if self._debug_auto_step:
            self.debug_state.release_waiters()
        self._dirty("debug_auto_step")

    def _on_num_step(self, handle, event, args):
        if not args or len(args) < 2:
            return
        field_name = str(args[0])
        direction = int(args[1])

        step_map = {
            "matches_per_ref": self._MATCHES_STEP,
            "max_points": self._MAX_POINTS_STEP,
        }
        step = step_map.get(field_name, 1)
        current = getattr(self.config, field_name, 0)
        new_val = current + direction * step

        range_map = {
            "matches_per_ref": (1000, 15000),
            "max_points": (0, 10000000),
        }
        vmin, vmax = range_map.get(field_name, (0, 999999999))
        new_val = max(vmin, min(vmax, new_val))

        if new_val != current:
            setattr(self.config, field_name, new_val)
            self._dirty(f"{field_name}_str")

    # ── Job control ──────────────────────────────────────

    def _is_running(self) -> bool:
        return self.job is not None and self.job.is_running()

    def _on_do_start(self, handle, event, args):
        self._start()

    def _on_do_cancel(self, handle, event, args):
        if self.job:
            self.job.cancel()

    def _start(self):
        if not self._has_training_data():
            lf.log.warn("No training cameras found in scene")
            self.last_result = DensifyResult(
                success=False,
                error="No training cameras found. Please load a dataset first."
            )
            return

        self.last_result = None

        self.debug_state.set_enabled(self._debug_enabled)
        self.debug_state.set_auto_step(self._debug_auto_step)
        self.debug_state.release_waiters()

        config = replace(
            self.config,
            output_path=self._get_temp_output_path(),
        )

        self.job = DensifyJob(
            config=config,
            on_complete=self._on_complete,
            on_error=self._on_error,
            on_sequential_viz=self._on_sequential_viz,
            debug_state=self.debug_state,
        )
        self.job.start()

    def _on_sequential_viz(self, ply_path: str):
        self._pending_import = ply_path

    def _on_complete(self, result: DensifyResult):
        lf.log.info(f"Densification complete: {result.num_points:,} points")
        self.last_result = result
        if self._auto_import and result.output_path:
            self._pending_import = result.output_path

    def _on_error(self, error: Exception):
        lf.log.error(f"Densification failed: {error}")
        self.last_result = DensifyResult(success=False, error=str(error))

    def _import_ply(self, ply_path: str):
        """Replace the existing point cloud node with the dense PLY data."""
        if not ply_path or not os.path.exists(ply_path):
            lf.log.warn(f"PLY file not found: {ply_path}")
            return

        try:
            lf.log.debug(f"Loading PLY: {ply_path}")

            scene = lf.get_scene()
            if scene is None:
                lf.log.error("No scene available")
                return

            target = None
            for n in scene.get_nodes():
                if n.type == lf.scene.NodeType.POINTCLOUD:
                    target = n
                    break

            if not target:
                lf.log.error("No point cloud node found to replace")
                return

            means, colors = lf.io.load_point_cloud(ply_path)

            lf.log.debug(
                f"Loaded PLY: {means.shape[0]:,} points, "
                f"bounds=[{means.min(0).numpy()}, {means.max(0).numpy()}]"
            )

            pc = target.point_cloud()
            if not pc:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return

            pc.set_data(means, colors)

            lf.log.debug(
                f"Replaced '{target.name}' with dense cloud "
                f"({means.shape[0]:,} points)"
            )

        except Exception as e:
            lf.log.error(f"Failed to import PLY: {e}")
