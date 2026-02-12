# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Panel using RoMa v2.

This panel allows users to densify a sparse point cloud using the cameras
already loaded in LichtFeld Studio. No additional file paths are needed -
simply load your scene, adjust parameters if desired, and click Start.
"""

import os
import random
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional, List

import numpy as np
from PIL import Image

import lichtfeld as lf
from lfs_plugins.types import Panel

from ..core.config import DensePipelineConfig
from ..core.debug_viz import MatchDebugState, MatchPreview


class DensifyStage(Enum):
    """Pipeline execution stage."""

    IDLE = "idle"
    LOADING = "loading"
    MATCHING = "matching"
    TRIANGULATING = "triangulating"
    WRITING = "writing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


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
            )

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


class DensificationPanel(Panel):
    """GUI panel for dense point cloud initialization workflow.
    
    This panel uses cameras already loaded in LichtFeld Studio.
    Simply load your scene, adjust parameters if desired, and click Start.
    The resulting dense point cloud will be automatically added to the scene.
    """

    label = "Dense Initialization"
    space = "MAIN_PANEL_TAB"
    order = 21

    def __init__(self):
        self.job = None
        self.last_result = None
        self._pending_import = None
        self._auto_import = True  # Auto-import result after completion

        self.debug_state = MatchDebugState()

        self.debug_enabled = False
        self.debug_auto_step = True

        self.config = DensePipelineConfig(output_path=self._get_temp_output_path())

        # Quality settings
        # Keep a UI index for the quality dropdown; the actual value is stored in self.config.roma_setting.
        self.roma_setting_idx = 3  # "fast" is default
        self.roma_settings = ["precise", "high", "base", "fast", "turbo"]
        self.roma_descriptions = [
            "Precise: Highest quality, slowest (800px bidirectional)",
            "High: High quality, moderate speed (640px bidirectional)", 
            "Base: Balanced quality/speed (640px)",
            "Fast: Good quality, fast (512px) - Recommended",
            "Turbo: Fastest, lower quality (320px)",
        ]

    def _get_temp_output_path(self) -> str:
        """Generate a temporary path for the output PLY file."""
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, "lfs_dense_init.ply")

    def _has_training_data(self) -> bool:
        """Check if the scene has training cameras loaded."""
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return cameras is not None and len(cameras) > 0
        except Exception:
            return False

    def _get_camera_count(self) -> int:
        """Get the number of cameras in the scene."""
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return len(cameras)
        except Exception:
            return 0

    def draw(self, layout):
        # Check for pending import (must happen on main thread)
        if self._pending_import:
            path = self._pending_import
            self._pending_import = None
            lf.log.info(f"Loading dense point cloud: {path}")
            self._import_ply(path)

        layout.heading("Dense Point Cloud Initialization")
        layout.label("Densify sparse reconstruction using RoMa v2 matching")
        layout.separator()

        # Scene status
        has_data = self._has_training_data()
        camera_count = self._get_camera_count() if has_data else 0

        if has_data:
            layout.label(f"Scene loaded with {camera_count} cameras")
        else:
            layout.text_colored("No scene loaded. Please load a dataset first.", (1.0, 0.6, 0.2, 1.0))
            layout.separator()
            layout.label("Load a COLMAP dataset or scene to enable densification.")
            return  # Don't show rest of UI if no scene

        layout.separator()

        # Quality settings
        if layout.collapsing_header("Quality Settings", default_open=True):
            _, self.roma_setting_idx = layout.combo("Matching Quality", self.roma_setting_idx, self.roma_settings)
            self.config.roma_setting = self.roma_settings[self.roma_setting_idx]
            layout.label(self.roma_descriptions[self.roma_setting_idx])

        # Advanced settings
        if layout.collapsing_header("Advanced Settings", default_open=False):
            layout.label("Reference View Selection:")
            _, self.config.num_refs = layout.drag_float("Reference Fraction", self.config.num_refs, 0.01, 0.1, 1.0)
            
            _, self.config.nns_per_ref = layout.drag_int("Neighbors per Ref", self.config.nns_per_ref, 1, 1, 10)

            layout.separator()
            layout.label("Matching Parameters:")
            
            _, self.config.matches_per_ref = layout.drag_int("Matches per Ref", self.config.matches_per_ref, 100, 1000, 50000)
            
            _, self.config.certainty_thresh = layout.drag_float("Min Certainty", self.config.certainty_thresh, 0.01, 0.0, 1.0)

            layout.separator()
            layout.label("Geometric Filtering:")
            
            _, self.config.reproj_thresh = layout.drag_float("Max Reproj Error (px)", self.config.reproj_thresh, 0.1, 0.1, 10.0)
            
            _, self.config.sampson_thresh = layout.drag_float("Max Sampson Error", self.config.sampson_thresh, 0.5, 0.0, 50.0)
            
            _, self.config.min_parallax_deg = layout.drag_float("Min Parallax (deg)", self.config.min_parallax_deg, 0.1, 0.0, 10.0)

            layout.separator()
            layout.label("Output Options:")
            
            _, self.config.max_points = layout.drag_int("Max Points (0=unlimited)", self.config.max_points, 1000, 0, 10000000)
            
            _, self.config.no_filter = layout.checkbox("Disable Filtering", self.config.no_filter)

            layout.separator()
            layout.label("Live Preview:")
            
            _, self.config.viz_interval = layout.drag_int("Update Every N Pairs", self.config.viz_interval, 1, 0, 100)
            layout.label("(0 = only show final result)")

            layout.separator()
            layout.label("Debugging:")
            changed_debug, self.debug_enabled = layout.checkbox("Debug matches (floating window)", self.debug_enabled)
            if changed_debug:
                self.debug_state.set_enabled(self.debug_enabled)
                if not self.debug_enabled:
                    # Ensure the pipeline is unblocked
                    self.debug_state.set_auto_step(True)
                    self.debug_state.release_waiters()

            changed_auto, self.debug_auto_step = layout.checkbox("Auto-step", self.debug_auto_step)
            if changed_auto:
                self.debug_state.set_auto_step(self.debug_auto_step)
                if self.debug_auto_step:
                    self.debug_state.release_waiters()

            if not self.debug_auto_step:
                layout.text_disabled("Manual stepping: use 'Next pair' in the debug window.")

        layout.separator()

        # Auto-import option
        _, self._auto_import = layout.checkbox("Auto-add to scene", self._auto_import)

        layout.separator()

        # Job status / start button
        if self.job and self.job.is_running():
            stage = self.job.stage.value
            progress = self.job.progress

            layout.label(f"Stage: {stage.capitalize()}")
            layout.progress_bar(progress / 100.0, self.job.status)

            if layout.button("Cancel"):
                self.job.cancel()
        else:
            if layout.button("Start Densification", (0, 36)):
                self._start()

        # Results display
        if self.last_result and self.last_result.success:
            layout.separator()
            layout.heading("Results")
            layout.label(f"Points generated: {self.last_result.num_points:,}")
            layout.label(f"Time: {self.last_result.elapsed_time:.1f}s")

            if self.last_result.output_path:
                if not self._auto_import:
                    if layout.button("Add to Scene", (0, 36)):
                        self._import_ply(self.last_result.output_path)

        if self.last_result and not self.last_result.success:
            layout.separator()
            layout.text_colored("Error:", (1.0, 0.3, 0.3, 1.0))
            layout.text_selectable(self.last_result.error or "Unknown error", 60)

        # Floating debug window (if enabled)
        self._draw_debug_window(layout)

    def _draw_debug_window(self, layout):
        if not self.debug_state.is_enabled():
            return

        layout.set_next_window_size((960, 560), first_use=True)
        visible, still_open = layout.begin_window_closable("Dense Match Debug", flags=0)
        try:
            if not visible:
                if not still_open:
                    self.debug_enabled = False
                    self.debug_state.set_enabled(False)
                return

            layout.label("Live match previews")
            changed_auto, auto_val = layout.checkbox("Auto-step", self.debug_state.is_auto_step())
            if changed_auto:
                self.debug_auto_step = auto_val
                self.debug_state.set_auto_step(auto_val)
                if auto_val:
                    self.debug_state.release_waiters()

            if not self.debug_state.is_auto_step():
                if layout.button("Next pair", (120, 0)):
                    self.debug_state.step_once()

            layout.separator()

            # --- Match visibility controls ---
            cur_max = self.debug_state.max_visible_matches()
            _, new_max = layout.drag_int("Visible matches (0=all)", cur_max, 1, 0, 5000)
            if new_max != cur_max:
                self.debug_state.set_max_visible_matches(new_max)

            single = self.debug_state.is_single_match_mode()
            changed_single, single_val = layout.checkbox("Single match mode", single)
            if changed_single:
                self.debug_state.set_single_match_mode(single_val)

            preview = self.debug_state.latest()

            if single_val and preview is not None:
                total_m = preview.matches.shape[0] if preview.matches is not None else 0
                cur_idx = self.debug_state.current_match_index()
                layout.label(f"Match {min(cur_idx + 1, total_m)}/{total_m}")
                with layout.row() as row:
                    if row.button("<< Prev", (80, 0)):
                        self.debug_state.prev_match(total_m)
                    if row.button("Next >>", (80, 0)):
                        self.debug_state.next_match(total_m)
                _, idx_val = layout.drag_int("Match index", cur_idx, 1, 0, max(0, total_m - 1))
                if idx_val != cur_idx:
                    self.debug_state.set_current_match_index(idx_val)

            layout.separator()
            total_pairs = max(self.debug_state.total_pairs(), 1)
            if preview:
                layout.label(f"Pair {preview.pair_index}/{preview.total_pairs or total_pairs}")
                layout.label(f"{preview.ref_label} ↔ {preview.nbr_label}")
                layout.label(f"Total matches: {preview.match_count}")
                self._draw_match_preview(layout, preview, pair_index=int(preview.pair_index))
            else:
                layout.text_disabled("Waiting for matches...")
        finally:
            layout.end_window()

    _DEBUG_PREVIEW_DRAW_SIZE = (512, 512)
    _DEBUG_PREVIEW_LINE_THICKNESS = 0.1

    def _draw_match_preview(self, layout, preview: MatchPreview, *, pair_index: int):
        left = preview.left_image
        right = preview.right_image
        matches = preview.matches

        if (
            left is None
            or right is None
            or matches is None
            or left.size == 0
            or right.size == 0
            or len(matches) == 0
        ):
            layout.text_disabled("No preview available")
            return

        disp_w, disp_h = self._DEBUG_PREVIEW_DRAW_SIZE

        # Prepare images for display
        img_left = np.asarray(
            Image.fromarray(left).resize((disp_w, disp_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        img_right = np.asarray(
            Image.fromarray(right).resize((disp_w, disp_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        tensor_left = lf.Tensor.from_numpy(img_left)
        tensor_right = lf.Tensor.from_numpy(img_right)

        # Source (match) resolution
        src_h, src_w = left.shape[:2]
        sx = disp_w / float(max(1, src_w))
        sy = disp_h / float(max(1, src_h))

        layout.new_line()
        # Draw images and capture exact anchors
        base_x, base_y = layout.get_cursor_screen_pos()

        left_anchor = None
        right_anchor = None
        with layout.row() as row:
            try:
                left_anchor = row.get_cursor_screen_pos()
            except Exception:
                left_anchor = None

            row.image_tensor("left", tensor_left, (disp_w, disp_h))

            # Cursor now points to where the NEXT widget will go (i.e., the right image start)
            try:
                right_anchor = row.get_cursor_screen_pos()
            except Exception:
                right_anchor = None

            row.image_tensor("right", tensor_right, (disp_w, disp_h))

        left_x, left_y = left_anchor if left_anchor is not None else (base_x, base_y)
        if right_anchor is not None:
            right_x, right_y = right_anchor
        else:
            # Fallback if SubLayout doesn't expose get_cursor_screen_pos
            right_x, right_y = base_x + disp_w, base_y

        # Draw lines. IMPORTANT: matches are already pixel coords in MATCH resolution.
        thickness = float(self._DEBUG_PREVIEW_LINE_THICKNESS)
        total_matches = int(matches.shape[0])
        visible_indices = self.debug_state.visible_match_indices(total_matches)
        if not visible_indices:
            layout.text_disabled("No matches to display")
            return

        for idx in visible_indices:
            xa, ya, xb, yb = matches[idx]
            x1 = left_x + float(xa) * sx
            y1 = left_y + float(ya) * sy
            x2 = right_x + float(xb) * sx
            y2 = right_y + float(yb) * sy

            seed = (int(pair_index) * 1000003) ^ (int(idx) * 9176)
            rng = random.Random(seed & 0xFFFFFFFF)
            color = (
                0.2 + 0.8 * rng.random(),
                0.2 + 0.8 * rng.random(),
                0.2 + 0.8 * rng.random(),
                1.0,
            )
            layout.draw_line(x1, y1, x2, y2, color, thickness)





    def _start(self):
        if not self._has_training_data():
            lf.log.warn("No training cameras found in scene")
            self.last_result = DensifyResult(
                success=False,
                error="No training cameras found. Please load a dataset first."
            )
            return

        self.last_result = None

        # Sync debug controller with current UI settings before launching job
        self.debug_state.set_enabled(self.debug_enabled)
        self.debug_state.set_auto_step(self.debug_auto_step)
        self.debug_state.release_waiters()

        # Snapshot the current config for the background job.
        # (Avoids mutations from the UI while the job is running.)
        config = replace(
            self.config,
            output_path=self._get_temp_output_path(),
            roma_setting=self.roma_settings[self.roma_setting_idx],
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
        """Handle periodic visualization of intermediate PLY files during processing."""
        # Schedule the import to happen on the main thread
        self._pending_import = ply_path

    def _on_complete(self, result: DensifyResult):
        lf.log.info(f"Densification complete: {result.num_points:,} points")
        self.last_result = result
        
        # Auto-import if enabled
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

            # Find the first POINTCLOUD node in the scene
            target = None
            for n in scene.get_nodes():
                if n.type == lf.scene.NodeType.POINTCLOUD:
                    target = n
                    break

            if not target:
                lf.log.error("No point cloud node found to replace")
                return

            # Load raw point cloud data (positions + colors)
            means, colors = lf.io.load_point_cloud(ply_path)

            lf.log.debug(
                f"Loaded PLY: {means.shape[0]:,} points, "
                f"bounds=[{means.min(0).numpy()}, {means.max(0).numpy()}]"
            )

            pc = target.point_cloud()
            if not pc:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return

            # Replace data in-place
            pc.set_data(means, colors)

            lf.log.debug(
                f"Replaced '{target.name}' with dense cloud "
                f"({means.shape[0]:,} points)"
            )

        except Exception as e:
            lf.log.error(f"Failed to import PLY: {e}")

