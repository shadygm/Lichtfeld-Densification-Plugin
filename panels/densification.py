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
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, List

import lichtfeld as lf
from lfs_plugins.types import Panel


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
class DensifyConfig:
    """Configuration for dense initialization pipeline."""

    output_path: str  # Path for output PLY file
    roma_setting: str = "fast"
    num_refs: float = 0.75
    nns_per_ref: int = 4
    matches_per_ref: int = 12000
    certainty_thresh: float = 0.20
    reproj_thresh: float = 1.5
    sampson_thresh: float = 5.0
    min_parallax_deg: float = 0.5
    max_points: int = 0
    no_filter: bool = False
    seed: int = 0


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
        config: DensifyConfig,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_complete: Optional[Callable[[DensifyResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

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
            from ..densify import dense_init_from_lfs, LFSDenseConfig

            lfs_config = LFSDenseConfig(
                output_path=self.config.output_path,
                roma_setting=self.config.roma_setting,
                num_refs=self.config.num_refs,
                nns_per_ref=self.config.nns_per_ref,
                matches_per_ref=self.config.matches_per_ref,
                certainty_thresh=self.config.certainty_thresh,
                reproj_thresh=self.config.reproj_thresh,
                sampson_thresh=self.config.sampson_thresh,
                min_parallax_deg=self.config.min_parallax_deg,
                max_points=self.config.max_points,
                no_filter=self.config.no_filter,
                seed=self.config.seed,
            )

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
                camera_nodes, lfs_config, progress_callback=progress_cb
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


class DensificationPanel(Panel):
    """GUI panel for dense point cloud initialization workflow.
    
    This panel uses cameras already loaded in LichtFeld Studio.
    Simply load your scene, adjust parameters if desired, and click Start.
    The resulting dense point cloud will be automatically added to the scene.
    """

    label = "Dense Initialization"
    space = "MAIN_PANEL_TAB"
    order = 5

    def __init__(self):
        self.job = None
        self.last_result = None
        self._pending_import = None
        self._auto_import = True  # Auto-import result after completion

        # Quality settings
        self.roma_setting_idx = 3  # "fast" is default
        self.roma_settings = ["precise", "high", "base", "fast", "turbo"]
        self.roma_descriptions = [
            "Precise: Highest quality, slowest (800px bidirectional)",
            "High: High quality, moderate speed (640px bidirectional)", 
            "Base: Balanced quality/speed (640px)",
            "Fast: Good quality, fast (512px) - Recommended",
            "Turbo: Fastest, lower quality (320px)",
        ]

        # Advanced settings with reasonable defaults
        self.num_refs = 0.75
        self.nns_per_ref = 4
        self.matches_per_ref = 12000
        self.certainty_thresh = 0.20
        self.reproj_thresh = 1.5
        self.sampson_thresh = 5.0
        self.min_parallax_deg = 0.5
        self.max_points = 0
        self.no_filter = False

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
            _, self.roma_setting_idx = layout.combo(
                "Matching Quality", self.roma_setting_idx, self.roma_settings
            )
            layout.label(self.roma_descriptions[self.roma_setting_idx])

        # Advanced settings
        if layout.collapsing_header("Advanced Settings", default_open=False):
            layout.label("Reference View Selection:")
            _, self.num_refs = layout.drag_float(
                "Reference Fraction", self.num_refs, 0.01, 0.1, 1.0
            )
            
            _, self.nns_per_ref = layout.drag_int(
                "Neighbors per Ref", self.nns_per_ref, 1, 1, 10
            )

            layout.separator()
            layout.label("Matching Parameters:")
            
            _, self.matches_per_ref = layout.drag_int(
                "Matches per Ref", self.matches_per_ref, 100, 1000, 50000
            )
            
            _, self.certainty_thresh = layout.drag_float(
                "Min Certainty", self.certainty_thresh, 0.01, 0.0, 1.0
            )

            layout.separator()
            layout.label("Geometric Filtering:")
            
            _, self.reproj_thresh = layout.drag_float(
                "Max Reproj Error (px)", self.reproj_thresh, 0.1, 0.1, 10.0
            )
            
            _, self.sampson_thresh = layout.drag_float(
                "Max Sampson Error", self.sampson_thresh, 0.5, 0.0, 50.0
            )
            
            _, self.min_parallax_deg = layout.drag_float(
                "Min Parallax (deg)", self.min_parallax_deg, 0.1, 0.0, 10.0
            )

            layout.separator()
            layout.label("Output Options:")
            
            _, self.max_points = layout.drag_int(
                "Max Points (0=unlimited)", self.max_points, 1000, 0, 10000000
            )
            
            _, self.no_filter = layout.checkbox("Disable Filtering", self.no_filter)

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

    def _start(self):
        if not self._has_training_data():
            lf.log.warn("No training cameras found in scene")
            self.last_result = DensifyResult(
                success=False,
                error="No training cameras found. Please load a dataset first."
            )
            return

        self.last_result = None

        config = DensifyConfig(
            output_path=self._get_temp_output_path(),
            roma_setting=self.roma_settings[self.roma_setting_idx],
            num_refs=self.num_refs,
            nns_per_ref=self.nns_per_ref,
            matches_per_ref=self.matches_per_ref,
            certainty_thresh=self.certainty_thresh,
            reproj_thresh=self.reproj_thresh,
            sampson_thresh=self.sampson_thresh,
            min_parallax_deg=self.min_parallax_deg,
            max_points=self.max_points,
            no_filter=self.no_filter,
        )

        self.job = DensifyJob(
            config=config,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )
        self.job.start()

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
            lf.log.info(f"Loading PLY: {ply_path}")

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

            lf.log.info(
                f"Loaded PLY: {means.shape[0]:,} points, "
                f"bounds=[{means.min(0).numpy()}, {means.max(0).numpy()}]"
            )

            pc = target.point_cloud()
            if not pc:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return

            # Replace data in-place
            pc.set_data(means, colors)

            lf.log.info(
                f"Replaced '{target.name}' with dense cloud "
                f"({means.shape[0]:,} points)"
            )

        except Exception as e:
            lf.log.error(f"Failed to import PLY: {e}")

