# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Panel using RoMa v2."""

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List

import lichtfeld as lf


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

    scene_path: Path
    images_subdir: str = "images"
    out_name: str = "points3D_dense.bin"
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

    def __post_init__(self):
        self.scene_path = Path(self.scene_path)


@dataclass
class DensifyResult:
    """Result of dense initialization."""

    success: bool
    output_path: Optional[Path] = None
    scene_path: Optional[Path] = None
    num_points: int = 0
    elapsed_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class DensifyJob:
    """Background densification job with progress tracking."""

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

            self._update(DensifyStage.LOADING, 5.0, "Loading reconstruction...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 0.0, "Cancelled")
                return

            # Build args namespace to match densify.py expectations
            from argparse import Namespace
            args = Namespace(
                scene_root=str(self.config.scene_path),
                images_subdir=self.config.images_subdir,
                out_name=self.config.out_name,
                roma_model="outdoor",  # Kept for compatibility
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
                viz=False,
                seed=self.config.seed,
            )

            self._update(DensifyStage.MATCHING, 10.0, "Initializing the System...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 10.0, "Cancelled")
                return

            # Import and run the densify pipeline
            from ..densify import dense_init

            def progress_cb(pct: float, msg: str):
                if check_cancelled():
                    # raising exception to abort dense_init might be abrupt, 
                    # but dense_init doesn't have a cancel check. 
                    # For now we just update status and let it finish or wait 
                    # for the next check points in dense_init if we added them.
                    # Since we didn't add cancellation hooks in dense_init loops,
                    # we just update the UI to show cancellation is pending or ignored.
                    return
                
                stage = DensifyStage.MATCHING
                if pct >= 90.0:
                    stage = DensifyStage.WRITING
                elif "triangula" in msg.lower():
                    stage = DensifyStage.TRIANGULATING
                
                self._update(stage, pct, msg)

            # Run the dense initialization
            result_code = dense_init(args, progress_callback=progress_cb)

            if result_code != 0:
                raise RuntimeError(f"Densification failed with code {result_code}")

            elapsed = time.time() - t0
            output_path = self.config.scene_path / "sparse" / "0" / "points3D.bin"

            # Try to count points from the output file
            num_points = 0
            if output_path.exists():
                import struct
                with open(output_path, "rb") as f:
                    num_points = struct.unpack("<Q", f.read(8))[0]

            result = DensifyResult(
                success=True,
                output_path=output_path,
                scene_path=self.config.scene_path,
                num_points=num_points,
                elapsed_time=elapsed,
            )

            with self._lock:
                self._result = result

            self._update(DensifyStage.DONE, 100.0, "Complete")
            lf.log.info(f"Densification complete: {num_points} points in {elapsed:.1f}s")

            if self.on_complete:
                self.on_complete(result)

        except Exception as e:
            lf.log.error(f"Densification error: {e}")
            self._update(DensifyStage.ERROR, self._progress, str(e))
            with self._lock:
                self._result = DensifyResult(success=False, error=str(e))

            if self.on_error:
                self.on_error(e)


class DensificationPanel:
    """GUI panel for dense point cloud initialization workflow."""

    panel_label = "Dense Initialization"
    panel_space = "SIDE_PANEL"
    panel_order = 5

    def __init__(self):
        self.job = None
        self.scene_path = ""
        self.last_result = None
        self._pending_import = None

        # Settings
        self.images_subdir = "images"
        self.roma_setting_idx = 3  # "fast" is default
        self.roma_settings = ["precise", "high", "base", "fast", "turbo"]

        # Advanced settings
        self.num_refs = 0.75
        self.nns_per_ref = 4
        self.matches_per_ref = 12000
        self.certainty_thresh = 0.20
        self.reproj_thresh = 1.5
        self.sampson_thresh = 5.0
        self.min_parallax_deg = 0.5
        self.max_points = 0
        self.no_filter = False

    def draw(self, layout):
        # Check for pending import (must happen on main thread)
        if self._pending_import:
            path = self._pending_import
            self._pending_import = None
            lf.log.info(f"Executing deferred import on main thread: {path}")
            lf.app.open(path)

        layout.heading("Dense Point Cloud Initialization")
        layout.label("Densify sparse COLMAP reconstruction using RoMa v2")
        layout.separator()

        layout.label("Scene Folder (with sparse/0/):")
        _, self.scene_path = layout.path_input(
            "##scenepath", self.scene_path, True, "Select Scene Folder"
        )

        layout.separator()

        # Basic settings
        if layout.collapsing_header("Settings", default_open=True):
            layout.label("Images Subfolder:")
            _, self.images_subdir = layout.path_input("##imgsubdir", self.images_subdir, True, "Select Image Folder")

            _, self.roma_setting_idx = layout.combo(
                "Quality", self.roma_setting_idx, self.roma_settings
            )
            layout.label("(precise > high > base > fast > turbo)")

        # Advanced settings
        if layout.collapsing_header("Advanced Settings", default_open=False):
            _, self.num_refs = layout.drag_float(
                "Reference Fraction", self.num_refs, 0.01, 0.1, 1.0
            )
            _, self.nns_per_ref = layout.drag_int(
                "Neighbors per Ref", self.nns_per_ref, 1, 1, 10
            )
            _, self.matches_per_ref = layout.drag_int(
                "Matches per Ref", self.matches_per_ref, 100, 1000, 50000
            )
            _, self.certainty_thresh = layout.drag_float(
                "Min Certainty", self.certainty_thresh, 0.01, 0.0, 1.0
            )
            _, self.reproj_thresh = layout.drag_float(
                "Max Reproj Error (px)", self.reproj_thresh, 0.1, 0.1, 10.0
            )
            _, self.sampson_thresh = layout.drag_float(
                "Max Sampson Error", self.sampson_thresh, 0.5, 0.0, 50.0
            )
            _, self.min_parallax_deg = layout.drag_float(
                "Min Parallax (deg)", self.min_parallax_deg, 0.1, 0.0, 10.0
            )
            _, self.max_points = layout.drag_int(
                "Max Points (0=unlimited)", self.max_points, 1000, 0, 10000000
            )
            _, self.no_filter = layout.checkbox("No Filtering", self.no_filter)

        layout.separator()

        # Job status / start button
        if self.job and self.job.is_running():
            stage = self.job.stage.value
            progress = self.job.progress

            layout.label(f"Stage: {stage}")
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
            layout.label(f"Points: {self.last_result.num_points:,}")
            layout.label(f"Time: {self.last_result.elapsed_time:.1f}s")

            if layout.button("Import to Scene##densify_import", (0, 36)):
                lf.log.info("Import to Scene button clicked")
                self._import_scene()

        if self.last_result and not self.last_result.success:
            layout.separator()
            layout.label("Error:")
            layout.text_selectable(self.last_result.error or "Unknown error", 60)

    def _start(self):
        if not self.scene_path:
            lf.log.warn("No scene path specified")
            return

        scene_path = Path(self.scene_path)
        sparse_dir = scene_path / "sparse" / "0"

        if not sparse_dir.exists():
            lf.log.error(f"Sparse reconstruction not found at {sparse_dir}")
            self.last_result = DensifyResult(
                success=False,
                error=f"Sparse reconstruction not found at {sparse_dir}"
            )
            return

        self.last_result = None

        config = DensifyConfig(
            scene_path=scene_path,
            images_subdir=self.images_subdir,
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
        lf.log.info(f"Densification complete: {result.num_points} points")
        self.last_result = result

    def _on_error(self, error: Exception):
        lf.log.error(f"Densification failed: {error}")
        self.last_result = DensifyResult(success=False, error=str(error))

    def _import_scene(self):
        """Import the scene into LichtFeld Studio."""
        if not self.last_result or not self.last_result.scene_path:
            lf.log.warn("No scene path available to import")
            return

        # Queue import for main thread (scene_path is the root with sparse/0/)
        scene_path = str(self.last_result.scene_path)
        lf.log.info(f"Queueing import: {scene_path}")
        self._pending_import = scene_path
