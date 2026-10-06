# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Background densification jobs, progress and cancellation."""

import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, List, ClassVar
import lichtfeld as lf
from ...core.pipeline.config import DensePipelineConfig
from ...core.reconstruction.cloud import DenseCloud
from ...core.previews.matches import MatchDebugState


class DensifyStage(Enum):
    """Pipeline execution stage."""

    IDLE = "Idle"
    LOADING = "Loading"
    MATCHING = "Matching"
    TRIANGULATING = "Triangulating"
    FINALIZING = "Finalizing"
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
    error: Optional[str] = None
    cloud: Optional[DenseCloud] = None


class DensifyJob:
    """Background densification job with progress tracking.
    
    Uses cameras from LichtFeld Studio directly - no file paths needed.
    """
    _instances: ClassVar[weakref.WeakSet["DensifyJob"]] = weakref.WeakSet()
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: DensePipelineConfig,
        camera_nodes: Optional[List] = None,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_complete: Optional[Callable[[DensifyResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_sequential_viz: Optional[Callable[[str], None]] = None,
        debug_state: Optional[MatchDebugState] = None,
        write_colmap: bool = False,
        on_cloud_preview: Optional[Callable[[DenseCloud], None]] = None,
    ):
        self.config = config
        self.camera_nodes = list(camera_nodes) if camera_nodes is not None else None
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_sequential_viz = on_sequential_viz
        self.debug_state = debug_state
        self.write_colmap = write_colmap
        self.on_cloud_preview = on_cloud_preview

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
            DensifyStage.FINALIZING,
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

            scope_label = "selected ROI cameras" if self.config.roi_only_selected else "scene cameras"
            self._update(DensifyStage.LOADING, 5.0, f"Loading {scope_label}...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 0.0, "Cancelled")
                return

            if self.camera_nodes is None:
                scene = lf.get_scene()
                camera_nodes = [n for n in scene.get_nodes() if n.has_camera]
            else:
                camera_nodes = list(self.camera_nodes)
            if not camera_nodes:
                raise RuntimeError("No cameras found in scene. Please load a dataset first.")
            
            lf.log.info(f"Found {len(camera_nodes)} cameras in scene")

            self._update(DensifyStage.MATCHING, 10.0, "Initializing RoMa v2...")
            if check_cancelled():
                self._update(DensifyStage.CANCELLED, 10.0, "Cancelled")
                return

            # Import and run the LFS-based densify pipeline
            from ...densify import dense_init_from_lfs

            def progress_cb(pct: float, msg: str):
                if check_cancelled():
                    return
                
                stage = DensifyStage.MATCHING
                if pct >= 95.0:
                    stage = DensifyStage.WRITING if self.write_colmap else DensifyStage.FINALIZING
                elif pct >= 90.0 or "triangula" in msg.lower():
                    stage = DensifyStage.TRIANGULATING
                
                self._update(stage, pct, msg)

            # Run the dense initialization
            result_code, result_info = dense_init_from_lfs(
                camera_nodes,
                self.config,
                progress_callback=progress_cb,
                on_sequential_viz=self.on_sequential_viz,
                on_cloud_preview=self.on_cloud_preview,
                debug_state=self.debug_state,
                cancel_requested=check_cancelled,
                write_colmap=self.write_colmap,
            )

            if result_code == 2 or check_cancelled():
                with self._lock:
                    self._result = DensifyResult(success=False, error="Cancelled")
                self._update(DensifyStage.CANCELLED, self._progress, "Cancelled")
                return

            if result_code != 0:
                raise RuntimeError(result_info or "Densification failed")

            elapsed = time.time() - t0
            num_points = len(result_info.points)

            result = DensifyResult(
                success=True,
                output_path=result_info.output_path,
                num_points=num_points,
                elapsed_time=elapsed,
                cloud=result_info,
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
            self.camera_nodes = None
            if self.debug_state:
                self.debug_state.release_waiters()
