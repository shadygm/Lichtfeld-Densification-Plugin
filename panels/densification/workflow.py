# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Start jobs and import preview and completed point clouds."""

from dataclasses import replace
import lichtfeld as lf
from .job import DensifyJob, DensifyResult
from ...core.reconstruction.cloud import DenseCloud


class DensificationWorkflow:
    def _start_training_after_import(self):
        try:
            lf.start_training()
        except Exception as exc:
            lf.log.warn(f"Failed to start training after densification: {exc}")

    def _is_running(self) -> bool:
        with self._handoff_lock:
            pending = self._pending_error is not None or self._pending_import is not None
        return (pending
                or self._cloud_update is not None or (self.job is not None and self.job.is_running()))

    def _on_do_start(self, handle, event, args):
        self._start()

    def _on_do_cancel(self, handle, event, args):
        with self._handoff_lock:
            self._cancel_requested = True
            self._pending_error = "Cancelled"
        if self.job:
            self.job.cancel()

    def _start(self):
        if self._is_running():
            return
        if not self._has_training_data():
            lf.log.warn("No training cameras found in scene")
            self.last_result = DensifyResult(
                success=False,
                error="No training cameras found. Please load a dataset first."
            )
            return

        camera_nodes = self._get_effective_camera_nodes()
        if self.config.roi_only_selected and len(camera_nodes) < 2:
            self.last_result = DensifyResult(
                success=False,
                error="ROI densification requires at least 2 selected cameras.",
            )
            return
        if len(camera_nodes) < 2:
            self.last_result = DensifyResult(
                success=False,
                error="Need at least 2 cameras for densification.",
            )
            return
        if not self._capture_base_point_cloud():
            return

        self.last_result = None
        with self._handoff_lock:
            self._cancel_requested = False
            self._pending_start_training = False

        self._active_run_roi_only_selected = bool(self.config.roi_only_selected)

        config = replace(
            self.config,
            output_path=self._get_temp_output_path(),
        )

        self.job = DensifyJob(
            config=config,
            camera_nodes=camera_nodes,
            on_complete=self._on_complete,
            on_error=self._on_error,
            on_cloud_preview=self._on_cloud_preview,
            debug_state=self.debug_state,
            write_colmap=self._write_colmap,
        )
        self.job.start()

    def _on_cloud_preview(self, cloud: DenseCloud):
        with self._handoff_lock:
            if not self._cancel_requested:
                self._pending_import = cloud

    def _import_output(self, output: DenseCloud | DensifyResult, *, start_training=False) -> bool:
        final = isinstance(output, DensifyResult)
        cloud = output.cloud if final else output
        if cloud is None:
            return False
        try:
            scene = lf.get_scene()
            target = self._resolve_target_point_cloud_node(scene)
            if target is None:
                lf.log.error("No point cloud node found to merge into")
                return False
            point_cloud = target.point_cloud()
            if point_cloud is None:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return False
            return self._apply_dense_point_cloud(target, point_cloud, cloud.points, cloud.colors,
                                                 result=output if final else None, start_training=start_training)
        except Exception as exc:
            lf.log.error(f"Failed to import dense point cloud: {exc}")
            if final:
                self.last_result = DensifyResult(success=False, error=str(exc))
            return False
        finally:
            # The async API owns the immutable inputs until inputs_released.
            if final:
                output.cloud = None

    def _on_complete(self, result: DensifyResult):
        with self._handoff_lock:
            if self._cancel_requested:
                result.cloud = None
                return
            base_count = len(self._base_point_cloud_points) if self._base_point_cloud_points is not None else 0
            run_roi_only_selected = self._run_roi_only_selected()
            if run_roi_only_selected and base_count > 0 and result.success:
                result.num_points = base_count + result.num_points
            if self._auto_import and result.success and result.cloud is not None:
                self.last_result = None  # Completion is visible only after publication.
                self._pending_import = result
                self._pending_start_training = bool(self._start_training_when_complete and result.success)
            else:
                self.last_result = result
                result.cloud = None
                self._pending_start_training = False

    def _on_error(self, error: Exception):
        with self._handoff_lock:
            if self._cancel_requested:
                return
            # Worker callback: keep scene changes and snapshot cleanup on the UI thread.
            self._pending_error = str(error)
        lf.log.error(f"Densification failed: {error}")

    def _apply_dense_point_cloud(self, target, point_cloud, points, colors, *,
                                 result=None, start_training=False) -> bool:
        if self._run_roi_only_selected():
            points, colors = self._build_roi_merge_arrays(points, colors)
        self._queue_cloud_update(target, point_cloud, points, colors,
                                 kind="final" if result is not None else "preview",
                                 result=result, start_training=start_training)
        return True
