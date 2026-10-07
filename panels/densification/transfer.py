# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Submit immutable cloud inputs and poll publication on the scene thread."""
from __future__ import annotations

from dataclasses import dataclass

import lichtfeld as lf
import numpy as np

from .job import DensifyResult, DensifyStage


@dataclass
class _CloudUpdate:
    ticket: lf.scene.PointCloudUpdateTicket
    node_uuid: str
    kind: str
    result: DensifyResult | None = None
    start_training: bool = False


class DensificationTransfer:
    def _release_cloud_snapshot(self):
        # Drop our references; the async API retains any inputs still in flight.
        self._base_point_cloud_points = None
        self._base_point_cloud_colors = None
        self._active_run_roi_only_selected = None
        self._preview_override_active = False

    @staticmethod
    def _set_point_cloud_data(point_cloud, points, colors):
        # The API retains the input owners; never mutate submitted arrays/tensors.
        if not isinstance(points, lf.Tensor):
            points = np.ascontiguousarray(points, dtype=np.float32)
        if not isinstance(colors, lf.Tensor):
            colors = np.asarray(colors)
            colors = np.ascontiguousarray(colors, dtype=np.uint8 if colors.dtype == np.uint8 else np.float32)
        return point_cloud.set_data_async(points, colors, queue_policy="latest")

    def _queue_cloud_update(self, target, point_cloud, points, colors, *, kind,
                            result=None, start_training=False):
        ticket = self._set_point_cloud_data(point_cloud, points, colors)
        self._cloud_update = _CloudUpdate(ticket, target.uuid, kind, result, start_training)
        self._target_point_cloud_uuid = target.uuid
        # Even an unobserved publication must be rolled back on job failure.
        if kind != "restore":
            self._preview_override_active = True
        lf.log.debug(f"Queued {kind} cloud update for '{target.name}'")

    def _cancel_cloud_update(self):
        if self._cloud_update is not None:
            if self._cloud_update.ticket.state in ("superseded", "cancelled"):
                self._preview_override_active = False
            self._cloud_update.ticket.cancel()
            self._cloud_update = None

    def _discard_pending_import(self):
        output, _ = self._take_pending_import()
        if isinstance(output, DensifyResult):
            output.cloud = None

    def _take_pending_import(self):
        with self._handoff_lock:
            output, start_training = self._pending_import, self._pending_start_training
            self._pending_import = None
            self._pending_start_training = False
        return output, start_training

    def _recover_cloud(self, error):
        with self._handoff_lock:
            self._cancel_requested = True
            # Keep snapshots alive if the host emits scene notifications while
            # a rollback is being submitted, before its ticket is available.
            self._pending_error = error
        try:
            self._cancel_cloud_update()
            self._discard_pending_import()
            if self._preview_override_active:
                self._restore_base_point_cloud()
            if self._cloud_update is None:
                self._active_run_roi_only_selected = None
            self.last_result = DensifyResult(success=False, error=error)
        finally:
            with self._handoff_lock:
                self._pending_error = None

    def _poll_cloud_update(self):
        update = self._cloud_update
        if update is None or update.ticket.state in ("queued", "uploading"):
            return False
        state = update.ticket.state
        error = None
        if state == "published":
            # Publication replaces the payload: do not reuse an old PointCloud wrapper.
            scene = lf.get_scene()
            node = scene.get_node_by_uuid(update.node_uuid) if scene is not None else None
            if node is None or node.point_cloud() is None:
                state = "failed"
                error = "Densification target is no longer available"
            elif update.kind == "restore":
                self._preview_override_active = False
                self._active_run_roi_only_selected = None
            elif update.kind == "final":
                self.last_result = update.result
                self._active_run_roi_only_selected = None
                lf.log.info("Dense point cloud published")
        if state != "published":
            error = error or update.ticket.error or f"Cloud update {state}"
            lf.log.warn(f"Point cloud {update.kind} update: {error}")
            # Supersession/cancellation can belong to an external edit. A failed
            # upload leaves our earlier published previews in place instead.
            # Do not retry an unsuccessful rollback indefinitely.
            if state in ("superseded", "cancelled") or update.kind == "restore":
                self._preview_override_active = False
            self._cancel_requested = True
            if self.job:
                self.job.cancel()
            self._cloud_update = None
            self._recover_cloud(error)
            return True
        self._cloud_update = None
        if state == "published" and update.kind == "final" and update.start_training:
            self._start_training_after_import()
        return True

    def _update_cloud_transfer(self):
        with self._handoff_lock:
            error = self._pending_error
        if error is not None:
            self._recover_cloud(error)
            return True
        if self.job and self.job.stage == DensifyStage.CANCELLED and self.last_result is None:
            self._cancel_requested = True
            self._recover_cloud("Cancelled")
            return True
        changed = self._poll_cloud_update()
        output, start_training = self._take_pending_import()
        if output is not None:
            final = isinstance(output, DensifyResult)
            imported = self._import_output(output, start_training=start_training)
            if final and not imported:
                error = self.last_result.error if self.last_result and not self.last_result.success else None
                self._recover_cloud(error or "Failed to queue dense point cloud")
            changed = True
        return changed

    def _display_stage(self):
        if self._cloud_update is not None and self._cloud_update.kind != "preview":
            return "Restoring" if self._cloud_update.kind == "restore" else "Uploading"
        return self.job.stage.value.capitalize() if self.job else "Idle"

    def _display_status(self):
        if self._cloud_update is not None and self._cloud_update.kind != "preview":
            return "Restoring original point cloud..." if self._cloud_update.kind == "restore" else "Publishing dense point cloud..."
        return self.job.status if self.job else ""

    def _display_progress(self):
        if self._cloud_update is not None and self._cloud_update.kind != "preview":
            return 99.0
        return self.job.progress if self.job else 0.0
