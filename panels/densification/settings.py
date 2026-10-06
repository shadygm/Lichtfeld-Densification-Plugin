# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scrub controls and densification settings."""

from dataclasses import replace
from lfs_plugins import ScrubFieldSpec


SCRUB_FIELD_SPECS = {
    "certainty_thresh": ScrubFieldSpec(
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        fmt="%.2f",
        data_type=float,
    ),
    "num_refs": ScrubFieldSpec(
        min_value=0.1,
        max_value=1.0,
        step=0.01,
        fmt="%.2f",
        data_type=float,
    ),
    "nns_per_ref": ScrubFieldSpec(
        min_value=1.0,
        max_value=10.0,
        step=1.0,
        fmt="%d",
        data_type=int,
    ),
    "reproj_thresh": ScrubFieldSpec(
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        fmt="%.1f",
        data_type=float,
    ),
    "sampson_thresh": ScrubFieldSpec(
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        fmt="%.1f",
        data_type=float,
    ),
    "min_parallax_deg": ScrubFieldSpec(
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        fmt="%.1f",
        data_type=float,
    ),
    "voxel_size": ScrubFieldSpec(
        min_value=0.001,
        max_value=0.1,
        step=0.001,
        fmt="%.3f",
        data_type=float,
    ),
    "viz_interval": ScrubFieldSpec(
        min_value=0.0,
        max_value=10.0,
        step=1.0,
        fmt="%d",
        data_type=int,
    ),
    "min_track_length": ScrubFieldSpec(
        min_value=0.0,
        max_value=10.0,
        step=1.0,
        fmt="%d",
        data_type=int,
    ),
}


class DensificationSettings:
    def _get_scrub_field_value(self, prop: str) -> float:
        if prop not in SCRUB_FIELD_SPECS:
            raise KeyError(prop)
        return float(self._voxel_size_ui if prop == "voxel_size" else getattr(self.config, prop))

    def _set_scrub_field_value(self, prop: str, value: float) -> None:
        spec = self._scrub_specs[prop]
        if prop == "voxel_size":
            self._set_voxel_size(value)
            return
        setter = self._set_int_config if spec.data_type is int else self._set_float_config
        setter(prop, value, spec.min_value, spec.max_value)

    def _sync_scrub_specs(self) -> bool:
        changed = self._update_scrub_spec(
            "nns_per_ref",
            max_value=float(self._nns_per_ref_max()),
        )
        if self._update_scrub_spec("min_track_length", max_value=float(self._min_track_length_max())):
            changed = True
        max_nns = self._nns_per_ref_max()
        if self.config.nns_per_ref > max_nns:
            self.config.nns_per_ref = max_nns
            self._dirty("nns_per_ref")
            changed = True
        max_track = self._min_track_length_max()
        if self.config.min_track_length > max_track:
            self.config.min_track_length = max_track
            self._dirty("min_track_length")
            changed = True
        return changed

    def _update_scrub_spec(self, prop: str, *, max_value: float) -> bool:
        current_spec = self._scrub_specs[prop]
        if abs(current_spec.max_value - max_value) <= 1.0e-9:
            return False
        next_spec = replace(current_spec, max_value=max_value)
        self._scrub_specs[prop] = next_spec
        self._scrub_fields.set_spec(prop, next_spec)
        return True

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

    def _set_roi_only_selected(self, value):
        if self._is_running():
            self._dirty("roi_only_selected")
            return
        enabled = bool(value)
        if enabled == self.config.roi_only_selected:
            return
        self.config.roi_only_selected = enabled
        max_nns = self._nns_per_ref_max()
        if self.config.nns_per_ref > max_nns:
            self.config.nns_per_ref = max_nns
        max_track = self._min_track_length_max()
        if self.config.min_track_length > max_track:
            self.config.min_track_length = max_track
        self._dirty(
            "roi_only_selected",
            "camera_scope_text",
            "show_roi_selection_warning",
            "nns_per_ref",
            "min_track_length",
        )

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
            v = int(max(vmin, min(vmax, int(float(value)))))
        except (TypeError, ValueError):
            return
        if v == getattr(self.config, attr) and type(getattr(self.config, attr)) is int:
            return
        setattr(self.config, attr, v)
        self._dirty(attr)

    def _set_start_training_when_complete(self, value):
        self._start_training_when_complete = bool(value)
        self._dirty("start_training_when_complete")

    def _set_write_colmap(self, value):
        self._write_colmap = bool(value)
        self._dirty("write_colmap")

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
            "matches_per_ref": (1000, 30000),
            "max_points": (0, 10000000),
        }
        vmin, vmax = range_map.get(field_name, (0, 999999999))
        new_val = max(vmin, min(vmax, new_val))

        if new_val != current:
            setattr(self.config, field_name, new_val)
            self._dirty(f"{field_name}_str")
