"""Run the core without LichtFeld: python -m core --scene-root DATA --output-path OUT.ply."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np

from .camera_models import CameraRecord
from .config import DensePipelineConfig
from .geometry import K_from_camera, P_from_KRt, cam_center_world, pose_world2cam
from .image_utils import find_image, image_dir, to_uint8_rgb
from .pipeline import run_dense_pipeline
from .selection import nearest_neighbors, select_cameras_kcenters
from .writers import write_ply


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone RoMa densification core")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True, help="Output PLY (input dataset is read-only)")
    parser.add_argument("--images-subdir", default="images_2")
    defaults = DensePipelineConfig(output_path="")
    for name in (
        "num_refs", "nns_per_ref", "matches_per_ref", "certainty_thresh",
        "reproj_thresh", "sampson_thresh", "min_parallax_deg", "min_track_length",
        "max_points", "seed", "prefetch_packages", "pack_workers",
    ):
        value = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    parser.add_argument("--roma-setting", choices=("precise", "high", "base", "fast", "turbo"), default=defaults.roma_setting)
    parser.add_argument("--no-filter", action="store_true")
    options = vars(parser.parse_args())
    scene_root = options.pop("scene_root")
    output_path = options.pop("output_path")
    images = image_dir(str(scene_root), options.pop("images_subdir"))
    if output_path.suffix.lower() != ".ply":
        parser.error("--output-path must end in .ply")
    config = DensePipelineConfig(output_path=str(output_path), viz_interval=0, **options)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import pycolmap

    started = time.perf_counter()
    reconstruction = pycolmap.Reconstruction(str(scene_root / "sparse" / "0"))
    records = []
    for uid, image in sorted(reconstruction.images.items()):
        camera = reconstruction.cameras[image.camera_id]
        K = K_from_camera(camera)
        R, t = pose_world2cam(image)
        records.append(CameraRecord(
            uid=uid, image_path=find_image(images, image.name),
            width=camera.width, height=camera.height, K=K, R=R, t=t,
            P=P_from_KRt(K, R, t), C=cam_center_world(R, t), colmap_camera=camera,
        ))
    if len(records) < 2:
        parser.error("the reconstruction must contain at least two cameras")
    poses = np.stack([camera.flat_pose() for camera in records])
    ref_count = round(config.num_refs * len(records)) if config.num_refs <= 1 else int(config.num_refs)
    refs = select_cameras_kcenters(poses, max(1, ref_count))
    neighbors = nearest_neighbors(poses, config.nns_per_ref)
    result = run_dense_pipeline(
        records, refs, neighbors, config,
        progress_callback=lambda pct, message: print(f"{pct:5.1f}% {message}", flush=True),
    )
    keep = np.flatnonzero(np.asarray([len(track) >= config.min_track_length for track in result.tracks]))
    if config.max_points > 0 and len(keep) > config.max_points:
        keep = np.random.default_rng(config.seed).choice(keep, size=config.max_points, replace=False)
    if not len(keep):
        raise RuntimeError("No points remain after track-length filtering")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(str(output_path), result.xyz[keep], to_uint8_rgb(result.rgb[keep]))
    print(json.dumps(dict(
        cameras=len(records), references=len(refs), points=len(keep),
        pipeline_seconds=result.elapsed_seconds, total_seconds=time.perf_counter() - started,
        output_path=str(output_path),
    )), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
