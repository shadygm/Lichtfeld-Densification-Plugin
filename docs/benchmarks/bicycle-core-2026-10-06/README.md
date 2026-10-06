# Bicycle core benchmark

Standalone CLI only, on `/home/shadygm/projects/personal/data/360_v2/bicycle`.
Baseline: `9414b31`; optimization: batching, bounded image/feature reuse, and
one CPU triangulation task overlapping GPU matching. RoMaV2 sources are unchanged.

Environment: RTX 5070 Ti (16 GB), Python 3.12.13, PyTorch 2.11.0+cu128,
CUDA runtime 12.8, NumPy 2.4.1, pycolmap 4.0.2.

```bash
.venv/bin/python -m core \
  --scene-root /home/shadygm/projects/personal/data/360_v2/bicycle \
  --output-path /tmp/bicycle.ply
.venv/bin/python -m unittest discover -s core/tests -q
```

All reconstruction settings stay at their defaults: 194 cameras, 155 references,
465 pairs, `fast` at 512 pixels, 10,000 requested samples per reference, and
0.8 original-camera-pixel reprojection threshold. Timings include model startup,
matching, geometry, PLY writing and independent final-point error metrics.

The ordinary threaded loader emits packages in completion order. Even with seed
zero, the resulting assignment of random draws to references varies across runs.
For a controlled quality comparison, both versions also ran with
`--prefetch-packages 1 --pack-workers 1`, preserving reference order. Raw metric
sidecars are retained alongside this report. Output PLYs and runtime logs are in
`/tmp/densification-speed`; these temporary artifacts are not versioned.

## Final default run

| Measure | Baseline | Optimized |
| --- | ---: | ---: |
| Complete CLI seconds | 85.223 | 34.176 |
| Pipeline seconds | 83.488 | 32.846 |
| Final points | 1,067,460 | 1,066,317 |
| Mean reprojection error (pixels) | 0.315798 | 0.315773 |
| Reprojection RMSE (pixels) | 0.388920 | 0.388742 |
| p95 reprojection error (pixels) | 0.722452 | 0.722175 |
| Invalid observations | 0 | 0 |

The final default run is **2.49× faster**, a 59.9% reduction in elapsed time.
A prior default overlap run took 34.135 seconds. All reference/pair counts,
matching resolution, sample budgets and reconstruction filters remain unchanged.
Final production code is at `48b89a7`; the earlier overlap run preceded the
threshold-rounding correction. `baseline.json` and `optimized.json` retain the
complete metric sidecars for the baseline and final production run.

## Quality validation

Fixed-order, separate full runs measured 86.364 seconds at baseline and 39.041
seconds after optimization. Mean/RMSE improved from 0.316394/0.389357 to
0.316096/0.389088 pixels; point counts were 1,066,840 and 1,066,370.
Fixed loading order removes one source of variation, but separate model runs
still vary slightly. These statistics alone are not an exact-equivalence test.

A stronger full-dataset check invoked the old and new triangulation functions
for each of the same 155 matched references, resetting NumPy's sampling state
between calls. All 1,067,255 point coordinates and all observation tracks were
identical. Independent reprojection metrics are functions of those coordinates,
tracks and unchanged cameras, so they are identical for these fixed inputs.
The validation run executed both triangulation paths and is excluded from speed
measurements. See `geometry-equivalence.json` and `geometry-validation.json`.

Batched matrix multiplication can round differently from individual projection.
Observations within 0.002 pixels of the acceptance threshold use the original
individual calculation to preserve filtering decisions. The internal error
proxy can otherwise differ by less than 0.001 pixels in the captured fixtures;
final metrics are calculated independently, rather than using that proxy.

Feature-cache validation compared three real bicycle pairs against the original
matcher, both cold and on cache hits: warp grids and certainty maps were bitwise
identical. Filtered and unfiltered fixture checks also preserved point colors
and optional preview matches. Seventeen unit tests cover geometry, metrics,
sampling, bounded caches/loading, ordered overlap, previews and cancellation.

Caches retain at most 256 prepared images/descriptors and 24 refinement-feature
entries at 512 pixels, with smaller limits at higher resolutions. They are
cleared when the matcher closes. This exchanges bounded GPU memory for avoided
feature extraction; the 16 GB test GPU supported the full run. Performance and
memory requirements on other hardware/datasets are not established here.
