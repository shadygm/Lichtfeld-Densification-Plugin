# COLMAP Plugin for LichtFeld Studio

Structure-from-Motion reconstruction plugin using [pycolmap](https://github.com/colmap/pycolmap).

## Features

- SIFT feature extraction
- Multiple matching strategies (exhaustive, sequential, vocab_tree, spatial)
- Incremental Structure-from-Motion
- Background processing with progress tracking
- GUI panel for easy workflow

## Installation

### From GitHub (LichtFeld Studio 1.x+)

In LichtFeld Studio:
1. Open **Plugins** panel
2. Enter: `https://github.com/YOUR_USERNAME/lichtfeld-plugin-colmap`
3. Click **Install**

### Manual Installation

```bash
git clone https://github.com/YOUR_USERNAME/lichtfeld-plugin-colmap.git ~/.lichtfeld/plugins/colmap
```

## Usage

### GUI

1. Open the **COLMAP Reconstruction** panel (side panel)
2. Select your image folder
3. Configure camera model and matching strategy
4. Click **Start Reconstruction**
5. Click **Import to Scene** when complete

### Python API

```python
import colmap

# Synchronous
result = colmap.run_pipeline(
    image_path="/path/to/images",
    output_path="/path/to/output",
    camera_model="OPENCV",
    match_type="exhaustive"
)

if result.success:
    print(f"Reconstructed {result.num_images} images")
    print(f"3D points: {result.num_points}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.3f}")

# Asynchronous with progress
job = colmap.run_pipeline_async(
    image_path="/path/to/images",
    on_progress=lambda stage, pct, msg: print(f"{stage}: {pct:.0f}% - {msg}"),
    on_complete=lambda result: print(f"Done: {result.success}")
)

# Cancel if needed
job.cancel()
```

## Configuration

### Camera Models

- `PINHOLE` - No distortion
- `OPENCV` - OpenCV distortion model (recommended)
- `SIMPLE_RADIAL` - Single radial distortion parameter
- `RADIAL` - Two radial distortion parameters

### Matching Strategies

- `exhaustive` - Match all image pairs (best quality, slow)
- `sequential` - Match consecutive images (fast, for video)
- `vocab_tree` - Vocabulary tree matching (large datasets)
- `spatial` - GPS-based spatial matching

## Dependencies

Automatically installed in plugin's virtual environment:

- `pycolmap>=0.6.0`

## License

GPL-3.0-or-later (same as LichtFeld Studio)
