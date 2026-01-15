# Dense Initialization Plugin for LichtFeld Studio

A densification preprocessing plugin for COLMAP-based workflows in LichtFeld Studio. This tool performs densification pass on sparse COLMAP reconstructions to create dense point clouds, enabling faster downstream processing, model training, and higher quality.

Based on bounty #2 ([link](https://github.com/MrNeRF/LichtFeld-Studio/pull/413)), with an upgrade to RoMaV2 for improved dense matching.

## Features

- **RoMaV2 Matching**: Upgraded dense correspondence matching for better accuracy.
- **GUI Panel**: Easy-to-use interface with progress tracking and scene import.
- **CLI Support**: Run via `densify.py` with full argument control.

## Installation

### From GitHub (LichtFeld Studio 1.x+)

In LichtFeld Studio:
1. Open the **Plugins** panel.
2. Enter: `https://github.com/shadygm/lichtfeld-densification-plugin`
3. Click **Install**.

### Manual Installation

```bash
git clone https://github.com/shadygm/lichtfeld-densification-plugin.git ~/.lichtfeld/plugins/lichtfeld-densification-plugin
```

## Usage

### GUI

1. Open the **Dense Initialization** panel (side panel).
2. Select your scene folder.
3. Configure settings (e.g., RoMa quality, reference fraction, filtering thresholds).
4. Click **Start Densification**.
5. Monitor progress and click **Import to Scene** when complete.

### Python API

```python
from densify import dense_init
from argparse import Namespace

# Synchronous run
args = Namespace(
    scene_root="/path/to/scene",
    images_subdir="images_2",
    roma_setting="fast",
    num_refs=0.75,
    nns_per_ref=4,
    matches_per_ref=12000,
    certainty_thresh=0.05,
    reproj_thresh=4.0,
    sampson_thresh=50.0,
    min_parallax_deg=0.1,
    no_filter=False,
    max_points=0,
    seed=0
)

result_code = dense_init(args)
if result_code == 0:
    print("Densification successful!")
```

For asynchronous usage, use the panel's job system or extend with threading.

## Configuration

### RoMa Settings

- `precise`: High quality, slow (H_lr=800, bidirectional).
- `base`: Balanced (H_lr=640, no high-res).
- `fast`: Default, fast (H_lr=512).
- `turbo`: Fastest (H_lr=320).

### Filtering Thresholds

- **Certainty Thresh**: Min overlap certainty (0.0-1.0).
- **Reproj Thresh**: Max reprojection error (px).
- **Sampson Thresh**: Max Sampson error (px²).
- **Min Parallax Deg**: Min parallax angle (deg).
- **No Filter**: Disable all geometric checks for raw output.

Adjust these in the GUI's Advanced Settings or via CLI flags.

## Dependencies

Automatically installed in the plugin's virtual environment:

- `pycolmap>=0.6.0`
- `romav2` (RoMaV2)
- `Pillow`
- `numpy`
- `scipy`
- `tqdm`

## License

This plugin's code is released under GPL-3.0-or-later (same as LichtFeld Studio).

RoMa (matching model) and DINOv3 have their own licenses—review them separately for redistribution or commercial use.
