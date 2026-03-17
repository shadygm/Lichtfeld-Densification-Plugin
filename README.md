# Dense Initialization Plugin for LichtFeld Studio
![2026-03-0223-47-26-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/6bf3aa4d-7022-4813-b72b-f9d7f931c5a7)

A densification preprocessing plugin for LichtFeld Studio. It performs a densification pass on sparse reconstructions to generate dense point clouds using RoMa v2 matching.

_Based on [bounty #2](https://github.com/MrNeRF/LichtFeld-Studio/pull/413)._

## Features

- **RoMa v2 Matching**
- **Live Reconstruction Updates**
- **Configurable Parameters via GUI**
- **Densify Regions from Camera Subsets**
- **Scene Integration**
- **Installable via LFS Plugin Marketplace**

## Installation

### From GitHub (LichtFeld Studio Nightly)

In LichtFeld Studio:
1. Open the **Plugins Marketplace** panel under "View" menu bar.
2. Click **Install** on the `Lichtfeld-Densification-Plugin` card.

### Manual Installation

```bash
git clone https://github.com/shadygm/lichtfeld-densification-plugin.git ~/.lichtfeld/plugins/lichtfeld-densification-plugin
```

## Usage

### GUI

1. Open the **Dense Initialization** panel (side panel).
2. Ensure a scene is loaded in LichtFeld-Studio.
3. Optional: Select a subset of cameras in the scene graph and keep **ROI: Selected Cameras Only** enabled to densify only that region of views.
4. Optional: Configure settings (e.g., RoMa quality, reference fraction, filtering thresholds) using the scrub controls.
5. Click **Start Densification**.
6. Monitor progress.
7. To reuse the result, right-click the point cloud in the scene graph and select **Save to Disk**.

## Configuration

### RoMa Settings

- `precise`: Highest quality, slow and VRAM heavy (H_lr=800, bidirectional).
- `high`: High quality, (H_lr=64, bidirectional matching)
- `base`: Balanced (H_lr=640).
- `fast`: Default, fast (H_lr=512).
- `turbo`: Fastest (H_lr=320).

### Filtering Thresholds

- **Certainty Thresh**: Min overlap certainty (0.0-1.0).
- **Reproj Thresh**: Max reprojection error (px).
- **Sampson Thresh**: Max Sampson error (px²).
- **Min Parallax Deg**: Min parallax angle (deg).
- **No Filter**: Disable all geometric checks for raw output.

Adjust these in the GUI's Advanced Settings.

### Camera Selection

- **Selected Cameras Only**: Use only the currently selected cameras for densification.
- **Reference Fraction**: Fraction of active cameras used as reference views.
- **Neighbors per Ref**: Number of nearest neighbor views matched per reference.

## Results

### Quality Comparison Using MipNerf360
| Scene    | Method | PSNR (dB) | SSIM     | Num Gaussians |
|----------|--------|-----------|----------|---------------|
| garden   | DENSE  | 28.0025   | 0.867532 | 1,000,000     |
| garden   | SPARSE | 27.8082   | 0.857569 | 1,000,000     |
| bicycle  | DENSE  | 25.0693   | 0.812810 | 1,000,000     |
| bicycle  | SPARSE | 24.9199   | 0.786084 | 1,000,000     |
| stump    | DENSE  | 27.5625   | 0.859106 | 1,000,000     |
| stump    | SPARSE | 26.6290   | 0.807915 | 1,000,000     |
| bonsai   | DENSE  | 32.7332   | 0.951858 | 1,000,000     |
| bonsai   | SPARSE | 32.9545   | 0.951101 | 1,000,000     |
| counter  | DENSE  | 30.5424   | 0.929821 | 1,000,000     |
| counter  | SPARSE | 30.3430   | 0.924391 | 1,000,000     |
| kitchen  | DENSE  | 32.2915   | 0.938054 | 1,000,000     |
| kitchen  | SPARSE | 32.3918   | 0.936084 | 1,000,000     |
| room     | DENSE  | 33.8272   | 0.938879 | 1,000,000     |
| room     | SPARSE | 33.6627   | 0.936690 | 1,000,000     |
| **Mean** | DENSE  | **30.0041**| **0.899723**| 1,000,000     |
| **Mean** | SPARSE | **29.8156**| **0.885691**| 1,000,000     |

### Training Efficiency
<img width="2151" height="1255" alt="image" src="https://github.com/user-attachments/assets/9917985b-0f02-46de-936a-2d7351f4ac50" />
*Max PSNR reached across all scenes with/without densification*

### Key Improvements
- **Quality Gain**:  
  DENSE achieves **+0.63% average PSNR** (30.0041 dB vs 29.8156 dB) and **+1.58% SSIM** (0.8997 vs 0.8857) compared to MASTER at 30k iterations.
  
- **Training Speed**:  
  Densification reaches the same PSNR quality **13% faster** (at ~26k iterations) than non-densified training (requires 30k iterations).

## License

This plugin's code is released under GPL-3.0-or-later.

RoMa and DINOv3 have their own licenses—review them separately for redistribution or commercial use.
