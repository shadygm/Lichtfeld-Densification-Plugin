# Dense Initialization Plugin for LichtFeld Studio

A densification preprocessing plugin in LichtFeld Studio. This tool performs densification pass on sparse reconstructions to create dense point clouds.

Based on [bounty #2](https://github.com/MrNeRF/LichtFeld-Studio/pull/413), with an upgrade to RoMaV2.

## Features

- **RoMaV2 Matching**: Upgraded dense matching for better accuracy.
- **GUI Panel**: Easy-to-use interface with progress tracking and scene import.

## Installation

### From GitHub (LichtFeld Studio v0.5+)

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
2. Ensure a scene is loaded in LichtFeld Studio.
3. Configure settings (e.g., RoMa quality, reference fraction, filtering thresholds).
4. Click **Start Densification**.
5. Monitor progress and click **Import to Scene** when complete.


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

Adjust these in the GUI's Advanced Settings.

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
