# muPIU-Net: Micro-CT Projection Infilling U-Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for **projection infilling** in micro-CT imaging - recovering missing CT projections to enable undersampled scanning with reduced radiation dose.

## Overview

This repository contains the code for our paper on CT sinogram infilling using deep learning. The framework implements two main approaches:

1. **muPIU-Net (U-Net)** - Custom U-Net architecture for direct projection interpolation in the sinogram domain
2. **Base Model Comparison** - Four state-of-the-art image inpainting models (LaMa, MAT, DeepFill v2, RePaint) adapted for CT sinogram inpainting

Both approaches use FDK reconstruction for fair comparison, followed by quantitative image quality metrics (MTF, NPS, NEQ).

## Repository Structure

```
muPIU-Net/
├── ct_core/                    # Core library package
│   ├── vff_io.py               # VFF file I/O
│   ├── calibration.py          # HU calibration
│   ├── field_correction.py     # Bright/dark field correction
│   └── paths.py                # Centralized path configuration
│
├── unet_pipeline/              # U-Net training and inference
│   ├── model.py                # U-Net architecture
│   ├── train.py                # Training script
│   ├── infer.py                # Inference script
│   └── hpc/                    # SLURM job scripts
│
├── reconstruction/             # FDK reconstruction
│   └── fdk.py                  # GPU-accelerated FDK
│
├── metric_calculators/         # Image quality metrics
│   ├── mtf_calculator.py       # Modulation Transfer Function
│   ├── nps_calculator.py       # Noise Power Spectrum
│   ├── neq_calculator.py       # Noise Equivalent Quanta
│   └── helper_scripts/         # Comparison plotting
│
├── base_models/                # Base model comparison framework
│   ├── models/
│   │   ├── lama/               # LaMa (Large Mask Inpainting)
│   │   ├── mat/                # MAT (Mask-Aware Transformer)
│   │   ├── deepfill/           # DeepFill v2
│   │   └── repaint/            # RePaint (Diffusion-based)
│   └── shared/
│       └── utils/              # Shared utilities
│
└── data/                       # Data directories (not tracked)
    ├── scans/                  # Raw projection data
    ├── results/                # Processed results
    └── models/                 # Trained model checkpoints
```

## Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

### Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/UBC-Ford-lab/muPIU-Net-microCT-sinogram-infilling-network.git
cd muPIU-Net-microCT-sinogram-infilling-network

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ct_core import vff_io"
```

## Usage

### U-Net Training

```bash
# Create data split
python unet_pipeline/create_data_split.py \
    --train_test_split=0.9 \
    --desired_scans_in_testing=['Scan_1680','Scan_1681'] \
    --number_of_scans_in_total=80

# Train model
python unet_pipeline/train.py --epochs=75 --batch_size=3
```

### U-Net Inference

```bash
# Generate predictions for missing projections
python unet_pipeline/infer.py

# Run FDK reconstruction
python reconstruction/fdk.py
```

### Base Model Comparison

Each base model follows the same workflow:

```bash
cd base_models/models/lama  # or mat, deepfill, repaint

# Run inference
python scripts/run_inference.py

# Reconstruct volume
python scripts/reconstruct.py --scan_folder /path/to/scan

# Calculate domain comparison metrics
python scripts/domain_comparison.py --gt_volume /path/to/gt.vff
```

### Metric Calculation

```bash
cd metric_calculators

# Individual metrics
python mtf_calculator.py
python nps_calculator.py
python neq_calculator.py

# Or all at once
python all_metrics_calculator.py
```

## Data Format

The primary data format is **VFF (Volume File Format)**:
- ASCII header + binary big-endian data
- Arrays in (z, y, x) ordering
- See [DATA_PREPARATION.md](DATA_PREPARATION.md) for details

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wiegmann2026mupiunet,
  title={muPIU-Net: Deep Learning for Micro-CT Projection Infilling},
  author={Wiegmann, Falk and others},
  journal={TBD},
  year={2026}
}
```

See [CITATION.cff](CITATION.cff) for the full citation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted at the UBC Ford Lab. We thank the developers of the following open-source projects:

- [LaMa](https://github.com/advimman/lama) - Large Mask Inpainting
- [MAT](https://github.com/fenglinglwb/MAT) - Mask-Aware Transformer
- [DeepFill v2](https://github.com/JiahuiYu/generative_inpainting) - Generative Inpainting
- [RePaint](https://github.com/andreas128/RePaint) - Diffusion-based Inpainting
