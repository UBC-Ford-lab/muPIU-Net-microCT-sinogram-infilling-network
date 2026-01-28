# Data Preparation Guide

## Data Availability

The raw CT projection data used in this research is available upon request for academic purposes. Please contact the UBC Ford Lab for data access.

## VFF File Format

This project uses the **VFF (Volume File Format)** for CT data:

### Structure

```
[ASCII Header]
\f (form-feed character)
[Binary Data]
```

### Header Fields

```
ncaa=1;
rank=3;
type=density;f
format=slice;
bits=16;
bands=1;
size=2476 2024 438;                    # (width, height, depth)
spacing=0.0850 0.0850 0.4000;          # Voxel spacing in mm
origin=0.000000 0.000000 0.000000;
rawsize=2024 2476 438;
byteorder=big-endian;                  # Important: requires byteswap on x86
```

### Reading VFF Files

```python
from ct_core import vff_io

# Read VFF file
header, data = vff_io.read_vff("/path/to/file.vff")

# Access metadata
print(f"Shape: {header['size']}")
print(f"Spacing: {header['spacing']}")

# Data is a numpy array in (z, y, x) order
print(f"Data shape: {data.shape}")
```

## Directory Structure

Place your data in the following structure:

```
data/
├── scans/
│   ├── Scan_1680/
│   │   ├── scan.xml           # Geometry parameters
│   │   ├── projection_0001.vff
│   │   ├── projection_0002.vff
│   │   └── ...
│   ├── Scan_1681/
│   └── ...
├── results/
│   ├── ground_truth_reconstruction.vff
│   └── ...
└── models/
    └── model_epoch_75.pth
```

## Creating Sinogram Dataset

Before running base model comparison, convert projections to sinograms:

```bash
cd base_models/shared/utils

python create_sinogram_dataset.py \
    --scan_folder /path/to/data/scans/Scan_1680 \
    --output_folder ../sinogram_dataset
```

This creates:
- `sinograms_gt/` - Ground truth sinograms (PNG)
- `sinograms_masked/` - Masked sinograms with every 2nd projection removed
- `masks/` - Binary masks for inpainting
- `metadata.json` - Normalization parameters for reconstruction

## Geometry Configuration

FDK reconstruction requires geometry from `scan.xml`:

```python
geometry = {
    'R_s': source_to_isocenter,      # mm
    'R_d': detector_to_isocenter,    # mm
    'da': detector_pixel_width,      # mm
    'db': detector_pixel_height,     # mm
    'vol_shape': (1100, 1100, 300),  # Reconstruction grid
    'vol_origin': (0, 0, 0),
    'dx': 0.085,                     # Voxel size (mm)
    'dz': 0.4,                       # Z voxel size (mm)
    'central_pixel_a': center_col,
    'central_pixel_b': center_row
}
```

## Image Quality Metrics

For metric calculation, you need phantom scans with:

1. **Slanted edge** - For MTF calculation
2. **Uniform region** - For NPS calculation
3. **Material inserts** - For TTF calculation

See the metric calculator scripts for ROI specification examples.
