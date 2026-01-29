#!/usr/bin/env python
"""U-Net inference for projection infilling."""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from ct_core import vff_io
from ct_core.field_correction import write_vff
from unet_pipeline.model import UNet


def natural_sort_key(path):
    """Sort key for natural ordering of filenames."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.name)]


def parse_args():
    p = argparse.ArgumentParser(description='U-Net projection infilling')
    p.add_argument('--scan_folder', type=str, required=True,
                   help='Path to folder containing VFF projection files')
    p.add_argument('--checkpoint', type=str,
                   default='data/models/mupiu-net_final_model.pth',
                   help='Path to model checkpoint')
    p.add_argument('--output_dir', type=str,
                   default='data/results',
                   help='Output directory for infilled projections')
    p.add_argument('--device', type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    # Load VFF files from scan folder
    scan_path = Path(args.scan_folder)

    # Exclude calibration files (dark.vff, bright.vff) from projection list
    calibration_files = {'dark.vff', 'bright.vff'}
    vff_files = sorted(
        [f for f in scan_path.glob('*.vff') if f.name.lower() not in calibration_files],
        key=natural_sort_key
    )

    if len(vff_files) == 0:
        raise ValueError(f"No VFF projection files found in {scan_path}")

    print(f"Found {len(vff_files)} projection files")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy auxiliary files (calibration files and metadata)
    auxiliary_files = ['dark.vff', 'bright.vff', 'detector_values.dat', 'scan.xml']
    print("Copying auxiliary files...")
    for aux_file in auxiliary_files:
        src = scan_path / aux_file
        if src.exists():
            shutil.copy(src, output_dir / aux_file)
            print(f"  Copied {aux_file}")

    # Load model
    device = torch.device(args.device)
    model = UNet(in_ch=2, out_ch=1).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Determine missing projections (odd indices: 1, 3, 5, ...)
    missing_indices = list(range(1, len(vff_files), 2))
    # Even indices are the neighbors we keep
    even_indices = set(range(0, len(vff_files), 2))
    print(f"Predicting {len(missing_indices)} missing projections")

    # Copy only even-indexed projections (neighbors used for prediction)
    print("Copying neighbor projections...")
    for idx in even_indices:
        vff_file = vff_files[idx]
        shutil.copy(vff_file, output_dir / vff_file.name)

    # Predict missing projections
    print("Running inference...")
    for i, missing_idx in enumerate(missing_indices):
        if missing_idx + 1 >= len(vff_files):
            break  # Can't predict last if odd count

        # Load surrounding projections
        h1, a1 = vff_io.read_vff(str(vff_files[missing_idx - 1]), verbose=False)
        h3, a3 = vff_io.read_vff(str(vff_files[missing_idx + 1]), verbose=False)

        # Compute middle angle
        angle1 = float(h1['gantryPosition'])
        angle3 = float(h3['gantryPosition'])
        angle2 = (angle1 + angle3) / 2

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i + 1}/{len(missing_indices)}] "
                  f"Predicting index {missing_idx}: angles {angle1:.2f}° -> {angle2:.2f}° -> {angle3:.2f}°")

        # Create header for predicted projection
        h2 = h1.copy()
        h2['gantryPosition'] = angle2

        # Convert to same format as training
        a1 = a1.squeeze(0).byteswap().view(a1.dtype.newbyteorder())
        a3 = a3.squeeze(0).byteswap().view(a3.dtype.newbyteorder())
        inp = np.stack([a1, a3], axis=0)
        t = torch.from_numpy(inp).unsqueeze(0).float().to(device, non_blocking=True)

        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                pred = model(t).squeeze(0).cpu().numpy()

        # Scale to match input range
        pred *= (a1.max() / 2 + a3.max() / 2) / (pred.max())
        pred[pred < 0] = 0  # Ensure no negative values

        # Build output filename with _pred suffix
        original_name = vff_files[missing_idx].stem
        pred_filename = f"{original_name}_pred.vff"
        pred_path = output_dir / pred_filename

        # Write predicted projection
        write_vff(str(pred_path), h2, pred, verbose=False)

    print(f"Done. Output written to {output_dir}")


if __name__ == '__main__':
    main()
