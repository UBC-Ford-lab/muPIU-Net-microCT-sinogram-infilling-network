"""
Run FDK reconstructions (GT, Undersampled, U-Net) with proper angular normalization.

Usage:
    python run_all.py --scan_folder /path/to/scan --mode all
    python run_all.py --scan_folder /path/to/scan --unet_folder data/results/unet_output --mode unet
    python run_all.py --scan_folder /path/to/scan --mode gt

Options:
    --scan_folder   Required: Path to original scan folder (with bright/dark fields, scan.xml)
    --unet_folder   Folder with U-Net predictions (from infer.py output)
    --output_dir    Base output directory (default: data/results)
    --mode          Which reconstruction(s) to run: gt, under, unet, or all (default: all)
"""

import argparse
import torch
import numpy as np
import xmltodict
import time
import os
from pathlib import Path

from reconstruction.fdk import FDKReconstructor
from ct_core.vff_io import VFFDataset

# Default projection spacing (degrees per projection)
DEFAULT_PROJECTION_SPACING = 0.878049


def parse_args():
    p = argparse.ArgumentParser(
        description="Run FDK reconstructions (GT, Undersampled, U-Net)"
    )
    p.add_argument('--scan_folder', type=str, required=True,
                   help='Path to original scan folder (with bright/dark fields, scan.xml)')
    p.add_argument('--unet_folder', type=str, default=None,
                   help='Folder with U-Net predictions (from infer.py output)')
    p.add_argument('--output_dir', type=str, default='data/results',
                   help='Base output directory (default: data/results)')
    p.add_argument('--mode', type=str, default='all', choices=['gt', 'under', 'unet', 'all'],
                   help='Which reconstruction(s) to run: gt, under, unet, or all (default: all)')
    return p.parse_args()


def run_reconstruction(data_folder, scan_folder, output_folder, name, projection_spacing,
                       index_stride=1):
    """Run a single reconstruction.

    Args:
        data_folder: Folder containing projection VFF files
        scan_folder: Original scan folder (for bright/dark fields and XML)
        output_folder: Output path for reconstruction (without .vff extension)
        name: Display name for this reconstruction
        projection_spacing: Angular spacing between projections in degrees
        index_stride: Take every Nth projection (1 = all, 2 = every 2nd, etc.)
    """
    print("=" * 80)
    print(f"RECONSTRUCTION: {name}")
    if index_stride > 1:
        print(f"SUBSAMPLING: Using every {index_stride}th projection")
    print("=" * 80)

    start = time.time()

    xml_file = os.path.join(scan_folder, 'scan.xml')

    if not os.path.exists(xml_file):
        print(f"ERROR: XML file not found: {xml_file}")
        return False

    print(f"Data folder: {data_folder}")
    print(f"Output: {output_folder}.vff")
    print(f"Projection spacing: {projection_spacing:.6f} degrees")

    # Load all projections and angles
    print("\nLoading projections...")
    dataset = VFFDataset(
        data_folder,
        xml_file,
        paths_str="uwarp*",
        projection_spacing=projection_spacing,
        index_stride=index_stride
    )
    projections = dataset.projections  # shape (N_angles, N_b, N_a)
    angles = dataset.angles_rad        # shape (N_angles,)

    print(f"Loaded {len(angles)} projections")
    print(f"Projection shape: {projections.shape}")
    print(f"Angular range: {float(angles[0]):.4f} to {float(angles[-1]):.4f} rad")
    print(f"Angular range: {float(angles[0]) * 180 / np.pi:.2f} to {float(angles[-1]) * 180 / np.pi:.2f} degrees")

    # Define geometry from XML
    header = xmltodict.parse(open(xml_file).read())
    source_to_isocenter = float(header['Series']['ObjectPosition'])
    detector_to_isocenter = float(header['Series']['DetectorPosition']) - source_to_isocenter

    geometry = {
        'R_d': detector_to_isocenter,
        'R_s': source_to_isocenter,
        'da': float(header['Series']['DetectorSpacing']),
        'db': float(header['Series']['DetectorSpacing']),
        'vol_shape': (1100, 1100, 300),
        'vol_origin': (0, 0, 0),
        'dx': 0.085,
        'dz': 0.4,
        'central_pixel_a': float(header['Series']['CentreOfRotation']),
        'central_pixel_b': float(header['Series']['CentralSlice'])
    }

    print(f"\nGeometry:")
    print(f"  Source-to-isocenter: {geometry['R_s']:.2f} mm")
    print(f"  Detector-to-isocenter: {geometry['R_d']:.2f} mm")
    print(f"  Volume shape: {geometry['vol_shape']}")
    print(f"  Voxel size: {geometry['dx']:.3f} x {geometry['dx']:.3f} x {geometry['dz']:.3f} mm")

    # Initialize reconstructor
    print("\nInitializing FDK reconstructor...")
    reconstructor = FDKReconstructor(
        projections=projections,
        angles=angles,
        geometry=geometry,
        source_locations=None,
        folder_name=output_folder
    )

    # Run full reconstruction
    print("\nRunning reconstruction pipeline...")
    reconstructor.reconstruct(display_volume=False)

    end = time.time()
    elapsed = (end - start) / 60

    print(f"\n{name} reconstruction finished in {elapsed:.2f} minutes.")
    print("=" * 80)

    return True


def main():
    """Main function to run reconstructions."""
    args = parse_args()

    # Derive scan name from scan_folder
    scan_name = Path(args.scan_folder).name

    # Determine which modes to run
    if args.mode == 'all':
        modes_to_run = ['gt', 'under', 'unet']
    else:
        modes_to_run = [args.mode]

    # Build configuration for each mode
    configs = {}

    if 'gt' in modes_to_run:
        configs['gt'] = {
            'name': 'Ground Truth',
            'data_folder': args.scan_folder,
            'output_folder': os.path.join(args.output_dir, f'{scan_name}_gt_recon'),
            'projection_spacing': DEFAULT_PROJECTION_SPACING,  # Single step (full sampling)
            'index_stride': 1,  # Use all projections
        }

    if 'under' in modes_to_run:
        configs['under'] = {
            'name': 'Undersampled (No Prediction)',
            'data_folder': args.scan_folder,
            'output_folder': os.path.join(args.output_dir, f'{scan_name}_under_recon'),
            'projection_spacing': 2 * DEFAULT_PROJECTION_SPACING,  # Double step (50% undersampling)
            'index_stride': 2,  # Take every 2nd projection
        }

    if 'unet' in modes_to_run:
        if args.unet_folder is None:
            print("WARNING: --unet_folder not provided, skipping 'unet' mode")
        else:
            configs['unet'] = {
                'name': 'U-Net (With Prediction)',
                'data_folder': args.unet_folder,
                'output_folder': os.path.join(args.output_dir, f'{scan_name}_unet_recon'),
                'projection_spacing': DEFAULT_PROJECTION_SPACING,  # Single step (predictions fill gaps)
                'index_stride': 1,  # Use all files from U-Net output
            }

    if not configs:
        print("ERROR: No valid reconstructions to run. Check your arguments.")
        return

    print("\n" + "=" * 80)
    print("FDK RECONSTRUCTION WITH ANGULAR NORMALIZATION")
    print("=" * 80)
    print(f"Scan folder: {args.scan_folder}")
    print(f"Reconstructions to run: {list(configs.keys())}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 80 + "\n")

    total_start = time.time()
    results = {}

    for key, config in configs.items():
        try:
            success = run_reconstruction(
                data_folder=config['data_folder'],
                scan_folder=args.scan_folder,
                output_folder=config['output_folder'],
                name=config['name'],
                projection_spacing=config['projection_spacing'],
                index_stride=config['index_stride'],
            )
            results[key] = 'SUCCESS' if success else 'FAILED'
        except Exception as e:
            print(f"ERROR in {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = f'ERROR: {e}'

    total_elapsed = (time.time() - total_start) / 60

    # Summary
    print("\n" + "=" * 80)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 80)
    for key, status in results.items():
        config = configs[key]
        print(f"  {config['name']}: {status}")
    print(f"\nTotal time: {total_elapsed:.2f} minutes")
    print("=" * 80)


if __name__ == '__main__':
    main()
