"""
Run all three reconstructions (GT, Undersampled, U-Net) with the fixed FDK
that includes proper angular normalization.

Usage:
    python run_all_reconstructions.py           # Run all three (no HU)
    python run_all_reconstructions.py gt        # Run only GT (no HU)
    python run_all_reconstructions.py --hu      # Run all with HU calibration
    python run_all_reconstructions.py --hu gt   # Run only GT with HU calibration

Options:
    --hu    Enable Hounsfield Unit calibration (requires log transformation)
"""

import torch
import numpy as np
import xmltodict
import sys
import time
import os
from pathlib import Path

from reconstruction.fdk import FDKReconstructor
from ct_core.vff_io import VFFDataset
from ct_core.calibration import parse_calibration_from_xml, load_calibration_fields

# Default mu_water value (literature value at 80 keV)
# This can be overridden with empirical calibration
MU_WATER_DEFAULT = 0.00184  # mm^-1


# Configuration for each reconstruction
RECONSTRUCTIONS = {
    'gt': {
        'name': 'Ground Truth',
        'data_folder': 'data/results/Scan_1681_uwarp_gt',
        'output_folder': 'data/results/Scan_1681_gt_recon',
        'projection_spacing': 0.878049,  # Single step (full sampling)
        'n_expected_projections': 410,
        'scan_folder': 'data/scans/Scan_1681',  # Original scan with bright/dark fields
    },
    'under': {
        'name': 'Undersampled (No Prediction)',
        'data_folder': 'data/results/Scan_1681_uwarp_no_pred',
        'output_folder': 'data/results/Scan_1681_no_pred_recon',
        'projection_spacing': 2 * 0.878049,  # Double step (50% undersampling)
        'n_expected_projections': 205,
        'scan_folder': 'data/scans/Scan_1681',
    },
    'unet': {
        'name': 'U-Net (With Prediction)',
        'data_folder': 'data/results/Scan_1681_uwarp_with_pred',
        'output_folder': 'data/results/Scan_1681_with_pred_recon',
        'projection_spacing': 0.878049,  # Single step (predictions fill gaps)
        'n_expected_projections': 409,
        'scan_folder': 'data/scans/Scan_1681',
    },
}


def run_reconstruction(config_key, enable_hu=False, mu_water=None):
    """Run a single reconstruction based on configuration.

    Args:
        config_key: Key from RECONSTRUCTIONS dict ('gt', 'under', 'unet')
        enable_hu: If True, apply log transformation and HU conversion
        mu_water: Linear attenuation coefficient of water in mm^-1 (for HU conversion)
    """
    config = RECONSTRUCTIONS[config_key]

    print("=" * 80)
    print(f"RECONSTRUCTION: {config['name']}")
    if enable_hu:
        print("MODE: Hounsfield Unit calibration enabled")
    print("=" * 80)

    start = time.time()

    data_folder = config['data_folder']
    xml_file = os.path.join(data_folder, 'scan.xml')

    if not os.path.exists(xml_file):
        print(f"ERROR: XML file not found: {xml_file}")
        return False

    print(f"Data folder: {data_folder}")
    output_suffix = "_hu" if enable_hu else ""
    output_folder = config['output_folder'] + output_suffix
    print(f"Output: {output_folder}.vff")
    print(f"Projection spacing: {config['projection_spacing']:.6f} degrees")

    # Load HU calibration data if enabled
    air_value = None
    bright_field = None
    dark_field = None

    if enable_hu:
        print("\nLoading HU calibration data...")

        # Load bright/dark fields from original scan folder
        scan_folder = config['scan_folder']
        print(f"  Scan folder: {scan_folder}")

        try:
            bright_field, dark_field = load_calibration_fields(scan_folder)
            print(f"  Bright field loaded: {bright_field.shape}, range [{bright_field.min():.0f}, {bright_field.max():.0f}]")
            print(f"  Dark field loaded: {dark_field.shape}, range [{dark_field.min():.0f}, {dark_field.max():.0f}]")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Falling back to empirical calibration...")

        # Parse XML calibration parameters for reference
        calibration = parse_calibration_from_xml(xml_file)
        print(f"  XML Air Value: {calibration['air_value']}")
        print(f"  XML Water Value: {calibration['water_value']}")
        print(f"  XML Bone HU: {calibration['bone_hu']}")

        if mu_water is None:
            mu_water = MU_WATER_DEFAULT
            print(f"  Using literature μ_water: {mu_water:.6f} mm⁻¹ (80 keV)")

    # Load all projections and angles
    print("\nLoading projections...")
    dataset = VFFDataset(
        data_folder,
        xml_file,
        paths_str="uwarp*",
        projection_spacing=config['projection_spacing']
    )
    projections = dataset.projections  # shape (N_angles, N_b, N_a)
    angles = dataset.angles_rad        # shape (N_angles,)

    print(f"Loaded {len(angles)} projections")
    print(f"Expected: {config['n_expected_projections']}")
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
        folder_name=output_folder,
        air_value=air_value,
        mu_water=mu_water,
        output_hu=enable_hu,
        bright_field=bright_field,
        dark_field=dark_field
    )

    # Run full reconstruction
    print("\nRunning reconstruction pipeline...")
    reconstructor.reconstruct(display_volume=False)

    end = time.time()
    elapsed = (end - start) / 60

    print(f"\n{config['name']} reconstruction finished in {elapsed:.2f} minutes.")
    print("=" * 80)

    return True


def main():
    """Main function to run reconstructions."""

    # Parse command line arguments
    args = sys.argv[1:]
    enable_hu = '--hu' in args
    if enable_hu:
        args.remove('--hu')

    if len(args) > 0:
        keys_to_run = [arg.lower() for arg in args]
        for key in keys_to_run:
            if key not in RECONSTRUCTIONS:
                print(f"Unknown reconstruction key: {key}")
                print(f"Valid keys: {list(RECONSTRUCTIONS.keys())}")
                print("Options: --hu (enable Hounsfield Unit calibration)")
                sys.exit(1)
    else:
        # Run all reconstructions
        keys_to_run = list(RECONSTRUCTIONS.keys())

    print("\n" + "=" * 80)
    if enable_hu:
        print("FDK RECONSTRUCTION WITH HOUNSFIELD UNIT CALIBRATION")
    else:
        print("FDK RECONSTRUCTION WITH ANGULAR NORMALIZATION")
    print("=" * 80)
    print(f"Reconstructions to run: {keys_to_run}")
    print(f"HU calibration: {'ENABLED' if enable_hu else 'disabled'}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 80 + "\n")

    total_start = time.time()
    results = {}

    for key in keys_to_run:
        try:
            success = run_reconstruction(key, enable_hu=enable_hu)
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
        config = RECONSTRUCTIONS[key]
        suffix = " (HU)" if enable_hu else ""
        print(f"  {config['name']}{suffix}: {status}")
    print(f"\nTotal time: {total_elapsed:.2f} minutes")
    print("=" * 80)


if __name__ == '__main__':
    main()
