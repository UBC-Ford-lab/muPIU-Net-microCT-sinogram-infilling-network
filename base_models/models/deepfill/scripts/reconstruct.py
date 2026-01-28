#!/usr/bin/env python3
"""
Reconstruction Pipeline for DeepFill v2-Infilled Sinograms
==========================================================
This script:
1. Loads DeepFill v2-infilled (merged) sinogram PNGs using memory-mapped arrays (RAM-safe)
2. Reconstructs them back into projection format
3. Runs FDK reconstruction using existing pipeline
4. Saves reconstruction results

Memory-safe implementation:
- Uses np.memmap to avoid loading 13GB into RAM
- Processes sinograms one at a time (~5-10 MB RAM usage)
- Flushes to disk every 100 sinograms
- FDKReconstructor already handles memmap arrays

Based on reconstruct_from_repaint.py structure.

Author: Claude (Anthropic)
Date: 2025-11-29
"""

import torch
import numpy as np
import xmltodict
import sys
import time
import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add ct_recon directory to path for imports (5 levels up from scripts/)
CT_RECON_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(CT_RECON_DIR))

from reconstruction.fdk import FDKReconstructor
from ct_core import tiff_converter


def load_metadata(metadata_path, tiling_metadata_path=None):
    """
    Load sinogram dataset metadata and optionally tiling metadata.

    Args:
        metadata_path: Path to original sinogram dataset metadata
        tiling_metadata_path: Path to tiling metadata (optional)

    Returns:
        metadata: Combined metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # If tiling metadata provided, merge it
    if tiling_metadata_path and os.path.exists(tiling_metadata_path):
        with open(tiling_metadata_path, 'r') as f:
            tiling_meta = json.load(f)
        metadata['tiling'] = tiling_meta

    return metadata


def load_infilled_sinograms(sinogram_folder, metadata, memmap_path, force_reload=False):
    """
    Load all infilled sinogram PNGs and reconstruct projection stack using memmap
    Memory-safe: processes sinograms one at a time, writes to disk-backed array

    Args:
        sinogram_folder: Path to folder containing sino_XXXX.png files
        metadata: Dictionary with dataset info and normalization parameters
        memmap_path: Path for temporary memory-mapped file
        force_reload: If True, recreate memmap even if it exists

    Returns:
        projections: memory-mapped array of shape (n_projections, height, width)
    """
    n_projections = metadata['n_projections']
    original_height = metadata['projection_height']
    original_width = metadata['projection_width']
    n_sinograms = metadata['n_sinograms']

    print(f"\nLoading {n_sinograms} infilled sinograms from {sinogram_folder}")
    print(f"Target projection shape: ({n_projections}, {original_height}, {original_width})")

    # Calculate expected file size
    expected_size = n_projections * original_height * original_width * 4  # 4 bytes per float32
    mem_gb = expected_size / (1024**3)

    # Check if memmap file already exists and is valid
    if os.path.exists(memmap_path) and not force_reload:
        actual_size = os.path.getsize(memmap_path)
        if actual_size == expected_size:
            print(f"Found existing memmap file: {memmap_path}")
            print(f"  Size: {mem_gb:.2f} GB")
            print(f"  Reusing cached projection data (skipping reload)")

            # Open existing memmap in read-write mode (FDKReconstructor modifies in-place)
            projections = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode='r+',
                shape=(n_projections, original_height, original_width)
            )
            return projections
        else:
            print(f"Existing memmap file has wrong size ({actual_size} vs {expected_size} expected)")
            print(f"  Recreating memmap file...")

    print(f"Creating memory-mapped file: {memmap_path}")
    print(f"Size: {mem_gb:.2f} GB (disk-backed, RAM-safe)")

    # Create memory-mapped array (disk-backed, won't fill RAM)
    projections = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode='w+',
        shape=(n_projections, original_height, original_width)
    )

    # Load each sinogram
    sinogram_files = sorted(Path(sinogram_folder).glob('sino_*.png'))

    if len(sinogram_files) == 0:
        raise FileNotFoundError(f"No sinogram files found in {sinogram_folder}")

    print(f"Found {len(sinogram_files)} sinogram files")
    print(f"Processing sinograms (RAM usage: ~5-10 MB)...")

    flush_interval = 100  # Flush to disk every 100 sinograms

    for idx, sino_file in enumerate(tqdm(sinogram_files, desc="Loading sinograms")):
        # Extract height index from filename: sino_0042.png -> 42
        height_idx = int(sino_file.stem.split('_')[1])

        # Load PNG with preserved bit depth using PIL
        try:
            img_pil = Image.open(sino_file)
            img_raw = np.array(img_pil)
        except Exception as e:
            print(f"\nWarning: Could not load {sino_file}, skipping ({e})")
            continue

        # Detect bit depth (8-bit or 16-bit)
        is_16bit = img_raw.dtype == np.uint16
        max_value = 65535.0 if is_16bit else 255.0

        # Handle both grayscale and RGB (for backward compatibility)
        if img_raw.ndim == 3:
            # RGB - take first channel (all channels identical for grayscale-converted-to-RGB)
            img = img_raw[:, :, 0].astype(np.float32)
        else:
            # Already grayscale (new 16-bit workflow)
            img = img_raw.astype(np.float32)

        # Get sinogram metadata
        sino_metadata = metadata['sinograms'][height_idx]

        # Sinograms should already have padding removed by merge_deepfill_tiles.py
        img_unpadded = img

        # Verify shape matches expected dimensions
        if img_unpadded.shape != (n_projections, original_width):
            print(f"\nWarning: Unexpected shape {img_unpadded.shape} for {sino_file}")
            print(f"  Expected: ({n_projections}, {original_width})")
            continue

        # Get normalization parameters for this sinogram
        min_val = sino_metadata['normalization_gt']['min']
        max_val = sino_metadata['normalization_gt']['max']

        # Denormalize: PNG is [0, max_value], map back to [min, max]
        img_denorm = (img_unpadded / max_value) * (max_val - min_val) + min_val

        # Place into projection array (write directly to disk-backed memmap)
        # Sinogram is shape (n_projections, width) and represents all projections at this height
        projections[:, height_idx, :] = img_denorm

        # Flush to disk periodically to ensure data is written
        if (idx + 1) % flush_interval == 0:
            projections.flush()

    # Final flush to ensure all data is written
    projections.flush()

    print(f"Loaded all sinograms into memory-mapped array")
    print(f"  Final shape: {projections.shape}")
    print(f"  Memmap file: {memmap_path}")
    print(f"  Computing value range (this may take a moment)...")

    # Sample-based range calculation to avoid loading all data
    sample_indices = np.random.choice(n_sinograms, size=min(100, n_sinograms), replace=False)
    sample_min = float('inf')
    sample_max = float('-inf')
    for idx in sample_indices:
        sample_min = min(sample_min, projections[:, idx, :].min())
        sample_max = max(sample_max, projections[:, idx, :].max())

    print(f"  Value range (sampled): [{sample_min:.1f}, {sample_max:.1f}]")

    return projections


def reconstruct_from_deepfill_output(
    infilled_folder='../data/sinograms_infilled',
    metadata_path='../../../shared/sinogram_dataset/metadata.json',
    tiling_metadata_path='../../repaint/data/sinogram_tiles/tiling_metadata.json',
    scan_folder=None,  # Must be provided - path to scan folder with scan.xml,
    output_folder='../results/reconstructed_volume',
    projection_spacing=None,
    keep_memmap=True,
    memmap_path=None,
    force_reload=False
    ):
    """
    Main reconstruction pipeline for DeepFill v2-infilled sinograms

    Args:
        infilled_folder: Path to DeepFill v2 merged output sinograms
        metadata_path: Path to dataset metadata JSON
        tiling_metadata_path: Path to tiling metadata JSON
        scan_folder: Path to original scan folder (for XML geometry)
        output_folder: Where to save reconstruction
        projection_spacing: Optional projection spacing override
        keep_memmap: If True, keep memmap file after reconstruction
        memmap_path: Path for temporary memory-mapped file
        force_reload: If True, recreate memmap even if it exists
    """
    start = time.time()

    # Convert to absolute paths
    infilled_folder = Path(infilled_folder).resolve()
    metadata_path = Path(metadata_path).resolve()
    scan_folder = Path(scan_folder).resolve()
    output_folder = Path(output_folder).resolve()

    print("="*70)
    print("DeepFill v2 Sinogram Reconstruction Pipeline")
    print("="*70)
    print(f"Infilled sinograms: {infilled_folder}")
    print(f"Metadata: {metadata_path}")
    print(f"Scan folder: {scan_folder}")
    print(f"Output folder: {output_folder}")

    # Load metadata
    metadata = load_metadata(metadata_path, tiling_metadata_path)

    # Auto-select memmap path if not provided
    if memmap_path is None:
        memmap_path = str(output_folder.parent / 'deepfill_projections_memmap.dat')
        print(f"Auto-selected memmap path: {memmap_path}")

    # Check available disk space
    memmap_dir = Path(memmap_path).parent
    memmap_dir.mkdir(parents=True, exist_ok=True)

    stat = shutil.disk_usage(memmap_dir)
    required_gb = (metadata['n_projections'] * metadata['projection_height'] *
                   metadata['projection_width'] * 4) / (1024**3)
    available_gb = stat.free / (1024**3)

    print(f"\nDisk space check:")
    print(f"  Required: {required_gb:.2f} GB")
    print(f"  Available: {available_gb:.2f} GB in {memmap_dir}")

    if available_gb < required_gb + 1:  # +1 GB buffer
        raise RuntimeError(
            f"Insufficient disk space! Need {required_gb:.2f} GB but only "
            f"{available_gb:.2f} GB available in {memmap_dir}"
        )

    # Load infilled sinograms and reconstruct projection array using memmap
    projections_readonly = load_infilled_sinograms(infilled_folder, metadata, memmap_path, force_reload)

    # Create a working copy for FDK (since it modifies data in-place)
    working_memmap_path = str(Path(memmap_path).parent / 'deepfill_projections_working.dat')
    print(f"\nCreating working copy for FDK reconstruction...")
    print(f"  Working memmap: {working_memmap_path}")

    # Create writable copy
    projections = np.memmap(
        working_memmap_path,
        dtype=np.float32,
        mode='w+',
        shape=projections_readonly.shape
    )

    # Copy data in chunks to avoid RAM overflow
    print(f"  Copying projection data (RAM-safe)...")
    chunk_size = 20  # Process 20 projections at a time
    for i in range(0, projections_readonly.shape[0], chunk_size):
        end_idx = min(i + chunk_size, projections_readonly.shape[0])
        projections[i:end_idx] = projections_readonly[i:end_idx]
        if (i + chunk_size) % 100 == 0:
            projections.flush()

    projections.flush()
    print(f"  Working copy created")

    # Load geometry from XML
    xml_file = scan_folder / 'scan.xml'
    if not xml_file.exists():
        raise FileNotFoundError(f"scan.xml not found at {xml_file}")

    print(f"\nLoading geometry from {xml_file}")
    header = xmltodict.parse(open(xml_file).read())

    # Extract geometry parameters
    sp = header['Series']['SeriesParams']
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

    print(f"Geometry parameters:")
    print(f"  Source-to-isocenter (R_s): {geometry['R_s']:.2f} mm")
    print(f"  Detector-to-isocenter (R_d): {geometry['R_d']:.2f} mm")
    print(f"  Detector spacing: {geometry['da']:.4f} mm")
    print(f"  Volume shape: {geometry['vol_shape']}")
    print(f"  Voxel size (dx): {geometry['dx']:.4f} mm")

    # Generate angles
    projection_spacing = 0.878049
    n_projections = metadata['n_projections']
    starting_angle_offset = float(header['Series']['AngleOffset']) + 120
    imaging_angle = projection_spacing * n_projections

    angles_deg = torch.linspace(
            starting_angle_offset,
            starting_angle_offset + imaging_angle,
            steps=n_projections,
            dtype=torch.float32
        ) % 360
    angles = torch.deg2rad(angles_deg)

    print(f"\nAngles:")
    print(f"  Number of projections: {n_projections}")
    print(f"  Angular range: [0, 2*pi]")
    print(f"  Angular spacing: {(2 * np.pi / n_projections):.6f} rad ({360 / n_projections:.2f} deg)")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize FDK reconstructor
    print(f"\nInitializing FDK reconstructor...")

    # Detect device (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Using GPU acceleration")
    else:
        print(f"No GPU available, using CPU")

    try:
        reconstructor = FDKReconstructor(
            projections=projections,
            angles=angles,
            geometry=geometry,
            source_locations=None,
            folder_name=str(output_folder)
        )

        # Run reconstruction
        print(f"\nStarting FDK reconstruction...")
        print(f"This may take 5-10 minutes with GPU acceleration...")
        reconstructor.reconstruct(display_volume=False)

        end = time.time()
        print(f"\n{'='*70}")
        print(f"Reconstruction complete!")
        print(f"Time elapsed: {(end - start)/60:.2f} minutes")
        print(f"Results saved to: {output_folder}")
        print(f"{'='*70}")

        # Cleanup memmap files after successful completion
        try:
            del projections
            if os.path.exists(working_memmap_path):
                os.unlink(working_memmap_path)
                print(f"\nCleaned up working memmap: {working_memmap_path}")
        except Exception as e:
            print(f"\nWarning: Could not delete working memmap {working_memmap_path}: {e}")

        if keep_memmap:
            print(f"Cached sinogram data preserved for reuse: {memmap_path}")
            print(f"  Size: {os.path.getsize(memmap_path) / (1024**3):.2f} GB")
        else:
            try:
                del projections_readonly
                if os.path.exists(memmap_path):
                    os.unlink(memmap_path)
                    print(f"Cleaned up cached sinogram data: {memmap_path}")
            except Exception as e:
                print(f"Warning: Could not delete cached data {memmap_path}: {e}")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Reconstruction failed!")
        print(f"{'='*70}")
        print(f"Error: {e}")
        print(f"\nMemmap files preserved for debugging:")
        print(f"  Cached sinograms: {memmap_path}")
        if 'working_memmap_path' in locals():
            print(f"  Working copy: {working_memmap_path}")
        raise


def main():
    """Main entry point with command-line argument support"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Reconstruct CT volumes from DeepFill v2-infilled sinograms'
    )
    parser.add_argument(
        '--sinogram_dir',
        type=str,
        default='../data/sinograms_infilled',
        help='Path to directory containing infilled sinogram PNGs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results/reconstructed_volume',
        help='Output directory for reconstruction results'
    )
    parser.add_argument(
        '--scan_folder',
        type=str,
        required=True,
        help='Path to scan folder containing original projections (for geometry/angles)'
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        default='../../../shared/sinogram_dataset/metadata.json',
        help='Path to dataset metadata file'
    )
    parser.add_argument(
        '--tiling_metadata_path',
        type=str,
        default='../../repaint/data/sinogram_tiles/tiling_metadata.json',
        help='Path to tiling metadata file'
    )
    parser.add_argument(
        '--force_reload',
        action='store_true',
        help='Force reload of cached memmap data'
    )

    args = parser.parse_args()

    # Handle paths
    sinogram_dir = Path(args.sinogram_dir) if not Path(args.sinogram_dir).is_absolute() else Path(args.sinogram_dir)
    output_dir = Path(args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    scan_folder = Path(args.scan_folder) if not Path(args.scan_folder).is_absolute() else Path(args.scan_folder)
    metadata_path = Path(args.metadata_path) if not Path(args.metadata_path).is_absolute() else Path(args.metadata_path)
    tiling_metadata_path = Path(args.tiling_metadata_path) if not Path(args.tiling_metadata_path).is_absolute() else Path(args.tiling_metadata_path)

    # Reconstruct DeepFill v2 infilled sinograms
    print("\n" + "="*80)
    print("RECONSTRUCTING DEEPFILL v2 INFILLED")
    print("="*80)
    reconstruct_from_deepfill_output(
        infilled_folder=sinogram_dir,
        metadata_path=metadata_path,
        tiling_metadata_path=tiling_metadata_path,
        scan_folder=scan_folder,
        output_folder=output_dir,
        force_reload=args.force_reload,
        keep_memmap=True
    )

    print("\n" + "="*80)
    print("RECONSTRUCTION COMPLETE!")
    print("="*80)
    print(f"DeepFill v2 reconstruction: {output_dir}.vff")

if __name__ == '__main__':
    main()
