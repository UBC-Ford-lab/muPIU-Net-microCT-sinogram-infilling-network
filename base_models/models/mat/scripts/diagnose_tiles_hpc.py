#!/usr/bin/env python3
"""
MAT Tile Zero Pixel Diagnostic Script (HPC Version)
=====================================================
Analyzes MAT output tiles to determine if zeros originate from:
1. The MAT model itself (tiles contain zeros before merging)
2. The merging process (Gaussian blending creates zeros)

Run via SLURM: sbatch diagnose_tiles.sh

Author: Claude (Anthropic)
Date: 2025-01-03
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image

# HPC paths
PROJECT = Path('/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison')
MAT_TILES_DIR = PROJECT / 'models/mat/data/tiles_infilled'
DEEPFILL_TILES_DIR = PROJECT / 'models/deepfill/data/tiles_infilled'
MAT_SINOGRAMS_DIR = PROJECT / 'models/mat/data/sinograms_infilled'
DEEPFILL_SINOGRAMS_DIR = PROJECT / 'models/deepfill/data/sinograms_infilled'


def analyze_directory(dir_path, name, sample_size=100):
    """Analyze tiles/sinograms in a directory for zeros."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Path: {dir_path}")
    print('='*60)

    if not dir_path.exists():
        print(f"  ERROR: Directory does not exist!")
        return None

    # Find all PNG files
    png_files = sorted(dir_path.glob('*.png'))
    total_files = len(png_files)

    if total_files == 0:
        print(f"  ERROR: No PNG files found!")
        return None

    print(f"  Total files: {total_files}")

    # Sample files for analysis
    if total_files <= sample_size:
        sample_files = png_files
    else:
        sample_files = random.sample(png_files, sample_size)

    print(f"  Sampling: {len(sample_files)} files")

    # Statistics
    stats = {
        'total_files': total_files,
        'sampled_files': len(sample_files),
        'files_with_zeros': 0,
        'total_zeros': 0,
        'total_pixels': 0,
        'min_vals': [],
        'max_vals': [],
        'zero_files': []
    }

    for i, f in enumerate(sample_files):
        try:
            img = np.array(Image.open(f))
            stats['total_pixels'] += img.size

            min_val = img.min()
            max_val = img.max()
            stats['min_vals'].append(min_val)
            stats['max_vals'].append(max_val)

            zero_count = np.sum(img == 0)
            if zero_count > 0:
                stats['files_with_zeros'] += 1
                stats['total_zeros'] += zero_count
                stats['zero_files'].append((f.name, zero_count, img.shape))

            # Progress
            if (i + 1) % 20 == 0:
                print(f"    Processed {i+1}/{len(sample_files)} files...")

        except Exception as e:
            print(f"    Error loading {f.name}: {e}")

    # Report
    print(f"\n  Results:")
    print(f"    Files with zeros: {stats['files_with_zeros']} / {stats['sampled_files']} ({100*stats['files_with_zeros']/stats['sampled_files']:.1f}%)")
    print(f"    Total zero pixels: {stats['total_zeros']} / {stats['total_pixels']} ({100*stats['total_zeros']/stats['total_pixels']:.4f}%)")
    print(f"    Min value range: {min(stats['min_vals'])} - {max(stats['min_vals'])}")
    print(f"    Max value range: {min(stats['max_vals'])} - {max(stats['max_vals'])}")

    if stats['zero_files']:
        print(f"\n  Sample files with zeros (first 10):")
        for fname, zcount, shape in stats['zero_files'][:10]:
            print(f"    {fname}: {zcount} zeros, shape {shape}")

    return stats


def compare_results(mat_stats, deepfill_stats):
    """Compare MAT vs DeepFill results."""
    print("\n" + "="*60)
    print("COMPARISON: MAT vs DeepFill")
    print("="*60)

    if mat_stats is None or deepfill_stats is None:
        print("  Cannot compare - missing data")
        return

    print(f"\n  {'Metric':<30} {'MAT':>15} {'DeepFill':>15}")
    print(f"  {'-'*60}")

    print(f"  {'Files with zeros':<30} {mat_stats['files_with_zeros']:>15} {deepfill_stats['files_with_zeros']:>15}")
    print(f"  {'Total zeros':<30} {mat_stats['total_zeros']:>15} {deepfill_stats['total_zeros']:>15}")
    print(f"  {'Min value (lowest)':<30} {min(mat_stats['min_vals']):>15} {min(deepfill_stats['min_vals']):>15}")
    print(f"  {'Max value (highest)':<30} {max(mat_stats['max_vals']):>15} {max(deepfill_stats['max_vals']):>15}")


def diagnose(mat_stats, deepfill_stats):
    """Provide diagnosis based on findings."""
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if mat_stats is None:
        print("  Cannot diagnose - MAT data missing")
        return

    mat_has_zeros = mat_stats['files_with_zeros'] > 0
    mat_min_zero = min(mat_stats['min_vals']) == 0

    deepfill_has_zeros = deepfill_stats is not None and deepfill_stats['files_with_zeros'] > 0
    deepfill_min_zero = deepfill_stats is not None and min(deepfill_stats['min_vals']) == 0

    print(f"\n  MAT tiles contain zeros: {'YES' if mat_has_zeros else 'NO'}")
    print(f"  MAT minimum value is 0: {'YES' if mat_min_zero else 'NO'}")
    print(f"  DeepFill tiles contain zeros: {'YES' if deepfill_has_zeros else 'NO'}")
    print(f"  DeepFill minimum value is 0: {'YES' if deepfill_min_zero else 'NO'}")

    print("\n  CONCLUSION:")
    if mat_has_zeros and not deepfill_has_zeros:
        print("  >>> MAT MODEL outputs zeros - the issue is in MAT inference, not merging!")
        print("  >>> The zeros exist BEFORE tile merging occurs.")
        print("  >>> Possible causes:")
        print("      - MAT model outputs -1.0 for boundary/masked regions")
        print("      - MAT model doesn't fully inpaint masked regions")
        print("      - Normalization issue in run_inference.py")
    elif mat_has_zeros and deepfill_has_zeros:
        print("  >>> BOTH models output zeros - check if this is expected behavior")
    elif not mat_has_zeros:
        print("  >>> MAT tiles are CLEAN (no zeros)")
        print("  >>> The zeros in sinograms must come from the MERGING process")
        print("  >>> Solution: Re-merge with --blend_mode nearest")
    else:
        print("  >>> Inconclusive - need more investigation")


def main():
    print("="*60)
    print("MAT Tile Zero Pixel Diagnostic")
    print("="*60)

    # Set random seed for reproducible sampling
    random.seed(42)

    # Check what directories exist
    print("\nChecking directories...")
    print(f"  MAT tiles:       {'EXISTS' if MAT_TILES_DIR.exists() else 'NOT FOUND'}")
    print(f"  DeepFill tiles:  {'EXISTS' if DEEPFILL_TILES_DIR.exists() else 'NOT FOUND'}")
    print(f"  MAT sinograms:   {'EXISTS' if MAT_SINOGRAMS_DIR.exists() else 'NOT FOUND'}")
    print(f"  DeepFill sinos:  {'EXISTS' if DEEPFILL_SINOGRAMS_DIR.exists() else 'NOT FOUND'}")

    # Analyze MAT tiles (primary focus)
    mat_tile_stats = analyze_directory(MAT_TILES_DIR, "MAT Tiles (before merging)", sample_size=100)

    # Analyze DeepFill tiles for comparison
    deepfill_tile_stats = analyze_directory(DEEPFILL_TILES_DIR, "DeepFill Tiles (for comparison)", sample_size=100)

    # Also check merged sinograms
    mat_sino_stats = analyze_directory(MAT_SINOGRAMS_DIR, "MAT Sinograms (after merging)", sample_size=50)
    deepfill_sino_stats = analyze_directory(DEEPFILL_SINOGRAMS_DIR, "DeepFill Sinograms (after merging)", sample_size=50)

    # Compare results
    print("\n\n" + "#"*60)
    print("# TILE COMPARISON (before merging)")
    print("#"*60)
    compare_results(mat_tile_stats, deepfill_tile_stats)

    print("\n\n" + "#"*60)
    print("# SINOGRAM COMPARISON (after merging)")
    print("#"*60)
    compare_results(mat_sino_stats, deepfill_sino_stats)

    # Diagnosis
    diagnose(mat_tile_stats, deepfill_tile_stats)

    print("\n" + "="*60)
    print("Diagnostic Complete")
    print("="*60)


if __name__ == '__main__':
    main()
