#!/usr/bin/env python3
"""
DEPRECATED: Split Sinogram Tiles for SLURM Array Jobs

================================================================================
WARNING: THIS SCRIPT IS DEPRECATED!

The new workflow uses start_idx/end_idx parameters in the data loader config
to select tile ranges for each GPU. This eliminates file duplication and saves
~200k files across 4 GPUs.

Instead of running this script, the run_repaint_cedar_h100.sh script now:
1. Copies tiles ONCE to local NVMe storage
2. Each GPU processes a range using start_idx/end_idx in the config
3. No splits directory is created

To clean up existing splits directories, run:
    bash cleanup_repaint_files.sh

================================================================================

Original description:
Splits the tile dataset into N equal batches for parallel processing
across multiple GPUs using SLURM array jobs.

Author: Claude (Anthropic)
Date: 2025-11-20
"""

import warnings
warnings.warn(
    "split_sinograms_for_array.py is DEPRECATED. "
    "The new workflow uses start_idx/end_idx for range selection. "
    "Run cleanup_repaint_files.sh to remove the splits directory.",
    DeprecationWarning
)

import argparse
import shutil
from pathlib import Path
from typing import List
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split tiles for parallel processing across array jobs'
    )
    parser.add_argument(
        '--input_gt_dir',
        type=str,
        required=True,
        help='Input directory containing GT tiles'
    )
    parser.add_argument(
        '--input_mask_dir',
        type=str,
        required=True,
        help='Input directory containing mask tiles'
    )
    parser.add_argument(
        '--output_base',
        type=str,
        required=True,
        help='Base output directory for splits'
    )
    parser.add_argument(
        '--num_splits',
        type=int,
        required=True,
        help='Number of splits (should match SLURM array size)'
    )
    return parser.parse_args()


def split_tiles(
    input_gt_dir: Path,
    input_mask_dir: Path,
    output_base: Path,
    num_splits: int
):
    """
    Split tiles into N equal batches for parallel processing.

    Creates directory structure:
    output_base/
        split_00/
            sinograms_gt/
            masks/
        split_01/
            sinograms_gt/
            masks/
        ...
    """

    # Get all GT tiles
    gt_path = Path(input_gt_dir)
    mask_path = Path(input_mask_dir)

    if not gt_path.exists():
        raise FileNotFoundError(f"GT directory not found: {input_gt_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {input_mask_dir}")

    # Get sorted list of tiles
    gt_files = sorted(gt_path.glob('*.png'))
    total_tiles = len(gt_files)

    if total_tiles == 0:
        raise ValueError(f"No PNG files found in {input_gt_dir}")

    # Calculate tiles per split
    tiles_per_split = total_tiles // num_splits
    remainder = total_tiles % num_splits

    print(f"{'='*70}")
    print(f"SPLITTING {total_tiles:,} TILES INTO {num_splits} BATCHES")
    print(f"{'='*70}")
    print(f"Tiles per split: {tiles_per_split:,}")
    if remainder > 0:
        print(f"Remainder: {remainder} (will be distributed to first {remainder} splits)")
    print()

    # Create splits
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)

    tile_idx = 0

    for split_id in range(num_splits):
        # Determine number of tiles for this split
        split_size = tiles_per_split + (1 if split_id < remainder else 0)

        # Create split directories
        split_dir = output_path / f'split_{split_id:02d}'
        split_gt_dir = split_dir / 'sinograms_gt'
        split_mask_dir = split_dir / 'masks'

        split_gt_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating split {split_id:02d}...")
        print(f"  Output: {split_dir}")
        print(f"  Tiles: {split_size:,}")

        # Copy tiles for this split
        tiles_copied = 0
        for _ in range(split_size):
            if tile_idx >= total_tiles:
                break

            gt_file = gt_files[tile_idx]
            tile_name = gt_file.name

            # Find corresponding mask
            mask_file = mask_path / tile_name

            if not mask_file.exists():
                print(f"  WARNING: Mask not found for {tile_name}")
                tile_idx += 1
                continue

            # Create symlinks (faster than copying)
            gt_link = split_gt_dir / tile_name
            mask_link = split_mask_dir / tile_name

            # Remove existing links if any
            if gt_link.exists():
                gt_link.unlink()
            if mask_link.exists():
                mask_link.unlink()

            # Create relative symlinks
            gt_link.symlink_to(gt_file.resolve())
            mask_link.symlink_to(mask_file.resolve())

            tiles_copied += 1
            tile_idx += 1

        print(f"  ✓ Created {tiles_copied:,} tile links")
        print()

    # Verify split
    print(f"{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    total_split_tiles = 0
    for split_id in range(num_splits):
        split_dir = output_path / f'split_{split_id:02d}'
        split_tiles = len(list((split_dir / 'sinograms_gt').glob('*.png')))
        total_split_tiles += split_tiles
        print(f"Split {split_id:02d}: {split_tiles:,} tiles")

    print(f"\nTotal: {total_split_tiles:,} tiles")

    if total_split_tiles == total_tiles:
        print("✓ All tiles successfully distributed!")
    else:
        print(f"⚠ WARNING: Expected {total_tiles:,} but got {total_split_tiles:,}")

    # Save split metadata
    metadata = {
        'total_tiles': total_tiles,
        'num_splits': num_splits,
        'tiles_per_split': tiles_per_split,
        'splits': {}
    }

    for split_id in range(num_splits):
        split_dir = output_path / f'split_{split_id:02d}'
        split_tiles = len(list((split_dir / 'sinograms_gt').glob('*.png')))
        metadata['splits'][split_id] = {
            'path': str(split_dir),
            'num_tiles': split_tiles
        }

    metadata_file = output_path / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_file}")
    print(f"{'='*70}")


def main():
    args = parse_args()

    split_tiles(
        input_gt_dir=args.input_gt_dir,
        input_mask_dir=args.input_mask_dir,
        output_base=args.output_base,
        num_splits=args.num_splits
    )


if __name__ == '__main__':
    main()
