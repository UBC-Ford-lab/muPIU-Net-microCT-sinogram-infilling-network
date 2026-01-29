#!/usr/bin/env python3
"""
Create Tiled Dataset for Inpainting Model Inference
====================================================
This script takes the full-resolution sinogram dataset and creates 256×256 tiles
for inpainting models, with proper overlap for seamless merging.

Shared by: RePaint, MAT, DeepFill

Key features:
- Splits 416×3504 sinograms into 256×256 tiles with 32-pixel overlap
- Inverts masks (LaMa: 255=inpaint → inpainting models: 0=inpaint)
- Preserves normalization metadata for later merging
- Creates tiles for both GT and masked sinograms

Author: Claude (Anthropic)
Date: 2025-11-18
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Script directory references
SCRIPT_DIR = Path(__file__).parent
SHARED_DIR = SCRIPT_DIR.parent  # shared/


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create 256×256 tiles from sinogram dataset for inpainting models'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='../sinogram_dataset',
        help='Input directory containing original sinograms'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../sinogram_tiles',
        help='Output directory for tiled dataset'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Size of each tile (default: 256)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=32,
        help='Overlap between tiles in pixels (default: 32 for blending)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of tiles even if they exist'
    )
    return parser.parse_args()


class TileCreator:
    """Creates overlapping tiles from full-resolution sinograms."""

    def __init__(self, tile_size: int = 256, overlap: int = 32):
        """
        Initialize tile creator.

        Args:
            tile_size: Size of each square tile (default: 256)
            overlap: Overlap between adjacent tiles (default: 32)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def calculate_tile_grid(self, height: int, width: int) -> Tuple[int, int, List, List]:
        """
        Calculate tile positions for a given image size.

        Args:
            height: Image height
            width: Image width

        Returns:
            n_tiles_h: Number of tiles vertically
            n_tiles_w: Number of tiles horizontally
            tile_positions_h: List of (start, end) positions for height
            tile_positions_w: List of (start, end) positions for width
        """
        # Calculate number of tiles needed
        n_tiles_h = int(np.ceil((height - self.overlap) / self.stride))
        n_tiles_w = int(np.ceil((width - self.overlap) / self.stride))

        # Calculate tile positions
        tile_positions_h = []
        for i in range(n_tiles_h):
            start = i * self.stride
            end = min(start + self.tile_size, height)
            # Adjust start if we're at the edge
            if end == height and end - start < self.tile_size:
                start = max(0, height - self.tile_size)
            tile_positions_h.append((start, end))

        tile_positions_w = []
        for i in range(n_tiles_w):
            start = i * self.stride
            end = min(start + self.tile_size, width)
            # Adjust start if we're at the edge
            if end == width and end - start < self.tile_size:
                start = max(0, width - self.tile_size)
            tile_positions_w.append((start, end))

        return n_tiles_h, n_tiles_w, tile_positions_h, tile_positions_w

    def extract_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract overlapping tiles from an image.

        Args:
            image: Input image array (H, W) or (H, W, C)

        Returns:
            tiles: List of tile arrays
            tile_info: Dictionary with tiling metadata
        """
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width = image.shape[:2]

        n_tiles_h, n_tiles_w, pos_h, pos_w = self.calculate_tile_grid(height, width)

        tiles = []
        tile_metadata = []

        for i, (y_start, y_end) in enumerate(pos_h):
            for j, (x_start, x_end) in enumerate(pos_w):
                # Extract tile
                if image.ndim == 2:
                    tile = image[y_start:y_end, x_start:x_end]
                else:
                    tile = image[y_start:y_end, x_start:x_end, :]

                tiles.append(tile)

                # Store metadata for reconstruction
                tile_metadata.append({
                    'tile_idx': len(tiles) - 1,
                    'row': i,
                    'col': j,
                    'y_start': int(y_start),
                    'y_end': int(y_end),
                    'x_start': int(x_start),
                    'x_end': int(x_end),
                })

        tile_info = {
            'n_tiles_h': n_tiles_h,
            'n_tiles_w': n_tiles_w,
            'n_tiles_total': len(tiles),
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'original_shape': [int(height), int(width)],
            'tiles': tile_metadata
        }

        return tiles, tile_info


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Invert mask from LaMa format to inpainting model format.

    LaMa: 255 = inpaint region, 0 = keep
    Inpainting models: 0 = inpaint region, 255 = keep

    Args:
        mask: Input mask array

    Returns:
        Inverted mask array
    """
    return 255 - mask


def normalize_for_repaint(image: np.ndarray, dtype=np.uint16) -> np.ndarray:
    """
    Normalize image for inpainting models (preserving uint16 precision).
    Model dataloaders will normalize to [-1, 1] float32.

    Args:
        image: Input image (any bit depth)
        dtype: Output dtype (default: uint16 for precision)

    Returns:
        Normalized image in [0, 65535] uint16
    """
    if image.dtype == np.uint16:
        # Keep as uint16
        image_normalized = image.astype(dtype)
    elif image.dtype == np.uint8:
        # Convert 8-bit to 16-bit
        image_normalized = (image.astype(np.float32) / 255.0 * 65535.0).astype(dtype)
    else:
        # Float or other - assume already in [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max > 1.0:
            # Assume [0, 65535] range
            image_normalized = image.astype(dtype)
        else:
            # Assume [0, 1] range
            image_normalized = (image * 65535.0).astype(dtype)

    return image_normalized


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    metadata_path = input_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    n_sinograms = metadata['n_sinograms']
    print(f"\nDataset info:")
    print(f"  Sinograms: {n_sinograms}")
    print(f"  Original shape: {metadata['sinograms'][0]['original_shape']}")
    print(f"  Padded shape: {metadata['sinograms'][0]['padded_shape']}")

    # Create output directories
    output_gt_dir = output_dir / 'sinograms_gt'
    output_masked_dir = output_dir / 'sinograms_masked'
    output_mask_dir = output_dir / 'masks'

    for dir_path in [output_gt_dir, output_masked_dir, output_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Check if tiles already exist
    if not args.force:
        existing_tiles = list(output_gt_dir.glob('sino_*_tile_*.png'))
        if len(existing_tiles) > 0:
            print(f"\nFound {len(existing_tiles)} existing tiles.")
            response = input("Recreate tiles? (y/N): ")
            if response.lower() != 'y':
                print("Skipping tile creation.")
                return

    # Initialize tile creator
    tile_creator = TileCreator(tile_size=args.tile_size, overlap=args.overlap)

    # Process each sinogram
    print(f"\nCreating tiles:")
    print(f"  Tile size: {args.tile_size}×{args.tile_size}")
    print(f"  Overlap: {args.overlap} pixels")
    print(f"  Stride: {tile_creator.stride} pixels")

    # Storage for tiling metadata
    tiling_metadata = {
        'tile_size': args.tile_size,
        'overlap': args.overlap,
        'stride': tile_creator.stride,
        'sinograms': {}
    }

    total_tiles = 0

    for sino_idx in tqdm(range(n_sinograms), desc="Processing sinograms"):
        sino_name = f"sino_{sino_idx:04d}.png"

        # Load GT sinogram
        gt_path = input_dir / 'sinograms_gt' / sino_name
        gt_img = Image.open(gt_path)
        gt_array = np.array(gt_img)

        # Load mask
        mask_name = f"sino_{sino_idx:04d}_mask001.png"
        mask_path = input_dir / 'masks' / mask_name
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)

        # Normalize GT image to uint16 for inpainting models
        gt_normalized = normalize_for_repaint(gt_array)

        # Create masked version by applying the mask
        # LaMa mask: 255 = inpaint region, 0 = keep
        # Set masked regions to 0 (will be inpainted by model)
        masked_normalized = gt_normalized.copy()
        masked_normalized[mask_array > 127] = 0  # Set inpaint regions to 0

        # Invert mask for inpainting models (0 = inpaint, 255 = keep)
        mask_array_repaint = invert_mask(mask_array)

        # Extract tiles
        gt_tiles, tile_info = tile_creator.extract_tiles(gt_normalized)
        masked_tiles, _ = tile_creator.extract_tiles(masked_normalized)
        mask_tiles, _ = tile_creator.extract_tiles(mask_array_repaint)

        # Save tiles
        for tile_idx, (gt_tile, masked_tile, mask_tile) in enumerate(zip(gt_tiles, masked_tiles, mask_tiles)):
            tile_name = f"sino_{sino_idx:04d}_tile_{tile_idx:02d}.png"

            # Save GT and masked tiles as 16-bit grayscale PNG
            # Dataloader will convert grayscale → RGB when loading
            Image.fromarray(gt_tile.astype(np.uint16)).save(output_gt_dir / tile_name)
            Image.fromarray(masked_tile.astype(np.uint16)).save(output_masked_dir / tile_name)

            # Save mask as 8-bit PNG (binary mask doesn't need 16-bit precision)
            # This ensures proper normalization in dataloader (0-255 → 0-1)
            Image.fromarray(mask_tile.astype(np.uint8)).save(output_mask_dir / tile_name)

        total_tiles += len(gt_tiles)

        # Store tiling info for this sinogram
        tiling_metadata['sinograms'][sino_idx] = {
            'sino_name': sino_name,
            'original_shape': metadata['sinograms'][sino_idx]['original_shape'],
            'padded_shape': metadata['sinograms'][sino_idx]['padded_shape'],
            'padding': metadata['sinograms'][sino_idx]['padding'],
            'normalization_gt': metadata['sinograms'][sino_idx]['normalization_gt'],
            'tile_info': tile_info
        }

    # Save tiling metadata
    tiling_metadata['n_sinograms'] = n_sinograms
    tiling_metadata['total_tiles'] = total_tiles

    metadata_out_path = output_dir / 'tiling_metadata.json'
    with open(metadata_out_path, 'w') as f:
        json.dump(tiling_metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TILE CREATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total tiles created: {total_tiles}")
    print(f"  GT tiles: {output_gt_dir}")
    print(f"  Masked tiles: {output_masked_dir}")
    print(f"  Masks: {output_mask_dir}")
    print(f"  Metadata: {metadata_out_path}")
    print(f"\nAverage tiles per sinogram: {total_tiles / n_sinograms:.1f}")

    # Estimate disk space
    sample_tile = Image.open(output_gt_dir / f"sino_0000_tile_00.png")
    tile_size_kb = len(sample_tile.tobytes()) / 1024
    estimated_gb = (total_tiles * 3 * tile_size_kb) / (1024 * 1024)
    print(f"Estimated disk space: {estimated_gb:.2f} GB")

    print(f"\nNext steps:")
    print(f"  1. Run the inpainting model on the tiles")
    print(f"  2. Merge results using the appropriate merge script")


if __name__ == '__main__':
    main()
