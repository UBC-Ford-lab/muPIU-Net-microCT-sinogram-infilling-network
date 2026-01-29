#!/usr/bin/env python3
"""
Merge DeepFill v2 Infilled Tiles Back to Full Resolution
=========================================================
This script takes 256×256 infilled tiles from DeepFill v2 and merges them back
into full-resolution (410×3500) sinograms with proper overlap blending.

Key features:
- Nearest/linear/Gaussian blending in overlap regions for seamless merging
- Preserves NORMALIZED [0, 65535] uint16 range (no denormalization here!)
- Denormalization is handled by the reconstruction script using stored metadata
- Removes padding to restore original sinogram dimensions

IMPORTANT: This script outputs NORMALIZED sinograms in [0, 65535] uint16 range.
The reconstruction script (reconstruct_from_deepfill.py) will denormalize using
the per-sinogram min/max values stored in metadata.

Based on merge_repaint_tiles.py structure.

Author: Claude (Anthropic)
Date: 2025-11-29 (updated 2025-12-04: fixed double denormalization bug)
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge DeepFill v2 tiles back to full-resolution sinograms'
    )
    parser.add_argument(
        '--tiles_dir',
        type=str,
        default='../data/tiles_infilled',
        help='Input directory containing infilled tiles from DeepFill v2'
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        default='../../../shared/sinogram_tiles/tiling_metadata.json',
        help='Path to tiling metadata JSON (shared by all models)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/sinograms_infilled',
        help='Output directory for merged sinograms'
    )
    parser.add_argument(
        '--blend_mode',
        type=str,
        choices=['linear', 'gaussian', 'nearest'],
        default='nearest',
        help='Blending method for overlaps: linear, gaussian, or nearest (default: nearest)'
    )
    parser.add_argument(
        '--gaussian_sigma',
        type=float,
        default=8.0,
        help='Sigma for Gaussian blending (default: 8.0)'
    )
    return parser.parse_args()


class TileMerger:
    """Merges overlapping tiles back into full-resolution images."""

    def __init__(self, blend_mode: str = 'gaussian', gaussian_sigma: float = 8.0):
        """
        Initialize tile merger.

        Args:
            blend_mode: 'linear', 'gaussian', or 'nearest' blending in overlap regions
                       'nearest' = no blending, uses tile with pixel furthest from edge
            gaussian_sigma: Sigma parameter for Gaussian blending
        """
        self.blend_mode = blend_mode
        self.gaussian_sigma = gaussian_sigma

    def create_blend_weights(self, tile_size: int, overlap: int) -> np.ndarray:
        """
        Create blending weights for a tile.

        Args:
            tile_size: Size of square tile
            overlap: Overlap size in pixels

        Returns:
            weights: 2D array of blending weights [0, 1]
        """
        if self.blend_mode == 'nearest':
            # No blending - weight based on distance from edges
            # Pixels furthest from tile edges get priority
            # This creates hard cutoffs in overlap regions with NO averaging
            y_dist = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])
            x_dist = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])

            # Distance from nearest edge (Manhattan distance)
            dist_from_edge = np.minimum(y_dist[:, np.newaxis], x_dist[np.newaxis, :])

            # Return distance values (will be used for priority, not averaging)
            # Higher value = further from edge = higher priority
            weights = dist_from_edge.astype(np.float32)

        elif self.blend_mode == 'linear':
            # Linear ramp in overlap regions
            weights = np.ones((tile_size, tile_size), dtype=np.float32)

            # Create linear ramps for edges
            ramp = np.linspace(0, 1, overlap)

            # Apply ramps to edges
            # Top edge
            weights[:overlap, :] = weights[:overlap, :] * ramp[:, np.newaxis]
            # Bottom edge
            weights[-overlap:, :] = weights[-overlap:, :] * ramp[::-1, np.newaxis]
            # Left edge
            weights[:, :overlap] = weights[:, :overlap] * ramp[np.newaxis, :]
            # Right edge
            weights[:, -overlap:] = weights[:, -overlap:] * ramp[::-1][np.newaxis, :]

        elif self.blend_mode == 'gaussian':
            # Gaussian-weighted blend (smoother)
            # Create distance from edges
            y_dist = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])
            x_dist = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])

            y_weights = np.exp(-(overlap - np.minimum(y_dist, overlap))**2 / (2 * self.gaussian_sigma**2))
            x_weights = np.exp(-(overlap - np.minimum(x_dist, overlap))**2 / (2 * self.gaussian_sigma**2))

            weights = np.minimum(y_weights[:, np.newaxis], x_weights[np.newaxis, :])
            weights = np.clip(weights, 0, 1)

        return weights

    def merge_tiles(
        self,
        tiles: List[np.ndarray],
        tile_info: Dict,
        output_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Merge tiles back into full-resolution image with blending.

        Args:
            tiles: List of tile arrays
            tile_info: Dictionary with tiling metadata
            output_shape: (height, width) of output image

        Returns:
            merged: Merged image array
        """
        height, width = output_shape
        tile_size = tile_info['tile_size']
        overlap = tile_info['overlap']

        # Initialize accumulation arrays
        merged = np.zeros((height, width), dtype=np.float64)

        if self.blend_mode == 'nearest':
            # For nearest mode: track which tile has priority for each pixel
            # Priority is based on distance from tile edge
            priority_map = np.full((height, width), -np.inf, dtype=np.float64)

            # Create distance-based priority weights for a tile
            blend_weights = self.create_blend_weights(tile_size, overlap)

            # Merge each tile - keep pixel from tile with highest priority (furthest from edge)
            for tile, tile_meta in zip(tiles, tile_info['tiles']):
                y_start = tile_meta['y_start']
                y_end = tile_meta['y_end']
                x_start = tile_meta['x_start']
                x_end = tile_meta['x_end']

                tile_h = y_end - y_start
                tile_w = x_end - x_start

                # Get priority weights for this tile
                current_weights = blend_weights[:tile_h, :tile_w]

                # Convert tile to float
                tile_float = tile.astype(np.float64)

                # Update pixels where this tile has higher priority
                # This creates hard boundaries (no averaging) based on distance from edge
                higher_priority = current_weights > priority_map[y_start:y_end, x_start:x_end]
                merged[y_start:y_end, x_start:x_end][higher_priority] = tile_float[higher_priority]
                priority_map[y_start:y_end, x_start:x_end] = np.maximum(
                    priority_map[y_start:y_end, x_start:x_end],
                    current_weights
                )

        else:
            # Original weighted blending for 'linear' and 'gaussian' modes
            weights_sum = np.zeros((height, width), dtype=np.float64)

            # Create blend weights for a tile
            blend_weights = self.create_blend_weights(tile_size, overlap)

            # Merge each tile
            for tile, tile_meta in zip(tiles, tile_info['tiles']):
                y_start = tile_meta['y_start']
                y_end = tile_meta['y_end']
                x_start = tile_meta['x_start']
                x_end = tile_meta['x_end']

                tile_h = y_end - y_start
                tile_w = x_end - x_start

                # Handle edge tiles that might be smaller
                current_weights = blend_weights[:tile_h, :tile_w]

                # Convert tile to float
                tile_float = tile.astype(np.float64)

                # Accumulate weighted tile
                merged[y_start:y_end, x_start:x_end] += tile_float * current_weights
                weights_sum[y_start:y_end, x_start:x_end] += current_weights

            # Normalize by total weights
            # Avoid division by zero (shouldn't happen, but be safe)
            weights_sum = np.maximum(weights_sum, 1e-8)
            merged = merged / weights_sum

        return merged


def normalize_to_uint16(
    image: np.ndarray,
    source_max: float = None
) -> np.ndarray:
    """
    Convert merged tile data to uint16 [0, 65535] range for saving.

    This function handles tiles that may be uint8 [0, 255] or uint16 [0, 65535].
    It ensures the output is always uint16 in [0, 65535] range, preserving
    the NORMALIZED values so that denormalization happens only once in the
    reconstruction script.

    Args:
        image: Input image (float64 from merge, with values in [0, 255] or [0, 65535])
        source_max: Maximum value of source tiles (255 for uint8, 65535 for uint16).
                   If None, auto-detected from image.max().

    Returns:
        uint16 image in [0, 65535] range (still normalized, NOT denormalized)
    """
    # Auto-detect source bit depth if not provided
    if source_max is None:
        # If max value is close to 255, assume uint8 source tiles
        # If max value is larger, assume uint16 source tiles
        if image.max() <= 256:
            source_max = 255.0
        else:
            source_max = 65535.0

    # Scale to [0, 65535] range
    if source_max <= 256:
        # uint8 source: scale up from [0, 255] to [0, 65535]
        image_scaled = image * (65535.0 / 255.0)
    else:
        # uint16 source: already in [0, 65535] range
        image_scaled = image

    # Clip and convert to uint16
    image_out = np.clip(image_scaled, 0, 65535).astype(np.uint16)

    return image_out


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    tiles_dir = Path(args.tiles_dir)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load tiling metadata
    print(f"Loading tiling metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    n_sinograms = metadata.get('n_sinograms', len(metadata['sinograms']))
    total_tiles = metadata.get('total_tiles', n_sinograms * 32)

    print(f"\nDataset info:")
    print(f"  Sinograms: {n_sinograms}")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Tile size: {metadata['tile_size']}×{metadata['tile_size']}")
    print(f"  Overlap: {metadata['overlap']} pixels")
    print(f"  Blend mode: {args.blend_mode}")
    if args.blend_mode == 'nearest':
        print(f"  ⚠ NEAREST mode: NO blending/averaging - preserves true model noise")
        print(f"     Overlap regions use tile with pixel furthest from edge")
    elif args.blend_mode == 'linear':
        print(f"  ⚠ LINEAR mode: Minimal blending - small noise reduction (~8%)")
    elif args.blend_mode == 'gaussian':
        print(f"  ⚠ GAUSSIAN mode: Heavy smoothing - significant noise reduction (~90%)")
        print(f"     Sigma: {args.gaussian_sigma}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize merger
    merger = TileMerger(blend_mode=args.blend_mode, gaussian_sigma=args.gaussian_sigma)

    # Process each sinogram
    print(f"\nMerging tiles back to full resolution...")

    # Track source bit depth (detected from first tile)
    source_bit_depth = None

    for sino_idx in tqdm(range(n_sinograms), desc="Merging sinograms"):
        sino_meta = metadata['sinograms'][str(sino_idx)]
        tile_info = sino_meta['tile_info']
        n_tiles = tile_info['n_tiles_total']

        # Load all tiles for this sinogram
        tiles = []
        missing_tiles = []

        for tile_idx in range(n_tiles):
            tile_name = f"sino_{sino_idx:04d}_tile_{tile_idx:02d}.png"
            tile_path = tiles_dir / tile_name

            if not tile_path.exists():
                missing_tiles.append(tile_name)
                continue

            # Load tile
            tile_img = Image.open(tile_path)
            tile_array = np.array(tile_img)

            # Handle RGB tiles (convert to grayscale if needed)
            if tile_array.ndim == 3:
                # Take first channel (should all be equal for grayscale-converted-to-RGB)
                tile_array = tile_array[:, :, 0]

            # Detect source bit depth from first tile
            if source_bit_depth is None:
                if tile_array.dtype == np.uint16:
                    source_bit_depth = 16
                    source_max = 65535.0
                else:
                    source_bit_depth = 8
                    source_max = 255.0
                print(f"\n  Detected source bit depth: {source_bit_depth}-bit (max={source_max})")

            tiles.append(tile_array)

        if len(missing_tiles) > 0:
            print(f"\nWarning: Missing {len(missing_tiles)} tiles for sinogram {sino_idx}")
            if len(missing_tiles) <= 5:
                for name in missing_tiles:
                    print(f"  - {name}")
            continue

        if len(tiles) != n_tiles:
            print(f"\nWarning: Expected {n_tiles} tiles, got {len(tiles)} for sinogram {sino_idx}")
            continue

        # Merge tiles with blending
        padded_shape = sino_meta['padded_shape']  # [416, 3504]
        merged = merger.merge_tiles(tiles, tile_info, tuple(padded_shape))

        # Convert to uint16 [0, 65535] range for saving
        # NOTE: We do NOT denormalize here! The reconstruction script will
        # denormalize using the per-sinogram min/max stored in metadata.
        # This avoids the double denormalization bug.
        merged_uint16 = normalize_to_uint16(merged, source_max=source_max)

        # Remove padding to restore original shape
        # Padding format: [top, bottom, left, right]
        padding = sino_meta['padding']
        top, bottom, left, right = padding

        # Crop to original shape (e.g., 410×3500 from padded 416×3504)
        h_start = top
        h_end = merged_uint16.shape[0] - bottom
        w_start = left
        w_end = merged_uint16.shape[1] - right

        merged_unpadded = merged_uint16[h_start:h_end, w_start:w_end]

        # Save merged sinogram (original shape, no padding)
        output_name = f"sino_{sino_idx:04d}.png"
        output_path = output_dir / output_name

        # Save as 16-bit PNG (normalized [0, 65535] values)
        Image.fromarray(merged_unpadded).save(output_path)

    print(f"\n{'='*70}")
    print(f"TILE MERGING COMPLETE!")
    print(f"{'='*70}")
    print(f"Merged sinograms saved to: {output_dir}")
    print(f"Total sinograms: {n_sinograms}")

    # Verify output with original shape info
    output_files = sorted(output_dir.glob('sino_*.png'))
    if len(output_files) > 0:
        sample_path = output_files[0]
        sample_img = Image.open(sample_path)
        sample_array = np.array(sample_img)
        sample_meta = metadata['sinograms']['0']
        original_shape = sample_meta['original_shape']
        padded_shape = sample_meta['padded_shape']
        normalization = sample_meta['normalization_gt']

        print(f"\nOutput verification:")
        print(f"  Final shape (padding removed): {sample_array.shape}")
        print(f"  Expected original shape: {tuple(original_shape)}")
        print(f"  Intermediate padded shape: {tuple(padded_shape)}")
        print(f"  Dtype: {sample_array.dtype}")
        print(f"  Value range: [{sample_array.min()}, {sample_array.max()}]")
        print(f"  Expected range: [0, 65535] (NORMALIZED)")
        print(f"  Output files: {len(output_files)}")
        print(f"\n  NOTE: Sinograms are saved in NORMALIZED [0, 65535] uint16 range.")
        print(f"        Denormalization to original range [{normalization['min']:.1f}, {normalization['max']:.1f}]")
        print(f"        will be performed by the reconstruction script.")

    print(f"\nNext steps:")
    print(f"  python reconstruct.py")


if __name__ == '__main__':
    main()
