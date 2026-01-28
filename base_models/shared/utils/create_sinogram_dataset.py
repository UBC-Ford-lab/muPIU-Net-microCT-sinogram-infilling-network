#!/usr/bin/env python3
#region Imports
"""
Sinogram Dataset Creator Using Memory Maps
Approach:
1. Create temporary memmap file [n_projections, height, width]
2. Load each projection ONCE into the memmap
3. Extract sinograms by indexing: memmap[:, height_idx, :]
4. Clean up memmap
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image

# Try to import tqdm, fallback to simple iterator if not available
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found, progress bars disabled")
    def tqdm(iterable, desc=None, total=None):
        return iterable

# Import from ct_core package
from ct_core import vff_io
print("Using ct_core.vff_io.read_vff()")
#endregion

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Create sinogram dataset using memory-mapped projection stack"
    )
    p.add_argument(
        '--scan_folder',
        type=str,
        required=True,
        help='Path to scan folder containing projection .vff files (MUST match original U-Net ou source!)'
    )
    p.add_argument(
        '--output_dir',
        type=str,
        default='sinogram_dataset',
        help='Output directory for sinograms and masks'
    )
    p.add_argument(
        '--mask_type',
        type=str,
        choices=['lama', 'repaint'],
        default='lama',
        help='Mask convention: lama (255=inpaint) or repaint (0=inpaint)'
    )
    p.add_argument(
        '--missing_projection_indices',
        type=int,
        nargs='+',
        default=None,
        help='Indices of projections to mask out. If None, uses every 4th projection.'
    )
    p.add_argument(
        '--height_range',
        type=int,
        nargs=2,
        default=None,
        help='Height range to process [start, end]. If None, processes all heights.'
    )
    p.add_argument(
        '--pad_to_modulo',
        type=int,
        default=8,
        help='Pad images to be divisible by this value (default: 8 for LaMa)'
    )
    p.add_argument(
        '--normalize_globally',
        default=True,
        action='store_true',
        help='Use global normalization across all sinograms instead of per-sinogram'
    )
    p.add_argument(
        '--memmap_dir',
        type=str,
        default='/tmp',
        help='Directory for temporary memmap files (default: /tmp)'
    )
    p.add_argument(
        '--reuse_memmap',
        default=True,
        action='store_true',
        help='Reuse existing memmap file if found (saves time on repeated runs)'
    )
    p.add_argument(
        '--keep_memmap',
        default=True,
        action='store_true',
        help='Keep memmap file after completion (for reuse in future runs)'
    )
    p.add_argument(
        '--mask_value',
        type=str,
        choices=['zero', 'mean', 'noise'],
        default='zero',
        help='Value to use for masked-out regions: zero, mean, or noise (default: zero)'
    )

    return p.parse_args()

def natural_sort_key(text):
    """Natural sorting key for filenames with numbers."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def load_projections_to_memmap(vff_files: List[Path], memmap_path: str, reuse_if_exists: bool = False) -> Tuple[np.memmap, int, int, int]:
    """
    Load all projections into a memory-mapped file.

    Uses the proven vff_io.read_vff() method.

    Args:
        vff_files: List of VFF file paths
        memmap_path: Path for temporary memmap file
        reuse_if_exists: If True, reuse existing memmap file if found

    Returns:
        memmap: Memory-mapped array [n_projections, height, width]
        n_projections, height, width: Dimensions
    """
    # Get dimensions from first projection
    _, first_proj = vff_io.read_vff(str(vff_files[0]), verbose=False)
    first_proj = first_proj.squeeze(0).byteswap().view(first_proj.dtype.newbyteorder())

    height, width = first_proj.shape
    n_projections = len(vff_files)

    # Check if we can reuse existing memmap
    if reuse_if_exists and os.path.exists(memmap_path):
        print("Checking for existing memmap file...")
        try:
            # Try to load existing memmap
            existing_memmap = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode='r+',
                shape=(n_projections, height, width)
            )
            print(f"  ✓ Reusing existing memmap: {memmap_path}")
            print(f"  ✓ Skipped loading {n_projections} projections (saved ~9 minutes!)")
            return existing_memmap, n_projections, height, width
        except Exception as e:
            print(f"  ✗ Could not reuse existing memmap ({e}), creating new one...")

    print("Loading projections into memory-mapped file...")

    print(f"  Projection dimensions: {height} × {width}")
    print(f"  Number of projections: {n_projections}")
    print(f"  Memmap file: {memmap_path}")

    # Calculate memmap size
    memmap_size_gb = (n_projections * height * width * 4) / (1024**3)
    print(f"  Memmap size: {memmap_size_gb:.2f} GB (on disk)")

    # Create memory-mapped array
    projection_memmap = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode='w+',
        shape=(n_projections, height, width)
    )

    # Load first projection
    projection_memmap[0] = first_proj.astype(np.float32)

    # Load remaining projections with batching (like projection_infilling_domain_comparison.py)
    print("  Reading projection files...")
    batch_size = min(32, n_projections)

    for batch_start in tqdm(range(0, n_projections, batch_size), desc="  Loading batches"):
        batch_end = min(batch_start + batch_size, n_projections)

        # Skip first batch if we already loaded index 0
        start_idx = batch_start if batch_start > 0 else 1

        for idx in range(start_idx, batch_end):
            _, proj = vff_io.read_vff(str(vff_files[idx]), verbose=False)
            proj = proj.squeeze(0).byteswap().view(proj.dtype.newbyteorder())
            projection_memmap[idx] = proj.astype(np.float32)

        # Flush periodically
        projection_memmap.flush()

    print(f"  ✓ All {n_projections} projections loaded to memmap")

    return projection_memmap, n_projections, height, width

def create_missing_projection_mask(n_projections: int, missing_indices: List[int] = None) -> np.ndarray:
    """Create 1D mask indicating which projections are missing."""
    mask = np.zeros(n_projections, dtype=np.uint8)

    if missing_indices is None:
        # Default: every second projection (odd indices: 1, 3, 5, ...)
        # Corresponds to acq-00-0001.vff, acq-00-0003.vff, etc.
        missing_indices = list(range(1, n_projections, 2))
        print(f"Using default masking: every 2nd projection, odd indices ({len(missing_indices)} missing)")
    else:
        print(f"Using custom masking: {len(missing_indices)} projections missing")

    # Validate indices
    invalid_indices = [idx for idx in missing_indices if idx < 0 or idx >= n_projections]
    if invalid_indices:
        raise ValueError(f"Invalid projection indices: {invalid_indices}")

    mask[missing_indices] = 1
    print(f"Masking {mask.sum()}/{n_projections} projections ({100*mask.sum()/n_projections:.1f}%)")

    return mask

def compute_global_statistics_from_memmap(projection_memmap: np.memmap, sample_size: int = None) -> Tuple[float, float]:
    """
    Compute EXACT global min/max from ALL projections (no sampling).

    CRITICAL FOR RECONSTRUCTION SCALE MATCHING:
    This must compute the EXACT min/max from ALL projections to ensure the
    Base_model_comparison reconstruction has the SAME intensity scale as the
    original Scanner reconstruction. Any sampling could miss the true global
    min/max and cause reconstruction scale mismatch.

    The previous implementation sampled only 50 projections, which caused
    Base_model_comparison reconstructions to have 1.67× larger intensity range
    (65,394 vs 39,054), resulting in smaller ERF dropoff in MTF measurements.

    Args:
        projection_memmap: [n_projections, height, width] memmap
        sample_size: Ignored (kept for backward compatibility)

    Returns:
        global_min, global_max: EXACT min/max from ALL projections
    """
    n_projections = projection_memmap.shape[0]
    print(f"\nComputing EXACT global normalization from ALL {n_projections} projections...")
    print(f"  (Critical: ensures reconstruction intensity scale matches Scanner reconstruction)")

    global_min = float('inf')
    global_max = float('-inf')

    # Process ALL projections (no sampling) to get exact min/max
    for idx in tqdm(range(n_projections), desc="  Processing all projections"):
        # Access memmap one projection at a time to minimize RAM
        proj_min = float(projection_memmap[idx].min())
        proj_max = float(projection_memmap[idx].max())
        global_min = min(global_min, proj_min)
        global_max = max(global_max, proj_max)

    print(f"  EXACT global range: [{global_min:.10f}, {global_max:.10f}]")
    print(f"  Range: {global_max - global_min:.10f}")
    return global_min, global_max

def normalize_sinogram(sinogram: np.ndarray, global_min: float = None, global_max: float = None) -> Tuple[np.ndarray, Dict]:
    """Normalize sinogram to [0, 65535] uint16 for better precision (260x better than uint8)."""
    data_min = global_min if global_min is not None else float(sinogram.min())
    data_max = global_max if global_max is not None else float(sinogram.max())

    if data_max - data_min < 1e-8:
        normalized = np.full(sinogram.shape, 32767, dtype=np.uint16)  # Middle value for uint16
        return normalized, {'min': data_min, 'max': data_max, 'range': 0}

    # Normalize to [0, 65535] instead of [0, 255]
    normalized = ((sinogram - data_min) / (data_max - data_min) * 65535.0).clip(0, 65535)
    normalized = normalized.astype(np.uint16)

    return normalized, {'min': data_min, 'max': data_max, 'range': data_max - data_min}

def pad_to_modulo(image: np.ndarray, modulo: int = 8, mode: str = 'reflect') -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad image to be divisible by modulo."""
    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        h, w, _ = image.shape
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.shape}")

    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if image.ndim == 2:
        padded = np.pad(image, ((top, bottom), (left, right)), mode=mode)
    else:
        padded = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode=mode)

    return padded, (top, bottom, left, right)

def convert_to_rgb(grayscale: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB by repeating channels. Works with both uint8 and uint16."""
    return np.stack([grayscale, grayscale, grayscale], axis=-1)

def apply_mask_to_sinogram(sinogram: np.ndarray, projection_mask: np.ndarray, mask_value: str = 'zero') -> np.ndarray:
    """
    Apply mask to sinogram by setting masked regions to specified value.

    This ensures LaMa doesn't see the ground truth in "missing" regions!

    Args:
        sinogram: [n_projections, width] sinogram
        projection_mask: [n_projections] mask (1=missing, 0=keep)
        mask_value: 'zero', 'mean', or 'noise'

    Returns:
        masked_sinogram: Sinogram with missing projections masked out
    """
    masked_sinogram = sinogram.copy()
    missing_indices = np.where(projection_mask == 1)[0]

    if mask_value == 'zero':
        # Set missing projections to zero
        masked_sinogram[missing_indices, :] = 0

    elif mask_value == 'mean':
        # Set missing projections to mean of known projections
        known_indices = np.where(projection_mask == 0)[0]
        mean_value = sinogram[known_indices, :].mean()
        masked_sinogram[missing_indices, :] = mean_value

    elif mask_value == 'noise':
        # Set missing projections to random noise matching known data statistics
        known_indices = np.where(projection_mask == 0)[0]
        mean_val = sinogram[known_indices, :].mean()
        std_val = sinogram[known_indices, :].std()
        noise_shape = (len(missing_indices), sinogram.shape[1])
        masked_sinogram[missing_indices, :] = np.random.normal(mean_val, std_val, noise_shape)

    return masked_sinogram

def save_image_png(image: np.ndarray, filepath: str):
    """Save numpy array as PNG. Supports uint8 and uint16."""
    if image.ndim == 2:
        # Grayscale image
        if image.dtype == np.uint16:
            img = Image.fromarray(image, mode='I;16')  # 16-bit grayscale
        elif image.dtype == np.uint8:
            img = Image.fromarray(image, mode='L')     # 8-bit grayscale
        else:
            raise ValueError(f"Unsupported dtype for grayscale: {image.dtype}")
    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB image
        if image.dtype == np.uint16:
            img = Image.fromarray(image, mode='RGB')   # PIL automatically handles uint16 RGB
        elif image.dtype == np.uint8:
            img = Image.fromarray(image, mode='RGB')   # 8-bit RGB
        else:
            raise ValueError(f"Unsupported dtype for RGB: {image.dtype}")
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # PNG compression level 6 is a good balance (lossless compression)
    img.save(filepath, format='PNG', compress_level=6)

def create_sinogram_dataset(args):
    """Main function using memory-mapped projection stack."""

    # Setup output directories
    output_path = Path(args.output_dir)
    sinogram_lama_dir = output_path / 'sinograms_lama'  # Masked sinograms for LaMa
    sinogram_gt_dir = output_path / 'sinograms_gt'      # Ground truth sinograms
    mask_dir = output_path / 'masks'

    sinogram_lama_dir.mkdir(parents=True, exist_ok=True)
    sinogram_gt_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SINOGRAM DATASET CREATOR (MEMMAP VERSION)")
    print("=" * 70)
    print(f"Scan folder: {args.scan_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mask type: {args.mask_type}")
    print(f"Memmap directory: {args.memmap_dir}")
    print("=" * 70)

    # Get projection files
    scan_path = Path(args.scan_folder)
    vff_files = sorted(scan_path.glob('*.vff'), key=natural_sort_key)

    if len(vff_files) == 0:
        raise ValueError(f"No .vff files found in {args.scan_folder}")

    print(f"\nFound {len(vff_files)} projection files")

    # Create temporary memmap file
    memmap_path = os.path.join(args.memmap_dir, f'ct_projections_{os.getpid()}.dat')

    try:
        # Load all projections into memmap (ONE-TIME LOAD)
        projection_memmap, n_projections, height, width = load_projections_to_memmap(
            vff_files,
            memmap_path,
            reuse_if_exists=args.reuse_memmap
        )

        # Determine height range
        if args.height_range is not None:
            height_start, height_end = args.height_range
            height_start = max(0, height_start)
            height_end = min(height, height_end)
        else:
            height_start, height_end = 0, height

        n_sinograms = height_end - height_start
        print(f"\nProcessing height range: {height_start} to {height_end} ({n_sinograms} sinograms)")

        # Create projection mask
        projection_mask_1d = create_missing_projection_mask(
            n_projections,
            args.missing_projection_indices
        )

        # Global normalization if requested
        global_min, global_max = None, None
        if args.normalize_globally:
            global_min, global_max = compute_global_statistics_from_memmap(projection_memmap)

        # Metadata
        metadata = {
            'scan_folder': args.scan_folder,
            'n_projections': int(n_projections),
            'projection_height': int(height),
            'projection_width': int(width),
            'n_sinograms': int(n_sinograms),
            'height_range': [int(height_start), int(height_end)],
            'mask_type': args.mask_type,
            'mask_value': args.mask_value,
            'pad_to_modulo': args.pad_to_modulo,
            'missing_projection_indices': projection_mask_1d.nonzero()[0].tolist(),
            'n_missing_projections': int(projection_mask_1d.sum()),
            'undersampling_ratio': float(projection_mask_1d.sum() / n_projections),
            'normalize_globally': args.normalize_globally,
            'global_min': global_min,
            'global_max': global_max,
            'projection_filenames': [f.name for f in vff_files],
            'sinograms': [],
            'output_folders': {
                'sinograms_lama': 'Masked sinograms for LaMa input (missing regions set to mask_value)',
                'sinograms_gt': 'Ground truth sinograms (complete, for comparison)',
                'masks': 'Binary masks indicating missing regions'
            },
            'note': 'Missing projections are masked out in sinograms_lama (not visible to inpainting model)'
        }

        # Process each height slice by extracting from memmap
        print(f"\nGenerating {n_sinograms} sinograms (memmap extraction)...")

        # Process in batches to control memory usage
        height_batch_size = 16

        for height_batch_start in tqdm(range(height_start, height_end, height_batch_size),
                                       desc="Creating sinograms"):
            height_batch_end = min(height_batch_start + height_batch_size, height_end)

            for height_idx in range(height_batch_start, height_batch_end):
                # Extract sinogram: memmap[:, height_idx, :] -> [n_projections, width]
                # Copy to avoid holding reference to memmap
                sinogram_gt = projection_memmap[:, height_idx, :].copy()

                # CRITICAL: Mask out the "missing" projections so LaMa doesn't see ground truth!
                sinogram_masked = apply_mask_to_sinogram(
                    sinogram_gt,
                    projection_mask_1d,
                    mask_value=args.mask_value
                )

                # Normalize both GT and masked versions
                sinogram_gt_norm, norm_params_gt = normalize_sinogram(sinogram_gt, global_min, global_max)
                sinogram_masked_norm, norm_params_masked = normalize_sinogram(sinogram_masked, global_min, global_max)

                # Pad both versions
                sinogram_gt_padded, padding = pad_to_modulo(sinogram_gt_norm, args.pad_to_modulo, mode='reflect')
                sinogram_masked_padded, _ = pad_to_modulo(sinogram_masked_norm, args.pad_to_modulo, mode='reflect')

                # Keep as 16-bit grayscale (no RGB conversion)
                # LaMa can work with grayscale images directly, and 16-bit preserves precision
                sinogram_gt_gray = sinogram_gt_padded
                sinogram_masked_gray = sinogram_masked_padded

                # Create 2D mask
                mask_2d = np.repeat(projection_mask_1d[:, np.newaxis], width, axis=1)

                # Apply mask convention
                if args.mask_type == 'lama':
                    mask_2d_formatted = mask_2d * 255
                else:  # repaint
                    mask_2d_formatted = (1 - mask_2d) * 255

                mask_2d_formatted = mask_2d_formatted.astype(np.uint8)
                mask_padded, _ = pad_to_modulo(mask_2d_formatted, args.pad_to_modulo, mode='constant')

                # Generate filenames
                sino_idx = height_idx - height_start
                sino_filename = f"sino_{sino_idx:04d}.png"
                mask_filename = f"sino_{sino_idx:04d}_mask001.png"

                # Save files to THREE folders (all as 16-bit grayscale)
                save_image_png(sinogram_masked_gray, str(sinogram_lama_dir / sino_filename))  # For LaMa input (16-bit)
                save_image_png(sinogram_gt_gray, str(sinogram_gt_dir / sino_filename))        # Ground truth (16-bit)
                save_image_png(mask_padded, str(mask_dir / mask_filename))                    # Mask (8-bit)

                # Store metadata
                sino_metadata = {
                    'filename': sino_filename,
                    'mask_filename': mask_filename,
                    'height_index': int(height_idx),
                    'original_shape': [int(n_projections), int(width)],
                    'padded_shape': list(sinogram_gt_padded.shape),
                    'padding': [int(p) for p in padding],
                    'normalization_gt': {
                        'min': float(norm_params_gt['min']),
                        'max': float(norm_params_gt['max']),
                        'range': float(norm_params_gt['range'])
                    },
                    'normalization_masked': {
                        'min': float(norm_params_masked['min']),
                        'max': float(norm_params_masked['max']),
                        'range': float(norm_params_masked['range'])
                    }
                }
                metadata['sinograms'].append(sino_metadata)

        # Save metadata
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print("DATASET CREATION COMPLETE")
        print("=" * 70)
        print(f"Sinograms (LaMa input): {sinogram_lama_dir}")
        print(f"Sinograms (Ground Truth): {sinogram_gt_dir}")
        print(f"Masks: {mask_dir}")
        print(f"Metadata: {metadata_path}")
        print(f"\nTotal sinograms: {len(metadata['sinograms'])}")
        print(f"Undersampling: {metadata['undersampling_ratio']*100:.1f}%")
        print(f"Missing projections: {metadata['missing_projection_indices'][:5]}... (every 2nd)")
        print("=" * 70)

    finally:
        # Clean up memmap file (unless --keep_memmap specified)
        if args.keep_memmap:
            print(f"\nKeeping memmap file for reuse: {memmap_path}")
            print(f"  Reuse with: --reuse_memmap")
            print(f"  Size: {os.path.getsize(memmap_path) / (1024**3):.2f} GB")
        else:
            print("\nCleaning up temporary memmap file...")
            try:
                del projection_memmap
                if os.path.exists(memmap_path):
                    os.unlink(memmap_path)
                    print(f"  ✓ Removed: {memmap_path}")
            except Exception as e:
                print(f"  Warning: Could not remove memmap file: {e}")
                print(f"  You can manually remove: rm {memmap_path}")

def main():
    args = parse_args()
    create_sinogram_dataset(args)

if __name__ == '__main__':
    main()
