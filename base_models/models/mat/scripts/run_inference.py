#!/usr/bin/env python3
"""
MAT (Mask-Aware Transformer) Inference on Sinogram Tiles
=========================================================
This script runs MAT inpainting on 256x256 sinogram tiles.

Key features:
- Handles int16 input tiles (preserves precision through uint16)
- Uses CelebA-HQ 256x256 pretrained model
- Batch processing for efficient GPU utilization
- Progress tracking with estimated time remaining
- Automatic GPU/CPU detection
- Supports resuming from interrupted runs

Input: 16-bit grayscale MASKED tiles (H, W) uint16 + binary masks
       Masked tiles have the regions to inpaint already zeroed/masked
       Masks use RePaint format (0=inpaint, 255=keep) - same as MAT!
Output: 16-bit grayscale tiles (H, W) uint16 (infilled)

MAT Mask Convention:
- 0 = pixels to inpaint (masked region)
- 1 = pixels to keep (unmasked region)
This matches RePaint format, so NO mask inversion needed!

Author: Claude (Anthropic)
Date: 2025-12-01
"""

import argparse
import json
import os
import sys
import time
import pickle
import copy
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Add MAT repository to path
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent  # models/mat/
MAT_DIR = MODEL_DIR / 'MAT'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MAT inference on sinogram tiles'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Directory containing input tiles (16-bit grayscale PNG). Preferred: sinograms_masked'
    )
    # Backward compatibility alias
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=None,
        help='DEPRECATED: Alias for --input_dir (kept for backward compatibility)'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='../../../shared/sinogram_tiles/masks',
        help='Directory containing mask tiles (8-bit, RePaint format: 0=inpaint)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/tiles_infilled',
        help='Output directory for infilled tiles'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../MAT/pretrained/CelebA-HQ_256.pkl',
        help='Path to MAT checkpoint (.pkl file)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8, reduce if OOM)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (auto, cuda, cpu)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output (skip completed tiles)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Start from this tile index (for parallel processing)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=-1,
        help='End at this tile index, -1 for all (for parallel processing)'
    )
    parser.add_argument(
        '--save_grayscale',
        action='store_true',
        default=True,
        help='Save output as grayscale uint16 (for tile merging)'
    )
    parser.add_argument(
        '--truncation_psi',
        type=float,
        default=1.0,
        help='Truncation psi for style mixing (default: 1.0, no truncation)'
    )
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='const',
        choices=['const', 'random', 'none'],
        help='Noise mode for synthesis (default: const)'
    )
    return parser.parse_args()


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    """Copy parameters and buffers from src to dst module."""
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in src_module.named_parameters()}
    src_tensors.update({name: tensor for name, tensor in src_module.named_buffers()})
    for name, tensor in dst_module.named_parameters():
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
    for name, tensor in dst_module.named_buffers():
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach())


def load_mat_generator(checkpoint_path: str, device: torch.device):
    """
    Load MAT generator model from pickle checkpoint.

    Args:
        checkpoint_path: Path to .pkl checkpoint file
        device: Torch device

    Returns:
        Generator model in eval mode
    """
    print(f"Loading MAT checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download the CelebA-HQ 256x256 weights first."
        )

    # Add MAT networks directory to path
    sys.path.insert(0, str(MAT_DIR))

    # Load the pickle file
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    # The pickle contains 'G' (generator), 'D' (discriminator), 'G_ema' (EMA generator)
    # We use G_ema for inference as it typically produces better results
    if 'G_ema' in data:
        G_src = data['G_ema']
        print("  Using G_ema (EMA generator)")
    elif 'G' in data:
        G_src = data['G']
        print("  Using G (generator)")
    else:
        raise KeyError(f"Checkpoint does not contain 'G' or 'G_ema'. Keys: {data.keys()}")

    # Get model configuration
    c_dim = getattr(G_src, 'c_dim', 0)
    print(f"  Model config:")
    print(f"    z_dim: {G_src.z_dim}")
    print(f"    w_dim: {G_src.w_dim}")
    print(f"    c_dim: {c_dim} (0 = unconditional)")
    print(f"    img_resolution: {G_src.img_resolution}")
    print(f"    img_channels: {G_src.img_channels}")

    # Move to device
    G_src = G_src.to(device)
    G_src.eval()

    print(f"  Model loaded successfully")
    return G_src


def process_batch_mat(
    generator,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    truncation_psi: float = 1.0,
    noise_mode: str = 'const'
) -> torch.Tensor:
    """
    Process a batch of images through MAT.

    Args:
        generator: MAT generator model
        images: Batch of images (B, 3, H, W) in [-1, 1]
        masks: Batch of masks (B, 1, H, W) where 0=inpaint, 1=keep
        device: Torch device
        truncation_psi: Truncation parameter for style mixing
        noise_mode: Noise mode for synthesis

    Returns:
        Inpainted images (B, 3, H, W) in [-1, 1]
    """
    batch_size = images.shape[0]

    # Move to device
    images = images.to(device)
    masks = masks.to(device)

    # Generate random latent codes
    z = torch.randn(batch_size, generator.z_dim, device=device)

    # Create label tensor (required even if unconditional - c_dim=0 means empty tensor)
    # MAT requires c as a positional argument, so we always create a tensor
    c_dim = getattr(generator, 'c_dim', 0)
    label = torch.zeros([batch_size, max(c_dim, 0)], device=device)

    with torch.inference_mode():
        # MAT generator signature: forward(img, mask, z, c, truncation_psi=1, noise_mode='const')
        # All positional arguments are required, optional ones can be passed as kwargs

        # Log shapes once for debugging
        if not hasattr(process_batch_mat, '_logged'):
            print(f"\nGenerator call tensors:")
            print(f"  images: {images.shape}, device={images.device}")
            print(f"  masks: {masks.shape}, device={masks.device}")
            print(f"  z: {z.shape}, device={z.device}")
            print(f"  label: {label.shape}, device={label.device}")
            process_batch_mat._logged = True

        # Try different calling conventions based on MAT version
        try:
            # First try: full signature with truncation and noise mode
            output = generator(
                images, masks, z, label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode
            )
        except TypeError as e:
            if 'truncation_psi' in str(e) or 'noise_mode' in str(e):
                # Fallback: without optional parameters
                print(f"    Note: Generator doesn't accept truncation/noise params, using defaults")
                try:
                    output = generator(images, masks, z, label)
                except Exception as e2:
                    print(f"    Error with 4 args: {e2}")
                    raise
            else:
                # Different error - re-raise with more context
                print(f"    Error calling generator: {e}")
                print(f"    Tensor shapes: images={images.shape}, masks={masks.shape}, z={z.shape}, label={label.shape}")
                raise

    # Output is already in [-1, 1]
    return output


def normalize_uint16_to_float(img_array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize uint16 image to [-1, 1] float range for MAT.

    Args:
        img_array: Input image array (uint16 or uint8)

    Returns:
        normalized: Image normalized to [-1, 1] float32
        scale: Scale factor used
        offset: Offset used
    """
    if img_array.dtype == np.uint8:
        # uint8: [0, 255] -> [-1, 1]
        normalized = img_array.astype(np.float32) / 127.5 - 1.0
        return normalized, 127.5, 1.0
    elif img_array.dtype == np.uint16:
        # uint16: [0, 65535] -> [-1, 1]
        normalized = img_array.astype(np.float32) / 32767.5 - 1.0
        return normalized, 32767.5, 1.0
    elif img_array.dtype in [np.int32, np.int16]:
        # PIL mode 'I' loads as int32
        normalized = img_array.astype(np.float32) / 32767.5 - 1.0
        return normalized, 32767.5, 1.0
    else:
        # Assume already float in [0, 1] or [-1, 1]
        if img_array.min() >= 0:
            normalized = img_array.astype(np.float32) * 2.0 - 1.0
            return normalized, 0.5, 1.0
        else:
            return img_array.astype(np.float32), 1.0, 0.0


def float_to_uint16(img_float: np.ndarray) -> np.ndarray:
    """
    Convert [-1, 1] float image to uint16.

    Args:
        img_float: Image in [-1, 1] float range

    Returns:
        Grayscale image (H, W) uint16
    """
    # [-1, 1] -> [0, 65535]
    img_scaled = (img_float + 1.0) * 32767.5
    img_clamped = np.clip(img_scaled, 0, 65535)
    return img_clamped.astype(np.uint16)


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths - handle both --input_dir and --gt_dir for backward compatibility
    # Priority: --input_dir > --gt_dir > default (sinograms_masked)
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
    elif args.gt_dir is not None:
        input_dir = Path(args.gt_dir)
        print(f"Note: Using --gt_dir (deprecated, use --input_dir instead)")
    else:
        # Default to sinograms_masked (the proper input for inpainting)
        input_dir = SCRIPT_DIR / '../../../shared/sinogram_tiles/sinograms_masked'
        print(f"Using default input directory: {input_dir}")

    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input tiles not found at: {input_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask tiles not found at: {mask_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("MAT (Mask-Aware Transformer) Inference")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Input tiles directory: {input_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Truncation psi: {args.truncation_psi}")
    print(f"Noise mode: {args.noise_mode}")

    # Load model
    generator = load_mat_generator(str(checkpoint_path), device)

    # Get expected resolution from model
    expected_resolution = generator.img_resolution
    print(f"\nModel expects {expected_resolution}x{expected_resolution} images")

    # Get list of tiles
    tile_files = sorted(input_dir.glob('sino_*_tile_*.png'))
    total_tiles = len(tile_files)

    print(f"\nTotal tiles: {total_tiles}")

    # Handle start/end indices for parallel processing
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else total_tiles

    tile_files = tile_files[start_idx:end_idx]
    print(f"Processing tiles {start_idx} to {end_idx} ({len(tile_files)} tiles)")

    # Filter already processed if resuming
    if args.resume:
        remaining_tiles = []
        for tile_path in tile_files:
            output_path = output_dir / tile_path.name
            if not output_path.exists():
                remaining_tiles.append(tile_path)
        print(f"Resuming: {len(tile_files) - len(remaining_tiles)} tiles already done")
        tile_files = remaining_tiles

    if len(tile_files) == 0:
        print("All tiles already processed!")
        return

    print(f"Tiles to process: {len(tile_files)}")

    # Process in batches
    batch_size = args.batch_size
    n_batches = (len(tile_files) + batch_size - 1) // batch_size

    start_time = time.time()
    processed = 0
    input_format_logged = False

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(tile_files))
        batch_files = tile_files[batch_start:batch_end]

        # Load batch
        images = []
        masks = []
        valid_files = []

        for tile_path in batch_files:
            # Load image (16-bit grayscale)
            img = Image.open(tile_path)
            img_array = np.array(img)

            # Log input format once for debugging
            if not input_format_logged:
                print(f"\nInput tile format detected:")
                print(f"  PIL mode: {img.mode}")
                print(f"  Array dtype: {img_array.dtype}")
                print(f"  Array shape: {img_array.shape}")
                print(f"  Value range: [{img_array.min()}, {img_array.max()}]")
                input_format_logged = True

            # Check resolution matches model
            if img_array.shape[0] != expected_resolution or img_array.shape[1] != expected_resolution:
                print(f"\nWarning: Tile {tile_path.name} has shape {img_array.shape}, "
                      f"expected {expected_resolution}x{expected_resolution}")
                continue

            # Normalize to [-1, 1] for MAT
            img_normalized, _, _ = normalize_uint16_to_float(img_array)

            # Log normalization once
            if batch_idx == 0 and len(images) == 0:
                print(f"  Normalized range: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")

            # Convert grayscale to RGB (MAT expects 3-channel input)
            if img_normalized.ndim == 2:
                img_rgb = np.stack([img_normalized] * 3, axis=0)  # (3, H, W)
            else:
                img_rgb = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

            # Load corresponding mask
            mask_path = mask_dir / tile_path.name
            if not mask_path.exists():
                print(f"\nWarning: Mask not found for {tile_path.name}, skipping")
                continue

            mask_array = np.array(Image.open(mask_path))

            # MAT mask convention: 0 = inpaint, 1 = keep
            # RePaint mask format: 0 = inpaint, 255 = keep
            # These MATCH! Just normalize to [0, 1], NO inversion needed
            mask_normalized = mask_array.astype(np.float32) / 255.0

            # Convert to tensors
            img_tensor = torch.from_numpy(img_rgb).float()
            mask_tensor = torch.from_numpy(mask_normalized).float().unsqueeze(0)  # (1, H, W)

            images.append(img_tensor)
            masks.append(mask_tensor)
            valid_files.append(tile_path)

        if len(images) == 0:
            continue

        # Stack into batch
        images_batch = torch.stack(images, dim=0)  # (B, 3, H, W)
        masks_batch = torch.stack(masks, dim=0)    # (B, 1, H, W)

        # Log tensor shapes for first batch
        if batch_idx == 0:
            print(f"\nFirst batch tensor shapes:")
            print(f"  images: {images_batch.shape} (expected: B, 3, H, W)")
            print(f"  masks: {masks_batch.shape} (expected: B, 1, H, W)")
            print(f"  images range: [{images_batch.min():.3f}, {images_batch.max():.3f}]")
            print(f"  masks range: [{masks_batch.min():.3f}, {masks_batch.max():.3f}]")

        # Process batch
        try:
            results = process_batch_mat(
                generator, images_batch, masks_batch, device,
                truncation_psi=args.truncation_psi,
                noise_mode=args.noise_mode
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nGPU out of memory with batch size {batch_size}")
                print("Trying batch size 1...")
                torch.cuda.empty_cache()

                # Process one at a time
                results_list = []
                for i in range(len(images)):
                    img = images[i].unsqueeze(0)
                    msk = masks[i].unsqueeze(0)
                    result = process_batch_mat(
                        generator, img, msk, device,
                        truncation_psi=args.truncation_psi,
                        noise_mode=args.noise_mode
                    )
                    results_list.append(result.squeeze(0))
                results = torch.stack(results_list, dim=0)
            else:
                raise

        # Save results
        results_cpu = results.cpu().numpy()

        for i, tile_path in enumerate(valid_files):
            result = results_cpu[i]  # (3, H, W) in [-1, 1]

            output_path = output_dir / tile_path.name

            if args.save_grayscale:
                # Convert to grayscale uint16 for better precision in merging
                # Since all RGB channels should be similar (from grayscale input), take mean
                result_gray = np.mean(result, axis=0)  # (H, W) in [-1, 1]
                result_uint16 = float_to_uint16(result_gray)
                Image.fromarray(result_uint16).save(output_path)
            else:
                # Convert to uint8 RGB
                result_hwc = np.transpose(result, (1, 2, 0))  # (H, W, 3)
                result_uint8 = ((result_hwc + 1) * 127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(result_uint8).save(output_path)

        processed += len(valid_files)

        # Progress update
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(tile_files) - processed) / rate if rate > 0 else 0
            tqdm.write(
                f"  Processed {processed}/{len(tile_files)} tiles "
                f"({rate:.1f} tiles/s, ETA: {remaining/60:.1f} min)"
            )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed: {processed} tiles")
    print(f"Time: {elapsed/60:.2f} minutes ({processed/elapsed:.1f} tiles/s)")
    print(f"Output: {output_dir}")

    # Verify output
    output_tiles = list(output_dir.glob('sino_*_tile_*.png'))
    print(f"\nOutput tiles: {len(output_tiles)}")

    if len(output_tiles) > 0:
        sample = np.array(Image.open(output_tiles[0]))
        print(f"Sample output shape: {sample.shape}")
        print(f"Sample output dtype: {sample.dtype}")
        print(f"Sample output range: [{sample.min()}, {sample.max()}]")

    print(f"\nNext steps:")
    print(f"  python merge_mat_tiles.py")


if __name__ == '__main__':
    main()
