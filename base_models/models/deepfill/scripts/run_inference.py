#!/usr/bin/env python3
"""
DeepFill v2 Inference on Sinogram Tiles
========================================
This script runs DeepFill v2 inpainting on 256×256 sinogram tiles.

Key features:
- Batch processing for efficient GPU utilization
- Progress tracking with estimated time remaining
- Automatic GPU/CPU detection
- Preserves output precision for tile merging
- Supports resuming from interrupted runs
- On-the-fly mask inversion (can read RePaint masks directly)

Input: RGB tiles (H, W, 3) uint8 + binary masks
       Masks can be RePaint format (0=inpaint) or DeepFill format (255=inpaint)
Output: RGB tiles (H, W, 3) uint8 (infilled)

Author: Claude (Anthropic)
Date: 2025-11-29
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Add DeepFill repository to path
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent  # models/deepfill/
DEEPFILL_DIR = MODEL_DIR / 'DeepFillv2'
sys.path.insert(0, str(DEEPFILL_DIR))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run DeepFill v2 inference on sinogram tiles'
    )
    parser.add_argument(
        '--tiles_dir',
        type=str,
        default='../../../shared/sinogram_tiles',
        help='Directory containing prepared tiles'
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
        default='../DeepFillv2/pretrained/states_tf_celebahq.pth',
        help='Path to DeepFill v2 checkpoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
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
        '--invert_masks',
        action='store_true',
        default=False,
        help='Invert masks on-the-fly (use when reading RePaint masks where 0=inpaint)'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=None,
        help='Explicit path to GT tiles directory (overrides tiles_dir/sinograms_gt)'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default=None,
        help='Explicit path to masks directory (overrides tiles_dir/masks)'
    )
    return parser.parse_args()


def load_generator(checkpoint_path: str, device: torch.device):
    """
    Load DeepFill v2 generator model.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device

    Returns:
        Generator model in eval mode
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download the CelebA-HQ weights first."
        )

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    generator_state_dict = state_dict['G']

    # Detect network type based on keys
    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
        print("  Using networks.py (PyTorch-native)")
    else:
        from model.networks_tf import Generator
        print("  Using networks_tf.py (TensorFlow-compatible)")

    # Create generator
    generator = Generator(cnum_in=5, cnum=48, return_flow=False)
    generator.load_state_dict(generator_state_dict, strict=True)
    generator.to(device)
    generator.eval()

    print(f"  Model loaded successfully")
    return generator


def process_batch(
    generator,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Process a batch of images through DeepFill v2.

    Args:
        generator: DeepFill v2 generator model
        images: Batch of images (B, 3, H, W) in [0, 1]
        masks: Batch of masks (B, 1, H, W) in {0, 1} where 1=inpaint
        device: Torch device

    Returns:
        Inpainted images (B, 3, H, W) in [0, 1]
    """
    # Move to device and normalize to [-1, 1]
    images = images.to(device)
    masks = masks.to(device)

    # Normalize image to [-1, 1]
    images_norm = images * 2 - 1

    # Mask image (set masked regions to 0)
    images_masked = images_norm * (1. - masks)

    # Create 5-channel input: [masked_image(3), ones(1), mask(1)]
    ones = torch.ones_like(images_masked[:, 0:1, :, :])
    x = torch.cat([images_masked, ones, ones * masks], dim=1)

    with torch.inference_mode():
        _, x_stage2 = generator(x, masks)

    # Composite: original * (1-mask) + inpainted * mask
    result = images_norm * (1. - masks) + x_stage2 * masks

    # Convert back to [0, 1]
    result = (result + 1) / 2
    result = torch.clamp(result, 0, 1)

    return result


def normalize_to_float(img_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize image to [0, 1] float range, detecting bit depth automatically.

    Args:
        img_array: Input image array (any dtype)

    Returns:
        normalized: Image normalized to [0, 1] float32
        scale_factor: The divisor used (255.0 for 8-bit, 65535.0 for 16-bit)
    """
    if img_array.dtype == np.uint8:
        return img_array.astype(np.float32) / 255.0, 255.0
    elif img_array.dtype in [np.uint16, np.int32, np.int16]:
        # 16-bit or 32-bit integer (PIL mode "I" loads as int32)
        return img_array.astype(np.float32) / 65535.0, 65535.0
    elif img_array.dtype in [np.float32, np.float64]:
        # Already float - check if already normalized
        if img_array.max() <= 1.0:
            return img_array.astype(np.float32), 1.0
        elif img_array.max() <= 255.0:
            return img_array.astype(np.float32) / 255.0, 255.0
        else:
            return img_array.astype(np.float32) / 65535.0, 65535.0
    else:
        # Default to 16-bit assumption for safety
        return img_array.astype(np.float32) / 65535.0, 65535.0


def float_to_uint16(img_float: np.ndarray) -> np.ndarray:
    """
    Convert [0, 1] float image to uint16.

    Args:
        img_float: Image in [0, 1] float range

    Returns:
        Grayscale image (H, W) uint16
    """
    # Clamp to [0, 1] range
    img_clamped = np.clip(img_float, 0.0, 1.0)
    # Scale to uint16
    return (img_clamped * 65535.0).astype(np.uint16)


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)

    # Allow explicit gt_dir and mask_dir overrides (for using RePaint tiles directly)
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        gt_dir = tiles_dir / 'sinograms_gt'

    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
    else:
        mask_dir = tiles_dir / 'masks'

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Tiles not found at: {gt_dir}\n"
            f"Please ensure tiles exist or use --gt_dir to specify location."
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("DeepFill v2 Inference")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GT tiles directory: {gt_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Invert masks: {args.invert_masks}")

    # Load model
    generator = load_generator(args.checkpoint, device)

    # Get list of tiles
    tile_files = sorted(gt_dir.glob('sino_*_tile_*.png'))
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
            # Load image
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

            # Normalize to [0, 1] range, handling both uint8 and uint16/int32 inputs
            img_normalized, scale_factor = normalize_to_float(img_array)

            # Log normalization once for debugging
            if batch_idx == 0 and len(images) == 0:
                print(f"  Normalized range: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")
                print(f"  Scale factor used: {scale_factor}")

            # Ensure RGB (DeepFill expects 3-channel input)
            if img_normalized.ndim == 2:
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
            elif img_normalized.shape[-1] == 1:
                img_rgb = np.repeat(img_normalized, 3, axis=-1)
            else:
                img_rgb = img_normalized

            # Load corresponding mask
            mask_path = mask_dir / tile_path.name
            if not mask_path.exists():
                print(f"\nWarning: Mask not found for {tile_path.name}, skipping")
                continue

            mask = np.array(Image.open(mask_path))

            # Invert mask on-the-fly if reading RePaint masks (0=inpaint -> 255=inpaint)
            if args.invert_masks:
                mask = 255 - mask

            # Convert to tensors
            # img_rgb is already normalized to [0, 1]
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            # Mask is uint8, normalize to [0, 1]
            mask_tensor = torch.from_numpy(mask).float() / 255.0

            # Binarize mask (threshold at 0.5): DeepFill expects 1=inpaint
            mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0)

            images.append(img_tensor)
            masks.append(mask_tensor)
            valid_files.append(tile_path)

        if len(images) == 0:
            continue

        # Stack into batch
        images_batch = torch.stack(images, dim=0)
        masks_batch = torch.stack(masks, dim=0)

        # Process batch
        try:
            results = process_batch(generator, images_batch, masks_batch, device)
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
                    result = process_batch(generator, img, msk, device)
                    results_list.append(result.squeeze(0))
                results = torch.stack(results_list, dim=0)
            else:
                raise

        # Save results
        results_cpu = results.cpu().numpy()

        for i, tile_path in enumerate(valid_files):
            result = results_cpu[i]

            # Convert from (C, H, W) to (H, W, C)
            result_hwc = np.transpose(result, (1, 2, 0))

            output_path = output_dir / tile_path.name

            if args.save_grayscale:
                # Convert to grayscale uint16 for better precision in merging
                # Since all RGB channels are the same (grayscale), just take the first
                result_gray = result_hwc[:, :, 0]  # Result is already in [0, 1]
                result_uint16 = float_to_uint16(result_gray)
                Image.fromarray(result_uint16).save(output_path)
            else:
                # Convert to uint8 RGB
                result_uint8 = (np.clip(result_hwc, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(result_uint8).save(output_path)

        processed += len(valid_files)

        # Progress update
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
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
    print(f"  python merge_tiles.py")


if __name__ == '__main__':
    main()
