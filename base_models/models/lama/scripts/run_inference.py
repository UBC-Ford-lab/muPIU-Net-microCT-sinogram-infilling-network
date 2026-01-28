#!/usr/bin/env python3
"""
LaMa Inference Script for CT Sinogram Inpainting
=================================================

Runs LaMa inference on sinogram dataset with progress tracking and batch processing.

Usage:
    python3 run_inference.py                         # Process all (resumes if interrupted)
    python3 run_inference.py --start 0 --end 100     # Process first 100
    python3 run_inference.py --no-resume             # Reprocess all, overwriting existing
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Run LaMa inference on sinograms")
    p.add_argument('--start', type=int, default=0, help='Start index (default: 0)')
    p.add_argument('--end', type=int, default=None, help='End index (default: all)')
    p.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    p.add_argument('--model_path', type=str, default='../lama-repo/big-lama', help='Path to LaMa model')
    p.add_argument('--input_dir', type=str, default='../../../shared/sinogram_dataset', help='Input dataset directory')
    p.add_argument('--output_dir', type=str, default='../data/sinograms_infilled', help='Output directory')
    p.add_argument('--skip_setup', action='store_true', help='Skip dependency check and model download')
    p.add_argument('--resume', action='store_true', default=True, help='Resume from where left off, skipping existing outputs (default: True)')
    p.add_argument('--no-resume', dest='resume', action='store_false', help='Process all sinograms, overwriting existing outputs')
    return p.parse_args()


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available, will use CPU (slower)")
    except ImportError:
        print("  ✗ PyTorch not found!")
        print("  Install with: pip3 install --user torch==1.8.0 torchvision==0.9.0")
        return False

    try:
        import yaml
        import hydra
        print("  ✓ LaMa dependencies found")
    except ImportError:
        print("  ✗ LaMa dependencies not found!")
        print("  Install with: cd lama && pip3 install --user -r requirements.txt")
        return False

    return True


def download_model(model_path):
    """Download pre-trained LaMa model if not exists."""
    if Path(model_path).exists():
        print(f"✓ Model already exists: {model_path}")
        return True

    print(f"Downloading LaMa model to {model_path}...")

    # Navigate to lama directory
    lama_dir = Path(__file__).parent.parent / 'lama-repo'
    if not lama_dir.exists():
        print(f"  ✗ LaMa directory not found: {lama_dir}")
        return False

    os.chdir(lama_dir)

    try:
        # Download model
        print("  Downloading big-lama.zip...")
        subprocess.run(['curl', '-LJO', 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip'], check=True)

        # Extract
        print("  Extracting...")
        subprocess.run(['unzip', '-q', 'big-lama.zip'], check=True)

        # Cleanup
        Path('big-lama.zip').unlink()

        print("  ✓ Model downloaded")
        os.chdir('..')
        return True

    except Exception as e:
        print(f"  ✗ Failed to download model: {e}")
        os.chdir('..')
        return False


def prepare_batch_data(input_dir, start_idx, end_idx, batch_dir, output_dir=None, resume=True):
    """Copy a batch of sinograms and masks to temporary directory.

    Args:
        input_dir: Input dataset directory
        start_idx: Start index for batch
        end_idx: End index for batch
        batch_dir: Temporary directory for batch processing
        output_dir: Output directory to check for existing files (for resume)
        resume: If True, skip sinograms that already have output files

    Returns:
        tuple: (copied_count, skipped_count)
    """
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(exist_ok=True, parents=True)

    # Clear previous batch
    for f in batch_dir.glob('*.png'):
        f.unlink()

    # Copy sinograms
    sino_lama = Path(input_dir) / 'sinograms_lama'
    masks = Path(input_dir) / 'masks'
    output_path = Path(output_dir) if output_dir else None

    copied_count = 0
    skipped_count = 0
    for i in range(start_idx, end_idx):
        sino_file = sino_lama / f"sino_{i:04d}.png"
        mask_file = masks / f"sino_{i:04d}_mask001.png"

        # Check if output already exists (resume mode)
        if resume and output_path:
            output_file = output_path / f"sino_{i:04d}_mask001.png"
            if output_file.exists():
                skipped_count += 1
                continue

        if sino_file.exists() and mask_file.exists():
            shutil.copy(sino_file, batch_dir)
            shutil.copy(mask_file, batch_dir)
            copied_count += 1

    return copied_count, skipped_count


def run_lama_batch(model_path, input_dir, output_dir):
    """Run LaMa inference on a batch of images."""
    # Set environment
    lama_dir = (Path(__file__).parent.parent / 'lama-repo').resolve()
    env = os.environ.copy()
    env['TORCH_HOME'] = str(lama_dir)
    env['PYTHONPATH'] = str(lama_dir)

    # Run inference
    cmd = [
        'python3', 'bin/predict.py',
        f'model.path={Path(model_path).resolve()}',
        f'indir={Path(input_dir).resolve()}',
        f'outdir={Path(output_dir).resolve()}'
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=lama_dir,
            env=env,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e)


def main():
    args = parse_args()

    print("=" * 70)
    print("LaMa Inference for CT Sinogram Inpainting")
    print("=" * 70)

    # Check setup
    if not args.skip_setup:
        if not check_dependencies():
            print("\n✗ Dependency check failed!")
            print("Run: cd lama && pip3 install --user -r requirements.txt")
            return

        if not download_model(args.model_path):
            print("\n✗ Model download failed!")
            return

    # Get sinogram count
    sino_lama = Path(args.input_dir) / 'sinograms_lama'
    if not sino_lama.exists():
        print(f"\n✗ Input directory not found: {sino_lama}")
        return

    all_sinos = sorted(sino_lama.glob('sino_*.png'))
    total_sinos = len(all_sinos)

    start_idx = args.start
    end_idx = args.end if args.end is not None else total_sinos

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Count existing outputs for resume mode
    existing_outputs = 0
    if args.resume:
        existing_outputs = len(list(output_dir.glob('sino_*_mask001.png')))
        if existing_outputs > 0:
            print(f"\n✓ Resume mode: Found {existing_outputs} existing outputs, will skip those")

    print(f"\nProcessing sinograms {start_idx} to {end_idx} (total: {end_idx - start_idx})")
    print(f"Batch size: {args.batch_size}")
    print(f"Resume mode: {'enabled' if args.resume else 'disabled'}")
    print(f"Output: {args.output_dir}")

    # Process in batches
    batch_temp_dir = Path(__file__).parent.parent / 'lama_batch_temp'
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for batch_start in tqdm(range(start_idx, end_idx, args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, end_idx)

        # Prepare batch (with resume support)
        batch_size, batch_skipped = prepare_batch_data(
            args.input_dir,
            batch_start,
            batch_end,
            batch_temp_dir,
            output_dir=args.output_dir,
            resume=args.resume
        )
        skipped_count += batch_skipped

        if batch_size == 0:
            # All files in this batch were skipped (already processed)
            continue

        # Run inference
        success, output = run_lama_batch(
            args.model_path,
            batch_temp_dir,
            output_dir
        )

        if success:
            processed_count += batch_size
        else:
            failed_count += batch_size
            print(f"\n  ✗ Batch {batch_start}-{batch_end} failed:")
            print(f"  {output[:500]}")

    # Cleanup
    if batch_temp_dir.exists():
        shutil.rmtree(batch_temp_dir)

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Processed: {processed_count} sinograms")
    print(f"Skipped (already existed): {skipped_count} sinograms")
    print(f"Failed: {failed_count} sinograms")
    print(f"Total outputs: {len(list(output_dir.glob('sino_*_mask001.png')))} sinograms")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
