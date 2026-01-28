#!/usr/bin/env python3
"""
Compute Undersampled Baseline Metrics
=====================================

This script computes baseline SSIM and PSNR metrics for undersampled data
compared to ground truth across two domains:

1. Sinogram domain: Undersampled sinograms (every other projection zeroed)
2. Reconstruction domain: VFF reconstruction files (GT vs undersampled recon)

This provides a reference baseline showing how much quality is lost by 50%
undersampling without any model infilling.

Usage:
    # Full computation (sinogram + reconstruction)
    python compute_undersampled_baseline.py

    # Reconstruction domain only
    python compute_undersampled_baseline.py --recon-only

    # Custom VFF paths
    python compute_undersampled_baseline.py --recon-only \
        --gt-recon data/results/Scan_1681_gt_recon.vff \
        --undersampled-recon data/results/Scan_1681_no_pred_recon.vff

Output:
    Saves results to Base_model_comparison/shared/sinogram_dataset/undersampled_baseline_metrics.txt
"""

import argparse
import os
import re
import sys
import numpy as np
import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Add ct_core to path for VFF I/O
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # ct_recon/
sys.path.insert(0, str(project_root))
from ct_core import vff_io


def natural_sort_key(text):
    """Natural sorting key to handle numerical sequences in filenames."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def compute_ssim(gt: torch.Tensor, pred: torch.Tensor):
    """
    Compute SSIM between gt and pred tensors.

    Args:
        gt: Ground truth tensor
        pred: Predicted tensor

    Returns:
        SSIM value as float
    """
    gt = gt.float()
    pred = pred.float()

    if torch.any(torch.isnan(gt)) or torch.any(torch.isnan(pred)):
        return 0.0
    if torch.any(torch.isinf(gt)) or torch.any(torch.isinf(pred)):
        return 0.0

    # Add batch and channel dimensions
    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

    # Compute data range
    combined_min = torch.min(torch.min(gt), torch.min(pred))
    combined_max = torch.max(torch.max(gt), torch.max(pred))
    data_range = float(combined_max - combined_min)

    if data_range == 0:
        return 1.0 if torch.allclose(gt, pred, atol=1e-8) else 0.0

    data_range = max(data_range, 1e-8)

    try:
        ssim_vals = structural_similarity_index_measure(pred, gt, data_range=data_range)
        if torch.isnan(ssim_vals) or torch.isinf(ssim_vals):
            return 0.0
        return ssim_vals.mean().item()
    except Exception as e:
        print(f"Error in SSIM computation: {e}")
        return 0.0


def compute_psnr(gt: torch.Tensor, pred: torch.Tensor):
    """
    Compute PSNR between gt and pred tensors.

    Args:
        gt: Ground truth tensor
        pred: Predicted tensor

    Returns:
        PSNR value in dB as float
    """
    gt = gt.float()
    pred = pred.float()

    if torch.any(torch.isnan(gt)) or torch.any(torch.isnan(pred)):
        return 0.0
    if torch.any(torch.isinf(gt)) or torch.any(torch.isinf(pred)):
        return 0.0

    mse = torch.mean((gt - pred) ** 2)

    epsilon = 1e-10
    if mse < epsilon:
        return 100.0

    max_val = torch.max(torch.max(gt), torch.max(pred))
    if max_val == 0:
        return 100.0 if mse == 0 else 0.0

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    if torch.isnan(psnr):
        return 0.0
    if torch.isinf(psnr) or psnr > 100.0:
        return 100.0

    return psnr.item()


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Compute undersampled baseline SSIM/PSNR metrics"
    )
    p.add_argument('--recon-only', action='store_true',
                   help='Only compute reconstruction domain metrics (skip sinogram)')
    p.add_argument('--gt-recon', type=str,
                   default='data/results/Scan_1681_gt_recon.vff',
                   help='Ground truth reconstruction VFF file')
    p.add_argument('--undersampled-recon', type=str,
                   default='data/results/Scan_1681_no_pred_recon.vff',
                   help='Undersampled reconstruction VFF file (no model infilling)')
    p.add_argument('--psnr-threshold', type=float, default=26.0,
                   help='PSNR threshold for filtering (slices with PSNR >= threshold are excluded)')
    return p.parse_args()


def compute_reconstruction_domain_metrics(gt_recon_file, undersampled_recon_file, psnr_threshold=26.0):
    """
    Compute SSIM and PSNR in the reconstruction domain using VFF files.

    Computes metrics on a per-slice basis across the 3D volume, matching
    the methodology in domain_comparison.py.

    Args:
        gt_recon_file: Path to ground truth reconstruction VFF
        undersampled_recon_file: Path to undersampled reconstruction VFF
        psnr_threshold: PSNR threshold for filtering (default 26 dB)

    Returns:
        Dictionary with 'ssim', 'psnr' arrays and filtered versions
    """
    print("\n" + "=" * 80)
    print("RECONSTRUCTION DOMAIN METRICS")
    print("=" * 80)

    results = {}

    # Validate paths
    if not os.path.exists(gt_recon_file):
        print(f"ERROR: Ground truth reconstruction not found: {gt_recon_file}")
        return results

    if not os.path.exists(undersampled_recon_file):
        print(f"ERROR: Undersampled reconstruction not found: {undersampled_recon_file}")
        return results

    print(f"GT reconstruction: {gt_recon_file}")
    print(f"Undersampled reconstruction: {undersampled_recon_file}")
    print(f"PSNR filter threshold: {psnr_threshold} dB")

    # Load ground truth reconstruction
    print("\nLoading ground truth reconstruction...")
    _, gt_recon = vff_io.read_vff(gt_recon_file, verbose=False)
    gt_recon = gt_recon.byteswap().view(gt_recon.dtype.newbyteorder()).astype(np.float32)
    print(f"  GT shape: {gt_recon.shape}")

    # Load undersampled reconstruction
    print("Loading undersampled reconstruction...")
    _, under_recon = vff_io.read_vff(undersampled_recon_file, verbose=False)
    under_recon = under_recon.byteswap().view(under_recon.dtype.newbyteorder()).astype(np.float32)
    print(f"  Undersampled shape: {under_recon.shape}")

    # Validate shapes match
    if gt_recon.shape != under_recon.shape:
        print(f"ERROR: Shape mismatch - GT: {gt_recon.shape}, Undersampled: {under_recon.shape}")
        return results

    # Compute per-slice metrics
    n_slices = gt_recon.shape[0]
    print(f"\nComputing per-slice metrics for {n_slices} slices...")

    ssim_values = []
    psnr_values = []

    for slice_idx in tqdm(range(n_slices), desc="  Per-slice SSIM/PSNR"):
        gt_slice = torch.from_numpy(gt_recon[slice_idx])
        under_slice = torch.from_numpy(under_recon[slice_idx])

        ssim_val = compute_ssim(gt_slice, under_slice)
        psnr_val = compute_psnr(gt_slice, under_slice)

        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)

    ssim_values = np.array(ssim_values)
    psnr_values = np.array(psnr_values)

    # Store unfiltered results
    results['ssim_unfiltered'] = ssim_values
    results['psnr_unfiltered'] = psnr_values

    # Apply PSNR threshold filter (same as domain_comparison.py)
    # Filter out slices where PSNR >= threshold (these are typically edge/empty slices)
    filter_mask = psnr_values < psnr_threshold
    n_filtered = filter_mask.sum()

    results['ssim_filtered'] = ssim_values[filter_mask]
    results['psnr_filtered'] = psnr_values[filter_mask]
    results['filter_mask'] = filter_mask
    results['n_total'] = n_slices
    results['n_filtered'] = n_filtered

    # Print results
    print("\n" + "-" * 80)
    print("RECONSTRUCTION DOMAIN RESULTS")
    print("-" * 80)

    print(f"\nUnfiltered (all {n_slices} slices):")
    print(f"  SSIM: mean={ssim_values.mean():.4f}, std={ssim_values.std():.4f}, "
          f"min={ssim_values.min():.4f}, max={ssim_values.max():.4f}")
    print(f"  PSNR: mean={psnr_values.mean():.2f} dB, std={psnr_values.std():.2f} dB, "
          f"min={psnr_values.min():.2f} dB, max={psnr_values.max():.2f} dB")

    print(f"\nFiltered (PSNR < {psnr_threshold} dB): {n_filtered}/{n_slices} slices")
    if n_filtered > 0:
        print(f"  SSIM: mean={results['ssim_filtered'].mean():.4f}, std={results['ssim_filtered'].std():.4f}, "
              f"min={results['ssim_filtered'].min():.4f}, max={results['ssim_filtered'].max():.4f}")
        print(f"  PSNR: mean={results['psnr_filtered'].mean():.2f} dB, std={results['psnr_filtered'].std():.2f} dB, "
              f"min={results['psnr_filtered'].min():.2f} dB, max={results['psnr_filtered'].max():.2f} dB")
    else:
        print("  No slices passed filter!")

    return results


def compute_sinogram_domain_metrics():
    """
    Compute sinogram domain metrics for undersampled baseline.

    Returns:
        Dictionary with ssim_values, psnr_values arrays
    """
    # Define paths relative to script location
    script_dir_local = Path(__file__).resolve().parent
    sinogram_dataset_dir = script_dir_local.parent / 'sinogram_dataset'
    gt_folder = sinogram_dataset_dir / 'sinograms_gt'
    masks_folder = sinogram_dataset_dir / 'masks'

    print("\n" + "=" * 80)
    print("SINOGRAM DOMAIN METRICS")
    print("=" * 80)
    print(f"GT folder: {gt_folder}")
    print(f"Masks folder: {masks_folder}")

    results = {}

    # Validate paths
    if not gt_folder.exists():
        print(f"ERROR: GT sinogram folder not found: {gt_folder}")
        return results
    if not masks_folder.exists():
        print(f"ERROR: Masks folder not found: {masks_folder}")
        return results

    # Get all GT sinogram files
    gt_files = sorted(
        [f for f in os.listdir(gt_folder) if f.endswith('.png')],
        key=natural_sort_key
    )
    print(f"Found {len(gt_files)} GT sinograms")

    # Compute metrics
    ssim_values = []
    psnr_values = []

    print("\nComputing undersampled baseline metrics...")
    for gt_fname in tqdm(gt_files, desc="Processing sinograms"):
        # Load GT sinogram
        gt_path = gt_folder / gt_fname
        gt_img = np.array(Image.open(gt_path), dtype=np.float32)

        # Load corresponding mask
        mask_fname = gt_fname.replace('.png', '_mask001.png')
        mask_path = masks_folder / mask_fname

        if not mask_path.exists():
            print(f"Warning: Mask not found for {gt_fname}, skipping")
            continue

        mask_img = np.array(Image.open(mask_path))

        # Create undersampled sinogram by zeroing masked rows
        # Mask convention: 255 = inpaint region (missing), 0 = keep
        undersampled = gt_img.copy()
        undersampled[mask_img == 255] = 0

        # Convert to tensors
        gt_tensor = torch.from_numpy(gt_img)
        undersampled_tensor = torch.from_numpy(undersampled)

        # Compute metrics
        ssim_val = compute_ssim(gt_tensor, undersampled_tensor)
        psnr_val = compute_psnr(gt_tensor, undersampled_tensor)

        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)

    # Convert to arrays
    ssim_values = np.array(ssim_values)
    psnr_values = np.array(psnr_values)

    results['ssim'] = ssim_values
    results['psnr'] = psnr_values

    # Print results
    print("\n" + "-" * 80)
    print("SINOGRAM DOMAIN RESULTS")
    print("-" * 80)
    print(f"\nSinogram Domain Metrics (n={len(ssim_values)}):")
    print(f"  SSIM: mean={ssim_values.mean():.4f}, std={ssim_values.std():.4f}, "
          f"min={ssim_values.min():.4f}, max={ssim_values.max():.4f}")
    print(f"  PSNR: mean={psnr_values.mean():.2f} dB, std={psnr_values.std():.2f} dB, "
          f"min={psnr_values.min():.2f} dB, max={psnr_values.max():.2f} dB")

    return results


def main():
    """Main execution function."""
    args = parse_args()

    # Define output file path
    script_dir_local = Path(__file__).resolve().parent
    sinogram_dataset_dir = script_dir_local.parent / 'sinogram_dataset'
    output_file = sinogram_dataset_dir / 'undersampled_baseline_metrics.txt'

    print("=" * 80)
    print("UNDERSAMPLED BASELINE METRICS COMPUTATION")
    print("=" * 80)

    sino_results = {}
    recon_results = {}

    # Compute sinogram domain metrics (unless --recon-only)
    if not args.recon_only:
        sino_results = compute_sinogram_domain_metrics()

    # Compute reconstruction domain metrics
    # Resolve paths relative to project root
    gt_recon = Path(args.gt_recon)
    under_recon = Path(args.undersampled_recon)

    if not gt_recon.is_absolute():
        gt_recon = project_root / gt_recon
    if not under_recon.is_absolute():
        under_recon = project_root / under_recon

    recon_results = compute_reconstruction_domain_metrics(
        str(gt_recon),
        str(under_recon),
        args.psnr_threshold
    )

    # Save results to file
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UNDERSAMPLED BASELINE METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Description: Metrics comparing undersampled data against ground truth.\n")
        f.write("This represents the baseline quality loss from 50% undersampling\n")
        f.write("without any model infilling.\n\n")

        # Sinogram domain results
        if sino_results:
            f.write("-" * 80 + "\n")
            f.write("SINOGRAM DOMAIN METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total sinograms: {len(sino_results['ssim'])}\n\n")
            f.write(f"  SSIM: mean={sino_results['ssim'].mean():.4f}, std={sino_results['ssim'].std():.4f}, "
                    f"min={sino_results['ssim'].min():.4f}, max={sino_results['ssim'].max():.4f}\n")
            f.write(f"  PSNR: mean={sino_results['psnr'].mean():.2f} dB, std={sino_results['psnr'].std():.2f} dB, "
                    f"min={sino_results['psnr'].min():.2f} dB, max={sino_results['psnr'].max():.2f} dB\n\n")

        # Reconstruction domain results
        if recon_results:
            f.write("-" * 80 + "\n")
            f.write("RECONSTRUCTION DOMAIN METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"GT file: {gt_recon}\n")
            f.write(f"Undersampled file: {under_recon}\n")
            f.write(f"PSNR filter threshold: {args.psnr_threshold} dB\n\n")

            f.write(f"Unfiltered (all {recon_results['n_total']} slices):\n")
            f.write(f"  SSIM: mean={recon_results['ssim_unfiltered'].mean():.4f}, "
                    f"std={recon_results['ssim_unfiltered'].std():.4f}\n")
            f.write(f"  PSNR: mean={recon_results['psnr_unfiltered'].mean():.2f} dB, "
                    f"std={recon_results['psnr_unfiltered'].std():.2f} dB\n\n")

            f.write(f"Filtered (PSNR < {args.psnr_threshold} dB): {recon_results['n_filtered']}/{recon_results['n_total']} slices\n")
            if recon_results['n_filtered'] > 0:
                f.write(f"  SSIM: mean={recon_results['ssim_filtered'].mean():.4f}, "
                        f"std={recon_results['ssim_filtered'].std():.4f}\n")
                f.write(f"  PSNR: mean={recon_results['psnr_filtered'].mean():.2f} dB, "
                        f"std={recon_results['psnr_filtered'].std():.2f} dB\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Results saved to: {output_file}")

    # Print summary comparison with expected U-Net baseline values
    print("\n" + "=" * 80)
    print("VERIFICATION AGAINST U-NET BASELINE")
    print("=" * 80)
    print("\nExpected U-Net baseline values (from domain_comparison.py):")
    print("  Reconstruction Domain (Undersampled): SSIM=0.3478, PSNR=18.31 dB")
    print("\nNote: U-Net domain_comparison.py filters using predicted_psnr mask,")
    print("      then applies same mask to undersampled metrics. Small differences")
    print("      are expected when filtering directly on undersampled PSNR.")

    if recon_results and recon_results['n_filtered'] > 0:
        computed_ssim = recon_results['ssim_filtered'].mean()
        computed_psnr = recon_results['psnr_filtered'].mean()
        print(f"\nComputed values (filtered, PSNR < {args.psnr_threshold} dB):")
        print(f"  Reconstruction Domain (Undersampled): SSIM={computed_ssim:.4f}, PSNR={computed_psnr:.2f} dB")

        ssim_diff = abs(computed_ssim - 0.3478)
        psnr_diff = abs(computed_psnr - 18.31)
        ssim_pct = ssim_diff / 0.3478 * 100
        psnr_pct = psnr_diff / 18.31 * 100

        print(f"\n  Difference: SSIM={ssim_diff:.4f} ({ssim_pct:.1f}%), PSNR={psnr_diff:.2f} dB ({psnr_pct:.1f}%)")

        # Use 1% tolerance for verification (accounts for filtering differences)
        if ssim_pct < 1.0 and psnr_pct < 1.0:
            print("\n✓ VERIFICATION PASSED: Computed values match U-Net baseline (<1% difference)")
        elif ssim_pct < 5.0 and psnr_pct < 5.0:
            print("\n~ ACCEPTABLE MATCH: Small difference due to filtering approach (<5%)")
        else:
            print(f"\n⚠ SIGNIFICANT DISCREPANCY DETECTED (>5% difference)")

    print("\n" + "=" * 80)
    print("BASELINE COMPUTATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
