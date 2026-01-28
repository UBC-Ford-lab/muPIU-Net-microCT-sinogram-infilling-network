#!/usr/bin/env python3
"""
MAT Quality Diagnostic Script
==============================
Creates diagnostic visualizations to identify why MAT metrics appear noisy.

Generates:
1. Reconstructed slice comparison (MTF region) - all models vs ground truth
2. Sinogram quality check - zero pixel detection
3. Histogram comparison - distribution analysis
4. Difference maps - where does MAT diverge from ground truth?

Author: Claude (Anthropic)
Date: 2025-01-03
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
CT_RECON_DIR = SCRIPT_DIR.parent.parent.parent.parent
METRIC_CALC_DIR = CT_RECON_DIR / 'metric_calculators'

sys.path.insert(0, str(CT_RECON_DIR))
sys.path.insert(0, str(METRIC_CALC_DIR))

from ct_core import vff_io as vff

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parent / 'metrics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_reconstructions():
    """Load all reconstruction volumes."""
    print("Loading reconstruction volumes...")

    volumes = {}

    # Ground truth
    gt_path = CT_RECON_DIR / "data/results/Scan_1681_gt_recon.vff"
    if gt_path.exists():
        _, volumes['Ground Truth'] = vff.read_vff(str(gt_path), verbose=False)
        print(f"  Ground Truth: {volumes['Ground Truth'].shape}")

    # MAT
    mat_path = SCRIPT_DIR.parent / "results/reconstructed_volume.vff"
    if mat_path.exists():
        _, volumes['MAT'] = vff.read_vff(str(mat_path), verbose=False)
        print(f"  MAT: {volumes['MAT'].shape}")

    # DeepFill
    deepfill_path = CT_RECON_DIR / "Base_model_comparison/models/deepfill/results/reconstructed_volume.vff"
    if deepfill_path.exists():
        _, volumes['DeepFill'] = vff.read_vff(str(deepfill_path), verbose=False)
        print(f"  DeepFill: {volumes['DeepFill'].shape}")

    # LaMa
    lama_path = CT_RECON_DIR / "Base_model_comparison/models/lama/results/lama_reconstruction.vff"
    if lama_path.exists():
        _, volumes['LaMa'] = vff.read_vff(str(lama_path), verbose=False)
        print(f"  LaMa: {volumes['LaMa'].shape}")

    # RePaint
    repaint_path = CT_RECON_DIR / "Base_model_comparison/models/repaint/results/reconstructed_volume.vff"
    if repaint_path.exists():
        _, volumes['RePaint'] = vff.read_vff(str(repaint_path), verbose=False)
        print(f"  RePaint: {volumes['RePaint'].shape}")

    return volumes


def load_sinograms():
    """Load sample sinograms from MAT and DeepFill."""
    print("\nLoading sample sinograms...")

    sinograms = {}

    # MAT sinogram
    mat_sino_path = SCRIPT_DIR.parent / "data/sinograms_infilled/sino_0500.png"
    if mat_sino_path.exists():
        sinograms['MAT'] = np.array(Image.open(mat_sino_path))
        print(f"  MAT sino: {sinograms['MAT'].shape}, dtype: {sinograms['MAT'].dtype}")

    # DeepFill sinogram
    deepfill_sino_path = CT_RECON_DIR / "Base_model_comparison/models/deepfill/data/sinograms_infilled/sino_0500.png"
    if deepfill_sino_path.exists():
        sinograms['DeepFill'] = np.array(Image.open(deepfill_sino_path))
        print(f"  DeepFill sino: {sinograms['DeepFill'].shape}, dtype: {sinograms['DeepFill'].dtype}")

    return sinograms


def analyze_sinogram_quality(sinograms):
    """Analyze sinogram quality and detect issues."""
    print("\n" + "="*60)
    print("SINOGRAM QUALITY ANALYSIS")
    print("="*60)

    for name, sino in sinograms.items():
        print(f"\n{name}:")
        print(f"  Shape: {sino.shape}")
        print(f"  Dtype: {sino.dtype}")
        print(f"  Min: {sino.min()}")
        print(f"  Max: {sino.max()}")
        print(f"  Mean: {sino.mean():.2f}")
        print(f"  Std: {sino.std():.2f}")

        # Count zeros
        zero_count = np.sum(sino == 0)
        zero_pct = 100 * zero_count / sino.size
        print(f"  Zero pixels: {zero_count} ({zero_pct:.4f}%)")

        # Count saturated
        if sino.dtype == np.uint16:
            sat_count = np.sum(sino == 65535)
        else:
            sat_count = np.sum(sino == 255)
        sat_pct = 100 * sat_count / sino.size
        print(f"  Saturated pixels: {sat_count} ({sat_pct:.4f}%)")


def analyze_reconstruction_quality(volumes):
    """Analyze reconstruction quality differences."""
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY ANALYSIS (Slice 228 - MTF region)")
    print("="*60)

    if 'Ground Truth' not in volumes:
        print("  Ground truth not available for comparison")
        return

    gt = volumes['Ground Truth'][228, :, :]

    for name, vol in volumes.items():
        if name == 'Ground Truth':
            continue

        slice_data = vol[228, :, :]
        diff = slice_data.astype(np.float64) - gt.astype(np.float64)

        print(f"\n{name} vs Ground Truth:")
        print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.2f}")
        print(f"  MAE: {np.mean(np.abs(diff)):.2f}")
        print(f"  Max abs diff: {np.max(np.abs(diff)):.2f}")
        print(f"  Slice range: [{slice_data.min()}, {slice_data.max()}]")


def create_diagnostic_figure(volumes, sinograms):
    """Create comprehensive diagnostic figure."""
    print("\nCreating diagnostic figure...")

    fig = plt.figure(figsize=(20, 16))

    # ============ ROW 1: Reconstructed slices comparison ============
    slice_idx = 228  # MTF region

    # Get consistent window/level from ground truth
    if 'Ground Truth' in volumes:
        gt_slice = volumes['Ground Truth'][slice_idx, :, :]
        vmin = np.percentile(gt_slice, 1)
        vmax = np.percentile(gt_slice, 99)
    else:
        vmin, vmax = None, None

    model_names = ['Ground Truth', 'MAT', 'DeepFill', 'LaMa', 'RePaint']

    for i, name in enumerate(model_names):
        ax = fig.add_subplot(4, 5, i + 1)
        if name in volumes:
            slice_data = volumes[name][slice_idx, :, :]
            im = ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)

            # Add MTF ROI rectangle
            crop_indices_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
            rect = Rectangle((crop_indices_MTF[2], crop_indices_MTF[0]),
                            crop_indices_MTF[3]-crop_indices_MTF[2],
                            crop_indices_MTF[1]-crop_indices_MTF[0],
                            linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{name}\n(slice {slice_idx})', fontsize=10)
        ax.axis('off')

    # ============ ROW 2: Zoomed MTF region ============
    crop = [270, 664, 522, 640]  # [y1, y2, x1, x2]

    for i, name in enumerate(model_names):
        ax = fig.add_subplot(4, 5, 6 + i)
        if name in volumes:
            slice_data = volumes[name][slice_idx, crop[0]:crop[1], crop[2]:crop[3]]
            ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} (MTF ROI)', fontsize=10)
        ax.axis('off')

    # ============ ROW 3: Sinogram comparison and histograms ============

    # MAT sinogram
    ax = fig.add_subplot(4, 5, 11)
    if 'MAT' in sinograms:
        sino = sinograms['MAT']
        ax.imshow(sino, cmap='gray', aspect='auto')
        zero_mask = sino == 0
        if np.any(zero_mask):
            # Overlay zeros in red
            overlay = np.zeros((*sino.shape, 4))
            overlay[zero_mask] = [1, 0, 0, 1]
            ax.imshow(overlay, aspect='auto')
    ax.set_title(f'MAT Sinogram\n(zeros: {np.sum(sinograms.get("MAT", np.array([]))==0)})', fontsize=10)
    ax.axis('off')

    # DeepFill sinogram
    ax = fig.add_subplot(4, 5, 12)
    if 'DeepFill' in sinograms:
        sino = sinograms['DeepFill']
        ax.imshow(sino, cmap='gray', aspect='auto')
        zero_mask = sino == 0
        if np.any(zero_mask):
            overlay = np.zeros((*sino.shape, 4))
            overlay[zero_mask] = [1, 0, 0, 1]
            ax.imshow(overlay, aspect='auto')
    ax.set_title(f'DeepFill Sinogram\n(zeros: {np.sum(sinograms.get("DeepFill", np.array([]))==0)})', fontsize=10)
    ax.axis('off')

    # Sinogram histograms
    ax = fig.add_subplot(4, 5, 13)
    if 'MAT' in sinograms:
        ax.hist(sinograms['MAT'].ravel(), bins=100, alpha=0.7, label='MAT', color='red')
    if 'DeepFill' in sinograms:
        ax.hist(sinograms['DeepFill'].ravel(), bins=100, alpha=0.7, label='DeepFill', color='blue')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.set_title('Sinogram Histograms', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # Zoomed histogram near zero
    ax = fig.add_subplot(4, 5, 14)
    if 'MAT' in sinograms:
        mat_low = sinograms['MAT'][sinograms['MAT'] < 5000]
        ax.hist(mat_low, bins=50, alpha=0.7, label='MAT', color='red')
    if 'DeepFill' in sinograms:
        df_low = sinograms['DeepFill'][sinograms['DeepFill'] < 5000]
        ax.hist(df_low, bins=50, alpha=0.7, label='DeepFill', color='blue')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.set_title('Low-value histogram (< 5000)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # Empty placeholder
    ax = fig.add_subplot(4, 5, 15)
    ax.axis('off')

    # ============ ROW 4: Difference maps ============
    if 'Ground Truth' in volumes:
        gt = volumes['Ground Truth'][slice_idx, :, :].astype(np.float64)

        diff_models = ['MAT', 'DeepFill', 'LaMa', 'RePaint']

        # Find global diff range for consistent colorbar
        all_diffs = []
        for name in diff_models:
            if name in volumes:
                diff = volumes[name][slice_idx, :, :].astype(np.float64) - gt
                all_diffs.append(diff)

        if all_diffs:
            diff_max = max(np.percentile(np.abs(d), 99) for d in all_diffs)
        else:
            diff_max = 100

        for i, name in enumerate(diff_models):
            ax = fig.add_subplot(4, 5, 16 + i)
            if name in volumes:
                diff = volumes[name][slice_idx, :, :].astype(np.float64) - gt
                im = ax.imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
                rmse = np.sqrt(np.mean(diff**2))
                ax.set_title(f'{name} - GT\n(RMSE: {rmse:.1f})', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        # Colorbar for difference maps
        ax = fig.add_subplot(4, 5, 20)
        ax.axis('off')

    plt.suptitle('MAT Quality Diagnostic - Slice 228 (MTF Region)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'diagnostic_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("="*60)
    print("MAT Quality Diagnostic")
    print("="*60)

    # Load data
    volumes = load_reconstructions()
    sinograms = load_sinograms()

    # Analyze quality
    analyze_sinogram_quality(sinograms)
    analyze_reconstruction_quality(volumes)

    # Create diagnostic figure
    create_diagnostic_figure(volumes, sinograms)

    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("="*60)


if __name__ == '__main__':
    main()
