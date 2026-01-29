#!/usr/bin/env python3
"""
RePaint MTF/NPS/NEQ Metric Calculator
======================================
This script calculates MTF, NPS, and NEQ metrics for RePaint reconstruction
and generates comparison plots against Ground Truth and U-Net.

Outputs:
- MTF_results_absolute.png - Individual MTF curve
- NPS_results.png - Individual NPS curve
- NEQ_plot.png - Individual NEQ curve
- MTF_comparison.png - MTF comparison plot
- NPS_comparison.png - NPS comparison plot
- NEQ_comparison.png - NEQ comparison plot
- MTF_NPS_NEQ_unified_comparison.png - Unified comparison figure

Based on Metric calculators/Helper scripts/MTF_NPS_NEQ_unified.py

Author: Claude (Anthropic)
Date: 2025-01-03
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add ct_recon directory to path for imports (5 levels up from scripts/)
SCRIPT_DIR = Path(__file__).parent
CT_RECON_DIR = SCRIPT_DIR.parent.parent.parent.parent
METRIC_CALC_DIR = CT_RECON_DIR / 'metric_calculators'

sys.path.insert(0, str(CT_RECON_DIR))
sys.path.insert(0, str(METRIC_CALC_DIR))
sys.path.insert(0, str(METRIC_CALC_DIR / 'helper_scripts'))

from ct_core import vff_io as vff
from metric_calculators import mtf_calculator as MTF_calculator
from metric_calculators import nps_calculator as NPS_calculator
from metric_calculators import neq_calculator as NEQ_calculator

# Default output directory
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / 'metrics'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate MTF/NPS/NEQ metrics for RePaint reconstruction'
    )
    parser.add_argument('--gt_recon', type=str,
                       default=str(CT_RECON_DIR / 'data/results/ground_truth_reconstruction.vff'),
                       help='Ground truth reconstruction VFF file')
    parser.add_argument('--model_recon', type=str,
                       default=str(SCRIPT_DIR.parent / 'results/reconstructed_volume.vff'),
                       help='Model reconstruction VFF file')
    parser.add_argument('--unet_recon', type=str,
                       default=str(CT_RECON_DIR / 'data/results/unet_reconstruction.vff'),
                       help='U-Net reconstruction VFF file')
    parser.add_argument('--output_dir', type=str,
                       default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for metric results')
    return parser.parse_args()


def create_individual_plots(image_data, dataset_name, output_dir, pixel_size,
                           mtf_slice_range, crop_indices_MTF, edge_angle,
                           nps_slice_range, ROI_bounds_NPS):
    """Create individual MTF, NPS, NEQ plots for a single dataset."""

    print(f"\n{'='*60}")
    print(f"Creating individual plots for: {dataset_name}")
    print(f"{'='*60}")

    # Extract slice ranges
    image_data_mtf = image_data[mtf_slice_range[0]:mtf_slice_range[1], :, :]
    image_data_nps = image_data[nps_slice_range, :, :]

    # Calculate and plot MTF
    print("  Calculating MTF...")
    mtf_freq, mtf = MTF_calculator.get_MTF(
        image_data_mtf,
        crop_indices_MTF.copy(),
        find_absolute_MTF=True,
        pixel_size=pixel_size,
        target_directory=str(output_dir),
        plot_results=True,
        edge_angle=edge_angle,
        high_to_low=True,
        process_LSF=True,
        normalize_MTF=True
    )

    # Calculate and plot NPS
    print("  Calculating NPS...")
    nps_freq, nps = NPS_calculator.get_NPS(
        image_data_nps,
        ROI_bounds_NPS.copy(),
        pixel_size=pixel_size,
        target_directory=str(output_dir),
        plot_results=True,
        filter_low_freq=False
    )

    # Calculate and plot NEQ
    print("  Calculating NEQ...")
    neq_freq, neq = NEQ_calculator.get_NEQ(
        image_data_mtf,
        image_data_nps,
        crop_indices_MTF.copy(),
        ROI_bounds_NPS.copy(),
        pixel_size=pixel_size,
        target_directory=str(output_dir),
        plot_results=True,
        high_to_low_MTF=True
    )

    return mtf_freq, mtf, nps_freq, nps, neq_freq, neq


def create_comparison_plots(results, dataset_names, colors, output_dir, pixel_size):
    """Create individual comparison plots for MTF, NPS, and NEQ."""

    mtf_freq_results = results['mtf_freq']
    mtf_results = results['mtf']
    nps_freq_results = results['nps_freq']
    nps_results = results['nps']
    neq_freq_results = results['neq_freq']
    neq_results = results['neq']

    plt.style.use('seaborn-v0_8-whitegrid')

    # ===================== MTF Comparison Plot =====================
    fig_mtf, ax_mtf = plt.subplots(figsize=(10, 7))

    for i, (mtf_freq, mtf, name, color) in enumerate(zip(mtf_freq_results, mtf_results, dataset_names, colors)):
        positive_freq_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]

        # Find MTF50 and MTF10
        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)
        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
        except:
            mtf50_freq = 0.0
        try:
            mtf10_idx = np.where((mtf_interp < 0.1) & (mtf_freq_interp > 0))[0][0]
            mtf10_freq = mtf_freq_interp[mtf10_idx]
        except:
            mtf10_freq = 0.0

        ax_mtf.plot(mtf_freq_pos, mtf_pos, color=color, linewidth=2.5,
                   label=f'{name} (MTF50={mtf50_freq:.2f}, MTF10={mtf10_freq:.2f})', alpha=0.8)
        ax_mtf.axvline(mtf50_freq, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

    ax_mtf.axhline(0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax_mtf.axhline(0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax_mtf.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax_mtf.set_ylabel('Modulation Transfer Function', fontsize=12, fontweight='bold')
    ax_mtf.set_title('MTF Comparison', fontsize=14, fontweight='bold')
    ax_mtf.legend(fontsize=10, frameon=True)
    ax_mtf.grid(True, alpha=0.3)
    ax_mtf.set_xlim([0, 1/(2*pixel_size)])
    ax_mtf.set_ylim([0, 1.05])

    plt.tight_layout()
    mtf_path = output_dir / 'MTF_comparison.png'
    plt.savefig(mtf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {mtf_path}")

    # ===================== NPS Comparison Plot =====================
    fig_nps, ax_nps = plt.subplots(figsize=(10, 7))

    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        nps_log_safe = np.maximum(nps, np.max(nps) * 1e-6)
        noise_variance = np.trapz(nps, nps_freq)
        ax_nps.semilogy(nps_freq, nps_log_safe, color=color, linewidth=2.5,
                       label=f'{name} (σ²={noise_variance:.1f})', alpha=0.8)

    ax_nps.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax_nps.set_ylabel('NPS (HU² mm²)', fontsize=12, fontweight='bold')
    ax_nps.set_title('NPS Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax_nps.legend(fontsize=10, frameon=True)
    ax_nps.grid(True, alpha=0.3)
    ax_nps.set_xlim([0, np.max(nps_freq_results[0])])

    plt.tight_layout()
    nps_path = output_dir / 'NPS_comparison.png'
    plt.savefig(nps_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {nps_path}")

    # ===================== NEQ Comparison Plot =====================
    fig_neq, ax_neq = plt.subplots(figsize=(10, 7))

    for i, (neq_freq, neq, name, color) in enumerate(zip(neq_freq_results, neq_results, dataset_names, colors)):
        valid_mask = np.isfinite(neq) & (neq > 0)
        neq_freq_clean = neq_freq[valid_mask]
        neq_clean = neq[valid_mask]

        if len(neq_clean) > 0:
            max_neq = np.max(neq_clean)
            ax_neq.plot(neq_freq_clean, neq_clean, color=color, linewidth=2.5,
                       label=f'{name} (Peak={max_neq:.2f})', alpha=0.8)

    ax_neq.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax_neq.set_ylabel('NEQ (mm⁻²)', fontsize=12, fontweight='bold')
    ax_neq.set_title('NEQ Comparison', fontsize=14, fontweight='bold')
    ax_neq.legend(fontsize=10, frameon=True)
    ax_neq.grid(True, alpha=0.3)
    ax_neq.set_xlim([0, np.max([np.max(f) for f in neq_freq_results])])
    ax_neq.set_ylim(bottom=0)

    plt.tight_layout()
    neq_path = output_dir / 'NEQ_comparison.png'
    plt.savefig(neq_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {neq_path}")


def create_unified_comparison_plot(results, dataset_names, colors, output_dir, pixel_size):
    """Create unified 3-panel comparison plot."""

    mtf_freq_results = results['mtf_freq']
    mtf_results = results['mtf']
    nps_freq_results = results['nps_freq']
    nps_results = results['nps']
    neq_freq_results = results['neq_freq']
    neq_results = results['neq']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # MTF Plot
    for i, (mtf_freq, mtf, name, color) in enumerate(zip(mtf_freq_results, mtf_results, dataset_names, colors)):
        positive_freq_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]

        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)
        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
            ax1.axvline(mtf50_freq, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        except:
            mtf50_freq = 0.0
        try:
            mtf10_idx = np.where((mtf_interp < 0.1) & (mtf_freq_interp > 0))[0][0]
            mtf10_freq = mtf_freq_interp[mtf10_idx]
        except:
            mtf10_freq = 0.0

        ax1.plot(mtf_freq_pos, mtf_pos, color=color, linewidth=2.5,
                label=f'{name} (MTF50={mtf50_freq:.2f}, MTF10={mtf10_freq:.2f})', alpha=0.8)

    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axhline(0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Modulation Transfer Function', fontsize=12, fontweight='bold')
    ax1.set_title('MTF', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1/(2*pixel_size)])
    ax1.set_ylim([0, 1.05])

    # NPS Plot
    for i, (nps_freq, nps, name, color) in enumerate(zip(nps_freq_results, nps_results, dataset_names, colors)):
        nps_log_safe = np.maximum(nps, np.max(nps) * 1e-6)
        noise_variance = np.trapz(nps, nps_freq)
        ax2.semilogy(nps_freq, nps_log_safe, color=color, linewidth=2.5,
                    label=f'{name} (σ²={noise_variance:.1f})', alpha=0.8)

    ax2.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NPS (HU² mm²)', fontsize=12, fontweight='bold')
    ax2.set_title('NPS (Log Scale)', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, np.max(nps_freq_results[0])])

    # NEQ Plot
    for i, (neq_freq, neq, name, color) in enumerate(zip(neq_freq_results, neq_results, dataset_names, colors)):
        valid_mask = np.isfinite(neq) & (neq > 0)
        neq_freq_clean = neq_freq[valid_mask]
        neq_clean = neq[valid_mask]

        if len(neq_clean) > 0:
            max_neq = np.max(neq_clean)
            ax3.plot(neq_freq_clean, neq_clean, color=color, linewidth=2.5,
                    label=f'{name} (Peak={max_neq:.2f})', alpha=0.8)

    ax3.set_xlabel('Spatial Frequency (lp/mm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('NEQ (mm⁻²)', fontsize=12, fontweight='bold')
    ax3.set_title('NEQ', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, np.max([np.max(f) for f in neq_freq_results])])
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    unified_path = output_dir / 'MTF_NPS_NEQ_unified_comparison.png'
    plt.savefig(unified_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {unified_path}")


def print_summary(results, dataset_names):
    """Print comprehensive summary statistics."""

    print("\n" + "="*70)
    print("COMPREHENSIVE IMAGE QUALITY ASSESSMENT SUMMARY")
    print("="*70)

    for i, name in enumerate(dataset_names):
        print(f"\n{name.upper()}:")
        print("-" * (len(name) + 1))

        mtf_freq = results['mtf_freq'][i]
        mtf = results['mtf'][i]
        nps_freq = results['nps_freq'][i]
        nps = results['nps'][i]
        neq_freq = results['neq_freq'][i]
        neq = results['neq'][i]

        # MTF metrics
        positive_freq_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[positive_freq_mask]
        mtf_pos = mtf[positive_freq_mask]

        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)

        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50_freq = mtf_freq_interp[mtf50_idx]
        except:
            mtf50_freq = 0.0

        try:
            mtf10_idx = np.where((mtf_interp < 0.1) & (mtf_freq_interp > 0))[0][0]
            mtf10_freq = mtf_freq_interp[mtf10_idx]
        except:
            mtf10_freq = 0.0

        print(f"  MTF Metrics:")
        print(f"    MTF50: {mtf50_freq:.3f} lp/mm")
        print(f"    MTF10: {mtf10_freq:.3f} lp/mm")

        # NPS metrics
        noise_variance = np.trapz(nps, nps_freq)
        peak_nps = np.max(nps)
        peak_nps_freq = nps_freq[np.argmax(nps)]

        print(f"  NPS Metrics:")
        print(f"    Noise variance (σ²): {noise_variance:.2f} HU²")
        print(f"    Peak NPS: {peak_nps:.3f} HU² mm² at {peak_nps_freq:.3f} lp/mm")

        # NEQ metrics
        valid_mask = np.isfinite(neq) & (neq > 0)

        if np.any(valid_mask):
            neq_freq_clean = neq_freq[valid_mask]
            neq_clean = neq[valid_mask]
            max_neq = np.max(neq_clean)
            max_neq_freq = neq_freq_clean[np.argmax(neq_clean)]
            integrated_neq = np.trapz(neq_clean, neq_freq_clean)

            print(f"  NEQ Metrics:")
            print(f"    Peak NEQ: {max_neq:.3f} mm^-2 at {max_neq_freq:.3f} lp/mm")
            print(f"    Integrated NEQ: {integrated_neq:.3f} mm^-1")

    print("\n" + "="*70)


def main():
    """Main function to calculate all metrics for RePaint."""

    args = parse_args()

    print("="*70)
    print("RePaint MTF/NPS/NEQ Metric Calculator")
    print("="*70)

    # Set output directory from args
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Define VFF files to compare from args
    vff_files = [
        Path(args.gt_recon),
        Path(args.model_recon),
        Path(args.unet_recon)
    ]

    dataset_names = [
        "Ground Truth",
        "RePaint",
        "Task-specific U-Net"
    ]

    colors = ['#2E86C1', '#E74C3C', '#28B463']  # Blue, Red, Green

    # Check files exist
    for vff_file, name in zip(vff_files, dataset_names):
        if not vff_file.exists():
            print(f"ERROR: {name} VFF file not found: {vff_file}")
            return
        print(f"  Found: {name} -> {vff_file.name}")

    # Common parameters
    pixel_size = 0.085  # mm

    # MTF parameters
    mtf_slice_range = [228, 229]
    crop_indices_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
    edge_angle = 5.4

    # NPS parameters
    nps_slice_range = np.concatenate((np.arange(204, 208), np.arange(216, 226)))
    ROI_bounds_NPS = np.array([
        [178, 294, 510, 626],
        [258, 374, 750, 866],
        [310, 426, 328, 444],
        [432, 548, 830, 946],
        [488, 604, 248, 364],
        [580, 696, 730, 846],
        [624, 740, 414, 530],
        [724, 840, 598, 714]
    ])

    # Storage for results
    results = {
        'mtf_freq': [],
        'mtf': [],
        'mtf_max': [],
        'nps_freq': [],
        'nps': [],
        'neq_freq': [],
        'neq': []
    }

    # Process each VFF file
    for i, (vff_file, name) in enumerate(zip(vff_files, dataset_names)):
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        # Load VFF data
        print(f"  Loading {vff_file.name}...")
        header, image_data = vff.read_vff(str(vff_file), verbose=False)
        print(f"  Shape: {image_data.shape}")

        # Extract slice ranges
        image_data_mtf = image_data[mtf_slice_range[0]:mtf_slice_range[1], :, :]
        image_data_nps = image_data[nps_slice_range, :, :]

        # Calculate MTF (unnormalized for global normalization later)
        print(f"  Calculating MTF...")
        mtf_freq, mtf = MTF_calculator.get_MTF(
            image_data_mtf,
            crop_indices_MTF.copy(),
            find_absolute_MTF=True,
            pixel_size=pixel_size,
            target_directory=str(output_dir),
            plot_results=False,
            edge_angle=edge_angle,
            high_to_low=True,
            process_LSF=True,
            normalize_MTF=False
        )

        # Calculate NPS
        print(f"  Calculating NPS...")
        nps_freq, nps = NPS_calculator.get_NPS(
            image_data_nps,
            ROI_bounds_NPS.copy(),
            pixel_size=pixel_size,
            target_directory=str(output_dir),
            plot_results=False,
            filter_low_freq=False
        )

        # Calculate NEQ
        print(f"  Calculating NEQ...")
        neq_freq, neq = NEQ_calculator.get_NEQ(
            image_data_mtf,
            image_data_nps,
            crop_indices_MTF.copy(),
            ROI_bounds_NPS.copy(),
            pixel_size=pixel_size,
            target_directory=str(output_dir),
            plot_results=False,
            high_to_low_MTF=True
        )

        # Store results
        results['mtf_freq'].append(mtf_freq)
        results['mtf'].append(mtf)
        results['mtf_max'].append(mtf.max())
        results['nps_freq'].append(nps_freq)
        results['nps'].append(nps)
        results['neq_freq'].append(neq_freq)
        results['neq'].append(neq)

    # Individual MTF normalization (each curve normalized to unity)
    results['mtf'] = [mtf / mtf.max() for mtf in results['mtf']]

    # Create individual plots for RePaint only
    print("\n" + "="*60)
    print("Creating individual plots for RePaint...")
    print("="*60)

    # Load RePaint data again for individual plots
    header, repaint_data = vff.read_vff(str(vff_files[1]), verbose=False)
    create_individual_plots(
        repaint_data, "RePaint", output_dir, pixel_size,
        mtf_slice_range, crop_indices_MTF, edge_angle,
        nps_slice_range, ROI_bounds_NPS
    )

    # Create comparison plots
    print("\n" + "="*60)
    print("Creating comparison plots...")
    print("="*60)

    create_comparison_plots(results, dataset_names, colors, output_dir, pixel_size)
    create_unified_comparison_plot(results, dataset_names, colors, output_dir, pixel_size)

    # Print summary
    print_summary(results, dataset_names)

    print(f"\nAll metrics saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
