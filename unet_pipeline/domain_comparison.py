#!/usr/bin/env python3
"""
SSIM and PSNR Domain Comparison Script
=====================================

This script computes SSIM and PSNR across three different domains:
1. Projection domain - Individual projections (as in original evaluation)
2. 2D Sinogram domain - Built from all projections, computed per sinogram
3. Reconstruction domain - Using VFF reconstruction files, computed per slice

Written for CT reconstruction evaluation.
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from ct_core import vff_io


def parse_args():
    p = argparse.ArgumentParser(description="Compare SSIM across projection, sinogram, and reconstruction domains")
    p.add_argument('--trained_projs_folders', type=str, nargs='+',
                     default=['data/results/Scan_1681_raw_with_preds'],
                     help='One or more folders with trained projections')
    p.add_argument('--ground_truth_projs_folders', type=str, nargs='+',
                     default=['data/scans/Scan_1681'],
                     help='One or more folders with ground truth projections')
    p.add_argument('--gt_recon_file', type=str,
                     default='data/results/Scan_1681_gt_recon.vff',
                     help='Ground truth reconstruction VFF file')
    p.add_argument('--pred_recon_file', type=str,
                     default='data/results/Scan_1681_with_pred_recon.vff',
                     help='Predicted reconstruction VFF file')
    p.add_argument('--undersampled_recon_file', type=str,
                     default='data/results/Scan_1681_no_pred_recon.vff',
                     help='Undersampled reconstruction VFF file (without predictions)')
    p.add_argument('--device', type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu',
                   help='Device to use for computations')
    return p.parse_args()


def compute_ssim(gt: torch.Tensor, pred: torch.Tensor):
    """
    Compute SSIM between gt and pred tensors.
    Handles different dimensionalities and ensures proper format for SSIM computation.
    """
    # Ensure float
    gt = gt.float()
    pred = pred.float()

    # Check for invalid values
    if torch.any(torch.isnan(gt)) or torch.any(torch.isnan(pred)):
        print("Warning: NaN values detected in input tensors")
        return 0.0

    if torch.any(torch.isinf(gt)) or torch.any(torch.isinf(pred)):
        print("Warning: Inf values detected in input tensors")
        return 0.0

    # Add batch and channel dimensions if needed
    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.ndim == 3:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)

    # Compute data range from both tensors to capture full dynamic range
    combined_min = torch.min(torch.min(gt), torch.min(pred))
    combined_max = torch.max(torch.max(gt), torch.max(pred))
    data_range = float(combined_max - combined_min)

    # Handle edge case where data_range is 0 (constant images)
    if data_range == 0:
        if torch.allclose(gt, pred, atol=1e-8):
            return 1.0  # Perfect match for constant images
        else:
            return 0.0  # Different constant values

    # Ensure minimum data range to avoid numerical issues
    data_range = max(data_range, 1e-8)

    try:
        # Compute SSIM
        ssim_vals = structural_similarity_index_measure(pred, gt, data_range=data_range)

        # Check if result is valid
        if torch.isnan(ssim_vals) or torch.isinf(ssim_vals):
            print(f"Warning: SSIM computation returned {ssim_vals}")
            return 0.0

        return ssim_vals.mean().item()
    except Exception as e:
        print(f"Error in SSIM computation: {e}")
        return 0.0


def compute_psnr(gt: torch.Tensor, pred: torch.Tensor, max_val=None):
    """
    Compute PSNR between gt and pred tensors.
    Handles different dimensionalities and ensures proper format for PSNR computation.
    """
    # Ensure float
    gt = gt.float()
    pred = pred.float()

    # Check for invalid values
    if torch.any(torch.isnan(gt)) or torch.any(torch.isnan(pred)):
        print("Warning: NaN values detected in input tensors")
        return 0.0

    if torch.any(torch.isinf(gt)) or torch.any(torch.isinf(pred)):
        print("Warning: Inf values detected in input tensors")
        return 0.0

    # Compute MSE
    mse = torch.mean((gt - pred) ** 2)

    # Handle perfect or near-perfect reconstruction
    # Use a small epsilon to avoid infinite PSNR values
    epsilon = 1e-10
    if mse < epsilon:
        return 100.0  # Cap PSNR at 100 dB for near-perfect reconstruction

    # Determine max_val if not provided
    if max_val is None:
        max_val = torch.max(torch.max(gt), torch.max(pred))

    # Handle edge case where max_val is 0
    if max_val == 0:
        if mse == 0:
            return 100.0  # Both images are zero and identical
        else:
            return 0.0  # Different zero/near-zero images

    # Compute PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    # Check if result is valid and cap extremely high values
    if torch.isnan(psnr):
        print(f"Warning: PSNR computation returned NaN")
        return 0.0

    if torch.isinf(psnr) or psnr > 100.0:
        return 100.0  # Cap at 100 dB

    return psnr.item()


def natural_sort_key(text):
    """Natural sorting key to handle numerical sequences in filenames"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


class ProjectionDataset(Dataset):
    """Dataset for loading projection pairs (GT and predicted)"""
    def __init__(self, gt_folder, pred_folder):
        self.pred_files = [
            f for f in os.listdir(pred_folder) if f.endswith('_pred.vff')
        ]
        # Sort files naturally to ensure proper ordering for sinogram construction
        self.pred_files.sort(key=natural_sort_key)
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        fname = self.pred_files[idx]
        gt_name = fname.replace('_pred.vff', '.vff')

        # Read ground truth projection
        _, gt = vff_io.read_vff(
            os.path.join(self.gt_folder, gt_name), verbose=False
        )
        gt = gt.squeeze(0).byteswap().view(gt.dtype.newbyteorder())

        # Read predicted projection
        _, pr = vff_io.read_vff(
            os.path.join(self.pred_folder, fname), verbose=False
        )
        pr = pr.squeeze(0).byteswap().view(pr.dtype.newbyteorder())

        return fname, gt, pr


def compute_projection_domain_metrics(gt_folders, pred_folders):
    """
    Compute SSIM and PSNR in the projection domain (individual projections).
    This replicates the original evaluation approach.
    """
    print("Computing SSIM and PSNR in Projection Domain...")
    ssim_values = []
    psnr_values = []

    for gt_folder in gt_folders:
        for pred_folder in pred_folders:
            print(f"  Processing: GT={gt_folder} | Pred={pred_folder}")
            ds = ProjectionDataset(gt_folder, pred_folder)
            dl = DataLoader(ds, batch_size=None, num_workers=4)

            for fname, gt, pr in tqdm(dl, total=len(ds), desc="  Projection SSIM/PSNR"):
                ssim_val = compute_ssim(gt, pr)
                psnr_val = compute_psnr(gt, pr)
                ssim_values.append(ssim_val)
                psnr_values.append(psnr_val)

    return np.array(ssim_values), np.array(psnr_values)


def compute_sinogram_domain_metrics(gt_folders, pred_folders):
    """
    Compute SSIM and PSNR in the 2D sinogram domain using memory maps.
    Build height-based sinograms (per 2D slice) from all projections and compute SSIM and PSNR per sinogram.
    Uses temporary memory maps and optimizations for speed.
    """
    print("Computing SSIM and PSNR in 2D Sinogram Domain (height-based slices)...")
    ssim_values = []
    psnr_values = []

    for gt_folder in gt_folders:
        for pred_folder in pred_folders:
            print(f"  Processing: GT={gt_folder} | Pred={pred_folder}")

            # Get projection file list
            ds = ProjectionDataset(gt_folder, pred_folder)

            if len(ds) == 0:
                continue

            # First pass: determine dimensions by loading one projection
            fname, gt_sample, pr_sample = ds[0]
            height, width = gt_sample.shape
            n_projections = len(ds)

            print(f"  Projection dimensions: {height}x{width}, count: {n_projections}")

            # Create temporary memory-mapped arrays for the projection stacks
            gt_memmap = np.memmap(f'/tmp/gt_projections_{os.getpid()}.dat',
                                dtype=np.float32, mode='w+',
                                shape=(n_projections, height, width))
            pred_memmap = np.memmap(f'/tmp/pred_projections_{os.getpid()}.dat',
                                  dtype=np.float32, mode='w+',
                                  shape=(n_projections, height, width))

            # Load projections into memory maps with batch processing for speed
            print("  Loading projections to memory maps...")
            batch_size = min(32, n_projections)  # Process in batches

            for batch_start in tqdm(range(0, n_projections, batch_size), desc="  Loading batches"):
                batch_end = min(batch_start + batch_size, n_projections)
                batch_indices = list(range(batch_start, batch_end))

                # Load batch
                for i, idx in enumerate(batch_indices):
                    fname, gt, pr = ds[idx]
                    gt_memmap[idx] = np.array(gt, dtype=np.float32)
                    pred_memmap[idx] = np.array(pr, dtype=np.float32)

                # Flush periodically
                gt_memmap.flush()
                pred_memmap.flush()

            print(f"  Computing sinogram SSIM and PSNR for {height} height positions (2D slices)")

            # Process height-based sinograms (per 2D slice) with vectorized operations
            batch_size = min(16, height)  # Process multiple height slices at once

            for height_start in tqdm(range(0, height, batch_size), desc="  Computing height sinograms"):
                height_end = min(height_start + batch_size, height)

                # Process multiple height positions in parallel
                for height_idx in range(height_start, height_end):
                    # Extract sinogram: [n_projections, width] for this height slice
                    gt_sinogram = torch.from_numpy(gt_memmap[:, height_idx, :].copy())
                    pred_sinogram = torch.from_numpy(pred_memmap[:, height_idx, :].copy())

                    # Compute SSIM and PSNR for this height-based sinogram
                    ssim_val = compute_ssim(gt_sinogram, pred_sinogram)
                    psnr_val = compute_psnr(gt_sinogram, pred_sinogram)
                    ssim_values.append(ssim_val)
                    psnr_values.append(psnr_val)

            # Clean up memory maps
            del gt_memmap, pred_memmap
            try:
                os.unlink(f'/tmp/gt_projections_{os.getpid()}.dat')
                os.unlink(f'/tmp/pred_projections_{os.getpid()}.dat')
            except OSError:
                pass  # Files may already be cleaned up

    return np.array(ssim_values), np.array(psnr_values)


def compute_reconstruction_domain_metrics(gt_recon_file, pred_recon_file, undersampled_recon_file=None):
    """
    Compute SSIM and PSNR in the reconstruction domain using VFF reconstruction files.
    Computes SSIM and PSNR on a per-slice basis across the 3D volume.
    Returns dictionary with 'predicted' and optionally 'undersampled' SSIM and PSNR values.
    """
    print("Computing SSIM and PSNR in Reconstruction Domain...")
    results = {}

    if not os.path.exists(gt_recon_file):
        print(f"  Warning: Ground truth reconstruction file not found: {gt_recon_file}")
        return results

    # Read ground truth reconstruction
    print("  Loading ground truth reconstruction...")
    _, gt_recon = vff_io.read_vff(gt_recon_file, verbose=False)
    gt_recon = gt_recon.byteswap().view(gt_recon.dtype.newbyteorder()).astype(np.float32)

    # Compute SSIM for predicted reconstruction
    if os.path.exists(pred_recon_file):
        print("  Loading predicted reconstruction...")
        _, pred_recon = vff_io.read_vff(pred_recon_file, verbose=False)

        if gt_recon.shape == pred_recon.shape:
            pred_recon = pred_recon.byteswap().view(pred_recon.dtype.newbyteorder()).astype(np.float32)

            print(f"  Computing predicted SSIM and PSNR for {gt_recon.shape[0]} slices...")
            pred_ssim_values = []
            pred_psnr_values = []

            for slice_idx in tqdm(range(gt_recon.shape[0]), desc="  Predicted SSIM/PSNR"):
                gt_slice = torch.from_numpy(gt_recon[slice_idx])
                pred_slice = torch.from_numpy(pred_recon[slice_idx])

                ssim_val = compute_ssim(gt_slice, pred_slice)
                psnr_val = compute_psnr(gt_slice, pred_slice)
                pred_ssim_values.append(ssim_val)
                pred_psnr_values.append(psnr_val)

            results['predicted_ssim'] = np.array(pred_ssim_values)
            results['predicted_psnr'] = np.array(pred_psnr_values)
        else:
            print(f"  Warning: Shape mismatch - GT: {gt_recon.shape}, Pred: {pred_recon.shape}")
    else:
        print(f"  Warning: Predicted reconstruction file not found: {pred_recon_file}")

    # Compute SSIM for undersampled reconstruction
    if undersampled_recon_file and os.path.exists(undersampled_recon_file):
        print("  Loading undersampled reconstruction...")
        _, under_recon = vff_io.read_vff(undersampled_recon_file, verbose=False)

        if gt_recon.shape == under_recon.shape:
            under_recon = under_recon.byteswap().view(under_recon.dtype.newbyteorder()).astype(np.float32)

            print(f"  Computing undersampled SSIM and PSNR for {gt_recon.shape[0]} slices...")
            under_ssim_values = []
            under_psnr_values = []

            for slice_idx in tqdm(range(gt_recon.shape[0]), desc="  Undersampled SSIM/PSNR"):
                gt_slice = torch.from_numpy(gt_recon[slice_idx])
                under_slice = torch.from_numpy(under_recon[slice_idx])

                ssim_val = compute_ssim(gt_slice, under_slice)
                psnr_val = compute_psnr(gt_slice, under_slice)
                under_ssim_values.append(ssim_val)
                under_psnr_values.append(psnr_val)

            results['undersampled_ssim'] = np.array(under_ssim_values)
            results['undersampled_psnr'] = np.array(under_psnr_values)
        else:
            print(f"  Warning: Shape mismatch - GT: {gt_recon.shape}, Undersampled: {under_recon.shape}")
    elif undersampled_recon_file:
        print(f"  Warning: Undersampled reconstruction file not found: {undersampled_recon_file}")

    return results


def print_statistics(values, domain_name, metric_type="SSIM"):
    """Print mean, std, and count for metric values"""
    if len(values) == 0:
        print(f"{domain_name:20s} ({metric_type}) -> No values computed")
        return

    mean_val = values.mean()
    std_val = values.std()
    count = len(values)

    if metric_type == "PSNR":
        print(f"{domain_name:20s} ({metric_type}) -> mean: {mean_val:.2f} dB,  std: {std_val:.2f} dB,  count: {count}")
    else:
        print(f"{domain_name:20s} ({metric_type}) -> mean: {mean_val:.4f},  std: {std_val:.4f},  count: {count}")


def create_comparison_plots(gt_folders, pred_folders, gt_recon_file, pred_recon_file, undersampled_recon_file=None):
    """
    Create high-resolution production-ready comparison plots:
    1. Middle projections (GT vs Predicted) - separate figure
    2. Central sinogram slice (GT vs Predicted) - separate figure
    3. Central reconstruction slices (GT vs Predicted vs Undersampled) - separate figure
    """
    print("Creating high-resolution comparison visualizations...")

    # Set high-quality matplotlib settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    gt_folder = gt_folders[0]
    pred_folder = pred_folders[0]
    ds = ProjectionDataset(gt_folder, pred_folder)

    # 1. PROJECTIONS COMPARISON
    print("  Creating projection comparison...")
    if len(ds) > 0:
        middle_idx = len(ds) // 2
        fname, gt_proj, pred_proj = ds[middle_idx]

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # GT projection
        im1 = ax1.imshow(gt_proj, cmap='gray', interpolation='bilinear')
        ax1.set_title('Ground Truth Projection', fontweight='bold')
        ax1.axis('off')

        # Predicted projection
        im2 = ax2.imshow(pred_proj, cmap='gray', interpolation='bilinear')
        ax2.set_title('Predicted Projection', fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig('projection_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("    Saved: projection_comparison.png")

    # 2. SINOGRAM COMPARISON
    print("  Creating sinogram comparison (using all projections)...")
    if len(ds) > 0:
        # First determine dimensions from a sample projection
        fname, gt_sample, pr_sample = ds[0]
        height, width = gt_sample.shape
        n_projections = len(ds)
        central_height = height // 2

        print(f"    Processing {n_projections} projections for sinogram (memory efficient)...")

        # Load GT projections and extract central sinogram slice
        print("    Loading GT projections...")
        gt_projections = []
        for i in tqdm(range(n_projections), desc="    GT projections"):
            fname, gt, _ = ds[i]
            gt_projections.append(np.array(gt, dtype=np.float32))

        # Build GT sinogram and immediately extract central slice
        gt_stack = np.stack(gt_projections, axis=0)
        gt_sino = gt_stack[:, central_height, :].copy()  # Extract central slice

        # Clear GT projections from memory
        del gt_projections, gt_stack

        # Load Predicted projections and extract central sinogram slice
        print("    Loading Predicted projections...")
        pred_projections = []
        for i in tqdm(range(n_projections), desc="    Pred projections"):
            fname, _, pr = ds[i]
            pred_projections.append(np.array(pr, dtype=np.float32))

        # Build Predicted sinogram and immediately extract central slice
        pred_stack = np.stack(pred_projections, axis=0)
        pred_sino = pred_stack[:, central_height, :].copy()  # Extract central slice

        # Clear Predicted projections from memory
        del pred_projections, pred_stack

        # Now create the plot with both sinograms
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # GT sinogram
        im1 = ax1.imshow(gt_sino, cmap='gray', aspect='auto', interpolation='bilinear')
        ax1.set_title('Ground Truth Sinogram (Central Slice)', fontweight='bold')
        ax1.set_xlabel('Detector Width')
        ax1.set_ylabel('Projection Angle')

        # Predicted sinogram
        im2 = ax2.imshow(pred_sino, cmap='gray', aspect='auto', interpolation='bilinear')
        ax2.set_title('Predicted Sinogram (Central Slice)', fontweight='bold')
        ax2.set_xlabel('Detector Width')
        ax2.set_ylabel('Projection Angle')

        plt.tight_layout()
        plt.savefig('sinogram_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("    Saved: sinogram_comparison.png")

        # Clean up sinogram data
        del gt_sino, pred_sino

    # 3. RECONSTRUCTION COMPARISON
    print("  Creating reconstruction comparison...")

    # Load GT reconstruction
    if os.path.exists(gt_recon_file):
        _, gt_recon = vff_io.read_vff(gt_recon_file, verbose=False)
        gt_recon = gt_recon.byteswap().view(gt_recon.dtype.newbyteorder()).astype(np.float32)

        central_slice_idx = gt_recon.shape[0] // 2
        gt_slice = gt_recon[central_slice_idx]

        # Determine number of subplots based on available data
        n_plots = 1  # GT always available
        if os.path.exists(pred_recon_file):
            n_plots += 1
        if undersampled_recon_file and os.path.exists(undersampled_recon_file):
            n_plots += 1

        fig3, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # GT reconstruction
        im1 = axes[plot_idx].imshow(gt_slice, cmap='gray', interpolation='bilinear')
        axes[plot_idx].set_title('Ground Truth Reconstruction', fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1

        # Predicted reconstruction
        if os.path.exists(pred_recon_file):
            _, pred_recon = vff_io.read_vff(pred_recon_file, verbose=False)
            pred_recon = pred_recon.byteswap().view(pred_recon.dtype.newbyteorder()).astype(np.float32)
            pred_slice = pred_recon[central_slice_idx]

            im2 = axes[plot_idx].imshow(pred_slice, cmap='gray', interpolation='bilinear')
            axes[plot_idx].set_title('Predicted Reconstruction', fontweight='bold')
            axes[plot_idx].axis('off')
            plot_idx += 1

        # Undersampled reconstruction
        if undersampled_recon_file and os.path.exists(undersampled_recon_file):
            _, under_recon = vff_io.read_vff(undersampled_recon_file, verbose=False)
            under_recon = under_recon.byteswap().view(under_recon.dtype.newbyteorder()).astype(np.float32)
            under_slice = under_recon[central_slice_idx]

            im3 = axes[plot_idx].imshow(under_slice, cmap='gray', interpolation='bilinear')
            axes[plot_idx].set_title('Undersampled Reconstruction', fontweight='bold')
            axes[plot_idx].axis('off')

        plt.tight_layout()
        plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("    Saved: reconstruction_comparison.png")

    # Reset matplotlib settings
    plt.rcdefaults()

    print("  All high-resolution comparison plots created successfully!")


def main():
    args = parse_args()

    print("SSIM and PSNR Domain Comparison Analysis")
    print("=" * 50)

    # 1. Projection Domain Metrics
    proj_ssim, proj_psnr = compute_projection_domain_metrics(
        args.ground_truth_projs_folders,
        args.trained_projs_folders)

    # 2. Sinogram Domain Metrics
    sino_ssim, sino_psnr = compute_sinogram_domain_metrics(
        args.ground_truth_projs_folders,
        args.trained_projs_folders)

    # 3. Reconstruction Domain Metrics
    recon_metrics = compute_reconstruction_domain_metrics(
        args.gt_recon_file,
        args.pred_recon_file,
        args.undersampled_recon_file)

    # Print results
    print("\n" + "=" * 50)
    print("SSIM AND PSNR COMPARISON RESULTS")
    print("=" * 50)

    # Projection Domain Results
    print_statistics(proj_ssim, "Projection Domain", "SSIM")
    print_statistics(proj_psnr, "Projection Domain", "PSNR")

    # Sinogram Domain Results
    print_statistics(sino_ssim, "Sinogram Domain", "SSIM")
    print_statistics(sino_psnr, "Sinogram Domain", "PSNR")

    # presort to only slices that actually have information in them
    sort_mask = recon_metrics['predicted_psnr'] < 26 # this is arbitrary and I found it by looking at the data and seeing where the bad/too good to be true slices were
    for key in recon_metrics:
        recon_metrics[key] = recon_metrics[key][sort_mask]

    # Reconstruction Domain Results
    if 'predicted_ssim' in recon_metrics:
        print_statistics(recon_metrics['predicted_ssim'], "Recon Domain (Pred)", "SSIM")
    if 'predicted_psnr' in recon_metrics:
        print_statistics(recon_metrics['predicted_psnr'], "Recon Domain (Pred)", "PSNR")
    if 'undersampled_ssim' in recon_metrics:
        print_statistics(recon_metrics['undersampled_ssim'], "Recon Domain (Under)", "SSIM")
    if 'undersampled_psnr' in recon_metrics:
        print_statistics(recon_metrics['undersampled_psnr'], "Recon Domain (Under)", "PSNR")

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == '__main__':
    main()
    # Create comparison plots
    create_plots = False
    if create_plots == True:
        args = parse_args()
        create_comparison_plots(
            args.ground_truth_projs_folders,
            args.trained_projs_folders,
            args.gt_recon_file,
            args.pred_recon_file,
            args.undersampled_recon_file
        )