#!/usr/bin/env python3
"""
MAT Infilling Domain Comparison Script
=======================================

This script computes SSIM and PSNR metrics for MAT-based sinogram infilling
across three different domains:
1. Projection domain - Individual sinogram slices (2D images)
2. Sinogram domain - Same as projection domain for this analysis
3. Reconstruction domain - 3D CT reconstruction volumes

Compares:
- Sinogram domain: GT sinograms vs MAT infilled sinograms
- Reconstruction domain: GT reconstruction vs MAT reconstruction vs Projection infilling reconstruction

Written for CT reconstruction base model comparison.
Based on deepfill_domain_comparison.py structure.

Author: Claude (Anthropic)
Date: 2025-12-01
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Add ct_recon directory to path for imports (5 levels up from scripts/)
CT_RECON_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(CT_RECON_DIR)
from ct_core import vff_io


def parse_args():
    """Parse command line arguments with sensible defaults for MAT comparison."""
    p = argparse.ArgumentParser(
        description="Compare SSIM and PSNR across sinogram and reconstruction domains for MAT infilling"
    )
    p.add_argument(
        '--gt_sinogram_folder',
        type=str,
        default='../../../shared/sinogram_dataset/sinograms_gt',
        help='Folder containing ground truth sinograms (PNG files)'
    )
    p.add_argument(
        '--mat_sinogram_folder',
        type=str,
        default='../data/sinograms_infilled',
        help='Folder containing MAT infilled sinograms (PNG files)'
    )
    p.add_argument(
        '--gt_recon_file',
        type=str,
        default=str(Path(__file__).resolve().parents[4] / 'data/results/Scan_1681_gt_recon.vff'),
        help='Ground truth reconstruction VFF file'
    )
    p.add_argument(
        '--mat_recon_file',
        type=str,
        default='../results/reconstructed_volume.vff',
        help='MAT reconstruction VFF file'
    )
    p.add_argument(
        '--projection_infilling_recon_file',
        type=str,
        required=True,
        help='Projection infilling reconstruction VFF file for comparison'
    )
    p.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computations'
    )
    p.add_argument(
        '--output_dir',
        type=str,
        default='../metrics',
        help='Directory to save output results and plots'
    )
    return p.parse_args()


def compute_ssim(gt: torch.Tensor, pred: torch.Tensor):
    """
    Compute SSIM between gt and pred tensors.
    Handles different dimensionalities and ensures proper format for SSIM computation.

    Args:
        gt: Ground truth tensor
        pred: Predicted tensor

    Returns:
        SSIM value as float
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

    Args:
        gt: Ground truth tensor
        pred: Predicted tensor
        max_val: Optional maximum value for PSNR computation

    Returns:
        PSNR value in dB as float
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
    epsilon = 1e-10
    if mse < epsilon:
        return 100.0  # Cap PSNR at 100 dB for near-perfect reconstruction

    # Determine max_val if not provided
    if max_val is None:
        max_val = torch.max(torch.max(gt), torch.max(pred))

    # Handle edge case where max_val is 0
    if max_val == 0:
        if mse == 0:
            return 100.0
        else:
            return 0.0

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


class SinogramDataset(Dataset):
    """Dataset for loading sinogram pairs (GT and MAT infilled)"""

    def __init__(self, gt_folder, mat_folder):
        """
        Initialize sinogram dataset.

        Args:
            gt_folder: Path to ground truth sinogram folder
            mat_folder: Path to MAT infilled sinogram folder
        """
        if not os.path.exists(gt_folder):
            raise ValueError(f"GT sinogram folder does not exist: {gt_folder}")
        if not os.path.exists(mat_folder):
            raise ValueError(f"MAT sinogram folder does not exist: {mat_folder}")

        # Get all GT sinogram files
        self.gt_files = [
            f for f in os.listdir(gt_folder) if f.endswith('.png')
        ]
        # Sort files naturally to ensure proper ordering
        self.gt_files.sort(key=natural_sort_key)

        self.gt_folder = gt_folder
        self.mat_folder = mat_folder

        # Check for shape mismatch (GT is padded, MAT is original shape)
        gt_sample = np.array(Image.open(os.path.join(gt_folder, self.gt_files[0])))
        mat_sample = np.array(Image.open(os.path.join(mat_folder, self.gt_files[0])))

        if gt_sample.shape != mat_sample.shape:
            print(f"  Note: Shape mismatch detected - GT {gt_sample.shape} vs MAT {mat_sample.shape}")
            print(f"  GT will be center-cropped to match MAT dimensions")

        print(f"Found {len(self.gt_files)} sinogram pairs")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        """
        Get a sinogram pair.

        Args:
            idx: Index of the sinogram pair

        Returns:
            Tuple of (filename, gt_sinogram, mat_sinogram)
        """
        gt_fname = self.gt_files[idx]

        # MAT outputs use the same filename as GT
        mat_fname = gt_fname
        mat_path = os.path.join(self.mat_folder, mat_fname)

        # Read ground truth sinogram
        gt_path = os.path.join(self.gt_folder, gt_fname)
        try:
            gt_img = Image.open(gt_path)
            gt = np.array(gt_img, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Error reading GT sinogram {gt_path}: {e}")

        # Read MAT infilled sinogram
        try:
            mat_img = Image.open(mat_path)
            mat = np.array(mat_img, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Error reading MAT sinogram {mat_path}: {e}")

        # Handle shape mismatch: GT is padded (416, 3504), MAT is original (410, 3500)
        if gt.shape != mat.shape:
            gt_h, gt_w = gt.shape
            mat_h, mat_w = mat.shape

            # Calculate crop offsets (center crop)
            h_offset = (gt_h - mat_h) // 2
            w_offset = (gt_w - mat_w) // 2

            # Only crop if GT is larger than MAT (expected case)
            if gt_h >= mat_h and gt_w >= mat_w:
                gt = gt[h_offset:h_offset + mat_h, w_offset:w_offset + mat_w]
            else:
                raise ValueError(
                    f"Unexpected shape mismatch for {gt_fname}: GT {gt.shape} vs MAT {mat.shape}. "
                    f"Expected GT to be larger or equal."
                )

        return gt_fname, gt, mat


def compute_sinogram_domain_metrics(gt_folder, mat_folder):
    """
    Compute SSIM and PSNR in the sinogram domain (individual sinogram slices).

    Args:
        gt_folder: Path to ground truth sinogram folder
        mat_folder: Path to MAT infilled sinogram folder

    Returns:
        Tuple of (ssim_values, psnr_values) as numpy arrays
    """
    print("\nComputing SSIM and PSNR in Sinogram Domain...")
    print(f"  GT folder: {gt_folder}")
    print(f"  MAT folder: {mat_folder}")

    ssim_values = []
    psnr_values = []

    try:
        ds = SinogramDataset(gt_folder, mat_folder)
        dl = DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True)

        for fname, gt, mat in tqdm(dl, total=len(ds), desc="  Computing metrics"):
            if isinstance(gt, np.ndarray):
                gt_tensor = torch.from_numpy(gt)
            else:
                gt_tensor = gt

            if isinstance(mat, np.ndarray):
                mat_tensor = torch.from_numpy(mat)
            else:
                mat_tensor = mat

            ssim_val = compute_ssim(gt_tensor, mat_tensor)
            psnr_val = compute_psnr(gt_tensor, mat_tensor)

            ssim_values.append(ssim_val)
            psnr_values.append(psnr_val)

    except Exception as e:
        print(f"  Error during sinogram metric computation: {e}")
        raise

    return np.array(ssim_values), np.array(psnr_values)


def compute_reconstruction_domain_metrics(
    gt_recon_file,
    mat_recon_file,
    projection_infilling_recon_file=None
):
    """
    Compute SSIM and PSNR in the reconstruction domain using VFF reconstruction files.
    Computes SSIM and PSNR on a per-slice basis across the 3D volume.

    Args:
        gt_recon_file: Path to ground truth reconstruction VFF file
        mat_recon_file: Path to MAT reconstruction VFF file
        projection_infilling_recon_file: Optional path to projection infilling reconstruction VFF file

    Returns:
        Dictionary with 'mat_ssim', 'mat_psnr', and optionally 'projection_ssim', 'projection_psnr'
    """
    print("\nComputing SSIM and PSNR in Reconstruction Domain...")
    results = {}

    if not os.path.exists(gt_recon_file):
        print(f"  Error: Ground truth reconstruction file not found: {gt_recon_file}")
        raise FileNotFoundError(f"GT reconstruction file not found: {gt_recon_file}")

    # Read ground truth reconstruction
    print("  Loading ground truth reconstruction...")
    try:
        _, gt_recon = vff_io.read_vff(gt_recon_file, verbose=False)
        gt_recon = gt_recon.byteswap().view(gt_recon.dtype.newbyteorder()).astype(np.float32)
        print(f"  GT reconstruction shape: {gt_recon.shape}")
    except Exception as e:
        print(f"  Error loading GT reconstruction: {e}")
        raise

    # Compute metrics for MAT reconstruction
    if os.path.exists(mat_recon_file):
        print("  Loading MAT reconstruction...")
        try:
            _, mat_recon = vff_io.read_vff(mat_recon_file, verbose=False)
            mat_recon = mat_recon.byteswap().view(mat_recon.dtype.newbyteorder()).astype(np.float32)
            print(f"  MAT reconstruction shape: {mat_recon.shape}")

            if gt_recon.shape == mat_recon.shape:
                print(f"  Computing MAT SSIM and PSNR for {gt_recon.shape[0]} slices...")
                mat_ssim_values = []
                mat_psnr_values = []

                for slice_idx in tqdm(range(gt_recon.shape[0]), desc="  MAT metrics"):
                    gt_slice = torch.from_numpy(gt_recon[slice_idx])
                    mat_slice = torch.from_numpy(mat_recon[slice_idx])

                    ssim_val = compute_ssim(gt_slice, mat_slice)
                    psnr_val = compute_psnr(gt_slice, mat_slice)
                    mat_ssim_values.append(ssim_val)
                    mat_psnr_values.append(psnr_val)

                results['mat_ssim'] = np.array(mat_ssim_values)
                results['mat_psnr'] = np.array(mat_psnr_values)
            else:
                print(f"  Warning: Shape mismatch - GT: {gt_recon.shape}, MAT: {mat_recon.shape}")
        except Exception as e:
            print(f"  Error processing MAT reconstruction: {e}")
    else:
        print(f"  Warning: MAT reconstruction file not found: {mat_recon_file}")

    # Compute metrics for projection infilling reconstruction
    if projection_infilling_recon_file and os.path.exists(projection_infilling_recon_file):
        print("  Loading projection infilling reconstruction...")
        try:
            _, proj_recon = vff_io.read_vff(
                projection_infilling_recon_file, verbose=False
            )
            proj_recon = proj_recon.byteswap().view(proj_recon.dtype.newbyteorder()).astype(np.float32)
            print(f"  Projection infilling reconstruction shape: {proj_recon.shape}")

            if gt_recon.shape == proj_recon.shape:
                print(f"  Computing projection infilling SSIM and PSNR for {gt_recon.shape[0]} slices...")
                proj_ssim_values = []
                proj_psnr_values = []

                for slice_idx in tqdm(range(gt_recon.shape[0]), desc="  Projection metrics"):
                    gt_slice = torch.from_numpy(gt_recon[slice_idx])
                    proj_slice = torch.from_numpy(proj_recon[slice_idx])

                    ssim_val = compute_ssim(gt_slice, proj_slice)
                    psnr_val = compute_psnr(gt_slice, proj_slice)
                    proj_ssim_values.append(ssim_val)
                    proj_psnr_values.append(psnr_val)

                results['projection_ssim'] = np.array(proj_ssim_values)
                results['projection_psnr'] = np.array(proj_psnr_values)
            else:
                print(f"  Warning: Shape mismatch - GT: {gt_recon.shape}, Projection: {proj_recon.shape}")
        except Exception as e:
            print(f"  Error processing projection infilling reconstruction: {e}")
    elif projection_infilling_recon_file:
        print(f"  Info: Projection infilling reconstruction file not found: {projection_infilling_recon_file}")
        print("  Skipping projection infilling comparison...")

    return results


def print_statistics(values, domain_name, metric_type="SSIM"):
    """
    Print mean, std, min, max, and count for metric values.

    Args:
        values: Array of metric values
        domain_name: Name of the domain for display
        metric_type: Type of metric (SSIM or PSNR)
    """
    if len(values) == 0:
        print(f"{domain_name:35s} ({metric_type}) -> No values computed")
        return

    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()
    count = len(values)

    if metric_type == "PSNR":
        print(
            f"{domain_name:35s} ({metric_type}) -> "
            f"mean: {mean_val:6.2f} dB,  std: {std_val:5.2f} dB,  "
            f"min: {min_val:6.2f} dB,  max: {max_val:6.2f} dB,  count: {count}"
        )
    else:
        print(
            f"{domain_name:35s} ({metric_type}) -> "
            f"mean: {mean_val:.4f},  std: {std_val:.4f},  "
            f"min: {min_val:.4f},  max: {max_val:.4f},  count: {count}"
        )


def save_results_to_file(
    sino_ssim, sino_psnr, recon_metrics, output_dir
):
    """
    Save numerical results to a text file.

    Args:
        sino_ssim: Sinogram SSIM values
        sino_psnr: Sinogram PSNR values
        recon_metrics: Dictionary of reconstruction metrics
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'mat_metrics_summary.txt')

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MAT INFILLING DOMAIN COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Sinogram Domain Results
        f.write("SINOGRAM DOMAIN METRICS\n")
        f.write("-" * 80 + "\n")
        if len(sino_ssim) > 0:
            f.write(f"SSIM: mean={sino_ssim.mean():.4f}, std={sino_ssim.std():.4f}, ")
            f.write(f"min={sino_ssim.min():.4f}, max={sino_ssim.max():.4f}, count={len(sino_ssim)}\n")
        if len(sino_psnr) > 0:
            f.write(f"PSNR: mean={sino_psnr.mean():.2f} dB, std={sino_psnr.std():.2f} dB, ")
            f.write(f"min={sino_psnr.min():.2f} dB, max={sino_psnr.max():.2f} dB, count={len(sino_psnr)}\n")
        f.write("\n")

        # Reconstruction Domain Results
        f.write("RECONSTRUCTION DOMAIN METRICS\n")
        f.write("-" * 80 + "\n")
        f.write("NOTE: Reconstruction metrics filtered to include only slices with PSNR < 26 dB\n")
        f.write("(Removes uninformative slices with near-perfect reconstruction)\n")
        f.write("-" * 80 + "\n")

        # MAT reconstruction
        if 'mat_ssim' in recon_metrics:
            mat_ssim = recon_metrics['mat_ssim']
            f.write(f"MAT SSIM: mean={mat_ssim.mean():.4f}, std={mat_ssim.std():.4f}, ")
            f.write(f"min={mat_ssim.min():.4f}, max={mat_ssim.max():.4f}, count={len(mat_ssim)}\n")
        if 'mat_psnr' in recon_metrics:
            mat_psnr = recon_metrics['mat_psnr']
            f.write(f"MAT PSNR: mean={mat_psnr.mean():.2f} dB, std={mat_psnr.std():.2f} dB, ")
            f.write(f"min={mat_psnr.min():.2f} dB, max={mat_psnr.max():.2f} dB, count={len(mat_psnr)}\n")
        f.write("\n")

        # Projection infilling reconstruction
        if 'projection_ssim' in recon_metrics:
            proj_ssim = recon_metrics['projection_ssim']
            f.write(f"Projection Infilling SSIM: mean={proj_ssim.mean():.4f}, std={proj_ssim.std():.4f}, ")
            f.write(f"min={proj_ssim.min():.4f}, max={proj_ssim.max():.4f}, count={len(proj_ssim)}\n")
        if 'projection_psnr' in recon_metrics:
            proj_psnr = recon_metrics['projection_psnr']
            f.write(f"Projection Infilling PSNR: mean={proj_psnr.mean():.2f} dB, std={proj_psnr.std():.2f} dB, ")
            f.write(f"min={proj_psnr.min():.2f} dB, max={proj_psnr.max():.2f} dB, count={len(proj_psnr)}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {output_file}")


def create_comparison_plots(
    gt_sino_folder,
    mat_sino_folder,
    gt_recon_file,
    mat_recon_file,
    projection_infilling_recon_file,
    output_dir
):
    """
    Create high-resolution production-ready comparison plots.

    Args:
        gt_sino_folder: Path to GT sinogram folder
        mat_sino_folder: Path to MAT sinogram folder
        gt_recon_file: Path to GT reconstruction VFF
        mat_recon_file: Path to MAT reconstruction VFF
        projection_infilling_recon_file: Path to projection infilling reconstruction VFF
        output_dir: Output directory for plots
    """
    print("\nCreating comparison visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # Set high-quality matplotlib settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    # 1. SINOGRAM COMPARISON
    print("  Creating sinogram comparison...")
    try:
        ds = SinogramDataset(gt_sino_folder, mat_sino_folder)
        if len(ds) > 0:
            middle_idx = len(ds) // 2
            fname, gt_sino, mat_sino = ds[middle_idx]

            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # GT sinogram
            im1 = ax1.imshow(gt_sino, cmap='gray', aspect='auto', interpolation='bilinear')
            ax1.set_title('Ground Truth Sinogram', fontweight='bold')
            ax1.set_xlabel('Detector Width')
            ax1.set_ylabel('Projection Angle')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # MAT sinogram
            im2 = ax2.imshow(mat_sino, cmap='gray', aspect='auto', interpolation='bilinear')
            ax2.set_title('MAT Infilled Sinogram', fontweight='bold')
            ax2.set_xlabel('Detector Width')
            ax2.set_ylabel('Projection Angle')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()
            sino_plot_path = os.path.join(output_dir, 'mat_sinogram_comparison.png')
            plt.savefig(sino_plot_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print(f"    Saved: {sino_plot_path}")
    except Exception as e:
        print(f"    Error creating sinogram comparison: {e}")

    # 2. RECONSTRUCTION COMPARISON
    print("  Creating reconstruction comparison...")
    try:
        if os.path.exists(gt_recon_file):
            _, gt_recon = vff_io.read_vff(gt_recon_file, verbose=False)
            gt_recon = gt_recon.byteswap().view(gt_recon.dtype.newbyteorder()).astype(np.float32)

            central_slice_idx = gt_recon.shape[0] // 2
            gt_slice = gt_recon[central_slice_idx]

            # Determine number of subplots based on available data
            n_plots = 1  # GT always available
            if os.path.exists(mat_recon_file):
                n_plots += 1
            if projection_infilling_recon_file and os.path.exists(projection_infilling_recon_file):
                n_plots += 1

            fig2, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
            if n_plots == 1:
                axes = [axes]

            plot_idx = 0

            # GT reconstruction
            im1 = axes[plot_idx].imshow(gt_slice, cmap='gray', interpolation='bilinear')
            axes[plot_idx].set_title('Ground Truth Reconstruction', fontweight='bold')
            axes[plot_idx].axis('off')
            plot_idx += 1

            # MAT reconstruction
            if os.path.exists(mat_recon_file):
                _, mat_recon = vff_io.read_vff(mat_recon_file, verbose=False)
                mat_recon = mat_recon.byteswap().view(mat_recon.dtype.newbyteorder()).astype(np.float32)
                mat_slice = mat_recon[central_slice_idx]

                im2 = axes[plot_idx].imshow(mat_slice, cmap='gray', interpolation='bilinear')
                axes[plot_idx].set_title('MAT Reconstruction', fontweight='bold')
                axes[plot_idx].axis('off')
                plot_idx += 1

            # Projection infilling reconstruction
            if projection_infilling_recon_file and os.path.exists(projection_infilling_recon_file):
                _, proj_recon = vff_io.read_vff(
                    projection_infilling_recon_file, verbose=False
                )
                proj_recon = proj_recon.byteswap().view(proj_recon.dtype.newbyteorder()).astype(np.float32)
                proj_slice = proj_recon[central_slice_idx]

                im3 = axes[plot_idx].imshow(proj_slice, cmap='gray', interpolation='bilinear')
                axes[plot_idx].set_title('Projection Infilling Reconstruction', fontweight='bold')
                axes[plot_idx].axis('off')

            plt.tight_layout()
            recon_plot_path = os.path.join(output_dir, 'mat_reconstruction_comparison.png')
            plt.savefig(recon_plot_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print(f"    Saved: {recon_plot_path}")
    except Exception as e:
        print(f"    Error creating reconstruction comparison: {e}")

    # Reset matplotlib settings
    plt.rcdefaults()
    print("  Comparison plots created successfully!")


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 80)
    print("MAT INFILLING DOMAIN COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")

    # Validate input paths
    if not os.path.exists(args.gt_sinogram_folder):
        raise FileNotFoundError(f"GT sinogram folder not found: {args.gt_sinogram_folder}")
    if not os.path.exists(args.mat_sinogram_folder):
        raise FileNotFoundError(f"MAT sinogram folder not found: {args.mat_sinogram_folder}")
    if not os.path.exists(args.gt_recon_file):
        raise FileNotFoundError(f"GT reconstruction file not found: {args.gt_recon_file}")

    # 1. Sinogram Domain Metrics
    sino_ssim, sino_psnr = compute_sinogram_domain_metrics(
        args.gt_sinogram_folder,
        args.mat_sinogram_folder
    )

    # 2. Reconstruction Domain Metrics
    recon_metrics = compute_reconstruction_domain_metrics(
        args.gt_recon_file,
        args.mat_recon_file,
        args.projection_infilling_recon_file
    )

    # Apply PSNR filtering
    print("\nApplying reconstruction domain filtering...")
    if 'projection_psnr' in recon_metrics:
        original_count = len(recon_metrics['projection_psnr'])
        filter_mask = recon_metrics['projection_psnr'] < 26
        n_kept = filter_mask.sum()
        n_removed = original_count - n_kept

        print(f"  Filter: PSNR < 26 dB")
        print(f"  Slices kept: {n_kept}/{original_count} ({100*n_kept/original_count:.1f}%)")
        print(f"  Slices removed: {n_removed}/{original_count} ({100*n_removed/original_count:.1f}%)")

        for key in list(recon_metrics.keys()):
            recon_metrics[key] = recon_metrics[key][filter_mask]
    elif 'mat_psnr' in recon_metrics:
        original_count = len(recon_metrics['mat_psnr'])
        filter_mask = recon_metrics['mat_psnr'] < 26
        n_kept = filter_mask.sum()
        n_removed = original_count - n_kept

        print(f"  Filter: PSNR < 26 dB (using MAT PSNR)")
        print(f"  Slices kept: {n_kept}/{original_count} ({100*n_kept/original_count:.1f}%)")
        print(f"  Slices removed: {n_removed}/{original_count} ({100*n_removed/original_count:.1f}%)")

        for key in list(recon_metrics.keys()):
            recon_metrics[key] = recon_metrics[key][filter_mask]
    else:
        print("  Warning: No PSNR metrics found, skipping filter")

    # Print results
    print("\n" + "=" * 80)
    print("SSIM AND PSNR COMPARISON RESULTS")
    print("=" * 80)

    # Sinogram Domain Results
    print("\nSINOGRAM DOMAIN:")
    print("-" * 80)
    print_statistics(sino_ssim, "Sinogram Domain", "SSIM")
    print_statistics(sino_psnr, "Sinogram Domain", "PSNR")

    # Reconstruction Domain Results
    print("\nRECONSTRUCTION DOMAIN:")
    print("-" * 80)
    if 'mat_ssim' in recon_metrics:
        print_statistics(recon_metrics['mat_ssim'], "Reconstruction (MAT)", "SSIM")
    if 'mat_psnr' in recon_metrics:
        print_statistics(recon_metrics['mat_psnr'], "Reconstruction (MAT)", "PSNR")

    if 'projection_ssim' in recon_metrics:
        print_statistics(recon_metrics['projection_ssim'], "Reconstruction (Projection Infilling)", "SSIM")
    if 'projection_psnr' in recon_metrics:
        print_statistics(recon_metrics['projection_psnr'], "Reconstruction (Projection Infilling)", "PSNR")

    # Save results to file
    save_results_to_file(sino_ssim, sino_psnr, recon_metrics, args.output_dir)

    # Create comparison plots
    create_comparison_plots(
        args.gt_sinogram_folder,
        args.mat_sinogram_folder,
        args.gt_recon_file,
        args.mat_recon_file,
        args.projection_infilling_recon_file,
        args.output_dir
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
