# In this script, calculate SSIM
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ct_core import vff_io

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trained_projs_folders', type=str, nargs='+',
                     default=['data/results/Scan_1681_raw_with_preds'],
                     help='One or more folders with trained projections')
    p.add_argument('--ground_truth_projs_folders', type=str, nargs='+',
                     default=['data/scans/Scan_1681'],
                     help='One or more folders with ground truth projections')
    p.add_argument('--device',     type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def compute_metrics(gt: torch.Tensor, pred: torch.Tensor):
    """
    Compute MSE, RMSE, MAE, NRMSE, PSNR, SSIM and NCC between gt and pred.
    Both tensors must be on the same device (e.g. CUDA) and of shape
    [B, C, H, W] or [C, H, W] or [H, W].
    """
    # ensure float
    gt = gt.float()
    pred = pred.float()

    # if no batch dim, add one so we can unify
    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.ndim == 3:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)

    # ---- basic pixel-wise ----
    mse = F.mse_loss(pred, gt, reduction='mean')
    rmse = torch.sqrt(mse)
    mae = F.l1_loss(pred, gt, reduction='mean')

    # ---- range-based metrics ----
    data_range = float(gt.max() - gt.min())
    # if batch, data_range is a [B] tensor; we want per-sample PSNR/SSIM
    psnr_vals = peak_signal_noise_ratio(pred, gt, data_range=data_range)
    ssim_vals = structural_similarity_index_measure(pred, gt, data_range=data_range)

    # average across batch
    psnr_val = psnr_vals.mean()
    ssim_val = ssim_vals.mean()

    # ---- normalized RMSE ----
    # avoid division by zero
    nrmse = rmse / data_range if data_range > 0 else torch.tensor(float('nan'), device=gt.device)

    # ---- normalized cross-correlation ----
    # flatten per sample
    B = gt.shape[0]
    ncc_list = []
    for b in range(B):
        g = gt[b].reshape(-1)
        p = pred[b].reshape(-1)
        gm, pm = g.mean(), p.mean()
        gstd, pstd = g.std(unbiased=True), p.std(unbiased=True)
        if (gstd > 0) and (pstd > 0):
            ncc = ((g - gm) * (p - pm)).sum() / ((g.numel() - 1) * gstd * pstd)
        else:
            ncc = torch.tensor(float('nan'), device=gt.device)
        ncc_list.append(ncc)
    ncc_val = torch.stack(ncc_list).mean()

    # ---- gather into Python scalars ----
    return {
        'MSE':      mse.item(),
        'RMSE':     rmse.item(),
        'MAE':      mae.item(),
        'NRMSE':    nrmse.item(),
        'PSNR (dB)': psnr_val.item(),
        'SSIM':     ssim_val.item(),
        'NCC':      ncc_val.item(),
    }

class ProjectionDataset(Dataset):
    def __init__(self, gt_folder, pred_folder):
        self.pred_files = [
            f for f in os.listdir(pred_folder) if f.endswith('_pred.vff')
        ]
        self.gt_folder   = gt_folder
        self.pred_folder = pred_folder

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        fname = self.pred_files[idx]
        gt_name = fname.replace('_pred.vff', '.vff')
        # same read + postprocess…
        _, gt = vff_io.read_vff(
            os.path.join(self.gt_folder, gt_name), verbose=False
        )
        gt = gt.squeeze(0).byteswap().view(gt.dtype.newbyteorder())
        _, pr = vff_io.read_vff(
            os.path.join(self.pred_folder, fname), verbose=False
        )
        pr = pr.squeeze(0).byteswap().view(pr.dtype.newbyteorder())
        return fname, gt, pr


def main():
    args = parse_args()

    all_metrics = {
        'MSE': [], 'RMSE': [], 'MAE': [],
        'NRMSE': [], 'PSNR (dB)': [],
        'SSIM': [], 'NCC': []
    }

    for gt_folder in args.ground_truth_projs_folders:
        for pred_folder in args.trained_projs_folders:
            print(f"\nEvaluating: GT={gt_folder} | Pred={pred_folder}")
            ds = ProjectionDataset(gt_folder, pred_folder)
            dl = DataLoader(ds, batch_size=None, num_workers=4)
            for fname, gt, pr in tqdm(dl, total=len(ds), desc="DataLoader eval"):
                metrics = compute_metrics(gt, pr)
                for k, v in metrics.items():
                    all_metrics[k].append(v)

    # 4) after loop: print mean ± std
    print("\nSummary across all projections:")
    for name, vals in all_metrics.items():
        arr = np.array(vals, dtype=np.float32)
        if arr.size:
            mean, std = arr.mean(), arr.std()
            array_size = arr.size
            print(f"  {name:10s} -> mean: {mean:.4f},  std: {std:.4f},  count: {array_size}")
        else:
            print(f"  {name:10s} -> no values computed")



if __name__ == '__main__':
    main()