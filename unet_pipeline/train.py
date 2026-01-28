import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # new import

import wandb

from ct_core import vff_io
from unet_pipeline.model import UNet
from unet_pipeline.evaluate import compute_metrics

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = total_params * next(model.parameters()).element_size()  # bytes
    param_size_mb = param_size_bytes / (1024 * 1024)  # Convert bytes to MB
    return param_size_mb

def get_tensor_size(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    return tensor.numel() * tensor.element_size() / (1024 * 1024)  # Convert bytes to MB

class CTArtifactDataset(Dataset):
    """
    Loads triplets of CT projection images from CSV file.
    Each sample: (x1, x3) -> y (the middle projection x2)
    """
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        row = self.csv_file.iloc[idx]
        inp_paths = eval(row['x'])  # Convert string to list
        tgt_path = row['y']
        
        # Load images
        _, x1 = vff_io.read_vff(inp_paths[0], verbose=False)
        _, x3 = vff_io.read_vff(inp_paths[1], verbose=False)
        _, x2 = vff_io.read_vff(tgt_path, verbose=False)

        x1 = x1.squeeze(0)
        x1 = x1.byteswap().view(x1.dtype.newbyteorder())
        x3 = x3.squeeze(0)
        x3 = x3.byteswap().view(x3.dtype.newbyteorder())
        x2 = x2.squeeze(0)
        x2 = x2.byteswap().view(x2.dtype.newbyteorder())

        # Stack inputs
        inp = np.stack([x1, x3], axis=0)  # shape: 2,H,W
        tgt = x2[np.newaxis, ...]         # shape: 1,H,W

        inp = torch.from_numpy(inp).half()
        tgt = torch.from_numpy(tgt).half()

        return inp, tgt


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    i = 0
    for inp, tgt in tqdm(loader, desc='Train', leave=False):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            out = model(inp)
            loss = criterion(out, tgt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inp.size(0)

        # check for nans and infs in weights
        for name, param in model.named_parameters(recurse=True):
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f'Weight {name}:', param)
                raise ValueError("NaN or Inf found in weight tensor")

        #i += 1
        #if i == 10:
        #    break

    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, scaler, device):
    model.eval()
    val_loss = 0.0
    all_metrics = {
        'MSE': [], 'RMSE': [], 'MAE': [],
        'NRMSE': [], 'PSNR (dB)': [],
        'SSIM': [], 'NCC': []
    }
    with torch.no_grad():
        for inp, tgt in tqdm(loader, desc='Val  ', leave=False):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                out = model(inp)
                loss = criterion(out, tgt)
            val_loss += loss.item() * inp.size(0)

            # Compute metrics
            metrics = compute_metrics(tgt, out)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    output_metrics = {}
    for k, v in all_metrics.items():
        if len(v) > 0:
            output_metrics[k] = np.mean(v)
            output_metrics[k+'_std'] = np.std(v)
        else:
            output_metrics[k] = float('nan')
            output_metrics[k+'_std'] = float('nan')

    return val_loss / len(loader.dataset), output_metrics

def main(args):
    wandb.init(project='ct-proj-infilling', name='train')

    # detect device and number of GPUs
    if torch.cuda.is_available():
        ngpus = torch.cuda.device_count()
        print(f"CUDA is available. {ngpus} GPU(s) detected.")
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                print(f"GPU:{i} can access GPU {j}: {torch.cuda.can_device_access_peer(i, j)}")
                if torch.cuda.can_device_access_peer(i, j) is False:
                    ngpus = 1


    else:
        ngpus = 0
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')

    train_test_csv = pd.read_csv(args.train_test_csv)

    # datasets and loaders
    train_base = CTArtifactDataset(train_test_csv[['train_x', 'train_y']].rename(columns={'train_x': 'x', 'train_y': 'y'}).dropna())
    val_base   = CTArtifactDataset(train_test_csv[['test_x', 'test_y']].rename(columns={'test_x': 'x', 'test_y': 'y'}).dropna())

    train_loader = DataLoader(
        train_base, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=False, prefetch_factor=1, persistent_workers=True
    )
    val_loader = DataLoader(
        val_base, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=False, prefetch_factor=1, persistent_workers=True
    )

    # model
    if ngpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)

        model = UNet(in_ch=2, out_ch=1)

        model = model.cuda(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = UNet(in_ch=2, out_ch=1).to(device)

    # load model
    if args.load_model_path:
        print(f"Loading model from {args.load_model_path}")
        if os.path.exists(args.load_model_path):
            state = torch.load(args.load_model_path, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"Model file {args.load_model_path} does not exist. Starting from scratch.")

    torch.backends.cudnn.benchmark = True
    #model = torch.compile(model)

    model_size = get_model_size(model)
    print(f"Model size: {model_size:.2f} MB")


    # criterion, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    scaler = torch.amp.GradScaler()

    # initialize loss & metric histories
    train_losses = []
    val_losses   = []
    mse_vals, mse_stds       = [], []
    psnr_vals, psnr_stds     = [], []
    ssim_vals, ssim_stds     = [], []
    ncc_vals, ncc_stds       = [], []

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        torch.cuda.empty_cache()
        val_loss, all_metrics = validate(model, val_loader, criterion, scaler, device)
        scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in all_metrics.items()}
        })
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # record metrics from output_metrics
        mse_vals.append(all_metrics['MSE'])
        mse_stds.append(all_metrics['MSE_std'])
        psnr_vals.append(all_metrics['PSNR (dB)'])
        psnr_stds.append(all_metrics['PSNR (dB)_std'])
        ssim_vals.append(all_metrics['SSIM'])
        ssim_stds.append(all_metrics['SSIM_std'])
        ncc_vals.append(all_metrics['NCC'])
        ncc_stds.append(all_metrics['NCC_std'])

        # make sure the directory exists
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

        best_val = val_loss
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        file_name = args.checkpoint + f'model_epoch_{epoch}_of_{args.epochs+1}.pth'
        torch.save(state, file_name)
        print(f"Saved model_epoch_{epoch}_of_{args.epochs+1}.pth")

    # plot losses after training
    epochs = range(1, args.epochs+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_yscale('linear')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Epoch (Linear Scale)')
    ax1.legend()

    ax2.plot(epochs, train_losses, label='Train Loss')
    ax2.plot(epochs, val_losses,   label='Val Loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss per Epoch (Log Scale)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.checkpoint+f'loss_evolution_of_{args.epochs+1}_epochs.png', dpi=300)
    plt.close(fig)

    # plot four key metrics with error bars
    epochs = range(1, args.epochs+1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # MSE
    axes[0,0].errorbar(epochs, mse_vals, yerr=mse_stds, fmt='-o')
    axes[0,0].set_title('MSE per Epoch')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('MSE')
    # PSNR
    axes[0,1].errorbar(epochs, psnr_vals, yerr=psnr_stds, fmt='-o')
    axes[0,1].set_title('PSNR per Epoch')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('PSNR (dB)')
    # SSIM
    axes[1,0].errorbar(epochs, ssim_vals, yerr=ssim_stds, fmt='-o')
    axes[1,0].set_title('SSIM per Epoch')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('SSIM')
    # NCC
    axes[1,1].errorbar(epochs, ncc_vals, yerr=ncc_stds, fmt='-o')
    axes[1,1].set_title('NCC per Epoch')
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('NCC')

    plt.tight_layout()
    plt.savefig(args.checkpoint+f'metric_evolution_of_{args.epochs+1}_epochs.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CT Artifact Reduction Training'
    )
    parser.add_argument('--train_test_csv', type=str, default='data/scans/training_testing_split.csv',
                        help='CSV file for training and testing data (train_x, train_y, test_x, test_y)')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default='data/models/')
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to the model to load for resuming training')
    args = parser.parse_args()

    main(args)
