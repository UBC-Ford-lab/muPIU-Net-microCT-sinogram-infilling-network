#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
import sys
from ct_core import vff_io
from unet_pipeline.model import UNet
from ct_core.field_correction import write_vff

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        type=str,
                   default='data/scans/training_testing_split.csv',
                   help='CSV with test_x, test_y columns')
    p.add_argument('--checkpoint', type=str, default='data/models/best_model_after_25+30_ep.pth',
                   help='path to best_model.pth')
    p.add_argument('--outdir',     type=str, default='data/results',
                   help='where to write the triplets')
    p.add_argument('--device',     type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load CSV & isolate test split
    df = pd.read_csv(args.csv)
    test = ( df[['test_x','test_y']]
             .dropna()
             .rename(columns={'test_x':'x','test_y':'y'}) )

    # model
    device = torch.device(args.device)
    model = UNet(in_ch=2, out_ch=1).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test = test[test['x'].str.contains('1539')]
    test = test[test['x'].str.contains('-00-')]
    print(f"Found {len(test)} test samples.")

    for idx, row in test.iterrows():
        # eval strings to lists
        inp_paths = eval(row['x'])   # e.g. ['proj001.vff','proj003.vff']
        # load headers + raw arrays
        h1, a1 = vff_io.read_vff(inp_paths[0], verbose=False)
        h3, a3 = vff_io.read_vff(inp_paths[1], verbose=False)

        # get resulting projection angle
        proj_angle_1 = float(h1['gantryPosition'])
        proj_angle_3 = float(h3['gantryPosition'])
        print(f"[{idx}] angles: {proj_angle_1:.2f}° and {proj_angle_3:.2f}°")
        proj_angle_2 = (proj_angle_1 + proj_angle_3) / 2
        print(f"[{idx}] infilled angle: {proj_angle_2:.2f}°")
        h2 = h1.copy()
        h2['gantryPosition'] = proj_angle_2

        # convert same as training
        a1 = a1.squeeze(0).byteswap().view(a1.dtype.newbyteorder())
        a3 = a3.squeeze(0).byteswap().view(a3.dtype.newbyteorder())
        inp = np.stack([a1,a3], axis=0)
        t = torch.from_numpy(inp).unsqueeze(0).float().to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                pred = model(t).squeeze(0).cpu().numpy()

        pred *= (a1.max()/2+a3.max()/2) / (pred.max())  # scale to match input range
        pred[pred < 0] = 0  # ensure no negative values

        # build filenames
        scan_folder_name = os.path.join(args.outdir, os.path.basename(os.path.dirname(inp_paths[0])))
        os.makedirs(scan_folder_name, exist_ok=True)
        b1 = os.path.splitext(os.path.basename(inp_paths[0]))[0]
        b3 = os.path.splitext(os.path.basename(inp_paths[1]))[0]
        o1 = os.path.join(scan_folder_name, f"{b1}.vff")
        pred_proj_angle_count = int(b1.split('-')[2]) + 1
        o2 = os.path.join(scan_folder_name, f"{b1[:-4]}{pred_proj_angle_count:04d}_pred.vff")
        o3 = os.path.join(scan_folder_name, f"{b3}.vff")

        # write
        write_vff(o1, h1, a1, verbose=False)
        write_vff(o3, h3, a3, verbose=False)
        write_vff(o2, h2, pred, verbose=False)

    #sys.exit()

if __name__ == '__main__':
    main()
