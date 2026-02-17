#!/usr/bin/env python
"""U-Net inference for projection infilling.

Two modes:
  interleave (default) — predict between every consecutive pair of projections,
      creating genuinely new intermediate views (no ground truth exists).
      Output: N originals + (N-1) predictions.

  subsample — treat odd-indexed projections as "missing" and predict them
      from their even-indexed neighbors.
      Output: N/2 originals + N/2 predictions.

Outputs:
    <outdir>/<scan>_with_pred/  — originals + predicted projections
    <outdir>/<scan>_no_pred/    — originals only (for baseline reconstruction)
"""
import os
import re
import shutil
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import xmltodict
from reconstruction.ct_core import vff_io
from unet_pipeline.model import UNet
from reconstruction.ct_core.vff_io import write_vff


def natural_sort_key(path):
    """Sort key for natural ordering of filenames."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.name)]


def parse_args():
    p = argparse.ArgumentParser(description='U-Net projection infilling')
    p.add_argument('--scan_folder', type=str, required=True,
                   help='Path to scan folder containing VFF projection files')
    p.add_argument('--sub_scan', type=str, default='-00-',
                   help='Sub-scan filter for acquisition files (default: -00-)')
    p.add_argument('--mode', type=str, default='interleave',
                   choices=['interleave', 'subsample'],
                   help='interleave (default): predict between every consecutive pair. '
                        'subsample: predict odd-indexed projections from even neighbors.')
    p.add_argument('--checkpoint', type=str,
                   default='data/models/mupiu-net_final_model.pth',
                   help='Path to model checkpoint')
    p.add_argument('--outdir', type=str, default='data/results',
                   help='Where to write the output folders')
    p.add_argument('--device', type=str,
                   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def write_updated_xml(original_xml, dst_path, scan_id, view_count, increment_angle):
    """Write a modified scan.xml with updated metadata fields."""
    doc = xmltodict.parse(original_xml)
    doc['Series']['@scanID'] = scan_id
    doc['Series']['Created'] = datetime.now().isoformat()
    doc['Series']['SeriesParams']['ViewCount'] = str(view_count)
    doc['Series']['SeriesParams']['IncrementAngle'] = str(increment_angle)
    with open(dst_path, 'w') as f:
        f.write(xmltodict.unparse(doc, pretty=True))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Enumerate projections from scan folder ---
    scan_path = Path(args.scan_folder)
    calibration_files = {'dark.vff', 'bright.vff'}

    # Auto-detect input format: proj-*.vff (from previous run) or acq* with phase filter
    proj_files = sorted(scan_path.glob('proj-*.vff'), key=natural_sort_key)
    if proj_files:
        vff_files = proj_files
    else:
        vff_files = sorted(
            [f for f in scan_path.glob('*.vff')
             if f.name.lower() not in calibration_files and args.sub_scan in f.name],
            key=natural_sort_key
        )

    if len(vff_files) == 0:
        raise ValueError(f"No VFF projection files found in {scan_path}")

    N = len(vff_files)
    print(f"Found {N} projections in {scan_path}")
    print(f"Mode: {args.mode}")

    # Derive scan basename and create output folders
    scan_basename = scan_path.name
    with_pred_folder = os.path.join(args.outdir, scan_basename + '_with_pred')
    no_pred_folder = os.path.join(args.outdir, scan_basename + '_no_pred')
    os.makedirs(with_pred_folder, exist_ok=True)
    os.makedirs(no_pred_folder, exist_ok=True)

    src_xml = scan_path / 'scan.xml'

    # Build list of (left_idx, right_idx) pairs depending on mode
    if args.mode == 'interleave':
        # Predict between every consecutive pair: (0,1), (1,2), ..., (N-2,N-1)
        pairs = [(i, i + 1) for i in range(N - 1)]
        n_out_with = 2 * N - 1
        n_out_no = N
        print(f"Predicting {len(pairs)} intermediate projections between consecutive pairs")
        print(f"Output: {n_out_with} in _with_pred, {n_out_no} in _no_pred")
    else:  # subsample
        # Odd indices are "missing" — predict from even neighbors
        pairs = [(m - 1, m + 1) for m in range(1, N, 2) if m + 1 < N]
        n_out_with = N
        n_out_no = (N + 1) // 2
        print(f"Predicting {len(pairs)} missing projections (odd indices from even neighbors)")
        print(f"Output: {n_out_with} in _with_pred, {n_out_no} in _no_pred")

    # Write updated scan.xml to both output folders
    if src_xml.exists():
        original_xml = src_xml.read_text()
        doc = xmltodict.parse(original_xml)
        orig_increment = float(doc['Series']['SeriesParams']['IncrementAngle'])

        # Preserve total_angle = increment * count (as the reconstruction expects)
        orig_total = orig_increment * N
        if args.mode == 'interleave':
            increment_with = orig_total / n_out_with  # (θ·N) / (2N−1)
            increment_no = orig_increment              # unchanged
        else:  # subsample
            increment_with = orig_increment             # unchanged
            increment_no = orig_total / n_out_no        # (θ·N) / ⌈N/2⌉

        write_updated_xml(original_xml,
                          os.path.join(with_pred_folder, 'scan.xml'),
                          scan_id=scan_basename + '_with_pred',
                          view_count=n_out_with,
                          increment_angle=increment_with)
        write_updated_xml(original_xml,
                          os.path.join(no_pred_folder, 'scan.xml'),
                          scan_id=scan_basename + '_no_pred',
                          view_count=n_out_no,
                          increment_angle=increment_no)

    # Load model
    device = torch.device(args.device)
    model = UNet(in_ch=2, out_ch=1).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # --- Optimization: read cache to avoid redundant VFF reads ---
    read_cache = {}

    def cached_read_vff(path):
        """Read VFF with caching. Returns (header, native_endian_2d_array)."""
        path_str = str(path)
        if path_str in read_cache:
            return read_cache[path_str]
        h, a = vff_io.read_vff(path_str, verbose=False)
        a = a.squeeze(0).byteswap().view(a.dtype.newbyteorder())
        read_cache[path_str] = (h, a)
        return h, a

    # --- Deduplicate original writes (by sequential index) ---
    written_with = set()
    written_no = set()

    # --- Optimization: background writer thread for I/O overlap ---
    executor = ThreadPoolExecutor(max_workers=2)
    futures = []

    def submit_write(fn, *a, **kw):
        futures.append(executor.submit(fn, *a, **kw))

    def copy_original(src_path, seq_with, seq_no):
        """Copy an original to both output folders with sequential naming."""
        if seq_with not in written_with:
            written_with.add(seq_with)
            dst = os.path.join(with_pred_folder, f"proj-{seq_with:06d}.vff")
            submit_write(shutil.copy2, str(src_path), dst)
        if seq_no is not None and seq_no not in written_no:
            written_no.add(seq_no)
            dst = os.path.join(no_pred_folder, f"proj-{seq_no:06d}.vff")
            submit_write(shutil.copy2, str(src_path), dst)

    for i, (left_idx, right_idx) in enumerate(pairs):
        left_path = vff_files[left_idx]
        right_path = vff_files[right_idx]

        # --- Optimization: cached reads ---
        h1, a1 = cached_read_vff(left_path)
        h3, a3 = cached_read_vff(right_path)

        # Evict left_path from cache — it won't be the left input of any future pair
        read_cache.pop(str(left_path), None)

        # Compute middle angle
        proj_angle_1 = float(h1['gantryPosition'])
        proj_angle_3 = float(h3['gantryPosition'])
        proj_angle_2 = (proj_angle_1 + proj_angle_3) / 2

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i + 1}/{len(pairs)}] "
                  f"angles: {proj_angle_1:.2f} -> {proj_angle_2:.2f} -> {proj_angle_3:.2f}")

        h2 = h1.copy()
        h2['gantryPosition'] = proj_angle_2

        # Convert same as training
        inp = np.stack([a1, a3], axis=0)
        t = torch.from_numpy(inp).unsqueeze(0).float().to(device, non_blocking=True)

        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                pred = model(t).squeeze(0).cpu().numpy()

        pred *= (a1.max() / 2 + a3.max() / 2) / (pred.max())
        pred[pred < 0] = 0

        # Compute sequential output indices
        if args.mode == 'interleave':
            left_seq_with = 2 * left_idx
            right_seq_with = 2 * right_idx
            pred_seq = 2 * left_idx + 1
            left_seq_no = left_idx
            right_seq_no = right_idx
        else:  # subsample
            left_seq_with = left_idx
            right_seq_with = right_idx
            pred_seq = left_idx + 1  # the missing odd position
            left_seq_no = left_idx // 2
            right_seq_no = right_idx // 2

        # Write prediction with sequential name
        pred_path = os.path.join(with_pred_folder, f"proj-{pred_seq:06d}.vff")
        pred_copy = pred.copy()  # snapshot before next iteration mutates pred
        submit_write(write_vff, pred_path, h2, pred_copy, False)

        # Copy originals (deduplicated) to both folders
        copy_original(left_path, left_seq_with, left_seq_no)
        copy_original(right_path, right_seq_with, right_seq_no)

    # In subsample mode with even N, the last odd projection has no right neighbor
    # and can't be predicted. Copy the original so _with_pred has all N projections.
    if args.mode == 'subsample' and N % 2 == 0:
        last_idx = N - 1  # odd index with no right neighbor
        if last_idx not in written_with:
            copy_original(vff_files[last_idx], last_idx, None)

    # Wait for all background writes to complete and propagate any exceptions
    executor.shutdown(wait=True)
    for f in futures:
        f.result()  # raises if the write failed

    print(f"Done. Wrote {len(futures)} files total.")
    print(f"  with_pred: {with_pred_folder}")
    print(f"  no_pred:   {no_pred_folder}")


if __name__ == '__main__':
    main()
