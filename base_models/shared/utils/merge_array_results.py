#!/usr/bin/env python3
"""
Merge results from SLURM array job splits
"""
import shutil
from pathlib import Path
import argparse
import json

def merge_splits(
    split_base='repaint/tiles_infilled_split',
    output_dir='repaint/tiles_infilled',
    num_splits=8,
    verify=True
):
    """
    Merge infilled tiles from all splits

    Args:
        split_base: Base directory containing split results
        output_dir: Output directory for merged results
        num_splits: Expected number of splits
        verify: Verify all splits are present before merging
    """

    print("="*70)
    print("Merging SLURM Array Job Results")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for split metadata
    metadata_file = Path(split_base).parent / 'sinogram_tiles_split' / 'split_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Found metadata:")
        print(f"  Total tiles expected: {metadata['total_tiles']:,}")
        print(f"  Splits: {metadata['num_splits']}")
    else:
        print("⚠ Warning: split_metadata.json not found")
        metadata = None

    print()

    # Check which splits exist
    missing_splits = []
    existing_splits = []

    for split_id in range(num_splits):
        split_dir = Path(split_base) / f'split_{split_id:02d}'
        if split_dir.exists():
            existing_splits.append(split_id)
        else:
            missing_splits.append(split_id)

    print(f"Split Status:")
    print(f"  Found: {len(existing_splits)} / {num_splits}")

    if missing_splits:
        print(f"  Missing splits: {missing_splits}")
        if verify:
            response = input("\nContinue with partial results? (y/n): ")
            if response.lower() != 'y':
                print("Merge cancelled.")
                return

    print()
    print("Copying files...")

    total_files = 0
    split_file_counts = {}

    for split_id in existing_splits:
        split_dir = Path(split_base) / f'split_{split_id:02d}'

        # Find all PNG files in split directory (16-bit PNG)
        png_files = list(split_dir.glob('*.png'))

        if len(png_files) == 0:
            print(f"⚠ Warning: Split {split_id:02d} has no PNG files")
            continue

        # Copy files
        for png_file in png_files:
            dest = output_path / png_file.name

            if dest.exists():
                print(f"⚠ Warning: {png_file.name} already exists, skipping...")
                continue

            shutil.copy2(png_file, dest)

        split_file_counts[split_id] = len(png_files)
        total_files += len(png_files)

        print(f"  ✓ Split {split_id:02d}: Copied {len(png_files):,} files")

    print()
    print("="*70)
    print("Merge Summary:")
    print("="*70)

    for split_id, count in sorted(split_file_counts.items()):
        print(f"  Split {split_id:02d}: {count:,} files")

    print()
    print(f"  Total files merged: {total_files:,}")

    if metadata:
        expected = metadata['total_tiles']
        print(f"  Expected files: {expected:,}")
        if total_files == expected:
            print(f"  ✓ All files accounted for!")
        else:
            print(f"  ⚠ Missing {expected - total_files:,} files ({(expected - total_files) / expected * 100:.1f}%)")

    print()
    print(f"✓ Merge complete! Output: {output_path}")
    print("="*70)

    # Save merge metadata
    merge_metadata = {
        'total_files': total_files,
        'num_splits_processed': len(existing_splits),
        'num_splits_expected': num_splits,
        'split_file_counts': split_file_counts,
        'missing_splits': missing_splits
    }

    with open(output_path / 'merge_metadata.json', 'w') as f:
        json.dump(merge_metadata, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge results from SLURM array job splits'
    )
    parser.add_argument(
        '--input', type=str, default='repaint/tiles_infilled_split',
        help='Base directory containing split results'
    )
    parser.add_argument(
        '--output', type=str, default='repaint/tiles_infilled',
        help='Output directory for merged results'
    )
    parser.add_argument(
        '--num-splits', type=int, default=8,
        help='Expected number of splits (default: 8)'
    )
    parser.add_argument(
        '--no-verify', action='store_true',
        help='Skip verification prompts'
    )

    args = parser.parse_args()

    merge_splits(
        split_base=args.input,
        output_dir=args.output,
        num_splits=args.num_splits,
        verify=not args.no_verify
    )
