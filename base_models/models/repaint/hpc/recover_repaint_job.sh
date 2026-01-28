#!/bin/bash
#SBATCH --time=01-00:00:00              # 1 day (conservative for infilling missing + merge + recon)
#SBATCH --job-name=repaint_recovery
#SBATCH --output=logs/repaint_recovery_%j.out
#SBATCH --error=logs/repaint_recovery_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1               # 1 H100 (cheaper than 4!)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

#==============================================================================
# RePaint Job Recovery Script
# ===========================
# This script recovers from the failed multi-GPU job where all workers
# processed all tiles instead of their assigned subsets.
#
# Steps:
# 1. ANALYZE: Scan all 4 task directories, find unique and missing tiles
# 2. CONSOLIDATE: Merge unique tiles into one directory, DELETE duplicates
#    (This REDUCES file count from ~185k to ~73k, freeing quota!)
# 3. INFILL: Process any missing tiles with single GPU
# 4. MERGE: Combine tiles back into full sinograms
# 5. RECONSTRUCT: Run FDK reconstruction
#
# Expected outcome:
# - File count reduced by ~100k (duplicates deleted)
# - Missing tiles filled in
# - Full reconstruction completed
#==============================================================================

# Detect cluster and set paths
HOSTNAME=$(hostname)
if [[ ${HOSTNAME} == *"cedar"* ]] || [[ ${HOSTNAME} == *"cdr"* ]]; then
    CLUSTER="cedar"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"fir"* ]]; then
    CLUSTER="fir"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"graham"* ]]; then
    CLUSTER="graham"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
else
    CLUSTER="unknown"
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot auto-detect project path"
        exit 1
    fi
fi

echo "========================================================================"
echo "RePaint Job Recovery"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "Start time: $(date)"
echo "========================================================================"

# Create logs directory
mkdir -p $PROJECT/logs

# Load Python environment early (needed for all phases)
module purge 2>/dev/null || true
module load python/3.10 scipy-stack 2>/dev/null || true
source ~/Python_virtual_env/bin/activate 2>/dev/null || true

#==============================================================================
# PHASE 1 & 2: ANALYSIS AND CONSOLIDATION (using Python for robustness)
#==============================================================================

echo ""
echo "========================================================================"
echo "PHASES 1 & 2: Analyzing and consolidating tiles"
echo "========================================================================"

# Run Python script for consolidation
python3 << 'CONSOLIDATE_SCRIPT'
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

project = os.environ.get('PROJECT', '/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison')
repaint_dir = Path(project) / 'repaint'

print("=" * 70)
print("PHASE 1: Analyzing existing tiles")
print("=" * 70)

# Find all task directories
task_dirs = []
for i in range(4):
    task_dir = repaint_dir / f'tiles_infilled_task_{i}'
    if task_dir.exists():
        task_dirs.append(task_dir)
        count = len(list(task_dir.glob('*.png')))
        print(f"  Task {i}: {count} tiles")

# Also check if tiles_infilled already exists
consolidated_dir = repaint_dir / 'tiles_infilled'
if consolidated_dir.exists() and consolidated_dir not in task_dirs:
    count = len(list(consolidated_dir.glob('*.png')))
    print(f"  tiles_infilled (existing): {count} tiles")

if not task_dirs and not consolidated_dir.exists():
    print("ERROR: No tile directories found!")
    sys.exit(1)

print("")
print("=" * 70)
print("PHASE 2: Consolidating tiles")
print("=" * 70)

# Build index of all tiles across all directories
# Track which directory has each tile (for moving vs deleting)
tile_locations = defaultdict(list)  # tile_name -> [(dir, full_path), ...]

for task_dir in task_dirs:
    for tile_path in task_dir.glob('*.png'):
        tile_locations[tile_path.name].append((task_dir, tile_path))

# If consolidated_dir exists, add those too
if consolidated_dir.exists():
    for tile_path in consolidated_dir.glob('*.png'):
        tile_locations[tile_path.name].append((consolidated_dir, tile_path))

total_files = sum(len(locs) for locs in tile_locations.values())
unique_tiles = len(tile_locations)
duplicates = total_files - unique_tiles

print(f"Total files across all directories: {total_files}")
print(f"Unique tiles: {unique_tiles}")
print(f"Duplicates to delete: {duplicates}")
print("")

# Create consolidated directory if needed
consolidated_dir.mkdir(parents=True, exist_ok=True)

# Process tiles: keep one copy in consolidated_dir, delete the rest
moved_count = 0
deleted_count = 0
already_in_place = 0

for idx, (tile_name, locations) in enumerate(tile_locations.items()):
    # Check if already in consolidated_dir
    in_consolidated = any(d == consolidated_dir for d, p in locations)

    if in_consolidated:
        # Keep the one in consolidated, delete all others
        already_in_place += 1
        for d, p in locations:
            if d != consolidated_dir:
                p.unlink()
                deleted_count += 1
    else:
        # Move first copy to consolidated, delete the rest
        first_dir, first_path = locations[0]
        dest_path = consolidated_dir / tile_name
        first_path.rename(dest_path)
        moved_count += 1

        # Delete remaining copies
        for d, p in locations[1:]:
            if p.exists():  # Check exists since we might have just moved it
                p.unlink()
                deleted_count += 1

    # Progress update
    if (idx + 1) % 10000 == 0:
        print(f"  Progress: {idx + 1}/{unique_tiles} tiles processed...")

print("")
print(f"Consolidation complete:")
print(f"  Already in place: {already_in_place}")
print(f"  Moved: {moved_count}")
print(f"  Duplicates deleted: {deleted_count}")
print(f"  File quota freed: ~{deleted_count} slots")

# Remove empty task directories
for task_dir in task_dirs:
    if task_dir.exists() and task_dir != consolidated_dir:
        remaining = list(task_dir.glob('*.png'))
        if not remaining:
            task_dir.rmdir()
            print(f"  Removed empty directory: {task_dir.name}")

# Verify final count
final_count = len(list(consolidated_dir.glob('*.png')))
print(f"\nFinal consolidated tile count: {final_count}")

print("")
print("=" * 70)
print("PHASE 3: Identifying missing tiles")
print("=" * 70)

# Load metadata
metadata_path = repaint_dir / 'sinogram_tiles' / 'tiling_metadata.json'
if not metadata_path.exists():
    print(f"ERROR: Tiling metadata not found at: {metadata_path}")
    sys.exit(1)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

n_sinograms = metadata['n_sinograms']
total_expected = metadata['total_tiles']

print(f"Expected tiles: {total_expected}")
print(f"Sinograms: {n_sinograms}")

# Build expected tile list
expected_tiles = set()
for sino_idx in range(n_sinograms):
    sino_meta = metadata['sinograms'][str(sino_idx)]
    n_tiles = sino_meta['tile_info']['n_tiles_total']
    for tile_idx in range(n_tiles):
        tile_name = f"sino_{sino_idx:04d}_tile_{tile_idx:02d}.png"
        expected_tiles.add(tile_name)

print(f"Built expected tile list: {len(expected_tiles)} tiles")

# Get existing tiles
existing_tiles = set(f.name for f in consolidated_dir.glob('*.png'))
print(f"Found existing tiles: {len(existing_tiles)}")

# Find missing tiles
missing_tiles = expected_tiles - existing_tiles
unexpected_tiles = existing_tiles - expected_tiles

print(f"Missing tiles: {len(missing_tiles)}")
if unexpected_tiles:
    print(f"Unexpected tiles (will be ignored): {len(unexpected_tiles)}")

# Write missing tiles to file
missing_file = repaint_dir / 'missing_tiles.txt'
with open(missing_file, 'w') as f:
    for tile in sorted(missing_tiles):
        f.write(f"{tile}\n")

if missing_tiles:
    print(f"Missing tiles list written to: {missing_file}")
else:
    print("No missing tiles - all tiles present!")

# Write summary for shell script
summary_file = repaint_dir / 'recovery_summary.txt'
with open(summary_file, 'w') as f:
    f.write(f"UNIQUE_TILES={unique_tiles}\n")
    f.write(f"DELETED_COUNT={deleted_count}\n")
    f.write(f"MISSING_COUNT={len(missing_tiles)}\n")
    f.write(f"FINAL_COUNT={final_count}\n")
    f.write(f"EXPECTED_TILES={total_expected}\n")

print("")
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Expected tiles: {total_expected}")
print(f"Existing tiles: {final_count}")
print(f"Missing tiles:  {len(missing_tiles)}")
print(f"Coverage:       {100 * final_count / total_expected:.2f}%")
print("=" * 70)
CONSOLIDATE_SCRIPT

# Source the summary
if [ -f "$PROJECT/repaint/recovery_summary.txt" ]; then
    source "$PROJECT/repaint/recovery_summary.txt"
else
    echo "ERROR: Recovery summary not generated!"
    exit 1
fi

echo ""
echo "Summary from Python:"
echo "  Unique tiles: ${UNIQUE_TILES}"
echo "  Deleted duplicates: ${DELETED_COUNT}"
echo "  Missing tiles: ${MISSING_COUNT}"
echo "  Final count: ${FINAL_COUNT}"
echo "  Expected: ${EXPECTED_TILES}"

#==============================================================================
# PHASE 4: INFILL MISSING TILES (if any)
#==============================================================================

if [ ${MISSING_COUNT} -gt 0 ]; then
    echo ""
    echo "========================================================================"
    echo "PHASE 4: Infilling ${MISSING_COUNT} missing tiles"
    echo "========================================================================"

    # Change to SLURM_TMPDIR for fast local storage
    WORK_DIR="${SLURM_TMPDIR:-/tmp/repaint_recovery_$$}"
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    echo "Working directory: $(pwd)"

    echo ""
    echo "Cloning repository..."
    if ! git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git 2>/dev/null; then
        echo "SSH clone failed, trying HTTPS..."
        git clone --depth=1 https://github.com/falkwiegmann/ct_recon.git
    fi

    cd ct_recon/Base_model_comparison

    echo ""
    echo "Loading modules..."
    module purge
    module load cuda/12.2 cudnn/8.9 python/3.10 scipy-stack
    source ~/Python_virtual_env/bin/activate

    echo ""
    echo "Setting up RePaint environment..."

    # Create directories
    mkdir -p repaint/sinogram_tiles/{sinograms_masked,masks}
    mkdir -p repaint/RePaint/data/pretrained
    mkdir -p repaint/RePaint/confs

    # Copy required data
    echo "Copying tile data..."
    rsync -a $PROJECT/repaint/sinogram_tiles/sinograms_masked/ repaint/sinogram_tiles/sinograms_masked/
    rsync -a $PROJECT/repaint/sinogram_tiles/masks/ repaint/sinogram_tiles/masks/
    cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/

    # Copy RePaint code
    rsync -a --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
        --exclude='data/pretrained/*' --exclude='test.py' \
        $PROJECT/repaint/RePaint/ repaint/RePaint/

    # Copy PATCHED image_datasets.py with missing_tiles_file support
    # This is critical for efficient recovery - only processes tiles in the missing list
    cp repaint_patches/image_datasets.py repaint/RePaint/guided_diffusion/image_datasets.py
    echo "Installed patched image_datasets.py with missing_tiles_file support"

    # Copy inference script
    cp run_repaint_inference.py repaint/RePaint/test.py

    # Copy model weights
    rsync -a $PROJECT/repaint/RePaint/data/pretrained/ repaint/RePaint/data/pretrained/

    # Copy config
    cp $PROJECT/configs/repaint_sinogram_h100.yml repaint/RePaint/confs/repaint_sinogram.yml

    # Create output directory for missing tiles
    mkdir -p repaint/tiles_infilled_missing

    # Copy the missing tiles list
    cp "$PROJECT/repaint/missing_tiles.txt" repaint/missing_tiles.txt

    echo ""
    echo "Creating config for missing tiles only..."
    cd repaint/RePaint

    # Create a config that processes only missing tiles
    python3 << 'CONFIG_SCRIPT'
import yaml
import os

# Load base config
with open('confs/repaint_sinogram.yml', 'r') as f:
    config = yaml.safe_load(f)

# Update paths for recovery job
config['data']['eval']['sinogram_tiles']['gt_path'] = '../../repaint/sinogram_tiles/sinograms_masked'
config['data']['eval']['sinogram_tiles']['mask_path'] = '../../repaint/sinogram_tiles/masks'
config['data']['eval']['sinogram_tiles']['paths']['srs'] = '../../repaint/tiles_infilled_missing'

# Add missing tiles filter - this is the key for efficient recovery!
config['data']['eval']['sinogram_tiles']['missing_tiles_file'] = '../../repaint/missing_tiles.txt'

# Remove any start_idx/end_idx that might be set (we want all tiles from the filtered list)
config['data']['eval']['sinogram_tiles'].pop('start_idx', None)
config['data']['eval']['sinogram_tiles'].pop('end_idx', None)

# Save config
with open('confs/repaint_missing.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Created config for missing tiles: confs/repaint_missing.yml")
print(f"  - Only tiles in missing_tiles.txt will be processed")
print(f"  - Output to: ../../repaint/tiles_infilled_missing/")
CONFIG_SCRIPT

    echo ""
    echo "Running RePaint on ${MISSING_COUNT} missing tiles..."
    echo "This may take a while depending on the number of missing tiles..."

    # Create logs directory (fix for tee failure)
    mkdir -p ../../logs

    CUDA_VISIBLE_DEVICES=0 python test.py \
        --conf_path confs/repaint_missing.yml \
        2>&1 | tee ../../logs/repaint_missing.log

    cd ../..

    # Transfer newly infilled tiles to project directory
    echo ""
    echo "Transferring newly infilled tiles..."
    NEW_TILES=$(find repaint/tiles_infilled_missing -name "*.png" 2>/dev/null | wc -l)
    echo "New tiles generated: ${NEW_TILES}"

    if [ ${NEW_TILES} -gt 0 ]; then
        rsync -a repaint/tiles_infilled_missing/ "$PROJECT/repaint/tiles_infilled/"
        echo "Transferred ${NEW_TILES} new tiles to consolidated directory"
    fi

    # Return to project directory (SLURM_TMPDIR is cleaned up automatically by SLURM)
    cd $PROJECT

else
    echo ""
    echo "========================================================================"
    echo "PHASE 4: SKIPPED - No missing tiles!"
    echo "========================================================================"
fi

#==============================================================================
# PHASE 5: MERGE AND RECONSTRUCT
#==============================================================================

echo ""
echo "========================================================================"
echo "PHASE 5: Merging tiles and reconstructing"
echo "========================================================================"

# Verify all tiles present
CONSOLIDATED_DIR="$PROJECT/repaint/tiles_infilled"
FINAL_TILE_COUNT=$(find "$CONSOLIDATED_DIR" -name "*.png" 2>/dev/null | wc -l)
echo "Final tile count: ${FINAL_TILE_COUNT}"

if [ ${FINAL_TILE_COUNT} -lt ${EXPECTED_TILES} ]; then
    echo "WARNING: Still missing $((EXPECTED_TILES - FINAL_TILE_COUNT)) tiles!"
    echo "Proceeding with available tiles..."
fi

# Change to temp directory for processing
WORK_DIR="${SLURM_TMPDIR:-/tmp/repaint_merge_$$}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository if not already done
if [ ! -d "ct_recon" ]; then
    echo ""
    echo "Cloning repository..."
    if ! git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git 2>/dev/null; then
        git clone --depth=1 https://github.com/falkwiegmann/ct_recon.git
    fi
fi

cd ct_recon/Base_model_comparison

# Load modules if not already loaded
module purge 2>/dev/null || true
module load python/3.10 scipy-stack 2>/dev/null || true
source ~/Python_virtual_env/bin/activate 2>/dev/null || true

# Copy required files for merging
echo ""
echo "Copying tiles for merging..."
mkdir -p repaint/tiles_infilled
mkdir -p repaint/sinogram_tiles
rsync -a "$CONSOLIDATED_DIR/" repaint/tiles_infilled/
cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/

echo ""
echo "Running tile merger..."
python merge_repaint_tiles.py \
    --tiles_dir repaint/tiles_infilled \
    --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --output_dir repaint/sinograms_infilled \
    --blend_mode nearest

# Transfer merged sinograms
echo ""
echo "Transferring merged sinograms..."
mkdir -p $PROJECT/repaint/sinograms_infilled
rsync -a repaint/sinograms_infilled/ $PROJECT/repaint/sinograms_infilled/

# Run reconstruction
echo ""
echo "Running FDK reconstruction..."
# data/results is a SIBLING of Base_model_comparison, not inside it
SCAN_FOLDER="${PROJECT%/Base_model_comparison}/data/results/Scan_1681_uwarp_gt"

if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
    echo "ERROR: scan.xml not found at ${SCAN_FOLDER}"
    echo "Skipping reconstruction - please run manually with correct scan folder"
else
    python reconstruct_from_repaint.py \
        --sinogram_dir repaint/sinograms_infilled \
        --output_dir repaint/reconstructed_volume \
        --metadata_path sinogram_dataset/metadata.json \
        --tiling_metadata_path repaint/sinogram_tiles/tiling_metadata.json \
        --scan_folder $SCAN_FOLDER

    # Transfer reconstruction results
    echo ""
    echo "Transferring reconstruction results..."
    mkdir -p $PROJECT/repaint/reconstructed_volume
    rsync -a repaint/reconstructed_volume/ $PROJECT/repaint/reconstructed_volume/

    # Also copy the .vff file (saved at same level as directory)
    if [ -f "repaint/reconstructed_volume.vff" ]; then
        cp repaint/reconstructed_volume.vff $PROJECT/repaint/reconstructed_volume.vff
        echo "Copied reconstructed_volume.vff to project directory"
    fi
fi

# Return to project directory (SLURM_TMPDIR is cleaned up automatically by SLURM)
cd $PROJECT

#==============================================================================
# COMPLETION
#==============================================================================

echo ""
echo "========================================================================"
echo "RECOVERY COMPLETE!"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  Tiles consolidated: ${FINAL_TILE_COUNT:-$FINAL_COUNT}"
echo "  Duplicates deleted: ${DELETED_COUNT}"
echo "  File quota freed:   ~${DELETED_COUNT} slots"
if [ ${MISSING_COUNT:-0} -gt 0 ]; then
    echo "  Missing tiles filled: ${MISSING_COUNT}"
fi
echo ""
echo "Output locations:"
echo "  Consolidated tiles:   $PROJECT/repaint/tiles_infilled/"
echo "  Merged sinograms:     $PROJECT/repaint/sinograms_infilled/"
echo "  Reconstructed volume: $PROJECT/repaint/reconstructed_volume/"
echo "========================================================================"
