#!/bin/bash
#SBATCH --time=02:00:00                 # 2 hours for merging and reconstruction
#SBATCH --job-name=mat_merge_recon
#SBATCH --output=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_merge_recon_%j.out
#SBATCH --error=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_merge_recon_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                    # 1 GPU for FDK reconstruction (any type)
#SBATCH --cpus-per-task=8               # Fewer CPUs (merge is I/O bound)
#SBATCH --mem=64G                       # 64GB for merge and reconstruction
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=ALL

set -eo pipefail

# Trap errors to provide more context
trap 'echo "ERROR at line $LINENO: command \"$BASH_COMMAND\" failed with exit code $?"' ERR

#==============================================================================
# MAT Merge Tiles and FDK Reconstruction
# =======================================
# This script:
# 1. Merges 73,472 infilled tiles (256x256) back to full-resolution sinograms
# 2. Runs FDK reconstruction to create CT volume
#
# Run this AFTER the main MAT inference job (run_mat_cedar_h100.sh) completes.
# Uses fewer resources than the full pipeline since no ML inference is needed.
#
# Expected runtime: 1-1.5 hours
# Works on: Cedar, Fir, Graham, or any Compute Canada cluster
#==============================================================================

# Detect cluster and set paths
HOSTNAME=$(hostname)
if [[ ${HOSTNAME} == *"cedar"* ]]; then
    CLUSTER="cedar"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"fir"* ]]; then
    CLUSTER="fir"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"graham"* ]]; then
    CLUSTER="graham"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"beluga"* ]]; then
    CLUSTER="beluga"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
elif [[ ${HOSTNAME} == *"narval"* ]]; then
    CLUSTER="narval"
    PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
else
    # Try to auto-detect
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    elif [ -d "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/project/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot auto-detect project path"
        exit 1
    fi
    CLUSTER=$(echo ${HOSTNAME} | cut -d'.' -f1)
fi

echo "========================================================================"
echo "MAT Tile Merge & FDK Reconstruction"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start time: $(date)"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p $PROJECT/logs

cd $SLURM_TMPDIR

#==============================================================================
# SETUP
#==============================================================================

echo ""
echo "Cloning repository..."
# Try SSH first, fall back to HTTPS if SSH fails
if ! git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git 2>/dev/null; then
    echo "SSH clone failed, trying HTTPS..."
    git clone --depth=1 https://github.com/falkwiegmann/ct_recon.git || {
        echo "ERROR: Git clone failed with both SSH and HTTPS"
        exit 1
    }
fi

cd ct_recon/Base_model_comparison

# Copy updated Python scripts from PROJECT to override GitHub versions
echo ""
echo "Updating Python scripts from PROJECT directory..."
cp "$PROJECT"/*.py . 2>/dev/null || echo "Warning: Could not copy .py files from PROJECT"
echo "Python scripts updated from PROJECT"

#==============================================================================
# MODULE LOADING
#==============================================================================

echo ""
echo "Loading modules..."

# Reset to default environment
module reset 2>/dev/null || true

# Load Python
echo "  Loading python/3.10..."
if ! module load python/3.10 2>&1; then
    if ! module load python/3.11 2>&1; then
        module load python 2>&1 || {
            echo "ERROR: No Python module available"
            exit 1
        }
    fi
fi

# Load scipy-stack for numpy, scipy, etc.
echo "  Loading scipy-stack..."
module load scipy-stack 2>&1 || echo "  scipy-stack not available, will use pip packages"

echo ""
echo "Loaded modules:"
module list 2>&1 | head -20

#==============================================================================
# VIRTUAL ENVIRONMENT SETUP
#==============================================================================

echo ""
echo "========================================================================"
echo "Setting up Python virtual environment..."
echo "========================================================================"

VENV_DIR="$SLURM_TMPDIR/venv"

echo "Creating virtual environment at: $VENV_DIR"
virtualenv --no-download "$VENV_DIR" || {
    echo "ERROR: Failed to create virtual environment"
    exit 1
}

source "$VENV_DIR/bin/activate" || {
    echo "ERROR: Failed to activate virtual environment"
    exit 1
}

echo "Virtual environment activated: $(which python)"

#==============================================================================
# INSTALL DEPENDENCIES
#==============================================================================

echo ""
echo "========================================================================"
echo "Installing Python dependencies..."
echo "========================================================================"

pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
if ! pip install --no-index torch 2>/dev/null; then
    echo "  Pre-built wheel not available, installing from PyPI..."
    pip install torch || {
        echo "ERROR: Failed to install PyTorch"
        exit 1
    }
fi

# Install other required packages
echo "Installing additional packages..."
PACKAGES="numpy pillow tqdm scipy pyyaml scikit-image"
for pkg in $PACKAGES; do
    echo "  Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "ERROR: Failed to install $pkg"
        exit 1
    }
done

# Install xmltodict for reconstruction
echo "  Installing xmltodict..."
pip install xmltodict || {
    echo "WARNING: xmltodict installation failed, reconstruction may not work"
}

echo ""
echo "All packages installed successfully!"

#==============================================================================
# VERIFY PYTORCH AND CUDA
#==============================================================================

echo ""
echo "Checking PyTorch and CUDA..."
python3 << 'CUDA_CHECK'
import torch
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
else:
    print("  WARNING: CUDA not available, reconstruction will use CPU (slower)")

print("\nPyTorch check passed!")
CUDA_CHECK

#==============================================================================
# STEP 1: VERIFY INFILLED TILES
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Verifying infilled tiles"
echo "========================================================================"

TILES_DIR="$PROJECT/mat/tiles_infilled"

if [ ! -d "${TILES_DIR}" ]; then
    echo "ERROR: Infilled tiles directory not found: ${TILES_DIR}"
    echo "Please run the MAT inference job first (run_mat_cedar_h100.sh)"
    exit 1
fi

TILE_COUNT=$(find ${TILES_DIR} -name "*.png" 2>/dev/null | wc -l)
echo "Found ${TILE_COUNT} infilled tiles in ${TILES_DIR}"

EXPECTED_TILES=73472
if [ ${TILE_COUNT} -lt ${EXPECTED_TILES} ]; then
    echo "WARNING: Expected ${EXPECTED_TILES} tiles but found ${TILE_COUNT}"
    echo "Proceeding with available tiles..."
fi

# Copy tiles to local storage for faster I/O
echo ""
echo "Copying tiles to local NVMe for faster processing..."
mkdir -p mat/tiles_infilled
rsync -av --info=progress2 ${TILES_DIR}/ mat/tiles_infilled/ 2>&1 | tail -10

# Copy metadata
echo ""
echo "Copying metadata files..."
mkdir -p repaint/sinogram_tiles
cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/ || {
    echo "ERROR: tiling_metadata.json not found"
    exit 1
}
echo "Tiling metadata copied"

# Verify dataset metadata exists (should be in git)
if [ ! -f "sinogram_dataset/metadata.json" ]; then
    echo "ERROR: sinogram_dataset/metadata.json not found in git repository!"
    exit 1
fi
echo "Dataset metadata found"

#==============================================================================
# STEP 2: MERGE TILES BACK TO FULL SINOGRAMS
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Merging tiles back to full-resolution sinograms"
echo "========================================================================"
echo "This will:"
echo "  - Merge ${TILE_COUNT} tiles (256x256) -> 2,296 sinograms (410x3500)"
echo "  - Apply nearest-neighbor blending in overlap regions"
echo "  - Keep tiles in normalized uint16 format [0, 65535]"
echo ""

START_TIME=$(date +%s)

python3 merge_mat_tiles.py \
    --tiles_dir mat/tiles_infilled \
    --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --output_dir mat/sinograms_infilled \
    --blend_mode nearest

END_TIME=$(date +%s)
MERGE_DURATION=$((END_TIME - START_TIME))

# Verify sinograms
NUM_SINOS=$(find mat/sinograms_infilled -name "*.png" 2>/dev/null | wc -l)
echo ""
echo "Tile merging complete in $((MERGE_DURATION / 60)) minutes $((MERGE_DURATION % 60)) seconds"
echo "Total sinograms: ${NUM_SINOS}"

EXPECTED_SINOS=2296
if [ ${NUM_SINOS} -ne ${EXPECTED_SINOS} ]; then
    echo "WARNING: Expected ${EXPECTED_SINOS} sinograms but got ${NUM_SINOS}"
fi

# Transfer merged sinograms to project directory (checkpoint)
echo ""
echo "Saving merged sinograms to project directory (checkpoint)..."
mkdir -p $PROJECT/mat/sinograms_infilled
rsync -av mat/sinograms_infilled/ $PROJECT/mat/sinograms_infilled/ 2>&1 | tail -5
echo "Checkpoint saved"

#==============================================================================
# STEP 3: FDK RECONSTRUCTION
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Running FDK reconstruction on infilled sinograms"
echo "========================================================================"
echo "This will reconstruct the full CT volume from ${NUM_SINOS} infilled sinograms"
echo ""

# Verify scan folder exists before reconstruction
# data/results is a SIBLING of Base_model_comparison, not inside it
# $PROJECT = .../ct_recon/Base_model_comparison
# scan.xml is at .../ct_recon/data/results/Scan_1681_uwarp_gt/scan.xml
SCAN_FOLDER="${PROJECT%/Base_model_comparison}/data/results/Scan_1681_uwarp_gt"

if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
    echo "ERROR: scan.xml not found at: ${SCAN_FOLDER}/scan.xml"
    echo "This file is required for FDK reconstruction (geometry, angles, detector specs)."
    exit 1
fi
echo "Scan folder verified: ${SCAN_FOLDER}"
echo ""

START_TIME=$(date +%s)

python3 reconstruct_from_mat.py \
    --sinogram_dir mat/sinograms_infilled \
    --output_dir mat/reconstructed_volume \
    --metadata_path sinogram_dataset/metadata.json \
    --tiling_metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --scan_folder "${SCAN_FOLDER}"

END_TIME=$(date +%s)
RECON_DURATION=$((END_TIME - START_TIME))
echo ""
echo "Reconstruction complete in $((RECON_DURATION / 60)) minutes $((RECON_DURATION % 60)) seconds"

#==============================================================================
# STEP 4: TRANSFER FINAL RESULTS TO PROJECT DIRECTORY
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 4: Transferring final results to project directory"
echo "========================================================================"

mkdir -p $PROJECT/mat

# FDKReconstructor saves the .vff file as folder_name + ".vff"
# So the file is at "mat/reconstructed_volume.vff", NOT inside a directory
echo "Transferring reconstructed volume (.vff file)..."
if [ -f "mat/reconstructed_volume.vff" ]; then
    cp -v mat/reconstructed_volume.vff $PROJECT/mat/
    echo "VFF file transferred successfully"
else
    echo "WARNING: mat/reconstructed_volume.vff not found!"
    echo "Checking for any .vff files..."
    find mat/ -name "*.vff" -ls
fi

# Also transfer any slice images if they were generated
if [ -d "mat/reconstructed_volume" ]; then
    echo "Transferring reconstruction slice images..."
    rsync -av mat/reconstructed_volume/ $PROJECT/mat/reconstructed_volume/ 2>&1 | tail -5
fi

#==============================================================================
# COMPLETION SUMMARY
#==============================================================================

# Verify .vff file exists
if [ -f "$PROJECT/mat/reconstructed_volume.vff" ]; then
    VFF_SIZE=$(du -h "$PROJECT/mat/reconstructed_volume.vff" | cut -f1)
    echo ""
    echo "Reconstruction .vff file: $PROJECT/mat/reconstructed_volume.vff ($VFF_SIZE)"
else
    echo ""
    echo "WARNING: .vff file not found at expected location!"
fi

echo ""
echo "========================================================================"
echo "PROCESSING COMPLETE!"
echo "========================================================================"
echo "Results saved to:"
echo "  Merged sinograms: $PROJECT/mat/sinograms_infilled/"
echo "  Reconstructed volume: $PROJECT/mat/reconstructed_volume.vff"
echo ""
echo "Statistics:"
echo "  Tiles merged: ${TILE_COUNT}"
echo "  Sinograms created: ${NUM_SINOS}"
echo "  Tile merge time: $((MERGE_DURATION / 60))m $((MERGE_DURATION % 60))s"
echo "  Reconstruction time: $((RECON_DURATION / 60))m $((RECON_DURATION % 60))s"
echo ""
echo "Job completed: $(date)"
echo "========================================================================"
