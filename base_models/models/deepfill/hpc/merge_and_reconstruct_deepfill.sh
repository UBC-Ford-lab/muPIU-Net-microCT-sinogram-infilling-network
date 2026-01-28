#!/bin/bash
#SBATCH --time=03:00:00               # 3 hours (merge ~10min + reconstruction ~1-2hr)
#SBATCH --job-name=deepfill_merge_recon
#SBATCH --output=logs/deepfill_merge_recon_%j.out
#SBATCH --error=logs/deepfill_merge_recon_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                  # 1 GPU is sufficient for reconstruction
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # 8 CPUs for data loading
#SBATCH --mem=64G                     # 64GB RAM for memmap operations
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Error trap to help diagnose failures
trap 'echo ""; echo "ERROR: Script failed at line $LINENO with exit code $?"; echo "Last command: $BASH_COMMAND"; echo "Check the .err file for additional details."' ERR

#==============================================================================
# DeepFill v2 Merge + Reconstruction Pipeline
#
# This script:
# 1. Takes already-infilled tiles from a previous DeepFill run
# 2. Merges them using "nearest" blending (no averaging)
# 3. Runs FDK reconstruction to produce final CT volume
#
# Prerequisites:
# - Infilled tiles exist in: $PROJECT/deepfill/tiles_infilled/
# - Tiling metadata exists in: $PROJECT/repaint/sinogram_tiles/tiling_metadata.json
# - Dataset metadata exists in: $PROJECT/sinogram_dataset/metadata.json
#==============================================================================

echo "========================================================================"
echo "DeepFill v2 Merge + Reconstruction Pipeline"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================================================"

#==============================================================================
# CLUSTER DETECTION AND PATH SETUP
#==============================================================================

HOSTNAME=$(hostname)
if [[ ${HOSTNAME} == *"cedar"* ]]; then
    CLUSTER="cedar"
elif [[ ${HOSTNAME} == *"fir"* ]]; then
    CLUSTER="fir"
elif [[ ${HOSTNAME} == *"graham"* ]]; then
    CLUSTER="graham"
elif [[ ${HOSTNAME} == *"narval"* ]]; then
    CLUSTER="narval"
elif [[ ${HOSTNAME} == *"beluga"* ]]; then
    CLUSTER="beluga"
else
    CLUSTER="unknown"
fi

echo "Detected cluster: ${CLUSTER}"

# Try multiple project path patterns
PROJECT=""
for path in \
    "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" \
    "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" \
    "/scratch/wiegmann/ct_recon/Base_model_comparison"; do
    if [ -d "$path" ]; then
        PROJECT="$path"
        break
    fi
done

if [ -z "$PROJECT" ]; then
    echo "ERROR: Cannot find project directory"
    echo "Tried: /home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison"
    echo "       /project/def-nlford/wiegmann/ct_recon/Base_model_comparison"
    echo "       /scratch/wiegmann/ct_recon/Base_model_comparison"
    exit 1
fi

echo "Project directory: ${PROJECT}"

# Define scan folder for geometry (go up one level from Base_model_comparison)
CT_RECON_ROOT="${PROJECT%/Base_model_comparison}"
SCAN_FOLDER="${CT_RECON_ROOT}/data/results/Scan_1681_uwarp_gt"

# Verify the scan folder exists
if [ ! -d "$SCAN_FOLDER" ]; then
    echo "WARNING: Scan folder not found at: $SCAN_FOLDER"
    echo "Trying alternative paths..."

    for alt_path in \
        "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/data/results/Scan_1681_uwarp_gt" \
        "/project/def-nlford/wiegmann/ct_recon/data/results/Scan_1681_uwarp_gt"; do
        if [ -d "$alt_path" ]; then
            SCAN_FOLDER="$alt_path"
            echo "Found scan folder at: $SCAN_FOLDER"
            break
        fi
    done
fi

# Create logs directory
mkdir -p "${PROJECT}/logs"

#==============================================================================
# VERIFY PREREQUISITES
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying prerequisites..."
echo "========================================================================"

# Check for infilled tiles
TILES_DIR="${PROJECT}/deepfill/tiles_infilled"
if [ ! -d "$TILES_DIR" ]; then
    echo "ERROR: Infilled tiles directory not found: $TILES_DIR"
    exit 1
fi

# Count tiles (directory already verified to exist above)
TILE_COUNT=$(find "$TILES_DIR" -name "*.png" | wc -l)
echo "Found ${TILE_COUNT} infilled tiles in: $TILES_DIR"

if [ $TILE_COUNT -lt 70000 ]; then
    echo "ERROR: Expected ~73,472 tiles, found only ${TILE_COUNT}"
    echo "Ensure the inference job completed successfully first."
    exit 1
fi

# Check for tiling metadata
TILING_META="${PROJECT}/repaint/sinogram_tiles/tiling_metadata.json"
if [ ! -f "$TILING_META" ]; then
    echo "ERROR: Tiling metadata not found: $TILING_META"
    exit 1
fi
echo "Tiling metadata: $TILING_META"

# Check for dataset metadata
DATASET_META="${PROJECT}/sinogram_dataset/metadata.json"
if [ ! -f "$DATASET_META" ]; then
    echo "ERROR: Dataset metadata not found: $DATASET_META"
    exit 1
fi
echo "Dataset metadata: $DATASET_META"

# Check for scan geometry
if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
    echo "ERROR: Scan geometry not found: ${SCAN_FOLDER}/scan.xml"
    echo "This file is required for FDK reconstruction."
    exit 1
fi
echo "Scan geometry: ${SCAN_FOLDER}/scan.xml"

echo ""
echo "All prerequisites verified!"

#==============================================================================
# MODULE LOADING - Following Alliance Canada official documentation
# Reference: https://docs.alliancecan.ca/wiki/PyTorch
#==============================================================================

echo ""
echo "========================================================================"
echo "Loading modules..."
echo "========================================================================"

# Reset to default environment (keeps StdEnv loaded)
module reset 2>/dev/null || true

# Load Python
echo "  Loading python/3.10..."
if ! module load python/3.10 2>&1; then
    echo "  python/3.10 not available, trying python/3.11..."
    if ! module load python/3.11 2>&1; then
        echo "  Trying default python..."
        module load python 2>&1 || {
            echo "ERROR: No Python module available"
            exit 1
        }
    fi
fi

# Load scipy-stack with specific version for scipy.io compatibility
# Note: Loading a specific version to avoid HDF5/scipy.io issues
echo "  Loading scipy-stack (specific version for scipy.io compatibility)..."
SCIPY_LOADED=0
for scipy_ver in scipy-stack/2024a scipy-stack/2023b scipy-stack/2023a scipy-stack; do
    if module load "$scipy_ver" 2>&1; then
        echo "    Loaded: $scipy_ver"
        SCIPY_LOADED=1
        break
    fi
done
if [ $SCIPY_LOADED -eq 0 ]; then
    echo "  Warning: scipy-stack not available (will use pip packages)"
fi

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

cd $SLURM_TMPDIR
echo "Working directory: $(pwd)"
echo "Available space:"
df -h $SLURM_TMPDIR

# Clone repository for merge/reconstruct scripts
echo ""
echo "Cloning repository..."
if ! git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git 2>/dev/null; then
    echo "SSH clone failed, trying HTTPS..."
    git clone --depth=1 https://github.com/falkwiegmann/ct_recon.git || {
        echo "ERROR: Git clone failed"
        exit 1
    }
fi

cd ct_recon/Base_model_comparison

# Create virtual environment
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

# Upgrade pip
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

# Install required packages
echo "Installing additional packages..."
PACKAGES="numpy pillow tqdm scipy pyyaml imageio matplotlib"
for pkg in $PACKAGES; do
    echo "  Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "ERROR: Failed to install $pkg"
        exit 1
    }
done

# xmltodict is required for loading scan.xml geometry
echo "  Installing xmltodict..."
pip install xmltodict || {
    echo "ERROR: xmltodict installation failed - required for reconstruction"
    exit 1
}

echo ""
echo "All packages installed successfully!"

#==============================================================================
# VERIFY INSTALLATION
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying installation..."
echo "========================================================================"

python3 << 'PYTHON_CHECK'
import sys
errors = []

packages = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('PIL', 'Pillow'),
    ('tqdm', 'tqdm'),
    ('scipy', 'SciPy'),
    ('scipy.io', 'scipy.io'),
    ('yaml', 'PyYAML'),
    ('xmltodict', 'xmltodict'),
    ('imageio', 'imageio'),
]

for import_name, display_name in packages:
    try:
        __import__(import_name)
        print(f"  OK: {display_name}")
    except ImportError as e:
        print(f"  MISSING: {display_name} ({e})")
        errors.append(display_name)

if errors:
    print(f"\nERROR: Missing required packages: {', '.join(errors)}")
    sys.exit(1)

print("\nAll required packages available!")

# Check CUDA
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available - reconstruction will be slow on CPU")
PYTHON_CHECK

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

#==============================================================================
# MERGE TILES
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 1/2: Merging tiles with NEAREST blending..."
echo "========================================================================"

OUTPUT_SINOS="${PROJECT}/deepfill/sinograms_infilled_nearest"

# Check if merged sinograms already exist
# Note: Use || true to prevent pipefail from killing script if directory doesn't exist
if [ -d "${OUTPUT_SINOS}" ]; then
    EXISTING_SINOS=$(find "${OUTPUT_SINOS}" -name "*.png" | wc -l)
else
    EXISTING_SINOS=0
fi
EXPECTED_SINOS=2296  # Expected number of sinograms based on dataset
echo "  Output directory: ${OUTPUT_SINOS}"
echo "  Existing sinograms: ${EXISTING_SINOS}/${EXPECTED_SINOS}"

SKIP_MERGE=0
if [ $EXISTING_SINOS -ge $EXPECTED_SINOS ]; then
    echo ""
    echo "Found ${EXISTING_SINOS} existing merged sinograms (complete)."
    echo "  Location: ${OUTPUT_SINOS}"
    echo "  Skipping merge step (delete directory to force re-merge)"
    SKIP_MERGE=1
elif [ $EXISTING_SINOS -gt 0 ]; then
    echo ""
    echo "Found ${EXISTING_SINOS}/${EXPECTED_SINOS} merged sinograms (incomplete)."
    echo "  Deleting and re-merging..."
    rm -rf "${OUTPUT_SINOS}"
fi

if [ "${SKIP_MERGE:-0}" -eq 0 ]; then
    echo ""
    echo "Running merge_deepfill_tiles.py..."
    echo "  Input tiles: ${TILES_DIR}"
    echo "  Metadata: ${TILING_META}"
    echo "  Output: ${OUTPUT_SINOS}"
    echo "  Blend mode: nearest (NO averaging - preserves true model output)"
    echo ""
    echo "Starting merge at: $(date)"

    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_SINOS}"

    python3 merge_deepfill_tiles.py \
        --tiles_dir "${TILES_DIR}" \
        --metadata_path "${TILING_META}" \
        --output_dir "${OUTPUT_SINOS}" \
        --blend_mode nearest

    MERGE_STATUS=$?
    echo "Merge finished at: $(date)"

    if [ $MERGE_STATUS -ne 0 ]; then
        echo "ERROR: Tile merging failed with status ${MERGE_STATUS}"
        exit 1
    fi

    # Verify merge output
    if [ -d "${OUTPUT_SINOS}" ]; then
        MERGED_COUNT=$(find "${OUTPUT_SINOS}" -name "*.png" | wc -l)
    else
        MERGED_COUNT=0
    fi
    echo ""
    echo "Merge complete: ${MERGED_COUNT} sinograms created"

    if [ $MERGED_COUNT -lt $EXPECTED_SINOS ]; then
        echo "ERROR: Expected ${EXPECTED_SINOS} sinograms but only got ${MERGED_COUNT}"
        exit 1
    fi
fi

#==============================================================================
# RECONSTRUCTION
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 2/2: Running FDK reconstruction..."
echo "========================================================================"

RECON_OUTPUT="${PROJECT}/deepfill/reconstructed_volume_nearest"

echo "Input sinograms: ${OUTPUT_SINOS}"
echo "Output volume: ${RECON_OUTPUT}"
echo "Scan geometry: ${SCAN_FOLDER}"
echo "Dataset metadata: ${DATASET_META}"
echo "Tiling metadata: ${TILING_META}"

# Verify sinograms exist before reconstruction
if [ -d "${OUTPUT_SINOS}" ]; then
    SINO_FOR_RECON=$(find "${OUTPUT_SINOS}" -name "*.png" | wc -l)
else
    SINO_FOR_RECON=0
fi

if [ $SINO_FOR_RECON -lt $EXPECTED_SINOS ]; then
    echo "ERROR: Not enough sinograms for reconstruction."
    echo "  Found: ${SINO_FOR_RECON}"
    echo "  Expected: ${EXPECTED_SINOS}"
    exit 1
fi

echo ""
echo "Starting reconstruction at: $(date)"
echo "This may take 1-2 hours with GPU acceleration..."

# Create output directory
mkdir -p "${RECON_OUTPUT}"

python3 reconstruct_from_deepfill.py \
    --sinogram_dir "${OUTPUT_SINOS}" \
    --output_dir "${RECON_OUTPUT}" \
    --scan_folder "${SCAN_FOLDER}" \
    --metadata_path "${DATASET_META}" \
    --tiling_metadata_path "${TILING_META}"

RECON_STATUS=$?
echo "Reconstruction finished at: $(date)"

if [ $RECON_STATUS -ne 0 ]; then
    echo "ERROR: Reconstruction failed with status ${RECON_STATUS}"
    exit 1
fi

#==============================================================================
# CLEANUP AND SUMMARY
#==============================================================================

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  Merged sinograms: ${OUTPUT_SINOS}/"
echo "  Reconstructed volume: ${RECON_OUTPUT}/"
echo ""

# Show output stats
if [ -d "${OUTPUT_SINOS}" ]; then
    SINO_COUNT=$(find "${OUTPUT_SINOS}" -name "*.png" | wc -l)
else
    SINO_COUNT=0
fi
echo "Statistics:"
echo "  Merged sinograms: ${SINO_COUNT}"

if [ -f "${RECON_OUTPUT}.vff" ]; then
    VFF_SIZE=$(du -h "${RECON_OUTPUT}.vff" | cut -f1)
    echo "  Reconstruction VFF: ${VFF_SIZE}"
fi

echo ""
echo "End time: $(date)"
echo "========================================================================"
