#!/bin/bash
#SBATCH --time=02:00:00                 # 2 hours (merge ~30min + reconstruction ~30min)
#SBATCH --job-name=deepfill_gaussian
#SBATCH --output=logs/deepfill_gaussian_%j.out
#SBATCH --error=logs/deepfill_gaussian_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1                    # 1 GPU for reconstruction (any type works)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4               # 4 CPUs sufficient for merge + reconstruction
#SBATCH --mem=32G                       # 32GB RAM (memmap handles large data)
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

#==============================================================================
# DeepFill v2 Gaussian Feathering Merge + Reconstruction
# Uses existing tiles_infilled to merge with Gaussian blending (sigma=16)
# Then reconstructs the sinograms to a volume
#
# Input: deepfill/tiles_infilled/ (already exists from previous run)
# Output: deepfill/sinograms_infilled_gaussian/
#         deepfill/reconstructed_volume_gaussian.vff
#==============================================================================

# Configuration
GAUSSIAN_SIGMA=16

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
else
    CLUSTER="unknown"
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    elif [ -d "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/project/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot auto-detect project path"
        exit 1
    fi
fi

echo "========================================================================"
echo "DeepFill v2 Gaussian Merge + Reconstruction"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "Gaussian sigma: ${GAUSSIAN_SIGMA}"
echo "Start time: $(date)"
echo "========================================================================"

# Create logs directory
mkdir -p $PROJECT/logs

# Change to SLURM_TMPDIR for fast local storage
cd $SLURM_TMPDIR
echo "Working directory: $(pwd)"
echo "Available space:"
df -h $SLURM_TMPDIR

#==============================================================================
# VERIFY REQUIRED FILES EXIST
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying required input files..."
echo "========================================================================"

# Check tiles exist
if [ ! -d "$PROJECT/deepfill/tiles_infilled" ]; then
    echo "ERROR: tiles_infilled not found at:"
    echo "  $PROJECT/deepfill/tiles_infilled"
    exit 1
fi

TILE_COUNT=$(find "$PROJECT/deepfill/tiles_infilled" -name "*.png" 2>/dev/null | wc -l)
echo "Found ${TILE_COUNT} infilled tiles"

if [ $TILE_COUNT -lt 70000 ]; then
    echo "ERROR: Not enough tiles. Expected ~73,472, found ${TILE_COUNT}"
    exit 1
fi

# Check tiling metadata exists
if [ ! -f "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" ]; then
    echo "ERROR: Tiling metadata not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/tiling_metadata.json"
    exit 1
fi
echo "Found tiling metadata"

# Check dataset metadata exists
if [ ! -f "$PROJECT/sinogram_dataset/metadata.json" ]; then
    echo "ERROR: Dataset metadata not found at:"
    echo "  $PROJECT/sinogram_dataset/metadata.json"
    exit 1
fi
echo "Found dataset metadata"

#==============================================================================
# SETUP: Clone repository and load modules
#==============================================================================

echo ""
echo "Cloning repository..."

if ! git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git 2>/dev/null; then
    echo "SSH clone failed, trying HTTPS..."
    git clone --depth=1 https://github.com/falkwiegmann/ct_recon.git || {
        echo "ERROR: Git clone failed"
        exit 1
    }
fi

cd ct_recon/Base_model_comparison || {
    echo "ERROR: Failed to change to repository directory"
    exit 1
}

#==============================================================================
# MODULE LOADING
#==============================================================================

echo ""
echo "Loading modules..."

module reset 2>/dev/null || true

echo "  Loading python/3.10..."
if ! module load python/3.10 2>&1; then
    echo "  python/3.10 not available, trying python/3.11..."
    if ! module load python/3.11 2>&1; then
        module load python 2>&1 || {
            echo "ERROR: No Python module available"
            exit 1
        }
    fi
fi

module load scipy-stack 2>&1 || echo "  scipy-stack not available"

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
    pip install torch || {
        echo "ERROR: Failed to install PyTorch"
        exit 1
    }
fi

# Install other required packages
PACKAGES="numpy pillow tqdm scipy pyyaml imageio"
for pkg in $PACKAGES; do
    echo "  Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "ERROR: Failed to install $pkg"
        exit 1
    }
done

# Install packages that may not have pre-built wheels
echo "  Installing xmltodict..."
pip install xmltodict || {
    echo "ERROR: Failed to install xmltodict"
    exit 1
}

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
    print("WARNING: CUDA not available, reconstruction will be slower on CPU")

print("\nPyTorch verification passed!")
CUDA_CHECK

#==============================================================================
# CHECK IF MERGED SINOGRAMS ALREADY EXIST (SKIP MERGE IF SO)
#==============================================================================

echo ""
echo "========================================================================"
echo "Checking for existing merged sinograms..."
echo "========================================================================"

EXPECTED_SINOGRAMS=2296  # Expected number of sinograms
EXISTING_SINOS=$(find "$PROJECT/deepfill/sinograms_infilled_gaussian" -name "*.png" 2>/dev/null | wc -l)

if [ $EXISTING_SINOS -ge $EXPECTED_SINOGRAMS ]; then
    echo "Found ${EXISTING_SINOS} existing sinograms in project directory"
    echo "Skipping tile merge - will use existing sinograms for reconstruction"
    SKIP_MERGE=1
else
    if [ $EXISTING_SINOS -gt 0 ]; then
        echo "Found partial sinograms (${EXISTING_SINOS}/${EXPECTED_SINOGRAMS})"
        echo "Will re-run merge to complete..."
    else
        echo "No existing sinograms found - will run merge"
    fi
    SKIP_MERGE=0
fi

#==============================================================================
# COPY DATA TO LOCAL NVMe
#==============================================================================

echo ""
echo "========================================================================"
echo "Copying data to local NVMe storage..."
echo "========================================================================"

# Create local directories
mkdir -p deepfill/sinograms_infilled_gaussian
mkdir -p repaint/sinogram_tiles
mkdir -p sinogram_dataset

# Copy metadata files (always needed)
echo "Copying metadata files..."
cp "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" repaint/sinogram_tiles/
cp "$PROJECT/sinogram_dataset/metadata.json" sinogram_dataset/

if [ $SKIP_MERGE -eq 1 ]; then
    # Copy existing sinograms instead of tiles
    echo "Copying existing sinograms to local NVMe..."
    rsync -av --info=progress2 "$PROJECT/deepfill/sinograms_infilled_gaussian/" deepfill/sinograms_infilled_gaussian/

    MERGED_COUNT=$(find deepfill/sinograms_infilled_gaussian -name "*.png" 2>/dev/null | wc -l)
    echo "Sinograms copied to local NVMe: ${MERGED_COUNT}"
else
    # Copy tiles for merging
    mkdir -p deepfill/tiles_infilled
    echo "Copying infilled tiles to local NVMe..."
    rsync -av --info=progress2 "$PROJECT/deepfill/tiles_infilled/" deepfill/tiles_infilled/

    LOCAL_TILES=$(find deepfill/tiles_infilled -name "*.png" 2>/dev/null | wc -l)
    echo "Tiles copied to local NVMe: ${LOCAL_TILES}"

    #==========================================================================
    # MERGE TILES WITH GAUSSIAN FEATHERING
    #==========================================================================

    echo ""
    echo "========================================================================"
    echo "Merging tiles with Gaussian feathering (sigma=${GAUSSIAN_SIGMA})..."
    echo "========================================================================"

    python3 merge_deepfill_tiles.py \
        --tiles_dir deepfill/tiles_infilled \
        --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
        --output_dir deepfill/sinograms_infilled_gaussian \
        --blend_mode gaussian \
        --gaussian_sigma ${GAUSSIAN_SIGMA}

    MERGE_STATUS=$?

    if [ $MERGE_STATUS -ne 0 ]; then
        echo "ERROR: Merge failed with status ${MERGE_STATUS}"
        exit 1
    fi

    # Verify merge output
    MERGED_COUNT=$(find deepfill/sinograms_infilled_gaussian -name "*.png" 2>/dev/null | wc -l)
    echo "Merged sinograms: ${MERGED_COUNT}"

    #==========================================================================
    # COPY MERGED SINOGRAMS BACK TO PROJECT
    #==========================================================================

    echo ""
    echo "========================================================================"
    echo "Copying merged sinograms back to project directory..."
    echo "========================================================================"

    mkdir -p $PROJECT/deepfill/sinograms_infilled_gaussian
    rsync -av --info=progress2 deepfill/sinograms_infilled_gaussian/ $PROJECT/deepfill/sinograms_infilled_gaussian/

    echo "Merged sinograms saved to: $PROJECT/deepfill/sinograms_infilled_gaussian/"
fi

#==============================================================================
# RUN RECONSTRUCTION
#==============================================================================

echo ""
echo "========================================================================"
echo "Running FDK reconstruction..."
echo "========================================================================"

# Compute scan folder path (scan.xml is needed for geometry parameters)
# PROJECT is .../ct_recon/Base_model_comparison, so strip that suffix to get ct_recon root
SCAN_FOLDER="${PROJECT%/Base_model_comparison}/data/results/Scan_1681_uwarp_gt"

if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
    echo "ERROR: scan.xml not found at ${SCAN_FOLDER}/scan.xml"
    echo "Trying alternative paths..."

    # Try alternative paths on Compute Canada
    for alt_path in \
        "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/data/results/Scan_1681_uwarp_gt" \
        "/project/def-nlford/wiegmann/ct_recon/data/results/Scan_1681_uwarp_gt"; do
        if [ -f "${alt_path}/scan.xml" ]; then
            SCAN_FOLDER="${alt_path}"
            echo "Found scan.xml at: ${SCAN_FOLDER}"
            break
        fi
    done

    # Final check
    if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
        echo "ERROR: Could not find scan.xml in any expected location"
        exit 1
    fi
fi

echo "Using scan folder: ${SCAN_FOLDER}"

python3 reconstruct_from_deepfill.py \
    --sinogram_dir deepfill/sinograms_infilled_gaussian \
    --output_dir deepfill/reconstructed_volume_gaussian \
    --metadata_path sinogram_dataset/metadata.json \
    --tiling_metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --scan_folder "${SCAN_FOLDER}"

RECON_STATUS=$?

if [ $RECON_STATUS -ne 0 ]; then
    echo "ERROR: Reconstruction failed with status ${RECON_STATUS}"
    exit 1
fi

#==============================================================================
# COPY RECONSTRUCTION RESULTS BACK TO PROJECT
#==============================================================================

echo ""
echo "========================================================================"
echo "Copying reconstruction results back to project directory..."
echo "========================================================================"

# Copy the VFF file and any other reconstruction outputs
if [ -f "deepfill/reconstructed_volume_gaussian.vff" ]; then
    cp deepfill/reconstructed_volume_gaussian.vff $PROJECT/deepfill/
    echo "Copied: reconstructed_volume_gaussian.vff"
fi

# Also copy the directory if it exists
if [ -d "deepfill/reconstructed_volume_gaussian" ]; then
    rsync -av deepfill/reconstructed_volume_gaussian/ $PROJECT/deepfill/reconstructed_volume_gaussian/
    echo "Copied: reconstructed_volume_gaussian/"
fi

#==============================================================================
# CLEANUP AND SUMMARY
#==============================================================================

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo "Gaussian sigma: ${GAUSSIAN_SIGMA}"
echo "Output files:"
echo "  Merged sinograms: $PROJECT/deepfill/sinograms_infilled_gaussian/"
echo "  Reconstruction: $PROJECT/deepfill/reconstructed_volume_gaussian.vff"
echo ""
echo "Finished at: $(date)"
echo "========================================================================"
