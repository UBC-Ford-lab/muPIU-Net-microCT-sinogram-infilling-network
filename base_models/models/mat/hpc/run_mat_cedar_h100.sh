#!/bin/bash
#SBATCH --time=04:00:00                  # 4 hours (MAT is moderately fast)
#SBATCH --job-name=mat_h100
#SBATCH --output=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_%A_%a.out
#SBATCH --error=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_%A_%a.err
#SBATCH --array=0-3                      # 4 parallel jobs (one per GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1               # 1 H100 per array job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # More CPUs for data loading
#SBATCH --mem=64G                        # 64GB RAM
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

# Exit on error, but with better error reporting
set -eo pipefail

# Trap errors to provide more context
trap 'echo "ERROR at line $LINENO: command \"$BASH_COMMAND\" failed with exit code $?"' ERR

#==============================================================================
# MAT (Mask-Aware Transformer) Sinogram Infilling on Compute Canada - H100 Optimized
# Processes 73,472 tiles across 4 H100 GPUs in parallel
# Expected runtime: 1-2 hours (transformer-based, batch processing)
# Works on: Cedar, Fir, or any cluster with H100 GPUs
#
# WORKFLOW:
# 1. All tasks copy RePaint tiles to local NVMe (in parallel)
# 2. All tasks run MAT inference in parallel on their subset of tiles
#    - Uses same tiles and masks as RePaint/DeepFill (256x256)
#    - Masks are NOT inverted (MAT uses same convention as RePaint)
# 3. Task 0 waits for all tasks, then runs merge + reconstruction
#
# TIME ESTIMATE:
# - 73,472 tiles / 4 GPUs = ~18,368 tiles per GPU
# - MAT processes ~30-50 tiles/sec with batch_size=8
# - Inference: ~10-15 minutes per GPU
# - Merge + Reconstruction: ~30 minutes
# - Total: ~1-2 hours
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
else
    CLUSTER="unknown"
    # Try to auto-detect
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    elif [ -d "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/project/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot auto-detect project path"
        exit 1
    fi
fi

# Get array task ID (0-3)
TASK_ID=${SLURM_ARRAY_TASK_ID}
N_TASKS=4

echo "========================================================================"
echo "MAT H100 Array Job ${TASK_ID}/${N_TASKS}"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "Start time: $(date)"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p $PROJECT/logs

# Change to SLURM_TMPDIR for fast local storage (7.84TB NVMe!)
cd $SLURM_TMPDIR
echo "Working directory: $(pwd)"
echo "Available space:"
df -h $SLURM_TMPDIR

#==============================================================================
# SETUP: Clone repository and load modules
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

cd ct_recon/Base_model_comparison || {
    echo "ERROR: Failed to change to repository directory"
    exit 1
}

# IMPORTANT: Copy updated Python scripts from PROJECT to override GitHub versions
# This ensures we use the latest local changes (e.g., --input_dir support, generator fixes)
echo ""
echo "Updating Python scripts from PROJECT directory..."
cp "$PROJECT"/*.py . 2>/dev/null || echo "Warning: Could not copy .py files from PROJECT"
echo "Python scripts updated from PROJECT"

#==============================================================================
# MODULE LOADING - Following Alliance Canada official documentation
# Reference: https://docs.alliancecan.ca/wiki/PyTorch
# IMPORTANT: Do NOT use 'module purge' as it removes StdEnv which is required
#            for loading CUDA modules on the hierarchical Lmod system
#==============================================================================

echo ""
echo "Loading modules..."

# Reset to default environment (keeps StdEnv loaded, unlike 'module purge')
module reset 2>/dev/null || true

# Load Python first (required for virtual environment)
echo "  Loading python/3.10..."
if ! module load python/3.10 2>&1; then
    echo "  python/3.10 not available, trying python/3.11..."
    if ! module load python/3.11 2>&1; then
        echo "  python/3.11 not available, trying default python..."
        module load python 2>&1 || {
            echo "ERROR: No Python module available"
            exit 1
        }
    fi
fi

# Load scipy-stack for numpy, scipy, etc.
echo "  Loading scipy-stack..."
module load scipy-stack 2>&1 || echo "  scipy-stack not available, will use pip packages"

# CRITICAL: Load OpenCV module BEFORE creating virtual environment
# Reference: https://docs.alliancecan.ca/wiki/OpenCV
echo "  Loading gcc (required for opencv)..."
module load gcc 2>&1 || echo "  gcc module not available"

echo "  Loading opencv..."
if module load opencv/4.8.1 2>&1; then
    echo "  opencv/4.8.1 loaded successfully"
elif module load opencv/4.8.0 2>&1; then
    echo "  opencv/4.8.0 loaded successfully"
elif module load opencv 2>&1; then
    echo "  opencv (default version) loaded successfully"
else
    echo "  WARNING: opencv module not available, will try pip install (may fail)"
fi

# Show loaded modules
echo ""
echo "Loaded modules:"
module list 2>&1 | head -20

#==============================================================================
# VIRTUAL ENVIRONMENT SETUP - Following Alliance Canada best practices
# Using a fresh virtualenv in SLURM_TMPDIR for faster I/O and to avoid quota issues
# Reference: https://docs.alliancecan.ca/wiki/PyTorch
#==============================================================================

echo ""
echo "========================================================================"
echo "Setting up Python virtual environment..."
echo "========================================================================"

# Create a fresh virtual environment in SLURM_TMPDIR (fast local storage)
# This avoids quota issues and provides faster package installation
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
# INSTALL DEPENDENCIES - Using Alliance Canada pre-built wheels
# IMPORTANT: Use --no-index to get pre-built wheels optimized for the cluster
# H100 GPUs require torch 2.3+ according to Alliance docs
#==============================================================================

echo ""
echo "========================================================================"
echo "Installing Python dependencies..."
echo "========================================================================"

# First, upgrade pip
pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip

# Install PyTorch with CUDA support using Alliance pre-built wheels
# --no-index uses wheels from $WHEELHOUSE which are optimized for the cluster
echo "Installing PyTorch (this may take a minute)..."
if ! pip install --no-index torch 2>/dev/null; then
    echo "  Pre-built wheel not available, installing from PyPI..."
    pip install torch || {
        echo "ERROR: Failed to install PyTorch"
        exit 1
    }
fi

# Install other required packages
# NOTE: opencv-python is NOT included - it's loaded as a module above
echo "Installing additional packages..."
PACKAGES="numpy pillow tqdm scipy pyyaml scikit-image"
for pkg in $PACKAGES; do
    echo "  Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "ERROR: Failed to install $pkg"
        exit 1
    }
done

# Install MAT-specific dependencies
echo "  Installing MAT-specific packages..."
MAT_PACKAGES="click requests timm psutil scikit-learn easydict"
for pkg in $MAT_PACKAGES; do
    echo "    Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "WARNING: Failed to install $pkg, continuing..."
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
# VERIFY INSTALLATION
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying installation..."
echo "========================================================================"

# Check all required imports
echo "Checking Python imports..."
python3 << 'PYTHON_CHECK'
import sys
errors = []

# Required packages
packages = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('PIL', 'Pillow'),
    ('tqdm', 'tqdm'),
    ('scipy', 'SciPy'),
    ('yaml', 'PyYAML'),
]

for import_name, display_name in packages:
    try:
        __import__(import_name)
        print(f"  OK: {display_name}")
    except ImportError as e:
        print(f"  MISSING: {display_name} ({e})")
        errors.append(display_name)

# Check OpenCV (loaded as module or via pip)
try:
    import cv2
    print(f"  OK: OpenCV (cv2)")
except ImportError:
    print(f"  WARNING: OpenCV not available (needed for MAT)")
    errors.append('OpenCV')

# Check optional packages
try:
    import xmltodict
    print(f"  OK: xmltodict")
except ImportError:
    print(f"  WARNING: xmltodict (optional, needed for reconstruction)")

if errors:
    print(f"\nERROR: Missing required packages: {', '.join(errors)}")
    sys.exit(1)

print("\nAll required packages are available!")
PYTHON_CHECK

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

# Check PyTorch and CUDA
echo ""
echo "Checking PyTorch and CUDA..."
python3 << 'CUDA_CHECK'
import torch
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nERROR: PyTorch cannot access CUDA!")
    print("This may be because:")
    print("  1. PyTorch was installed without CUDA support")
    print("  2. NVIDIA driver is incompatible")
    print("\nTo fix, try:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

print(f"  CUDA version: {torch.version.cuda}")
print(f"  GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

# Check torch version for H100 compatibility
major, minor = torch.__version__.split('.')[:2]
if int(major) < 2 or (int(major) == 2 and int(minor) < 3):
    print(f"\nWARNING: H100 GPUs require PyTorch 2.3+, you have {torch.__version__}")

print("\nCUDA verification passed!")
CUDA_CHECK

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA verification failed"
    exit 1
fi

echo ""
echo "GPU info (nvidia-smi):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || nvidia-smi -L

#==============================================================================
# VERIFY REPAINT TILES (same tiles used for MAT - 256x256 format)
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying RePaint tiles (will use directly for MAT)..."
echo "========================================================================"

# Use sinograms_masked (the proper input for inpainting)
INPUT_TILE_DIR="sinograms_masked"
if [ ! -d "$PROJECT/repaint/sinogram_tiles/sinograms_masked" ]; then
    echo "ERROR: sinograms_masked not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/sinograms_masked"
    echo ""
    echo "Please ensure RePaint masked tiles are transferred to Compute Canada first."
    echo "Note: sinograms_gt is NOT a valid fallback - we need the masked tiles for proper inpainting."
    exit 1
fi
echo "Found sinograms_masked directory"

REPAINT_INPUT_COUNT=$(find "$PROJECT/repaint/sinogram_tiles/$INPUT_TILE_DIR" -name "*.png" 2>/dev/null | wc -l)
REPAINT_MASK_COUNT=$(find "$PROJECT/repaint/sinogram_tiles/masks" -name "*.png" 2>/dev/null | wc -l)
echo "Found RePaint tiles: Input=${REPAINT_INPUT_COUNT} (${INPUT_TILE_DIR}), Masks=${REPAINT_MASK_COUNT}"

if [ $REPAINT_INPUT_COUNT -lt 70000 ]; then
    echo "ERROR: Not enough input tiles. Expected ~73,472, found ${REPAINT_INPUT_COUNT}"
    exit 1
fi

echo ""
echo "NOTE: Using ${INPUT_TILE_DIR} - 256x256 format works for MAT!"
echo "      Mask format is compatible (0=inpaint, 255=keep)"

#==============================================================================
# DATA TRANSFER: Copy RePaint tiles to local NVMe for fast I/O
#==============================================================================

echo ""
echo "========================================================================"
echo "Transferring RePaint tiles to local NVMe storage..."
echo "========================================================================"

# Create local directories using the detected tile dir name
mkdir -p "repaint/sinogram_tiles/${INPUT_TILE_DIR}"
mkdir -p repaint/sinogram_tiles/masks
mkdir -p mat/tiles_infilled

# Copy input tiles and masks from RePaint
echo "Copying input tiles (${INPUT_TILE_DIR}) from RePaint to local NVMe..."
rsync -av "$PROJECT/repaint/sinogram_tiles/${INPUT_TILE_DIR}/" "repaint/sinogram_tiles/${INPUT_TILE_DIR}/"

echo "Copying mask tiles from RePaint to local NVMe..."
rsync -av "$PROJECT/repaint/sinogram_tiles/masks/" repaint/sinogram_tiles/masks/

# Copy metadata
if [ -f "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" ]; then
    cp "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" repaint/sinogram_tiles/
fi

# Verify tiles copied successfully
INPUT_COPIED=$(find "repaint/sinogram_tiles/${INPUT_TILE_DIR}" -name "*.png" 2>/dev/null | wc -l)
MASK_COPIED=$(find repaint/sinogram_tiles/masks -name "*.png" 2>/dev/null | wc -l)

echo "RePaint tiles copied to local NVMe: Input=${INPUT_COPIED}, Masks=${MASK_COPIED}"

if [ $INPUT_COPIED -lt 70000 ]; then
    echo "ERROR: Input tile copy failed - expected ~73,472 tiles, got ${INPUT_COPIED}"
    exit 1
fi

if [ $MASK_COPIED -lt 70000 ]; then
    echo "ERROR: Mask tile copy failed - expected ~73,472 tiles, got ${MASK_COPIED}"
    exit 1
fi

#==============================================================================
# COPY MAT MODEL AND WEIGHTS
#==============================================================================

echo ""
echo "Copying MAT model and weights..."

mkdir -p mat/MAT/pretrained

# Copy the entire MAT repository structure
if [ -d "$PROJECT/mat/MAT" ]; then
    echo "Copying MAT repository..."
    rsync -av "$PROJECT/mat/MAT/" mat/MAT/
else
    echo "MAT repository not found, cloning from GitHub..."
    cd mat
    git clone --depth=1 https://github.com/fenglinglwb/MAT.git
    cd ..
fi

# Verify checkpoint exists
CHECKPOINT_FILE="mat/MAT/pretrained/CelebA-HQ_256.pkl"
if [ ! -f "$CHECKPOINT_FILE" ]; then
    # Try to copy from project directory
    if [ -f "$PROJECT/mat/MAT/pretrained/CelebA-HQ_256.pkl" ]; then
        cp "$PROJECT/mat/MAT/pretrained/CelebA-HQ_256.pkl" "$CHECKPOINT_FILE"
    else
        echo "ERROR: MAT checkpoint not found at:"
        echo "  $PROJECT/mat/MAT/pretrained/CelebA-HQ_256.pkl"
        echo ""
        echo "Please download the CelebA-HQ 256x256 checkpoint first."
        exit 1
    fi
fi

CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
echo "MAT checkpoint found: $CHECKPOINT_FILE ($CHECKPOINT_SIZE)"

#==============================================================================
# CALCULATE TILE DISTRIBUTION
#==============================================================================

TOTAL_TILES=$INPUT_COPIED
TILES_PER_TASK=$((TOTAL_TILES / N_TASKS))
REMAINDER=$((TOTAL_TILES % N_TASKS))

# Calculate start and end indices for this task
START_IDX=$((TASK_ID * TILES_PER_TASK))
if [ $TASK_ID -lt $REMAINDER ]; then
    START_IDX=$((START_IDX + TASK_ID))
    END_IDX=$((START_IDX + TILES_PER_TASK + 1))
else
    START_IDX=$((START_IDX + REMAINDER))
    END_IDX=$((START_IDX + TILES_PER_TASK))
fi

echo ""
echo "========================================================================"
echo "Tile distribution for Task ${TASK_ID}:"
echo "========================================================================"
echo "  Total tiles: ${TOTAL_TILES}"
echo "  Tiles per task: ~${TILES_PER_TASK}"
echo "  This task: tiles ${START_IDX} to ${END_IDX} ($((END_IDX - START_IDX)) tiles)"

#==============================================================================
# RUN MAT INFERENCE
#==============================================================================

echo ""
echo "========================================================================"
echo "Starting MAT inference..."
echo "========================================================================"

# Run inference with batch processing
# Uses auto-detected tile directory (sinograms_masked or sinograms_gt)
# MAT mask format matches RePaint (0=inpaint, 255=keep), so NO inversion needed
echo "Using input tiles from: repaint/sinogram_tiles/${INPUT_TILE_DIR}"
python3 run_mat_inference.py \
    --input_dir "repaint/sinogram_tiles/${INPUT_TILE_DIR}" \
    --mask_dir repaint/sinogram_tiles/masks \
    --output_dir mat/tiles_infilled \
    --checkpoint "$CHECKPOINT_FILE" \
    --batch_size 8 \
    --device cuda \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --save_grayscale

INFERENCE_STATUS=$?

if [ $INFERENCE_STATUS -ne 0 ]; then
    echo "ERROR: Inference failed with status ${INFERENCE_STATUS}"
    exit 1
fi

echo ""
echo "Inference complete for Task ${TASK_ID}!"

#==============================================================================
# COPY RESULTS BACK TO PROJECT DIRECTORY
#==============================================================================

echo ""
echo "========================================================================"
echo "Copying results back to project directory..."
echo "========================================================================"

# Copy infilled tiles back
mkdir -p $PROJECT/mat/tiles_infilled
rsync -av mat/tiles_infilled/ $PROJECT/mat/tiles_infilled/

TILES_PRODUCED=$(find mat/tiles_infilled -name "*.png" 2>/dev/null | wc -l)
echo "Tiles produced by Task ${TASK_ID}: ${TILES_PRODUCED}"

#==============================================================================
# CHECK IF ALL TASKS COMPLETE (only Task 0 does final merge)
#==============================================================================

if [ $TASK_ID -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Task 0: Waiting for all tasks to complete, then merge..."
    echo "========================================================================"

    # Wait for all tiles to be produced
    EXPECTED_TILES=$TOTAL_TILES
    MAX_WAIT=7200  # 2 hours max wait for inference
    WAIT_TIME=0

    while [ $WAIT_TIME -lt $MAX_WAIT ]; do
        CURRENT_TILES=$(find $PROJECT/mat/tiles_infilled -name "*.png" 2>/dev/null | wc -l)

        if [ $CURRENT_TILES -ge $EXPECTED_TILES ]; then
            echo "All tiles complete: ${CURRENT_TILES}/${EXPECTED_TILES}"
            break
        fi

        echo "Waiting for tiles: ${CURRENT_TILES}/${EXPECTED_TILES} (${WAIT_TIME}s elapsed)"
        sleep 60
        WAIT_TIME=$((WAIT_TIME + 60))
    done

    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "Warning: Timeout waiting for all tiles. Proceeding with merge anyway."
        echo "  Current tiles: $(find $PROJECT/mat/tiles_infilled -name '*.png' 2>/dev/null | wc -l)"
    fi

    # Run merge from project directory
    echo ""
    echo "Merging tiles..."
    cd $PROJECT
    python3 merge_mat_tiles.py \
        --tiles_dir mat/tiles_infilled \
        --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
        --output_dir mat/sinograms_infilled \
        --blend_mode nearest

    echo ""
    echo "Running reconstruction..."

    # Define scan folder path (contains scan.xml with geometry info)
    # data/results is a SIBLING of Base_model_comparison, not inside it
    # $PROJECT = .../ct_recon/Base_model_comparison
    # scan.xml is at .../ct_recon/data/results/Scan_1681_uwarp_gt/scan.xml
    SCAN_FOLDER="${PROJECT%/Base_model_comparison}/data/results/Scan_1681_uwarp_gt"

    # Verify scan.xml exists before reconstruction
    if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
        echo "ERROR: scan.xml not found at: ${SCAN_FOLDER}/scan.xml"
        echo "This file is required for FDK reconstruction."
        exit 1
    fi
    echo "Scan folder verified: ${SCAN_FOLDER}"

    python3 reconstruct_from_mat.py \
        --sinogram_dir mat/sinograms_infilled \
        --output_dir mat/reconstructed_volume \
        --metadata_path sinogram_dataset/metadata.json \
        --tiling_metadata_path repaint/sinogram_tiles/tiling_metadata.json \
        --scan_folder "${SCAN_FOLDER}"

    # Verify the .vff file was created
    if [ -f "$PROJECT/mat/reconstructed_volume.vff" ]; then
        VFF_SIZE=$(du -h "$PROJECT/mat/reconstructed_volume.vff" | cut -f1)
        echo "Reconstruction .vff file created: $PROJECT/mat/reconstructed_volume.vff ($VFF_SIZE)"
    else
        echo "WARNING: .vff file not found at $PROJECT/mat/reconstructed_volume.vff"
        echo "Checking for .vff files..."
        find $PROJECT/mat -name "*.vff" -ls
    fi

    echo ""
    echo "========================================================================"
    echo "PIPELINE COMPLETE!"
    echo "========================================================================"
    echo "Output files:"
    echo "  Infilled tiles: $PROJECT/mat/tiles_infilled/"
    echo "  Merged sinograms: $PROJECT/mat/sinograms_infilled/"
    echo "  Reconstruction: $PROJECT/mat/reconstructed_volume.vff"
fi

echo ""
echo "Task ${TASK_ID} finished at: $(date)"
echo "========================================================================"
