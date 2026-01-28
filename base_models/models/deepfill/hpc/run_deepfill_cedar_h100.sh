#!/bin/bash
#SBATCH --time=01-00:00:00              # 1 day (DeepFill is faster than RePaint)
#SBATCH --job-name=deepfill_h100
#SBATCH --output=logs/deepfill_%A_%a.out
#SBATCH --error=logs/deepfill_%A_%a.err
#SBATCH --array=0-3                      # 4 parallel jobs (one per GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1               # 1 H100 per array job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # More CPUs for data loading
#SBATCH --mem=64G                        # 64GB RAM
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

set -euo pipefail

#==============================================================================
# DeepFill v2 Sinogram Infilling on Compute Canada - H100 Optimized
# Processes 73,472 tiles across 4 H100 GPUs in parallel
# Expected runtime: 2-4 hours (DeepFill is much faster than diffusion models)
# Works on: Cedar, Fir, or any cluster with H100 GPUs
#
# WORKFLOW (optimized - no tile creation, all workers start immediately):
# 1. All tasks copy RePaint tiles to local NVMe (in parallel)
# 2. All tasks run inference in parallel on their subset of tiles
#    - Masks are inverted on-the-fly (no separate DeepFill mask files needed)
# 3. Task 0 waits for all tasks, then runs merge + reconstruction
#
# This approach eliminates ~73,000 extra mask files from disk!
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
echo "DeepFill v2 H100 Array Job ${TASK_ID}/${N_TASKS}"
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
echo "Installing additional packages..."
PACKAGES="numpy pillow tqdm scipy pyyaml"
for pkg in $PACKAGES; do
    echo "  Installing $pkg..."
    pip install --no-index $pkg 2>/dev/null || pip install $pkg || {
        echo "ERROR: Failed to install $pkg"
        exit 1
    }
done

# xmltodict may not have a pre-built wheel
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
# VERIFY REPAINT TILES (no separate DeepFill tiles needed - uses on-the-fly mask inversion)
#==============================================================================

echo ""
echo "========================================================================"
echo "Verifying RePaint tiles (will use directly with on-the-fly mask inversion)..."
echo "========================================================================"

# Check if RePaint tiles exist (required source)
if [ ! -d "$PROJECT/repaint/sinogram_tiles/sinograms_gt" ]; then
    echo "ERROR: RePaint tiles not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/sinograms_gt"
    echo ""
    echo "Please ensure RePaint tiles are transferred to Compute Canada first."
    exit 1
fi

REPAINT_GT_COUNT=$(find "$PROJECT/repaint/sinogram_tiles/sinograms_gt" -name "*.png" 2>/dev/null | wc -l)
REPAINT_MASK_COUNT=$(find "$PROJECT/repaint/sinogram_tiles/masks" -name "*.png" 2>/dev/null | wc -l)
echo "Found RePaint tiles: GT=${REPAINT_GT_COUNT}, Masks=${REPAINT_MASK_COUNT}"

if [ $REPAINT_GT_COUNT -lt 70000 ]; then
    echo "ERROR: Not enough RePaint tiles. Expected ~73,472, found ${REPAINT_GT_COUNT}"
    exit 1
fi

echo ""
echo "NOTE: Using on-the-fly mask inversion - no separate DeepFill tiles needed!"
echo "      This saves ~73,000 files on disk."

#==============================================================================
# DATA TRANSFER: Copy RePaint tiles to local NVMe for fast I/O
#==============================================================================

echo ""
echo "========================================================================"
echo "Transferring RePaint tiles to local NVMe storage..."
echo "========================================================================"

# Create local directories
mkdir -p repaint/sinogram_tiles/{sinograms_gt,masks}
mkdir -p deepfill/tiles_infilled

# Copy GT tiles and masks from RePaint (masks will be inverted on-the-fly during inference)
echo "Copying GT tiles from RePaint to local NVMe..."
rsync -av "$PROJECT/repaint/sinogram_tiles/sinograms_gt/" repaint/sinogram_tiles/sinograms_gt/

echo "Copying mask tiles from RePaint to local NVMe..."
rsync -av "$PROJECT/repaint/sinogram_tiles/masks/" repaint/sinogram_tiles/masks/

# Copy metadata
if [ -f "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" ]; then
    cp "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" repaint/sinogram_tiles/
fi

# Verify tiles copied successfully
GT_COPIED=$(find repaint/sinogram_tiles/sinograms_gt -name "*.png" 2>/dev/null | wc -l)
MASK_COPIED=$(find repaint/sinogram_tiles/masks -name "*.png" 2>/dev/null | wc -l)

echo "RePaint tiles copied to local NVMe: GT=${GT_COPIED}, Masks=${MASK_COPIED}"

if [ $GT_COPIED -lt 70000 ]; then
    echo "ERROR: GT tile copy failed - expected ~73,472 tiles, got ${GT_COPIED}"
    exit 1
fi

if [ $MASK_COPIED -lt 70000 ]; then
    echo "ERROR: Mask tile copy failed - expected ~73,472 tiles, got ${MASK_COPIED}"
    exit 1
fi

#==============================================================================
# COPY MODEL WEIGHTS
#==============================================================================

echo ""
echo "Copying DeepFill v2 model and weights..."

mkdir -p deepfill/DeepFillv2/pretrained
mkdir -p deepfill/DeepFillv2/model

# Copy model code
if [ -d "$PROJECT/deepfill/DeepFillv2/model" ]; then
    cp -r $PROJECT/deepfill/DeepFillv2/model/* deepfill/DeepFillv2/model/
else
    echo "ERROR: DeepFill v2 model code not found at:"
    echo "  $PROJECT/deepfill/DeepFillv2/model"
    echo ""
    echo "Please ensure the DeepFill v2 repository is cloned on Compute Canada."
    exit 1
fi

# Copy weights
if [ -f "$PROJECT/deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth" ]; then
    cp $PROJECT/deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth deepfill/DeepFillv2/pretrained/
else
    echo "ERROR: DeepFill v2 weights not found at:"
    echo "  $PROJECT/deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth"
    echo ""
    echo "Please download the CelebA-HQ weights first."
    exit 1
fi

echo "Model and weights copied successfully"

#==============================================================================
# CALCULATE TILE DISTRIBUTION
#==============================================================================

TOTAL_TILES=$GT_COPIED
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
# CHECK IF TILES ALREADY EXIST (SMART RESUME)
#==============================================================================

echo ""
echo "========================================================================"
echo "Checking for existing tiles (smart resume)..."
echo "========================================================================"

# Check if tiles already exist in project directory
EXISTING_TILES=$(find $PROJECT/deepfill/tiles_infilled -name "*.png" 2>/dev/null | wc -l)

echo "Found ${EXISTING_TILES}/${TOTAL_TILES} tiles in project directory"

SKIP_INFERENCE=0

if [ $EXISTING_TILES -ge $TOTAL_TILES ]; then
    echo ""
    echo "✓ All tiles already exist in project directory!"
    echo "  Skipping inference to save time..."
    echo "  (Delete $PROJECT/deepfill/tiles_infilled/ to force re-inference)"
    SKIP_INFERENCE=1
elif [ $EXISTING_TILES -gt 0 ]; then
    echo ""
    echo "⚠ Found partial tiles (${EXISTING_TILES}/${TOTAL_TILES})"
    echo "  Will run inference for missing tiles..."
fi

#==============================================================================
# RUN DEEPFILL v2 INFERENCE (if needed)
#==============================================================================

if [ $SKIP_INFERENCE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Starting DeepFill v2 inference..."
    echo "========================================================================"

    # Run inference with batch processing
    # Uses --invert_masks to invert RePaint masks on-the-fly (0=inpaint -> 255=inpaint)
    python3 run_deepfill_inference.py \
        --gt_dir repaint/sinogram_tiles/sinograms_gt \
        --mask_dir repaint/sinogram_tiles/masks \
        --output_dir deepfill/tiles_infilled \
        --checkpoint deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth \
        --batch_size 32 \
        --device cuda \
        --start_idx ${START_IDX} \
        --end_idx ${END_IDX} \
        --invert_masks \
        --save_grayscale

    INFERENCE_STATUS=$?

    if [ $INFERENCE_STATUS -ne 0 ]; then
        echo "ERROR: Inference failed with status ${INFERENCE_STATUS}"
        exit 1
    fi

    echo ""
    echo "Inference complete for Task ${TASK_ID}!"

    #==========================================================================
    # COPY RESULTS BACK TO PROJECT DIRECTORY
    #==========================================================================

    echo ""
    echo "========================================================================"
    echo "Copying results back to project directory..."
    echo "========================================================================"

    # Copy infilled tiles back
    mkdir -p $PROJECT/deepfill/tiles_infilled
    rsync -av deepfill/tiles_infilled/ $PROJECT/deepfill/tiles_infilled/

    TILES_PRODUCED=$(find deepfill/tiles_infilled -name "*.png" 2>/dev/null | wc -l)
    echo "Tiles produced by Task ${TASK_ID}: ${TILES_PRODUCED}"
else
    echo ""
    echo "========================================================================"
    echo "Inference skipped - using existing tiles"
    echo "========================================================================"
    TILES_PRODUCED=$EXISTING_TILES
fi

#==============================================================================
# CHECK IF ALL TASKS COMPLETE (only Task 0 does final merge)
#==============================================================================

if [ $TASK_ID -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Task 0: Waiting for all tasks to complete, then merge..."
    echo "========================================================================"

    # If inference was skipped, no need to wait
    if [ $SKIP_INFERENCE -eq 1 ]; then
        echo "All tiles already available, skipping wait..."
    else
        # Wait for all tiles to be produced
        EXPECTED_TILES=$TOTAL_TILES
        MAX_WAIT=14400  # 4 hours max wait for inference
        WAIT_TIME=0

        while [ $WAIT_TIME -lt $MAX_WAIT ]; do
            CURRENT_TILES=$(find $PROJECT/deepfill/tiles_infilled -name "*.png" 2>/dev/null | wc -l)

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
            echo "  Current tiles: $(find $PROJECT/deepfill/tiles_infilled -name '*.png' 2>/dev/null | wc -l)"
        fi
    fi

    # Check if sinograms already exist (allows re-running with different blend mode)
    echo ""
    echo "Checking for existing merged sinograms..."
    EXISTING_SINOS=$(find $PROJECT/deepfill/sinograms_infilled -name "*.png" 2>/dev/null | wc -l)

    if [ $EXISTING_SINOS -gt 0 ]; then
        echo ""
        echo "⚠ Found ${EXISTING_SINOS} existing sinograms in deepfill/sinograms_infilled/"
        echo "  To re-merge with different blend mode, delete this directory first:"
        echo "  rm -rf $PROJECT/deepfill/sinograms_infilled"
        echo ""
        echo "  Skipping merge and using existing sinograms..."
        SKIP_MERGE=1
    else
        SKIP_MERGE=0
    fi

    # Run merge from project directory
    if [ $SKIP_MERGE -eq 0 ]; then
        echo ""
        echo "Merging tiles with nearest blend mode..."
        cd $PROJECT
        python3 merge_deepfill_tiles.py \
            --tiles_dir deepfill/tiles_infilled \
            --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
            --output_dir deepfill/sinograms_infilled \
            --blend_mode nearest
    fi

    echo ""
    echo "Running reconstruction..."
    cd $PROJECT
    python3 reconstruct_from_deepfill.py \
        --sinogram_dir deepfill/sinograms_infilled \
        --output_dir deepfill/reconstructed_volume

    echo ""
    echo "========================================================================"
    echo "PIPELINE COMPLETE!"
    echo "========================================================================"
    echo "Output files:"
    echo "  Infilled tiles: $PROJECT/deepfill/tiles_infilled/"
    echo "  Merged sinograms: $PROJECT/deepfill/sinograms_infilled/"
    echo "  Reconstruction: $PROJECT/deepfill/reconstructed_volume/"
fi

echo ""
echo "Task ${TASK_ID} finished at: $(date)"
echo "========================================================================"
