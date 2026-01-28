#!/bin/bash
#SBATCH --time=02-12:00:00              # 2.5 days (conservative)
#SBATCH --job-name=repaint_h100
#SBATCH --output=logs/repaint_%A_%a.out
#SBATCH --error=logs/repaint_%A_%a.err
#SBATCH --array=0-3                      # 4 parallel jobs (one per GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1               # 1 H100 per array job (adjust for your cluster)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # More CPUs for H100
#SBATCH --mem=64G                        # 64GB per GPU (H100 has 80GB VRAM)
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

set -euo pipefail

#==============================================================================
# RePaint Sinogram Infilling on Compute Canada - H100 Optimized
# Processes 73,472 tiles across 4 H100 GPUs in parallel
# Expected runtime: 1.0-1.5 days (allocated 2 days for safety)
# Works on: Cedar, Fir, or any cluster with H100 GPUs
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

echo "========================================================================"
echo "RePaint H100 Array Job ${TASK_ID}/3"
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

echo ""
echo "Loading modules..."
module purge

# Load modules with error checking
for mod in cuda/12.2 cudnn/8.9 python/3.10 scipy-stack; do
    echo "  Loading $mod..."
    if ! module load $mod 2>&1; then
        echo "ERROR: Failed to load module: $mod"
        echo "Available modules:"
        module avail ${mod%%/*} 2>&1 | head -20
        exit 1
    fi
done

echo "✓ All modules loaded successfully"

echo ""
echo "Activating virtual environment..."
if [ ! -f "$HOME/Python_virtual_env/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at:"
    echo "  $HOME/Python_virtual_env/bin/activate"
    exit 1
fi

source ~/Python_virtual_env/bin/activate || {
    echo "ERROR: Failed to activate virtual environment"
    exit 1
}

echo "✓ Virtual environment activated"

echo ""
echo "Environment info:"
echo "  Host: $(hostname)"
echo "  Python: $(python --version)"
echo "  GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || nvidia-smi -L

#==============================================================================
# DATA TRANSFER: Copy tiles from project directory
#==============================================================================

echo ""
echo "========================================================================"
echo "Transferring tile data..."
echo "========================================================================"

# Create directories (NO splits directory - we use start_idx/end_idx instead)
mkdir -p repaint/sinogram_tiles/{sinograms_masked,masks}

# Validate tiles exist before copying (check masked sinograms, NOT gt)
if [ ! -d "$PROJECT/repaint/sinogram_tiles/sinograms_masked" ]; then
    echo "ERROR: Masked tiles not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/sinograms_masked"
    echo ""
    echo "Please run create_repaint_tiles.py first or transfer via Globus"
    exit 1
fi

if [ ! -d "$PROJECT/repaint/sinogram_tiles/masks" ]; then
    echo "ERROR: Masks not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/masks"
    exit 1
fi

# Copy ONLY sinograms_masked and masks (NOT sinograms_gt - saves 73k files)
# Note: sinograms_gt is only needed for evaluation, not inference
echo "Copying tiles from project directory (masked + masks only)..."
rsync -av --progress $PROJECT/repaint/sinogram_tiles/sinograms_masked/ repaint/sinogram_tiles/sinograms_masked/
rsync -av --progress $PROJECT/repaint/sinogram_tiles/masks/ repaint/sinogram_tiles/masks/

# Verify tiles copied successfully
MASKED_COPIED=$(find repaint/sinogram_tiles/sinograms_masked -name "*.png" 2>/dev/null | wc -l)
MASK_COPIED=$(find repaint/sinogram_tiles/masks -name "*.png" 2>/dev/null | wc -l)

echo "Tiles copied: Masked=${MASKED_COPIED}, Masks=${MASK_COPIED}"

if [ $MASKED_COPIED -lt 70000 ]; then
    echo "ERROR: Tile copy failed - expected ~73,472 tiles, got ${MASKED_COPIED}"
    exit 1
fi

# Copy metadata
if [ ! -f "$PROJECT/repaint/sinogram_tiles/tiling_metadata.json" ]; then
    echo "ERROR: Tiling metadata not found at:"
    echo "  $PROJECT/repaint/sinogram_tiles/tiling_metadata.json"
    echo "This file should have been created by create_repaint_tiles.py"
    exit 1
fi

cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/

echo "✓ Metadata copied"

# Create RePaint directory structure
# CRITICAL: repaint/ is in .gitignore, so it won't exist after git clone!
# We need to create all necessary subdirectories before copying files
mkdir -p repaint/RePaint/data/pretrained
mkdir -p repaint/RePaint/confs
mkdir -p repaint/RePaint/guided_diffusion
mkdir -p repaint/RePaint/utils
mkdir -p repaint/RePaint/conf_mgt

echo "Copying RePaint code and model weights..."

# Validate RePaint exists in project directory
if [ ! -d "$PROJECT/repaint/RePaint" ]; then
    echo "ERROR: RePaint directory not found at:"
    echo "  $PROJECT/repaint/RePaint"
    echo "Please ensure RePaint is cloned/installed"
    exit 1
fi

# Copy Python code files (guided_diffusion/, utils/, conf_mgt/, etc.)
rsync -av \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data/pretrained/*' \
    --exclude='test.py' \
    $PROJECT/repaint/RePaint/ repaint/RePaint/

echo "✓ RePaint code copied"

# Copy version-controlled inference script to RePaint directory
# CRITICAL: test.py needs to be in repaint/RePaint/ to import conf_mgt, utils, guided_diffusion
cp run_repaint_inference.py repaint/RePaint/test.py || {
    echo "ERROR: Failed to copy run_repaint_inference.py to repaint/RePaint/test.py"
    echo "Current directory: $(pwd)"
    ls -la run_repaint_inference.py
    exit 1
}

echo "✓ Inference script copied from version control (run_repaint_inference.py → test.py)"

# Copy model weights separately (large file)
if [ ! -f "$PROJECT/repaint/RePaint/data/pretrained/celeba256_250000.pt" ]; then
    echo "ERROR: Pretrained model not found at:"
    echo "  $PROJECT/repaint/RePaint/data/pretrained/celeba256_250000.pt"
    echo ""
    echo "Please download the model first using:"
    echo "  cd $PROJECT/repaint/RePaint"
    echo "  bash download.sh"
    exit 1
fi

rsync -av $PROJECT/repaint/RePaint/data/pretrained/ repaint/RePaint/data/pretrained/

# Verify model was copied
if [ ! -f "repaint/RePaint/data/pretrained/celeba256_250000.pt" ]; then
    echo "ERROR: Model copy failed!"
    exit 1
fi

MODEL_SIZE=$(stat -c%s "repaint/RePaint/data/pretrained/celeba256_250000.pt")
echo "Model copied successfully ($(numfmt --to=iec-i --suffix=B $MODEL_SIZE))"

# Copy H100-optimized config
if [ ! -f "$PROJECT/configs/repaint_sinogram_h100.yml" ]; then
    echo "ERROR: H100 config not found at:"
    echo "  $PROJECT/configs/repaint_sinogram_h100.yml"
    exit 1
fi

cp $PROJECT/configs/repaint_sinogram_h100.yml repaint/RePaint/confs/repaint_sinogram.yml

# Verify config was copied
if [ ! -f "repaint/RePaint/confs/repaint_sinogram.yml" ]; then
    echo "ERROR: Config copy failed!"
    exit 1
fi

echo "✓ Configuration copied successfully"

echo ""
echo "Data transfer complete!"
echo "Disk usage:"
du -sh repaint/

#==============================================================================
# TILE DISTRIBUTION: Calculate range for this task (NO FILE DUPLICATION!)
#==============================================================================

echo ""
echo "========================================================================"
echo "Determining tile assignment for task ${TASK_ID}..."
echo "========================================================================"

# IMPORTANT: We use start_idx/end_idx in the dataloader config instead of copying files!
# This eliminates file duplication and saves ~200k files across 4 GPUs.

# Count tiles from the masked sinograms directory (this is our input)
TOTAL_TILES=$(find repaint/sinogram_tiles/sinograms_masked -name "*.png" | wc -l)

echo "Total tiles available: ${TOTAL_TILES}"

# Calculate tile range for this task
TILES_PER_TASK=$((TOTAL_TILES / 4))
START_IDX=$((TASK_ID * TILES_PER_TASK))

if [ ${TASK_ID} -eq 3 ]; then
    # Last task gets any remainder tiles
    END_IDX=${TOTAL_TILES}
else
    END_IDX=$((START_IDX + TILES_PER_TASK))
fi

NUM_MY_TILES=$((END_IDX - START_IDX))

echo ""
echo "Task assignment (using start_idx/end_idx - NO file copying!):"
echo "  Task ${TASK_ID} processes tiles: ${START_IDX} to $((END_IDX - 1))"
echo "  Tile count for this task: ${NUM_MY_TILES}"
echo ""
echo "All task assignments:"
echo "  Task 0: tiles 0-$((TILES_PER_TASK - 1)) (${TILES_PER_TASK} tiles)"
echo "  Task 1: tiles ${TILES_PER_TASK}-$((2 * TILES_PER_TASK - 1)) (${TILES_PER_TASK} tiles)"
echo "  Task 2: tiles $((2 * TILES_PER_TASK))-$((3 * TILES_PER_TASK - 1)) (${TILES_PER_TASK} tiles)"
echo "  Task 3: tiles $((3 * TILES_PER_TASK))-$((TOTAL_TILES - 1)) ($((TOTAL_TILES - 3 * TILES_PER_TASK)) tiles)"
echo ""
echo "NOTE: All GPUs read from the SAME tile directory (repaint/sinogram_tiles/)."
echo "      Each GPU processes only its assigned range using start_idx=${START_IDX}, end_idx=${END_IDX}."
echo "      This eliminates file duplication and saves ~200k files!"

# Use the shared tile directory directly (no separate my_tiles directory needed!)
TILE_DIR="repaint/sinogram_tiles"

echo ""
echo "✓ Tile range calculated for task ${TASK_ID}: [${START_IDX}, ${END_IDX})"

#==============================================================================
# PROCESSING: Run RePaint on this task's tiles (H100 optimized)
#==============================================================================

echo ""
echo "========================================================================"
echo "Processing task ${TASK_ID} on H100"
echo "========================================================================"
echo "Tile directory: ${TILE_DIR}"
echo "Tiles to process: ${NUM_MY_TILES}"
echo ""
echo "IMPORTANT: Each array task processes DIFFERENT tiles independently:"
echo "  Task 0 → tiles 0-18367 (~18,368 tiles)"
echo "  Task 1 → tiles 18368-36735 (~18,368 tiles)"
echo "  Task 2 → tiles 36736-55103 (~18,368 tiles)"
echo "  Task 3 → tiles 55104-73471 (~18,368 tiles)"
echo ""
echo "THIS task (${TASK_ID}) processes: ${START_IDX} to $((END_IDX - 1)) (${NUM_MY_TILES} tiles)"
echo ""

echo "Expected time: ~24-36 hours with H100"

# Create output directory for this task (task-specific naming)
OUTPUT_DIR="repaint/tiles_infilled_task_${TASK_ID}"
mkdir -p ${OUTPUT_DIR}

# Create logs directory in working dir
mkdir -p logs

# Create project output directory (auto-saved hourly via checkpoint)
PROJECT_OUTPUT_DIR="$PROJECT/repaint/tiles_infilled_task_${TASK_ID}"
mkdir -p ${PROJECT_OUTPUT_DIR}

#==============================================================================
# PERIODIC CHECKPOINT: Transfer tiles every hour to prevent data loss
#==============================================================================

echo ""
echo "Setting up hourly checkpointing..."

# Background function to periodically save progress
checkpoint_tiles() {
    local output_dir=$1
    local project_dir=$2
    local checkpoint_interval=3600  # 1 hour in seconds

    while true; do
        sleep ${checkpoint_interval}

        # Count tiles before transfer
        local tile_count=$(find ${output_dir} -name "*.png" 2>/dev/null | wc -l)

        if [ ${tile_count} -gt 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint: Transferring ${tile_count} completed tiles..."

            # Rsync only new/modified files (very efficient)
            rsync -a --info=progress2 ${output_dir}/ ${project_dir}/ 2>&1 | head -5

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint complete: ${tile_count} tiles backed up"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint: No tiles yet, waiting..."
        fi
    done
}

# Start checkpoint process in background
checkpoint_tiles "${OUTPUT_DIR}" "${PROJECT_OUTPUT_DIR}" &
CHECKPOINT_PID=$!

echo "✓ Checkpoint process started (PID: ${CHECKPOINT_PID})"
echo "  - Backing up to: ${PROJECT_OUTPUT_DIR}"
echo "  - Frequency: Every 1 hour"
echo "  - Progress will be saved automatically!"

# Ensure checkpoint process is killed on exit/error
trap "echo 'Stopping checkpoint process...'; kill ${CHECKPOINT_PID} 2>/dev/null || true" EXIT INT TERM

#==============================================================================
# RESUME SUPPORT: Sync any existing outputs from project directory
#==============================================================================

echo ""
echo "========================================================================"
echo "Checking for existing outputs (resume support)..."
echo "========================================================================"

# If there are already completed tiles in the project directory, sync them
# This allows the inference to skip already-completed tiles
if [ -d "${PROJECT_OUTPUT_DIR}" ]; then
    EXISTING_TILES=$(find ${PROJECT_OUTPUT_DIR} -name "*.png" 2>/dev/null | wc -l)
    if [ ${EXISTING_TILES} -gt 0 ]; then
        echo "Found ${EXISTING_TILES} existing tiles in project directory"
        echo "Syncing to local storage for resume..."
        rsync -av ${PROJECT_OUTPUT_DIR}/ ${OUTPUT_DIR}/
        echo "✓ Synced ${EXISTING_TILES} tiles - these will be SKIPPED during inference"
    else
        echo "No existing tiles found - starting fresh"
    fi
else
    echo "No previous output directory - starting fresh"
fi

#==============================================================================
# REPAINT PROCESSING
#==============================================================================

echo ""
echo "Starting RePaint inference with H100 optimization..."
echo "Batch size: 32 (optimal for H100 80GB)"
echo "Resume mode: ENABLED (will skip already-completed tiles)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

# Change to RePaint directory
cd repaint/RePaint || {
    echo "ERROR: Failed to change to repaint/RePaint directory"
    exit 1
}

# Validate critical files and modules exist
echo "Validating RePaint installation..."

# Check inference script
if [ ! -f "test.py" ]; then
    echo "ERROR: test.py not found in $(pwd)"
    echo "Should have been copied from run_repaint_inference.py during setup"
    exit 1
fi
echo "✓ test.py exists"

# Check required Python modules
REQUIRED_MODULES=("conf_mgt" "utils" "guided_diffusion")
for module in "${REQUIRED_MODULES[@]}"; do
    if [ ! -d "${module}" ]; then
        echo "ERROR: Required module '${module}' not found in $(pwd)"
        echo "Contents of directory:"
        ls -la
        exit 1
    fi
    echo "✓ ${module}/ exists"
done

# Check config file
if [ ! -f "confs/repaint_sinogram.yml" ]; then
    echo "ERROR: Config file not found at confs/repaint_sinogram.yml"
    echo "Contents of confs/:"
    ls -la confs/ || echo "  Directory doesn't exist"
    exit 1
fi
echo "✓ confs/repaint_sinogram.yml exists"

# Check model weights
if [ ! -f "data/pretrained/celeba256_250000.pt" ]; then
    echo "ERROR: Model weights not found at data/pretrained/celeba256_250000.pt"
    exit 1
fi
echo "✓ data/pretrained/celeba256_250000.pt exists"

echo "✓ All RePaint files and modules validated"
echo ""

# System health check before starting long inference
echo "System health check:"
echo "  - GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "    Could not query GPU"
echo "  - Shared memory (/dev/shm):"
df -h /dev/shm 2>/dev/null || echo "    Could not query /dev/shm"
echo "  - Local storage:"
df -h $SLURM_TMPDIR 2>/dev/null || df -h . || echo "    Could not query storage"
echo ""

# Generate task-specific config with correct paths and tile range
echo "Generating task-specific config..."
TASK_CONFIG="confs/repaint_sinogram_task${TASK_ID}.yml"

# Create task-specific config by modifying paths in base config
# IMPORTANT: Uses sinograms_masked (NOT sinograms_gt!) and start_idx/end_idx for range selection
python3 -c "
import yaml
import sys

# Load base config
with open('confs/repaint_sinogram.yml', 'r') as f:
    config = yaml.safe_load(f)

# Update paths for this task
# IMPORTANT: Use sinograms_masked as input, NOT sinograms_gt!
# sinograms_masked contains the input with metal regions zeroed out
# sinograms_gt contains the ground truth - NOT used for inference
config['data']['eval']['sinogram_tiles']['gt_path'] = '../../${TILE_DIR}/sinograms_masked'
config['data']['eval']['sinogram_tiles']['mask_path'] = '../../${TILE_DIR}/masks'
config['data']['eval']['sinogram_tiles']['paths']['srs'] = '../../${OUTPUT_DIR}'

# Add tile range for this task (NO FILE DUPLICATION!)
# Each GPU processes only its assigned range from the shared directory
config['data']['eval']['sinogram_tiles']['start_idx'] = ${START_IDX}
config['data']['eval']['sinogram_tiles']['end_idx'] = ${END_IDX}

# Save task-specific config
with open('${TASK_CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print('✓ Task-specific config created')
print('  - Using sinograms_masked (NOT sinograms_gt) as input')
print('  - Using start_idx/end_idx for range selection (NO file duplication)')
sys.exit(0)
"

CONFIG_EXIT=$?
if [ $CONFIG_EXIT -ne 0 ]; then
    echo "ERROR: Failed to create task-specific config (exit code: ${CONFIG_EXIT})"
    exit 1
fi

# Verify config was created
if [ ! -f "${TASK_CONFIG}" ]; then
    echo "ERROR: Config file not found: ${TASK_CONFIG}"
    exit 1
fi

echo "✓ Task-specific config: ${TASK_CONFIG}"
echo "  - gt_path: ../../${TILE_DIR}/sinograms_masked (CORRECT: using masked input)"
echo "  - mask_path: ../../${TILE_DIR}/masks"
echo "  - output_path: ../../${OUTPUT_DIR}"
echo "  - start_idx: ${START_IDX}"
echo "  - end_idx: ${END_IDX}"
echo "  - Tiles to process: ${NUM_MY_TILES}"

# Run RePaint with task-specific config (no additional command-line arguments)
# Note: test.py was copied from version-controlled run_repaint_inference.py
echo ""
echo "Running RePaint with task-specific config..."
CUDA_VISIBLE_DEVICES=0 python test.py \
    --conf_path ${TASK_CONFIG} \
    2>&1 | tee ../../logs/repaint_task_${TASK_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "RePaint inference completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: RePaint failed with exit code ${EXIT_CODE}"
    echo "Check logs/repaint_task_${TASK_ID}.log for details"
    exit ${EXIT_CODE}
fi

cd ../..

# Verify output
NUM_OUTPUT=$(find ${OUTPUT_DIR} -name "*.png" | wc -l)
echo "Output tiles generated: ${NUM_OUTPUT}"

if [ ${NUM_OUTPUT} -lt ${NUM_MY_TILES} ]; then
    echo "WARNING: Expected ${NUM_MY_TILES} tiles, but only generated ${NUM_OUTPUT}"
fi

#==============================================================================
# FINAL CHECKPOINT: Transfer any remaining tiles
#==============================================================================

echo ""
echo "========================================================================"
echo "Final checkpoint: Transferring any remaining tiles..."
echo "========================================================================"

# Stop the hourly checkpoint process (we'll do final sync manually)
kill ${CHECKPOINT_PID} 2>/dev/null || true
wait ${CHECKPOINT_PID} 2>/dev/null || true

# Final comprehensive sync (catches any files missed by hourly checkpoints)
echo "Performing final comprehensive sync..."
rsync -av --progress ${OUTPUT_DIR}/ ${PROJECT_OUTPUT_DIR}/

# Verify final count in project directory
NUM_TRANSFERRED=$(find ${PROJECT_OUTPUT_DIR} -name "*.png" 2>/dev/null | wc -l)
echo "Tiles in project directory: ${NUM_TRANSFERRED}"

if [ ${NUM_TRANSFERRED} -eq ${NUM_OUTPUT} ]; then
    echo "✓ All tiles successfully backed up!"
elif [ ${NUM_TRANSFERRED} -lt ${NUM_OUTPUT} ]; then
    echo "⚠ WARNING: Only ${NUM_TRANSFERRED}/${NUM_OUTPUT} tiles in project directory"
    echo "  Attempting one more sync..."
    rsync -av ${OUTPUT_DIR}/ ${PROJECT_OUTPUT_DIR}/
else
    echo "✓ Transfer verified: ${NUM_TRANSFERRED} tiles"
fi

# Copy logs
cp logs/repaint_task_${TASK_ID}.log $PROJECT/logs/ 2>/dev/null || true

echo "Transfer complete!"

#==============================================================================
# CLEANUP AND SUMMARY
#==============================================================================

echo ""
echo "========================================================================"
echo "Array job ${TASK_ID} completed successfully!"
echo "========================================================================"
echo "End time: $(date)"
echo "Tiles processed: ${NUM_MY_TILES}"
echo "Output directory: $PROJECT/repaint/tiles_infilled_task_${TASK_ID}"
echo "========================================================================"

echo ""
echo "NOTE: Once all 4 array jobs complete, merge results with:"
echo "  All output directories are already in $PROJECT/repaint/"
echo "  tiles_infilled_task_0/"
echo "  tiles_infilled_task_1/"
echo "  tiles_infilled_task_2/"
echo "  tiles_infilled_task_3/"
echo ""
echo "  Merge them into a single directory for reconstruction."
