#!/bin/bash
#SBATCH --time=03:00:00                 # 3 hours for merging and reconstruction
#SBATCH --job-name=repaint_merge_h100
#SBATCH --output=logs/repaint_merge_%j.out
#SBATCH --error=logs/repaint_merge_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16              # More CPUs for faster processing
#SBATCH --mem=128G                      # 128GB for large tile merging
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=ALL

set -euo pipefail

#==============================================================================
# Merge RePaint H100 Results and Reconstruct CT Volume
# =====================================================
# This script:
# 1. Combines tiles from 4 H100 array jobs (task_0, task_1, task_2, task_3)
# 2. Merges 256×256 tiles back to full-resolution sinograms
# 3. Runs FDK reconstruction to create CT volume
# 4. Transfers final results to project directory
#
# Expected runtime: 1.5-2 hours
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
echo "RePaint H100 Results Merger & Reconstruction"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================================================"

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

echo ""
echo "Loading modules..."
module purge
module load python/3.10
module load scipy-stack

echo "Activating virtual environment..."
source ~/Python_virtual_env/bin/activate

#==============================================================================
# STEP 1: COMBINE TILES FROM ALL TASKS
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Combining tiles from 4 H100 tasks"
echo "========================================================================"

# Create combined output directory
mkdir -p repaint/tiles_infilled

# Transfer and combine tiles from all 4 tasks
TOTAL_TILES_TRANSFERRED=0

for TASK_ID in {0..3}; do
    echo ""
    echo "Processing Task ${TASK_ID}..."

    TASK_DIR="$PROJECT/repaint/tiles_infilled_task_${TASK_ID}"

    if [ ! -d "${TASK_DIR}" ]; then
        echo "WARNING: Task ${TASK_ID} directory not found: ${TASK_DIR}"
        echo "Skipping this task..."
        continue
    fi

    # Count tiles before transfer
    TASK_TILES=$(find ${TASK_DIR} -name "*.png" 2>/dev/null | wc -l)
    echo "  Task ${TASK_ID}: ${TASK_TILES} tiles found"

    # Transfer tiles from this task
    echo "  Transferring tiles from task ${TASK_ID}..."
    rsync -av --info=progress2 ${TASK_DIR}/ repaint/tiles_infilled/ 2>&1 | tail -10

    TOTAL_TILES_TRANSFERRED=$((TOTAL_TILES_TRANSFERRED + TASK_TILES))
done

echo ""
echo "Transfer complete!"
echo "Total tiles combined: ${TOTAL_TILES_TRANSFERRED}"

# Verify we got all expected tiles
EXPECTED_TILES=73472
if [ ${TOTAL_TILES_TRANSFERRED} -ne ${EXPECTED_TILES} ]; then
    echo ""
    echo "WARNING: Expected ${EXPECTED_TILES} tiles but got ${TOTAL_TILES_TRANSFERRED}"
    echo "This might be OK if some tasks are still running..."
    echo "Proceeding with available tiles..."
fi

# Copy metadata
echo ""
echo "Copying metadata files..."
mkdir -p repaint/sinogram_tiles
cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/
echo "✓ Tiling metadata copied"

# Verify dataset metadata exists (should be in git)
if [ ! -f "sinogram_dataset/metadata.json" ]; then
    echo "ERROR: sinogram_dataset/metadata.json not found in git repository!"
    echo "This file is required for reconstruction. Please ensure it's committed to git."
    exit 1
fi
echo "✓ Dataset metadata found in git repository"

#==============================================================================
# STEP 2: MERGE TILES BACK TO FULL SINOGRAMS
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Merging tiles back to full-resolution sinograms"
echo "========================================================================"

# Check if merged sinograms already exist in project directory (from previous run)
if [ -d "$PROJECT/repaint/sinograms_infilled" ]; then
    EXISTING_SINOS=$(find $PROJECT/repaint/sinograms_infilled -name "*.png" 2>/dev/null | wc -l)
    if [ ${EXISTING_SINOS} -eq 2296 ]; then
        echo "✓ Found existing merged sinograms in project directory (${EXISTING_SINOS} files)"
        echo "  Reusing from previous run to save time..."
        mkdir -p repaint/sinograms_infilled
        rsync -av --info=progress2 $PROJECT/repaint/sinograms_infilled/ repaint/sinograms_infilled/ 2>&1 | tail -10
        echo "✓ Merged sinograms loaded from checkpoint"
        NUM_SINOS=${EXISTING_SINOS}
        MERGE_DURATION=0
    else
        echo "⚠ Found incomplete sinograms in project directory (${EXISTING_SINOS}/2296)"
        echo "  Re-running merge from scratch..."
        DO_MERGE=1
    fi
else
    echo "No existing merged sinograms found, running merge..."
    DO_MERGE=1
fi

if [ "${DO_MERGE:-0}" -eq 1 ]; then
    echo "This will:"
    echo "  - Merge 73,472 tiles (256×256) → 2,296 sinograms (410×3500)"
    echo "  - Apply Gaussian blending in overlap regions"
    echo "  - Denormalize from uint16 back to int16"
    echo "  - Remove padding to restore original shape"
    echo ""

    START_TIME=$(date +%s)

    python merge_repaint_tiles.py \
        --tiles_dir repaint/tiles_infilled \
        --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
        --output_dir repaint/sinograms_infilled \
        --blend_mode nearest

    END_TIME=$(date +%s)
    MERGE_DURATION=$((END_TIME - START_TIME))

    # Immediately transfer merged sinograms to project directory (checkpoint)
    echo ""
    echo "Transferring merged sinograms to project directory (checkpoint)..."
    mkdir -p $PROJECT/repaint/sinograms_infilled
    rsync -av --info=progress2 repaint/sinograms_infilled/ $PROJECT/repaint/sinograms_infilled/ 2>&1 | tail -10
    echo "✓ Checkpoint saved"

    # Verify sinograms
    NUM_SINOS=$(find repaint/sinograms_infilled -name "*.png" 2>/dev/null | wc -l)
fi
echo ""
if [ ${MERGE_DURATION} -gt 0 ]; then
    echo "Tile merging complete in $((MERGE_DURATION / 60)) minutes $((MERGE_DURATION % 60)) seconds"
else
    echo "Tile merging skipped (loaded from checkpoint)"
fi
echo "Total sinograms: ${NUM_SINOS}"

EXPECTED_SINOS=2296
if [ ${NUM_SINOS} -ne ${EXPECTED_SINOS} ]; then
    echo "WARNING: Expected ${EXPECTED_SINOS} sinograms but got ${NUM_SINOS}"
    echo "This may be normal if only a subset of tiles was processed"
fi

#==============================================================================
# STEP 3: FDK RECONSTRUCTION
#==============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Running FDK reconstruction on infilled sinograms"
echo "========================================================================"
echo "This will reconstruct the full CT volume from ${NUM_SINOS} infilled sinograms"
echo "Expected shape: 410×3500 per sinogram (original dimensions, padding removed)"
echo ""

# Verify scan folder exists before reconstruction
echo "Verifying scan folder..."
# data/results is a SIBLING of Base_model_comparison, not inside it
SCAN_XML_PATH="${PROJECT%/Base_model_comparison}/data/results/Scan_1681_uwarp_gt/scan.xml"
if [ ! -f "${SCAN_XML_PATH}" ]; then
    echo ""
    echo "ERROR: scan.xml not found!"
    echo "  Expected path: ${SCAN_XML_PATH}"
    echo ""
    echo "This file is required for FDK reconstruction (geometry, angles, detector specs)."
    echo "Please verify the scan folder exists in your project directory."
    exit 1
fi
echo "✓ Scan folder verified: ${SCAN_XML_PATH}"
echo ""

START_TIME=$(date +%s)

# Use the parent directory of SCAN_XML_PATH for --scan_folder
SCAN_FOLDER="${SCAN_XML_PATH%/scan.xml}"

python reconstruct_from_repaint.py \
    --sinogram_dir repaint/sinograms_infilled \
    --output_dir repaint/reconstructed_volume \
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

# Create output directories in project
mkdir -p $PROJECT/repaint/sinograms_infilled
mkdir -p $PROJECT/repaint/reconstructed_volume

# Transfer sinograms
echo ""
echo "Transferring merged sinograms..."
rsync -av --info=progress2 repaint/sinograms_infilled/ $PROJECT/repaint/sinograms_infilled/ 2>&1 | tail -10

# Transfer reconstruction results
echo ""
echo "Transferring reconstructed volume..."
rsync -av --info=progress2 repaint/reconstructed_volume/ $PROJECT/repaint/reconstructed_volume/ 2>&1 | tail -10

# Also copy the .vff file (saved at same level as directory)
if [ -f "repaint/reconstructed_volume.vff" ]; then
    cp repaint/reconstructed_volume.vff $PROJECT/repaint/reconstructed_volume.vff
    echo "Copied reconstructed_volume.vff to project directory"
fi

#==============================================================================
# COMPLETION SUMMARY
#==============================================================================

echo ""
echo "========================================================================"
echo "PROCESSING COMPLETE!"
echo "========================================================================"
echo "Results saved to:"
echo "  Merged sinograms: $PROJECT/repaint/sinograms_infilled/"
echo "  Reconstructed volume: $PROJECT/repaint/reconstructed_volume/"
echo ""
echo "Statistics:"
echo "  Tiles merged: ${TOTAL_TILES_TRANSFERRED}"
echo "  Sinograms created: ${NUM_SINOS}"
echo "  Tile merge time: $((MERGE_DURATION / 60))m $((MERGE_DURATION % 60))s"
echo "  Reconstruction time: $((RECON_DURATION / 60))m $((RECON_DURATION % 60))s"
echo ""
echo "Job completed: $(date)"
echo "========================================================================"

echo ""
echo "Next steps:"
echo "  1. Transfer results to local server:"
echo "     rsync -av ${CLUSTER}.alliancecan.ca:$PROJECT/repaint/reconstructed_volume/ /path/to/local/"
echo ""
echo "  2. Visualize reconstruction (local):"
echo "     python visualize_reconstruction.py --volume_dir /path/to/local/reconstructed_volume/"
echo ""
