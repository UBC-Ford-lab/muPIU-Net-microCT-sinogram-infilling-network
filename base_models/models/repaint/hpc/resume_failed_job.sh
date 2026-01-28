#!/bin/bash

#==============================================================================
# Resume Failed or Timed-Out RePaint Job
# Allows you to resume processing from where it left off
# Works on any Compute Canada cluster
#
# NOTE: This script works with the new range-based approach (no splits directory)
#==============================================================================

set -euo pipefail

# Check if task ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 2  (to resume task 2)"
    echo ""
    echo "Available tasks: 0, 1, 2, 3"
    exit 1
fi

TASK_ID=$1

if [ ${TASK_ID} -lt 0 ] || [ ${TASK_ID} -gt 3 ]; then
    echo "Error: Task ID must be between 0 and 3"
    exit 1
fi

# Detect cluster and set paths
HOSTNAME=$(hostname)
if [[ ${HOSTNAME} == *"cedar"* ]] || [[ ${HOSTNAME} == *"fir"* ]] || [[ ${HOSTNAME} == *"graham"* ]]; then
    # Try common paths
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    elif [ -d "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/project/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot find project directory"
        exit 1
    fi
else
    PROJECT='.'
fi

# New approach: each task outputs to tiles_infilled_task_X
OUTPUT_DIR="${PROJECT}/repaint/tiles_infilled_task_${TASK_ID}"
TILE_SOURCE="${PROJECT}/repaint/sinogram_tiles/sinograms_masked"

echo "========================================================================"
echo "RePaint Job Recovery - Task ${TASK_ID}"
echo "========================================================================"

# Check if source tiles exist
if [ ! -d "${TILE_SOURCE}" ]; then
    echo "Error: Source tiles not found: ${TILE_SOURCE}"
    echo "Make sure create_repaint_tiles.py has been run first."
    exit 1
fi

# Count total tiles
TOTAL_TILES=$(find ${TILE_SOURCE} -name "*.png" | wc -l)
echo "Total tiles in dataset: ${TOTAL_TILES}"

# Calculate this task's expected tile count
TILES_PER_TASK=$((TOTAL_TILES / 4))
if [ ${TASK_ID} -eq 3 ]; then
    EXPECTED_TILES=$((TOTAL_TILES - 3 * TILES_PER_TASK))
else
    EXPECTED_TILES=${TILES_PER_TASK}
fi
echo "Expected tiles for task ${TASK_ID}: ${EXPECTED_TILES}"

# Count already completed tiles
if [ -d "${OUTPUT_DIR}" ]; then
    COMPLETED_TILES=$(find ${OUTPUT_DIR} -name "*.png" 2>/dev/null | wc -l)
    echo "Already completed: ${COMPLETED_TILES}"
    REMAINING=$((EXPECTED_TILES - COMPLETED_TILES))
    echo "Remaining to process: ${REMAINING}"

    if [ ${REMAINING} -le 0 ]; then
        echo ""
        echo "✓ This task is already complete! (${COMPLETED_TILES}/${EXPECTED_TILES} tiles)"
        echo "  Output directory: ${OUTPUT_DIR}"
        exit 0
    fi

    echo ""
    echo "Progress: ${COMPLETED_TILES}/${EXPECTED_TILES} ($(echo "scale=1; ${COMPLETED_TILES}*100/${EXPECTED_TILES}" | bc)%)"
else
    echo "No previous output found. Will start from scratch."
    COMPLETED_TILES=0
    REMAINING=${EXPECTED_TILES}
fi

echo ""
echo "========================================================================"
echo "Options for Resuming:"
echo "========================================================================"

echo ""
echo "Option 1: Resubmit just this task (RECOMMENDED)"
echo "  sbatch --array=${TASK_ID} run_repaint_cedar_h100.sh"
echo ""
echo "  This will:"
echo "  - Process tiles in range for task ${TASK_ID}"
echo "  - Keep using hourly checkpoints"
echo "  - NOTE: Currently restarts from beginning of range"
echo "         (completed tiles will be overwritten)"

echo ""
echo "Option 2: Extend time allocation"
echo "  If job timed out, you can request more time:"
echo "  sbatch --array=${TASK_ID} --time=03-00:00:00 run_repaint_cedar_h100.sh"

echo ""
echo "========================================================================"
echo "Recovery Status Summary"
echo "========================================================================"
echo "Task ${TASK_ID}:"
echo "  Tile range: start_idx=$((TASK_ID * TILES_PER_TASK))"
echo "  Expected tiles: ${EXPECTED_TILES}"
echo "  Completed: ${COMPLETED_TILES} ($(echo "scale=1; ${COMPLETED_TILES}*100/${EXPECTED_TILES}" | bc)%)"
echo "  Remaining: ${REMAINING}"
echo ""

if [ ${COMPLETED_TILES} -gt 0 ]; then
    echo "✓ Checkpoint data preserved at: ${OUTPUT_DIR}"
    echo "  (Note: Resubmitting will reprocess from the start of this task's range)"
else
    echo "⚠ No checkpoint data found - job will start from scratch"
fi

echo ""
echo "Recommended action: sbatch --array=${TASK_ID} run_repaint_cedar_h100.sh"
