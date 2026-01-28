#!/bin/bash

#==============================================================================
# Cleanup RePaint Files to Reduce File Count
#
# This script helps reduce file count on Compute Canada by removing:
# 1. The deprecated splits directory (100k+ files - no longer needed!)
# 2. Optionally, sinograms_gt tiles (only needed for evaluation, not inference)
#
# Run this BEFORE submitting jobs to stay under the 500k file limit.
#==============================================================================

set -euo pipefail

# Detect cluster and set paths
HOSTNAME=$(hostname)
if [[ ${HOSTNAME} == *"cedar"* ]] || [[ ${HOSTNAME} == *"fir"* ]] || [[ ${HOSTNAME} == *"graham"* ]]; then
    if [ -d "/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    elif [ -d "/project/def-nlford/wiegmann/ct_recon/Base_model_comparison" ]; then
        PROJECT='/project/def-nlford/wiegmann/ct_recon/Base_model_comparison'
    else
        echo "ERROR: Cannot find project directory"
        exit 1
    fi
else
    PROJECT="${1:-.}"
fi

echo "========================================================================"
echo "RePaint File Cleanup Utility"
echo "========================================================================"
echo "Project directory: ${PROJECT}"
echo ""

# Calculate current file usage
echo "Current file counts:"

# Check splits directory
SPLITS_DIR="${PROJECT}/repaint/splits"
if [ -d "${SPLITS_DIR}" ]; then
    SPLITS_COUNT=$(find "${SPLITS_DIR}" -type f 2>/dev/null | wc -l)
    echo "  - splits/: ${SPLITS_COUNT} files (DEPRECATED - safe to delete!)"
else
    SPLITS_COUNT=0
    echo "  - splits/: (not found)"
fi

# Check sinograms_gt directory
GT_DIR="${PROJECT}/repaint/sinogram_tiles/sinograms_gt"
if [ -d "${GT_DIR}" ]; then
    GT_COUNT=$(find "${GT_DIR}" -type f 2>/dev/null | wc -l)
    echo "  - sinograms_gt/: ${GT_COUNT} files (only needed for evaluation)"
else
    GT_COUNT=0
    echo "  - sinograms_gt/: (not found)"
fi

# Check sinograms_masked directory (needed for inference!)
MASKED_DIR="${PROJECT}/repaint/sinogram_tiles/sinograms_masked"
if [ -d "${MASKED_DIR}" ]; then
    MASKED_COUNT=$(find "${MASKED_DIR}" -type f 2>/dev/null | wc -l)
    echo "  - sinograms_masked/: ${MASKED_COUNT} files (REQUIRED for inference)"
else
    MASKED_COUNT=0
    echo "  - sinograms_masked/: (not found - REQUIRED!)"
fi

# Check masks directory (needed!)
MASKS_DIR="${PROJECT}/repaint/sinogram_tiles/masks"
if [ -d "${MASKS_DIR}" ]; then
    MASKS_COUNT=$(find "${MASKS_DIR}" -type f 2>/dev/null | wc -l)
    echo "  - masks/: ${MASKS_COUNT} files (REQUIRED for inference)"
else
    MASKS_COUNT=0
    echo "  - masks/: (not found - REQUIRED!)"
fi

TOTAL_DELETABLE=$((SPLITS_COUNT + GT_COUNT))
echo ""
echo "Files that can be safely deleted: ${TOTAL_DELETABLE}"
echo "  - splits/ (${SPLITS_COUNT}) - DEPRECATED: we now use start_idx/end_idx"
echo "  - sinograms_gt/ (${GT_COUNT}) - Only needed for evaluation metrics"
echo ""

if [ ${TOTAL_DELETABLE} -eq 0 ]; then
    echo "Nothing to clean up!"
    exit 0
fi

echo "========================================================================"
echo "Cleanup Options:"
echo "========================================================================"
echo ""
echo "1. Delete splits/ directory only (RECOMMENDED - always safe)"
echo "   rm -rf ${SPLITS_DIR}"
echo "   Saves: ${SPLITS_COUNT} files"
echo ""
echo "2. Delete both splits/ and sinograms_gt/ (if you don't need evaluation)"
echo "   rm -rf ${SPLITS_DIR} ${GT_DIR}"
echo "   Saves: ${TOTAL_DELETABLE} files"
echo ""
echo "========================================================================"

# Interactive mode
if [ -t 0 ]; then
    echo ""
    read -p "Delete splits/ directory now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "${SPLITS_DIR}" ]; then
            echo "Deleting ${SPLITS_DIR}..."
            rm -rf "${SPLITS_DIR}"
            echo "✓ Deleted ${SPLITS_COUNT} files from splits/"
        fi
    fi

    echo ""
    read -p "Delete sinograms_gt/ directory? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "${GT_DIR}" ]; then
            echo "Deleting ${GT_DIR}..."
            rm -rf "${GT_DIR}"
            echo "✓ Deleted ${GT_COUNT} files from sinograms_gt/"
        fi
    fi

    echo ""
    echo "Cleanup complete!"
else
    echo ""
    echo "Run this script interactively to perform cleanup, or use the commands above."
fi

echo ""
echo "NOTE: After cleanup, the inference workflow will use:"
echo "  - sinograms_masked/ as INPUT (the masked sinograms)"
echo "  - masks/ to identify regions to inpaint"
echo "  - start_idx/end_idx for GPU task distribution (NO file copying!)"
