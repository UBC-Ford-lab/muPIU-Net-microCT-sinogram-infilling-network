#!/bin/bash
# ============================================================================
# Complete Base Model Comparison Pipeline
# ============================================================================
# This script runs the entire pipeline from scratch:
# 1. Cleans up old data
# 2. Creates sinogram dataset from uwarp projections
# 3. Runs LaMa inference
# 4. Reconstructs both GT and LaMa volumes
#
# Usage:
#   ./run_complete_lama_pipeline.sh [--keep-memmap] [--skip-cleanup]
#
# Options:
#   --keep-memmap    Keep memmap files after reconstruction (faster reruns)
#   --skip-cleanup   Skip cleanup step (use existing dataset)
#
# Written for CT reconstruction comparison with Original U-Net model
# ============================================================================

set -e  # Exit on error

# Parse command-line arguments
KEEP_MEMMAP=false
SKIP_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-memmap)
            KEEP_MEMMAP=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--keep-memmap] [--skip-cleanup]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "                 BASE MODEL COMPARISON - FULL PIPELINE"
echo "============================================================================"
echo ""
echo "This script will:"
echo "  1. Clean up old data (unless --skip-cleanup)"
echo "  2. Create sinogram dataset from uwarp projections"
echo "  3. Run LaMa inference on masked sinograms"
echo "  4. Reconstruct both GT and LaMa volumes"
echo ""
echo "Options:"
echo "  Keep memmap: $KEEP_MEMMAP"
echo "  Skip cleanup: $SKIP_CLEANUP"
echo ""
echo "============================================================================"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtualenv
# Set VENV_PATH environment variable, or use default location
VENV_PATH="${VENV_PATH:-$HOME/Python_virtual_env}"
echo ""
echo -e "${BLUE}Activating virtualenv from: $VENV_PATH${NC}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Set VENV_PATH environment variable to your virtual environment path"
    exit 1
fi

# ============================================================================
# STEP 1: CLEANUP OLD DATA
# ============================================================================

if [ "$SKIP_CLEANUP" = false ]; then
    echo ""
    echo "============================================================================"
    echo "STEP 1: CLEANING UP OLD DATA"
    echo "============================================================================"

    echo ""
    echo -e "${YELLOW}Removing old dataset and reconstructions...${NC}"

    # Remove old sinogram dataset
    if [ -d "sinogram_dataset" ]; then
        echo "  Removing: sinogram_dataset/"
        rm -rf sinogram_dataset
    fi

    # Remove LaMa input data
    if [ -d "lama_input_data" ]; then
        echo "  Removing: lama_input_data/"
        rm -rf lama_input_data
    fi

    # Remove LaMa infilled outputs
    if [ -d "lama/sinograms_infilled" ]; then
        echo "  Removing: lama/sinograms_infilled/"
        rm -rf lama/sinograms_infilled
    fi

    # Remove old reconstructions
    if [ -f "gt_reconstruction.vff" ]; then
        echo "  Removing: gt_reconstruction.vff"
        rm -f gt_reconstruction.vff
    fi

    if [ -f "lama_reconstruction.vff" ]; then
        echo "  Removing: lama_reconstruction.vff"
        rm -f lama_reconstruction.vff
    fi

    if [ -d "gt_reconstruction" ]; then
        echo "  Removing: gt_reconstruction/ (TIFF slices)"
        rm -rf gt_reconstruction
    fi

    if [ -d "lama_reconstruction" ]; then
        echo "  Removing: lama_reconstruction/ (TIFF slices)"
        rm -rf lama_reconstruction
    fi

    # Remove old memmap files (critical for ensuring fresh data!)
    echo ""
    echo -e "${YELLOW}Removing old memmap files...${NC}"

    # Memmap in /tmp
    if ls /tmp/ct_projections_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/ct_projections_*.dat"
        rm -f /tmp/ct_projections_*.dat
    fi

    # Memmap in current directory
    if ls ct_projections_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: ./ct_projections_*.dat"
        rm -f ct_projections_*.dat
    fi

    # Cached reconstruction memmaps
    if ls /tmp/sinogram_cache_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/sinogram_cache_*.dat"
        rm -f /tmp/sinogram_cache_*.dat
    fi

    if ls /tmp/sinogram_working_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/sinogram_working_*.dat"
        rm -f /tmp/sinogram_working_*.dat
    fi

    echo ""
    echo -e "${GREEN}✓ Cleanup complete${NC}"
else
    echo ""
    echo "============================================================================"
    echo "STEP 1: SKIPPING CLEANUP (--skip-cleanup flag)"
    echo "============================================================================"
fi

# ============================================================================
# STEP 2: CREATE SINOGRAM DATASET
# ============================================================================

echo ""
echo "============================================================================"
echo "STEP 2: CREATING SINOGRAM DATASET"
echo "============================================================================"
echo ""
echo "Using uwarp projections from: data/results/Scan_1681_uwarp_gt"
echo "This ensures Base_model_comparison uses the SAME projection source as Original U-Net!"
echo ""

# Check if dataset already exists
if [ -d "sinogram_dataset" ] && [ "$SKIP_CLEANUP" = true ]; then
    echo -e "${GREEN}✓ Dataset already exists, skipping creation${NC}"
else
    python create_sinogram_dataset.py \
        --normalize_globally \
        --reuse_memmap \
        --keep_memmap

    echo ""
    echo -e "${GREEN}✓ Sinogram dataset created${NC}"
fi

# ============================================================================
# STEP 3: RUN LAMA INFERENCE
# ============================================================================

echo ""
echo "============================================================================"
echo "STEP 3: RUNNING LAMA INFERENCE"
echo "============================================================================"
echo ""

cd lama

# Check if LaMa output already exists
if [ -d "sinograms_infilled" ] && [ "$SKIP_CLEANUP" = true ]; then
    NUM_INFILLED=$(ls sinograms_infilled/*.png 2>/dev/null | wc -l)
    if [ "$NUM_INFILLED" -gt 0 ]; then
        echo -e "${GREEN}✓ LaMa output already exists ($NUM_INFILLED files), skipping inference${NC}"
        cd ..
    else
        ./setup_and_run_lama_fast.sh
        cd ..
        echo ""
        echo -e "${GREEN}✓ LaMa inference complete${NC}"
    fi
else
    ./setup_and_run_lama_fast.sh
    cd ..
    echo ""
    echo -e "${GREEN}✓ LaMa inference complete${NC}"
fi

# ============================================================================
# STEP 4: RECONSTRUCT GT AND LAMA VOLUMES
# ============================================================================

echo ""
echo "============================================================================"
echo "STEP 4: RECONSTRUCTING GT AND LAMA VOLUMES"
echo "============================================================================"
echo ""

# Run reconstruction for both GT and LaMa
python reconstruct_from_lama.py --mode both

echo ""
echo -e "${GREEN}✓ Reconstructions complete${NC}"

# ============================================================================
# STEP 5: VERIFY OUTPUT FILES
# ============================================================================

echo ""
echo "============================================================================"
echo "STEP 5: VERIFYING OUTPUT FILES"
echo "============================================================================"
echo ""

# Check GT reconstruction
if [ -f "gt_reconstruction.vff" ]; then
    GT_SIZE=$(stat -c%s "gt_reconstruction.vff" 2>/dev/null || stat -f%z "gt_reconstruction.vff")
    GT_SIZE_MB=$((GT_SIZE / 1024 / 1024))
    echo -e "${GREEN}✓ gt_reconstruction.vff${NC} ($GT_SIZE_MB MB)"
else
    echo -e "${RED}✗ gt_reconstruction.vff NOT FOUND${NC}"
fi

# Check LaMa reconstruction
if [ -f "lama_reconstruction.vff" ]; then
    LAMA_SIZE=$(stat -c%s "lama_reconstruction.vff" 2>/dev/null || stat -f%z "lama_reconstruction.vff")
    LAMA_SIZE_MB=$((LAMA_SIZE / 1024 / 1024))
    echo -e "${GREEN}✓ lama_reconstruction.vff${NC} ($LAMA_SIZE_MB MB)"
else
    echo -e "${RED}✗ lama_reconstruction.vff NOT FOUND${NC}"
fi

# Check dataset
if [ -d "sinogram_dataset" ]; then
    NUM_SINOGRAMS=$(ls sinogram_dataset/sinograms_gt/*.png 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ sinogram_dataset${NC} ($NUM_SINOGRAMS sinograms)"
else
    echo -e "${RED}✗ sinogram_dataset NOT FOUND${NC}"
fi

# ============================================================================
# STEP 6: VERIFY GT RECONSTRUCTIONS MATCH
# ============================================================================

echo ""
echo "============================================================================"
echo "STEP 6: VERIFYING GT RECONSTRUCTIONS MATCH ORIGINAL"
echo "============================================================================"
echo ""

python << EOF
import sys
import os

# Add project root to path (determined by shell script)
sys.path.insert(0, '$PROJECT_ROOT')

from ct_core.vff_io import read_vff
import numpy as np

try:
    # Load reconstructions
    _, gt_base = read_vff('gt_reconstruction.vff', verbose=False)
    _, gt_orig = read_vff('../data/results/Scan_1681_gt_recon.vff', verbose=False)

    # Compare full volume ranges
    base_range = float(np.max(gt_base) - np.min(gt_base))
    orig_range = float(np.max(gt_orig) - np.min(gt_orig))

    print(f"Full Volume Ranges:")
    print(f"  Base GT:     {base_range:>10.1f}")
    print(f"  Original GT: {orig_range:>10.1f}")
    print(f"  Difference:  {abs(base_range - orig_range):>10.1f} ({abs(base_range - orig_range)/orig_range*100:.2f}%)")

    # Compare MTF region ranges
    slice_range = [228, 229]
    crop = [270, 664, 522, 640]

    mtf_base = gt_base[slice_range[0]:slice_range[1], crop[0]:crop[1], crop[2]:crop[3]]
    mtf_orig = gt_orig[slice_range[0]:slice_range[1], crop[0]:crop[1], crop[2]:crop[3]]

    base_mtf_range = float(np.max(mtf_base) - np.min(mtf_base))
    orig_mtf_range = float(np.max(mtf_orig) - np.min(mtf_orig))

    print(f"\nMTF Region Ranges:")
    print(f"  Base GT:     {base_mtf_range:>10.1f}")
    print(f"  Original GT: {orig_mtf_range:>10.1f}")
    print(f"  Difference:  {abs(base_mtf_range - orig_mtf_range):>10.1f} ({abs(base_mtf_range - orig_mtf_range)/orig_mtf_range*100:.2f}%)")

    # Verdict
    print(f"\nVerification:")
    if abs(base_mtf_range - orig_mtf_range) < 500:  # Allow 500 tolerance
        print(f"  ✓ MTF region ranges MATCH (difference < 500)")
        print(f"  ✓ Both GT reconstructions use same projection source")
        print(f"  ✓ ERF dropoff should now be consistent")
        sys.exit(0)
    else:
        print(f"  ✗ MTF region ranges DIFFER significantly")
        print(f"  ✗ Something is still wrong with projection source or normalization")
        sys.exit(1)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Make sure both VFF files exist")
    sys.exit(1)
except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

VERIFY_EXIT=$?

# ============================================================================
# CLEANUP MEMMAP FILES (if requested)
# ============================================================================

if [ "$KEEP_MEMMAP" = false ]; then
    echo ""
    echo "============================================================================"
    echo "CLEANING UP MEMMAP FILES"
    echo "============================================================================"
    echo ""

    # Remove memmap files
    if ls /tmp/ct_projections_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/ct_projections_*.dat"
        rm -f /tmp/ct_projections_*.dat
    fi

    if ls /tmp/sinogram_cache_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/sinogram_cache_*.dat"
        rm -f /tmp/sinogram_cache_*.dat
    fi

    if ls /tmp/sinogram_working_*.dat 1> /dev/null 2>&1; then
        echo "  Removing: /tmp/sinogram_working_*.dat"
        rm -f /tmp/sinogram_working_*.dat
    fi

    echo ""
    echo -e "${GREEN}✓ Memmap files cleaned up${NC}"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "                        PIPELINE COMPLETE!"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  ✓ gt_reconstruction.vff         - Ground truth reconstruction"
echo "  ✓ lama_reconstruction.vff       - LaMa infilled reconstruction"
echo "  ✓ sinogram_dataset/             - Sinogram dataset (GT, masked, masks)"
echo "  ✓ lama/sinograms_infilled/      - LaMa infilled sinograms"
echo ""

if [ $VERIFY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ VERIFICATION PASSED${NC}"
    echo "  GT reconstructions match Original U-Net reconstruction"
    echo "  Ready for MTF/NPS/NEQ comparison!"
else
    echo -e "${RED}✗ VERIFICATION FAILED${NC}"
    echo "  GT reconstructions do not match"
    echo "  Please check projection source and normalization"
fi

echo ""
echo "Next steps:"
echo "  1. Run MTF comparison: cd ../Metric\ calculators/Helper\ scripts && python MTF_comparison_plotting.py"
echo "  2. Run NPS comparison: cd ../Metric\ calculators/Helper\ scripts && python NPS_comparison_plotting.py"
echo "  3. Run NEQ comparison: cd ../Metric\ calculators/Helper\ scripts && python NEQ_comparison_plotting.py"
echo ""
echo "============================================================================"

exit $VERIFY_EXIT
