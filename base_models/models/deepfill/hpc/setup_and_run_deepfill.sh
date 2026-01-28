#!/bin/bash
#==============================================================================
# DeepFill v2 Complete Pipeline - Local Setup and Run
#==============================================================================
# This script sets up and runs the complete DeepFill v2 inpainting pipeline:
# 1. Creates virtual environment (if needed)
# 2. Installs dependencies
# 3. Prepares tiles from existing RePaint tiles
# 4. Runs inference
# 5. Merges tiles back to full sinograms
# 6. Runs FDK reconstruction
#
# Requirements:
# - NVIDIA GPU with CUDA support (recommended)
# - At least 16GB GPU memory for batch processing
# - At least 32GB system RAM
# - Python 3.8+
#
# Usage:
#   ./setup_and_run_deepfill.sh [--skip-setup] [--cpu-only]
#
# Author: Claude (Anthropic)
# Date: 2025-11-29
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_SETUP=false
CPU_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-setup] [--cpu-only]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "DeepFill v2 Sinogram Inpainting Pipeline"
echo "========================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Start time: $(date)"
echo ""

#==============================================================================
# STEP 1: Environment Setup
#==============================================================================

VENV_DIR="$SCRIPT_DIR/deepfill_venv"

if [ "$SKIP_SETUP" = false ]; then
    echo "========================================================================"
    echo "Step 1: Setting up virtual environment..."
    echo "========================================================================"

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists."
    fi

    source "$VENV_DIR/bin/activate"

    echo "Installing dependencies..."
    pip install --upgrade pip

    # Install PyTorch with CUDA support (if GPU available)
    if [ "$CPU_ONLY" = false ] && command -v nvidia-smi &> /dev/null; then
        echo "GPU detected, installing PyTorch with CUDA..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install other dependencies
    pip install pillow numpy tqdm pyyaml scipy

    echo "Dependencies installed successfully!"
else
    echo "Skipping setup (--skip-setup flag)"
    source "$VENV_DIR/bin/activate"
fi

# Show environment info
echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if [ "$CPU_ONLY" = false ]; then
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    fi
fi

#==============================================================================
# STEP 2: Verify Prerequisites
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 2: Verifying prerequisites..."
echo "========================================================================"

# Check for DeepFill v2 model and weights
if [ ! -d "$SCRIPT_DIR/deepfill/DeepFillv2" ]; then
    echo "ERROR: DeepFill v2 repository not found."
    echo "Please clone it first:"
    echo "  cd $SCRIPT_DIR/deepfill"
    echo "  git clone https://github.com/nipponjo/deepfillv2-pytorch.git DeepFillv2"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth" ]; then
    echo "ERROR: CelebA-HQ weights not found."
    echo "Please download them:"
    echo "  cd $SCRIPT_DIR/deepfill/DeepFillv2"
    echo "  python download_files.py"
    exit 1
fi

# Check for RePaint tiles (source for DeepFill tiles)
if [ ! -d "$SCRIPT_DIR/repaint/sinogram_tiles/sinograms_gt" ]; then
    echo "ERROR: RePaint tiles not found."
    echo "Please run create_repaint_tiles.py first to generate the 256x256 tiles."
    exit 1
fi

TILE_COUNT=$(find "$SCRIPT_DIR/repaint/sinogram_tiles/sinograms_gt" -name "*.png" | wc -l)
echo "Found $TILE_COUNT RePaint tiles"

if [ "$TILE_COUNT" -lt 70000 ]; then
    echo "ERROR: Expected ~73,472 tiles, found only $TILE_COUNT"
    exit 1
fi

echo "Prerequisites verified!"

#==============================================================================
# STEP 3: Verify RePaint Tiles (no separate DeepFill tiles needed)
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 3: Verifying RePaint tiles..."
echo "========================================================================"

REPAINT_GT_DIR="$SCRIPT_DIR/repaint/sinogram_tiles/sinograms_gt"
REPAINT_MASK_DIR="$SCRIPT_DIR/repaint/sinogram_tiles/masks"

REPAINT_GT_COUNT=$(find "$REPAINT_GT_DIR" -name "*.png" 2>/dev/null | wc -l)
REPAINT_MASK_COUNT=$(find "$REPAINT_MASK_DIR" -name "*.png" 2>/dev/null | wc -l)

echo "RePaint tiles: GT=${REPAINT_GT_COUNT}, Masks=${REPAINT_MASK_COUNT}"

if [ "$REPAINT_GT_COUNT" -lt 70000 ] || [ "$REPAINT_MASK_COUNT" -lt 70000 ]; then
    echo "ERROR: Not enough RePaint tiles. Expected ~73,472 each."
    exit 1
fi

echo ""
echo "NOTE: Using on-the-fly mask inversion - no separate DeepFill tiles needed!"
echo "      This saves ~73,000 files on disk."

#==============================================================================
# STEP 4: Run DeepFill v2 Inference
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 4: Running DeepFill v2 inference..."
echo "========================================================================"

OUTPUT_DIR="$SCRIPT_DIR/deepfill/tiles_infilled"
mkdir -p "$OUTPUT_DIR"

# Check for existing output
EXISTING_OUTPUT=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
if [ "$EXISTING_OUTPUT" -ge 70000 ]; then
    echo "Found $EXISTING_OUTPUT existing output tiles."
    read -p "Skip inference and use existing tiles? (Y/n): " SKIP_INFERENCE
    if [ "$SKIP_INFERENCE" != "n" ] && [ "$SKIP_INFERENCE" != "N" ]; then
        echo "Using existing tiles."
    else
        DEVICE="cuda"
        if [ "$CPU_ONLY" = true ]; then
            DEVICE="cpu"
        fi

        # Uses --invert_masks to convert RePaint masks (0=inpaint) to DeepFill format (255=inpaint) on-the-fly
        python3 run_deepfill_inference.py \
            --gt_dir repaint/sinogram_tiles/sinograms_gt \
            --mask_dir repaint/sinogram_tiles/masks \
            --output_dir deepfill/tiles_infilled \
            --checkpoint deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth \
            --batch_size 16 \
            --device $DEVICE \
            --invert_masks \
            --save_grayscale
    fi
else
    DEVICE="cuda"
    if [ "$CPU_ONLY" = true ]; then
        DEVICE="cpu"
    fi

    # Uses --invert_masks to convert RePaint masks (0=inpaint) to DeepFill format (255=inpaint) on-the-fly
    python3 run_deepfill_inference.py \
        --gt_dir repaint/sinogram_tiles/sinograms_gt \
        --mask_dir repaint/sinogram_tiles/masks \
        --output_dir deepfill/tiles_infilled \
        --checkpoint deepfill/DeepFillv2/pretrained/states_tf_celebahq.pth \
        --batch_size 16 \
        --device $DEVICE \
        --invert_masks \
        --save_grayscale
fi

#==============================================================================
# STEP 5: Merge Tiles
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 5: Merging tiles back to full sinograms..."
echo "========================================================================"

python3 merge_deepfill_tiles.py \
    --tiles_dir deepfill/tiles_infilled \
    --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --output_dir deepfill/sinograms_infilled \
    --blend_mode gaussian

#==============================================================================
# STEP 6: FDK Reconstruction
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 6: Running FDK reconstruction..."
echo "========================================================================"

python3 reconstruct_from_deepfill.py \
    --sinogram_dir deepfill/sinograms_infilled \
    --output_dir deepfill/reconstructed_volume

#==============================================================================
# Summary
#==============================================================================

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  Infilled tiles: deepfill/tiles_infilled/"
echo "  Merged sinograms: deepfill/sinograms_infilled/"
echo "  Reconstruction: deepfill/reconstructed_volume/"
echo ""
echo "Next steps:"
echo "  python deepfill_domain_comparison.py  # Run metrics evaluation"
echo ""
echo "========================================================================"
