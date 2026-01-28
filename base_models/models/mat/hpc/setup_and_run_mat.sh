#!/bin/bash
#==============================================================================
# MAT (Mask-Aware Transformer) Complete Pipeline - Local Setup and Run
#==============================================================================
# This script sets up and runs the complete MAT inpainting pipeline:
# 1. Clones MAT repository and downloads CelebA-HQ 256x256 checkpoint
# 2. Creates virtual environment and installs dependencies
# 3. Uses existing RePaint tiles (256x256) directly
# 4. Runs inference with int16 handling
# 5. Merges tiles back to full sinograms
# 6. Runs FDK reconstruction
#
# Requirements:
# - NVIDIA GPU with CUDA 11.0+ support
# - At least 8GB GPU memory (16GB recommended for larger batches)
# - At least 32GB system RAM
# - Python 3.7+
#
# Usage:
#   ./setup_and_run_mat.sh [--skip-setup] [--cpu-only] [--download-only]
#
# Author: Claude (Anthropic)
# Date: 2025-12-01
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_SETUP=false
CPU_ONLY=false
DOWNLOAD_ONLY=false

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
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-setup] [--cpu-only] [--download-only]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "MAT (Mask-Aware Transformer) Sinogram Inpainting Pipeline"
echo "========================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Start time: $(date)"
echo ""

#==============================================================================
# STEP 1: Clone MAT Repository
#==============================================================================

MAT_DIR="$SCRIPT_DIR/mat/MAT"

echo "========================================================================"
echo "Step 1: Setting up MAT repository..."
echo "========================================================================"

if [ ! -d "$MAT_DIR" ]; then
    echo "Cloning MAT repository..."
    mkdir -p "$SCRIPT_DIR/mat"
    cd "$SCRIPT_DIR/mat"
    git clone https://github.com/fenglinglwb/MAT.git
    cd "$SCRIPT_DIR"
    echo "MAT repository cloned successfully!"
else
    echo "MAT repository already exists at $MAT_DIR"
fi

#==============================================================================
# STEP 2: Download CelebA-HQ 256x256 Checkpoint
#==============================================================================

CHECKPOINT_DIR="$MAT_DIR/pretrained"
CHECKPOINT_FILE="$CHECKPOINT_DIR/CelebA-HQ_256.pkl"

echo ""
echo "========================================================================"
echo "Step 2: Downloading CelebA-HQ 256x256 checkpoint..."
echo "========================================================================"

mkdir -p "$CHECKPOINT_DIR"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint not found. Downloading from OneDrive..."
    echo ""
    echo "IMPORTANT: The checkpoint must be downloaded manually from OneDrive."
    echo "Please download the CelebA-HQ 256x256 checkpoint from:"
    echo "  https://onedrive.live.com/?authkey=%21ADU4yMFZ5aRgeco&id=F9FC8EC0E334A1AE%21533&cid=F9FC8EC0E334A1AE"
    echo ""
    echo "Save it to: $CHECKPOINT_FILE"
    echo ""

    # Try using gdown if available (for Google Drive links)
    if command -v gdown &> /dev/null; then
        echo "gdown is available, but OneDrive links require manual download."
    fi

    # Check if wget/curl can be used with a direct link
    # Note: OneDrive links typically require authentication

    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo ""
        echo "After downloading, re-run this script."

        if [ "$DOWNLOAD_ONLY" = true ]; then
            echo "Exiting (--download-only mode)"
            exit 0
        fi

        read -p "Press Enter to continue if checkpoint is downloaded, or Ctrl+C to exit: "
    fi
fi

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint found: $CHECKPOINT_FILE"
    CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    echo "  Size: $CHECKPOINT_SIZE"
else
    echo "ERROR: Checkpoint not found at $CHECKPOINT_FILE"
    echo "Please download it manually and re-run this script."
    exit 1
fi

if [ "$DOWNLOAD_ONLY" = true ]; then
    echo ""
    echo "Download-only mode complete."
    exit 0
fi

#==============================================================================
# STEP 3: Environment Setup
#==============================================================================

VENV_DIR="$SCRIPT_DIR/mat_venv"

if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo "========================================================================"
    echo "Step 3: Setting up virtual environment..."
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

    # Install PyTorch with CUDA support
    # MAT requires PyTorch 1.7.1+ but works with newer versions
    if [ "$CPU_ONLY" = false ] && command -v nvidia-smi &> /dev/null; then
        echo "GPU detected, installing PyTorch with CUDA..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install MAT dependencies
    echo "Installing MAT dependencies..."
    pip install pillow numpy tqdm scipy pyyaml
    pip install opencv-python scikit-image
    pip install click requests pyspng ninja timm psutil scikit-learn
    pip install imageio-ffmpeg==0.4.3
    pip install easydict

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
# STEP 4: Verify Prerequisites
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 4: Verifying prerequisites..."
echo "========================================================================"

# Use sinograms_masked (the proper input for inpainting)
INPUT_TILE_DIR="sinograms_masked"
if [ ! -d "$SCRIPT_DIR/repaint/sinogram_tiles/sinograms_masked" ]; then
    echo "ERROR: sinograms_masked not found."
    echo "Please run create_repaint_tiles.py first to generate the 256x256 tiles."
    echo "Note: sinograms_gt is NOT a valid fallback - we need the masked tiles for proper inpainting."
    exit 1
fi
echo "Found sinograms_masked directory"

TILE_COUNT=$(find "$SCRIPT_DIR/repaint/sinogram_tiles/$INPUT_TILE_DIR" -name "*.png" | wc -l)
echo "Found $TILE_COUNT tiles in $INPUT_TILE_DIR (will use for MAT)"

if [ "$TILE_COUNT" -lt 70000 ]; then
    echo "ERROR: Expected ~73,472 tiles, found only $TILE_COUNT"
    exit 1
fi

# Verify mask tiles
MASK_COUNT=$(find "$SCRIPT_DIR/repaint/sinogram_tiles/masks" -name "*.png" | wc -l)
echo "Found $MASK_COUNT mask tiles"

echo "Prerequisites verified!"

#==============================================================================
# STEP 5: Run MAT Inference
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 5: Running MAT inference..."
echo "========================================================================"

OUTPUT_DIR="$SCRIPT_DIR/mat/tiles_infilled"
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

        python3 run_mat_inference.py \
            --input_dir "repaint/sinogram_tiles/$INPUT_TILE_DIR" \
            --mask_dir repaint/sinogram_tiles/masks \
            --output_dir mat/tiles_infilled \
            --checkpoint "$CHECKPOINT_FILE" \
            --batch_size 8 \
            --device $DEVICE \
            --save_grayscale
    fi
else
    DEVICE="cuda"
    if [ "$CPU_ONLY" = true ]; then
        DEVICE="cpu"
    fi

    python3 run_mat_inference.py \
        --input_dir "repaint/sinogram_tiles/$INPUT_TILE_DIR" \
        --mask_dir repaint/sinogram_tiles/masks \
        --output_dir mat/tiles_infilled \
        --checkpoint "$CHECKPOINT_FILE" \
        --batch_size 8 \
        --device $DEVICE \
        --save_grayscale
fi

#==============================================================================
# STEP 6: Merge Tiles
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 6: Merging tiles back to full sinograms..."
echo "========================================================================"

python3 merge_mat_tiles.py \
    --tiles_dir mat/tiles_infilled \
    --metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --output_dir mat/sinograms_infilled \
    --blend_mode nearest

#==============================================================================
# STEP 7: FDK Reconstruction
#==============================================================================

echo ""
echo "========================================================================"
echo "Step 7: Running FDK reconstruction..."
echo "========================================================================"

# Scan folder containing scan.xml with geometry info
SCAN_FOLDER="$SCRIPT_DIR/../data/results/Scan_1681_uwarp_gt"

if [ ! -f "${SCAN_FOLDER}/scan.xml" ]; then
    echo "ERROR: scan.xml not found at: ${SCAN_FOLDER}/scan.xml"
    echo "This file is required for FDK reconstruction."
    exit 1
fi
echo "Scan folder: ${SCAN_FOLDER}"

python3 reconstruct_from_mat.py \
    --sinogram_dir mat/sinograms_infilled \
    --output_dir mat/reconstructed_volume \
    --metadata_path sinogram_dataset/metadata.json \
    --tiling_metadata_path repaint/sinogram_tiles/tiling_metadata.json \
    --scan_folder "${SCAN_FOLDER}"

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
echo "  Infilled tiles: mat/tiles_infilled/"
echo "  Merged sinograms: mat/sinograms_infilled/"
echo "  Reconstruction: mat/reconstructed_volume/"
echo ""
echo "Next steps:"
echo "  python mat_domain_comparison.py  # Run metrics evaluation"
echo ""
echo "========================================================================"
