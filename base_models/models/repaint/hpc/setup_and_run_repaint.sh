#!/bin/bash
# RePaint Setup and Complete Workflow for CT Sinogram Inpainting
# ==============================================================
# This script handles the complete RePaint workflow:
# 1. Create 256×256 tiles from sinograms (if needed)
# 2. Run RePaint inference on all tiles
# 3. Merge infilled tiles back to full sinograms
# 4. Reconstruct CT volume from infilled sinograms

set -e  # Exit on error

echo "======================================================================"
echo "RePaint Complete Workflow for CT Sinogram Inpainting"
echo "======================================================================"

# Determine script and project directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../" && pwd )"
BASE_MODELS_DIR="$PROJECT_ROOT/base_models"

# Activate virtualenv
# Set VENV_PATH environment variable, or use default location
VENV_PATH="${VENV_PATH:-$HOME/Python_virtual_env}"
echo "Activating virtualenv from: $VENV_PATH"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Set VENV_PATH environment variable to your virtual environment path"
    exit 1
fi

# Navigate to base models directory
cd "$BASE_MODELS_DIR"

# Step 1: Check/Create tiles
echo ""
echo "Step 1: Checking for tiles..."
echo "----------------------------------------------------------------------"

TILES_DIR="repaint/sinogram_tiles/sinograms_gt"
if [ -d "$TILES_DIR" ] && [ "$(ls -A $TILES_DIR)" ]; then
    NUM_TILES=$(ls "$TILES_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Tiles already exist: $NUM_TILES tiles"
else
    echo "Creating tiles from sinogram dataset..."
    python3 create_repaint_tiles.py --force
    NUM_TILES=$(ls "$TILES_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Created $NUM_TILES tiles"
fi

# Step 2: Check for pre-trained model
echo ""
echo "Step 2: Checking pre-trained model..."
echo "----------------------------------------------------------------------"

MODEL_PATH="repaint/RePaint/data/pretrained/celeba256_250000.pt"
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "✓ Model exists: $MODEL_SIZE"
else
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run: gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX"
    echo "Then move celeba256_250000.pt to: $MODEL_PATH"
    exit 1
fi

# Step 3: Run RePaint inference
echo ""
echo "Step 3: Running RePaint inference on tiles..."
echo "----------------------------------------------------------------------"

OUTPUT_DIR="repaint/tiles_infilled"
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    NUM_INFILLED=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Infilled tiles already exist: $NUM_INFILLED tiles"
    read -p "Re-run RePaint inference? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping RePaint inference"
    else
        echo "Running RePaint inference..."
        cd repaint/RePaint
        python test.py --conf_path ../../configs/repaint_sinogram.yml 2>&1 | tee ../../repaint_inference.log
        cd ../..
        echo "✓ RePaint inference complete"
    fi
else
    echo "Running RePaint inference on $NUM_TILES tiles..."
    echo "This will take several hours depending on GPU speed..."
    cd repaint/RePaint
    python test.py --conf_path ../../configs/repaint_sinogram.yml 2>&1 | tee ../../repaint_inference.log
    cd ../..
    echo "✓ RePaint inference complete"
fi

# Step 4: Merge tiles back to full sinograms
echo ""
echo "Step 4: Merging infilled tiles back to full sinograms..."
echo "----------------------------------------------------------------------"

MERGED_DIR="repaint/sinograms_merged"
if [ -d "$MERGED_DIR" ] && [ "$(ls -A $MERGED_DIR)" ]; then
    NUM_MERGED=$(ls "$MERGED_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Merged sinograms already exist: $NUM_MERGED sinograms"
    read -p "Re-merge tiles? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping tile merging"
    else
        echo "Merging tiles..."
        python3 merge_repaint_tiles.py
        echo "✓ Tiles merged successfully"
    fi
else
    echo "Merging infilled tiles back to full resolution..."
    python3 merge_repaint_tiles.py
    NUM_MERGED=$(ls "$MERGED_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Merged $NUM_MERGED sinograms"
fi

# Step 5: Reconstruct CT volume
echo ""
echo "Step 5: Reconstructing CT volume from infilled sinograms..."
echo "----------------------------------------------------------------------"

RECON_FILE="repaint_reconstruction.vff"
if [ -f "$RECON_FILE" ]; then
    RECON_SIZE=$(du -h "$RECON_FILE" | cut -f1)
    echo "✓ Reconstruction already exists: $RECON_SIZE"
    read -p "Re-run reconstruction? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping reconstruction"
    else
        echo "Running reconstruction..."
        python3 reconstruct_from_repaint.py
        echo "✓ Reconstruction complete"
    fi
else
    echo "Reconstructing CT volume..."
    python3 reconstruct_from_repaint.py
    RECON_SIZE=$(du -h "$RECON_FILE" | cut -f1)
    echo "✓ Reconstruction complete: $RECON_SIZE"
fi

echo ""
echo "======================================================================"
echo "REPAINT WORKFLOW COMPLETE!"
echo "======================================================================"
echo "Results:"
echo "  Infilled tiles: $OUTPUT_DIR"
echo "  Merged sinograms: $MERGED_DIR"
echo "  Reconstructed volume: $RECON_FILE"
echo ""
echo "Next steps:"
echo "  1. Compare with ground truth: gt_reconstruction.vff"
echo "  2. Evaluate with MTF/NPS/NEQ metrics"
echo "  3. Compare with LaMa results using MTF_comparison_plotting.py"
echo "======================================================================"
