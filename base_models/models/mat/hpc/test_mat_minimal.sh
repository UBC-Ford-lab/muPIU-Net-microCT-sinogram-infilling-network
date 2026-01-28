#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=mat_test
#SBATCH --output=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_test_%j.out
#SBATCH --error=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/mat_test_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Minimal test - no strict error handling
echo "=== MAT Minimal Test ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo ""
echo "SLURM variables:"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  SLURM_TMPDIR: $SLURM_TMPDIR"
echo ""
echo "Checking PROJECT path..."
PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
if [ -d "$PROJECT" ]; then
    echo "  PROJECT exists: $PROJECT"
    ls -la "$PROJECT" | head -10
else
    echo "  ERROR: PROJECT does not exist: $PROJECT"
fi
echo ""
echo "Checking sinograms_masked..."
if [ -d "$PROJECT/repaint/sinogram_tiles/sinograms_masked" ]; then
    TILE_COUNT=$(find "$PROJECT/repaint/sinogram_tiles/sinograms_masked" -name "*.png" 2>/dev/null | wc -l)
    echo "  sinograms_masked exists with $TILE_COUNT tiles"
else
    echo "  ERROR: sinograms_masked does not exist"
fi
echo ""
echo "Checking git..."
which git && git --version
echo ""
echo "Checking python..."
which python3 && python3 --version
echo ""
echo "Checking GPU..."
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
echo ""
echo "=== Test Complete ==="
