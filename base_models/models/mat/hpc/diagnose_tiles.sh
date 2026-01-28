#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=mat_diagnose
#SBATCH --output=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/diagnose_tiles_%j.out
#SBATCH --error=/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison/logs/diagnose_tiles_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# MAT Tile Diagnostic Script
# Checks if MAT output tiles contain zeros before merging
# No GPU required - just file I/O and numpy

set -eo pipefail
trap 'echo "ERROR at line $LINENO: $BASH_COMMAND failed with exit code $?"' ERR

echo "============================================================"
echo "MAT Tile Zero Pixel Diagnostic"
echo "============================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Setup paths
PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon/Base_model_comparison'
SCRIPT_DIR="$PROJECT/models/mat/scripts"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT/logs"

# Verify project directory exists
if [ ! -d "$PROJECT" ]; then
    echo "ERROR: PROJECT directory not found: $PROJECT"
    exit 1
fi

# Verify script directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Script directory not found: $SCRIPT_DIR"
    exit 1
fi

# Load modules with fallbacks
echo "Loading modules..."
module load python/3.10 2>/dev/null || module load python/3.11 2>/dev/null || module load python 2>/dev/null || {
    echo "ERROR: Could not load python module"
    exit 1
}
module load scipy-stack 2>/dev/null || echo "WARNING: scipy-stack not available, continuing..."

# Activate virtual environment with fallback
if [ -f ~/Python_virtual_env/bin/activate ]; then
    source ~/Python_virtual_env/bin/activate
    echo "Activated virtual environment: ~/Python_virtual_env"
else
    echo "WARNING: Virtual env not found at ~/Python_virtual_env, using system Python"
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Verify diagnostic script exists
if [ ! -f "$SCRIPT_DIR/diagnose_tiles_hpc.py" ]; then
    echo "ERROR: Diagnostic script not found: $SCRIPT_DIR/diagnose_tiles_hpc.py"
    exit 1
fi

# Run diagnostic script
cd "$SCRIPT_DIR"
python diagnose_tiles_hpc.py

echo ""
echo "============================================================"
echo "Diagnostic Complete: $(date)"
echo "============================================================"
