#!/bin/bash
#SBATCH --time=02:00:00                 # 2 hours for merging
#SBATCH --job-name=repaint_merge
#SBATCH --output=logs/repaint_merge_%j.out
#SBATCH --error=logs/repaint_merge_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=ALL

set -euo pipefail

#==============================================================================
# Merge RePaint Results from Array Jobs
# Combines split results and reconstructs full sinograms
# Works on any Compute Canada cluster
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
echo "RePaint Results Merger"
echo "========================================================================"
echo "Cluster: ${CLUSTER}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT}"
echo "Start time: $(date)"
echo "========================================================================"

cd $SLURM_TMPDIR

#==============================================================================
# SETUP
#==============================================================================

echo ""
echo "Cloning repository..."
git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git
cd ct_recon/Base_model_comparison

echo ""
echo "Loading modules..."
module purge
module load python/3.10
module load scipy-stack

echo "Activating virtual environment..."
source ~/Python_virtual_env/bin/activate

#==============================================================================
# TRANSFER SPLIT RESULTS
#==============================================================================

echo ""
echo "========================================================================"
echo "Transferring split results..."
echo "========================================================================"

# Copy all split results
for i in {0..7}; do
    SPLIT_NUM=$(printf '%02d' $i)
    echo "Copying split ${SPLIT_NUM}..."
    mkdir -p repaint/tiles_infilled_split_${SPLIT_NUM}
    rsync -av $PROJECT/repaint/tiles_infilled_split_${SPLIT_NUM}/ \
        repaint/tiles_infilled_split_${SPLIT_NUM}/
done

# Copy metadata
cp $PROJECT/repaint/sinogram_tiles/tiling_metadata.json repaint/sinogram_tiles/

echo "Transfer complete!"

#==============================================================================
# MERGE SPLIT RESULTS
#==============================================================================

echo ""
echo "========================================================================"
echo "Merging array job results..."
echo "========================================================================"

python merge_array_results.py \
    --split_base repaint \
    --output_dir repaint/tiles_infilled

echo "Merge complete!"

# Verify merged results
NUM_TILES=$(find repaint/tiles_infilled -name "*.png" | wc -l)
echo "Total merged tiles: ${NUM_TILES}"

#==============================================================================
# MERGE TILES BACK TO SINOGRAMS
#==============================================================================

echo ""
echo "========================================================================"
echo "Merging tiles back to full sinograms..."
echo "========================================================================"

python merge_repaint_tiles.py \
    --tiles_dir repaint/tiles_infilled \
    --metadata_file repaint/sinogram_tiles/tiling_metadata.json \
    --output_dir repaint/sinograms_infilled \
    --blend_mode nearest

echo "Sinogram merging complete!"

# Verify sinograms
NUM_SINOS=$(find repaint/sinograms_infilled -name "*.png" | wc -l)
echo "Total sinograms: ${NUM_SINOS}"

#==============================================================================
# FDK RECONSTRUCTION
#==============================================================================

echo ""
echo "========================================================================"
echo "Running FDK reconstruction..."
echo "========================================================================"

python reconstruct_from_repaint.py \
    --sinogram_dir repaint/sinograms_infilled \
    --output_dir repaint/reconstructed_volume

echo "Reconstruction complete!"

#==============================================================================
# TRANSFER FINAL RESULTS
#==============================================================================

echo ""
echo "========================================================================"
echo "Transferring final results to project directory..."
echo "========================================================================"

# Copy merged tiles
mkdir -p $PROJECT/repaint/tiles_infilled
rsync -av repaint/tiles_infilled/ $PROJECT/repaint/tiles_infilled/

# Copy merged sinograms
mkdir -p $PROJECT/repaint/sinograms_infilled
rsync -av repaint/sinograms_infilled/ $PROJECT/repaint/sinograms_infilled/

# Copy reconstructed volume
mkdir -p $PROJECT/repaint/reconstructed_volume
rsync -av repaint/reconstructed_volume/ $PROJECT/repaint/reconstructed_volume/

echo "Transfer complete!"

#==============================================================================
# SUMMARY
#==============================================================================

echo ""
echo "========================================================================"
echo "ALL PROCESSING COMPLETE!"
echo "========================================================================"
echo "End time: $(date)"
echo "Merged tiles: ${NUM_TILES}"
echo "Merged sinograms: ${NUM_SINOS}"
echo ""
echo "Output directories:"
echo "  Tiles: $PROJECT/repaint/tiles_infilled"
echo "  Sinograms: $PROJECT/repaint/sinograms_infilled"
echo "  Volume: $PROJECT/repaint/reconstructed_volume"
echo "========================================================================"
