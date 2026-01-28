#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=file_name
#SBATCH --output=projection_infilling_training.out
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=wiegmann@phas.ubc.ca
#SBATCH --mail-type=ALL

set -euo pipefail

PROJECT='/home/wiegmann/projects/def-nlford/wiegmann/ct_recon'

cd $SLURM_TMPDIR
git clone --depth=1 git@github.com:falkwiegmann/ct_recon.git

df -h $SLURM_TMPDIR

cd ct_recon

#––– bring your scans folder into the tmp directory –––
mkdir -p data/scans data/models

CURRENT_TIME=$(date "+%Y.%m.%d-%H(hr).%M(min)")
echo "File transfer start time : $CURRENT_TIME"

cp -r $PROJECT/data/scans/. data/scans/
cp -r $PROJECT/data/models/. data/models/

CURRENT_TIME=$(date "+%Y.%m.%d-%H(hr).%M(min)")
echo "File transfer end time : $CURRENT_TIME"

module purge
module load cuda
module load cudnn
module load python scipy-stack

source ~/Python_virtual_env/bin/activate

export WANDB_API_KEY=80c734bef3fa29149d6d5b037a429991ec709fa0

wandb login 80c734bef3fa29149d6d5b037a429991ec709fa0

echo "Host: $(hostname)"
echo "GPU info:"; nvidia-smi
echo "Python:"; python --version

CURRENT_TIME=$(date "+%Y.%m.%d-%H(hr).%M(min)")
echo "Training Start Time : $CURRENT_TIME"

python projection_infilling_create_train_test_split.py --train_test_split=0.9 --desired_scans_in_testing=['Scan_1680','Scan_1681','Scan_1544'] --number_of_scans_in_total=80 || true

python projection_infilling_training.py --epochs=75 --batch_size=3 --load_model_path=data/models/model_epoch_30_of_31.pth || true

CURRENT_TIME=$(date "+%Y.%m.%d-%H(hr).%M(min)")
echo "Training End Time : $CURRENT_TIME"

#––– copy your trained-model folder back to your project tree –––
mkdir -p $PROJECT/data/models
cp -r data/models/. $PROJECT/data/models/
cp -r data/scans/training_testing_split.csv $PROJECT/data/scans/training_testing_split.csv
cp -r projection_infilling_training.out $PROJECT/projection_infilling_training.out

# commit and push changes to the repository
git add .
git commit -m "Compute Canada run commit"
git push -u origin main
