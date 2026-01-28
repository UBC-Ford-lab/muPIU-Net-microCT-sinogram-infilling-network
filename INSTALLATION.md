# Installation Guide

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended for large reconstructions)
- ~50GB disk space (for data and intermediate files)

## Local Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/UBC-Ford-lab/muPIU-Net-microCT-sinogram-infilling-network.git
cd muPIU-Net-microCT-sinogram-infilling-network
```

If you already cloned without `--recursive`, initialize submodules:

```bash
git submodule update --init --recursive
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from ct_core import vff_io; print('ct_core OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## HPC Installation (Compute Canada / SLURM)

### 1. Load Required Modules

```bash
module load python/3.10 cuda/11.8 cudnn/8.6 scipy-stack
```

### 2. Create Virtual Environment

```bash
virtualenv --no-download ~/Python_virtual_env
source ~/Python_virtual_env/bin/activate
pip install --no-index --upgrade pip
```

### 3. Install PyTorch

```bash
pip install --no-index torch torchvision
```

### 4. Install Other Dependencies

```bash
pip install --no-index numpy scipy scikit-image pillow matplotlib tqdm
pip install xmltodict opencv-python wandb
```

### 5. Set Environment Variable

Add to your `.bashrc`:

```bash
export VENV_PATH="$HOME/Python_virtual_env"
```

### 6. Verify GPU Access

```bash
salloc --gres=gpu:1 --time=0:10:00 --mem=8G
python -c "import torch; print(torch.cuda.get_device_name(0))"
exit
```

## Base Model Setup

Each base model requires additional setup:

### LaMa

```bash
cd base_models/models/lama/lama-repo

# Download pre-trained model
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
rm big-lama.zip

# Install LaMa dependencies
pip install -r requirements.txt
```

### MAT

```bash
cd base_models/models/mat/MAT

# Download checkpoint (via gdown)
pip install gdown
gdown https://drive.google.com/uc?id=1M3AFy7x9DqXaI-fINSynW7FJSXYROfv-
```

### DeepFill v2

```bash
cd base_models/models/deepfill/DeepFillv2

# Download model checkpoint
gdown https://drive.google.com/uc?id=1dyPD2hx0JTmMuHHS8s4LDkkF4vMpBV9E
```

### RePaint

```bash
cd base_models/models/repaint/RePaint

# Download pre-trained diffusion model
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX
mkdir -p data/pretrained
mv celeba256_250000.pt data/pretrained/
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in training scripts or use smaller tile sizes for base models.

### Import Errors

Ensure the project root is in your Python path:

```python
import sys
sys.path.insert(0, '/path/to/muPIU-Net-microCT-sinogram-infilling-network')
```

### VFF File Loading Issues

VFF files use big-endian byte order. Ensure byteswap is applied on little-endian systems (handled automatically by `ct_core.vff_io`).

### Submodule Issues

If submodules are empty:

```bash
git submodule update --init --recursive --force
```
