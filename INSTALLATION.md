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

# Create pretrained directory and download checkpoint
mkdir -p pretrained
pip install gdown
gdown https://drive.google.com/uc?id=1M3AFy7x9DqXaI-fINSynW7FJSXYROfv- -O pretrained/CelebA-HQ_256.pkl
```

### DeepFill v2

```bash
cd base_models/models/deepfill/DeepFillv2

# Create pretrained directory and download checkpoint
mkdir -p pretrained
gdown https://drive.google.com/uc?id=1dyPD2hx0JTmMuHHS8s4LDkkF4vMpBV9E -O pretrained/states_tf_celebahq.pth
```

### RePaint

```bash
cd base_models/models/repaint/RePaint

# Create pretrained directory and download checkpoint
mkdir -p data/pretrained
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX -O data/pretrained/celeba256_250000.pt
```
