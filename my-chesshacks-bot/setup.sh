#!/bin/bash
set -e

echo "=== Updating system ==="
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git wget unzip htop tmux



echo "=== Creating venv ==="
python3.11 -m venv rocm-env
source rocm-env/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch ROCm build ==="
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

echo "=== Installing Python deps ==="
pip install python-chess tqdm numpy

echo "=== Install Stockfish ==="
wget https://stockfishchess.org/files/stockfish-ubuntu-x86-64-avx2.zip
unzip stockfish-ubuntu-x86-64-avx2.zip
mv stockfish* stockfish
chmod +x stockfish

echo "=== Verify ROCm GPU visibility ==="
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA/ROCm available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("GPU", i, torch.cuda.get_device_name(i))
EOF
