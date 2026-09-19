#!/usr/bin/env bash
# Separate environment: CUDA 12.9 kernels must not change the planner runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
"$CUDA_HOME/bin/nvcc" --version
if ! command -v uv >/dev/null; then export PATH="$PWD/.bootstrap/bin:$PATH"; fi
uv venv --python 3.12 --allow-existing .venv-bev
uv pip install --python .venv-bev/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-bev/bin/python 'transformers==5.14.1' 'accelerate==1.12.0' 'safetensors==0.8.0' 'numpy==2.2.6' 'pillow==12.0.0' 'opencv-python==5.0.0.93' 'matplotlib==3.10.7' ninja setuptools wheel
uv pip install --python .venv-bev/bin/python --no-deps -e . -e vendor/qwen-drive
.venv/bin/python scripts/download_model.py --include-rl --include-perception
