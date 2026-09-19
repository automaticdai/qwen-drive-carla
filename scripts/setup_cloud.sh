#!/usr/bin/env bash
# Called inside the isolated release directory on a Linux NVIDIA GPU VM.
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi
if ! command -v uv >/dev/null; then
    python3 -m venv .bootstrap
    .bootstrap/bin/python -m pip install uv
    export PATH="$PWD/.bootstrap/bin:$PATH"
fi
bash scripts/setup.sh
.venv/bin/python scripts/download_model.py --include-rl
.venv-qwen/bin/python - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('This profile requires a CUDA GPU with BF16 support')
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
b = a @ a
print('GPU:', torch.cuda.get_device_name(0), 'BF16 matmul:', tuple(b.shape))
PY
