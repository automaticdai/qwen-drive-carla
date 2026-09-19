#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
revision=28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -e '.[carla,test]' huggingface-hub
.venv/bin/python scripts/download_carla_agents.py
if [[ ! -d vendor/qwen-drive ]]; then
    git clone https://github.com/QwenLM/Qwen-Drive-1.0.git vendor/qwen-drive
    git -C vendor/qwen-drive checkout "$revision"
fi
if [[ "$(git -C vendor/qwen-drive rev-parse HEAD)" != "$revision" ]]; then
    echo "Unexpected upstream revision; inspect vendor/qwen-drive before proceeding." >&2
    exit 1
fi
uv venv --python 3.12 --allow-existing .venv-qwen
# Minimal SDPA runtime. Native optional acceleration extensions are deliberately
# omitted: the machine's CUDA 12.0 compiler cannot target this Blackwell GPU.
uv pip install --python .venv-qwen/bin/python -r requirements-qwen.txt
uv pip install --python .venv-qwen/bin/python --no-deps -e . -e vendor/qwen-drive
uv pip install --python .venv-qwen/bin/python 'carla==0.9.16'
