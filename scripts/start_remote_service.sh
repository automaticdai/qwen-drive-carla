#!/usr/bin/env bash
# Run this on the cloud VM inside the deployed release.
set -euo pipefail
cd "$(dirname "$0")/.."
runtime='.venv-qwen/bin/python -m qwen_drive_carla.remote'
case "${1:-quality}" in
    quality) profile='--precision bf16 --planner sft --planning-mode direct --image-profile high' ;;
    reasoning) profile='--precision bf16 --planner rl --planning-mode reasoning --image-profile high' ;;
    baseline) profile='--precision nf4 --planner sft --planning-mode direct --image-profile small' ;;
    debug)
        runtime='.venv-bev/bin/python -m qwen_drive_carla.debug_inference'
        profile='--image-profile high'
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
        export PATH="$PWD/.venv-bev/bin:$CUDA_HOME/bin:$PATH"
        export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
        ;;
    *) echo 'Usage: bash scripts/start_remote_service.sh [quality|reasoning|baseline|debug]' >&2; exit 2 ;;
esac
if tmux has-session -t qwen-drive-agent 2>/dev/null; then
    echo "tmux session qwen-drive-agent already exists; inspect it before starting another service." >&2
    exit 1
fi
if [[ -f service.log ]]; then cp service.log "service-$(date -u +%Y%m%dT%H%M%SZ).log"; fi
tmux new-session -d -s qwen-drive-agent -c "$PWD" \
    "exec $runtime $profile --port 8765 > service.log 2>&1"
echo "Started tmux session qwen-drive-agent; readiness and errors are in $PWD/service.log"
echo "Stop with: tmux send-keys -t qwen-drive-agent C-c"
