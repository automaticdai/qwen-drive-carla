#!/usr/bin/env bash
# Deploy to an existing SSH-reachable GPU host (e.g. cheddar0). Changes nothing outside the release directory.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/deploy_ssh.sh SSH_HOST" >&2
    exit 2
fi
host=$1
[[ "$host" =~ ^[a-zA-Z0-9._@-]+$ ]] || { echo "Invalid SSH host" >&2; exit 2; }
cd "$(dirname "$0")/.."
release="qwen-drive-$(date -u +%Y%m%dT%H%M%SZ)-$$"
rsync -az --exclude='__pycache__' --exclude='*.pyc' \
    pyproject.toml requirements-qwen.txt README.md src scripts tests docs scenarios "$host:$release/"
ssh "$host" "cd '$release' && bash scripts/setup_cloud.sh"
printf 'Deployment directory on %s: ~/%s\n' "$host" "$release"
printf 'Start service: ssh %s "cd ~/%s && bash scripts/start_remote_service.sh quality"\n' "$host" "$release"
