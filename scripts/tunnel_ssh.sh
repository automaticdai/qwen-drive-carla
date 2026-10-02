#!/usr/bin/env bash
# Forward local 127.0.0.1:8765 to the inference service on an SSH host.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/tunnel_ssh.sh SSH_HOST" >&2
    exit 2
fi
exec ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$1"
