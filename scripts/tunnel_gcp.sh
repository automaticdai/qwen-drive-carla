#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/tunnel_gcp.sh PROJECT ZONE INSTANCE" >&2
    exit 2
fi
connection=(--project "$1" --zone "$2")
if [[ ${GCP_IAP:-0} == 1 ]]; then connection+=(--tunnel-through-iap); fi
exec gcloud compute ssh "$3" "${connection[@]}" -- \
    -N -L 127.0.0.1:8765:127.0.0.1:8765 \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3
