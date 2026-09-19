#!/usr/bin/env bash
# Deploy to an EXISTING GPU VM. Does not create/start a VM or change firewall rules.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/deploy_gcp.sh PROJECT ZONE INSTANCE" >&2
    exit 2
fi
cloud_project=$1
cloud_zone=$2
cloud_instance=$3
for value in "$cloud_project" "$cloud_zone" "$cloud_instance"; do
    [[ "$value" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "Invalid cloud identifier" >&2; exit 2; }
done
cd "$(dirname "$0")/.."
release="qwen-drive-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p outputs/cloud-packages
archive="outputs/cloud-packages/$release.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$archive" \
    pyproject.toml requirements-qwen.txt README.md src scripts tests docs scenarios
# Set GCP_IAP=1 when the VM is reachable only through Identity-Aware Proxy.
connection=(--project "$cloud_project" --zone "$cloud_zone")
if [[ ${GCP_IAP:-0} == 1 ]]; then connection+=(--tunnel-through-iap); fi
gcloud compute scp "${connection[@]}" "$archive" "$cloud_instance:~/$release.tar.gz"
gcloud compute ssh "${connection[@]}" "$cloud_instance" --command \
    "mkdir -p '$release' && tar -xzf '$release.tar.gz' -C '$release' && cd '$release' && bash scripts/setup_cloud.sh"
printf 'Deployment directory on VM: ~/%s\n' "$release"
printf 'Start service: cd ~/%s && bash scripts/start_remote_service.sh quality\n' "$release"
