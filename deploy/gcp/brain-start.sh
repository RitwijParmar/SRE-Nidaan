#!/bin/bash
set -euo pipefail

export PRODUCTION_ADAPTER_DIR="${PRODUCTION_ADAPTER_DIR:-/models/production_adapter}"
export HF_PRODUCTION_REPO_ID="${HF_PRODUCTION_REPO_ID:-ritwijar/SRE-Nidaan-Production}"
export PRODUCTION_ARTIFACT_LABEL="${PRODUCTION_ARTIFACT_LABEL:-checkpoint-1064}"

mkdir -p "${PRODUCTION_ADAPTER_DIR}"

python3 /app/scripts/08_prepare_production_adapter.py

exec python3 /app/inference_server.py
