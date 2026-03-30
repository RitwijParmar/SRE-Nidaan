#!/bin/bash
set -euo pipefail

export PRODUCTION_ADAPTER_DIR="${PRODUCTION_ADAPTER_DIR:-/models/production_adapter}"
export HF_PRODUCTION_REPO_ID="${HF_PRODUCTION_REPO_ID:-ritwijar/SRE-Nidaan-Production}"
export PRODUCTION_ARTIFACT_LABEL="${PRODUCTION_ARTIFACT_LABEL:-checkpoint-1064}"

mkdir -p "${PRODUCTION_ADAPTER_DIR}"

python3 /app/scripts/08_prepare_production_adapter.py

exec python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK:-64}" \
  --lora-modules "sre-lora=${PRODUCTION_ADAPTER_DIR}" \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
