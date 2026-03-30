#!/bin/bash
set -euo pipefail

GCLOUD_BIN="${GCLOUD_BIN:-$HOME/google-cloud-sdk/bin/gcloud}"
PROJECT_ID="${PROJECT_ID:-$($GCLOUD_BIN config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-east4}"
ARTIFACT_REPO="${ARTIFACT_REPO:-sre-nidaan}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN_SECRET_NAME="${HF_TOKEN_SECRET_NAME:-hf-token}"
HF_PRODUCTION_REPO_ID="${HF_PRODUCTION_REPO_ID:-ritwijar/SRE-Nidaan-Production}"
PRODUCTION_ARTIFACT_LABEL="${PRODUCTION_ARTIFACT_LABEL:-checkpoint-1064}"
BRAIN_STARTUP_PROBE="${BRAIN_STARTUP_PROBE:-timeoutSeconds=240,periodSeconds=240,failureThreshold=10,tcpSocket.port=8000}"

if [[ ! -x "$GCLOUD_BIN" ]]; then
  echo "gcloud CLI not found at $GCLOUD_BIN" >&2
  exit 1
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID or run 'gcloud init' first." >&2
  exit 1
fi

IMAGE_PREFIX="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}"
BODY_IMAGE="${IMAGE_PREFIX}/sre-nidaan-body:${TAG}"
FACE_IMAGE="${IMAGE_PREFIX}/sre-nidaan-face:${TAG}"
BRAIN_IMAGE="${IMAGE_PREFIX}/sre-nidaan-brain:${TAG}"

"$GCLOUD_BIN" services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com

if "$GCLOUD_BIN" secrets describe "$HF_TOKEN_SECRET_NAME" >/dev/null 2>&1; then
  if [[ -n "$HF_TOKEN" ]]; then
    printf "%s" "$HF_TOKEN" | "$GCLOUD_BIN" secrets versions add "$HF_TOKEN_SECRET_NAME" --data-file=-
  fi
elif [[ -n "$HF_TOKEN" ]]; then
  printf "%s" "$HF_TOKEN" | "$GCLOUD_BIN" secrets create "$HF_TOKEN_SECRET_NAME" --data-file=-
else
  echo "Set HF_TOKEN for the first deploy so the Hugging Face secret can be created." >&2
  exit 1
fi

if ! "$GCLOUD_BIN" artifacts repositories describe "$ARTIFACT_REPO" --location "$REGION" >/dev/null 2>&1; then
  "$GCLOUD_BIN" artifacts repositories create "$ARTIFACT_REPO" \
    --repository-format=docker \
    --location "$REGION" \
    --description="SRE-Nidaan Cloud Run images"
fi

"$GCLOUD_BIN" builds submit \
  --config deploy/gcp/cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_REPO="${ARTIFACT_REPO}",_TAG="${TAG}"

"$GCLOUD_BIN" run deploy sre-nidaan-brain \
  --image "$BRAIN_IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --command /bin/bash \
  --args /app/brain-start.sh \
  --port 8000 \
  --execution-environment gen2 \
  --cpu 8 \
  --memory 32Gi \
  --no-cpu-throttling \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --concurrency 1 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 3600 \
  --startup-probe="${BRAIN_STARTUP_PROBE}" \
  --set-env-vars "MODEL_ID=${MODEL_ID},HF_PRODUCTION_REPO_ID=${HF_PRODUCTION_REPO_ID},PRODUCTION_ARTIFACT_LABEL=${PRODUCTION_ARTIFACT_LABEL},MAX_MODEL_LEN=1024,GPU_MEMORY_UTILIZATION=0.82,VLLM_ENFORCE_EAGER=1,SERVING_BACKEND=transformers" \
  --set-secrets "HF_TOKEN=${HF_TOKEN_SECRET_NAME}:latest"

BRAIN_URL="$("$GCLOUD_BIN" run services describe sre-nidaan-brain --region "$REGION" --format='value(status.url)')"

"$GCLOUD_BIN" run deploy sre-nidaan-body \
  --image "$BODY_IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8001 \
  --cpu 1 \
  --memory 1Gi \
  --max-instances 5 \
  --timeout 300 \
  --set-env-vars "VLLM_ENDPOINT=${BRAIN_URL}/v1,MODEL_ID=${MODEL_ID},PRODUCTION_ARTIFACT_LABEL=${PRODUCTION_ARTIFACT_LABEL},GENERATION_CANDIDATES=1,VLLM_REQUEST_TIMEOUT_SECONDS=45,VLLM_MAX_RETRIES=0"

BODY_URL="$("$GCLOUD_BIN" run services describe sre-nidaan-body --region "$REGION" --format='value(status.url)')"

"$GCLOUD_BIN" run deploy sre-nidaan-face \
  --image "$FACE_IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 3000 \
  --cpu 1 \
  --memory 512Mi \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "NEXT_PUBLIC_API_URL=${BODY_URL}"

FACE_URL="$("$GCLOUD_BIN" run services describe sre-nidaan-face --region "$REGION" --format='value(status.url)')"

echo "FACE_URL=${FACE_URL}"
echo "BODY_URL=${BODY_URL}"
echo "BRAIN_URL=${BRAIN_URL}"
