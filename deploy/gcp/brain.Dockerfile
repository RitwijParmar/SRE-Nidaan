FROM vllm/vllm-openai:latest

WORKDIR /app

RUN pip install --no-cache-dir "huggingface_hub<1.0"

COPY scripts/08_prepare_production_adapter.py /app/scripts/08_prepare_production_adapter.py
COPY deploy/gcp/brain-start.sh /app/brain-start.sh

RUN chmod +x /app/brain-start.sh

ENV MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
ENV PRODUCTION_ADAPTER_DIR=/models/production_adapter
ENV HF_PRODUCTION_REPO_ID=ritwijar/SRE-Nidaan-Production
ENV PRODUCTION_ARTIFACT_LABEL=checkpoint-1064

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/brain-start.sh"]
