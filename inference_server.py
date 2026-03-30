"""
SRE-Nidaan: The Brain — vLLM Inference Server
================================================
OpenAI-compatible inference endpoint serving the NEXUS-CAUSAL v3.1
LoRA adapter on top of the configured instruct model.

Designed for GPU execution (Colab / cloud VM with ≥16 GB VRAM).
Exposes a public URL via pyngrok for split-compute integration.
"""

import os
import asyncio
import uuid
import json
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.model_utils import build_chat_prompt

try:
    from pyngrok import ngrok, conf as ngrok_conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# ---------------------------------------------------------------------------
# vLLM imports (deferred so the script still parses on CPU-only machines)
# ---------------------------------------------------------------------------
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not installed — running in MOCK mode for local dev.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
LORA_ADAPTER_PATH = os.environ.get(
    "NEXUS_LORA_PATH",
    os.environ.get("PRODUCTION_ADAPTER_DIR", "/models/production_adapter"),
)
LORA_ADAPTER_NAME = os.environ.get("NEXUS_LORA_NAME", "sre-nidaan-production")
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "64"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "1024"))
DTYPE = os.environ.get("VLLM_DTYPE", "half")
QUANTIZATION = os.environ.get("VLLM_QUANTIZATION") or None
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.82"))
ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "1") == "1"
ALLOW_MOCK_BRAIN = os.environ.get("ALLOW_MOCK_BRAIN", "0") == "1"
PORT = int(os.environ.get("PORT", "8000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sre-nidaan-brain")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SRE-Nidaan Brain — vLLM Inference",
    version="1.0.0",
    description="OpenAI-compatible causal inference endpoint with dynamic LoRA routing.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = BASE_MODEL
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.95
    use_lora: bool = Field(
        default=True,
        description="If true, route inference through the NEXUS-CAUSAL LoRA adapter.",
    )


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    model: str = BASE_MODEL
    choices: list[ChatCompletionChoice]


# ---------------------------------------------------------------------------
# Engine Initialization
# ---------------------------------------------------------------------------
engine: Optional[AsyncLLMEngine] = None
engine_loading_task: Optional[asyncio.Task] = None
engine_error: Optional[str] = None


async def init_engine() -> None:
    """Initialize the vLLM async engine with LoRA support."""
    global engine
    global engine_error

    if not VLLM_AVAILABLE:
        if ALLOW_MOCK_BRAIN:
            logger.warning("vLLM unavailable — engine init skipped (mock mode).")
            return
        raise RuntimeError("vLLM is not installed in this runtime.")

    engine_args = AsyncEngineArgs(
        model=BASE_MODEL,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        quantization=QUANTIZATION,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=2,                       # slots for live adapters
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
    )

    logger.info(
        "Initializing vLLM engine: base=%s | lora_rank_cap=%d | quant=%s",
        BASE_MODEL, MAX_LORA_RANK, QUANTIZATION,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_error = None
    logger.info("vLLM engine ready.")


async def _bootstrap_engine() -> None:
    global engine_error

    try:
        await init_engine()
    except Exception as exc:
        engine_error = str(exc)
        logger.exception("Brain engine initialization failed: %s", exc)


# ---------------------------------------------------------------------------
# LoRA Resolver
# ---------------------------------------------------------------------------

def resolve_lora(use_lora: bool) -> Optional[LoRARequest]:
    """Dynamically resolve the LoRA adapter for NEXUS-CAUSAL inference."""
    if not use_lora or not VLLM_AVAILABLE:
        return None

    return LoRARequest(
        lora_name=LORA_ADAPTER_NAME,
        lora_int_id=1,               # unique numeric ID for this adapter slot
        lora_path=LORA_ADAPTER_PATH,
    )


# ---------------------------------------------------------------------------
# Prompt Formatter
# ---------------------------------------------------------------------------

def format_prompt(messages: list[ChatMessage]) -> str:
    """Render chat turns using the active model's native prompt format."""
    return build_chat_prompt(
        [{"role": message.role, "content": message.content} for message in messages],
        model_name=BASE_MODEL,
        tokenizer=None,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint with LoRA routing."""

    prompt = format_prompt(request.messages)

    # ── Mock path (no GPU / vLLM) ──────────────────────────────────────
    if engine is None:
        if not ALLOW_MOCK_BRAIN:
            detail = "Brain is warming up."
            if engine_error:
                detail = f"Brain initialization failed: {engine_error}"
            raise HTTPException(status_code=503, detail=detail)

        mock_response = json.dumps({
            "root_cause": "Database connection pool exhaustion caused by auth_service retry storm",
            "intervention_simulation": (
                "do(Scale Up Auth_Service) is a confounding error: adding replicas "
                "increases concurrent DB connections, accelerating pool exhaustion "
                "and triggering cascading 503 failures across all upstream services."
            ),
            "recommended_action": "Rate limit frontend and increase DB max_connections",
            "dag_nodes": [
                {"id": "frontend", "label": "Frontend (503)"},
                {"id": "auth_service", "label": "Auth Service (96% CPU)"},
                {"id": "database", "label": "Database (99% Conn)"},
                {"id": "retry_storm", "label": "Retry Storm"},
                {"id": "conn_exhaustion", "label": "Connection Exhaustion"},
            ],
            "dag_edges": [
                {"id": "e1", "source": "frontend", "target": "auth_service", "animated": True},
                {"id": "e2", "source": "auth_service", "target": "retry_storm", "animated": True},
                {"id": "e3", "source": "retry_storm", "target": "database", "animated": True},
                {"id": "e4", "source": "database", "target": "conn_exhaustion", "animated": True},
                {"id": "e5", "source": "conn_exhaustion", "target": "frontend", "animated": True},
            ],
        }, indent=2)

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=mock_response)
                )
            ],
        )

    # ── Live vLLM path ─────────────────────────────────────────────────
    sampling = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
    )
    lora_request = resolve_lora(request.use_lora)
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    generated_text = ""
    async for output in engine.generate(prompt, sampling, request_id, lora_request=lora_request):
        if output.outputs:
            generated_text = output.outputs[0].text

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=generated_text.strip())
            )
        ],
    )


@app.get("/health")
async def health():
    if engine is not None:
        status = "ready"
    elif engine_error:
        status = "error"
    else:
        status = "warming"

    return {
        "status": status,
        "engine_loaded": engine is not None,
        "engine_loading": engine is None and engine_error is None,
        "engine_error": engine_error,
        "base_model": BASE_MODEL,
        "lora_adapter": LORA_ADAPTER_NAME,
        "max_lora_rank": MAX_LORA_RANK,
    }


# ---------------------------------------------------------------------------
# Startup: Engine + Ngrok Tunnel
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global engine_loading_task

    engine_loading_task = asyncio.create_task(_bootstrap_engine())

    # ── Ngrok tunnel ───────────────────────────────────────────────────
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token and NGROK_AVAILABLE:
        ngrok_conf.get_default().auth_token = ngrok_token
        public_url = ngrok.connect(PORT, "http").public_url
        logger.info("🧠 Brain is LIVE at: %s", public_url)
        logger.info("   └─ POST %s/v1/chat/completions", public_url)
    elif ngrok_token:
        logger.warning("NGROK_AUTHTOKEN set but pyngrok is not installed.")
    else:
        logger.warning(
            "NGROK_AUTHTOKEN not set — serving on localhost:%d only.", PORT
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
