"""
SRE-Nidaan: The Brain — vLLM Inference Server
================================================
OpenAI-compatible inference endpoint serving the NEXUS-CAUSAL v3.1
LoRA adapter on top of Mistral-7B-Instruct-v0.2.

Designed for GPU execution (Colab / cloud VM with ≥16 GB VRAM).
Exposes a public URL via pyngrok for split-compute integration.
"""

import os
import asyncio
import uuid
import json
import logging
from typing import Optional

import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyngrok import ngrok, conf as ngrok_conf

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
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_ADAPTER_PATH = os.environ.get(
    "NEXUS_LORA_PATH",
    "./nexus-causal-v3.1-lora",  # default local path
)
LORA_ADAPTER_NAME = "nexus-causal-v3.1"
MAX_LORA_RANK = 64          # ← caps cudaMemcpyAsync during adapter swaps
MAX_MODEL_LEN = 2048
DTYPE = "half"
QUANTIZATION = "awq"        # 4-bit quantization for memory efficiency
PORT = 8000

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


async def init_engine() -> None:
    """Initialize the vLLM async engine with LoRA support."""
    global engine

    if not VLLM_AVAILABLE:
        logger.warning("vLLM unavailable — engine init skipped (mock mode).")
        return

    engine_args = AsyncEngineArgs(
        model=BASE_MODEL,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        quantization=QUANTIZATION,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=2,                       # slots for live adapters
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    logger.info(
        "Initializing vLLM engine: base=%s | lora_rank_cap=%d | quant=%s",
        BASE_MODEL, MAX_LORA_RANK, QUANTIZATION,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM engine ready.")


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
# Mistral [INST] Prompt Formatter
# ---------------------------------------------------------------------------

def format_mistral_prompt(messages: list[ChatMessage]) -> str:
    """
    Converts a list of chat messages into Mistral-7B-Instruct-v0.2 format.
    Wraps system + user content inside [INST] / [/INST] tags.
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            user_block = msg.content
            if system_parts:
                # Prepend system context inside the first [INST] block
                user_block = "\n".join(system_parts) + "\n\n" + user_block
                system_parts.clear()
            conversation_parts.append(f"<s>[INST] {user_block} [/INST]")
        elif msg.role == "assistant":
            conversation_parts.append(f" {msg.content}</s>")

    return "".join(conversation_parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint with LoRA routing."""

    prompt = format_mistral_prompt(request.messages)

    # ── Mock path (no GPU / vLLM) ──────────────────────────────────────
    if engine is None:
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
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "base_model": BASE_MODEL,
        "lora_adapter": LORA_ADAPTER_NAME,
        "max_lora_rank": MAX_LORA_RANK,
    }


# ---------------------------------------------------------------------------
# Startup: Engine + Ngrok Tunnel
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    await init_engine()

    # ── Ngrok tunnel ───────────────────────────────────────────────────
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token:
        ngrok_conf.get_default().auth_token = ngrok_token
        public_url = ngrok.connect(PORT, "http").public_url
        logger.info("🧠 Brain is LIVE at: %s", public_url)
        logger.info("   └─ POST %s/v1/chat/completions", public_url)
    else:
        logger.warning(
            "NGROK_AUTHTOKEN not set — serving on localhost:%d only.", PORT
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
