"""
SRE-Nidaan: The Body — FastAPI Agentic Backend
================================================
Model Context Protocol (MCP) router with:
  • Strict Pydantic-guided decoding (deterministic JSON)
  • Two-plane safety architecture (read-only copilot pattern)
  • Causal refutation test simulation
"""

import os
import json
import random
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "not-needed")  # vLLM doesn't require a key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sre-nidaan-body")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SRE-Nidaan Body — Agentic Backend",
    version="1.0.0",
    description="MCP router with blast-radius controls and causal inference orchestration.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas — Deterministic Structured Output
# ---------------------------------------------------------------------------

class DAGNode(BaseModel):
    id: str
    label: str


class DAGEdge(BaseModel):
    id: str
    source: str
    target: str
    animated: bool = True


class CausalAnalysisResponse(BaseModel):
    """Strict schema enforced via guided decoding — never rely on prompt alone."""
    root_cause: str = Field(
        ..., description="The structural root cause identified via causal inference."
    )
    intervention_simulation: str = Field(
        ...,
        description=(
            "Explain why do(Scale Up Auth_Service) is a confounding error "
            "that crashes the DB — Pearl's do-calculus reasoning."
        ),
    )
    recommended_action: str = Field(
        ..., description="The safe, causal intervention to resolve the incident."
    )
    dag_nodes: list[DAGNode] = Field(
        ..., description="Nodes in the causal Directed Acyclic Graph."
    )
    dag_edges: list[DAGEdge] = Field(
        ..., description="Edges in the causal DAG with animation flags."
    )


class IncidentAnalysisResult(BaseModel):
    """Full API response with safety-plane metadata."""
    analysis: CausalAnalysisResponse
    requires_human_approval: bool = True
    safety_plane: str = "read-only-copilot"
    telemetry_snapshot: dict
    refutation_status: str = "pending"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Mock MCP Tools — Data Layer
# ---------------------------------------------------------------------------

def fetch_system_telemetry() -> dict:
    """
    Simulates a DB lock cascading failure scenario.
    In production this would pull from Prometheus/Datadog/CloudWatch.
    """
    return {
        "frontend": {
            "status": "503 Gateway Timeout",
            "error_rate": "Spiking",
        },
        "auth_service": {
            "cpu_utilization": "96%",
            "latency_ms": 4500,
            "replicas": 5,
        },
        "database": {
            "connections": "990/1000 (99%)",
            "wait_event": "ClientRead (Locked)",
        },
    }


# ---------------------------------------------------------------------------
# Mistral [INST] Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are NEXUS-CAUSAL, an AI SRE agent trained on Pearl's Causal Hierarchy.
Analyze the telemetry data below using formal causal reasoning (do-calculus).
Identify the STRUCTURAL root cause — not correlational symptoms.
Explain why a naive intervention (e.g., scaling up auth_service) would WORSEN the incident.
Respond with a valid JSON object matching the provided schema EXACTLY."""


def build_mistral_prompt(telemetry: dict) -> str:
    """
    Constructs a Mistral-7B-Instruct-v0.2 formatted prompt.
    Uses [INST] / [/INST] token wrapping as required by the tokenizer.
    Prompt is CONCISE — CausalCoT degrades performance to 41%.
    """
    user_content = f"""Telemetry Snapshot:
{json.dumps(telemetry, indent=2)}

Task: Apply do-calculus to identify the root cause and produce a causal DAG.
Return ONLY the JSON object with: root_cause, intervention_simulation, recommended_action, dag_nodes, dag_edges."""

    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"


# ---------------------------------------------------------------------------
# Refutation Test — Observability / Day-2 SRE
# ---------------------------------------------------------------------------

async def run_refutation_test(analysis: CausalAnalysisResponse) -> dict:
    """
    Simulates a causal refutation test by injecting a random confounder
    or placebo treatment into the DAG, then checking if the LLM's
    causal estimate remains robust.

    In production, this would re-run inference with the modified DAG
    and compare the causal effect estimates.
    """
    confounder_types = [
        {"type": "random_common_cause", "node": "network_jitter", "label": "Network Jitter (Placebo)"},
        {"type": "placebo_treatment", "node": "cache_miss", "label": "Cache Miss Rate (Placebo)"},
        {"type": "data_permutation", "node": "dns_latency", "label": "DNS Latency (Random)"},
        {"type": "subset_removal", "node": "log_volume", "label": "Log Volume Spike (Subset)"},
    ]

    confounder = random.choice(confounder_types)

    # Simulate a delay as if running actual re-inference
    await asyncio.sleep(random.uniform(0.5, 2.0))

    # Mock robustness check
    original_edges = len(analysis.dag_edges)
    estimate_shift = round(random.uniform(-0.05, 0.05), 4)
    is_robust = abs(estimate_shift) < 0.03

    result = {
        "test_type": confounder["type"],
        "injected_confounder": confounder["label"],
        "original_root_cause": analysis.root_cause,
        "estimate_shift": estimate_shift,
        "is_robust": is_robust,
        "verdict": (
            "PASS — Causal estimate is stable under placebo perturbation."
            if is_robust
            else "WARN — Estimate shifted; manual review recommended."
        ),
    }

    logger.info("Refutation test complete: %s → %s", confounder["label"], result["verdict"])
    return result


# Store the latest refutation result for polling
_refutation_results: dict = {}

async def _run_refutation_background(analysis: CausalAnalysisResponse):
    result = await run_refutation_test(analysis)
    _refutation_results["latest"] = result


# ---------------------------------------------------------------------------
# Mock LLM Fallback (when vLLM is unreachable)
# ---------------------------------------------------------------------------

def get_mock_analysis() -> CausalAnalysisResponse:
    """Deterministic fallback when Brain is offline."""
    return CausalAnalysisResponse(
        root_cause=(
            "Database connection pool exhaustion (990/1000) caused by "
            "auth_service retry storm. The auth_service at 96% CPU is generating "
            "cascading retries that hold open DB connections with ClientRead locks."
        ),
        intervention_simulation=(
            "Applying do(Scale Up Auth_Service) by adding replicas is a "
            "confounding error: each new replica opens additional persistent DB "
            "connections, accelerating pool exhaustion from 99% → 100%. This "
            "triggers a hard connection limit, causing ALL services — including "
            "frontend — to receive 503 errors. The naive intervention transforms "
            "a degraded state into a full outage."
        ),
        recommended_action=(
            "1. Rate limit frontend to reduce inbound request volume. "
            "2. Increase DB max_connections from 1000 → 2000. "
            "3. Implement auth_service circuit breaker to halt retry storm. "
            "4. Drain locked ClientRead connections."
        ),
        dag_nodes=[
            DAGNode(id="frontend", label="Frontend (503 Gateway Timeout)"),
            DAGNode(id="auth_service", label="Auth Service (96% CPU)"),
            DAGNode(id="database", label="Database (99% Conn Pool)"),
            DAGNode(id="retry_storm", label="Retry Storm (Cascading)"),
            DAGNode(id="conn_exhaustion", label="Connection Exhaustion"),
            DAGNode(id="client_lock", label="ClientRead Lock"),
        ],
        dag_edges=[
            DAGEdge(id="e1", source="frontend", target="auth_service", animated=True),
            DAGEdge(id="e2", source="auth_service", target="retry_storm", animated=True),
            DAGEdge(id="e3", source="retry_storm", target="database", animated=True),
            DAGEdge(id="e4", source="database", target="conn_exhaustion", animated=True),
            DAGEdge(id="e5", source="conn_exhaustion", target="client_lock", animated=True),
            DAGEdge(id="e6", source="client_lock", target="frontend", animated=True),
        ],
    )


# ---------------------------------------------------------------------------
# OpenAI SDK Client (pointed at vLLM)
# ---------------------------------------------------------------------------

client = AsyncOpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=VLLM_API_KEY,
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/analyze-incident", response_model=IncidentAnalysisResult)
async def analyze_incident(background_tasks: BackgroundTasks):
    """
    POST /api/analyze-incident
    
    Read-only copilot pattern:
    1. Fetch telemetry snapshot
    2. Build Mistral-formatted prompt
    3. Call vLLM with Pydantic schema enforcement
    4. Return analysis with requires_human_approval: true
    5. Kick off refutation test in background
    """
    # ── Step 1: Fetch telemetry ────────────────────────────────────────
    telemetry = fetch_system_telemetry()
    logger.info("Telemetry fetched: %s", json.dumps(telemetry, indent=2))

    # ── Step 2: Build prompt ───────────────────────────────────────────
    prompt = build_mistral_prompt(telemetry)

    # ── Step 3: Call vLLM with structured output ───────────────────────
    try:
        response = await client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Telemetry Snapshot:\n{json.dumps(telemetry, indent=2)}\n\n"
                        "Task: Apply do-calculus to identify the root cause and "
                        "produce a causal DAG. Return ONLY the JSON object with: "
                        "root_cause, intervention_simulation, recommended_action, "
                        "dag_nodes, dag_edges."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1024,
            extra_body={
                "guided_json": CausalAnalysisResponse.model_json_schema(),
            },
        )

        raw_content = response.choices[0].message.content
        analysis = CausalAnalysisResponse.model_validate_json(raw_content)
        logger.info("Live LLM analysis received and validated.")

    except Exception as e:
        logger.warning("vLLM unreachable (%s) — using mock analysis.", str(e))
        analysis = get_mock_analysis()

    # ── Step 4: Safety plane ───────────────────────────────────────────
    result = IncidentAnalysisResult(
        analysis=analysis,
        requires_human_approval=True,
        safety_plane="read-only-copilot",
        telemetry_snapshot=telemetry,
        refutation_status="running",
    )

    # ── Step 5: Background refutation test ─────────────────────────────
    background_tasks.add_task(_run_refutation_background, analysis)

    return result


@app.get("/api/refutation-result")
async def get_refutation_result():
    """Poll for the latest refutation test result."""
    if "latest" in _refutation_results:
        return _refutation_results["latest"]
    return {"status": "pending", "message": "Refutation test still running."}


@app.get("/api/telemetry")
async def get_telemetry():
    """Return raw telemetry for the dashboard."""
    return fetch_system_telemetry()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "sre-nidaan-body",
        "safety_plane": "read-only-copilot",
        "vllm_endpoint": VLLM_ENDPOINT,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
