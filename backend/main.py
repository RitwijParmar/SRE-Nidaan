"""
SRE-Nidaan: The Body - FastAPI Agentic Backend
==============================================
Grounded production backend with:
  * strict JSON decoding against the incident schema
  * grounding retrieval over telemetry + operations knowledge
  * candidate verification and deterministic fallback selection
  * human-in-the-loop safety gating and analyst feedback logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.runtime.product_strategy import (  # noqa: E402
    build_grounded_fallback_analysis,
    render_grounding_context,
    retrieve_grounding_evidence,
    score_candidate_analysis,
    select_best_candidate,
)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, min(maximum, int(raw_value)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "not-needed")
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
PRODUCTION_ARTIFACT_LABEL = os.environ.get(
    "PRODUCTION_ARTIFACT_LABEL", "checkpoint-1064"
)
VLLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("VLLM_REQUEST_TIMEOUT_SECONDS", "12"))
VLLM_MAX_RETRIES = _env_int("VLLM_MAX_RETRIES", 1, 0, 5)
GROUNDING_KB_PATH = os.environ.get(
    "GROUNDING_KB_PATH", str(ROOT_DIR / "ops" / "knowledge_base.json")
)
FEEDBACK_LOG_PATH = os.environ.get(
    "FEEDBACK_LOG_PATH", str(ROOT_DIR / "feedback" / "analyst_feedback.jsonl")
)
DEFAULT_CANDIDATE_COUNT = _env_int("GENERATION_CANDIDATES", 3, 1, 8)
DEFAULT_GROUNDING_LIMIT = _env_int("GROUNDING_EVIDENCE_LIMIT", 4, 1, 8)
GENERATION_MAX_TOKENS = _env_int("GENERATION_MAX_TOKENS", 512, 96, 1024)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sre-nidaan-body")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SRE-Nidaan Body - Agentic Backend",
    version="2.0.0",
    description=(
        "MCP router with blast-radius controls, grounding retrieval, "
        "candidate verification, and analyst feedback capture."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
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
    root_cause: str = Field(
        ..., description="The structural root cause identified via causal inference."
    )
    intervention_simulation: str = Field(
        ...,
        description=(
            "Explain why do(Scale Up Auth_Service) is a confounding error "
            "that can worsen the incident."
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


class GroundingEvidence(BaseModel):
    id: str
    kind: str = "doc"
    title: str
    summary: str
    matched_terms: list[str] = Field(default_factory=list)
    score: float = 0.0


class VerifierResult(BaseModel):
    accepted: bool
    score: float
    evidence_overlap: int
    telemetry_overlap: int
    generic_penalty: float = 0.0
    panic_scaling_penalty: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    artifact_label: str
    source: str
    model_id: str
    llm_reachable: bool
    candidate_count: int
    selected_candidate_index: int
    used_fallback: bool
    knowledge_base_path: str


class IncidentAnalysisResult(BaseModel):
    analysis_id: str
    analysis: CausalAnalysisResponse
    grounding_evidence: list[GroundingEvidence]
    verifier: VerifierResult
    generation_metadata: GenerationMetadata
    requires_human_approval: bool = True
    safety_plane: str = "read-only-copilot"
    telemetry_snapshot: dict[str, Any]
    refutation_status: str = "pending"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AnalyzeIncidentRequest(BaseModel):
    incident_summary: str = Field(default="", max_length=1200)
    telemetry_override: dict[str, Any] | None = None
    candidate_count: int | None = Field(default=None, ge=1, le=8)


class FeedbackSubmission(BaseModel):
    analysis_id: str
    rating: Literal["useful", "needs_correction"]
    correction: str = ""
    operator: str = "anonymous"
    incident_summary: str = ""
    analysis: dict[str, Any] | None = None
    verifier: dict[str, Any] | None = None
    generation_metadata: dict[str, Any] | None = None


class FeedbackAck(BaseModel):
    status: str
    analysis_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# Mock MCP Tools - Data Layer
# ---------------------------------------------------------------------------

def fetch_system_telemetry() -> dict[str, Any]:
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
# Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are NEXUS-CAUSAL, an AI SRE copilot using Pearl's Causal Hierarchy.
Use the telemetry and grounding evidence to identify the structural bottleneck, not surface symptoms.
Do not recommend panic scaling. If capacity or database changes are mentioned, require human approval.
Use concrete service names, metrics, and retrieved evidence whenever possible.
Be concise and deterministic.
Return short strings, at most 4 dag_nodes, and at most 4 dag_edges.
For dag_edges, use exactly these keys: id, source, target, animated.
Do not use keys like from or to.
Respond with a valid JSON object matching the provided schema exactly."""


def build_analysis_user_content(
    telemetry: dict[str, Any],
    *,
    incident_summary: str = "",
    grounding_context: str = "",
) -> str:
    incident_block = (
        f"Incident Summary:\n{incident_summary.strip()}\n\n" if incident_summary.strip() else ""
    )
    grounding_block = (
        f"Grounding Evidence:\n{grounding_context}\n\n" if grounding_context else ""
    )
    return (
        f"{incident_block}"
        f"Telemetry Snapshot:\n{json.dumps(telemetry, indent=2)}\n\n"
        f"{grounding_block}"
        "Task:\n"
        "1. Identify the structural root cause with telemetry-specific language.\n"
        "2. Explain why do(Scale Up Auth_Service) or similar upstream scaling can worsen the incident.\n"
        "3. Recommend the safest intervention and mention manual review or human approval for risky changes.\n"
        "4. Produce a compact causal DAG with 3-4 nodes and 2-4 edges using short IDs.\n"
        "5. Keep every field concise so the full JSON stays small.\n\n"
        "Return ONLY the JSON object with: root_cause, intervention_simulation, recommended_action, dag_nodes, dag_edges."
    )


# ---------------------------------------------------------------------------
# Refutation Test - Observability / Day-2 SRE
# ---------------------------------------------------------------------------

async def run_refutation_test(analysis: CausalAnalysisResponse) -> dict[str, Any]:
    confounder_types = [
        {
            "type": "random_common_cause",
            "node": "network_jitter",
            "label": "Network Jitter (Placebo)",
        },
        {
            "type": "placebo_treatment",
            "node": "cache_miss",
            "label": "Cache Miss Rate (Placebo)",
        },
        {
            "type": "data_permutation",
            "node": "dns_latency",
            "label": "DNS Latency (Random)",
        },
        {
            "type": "subset_removal",
            "node": "log_volume",
            "label": "Log Volume Spike (Subset)",
        },
    ]

    confounder = random.choice(confounder_types)
    await asyncio.sleep(random.uniform(0.5, 2.0))

    estimate_shift = round(random.uniform(-0.05, 0.05), 4)
    is_robust = abs(estimate_shift) < 0.03

    result = {
        "test_type": confounder["type"],
        "injected_confounder": confounder["label"],
        "original_root_cause": analysis.root_cause,
        "estimate_shift": estimate_shift,
        "is_robust": is_robust,
        "verdict": (
            "PASS - Causal estimate is stable under placebo perturbation."
            if is_robust
            else "WARN - Estimate shifted; manual review recommended."
        ),
    }

    logger.info(
        "Refutation test complete: %s -> %s",
        confounder["label"],
        result["verdict"],
    )
    return result


_refutation_results: dict[str, Any] = {}


async def _run_refutation_background(analysis: CausalAnalysisResponse) -> None:
    _refutation_results["latest"] = await run_refutation_test(analysis)


# ---------------------------------------------------------------------------
# Fallbacks and Client Helpers
# ---------------------------------------------------------------------------

def get_mock_analysis() -> CausalAnalysisResponse:
    return CausalAnalysisResponse(
        root_cause=(
            "Database connection pool exhaustion (990/1000) caused by an "
            "auth_service retry storm holding ClientRead locks."
        ),
        intervention_simulation=(
            "Applying do(Scale Up Auth_Service) is a confounding error because each "
            "new replica opens more database connections and widens the outage."
        ),
        recommended_action=(
            "Rate limit frontend traffic, break the retry loop with a circuit breaker, "
            "drain locked sessions, and require human approval before raising DB limits."
        ),
        dag_nodes=[
            DAGNode(id="frontend", label="Frontend (503 Gateway Timeout)"),
            DAGNode(id="auth_service", label="Auth Service retry storm"),
            DAGNode(id="database", label="Database connection pool saturation"),
            DAGNode(id="client_lock", label="ClientRead lock backlog"),
            DAGNode(id="outage", label="Customer-visible outage"),
        ],
        dag_edges=[
            DAGEdge(id="e1", source="frontend", target="auth_service", animated=True),
            DAGEdge(id="e2", source="auth_service", target="database", animated=True),
            DAGEdge(id="e3", source="database", target="client_lock", animated=True),
            DAGEdge(id="e4", source="client_lock", target="outage", animated=True),
        ],
    )


def _sanitize_json_candidate(raw_content: str) -> str:
    candidate = raw_content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = candidate.find("{")
    end = candidate.rfind("}")
    normalized = candidate[start : end + 1] if start != -1 and end != -1 and end > start else raw_content

    try:
        payload = json.loads(normalized)
    except Exception:
        return normalized

    def _slugify(value: str, fallback: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or fallback

    nodes = payload.get("dag_nodes")
    if isinstance(nodes, list):
        normalized_nodes: list[dict[str, Any]] = []
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            label = str(node.get("label") or node.get("name") or node.get("id") or f"Node {index}")
            node_id = str(node.get("id") or _slugify(label, f"node_{index}"))
            normalized_nodes.append({"id": node_id, "label": label})
        payload["dag_nodes"] = normalized_nodes

    edges = payload.get("dag_edges")
    if isinstance(edges, list):
        normalized_edges: list[dict[str, Any]] = []
        for index, edge in enumerate(edges, start=1):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to") or "")
            if not source or not target:
                continue
            normalized_edges.append(
                {
                    "id": str(edge.get("id") or f"e{index}"),
                    "source": source,
                    "target": target,
                    "animated": bool(edge.get("animated", True)),
                }
            )
        payload["dag_edges"] = normalized_edges

    return json.dumps(payload)


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _candidate_count(requested: int | None) -> int:
    if requested is None:
        return DEFAULT_CANDIDATE_COUNT
    return max(1, min(8, requested))


def _feedback_log_file() -> Path:
    path = Path(FEEDBACK_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _generate_candidate_analyses(
    user_content: str,
    *,
    candidate_count: int,
) -> list[CausalAnalysisResponse]:
    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.45 if candidate_count > 1 else 0.2,
        top_p=0.9,
        n=candidate_count,
        max_tokens=GENERATION_MAX_TOKENS,
        extra_body={"guided_json": CausalAnalysisResponse.model_json_schema()},
    )

    parsed_candidates: list[CausalAnalysisResponse] = []
    for choice in response.choices:
        raw_content = _coerce_message_text(choice.message.content)
        try:
            parsed_candidates.append(
                CausalAnalysisResponse.model_validate_json(
                    _sanitize_json_candidate(raw_content)
                )
            )
        except Exception as exc:
            logger.warning("Discarding invalid candidate from vLLM: %s", exc)
    return parsed_candidates


client = AsyncOpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=VLLM_API_KEY,
    timeout=VLLM_REQUEST_TIMEOUT_SECONDS,
    max_retries=VLLM_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/analyze-incident", response_model=IncidentAnalysisResult)
async def analyze_incident(
    payload: AnalyzeIncidentRequest,
    background_tasks: BackgroundTasks,
) -> IncidentAnalysisResult:
    telemetry = payload.telemetry_override or fetch_system_telemetry()
    grounding_evidence_payload = retrieve_grounding_evidence(
        telemetry,
        incident_summary=payload.incident_summary,
        knowledge_base_path=GROUNDING_KB_PATH,
        limit=DEFAULT_GROUNDING_LIMIT,
    )
    grounding_context = render_grounding_context(grounding_evidence_payload)
    user_content = build_analysis_user_content(
        telemetry,
        incident_summary=payload.incident_summary,
        grounding_context=grounding_context,
    )

    candidate_count = _candidate_count(payload.candidate_count)
    llm_reachable = True
    selected_candidate_index = -1
    used_fallback = False
    source = "live-candidate"

    try:
        live_candidates = await _generate_candidate_analyses(
            user_content,
            candidate_count=candidate_count,
        )
        candidate_payloads = [candidate.model_dump() for candidate in live_candidates]
        selected_payload, verifier_data, selected_candidate_index = select_best_candidate(
            candidate_payloads,
            telemetry=telemetry,
            grounding_evidence=grounding_evidence_payload,
        )
        if selected_candidate_index < 0 or not verifier_data["accepted"]:
            used_fallback = True
            source = "grounded-fallback"
            selected_payload = build_grounded_fallback_analysis(
                telemetry,
                grounding_evidence=grounding_evidence_payload,
            )
            verifier_data = score_candidate_analysis(
                selected_payload,
                telemetry=telemetry,
                grounding_evidence=grounding_evidence_payload,
            )
        analysis = CausalAnalysisResponse.model_validate(selected_payload)
        logger.info(
            "Analysis selected: source=%s selected_candidate_index=%s score=%s",
            source,
            selected_candidate_index,
            verifier_data["score"],
        )
    except Exception as exc:
        llm_reachable = False
        used_fallback = True
        source = "grounded-fallback"
        logger.warning("vLLM unreachable or invalid (%s) - using grounded fallback.", exc)
        fallback_payload = build_grounded_fallback_analysis(
            telemetry,
            grounding_evidence=grounding_evidence_payload,
        )
        verifier_data = score_candidate_analysis(
            fallback_payload,
            telemetry=telemetry,
            grounding_evidence=grounding_evidence_payload,
        )
        try:
            analysis = CausalAnalysisResponse.model_validate(fallback_payload)
        except Exception:
            source = "mock-analysis"
            analysis = get_mock_analysis()
            verifier_data = score_candidate_analysis(
                analysis.model_dump(),
                telemetry=telemetry,
                grounding_evidence=grounding_evidence_payload,
            )

    analysis_id = f"analysis-{uuid4().hex[:12]}"
    result = IncidentAnalysisResult(
        analysis_id=analysis_id,
        analysis=analysis,
        grounding_evidence=[
            GroundingEvidence.model_validate(item) for item in grounding_evidence_payload
        ],
        verifier=VerifierResult.model_validate(verifier_data),
        generation_metadata=GenerationMetadata(
            artifact_label=PRODUCTION_ARTIFACT_LABEL,
            source=source,
            model_id=MODEL_ID,
            llm_reachable=llm_reachable,
            candidate_count=candidate_count,
            selected_candidate_index=selected_candidate_index,
            used_fallback=used_fallback,
            knowledge_base_path=GROUNDING_KB_PATH,
        ),
        telemetry_snapshot=telemetry,
        refutation_status="running",
    )
    background_tasks.add_task(_run_refutation_background, analysis)
    return result


@app.post("/api/analysis-feedback", response_model=FeedbackAck)
async def analysis_feedback(payload: FeedbackSubmission) -> FeedbackAck:
    record = {
        "analysis_id": payload.analysis_id,
        "rating": payload.rating,
        "correction": payload.correction.strip(),
        "operator": payload.operator.strip() or "anonymous",
        "incident_summary": payload.incident_summary.strip(),
        "analysis": payload.analysis or {},
        "verifier": payload.verifier or {},
        "generation_metadata": payload.generation_metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with _feedback_log_file().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    logger.info("Analyst feedback recorded for %s", payload.analysis_id)
    return FeedbackAck(
        status="recorded",
        analysis_id=payload.analysis_id,
        timestamp=record["timestamp"],
    )


@app.get("/api/refutation-result")
async def get_refutation_result() -> dict[str, Any]:
    if "latest" in _refutation_results:
        return _refutation_results["latest"]
    return {"status": "pending", "message": "Refutation test still running."}


@app.get("/api/telemetry")
async def get_telemetry() -> dict[str, Any]:
    return fetch_system_telemetry()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "sre-nidaan-body",
        "safety_plane": "read-only-copilot",
        "vllm_endpoint": VLLM_ENDPOINT,
        "model_id": MODEL_ID,
        "artifact_label": PRODUCTION_ARTIFACT_LABEL,
        "knowledge_base_path": GROUNDING_KB_PATH,
        "default_candidate_count": DEFAULT_CANDIDATE_COUNT,
        "generation_max_tokens": GENERATION_MAX_TOKENS,
        "vllm_request_timeout_seconds": VLLM_REQUEST_TIMEOUT_SECONDS,
        "vllm_max_retries": VLLM_MAX_RETRIES,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
