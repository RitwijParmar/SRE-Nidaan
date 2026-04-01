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
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, min(maximum, float(raw_value)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw_value = os.environ.get(name, default)
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if values:
        return values
    return [item.strip() for item in default.split(",") if item.strip()]


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
FEEDBACK_DB_PATH = os.environ.get(
    "FEEDBACK_DB_PATH", str(ROOT_DIR / "feedback" / "analyst_feedback.db")
)
TELEMETRY_SOURCE_URL = os.environ.get("TELEMETRY_SOURCE_URL", "").strip()
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "").strip()
REQUIRE_TENANT_ID = _env_bool("REQUIRE_TENANT_ID", True)
ALLOWED_ORIGINS = _env_csv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://sre-nidaan-face-ciiiagnzaq-uk.a.run.app",
)
DEFAULT_CANDIDATE_COUNT = _env_int("GENERATION_CANDIDATES", 3, 1, 8)
DEFAULT_GROUNDING_LIMIT = _env_int("GROUNDING_EVIDENCE_LIMIT", 4, 1, 8)
GENERATION_MAX_TOKENS = _env_int("GENERATION_MAX_TOKENS", 512, 96, 1024)
LIVE_ANALYSIS_TIMEOUT_SECONDS = _env_float("LIVE_ANALYSIS_TIMEOUT_SECONDS", 18.0, 2.0, 120.0)

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
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_api_security(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/"):
        if request.method == "OPTIONS":
            return await call_next(request)
        tenant_id = request.headers.get("x-tenant-id", "").strip()
        if REQUIRE_TENANT_ID and not tenant_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "missing required header: x-tenant-id"},
            )
        if API_AUTH_TOKEN:
            provided_token = request.headers.get("x-api-key", "").strip()
            if provided_token != API_AUTH_TOKEN:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid api key"},
                )

    response = await call_next(request)
    return response


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


class InterventionAuthorizationRequest(BaseModel):
    analysis_id: str
    operator_id: str = Field(..., min_length=2, max_length=120)
    tenant_id: str = Field(..., min_length=1, max_length=120)
    reason: str = Field(..., min_length=8, max_length=2000)
    approved_action: str = Field(default="", max_length=2000)


class InterventionAuthorizationAck(BaseModel):
    status: str
    authorization_id: str
    analysis_id: str
    tenant_id: str
    timestamp: str


class MCPToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPCallResponse(BaseModel):
    tool_name: str
    result: dict[str, Any]


# ---------------------------------------------------------------------------
# Mock MCP Tools - Data Layer
# ---------------------------------------------------------------------------

STATIC_TELEMETRY: dict[str, Any] = {
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


class MCPRouter:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, dict[str, Any], Any]] = {}

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Any,
    ) -> None:
        self._tools[name] = (description, input_schema, handler)

    def list_tools(self) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name=name,
                description=description,
                input_schema=input_schema,
            )
            for name, (description, input_schema, _handler) in sorted(self._tools.items())
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        record = self._tools.get(tool_name)
        if not record:
            raise ValueError(f"unknown MCP tool: {tool_name}")
        _description, _input_schema, handler = record
        payload = handler(arguments or {})
        if not isinstance(payload, dict):
            raise ValueError(f"MCP tool {tool_name} returned non-dict payload")
        return payload


def _coerce_telemetry_payload(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("telemetry payload must be an object")

    candidate = raw_payload.get("telemetry") if "telemetry" in raw_payload else raw_payload
    if not isinstance(candidate, dict):
        raise ValueError("telemetry payload missing object map")

    normalized: dict[str, Any] = {}
    for service_name, metrics in candidate.items():
        if not isinstance(service_name, str) or not isinstance(metrics, dict):
            continue
        normalized[service_name] = metrics
    if not normalized:
        raise ValueError("telemetry payload has no valid services")
    return normalized


def _tool_get_telemetry(arguments: dict[str, Any]) -> dict[str, Any]:
    source_url = str(arguments.get("source_url") or TELEMETRY_SOURCE_URL).strip()
    if source_url:
        try:
            with urllib.request.urlopen(source_url, timeout=8.0) as response:
                body = response.read().decode("utf-8", errors="ignore")
                payload = json.loads(body)
                telemetry = _coerce_telemetry_payload(payload)
                return {
                    "telemetry": telemetry,
                    "source": f"live-http:{source_url}",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            logger.warning("Telemetry source fetch failed (%s), using static snapshot.", exc)

    return {
        "telemetry": STATIC_TELEMETRY,
        "source": "simulated-static",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


MCP_ROUTER = MCPRouter()
MCP_ROUTER.register_tool(
    name="sre.telemetry.get_snapshot",
    description="Return telemetry snapshot for incident analysis with source attribution.",
    input_schema={
        "type": "object",
        "properties": {
            "source_url": {"type": "string"},
        },
        "additionalProperties": False,
    },
    handler=_tool_get_telemetry,
)


def fetch_system_telemetry() -> dict[str, Any]:
    payload = MCP_ROUTER.call_tool("sre.telemetry.get_snapshot", {})
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict):
        return STATIC_TELEMETRY
    return telemetry


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
_analysis_cache: dict[str, dict[str, Any]] = {}
_analysis_tenants: dict[str, str] = {}


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


def _analysis_structure_is_viable(payload: dict[str, Any]) -> bool:
    nodes = payload.get("dag_nodes")
    edges = payload.get("dag_edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    if len(nodes) < 3 or len(edges) < 2:
        return False

    node_ids = {
        str(item.get("id"))
        for item in nodes
        if isinstance(item, dict) and item.get("id")
    }
    if len(node_ids) < 3:
        return False

    valid_edges = 0
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source and target and source in node_ids and target in node_ids:
            valid_edges += 1
    return valid_edges >= 2


def _feedback_log_file() -> Path:
    path = Path(FEEDBACK_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _feedback_db_file() -> Path:
    path = Path(FEEDBACK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _feedback_db_connection() -> sqlite3.Connection:
    db_path = _feedback_db_file()
    connection = sqlite3.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyst_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            correction TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            incident_summary TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            verifier_json TEXT NOT NULL,
            generation_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS intervention_authorizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            authorization_id TEXT NOT NULL UNIQUE,
            analysis_id TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            approved_action TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return connection


def _to_brain_health_url(vllm_endpoint: str) -> str:
    endpoint = (vllm_endpoint or "").rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]
    return f"{endpoint}/health"


async def _probe_json_url(url: str, timeout_seconds: float = 6.0) -> tuple[bool, dict[str, Any] | None]:
    def _fetch() -> tuple[bool, dict[str, Any] | None]:
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="ignore")
                if response.status < 200 or response.status >= 300:
                    return False, None
                try:
                    return True, json.loads(body)
                except Exception:
                    return True, None
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False, None

    return await asyncio.to_thread(_fetch)


async def _brain_ready_for_inference() -> bool:
    brain_health_url = _to_brain_health_url(VLLM_ENDPOINT)
    brain_ok, brain_payload = await _probe_json_url(brain_health_url, timeout_seconds=2.0)
    if not brain_ok:
        return False

    status_value = str((brain_payload or {}).get("status", "")).strip().lower()
    return status_value in {"ready", "online", "healthy", "ok"}


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
    request: Request,
) -> IncidentAnalysisResult:
    request_tenant_id = request.headers.get("x-tenant-id", "default").strip() or "default"
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
    llm_ready = await _brain_ready_for_inference()

    if not llm_ready:
        llm_reachable = False
        used_fallback = True
        source = "grounded-fallback"
        logger.info("Brain not ready for inference; returning grounded fallback immediately.")
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
    else:
        try:
            try:
                live_candidates = await asyncio.wait_for(
                    _generate_candidate_analyses(
                        user_content,
                        candidate_count=candidate_count,
                    ),
                    timeout=LIVE_ANALYSIS_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"live inference exceeded {LIVE_ANALYSIS_TIMEOUT_SECONDS:.1f}s budget"
                ) from exc
            candidate_payloads = [candidate.model_dump() for candidate in live_candidates]
            selected_payload, verifier_data, selected_candidate_index = select_best_candidate(
                candidate_payloads,
                telemetry=telemetry,
                grounding_evidence=grounding_evidence_payload,
            )
            structure_viable = _analysis_structure_is_viable(selected_payload)
            if (
                selected_candidate_index < 0
                or not verifier_data["accepted"]
                or not structure_viable
            ):
                used_fallback = True
                source = "grounded-fallback"
                if not structure_viable:
                    logger.info("Selected live candidate failed structural DAG checks; using grounded fallback.")
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
    _analysis_cache[analysis_id] = result.model_dump()
    _analysis_tenants[analysis_id] = request_tenant_id
    background_tasks.add_task(_run_refutation_background, analysis)
    return result


@app.post("/api/analysis-feedback", response_model=FeedbackAck)
async def analysis_feedback(payload: FeedbackSubmission, request: Request) -> FeedbackAck:
    tenant_id = request.headers.get("x-tenant-id", "default")
    record = {
        "analysis_id": payload.analysis_id,
        "rating": payload.rating,
        "correction": payload.correction.strip(),
        "operator": payload.operator.strip() or "anonymous",
        "tenant_id": tenant_id,
        "incident_summary": payload.incident_summary.strip(),
        "analysis": payload.analysis or {},
        "verifier": payload.verifier or {},
        "generation_metadata": payload.generation_metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with _feedback_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO analyst_feedback (
                analysis_id,
                rating,
                correction,
                operator_id,
                tenant_id,
                incident_summary,
                analysis_json,
                verifier_json,
                generation_metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["analysis_id"],
                record["rating"],
                record["correction"],
                record["operator"],
                record["tenant_id"],
                record["incident_summary"],
                json.dumps(record["analysis"]),
                json.dumps(record["verifier"]),
                json.dumps(record["generation_metadata"]),
                record["timestamp"],
            ),
        )

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


@app.get("/api/mcp/tools", response_model=list[MCPToolDescriptor])
async def list_mcp_tools() -> list[MCPToolDescriptor]:
    return MCP_ROUTER.list_tools()


@app.post("/api/mcp/call", response_model=MCPCallResponse)
async def call_mcp_tool(payload: MCPCallRequest) -> MCPCallResponse:
    try:
        result = MCP_ROUTER.call_tool(payload.tool_name, payload.arguments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPCallResponse(tool_name=payload.tool_name, result=result)


@app.get("/api/integration-check")
async def integration_check() -> dict[str, Any]:
    brain_health_url = _to_brain_health_url(VLLM_ENDPOINT)
    brain_ok, brain_payload = await _probe_json_url(brain_health_url)
    telemetry_probe = MCP_ROUTER.call_tool("sre.telemetry.get_snapshot", {})
    telemetry_source = str(telemetry_probe.get("source", "unknown"))
    brain_status = "offline"
    if brain_ok:
        brain_status = str((brain_payload or {}).get("status", "online"))

    return {
        "status": "ok" if brain_ok else "degraded",
        "service": "sre-nidaan-body",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "services": {
            "face": "browser-origin",
            "body": "online",
            "telemetry_api": "online",
            "brain": brain_status,
        },
        "endpoints": {
            "body_health": "/health",
            "telemetry_api": "/api/telemetry",
            "brain_health": brain_health_url,
        },
        "notes": [
            "Use this endpoint for one-click integration diagnostics.",
            "Brain status is probed from the configured VLLM endpoint.",
            f"Telemetry source: {telemetry_source}",
        ],
    }


@app.post("/api/interventions/authorize", response_model=InterventionAuthorizationAck)
async def authorize_intervention(
    payload: InterventionAuthorizationRequest,
    request: Request,
) -> InterventionAuthorizationAck:
    request_tenant_id = request.headers.get("x-tenant-id", "").strip()
    if request_tenant_id and request_tenant_id != payload.tenant_id.strip():
        raise HTTPException(
            status_code=400,
            detail="tenant_id mismatch between header and payload",
        )

    analysis_payload = _analysis_cache.get(payload.analysis_id)
    if not analysis_payload:
        raise HTTPException(
            status_code=404,
            detail=f"analysis_id not found: {payload.analysis_id}",
        )
    expected_tenant_id = _analysis_tenants.get(payload.analysis_id)
    if expected_tenant_id and expected_tenant_id != payload.tenant_id.strip():
        raise HTTPException(
            status_code=403,
            detail="analysis_id belongs to a different tenant",
        )

    if not bool(analysis_payload.get("requires_human_approval", True)):
        raise HTTPException(
            status_code=409,
            detail="analysis does not require human approval",
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    authorization_id = f"auth-{uuid4().hex[:12]}"
    approved_action = payload.approved_action.strip() or str(
        analysis_payload.get("analysis", {}).get("recommended_action", "")
    )
    record = {
        "authorization_id": authorization_id,
        "analysis_id": payload.analysis_id,
        "operator_id": payload.operator_id.strip(),
        "tenant_id": payload.tenant_id.strip(),
        "reason": payload.reason.strip(),
        "approved_action": approved_action,
        "analysis_payload": analysis_payload,
        "timestamp": timestamp,
    }

    with _feedback_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO intervention_authorizations (
                authorization_id,
                analysis_id,
                operator_id,
                tenant_id,
                reason,
                approved_action,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["authorization_id"],
                record["analysis_id"],
                record["operator_id"],
                record["tenant_id"],
                record["reason"],
                record["approved_action"],
                json.dumps(record["analysis_payload"]),
                record["timestamp"],
            ),
        )

    logger.info(
        "Intervention authorized: %s by %s tenant=%s",
        authorization_id,
        record["operator_id"],
        record["tenant_id"],
    )
    return InterventionAuthorizationAck(
        status="authorized",
        authorization_id=authorization_id,
        analysis_id=payload.analysis_id,
        tenant_id=record["tenant_id"],
        timestamp=timestamp,
    )


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
        "live_analysis_timeout_seconds": LIVE_ANALYSIS_TIMEOUT_SECONDS,
        "vllm_max_retries": VLLM_MAX_RETRIES,
        "mcp_tools_registered": len(MCP_ROUTER.list_tools()),
        "telemetry_source_url": TELEMETRY_SOURCE_URL or "simulated-static",
        "require_tenant_id": REQUIRE_TENANT_ID,
        "api_auth_enabled": bool(API_AUTH_TOKEN),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
