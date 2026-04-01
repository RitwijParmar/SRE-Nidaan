"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CausalGraph from "@/components/CausalGraph";

interface DAGNode {
  id: string;
  label: string;
}

interface DAGEdge {
  id: string;
  source: string;
  target: string;
  animated: boolean;
}

interface CausalAnalysis {
  root_cause: string;
  intervention_simulation: string;
  recommended_action: string;
  dag_nodes: DAGNode[];
  dag_edges: DAGEdge[];
}

interface GroundingEvidence {
  id: string;
  kind: string;
  title: string;
  summary: string;
  matched_terms: string[];
  score: number;
}

interface VerifierResult {
  accepted: boolean;
  score: number;
  evidence_overlap: number;
  telemetry_overlap: number;
  generic_penalty: number;
  panic_scaling_penalty: number;
  reasons: string[];
}

interface GenerationMetadata {
  artifact_label: string;
  source: string;
  model_id: string;
  llm_reachable: boolean;
  candidate_count: number;
  selected_candidate_index: number;
  used_fallback: boolean;
  knowledge_base_path: string;
}

interface IncidentResult {
  analysis_id: string;
  analysis: CausalAnalysis;
  grounding_evidence: GroundingEvidence[];
  verifier: VerifierResult;
  generation_metadata: GenerationMetadata;
  requires_human_approval: boolean;
  safety_plane: string;
  telemetry_snapshot: Record<string, Record<string, string | number>>;
  refutation_status: string;
  timestamp: string;
}

interface HealthPayload {
  status: string;
  service: string;
  safety_plane: string;
  model_id: string;
  artifact_label: string;
  vllm_endpoint: string;
  default_candidate_count: number;
  generation_max_tokens: number;
}

interface BrainHealthPayload {
  status: string;
  engine_loaded: boolean;
  engine_loading: boolean;
  engine_error: string | null;
  serving_backend: string;
  base_model: string;
  lora_adapter: string;
  max_lora_rank: number;
}

interface IntegrationCheck {
  face: "online" | "offline";
  body: "online" | "offline";
  telemetry: "online" | "offline";
  brain: "ready" | "warming" | "error" | "unknown";
  checked_at: string;
}

interface IntegrationSnapshot {
  status: string;
  service: string;
  checked_at: string;
  services: {
    face: string;
    body: string;
    telemetry_api: string;
    brain: string;
  };
  endpoints: {
    body_health: string;
    telemetry_api: string;
    brain_health: string;
  };
  notes: string[];
}

interface InterventionAuthorizationAck {
  status: string;
  authorization_id: string;
  analysis_id: string;
  tenant_id: string;
  timestamp: string;
}

const STATIC_TELEMETRY: Record<string, Record<string, string | number>> = {
  frontend: { status: "503 Gateway Timeout", error_rate: "Spiking" },
  auth_service: { cpu_utilization: "96%", latency_ms: 4500, replicas: 5 },
  database: {
    connections: "990/1000 (99%)",
    wait_event: "ClientRead (Locked)",
  },
};

const INCIDENT_PRESETS = [
  {
    id: "auth-retry-storm",
    title: "Auth Retry Storm",
    summary:
      "Login bursts trigger retry loops in auth-service. Frontend 503 spikes while DB connections climb to 99%.",
  },
  {
    id: "cache-eviction",
    title: "Cache Eviction Drift",
    summary:
      "Preference cache misses surged after deployment. API gateway latency doubled and datastore saturation is rising.",
  },
  {
    id: "kafka-rebalance",
    title: "Kafka Rebalance Cascade",
    summary:
      "Consumer lag jumped after rebalance storms. Worker CPU spikes and downstream writes back up across regions.",
  },
];

const NETWORK_TIMEOUT_MS = 12000;
const REFUTATION_TIMEOUT_MS = 5000;
const ANALYZE_TIMEOUT_MS = 45000;

function prettifyLabel(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function metricLooksCritical(value: string | number): boolean {
  if (typeof value === "number") {
    return value >= 90;
  }
  const compact = value.toLowerCase();
  return (
    compact.includes("503") ||
    compact.includes("spiking") ||
    compact.includes("99%") ||
    compact.includes("96%") ||
    compact.includes("locked")
  );
}

function serviceTone(service: string): {
  border: string;
  title: string;
  badge: string;
  badgeClass: string;
} {
  const map: Record<string, { border: string; title: string; badge: string; badgeClass: string }> = {
    frontend: {
      border: "border-nidaan-danger/35",
      title: "text-nidaan-danger",
      badge: "critical",
      badgeClass: "bg-nidaan-danger/10 text-nidaan-danger border-nidaan-danger/25",
    },
    auth_service: {
      border: "border-nidaan-warning/35",
      title: "text-nidaan-warning",
      badge: "warning",
      badgeClass: "bg-nidaan-warning/10 text-nidaan-warning border-nidaan-warning/25",
    },
    database: {
      border: "border-nidaan-danger/35",
      title: "text-nidaan-danger",
      badge: "critical",
      badgeClass: "bg-nidaan-danger/10 text-nidaan-danger border-nidaan-danger/25",
    },
  };
  return (
    map[service] ?? {
      border: "border-nidaan-border",
      title: "text-nidaan-accent",
      badge: "info",
      badgeClass: "bg-nidaan-accent/10 text-nidaan-accent border-nidaan-accent/25",
    }
  );
}

function shortTimestamp(value: string): string {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(parsed));
}

function normalizeOnlineState(value: string | undefined, fallback: "online" | "offline" = "offline"): "online" | "offline" {
  if (!value) {
    return fallback;
  }
  const compact = value.toLowerCase();
  return compact.includes("online") || compact.includes("browser-origin") ? "online" : "offline";
}

function normalizeBrainState(value: string | undefined): IntegrationCheck["brain"] {
  if (!value) {
    return "unknown";
  }
  const compact = value.toLowerCase();
  if (compact.includes("ready")) {
    return "ready";
  }
  if (compact.includes("warming")) {
    return "warming";
  }
  if (compact.includes("online")) {
    return "ready";
  }
  if (compact.includes("error") || compact.includes("offline") || compact.includes("fail")) {
    return "error";
  }
  return "unknown";
}

function toBrainHealthUrl(vllmEndpoint: string): string {
  return vllmEndpoint.replace(/\/v1\/?$/, "/health");
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = NETWORK_TIMEOUT_MS
): Promise<Response> {
  const timeoutController = new AbortController();
  const timeoutHandle = window.setTimeout(() => {
    timeoutController.abort("timeout");
  }, timeoutMs);

  let detachExternalAbort: (() => void) | undefined;
  if (init.signal) {
    const externalSignal = init.signal;
    if (externalSignal.aborted) {
      timeoutController.abort("upstream-abort");
    } else {
      const forwardAbort = () => {
        timeoutController.abort("upstream-abort");
      };
      externalSignal.addEventListener("abort", forwardAbort, { once: true });
      detachExternalAbort = () => {
        externalSignal.removeEventListener("abort", forwardAbort);
      };
    }
  }

  try {
    return await fetch(input, { ...init, signal: timeoutController.signal });
  } finally {
    window.clearTimeout(timeoutHandle);
    if (detachExternalAbort) {
      detachExternalAbort();
    }
  }
}

function ensureRenderableIncidentResult(payload: IncidentResult): IncidentResult {
  const nodes = payload.analysis.dag_nodes ?? [];
  const edges = payload.analysis.dag_edges ?? [];
  const nodeIds = new Set(nodes.map((node) => node.id));
  const filteredEdges = edges.filter(
    (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)
  );

  return {
    ...payload,
    analysis: {
      ...payload.analysis,
      dag_nodes: nodes,
      dag_edges: filteredEdges,
    },
  };
}

export default function DashboardPage() {
  const configuredBodyBase = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");
  const BODY_BASE = configuredBodyBase || "/body";
  const API_BASE = "/api";
  const DIRECT_API_BASE = configuredBodyBase ? `${configuredBodyBase}/api` : API_BASE;
  const TENANT_ID = (process.env.NEXT_PUBLIC_TENANT_ID || "default-tenant").trim();
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [brainHealth, setBrainHealth] = useState<BrainHealthPayload | null>(null);
  const [telemetry, setTelemetry] = useState<Record<string, Record<string, string | number>> | null>(null);
  const [incidentSummary, setIncidentSummary] = useState(INCIDENT_PRESETS[0].summary);
  const [selectedPreset, setSelectedPreset] = useState(INCIDENT_PRESETS[0].id);
  const [candidateCount, setCandidateCount] = useState(3);
  const [expertMode, setExpertMode] = useState(false);
  const [authorized, setAuthorized] = useState(false);
  const [authorizationSubmitting, setAuthorizationSubmitting] = useState(false);
  const [authorizationStatus, setAuthorizationStatus] = useState<string | null>(null);
  const [authorizationReason, setAuthorizationReason] = useState("");
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshingRuntime, setRefreshingRuntime] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackNote, setFeedbackNote] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Connecting to SRE-Nidaan runtime...");
  const [statusTone, setStatusTone] = useState<"info" | "success" | "error">("info");
  const [refutation, setRefutation] = useState<Record<string, unknown> | null>(null);
  const [integrationCheck, setIntegrationCheck] = useState<IntegrationCheck | null>(null);
  const [integrationSnapshot, setIntegrationSnapshot] = useState<IntegrationSnapshot | null>(null);
  const [checkingIntegration, setCheckingIntegration] = useState(false);
  const [integrationCheckMessage, setIntegrationCheckMessage] = useState<string | null>(null);
  const analysisAbortRef = useRef<AbortController | null>(null);
  const analysisRunRef = useRef(0);
  const latestHealthRef = useRef<HealthPayload | null>(null);
  const didBootstrapRef = useRef(false);

  const telemetryView = result?.telemetry_snapshot ?? telemetry ?? STATIC_TELEMETRY;
  const graphResult = useMemo(
    () => (result ? ensureRenderableIncidentResult(result) : null),
    [result]
  );
  const withTenantHeaders = useCallback(
    (headers?: HeadersInit): Headers => {
      const merged = new Headers(headers || {});
      if (!merged.has("x-tenant-id")) {
        merged.set("x-tenant-id", TENANT_ID);
      }
      return merged;
    },
    [TENANT_ID]
  );

  const refreshRuntime = useCallback(async (): Promise<HealthPayload> => {
    setRefreshingRuntime(true);
    setStatusTone("info");
    setStatusMessage("Refreshing runtime status...");
    try {
      const [healthRes, telemetryRes] = await Promise.all([
        fetchWithTimeout(`${BODY_BASE}/health`, {}, NETWORK_TIMEOUT_MS),
        fetchWithTimeout(
          `${API_BASE}/telemetry`,
          { headers: withTenantHeaders() },
          NETWORK_TIMEOUT_MS
        ),
      ]);

      if (!healthRes.ok) {
        throw new Error(`health check failed (${healthRes.status})`);
      }

      const healthPayload: HealthPayload = await healthRes.json();
      const telemetryPayload = telemetryRes.ok
        ? ((await telemetryRes.json()) as Record<string, Record<string, string | number>>)
        : null;

      let brainPayload: BrainHealthPayload | null = null;
      try {
        const brainHealthResponse = await fetchWithTimeout(
          toBrainHealthUrl(healthPayload.vllm_endpoint),
          {},
          NETWORK_TIMEOUT_MS
        );
        if (brainHealthResponse.ok) {
          brainPayload = (await brainHealthResponse.json()) as BrainHealthPayload;
        }
      } catch {
        // Best effort only.
      }

      setHealth(healthPayload);
      latestHealthRef.current = healthPayload;
      setBrainHealth(brainPayload);
      setTelemetry(telemetryPayload);
      setCandidateCount(Math.max(1, Math.min(8, healthPayload.default_candidate_count || 3)));
      setStatusTone("success");
      setStatusMessage("Runtime refreshed and synchronized.");
      return healthPayload;
    } catch (error) {
      setStatusTone("error");
      setStatusMessage("Runtime refresh failed or timed out. Verify backend/brain health.");
      throw error;
    } finally {
      setRefreshingRuntime(false);
    }
  }, [API_BASE, BODY_BASE, withTenantHeaders]);

  const runIntegrationCheck = useCallback(
    async (seedHealth: HealthPayload | null) => {
      setCheckingIntegration(true);
      setIntegrationCheckMessage("Running integration check...");

      try {
        const sourceHealth = seedHealth ?? latestHealthRef.current ?? (await refreshRuntime());
        const [integrationRes, telemetryRes] = await Promise.all([
          fetchWithTimeout(
            `${API_BASE}/integration-check`,
            { headers: withTenantHeaders() },
            NETWORK_TIMEOUT_MS
          ),
          fetchWithTimeout(
            `${API_BASE}/telemetry`,
            { headers: withTenantHeaders() },
            NETWORK_TIMEOUT_MS
          ),
        ]);

        if (telemetryRes.ok) {
          setTelemetry((await telemetryRes.json()) as Record<string, Record<string, string | number>>);
        }

        let brainPayload: BrainHealthPayload | null = null;
        try {
          const brainRes = await fetchWithTimeout(
            toBrainHealthUrl(sourceHealth.vllm_endpoint),
            {},
            NETWORK_TIMEOUT_MS
          );
          if (brainRes.ok) {
            brainPayload = (await brainRes.json()) as BrainHealthPayload;
            setBrainHealth(brainPayload);
          }
        } catch {
          // Best effort only.
        }

        let faceStatus: IntegrationCheck["face"] = "online";
        let bodyStatus: IntegrationCheck["body"] = "online";
        let telemetryStatus: IntegrationCheck["telemetry"] = telemetryRes.ok ? "online" : "offline";
        let brainStatus: IntegrationCheck["brain"] = normalizeBrainState(brainPayload?.status);
        let checkedAt = new Date().toISOString();

        if (integrationRes.ok) {
          const integrationPayload = (await integrationRes.json()) as IntegrationSnapshot;
          setIntegrationSnapshot(integrationPayload);
          checkedAt = integrationPayload.checked_at || checkedAt;
          faceStatus = normalizeOnlineState(integrationPayload.services?.face, "online");
          bodyStatus = normalizeOnlineState(integrationPayload.services?.body, "online");
          telemetryStatus = normalizeOnlineState(
            integrationPayload.services?.telemetry_api,
            telemetryStatus
          );
          brainStatus = normalizeBrainState(
            integrationPayload.services?.brain || brainPayload?.status
          );
          setIntegrationCheckMessage(
            `Integration ${integrationPayload.status} · checked ${shortTimestamp(checkedAt)} · brain ${brainStatus}.`
          );
        } else {
          setIntegrationSnapshot(null);
          setIntegrationCheckMessage(
            `Checked ${shortTimestamp(checkedAt)} · diagnostics endpoint unavailable (${integrationRes.status}).`
          );
        }

        setIntegrationCheck({
          face: faceStatus,
          body: bodyStatus,
          telemetry: telemetryStatus,
          brain: brainStatus,
          checked_at: checkedAt,
        });
      } catch {
        const checkedAt = new Date().toISOString();
        setIntegrationSnapshot(null);
        setIntegrationCheck({
          face: "online",
          body: "offline",
          telemetry: "offline",
          brain: "unknown",
          checked_at: checkedAt,
        });
        setIntegrationCheckMessage(
          `Checked ${shortTimestamp(checkedAt)} · integration check failed or timed out (body/telemetry unreachable).`
        );
      } finally {
        setCheckingIntegration(false);
      }
    },
    [API_BASE, refreshRuntime, withTenantHeaders]
  );

  useEffect(() => {
    if (didBootstrapRef.current) {
      return;
    }
    didBootstrapRef.current = true;

    let disposed = false;

    async function bootstrap() {
      try {
        const healthPayload = await refreshRuntime();
        if (!disposed) {
          void runIntegrationCheck(healthPayload);
        }
      } catch (err) {
        if (!disposed) {
          setStatusTone("error");
          setStatusMessage("Runtime connection failed. Verify backend URL and health endpoint.");
          console.error(err);
        }
      }
    }

    void bootstrap();

    return () => {
      disposed = true;
    };
  }, [refreshRuntime, runIntegrationCheck]);

  const pollRefutation = useCallback(async () => {
    for (let attempt = 0; attempt < 6; attempt += 1) {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 1200);
      });
      try {
        const response = await fetchWithTimeout(
          `${API_BASE}/refutation-result`,
          { headers: withTenantHeaders() },
          REFUTATION_TIMEOUT_MS
        );
        if (!response.ok) {
          continue;
        }
        const payload = (await response.json()) as Record<string, unknown>;
        setRefutation(payload);
        if (payload.status !== "pending") {
          break;
        }
      } catch {
        // Non-blocking poll.
      }
    }
  }, [API_BASE, withTenantHeaders]);

  const analyzeIncident = useCallback(async () => {
    if (loading) {
      return;
    }
    const runId = analysisRunRef.current + 1;
    analysisRunRef.current = runId;
    const abortController = new AbortController();
    analysisAbortRef.current = abortController;
    const timeoutHandle = window.setTimeout(() => {
      abortController.abort("timeout");
    }, ANALYZE_TIMEOUT_MS);

    setLoading(true);
    setAnalysisStage("Collecting telemetry and prompting the causal model...");
    setAuthorized(false);
    setAuthorizationStatus(null);
    setResult(null);
    setFeedbackStatus(null);
    setRefutation(null);
    setAnalysisError(null);
    setStatusTone("info");
    setStatusMessage("Running grounded causal analysis...");

    try {
      const response = await fetchWithTimeout(`${DIRECT_API_BASE}/analyze-incident`, {
        method: "POST",
        headers: withTenantHeaders({ "Content-Type": "application/json" }),
        signal: abortController.signal,
        body: JSON.stringify({
          incident_summary: incidentSummary,
          candidate_count: candidateCount,
        }),
      }, ANALYZE_TIMEOUT_MS + 1500);

      if (!response.ok) {
        throw new Error(`analysis failed (${response.status})`);
      }

      if (analysisRunRef.current !== runId) {
        return;
      }

      const payload = ensureRenderableIncidentResult((await response.json()) as IncidentResult);
      setAnalysisStage("Rendering causal graph...");
      setResult(payload);
      setTelemetry(payload.telemetry_snapshot);
      setStatusTone("success");
      setStatusMessage(
        payload.generation_metadata.used_fallback
          ? "Analysis complete with grounded fallback path."
          : "Analysis complete from live candidate selection."
      );
      void pollRefutation();
    } catch (err) {
      console.error(err);
      if (analysisRunRef.current !== runId) {
        return;
      }
      const aborted = abortController.signal.aborted;
      const stopReason = String(abortController.signal.reason || "");
      setAnalysisError(
        aborted
          ? "Analysis was cancelled before completion."
          : "Live analysis failed. Review runtime diagnostics and retry."
      );
      if (aborted) {
        setStatusTone("error");
        setStatusMessage(
          stopReason === "timeout"
            ? "Analysis timed out. No synthetic fallback shown."
            : "Analysis stopped. No synthetic fallback shown."
        );
      } else {
        setStatusTone("error");
        setStatusMessage("Live inference failed. No synthetic fallback shown.");
      }
    } finally {
      window.clearTimeout(timeoutHandle);
      if (analysisAbortRef.current === abortController && analysisRunRef.current === runId) {
        analysisAbortRef.current = null;
      }
      if (analysisRunRef.current === runId) {
        setLoading(false);
        setAnalysisStage(null);
      }
    }
  }, [DIRECT_API_BASE, candidateCount, incidentSummary, loading, pollRefutation, withTenantHeaders]);

  const stopAnalysis = useCallback(() => {
    if (analysisAbortRef.current) {
      setStatusTone("info");
      setStatusMessage("Stopping analysis request...");
      setAnalysisStage("Cancelling in-flight request...");
      analysisAbortRef.current.abort("manual-stop");
    }
  }, []);

  const submitFeedback = useCallback(
    async (rating: "useful" | "needs_correction") => {
      if (!result) {
        return;
      }
      setFeedbackSubmitting(true);
      setFeedbackStatus(null);
      try {
        const response = await fetchWithTimeout(`${DIRECT_API_BASE}/analysis-feedback`, {
          method: "POST",
          headers: withTenantHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            analysis_id: result.analysis_id,
            rating,
            correction: feedbackNote,
            incident_summary: incidentSummary,
            analysis: result.analysis,
            verifier: result.verifier,
            generation_metadata: result.generation_metadata,
          }),
        }, NETWORK_TIMEOUT_MS);

        if (!response.ok) {
          throw new Error(`feedback failed (${response.status})`);
        }

        setFeedbackStatus(
          rating === "useful"
            ? "Analyst feedback saved as useful."
            : "Correction note stored for preference tuning."
        );
        if (rating === "needs_correction") {
          setFeedbackNote("");
        }
      } catch (err) {
        console.error(err);
        setFeedbackStatus("Feedback submission failed. Retry in a moment.");
      } finally {
        setFeedbackSubmitting(false);
      }
    },
    [DIRECT_API_BASE, feedbackNote, incidentSummary, result, withTenantHeaders]
  );

  const authorizeIntervention = useCallback(async () => {
    if (!result || authorizationSubmitting) {
      return;
    }
    const reason = authorizationReason.trim();
    if (reason.length < 8) {
      setAuthorizationStatus("Provide a short approval reason (at least 8 characters).");
      return;
    }

    setAuthorizationSubmitting(true);
    setAuthorizationStatus(null);
    try {
      const response = await fetchWithTimeout(
        `${DIRECT_API_BASE}/interventions/authorize`,
        {
          method: "POST",
          headers: withTenantHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            analysis_id: result.analysis_id,
            operator_id: "incident-commander",
            tenant_id: TENANT_ID,
            reason,
            approved_action: result.analysis.recommended_action,
          }),
        },
        NETWORK_TIMEOUT_MS
      );
      if (!response.ok) {
        throw new Error(`authorization failed (${response.status})`);
      }
      const payload = (await response.json()) as InterventionAuthorizationAck;
      setAuthorized(true);
      setAuthorizationStatus(
        `Authorized as ${payload.authorization_id} at ${shortTimestamp(payload.timestamp)}.`
      );
      setAuthorizationReason("");
    } catch (error) {
      console.error(error);
      setAuthorizationStatus("Authorization failed. Check policy headers and retry.");
    } finally {
      setAuthorizationSubmitting(false);
    }
  }, [
    DIRECT_API_BASE,
    TENANT_ID,
    authorizationReason,
    authorizationSubmitting,
    result,
    withTenantHeaders,
  ]);

  return (
    <div className="nidaan-shell min-h-screen text-nidaan-ink">
      <div className="nidaan-glow nidaan-glow-left" />
      <div className="nidaan-glow nidaan-glow-right" />

      <header className="sticky top-0 z-50 border-b border-nidaan-border/80 bg-nidaan-paper/75 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-nidaan-accent to-nidaan-accent-strong shadow-lg shadow-nidaan-accent/35">
              <span className="nidaan-display text-lg font-bold text-white">N</span>
            </div>
            <div>
              <p className="nidaan-mono text-[11px] uppercase tracking-[0.2em] text-nidaan-muted">
                SRE-Nidaan Command Deck
              </p>
              <h1 className="nidaan-display text-2xl font-semibold text-nidaan-ink md:text-3xl">
                Causal Incident Response Copilot
              </h1>
              <p className="max-w-3xl text-sm text-nidaan-muted">
                Grounded root-cause analysis, intervention safety gating, and operator feedback loops in one live product surface.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setExpertMode((value) => !value)}
              className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent"
            >
              {expertMode ? "Expert: On" : "Expert: Off"}
            </button>
            <button
              id="analyze-incident-btn"
              onClick={analyzeIncident}
              disabled={loading}
              className="rounded-full bg-gradient-to-r from-nidaan-accent to-nidaan-accent-strong px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-nidaan-accent/25 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-55"
            >
              {loading ? "Analyzing..." : "Analyze Incident"}
            </button>
            {loading && (
              <button
                onClick={stopAnalysis}
                className="rounded-full border border-nidaan-danger/35 bg-nidaan-danger/10 px-4 py-2 text-sm font-semibold text-nidaan-danger transition hover:bg-nidaan-danger/15"
              >
                Stop
              </button>
            )}
            <button
              onClick={() => {
                void (async () => {
                  try {
                    const payload = await refreshRuntime();
                    await runIntegrationCheck(payload);
                  } catch {
                    // status banner already updated in refreshRuntime
                  }
                })();
              }}
              disabled={refreshingRuntime || checkingIntegration}
              className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {refreshingRuntime || checkingIntegration ? "Refreshing..." : "Refresh Runtime"}
            </button>
            {expertMode && (
              <a
                href={`${BODY_BASE}/docs`}
                className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent"
              >
                API Docs
              </a>
            )}
            {expertMode && (
              <button
                onClick={() => {
                  void runIntegrationCheck(health);
                }}
                disabled={checkingIntegration || refreshingRuntime}
                className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {checkingIntegration ? "Checking..." : "Integration Check"}
              </button>
            )}
          </div>
          {integrationCheckMessage && (
            <p className="nidaan-mono text-[11px] text-nidaan-muted lg:text-right">
              {integrationCheckMessage}
            </p>
          )}
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1600px] grid-cols-1 gap-6 px-5 py-6 lg:px-8 xl:grid-cols-12">
        <section className="space-y-5 xl:col-span-4">
          <article id="incident-command" className="nidaan-card p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Incident Command</h2>
              <span className="nidaan-chip">Human-in-the-loop enforced</span>
            </div>
            <p className="text-sm text-nidaan-muted">
              Pick a realistic scenario, adjust candidate search depth, and trigger the causal response engine.
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              {INCIDENT_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => {
                    setSelectedPreset(preset.id);
                    setIncidentSummary(preset.summary);
                  }}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${
                    selectedPreset === preset.id
                      ? "border-nidaan-accent/30 bg-nidaan-accent/12 text-nidaan-accent-strong"
                      : "border-nidaan-border bg-white text-nidaan-muted hover:border-nidaan-accent/30 hover:text-nidaan-accent-strong"
                  }`}
                >
                  {preset.title}
                </button>
              ))}
            </div>

            <textarea
              value={incidentSummary}
              onChange={(event) => setIncidentSummary(event.target.value)}
              rows={5}
              className="mt-4 w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm leading-relaxed text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
              placeholder="Describe blast radius, customer impact, and first observed symptoms."
            />

            {expertMode && (
              <div className="mt-4 rounded-2xl border border-nidaan-border bg-white/80 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-nidaan-muted">
                    Candidate Search Depth
                  </span>
                  <span className="nidaan-mono text-xs text-nidaan-ink">{candidateCount}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={8}
                  step={1}
                  value={candidateCount}
                  onChange={(event) => {
                    setCandidateCount(Number(event.target.value));
                  }}
                  className="w-full accent-[#0f766e]"
                />
                <p className="mt-2 text-xs text-nidaan-muted">
                  More candidates increase diversity before verifier ranking. Recommended for demos: 3-5.
                </p>
              </div>
            )}
          </article>

          <article className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Incident Workflow</h2>
              <span className="nidaan-chip">guided</span>
            </div>
            <div className="space-y-2 text-sm text-nidaan-muted">
              <p>
                <span className="nidaan-mono text-nidaan-ink">1.</span> Describe incident blast radius and trigger analysis.
              </p>
              <p>
                <span className="nidaan-mono text-nidaan-ink">2.</span> Validate root cause and graph evidence.
              </p>
              <p>
                <span className="nidaan-mono text-nidaan-ink">3.</span> Authorize intervention with reason for audit.
              </p>
              <p>
                <span className="nidaan-mono text-nidaan-ink">4.</span> Submit analyst feedback to improve future responses.
              </p>
            </div>
          </article>

          <article id="runtime-pulse" className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Runtime Pulse</h2>
              <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] ${
                statusTone === "success"
                  ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                  : statusTone === "error"
                    ? "border-nidaan-danger/30 bg-nidaan-danger/10 text-nidaan-danger"
                    : "border-nidaan-accent/30 bg-nidaan-accent/10 text-nidaan-accent-strong"
              }`}>
                <span className={`h-2 w-2 rounded-full ${statusTone === "error" ? "bg-nidaan-danger" : "bg-nidaan-success"}`} />
                {statusTone}
              </span>
            </div>
            <p className="text-sm text-nidaan-muted">{statusMessage}</p>

            {health && !expertMode && (
              <div className="mt-4 rounded-2xl border border-nidaan-border bg-white p-3">
                <p className="text-xs text-nidaan-ink">
                  Runtime is connected with artifact <span className="nidaan-mono">{health.artifact_label}</span>.
                </p>
                <p className="mt-1 text-xs text-nidaan-muted">
                  Detailed runtime metadata is available in expert mode.
                </p>
              </div>
            )}

            {health && expertMode && (
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-nidaan-border bg-white p-3">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">model</p>
                  <p className="mt-1 text-xs font-semibold text-nidaan-ink">{health.model_id}</p>
                </div>
                <div className="rounded-xl border border-nidaan-border bg-white p-3">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">artifact</p>
                  <p className="mt-1 text-xs font-semibold text-nidaan-ink">{health.artifact_label}</p>
                </div>
                <div className="rounded-xl border border-nidaan-border bg-white p-3">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">safety plane</p>
                  <p className="mt-1 text-xs font-semibold text-nidaan-ink">{health.safety_plane}</p>
                </div>
                <div className="rounded-xl border border-nidaan-border bg-white p-3">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">max tokens</p>
                  <p className="mt-1 text-xs font-semibold text-nidaan-ink">{health.generation_max_tokens}</p>
                </div>
              </div>
            )}

            {brainHealth && expertMode && (
              <div className="mt-3 rounded-2xl border border-nidaan-border bg-white p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">brain runtime</p>
                  <span className={`rounded-full border px-2 py-0.5 nidaan-mono text-[10px] uppercase ${
                    brainHealth.status === "ready"
                      ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                      : brainHealth.status === "warming"
                        ? "border-nidaan-warning/30 bg-nidaan-warning/10 text-nidaan-warning"
                        : "border-nidaan-danger/30 bg-nidaan-danger/10 text-nidaan-danger"
                  }`}>
                    {brainHealth.status}
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <p className="text-xs text-nidaan-ink"><span className="nidaan-mono text-nidaan-muted">backend:</span> {brainHealth.serving_backend}</p>
                  <p className="text-xs text-nidaan-ink"><span className="nidaan-mono text-nidaan-muted">engine loaded:</span> {String(brainHealth.engine_loaded)}</p>
                  <p className="break-all text-xs text-nidaan-ink"><span className="nidaan-mono text-nidaan-muted">base model:</span> {brainHealth.base_model}</p>
                  <p className="break-all text-xs text-nidaan-ink"><span className="nidaan-mono text-nidaan-muted">adapter:</span> {brainHealth.lora_adapter}</p>
                </div>
                {brainHealth.engine_error && (
                  <p className="mt-2 text-xs text-nidaan-danger">{brainHealth.engine_error}</p>
                )}
              </div>
            )}

            {brainHealth && !expertMode && (
              <p className="mt-3 text-xs text-nidaan-muted">
                Brain status: <span className="nidaan-mono text-nidaan-ink">{brainHealth.status}</span>
              </p>
            )}
          </article>

          {expertMode && (
          <article id="system-integration" className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">System Integration</h2>
              <span className="nidaan-chip">Face + Body + Brain</span>
            </div>
            <p className="text-sm text-nidaan-muted">
              Live wiring status across microservices, including model-serving readiness from the Brain endpoint.
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-nidaan-border bg-white p-3">
                <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">face</p>
                <p className={`mt-1 text-xs font-semibold ${
                  integrationCheck?.face === "offline" ? "text-nidaan-danger" : "text-nidaan-success"
                }`}>
                  {integrationCheck?.face ?? "online"}
                </p>
              </div>
              <div className="rounded-xl border border-nidaan-border bg-white p-3">
                <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">body</p>
                <p className={`mt-1 text-xs font-semibold ${
                  integrationCheck?.body === "online" ? "text-nidaan-success" : "text-nidaan-danger"
                }`}>
                  {integrationCheck?.body ?? (health ? "online" : "checking")}
                </p>
              </div>
              <div className="rounded-xl border border-nidaan-border bg-white p-3">
                <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">telemetry api</p>
                <p className={`mt-1 text-xs font-semibold ${
                  integrationCheck?.telemetry === "online" ? "text-nidaan-success" : "text-nidaan-danger"
                }`}>
                  {integrationCheck?.telemetry ?? "checking"}
                </p>
              </div>
              <div className="rounded-xl border border-nidaan-border bg-white p-3">
                <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">brain</p>
                <p className={`mt-1 text-xs font-semibold ${
                  (integrationCheck?.brain ?? "unknown") === "ready"
                    ? "text-nidaan-success"
                    : (integrationCheck?.brain ?? "unknown") === "warming"
                      ? "text-nidaan-warning"
                      : "text-nidaan-danger"
                }`}>
                  {integrationCheck?.brain ?? (brainHealth?.status ?? "checking")}
                </p>
              </div>
            </div>

            <div className="mt-3 rounded-xl border border-nidaan-border bg-white p-3">
              <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">service endpoints</p>
              <p className="mt-1 break-all text-xs text-nidaan-ink">face: browser origin</p>
              <p className="mt-1 break-all text-xs text-nidaan-ink">body: {BODY_BASE}</p>
              <p className="mt-1 break-all text-xs text-nidaan-ink">
                brain: {health ? toBrainHealthUrl(health.vllm_endpoint) : "waiting for body health"}
              </p>
              {integrationCheck?.checked_at && (
                <p className="mt-2 nidaan-mono text-[10px] text-nidaan-muted">
                  last check: {shortTimestamp(integrationCheck.checked_at)}
                </p>
              )}
            </div>

            {integrationSnapshot && (
              <div className="mt-3 rounded-xl border border-nidaan-border bg-white p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">diagnostics snapshot</p>
                  <span className={`rounded-full border px-2 py-0.5 nidaan-mono text-[10px] uppercase ${
                    integrationSnapshot.status === "ok"
                      ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                      : "border-nidaan-warning/30 bg-nidaan-warning/10 text-nidaan-warning"
                  }`}>
                    {integrationSnapshot.status}
                  </span>
                </div>
                <p className="text-xs text-nidaan-ink">
                  <span className="nidaan-mono text-nidaan-muted">checked:</span> {shortTimestamp(integrationSnapshot.checked_at)}
                </p>
                <p className="mt-1 break-all text-xs text-nidaan-ink">
                  <span className="nidaan-mono text-nidaan-muted">body health:</span> {integrationSnapshot.endpoints.body_health}
                </p>
                <p className="mt-1 break-all text-xs text-nidaan-ink">
                  <span className="nidaan-mono text-nidaan-muted">telemetry api:</span> {integrationSnapshot.endpoints.telemetry_api}
                </p>
                <p className="mt-1 break-all text-xs text-nidaan-ink">
                  <span className="nidaan-mono text-nidaan-muted">brain health:</span> {integrationSnapshot.endpoints.brain_health}
                </p>
                {integrationSnapshot.notes.length > 0 && (
                  <p className="mt-2 text-xs text-nidaan-muted">{integrationSnapshot.notes.join(" · ")}</p>
                )}
              </div>
            )}
          </article>
          )}

          {expertMode && (
          <article className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Telemetry Snapshot</h2>
              <span className="nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">current source snapshot</span>
            </div>
            <div className="space-y-3">
              {Object.entries(telemetryView).map(([service, metrics]) => {
                const tone = serviceTone(service);
                return (
                  <div key={service} className={`rounded-2xl border bg-white/90 p-4 ${tone.border}`}>
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <h3 className={`text-sm font-semibold ${tone.title}`}>{prettifyLabel(service)}</h3>
                      <span className={`rounded-full border px-2 py-0.5 nidaan-mono text-[10px] uppercase ${tone.badgeClass}`}>
                        {tone.badge}
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {Object.entries(metrics).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between gap-4">
                          <span className="nidaan-mono text-[11px] text-nidaan-muted">{key.replace(/_/g, " ")}</span>
                          <span className={`nidaan-mono text-xs font-semibold ${metricLooksCritical(value) ? "text-nidaan-danger" : "text-nidaan-ink"}`}>
                            {String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </article>
          )}
        </section>

        <section className="space-y-5 xl:col-span-8">
          <article id="causal-graph-workspace" className="nidaan-card overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-nidaan-border/80 bg-white/80 px-5 py-3">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Causal Graph Workspace</h2>
              {loading ? (
                <span className="rounded-full border border-nidaan-warning/30 bg-nidaan-warning/10 px-2 py-1 nidaan-mono text-[11px] text-nidaan-warning">
                  running analysis...
                </span>
              ) : graphResult ? (
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className={`rounded-full border px-2 py-1 nidaan-mono ${
                    graphResult.verifier.accepted
                      ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                      : "border-nidaan-warning/30 bg-nidaan-warning/10 text-nidaan-warning"
                  }`}>
                    verifier {graphResult.verifier.accepted ? "accepted" : "fallback"}
                  </span>
                  <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-nidaan-muted">
                    {graphResult.analysis.dag_nodes.length} nodes · {graphResult.analysis.dag_edges.length} edges
                  </span>
                  <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-nidaan-muted">
                    {shortTimestamp(graphResult.timestamp)}
                  </span>
                </div>
              ) : (
                <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-[11px] text-nidaan-muted">
                  waiting for first analysis
                </span>
              )}
            </div>
            <div className="h-[460px]">
              {loading && !graphResult ? (
                <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                  <div className="h-10 w-10 rounded-full border-4 border-nidaan-accent/25 border-t-nidaan-accent animate-spin" />
                  <p className="text-sm font-semibold text-nidaan-ink">
                    {analysisStage || "Running causal analysis..."}
                  </p>
                  <p className="max-w-md text-xs text-nidaan-muted">
                    First response may take up to 45 seconds. Use <span className="nidaan-mono">Stop</span> to cancel the in-flight request.
                  </p>
                </div>
              ) : graphResult ? (
                <div className="relative h-full">
                  <CausalGraph nodes={graphResult.analysis.dag_nodes} edges={graphResult.analysis.dag_edges} />
                  {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-nidaan-paper/70 backdrop-blur-[1px]">
                      <div className="rounded-2xl border border-nidaan-accent/25 bg-white/90 px-4 py-2 text-center">
                        <p className="text-xs font-semibold text-nidaan-accent-strong">
                          {analysisStage || "Running next analysis..."}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ) : analysisError ? (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                  <p className="text-sm font-semibold text-nidaan-danger">{analysisError}</p>
                  <p className="max-w-md text-xs text-nidaan-muted">
                    Synthetic fallback rendering is disabled in product mode. Retry once runtime is healthy.
                  </p>
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                  <div className="nidaan-display text-5xl text-nidaan-accent/30">DAG</div>
                  <p className="max-w-md text-sm text-nidaan-muted">
                    Trigger analysis to render the directed acyclic graph, then inspect root cause and intervention path.
                  </p>
                </div>
              )}
            </div>
          </article>

          {result && (
            <article className="grid grid-cols-1 gap-5 lg:grid-cols-3">
              <div className="nidaan-card p-5">
                <p className="mb-2 nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">root cause</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.root_cause}</p>
              </div>
              <div className="nidaan-card p-5">
                <p className="mb-2 nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">intervention simulation</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.intervention_simulation}</p>
              </div>
              <div className="nidaan-card p-5">
                <p className="mb-2 nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">recommended action</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.recommended_action}</p>
              </div>
            </article>
          )}

          {result && (
            <article className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <div className="nidaan-card p-5">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">Grounding Evidence</h3>
                  <span className="nidaan-chip">{result.grounding_evidence.length} docs</span>
                </div>
                <div className="space-y-3">
                  {result.grounding_evidence.map((evidence) => (
                    <div key={evidence.id} className="rounded-2xl border border-nidaan-border bg-white p-3">
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <span className="text-sm font-semibold text-nidaan-ink">{evidence.title}</span>
                        <span className="nidaan-mono rounded-full border border-nidaan-border px-2 py-0.5 text-[10px] uppercase text-nidaan-muted">
                          {evidence.kind}
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed text-nidaan-muted">{evidence.summary}</p>
                      {evidence.matched_terms.length > 0 && (
                        <p className="mt-2 nidaan-mono text-[10px] text-nidaan-accent-strong">
                          matched: {evidence.matched_terms.join(", ")}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-5">
                <div className="nidaan-card p-5">
                  <h3 className="nidaan-display mb-3 text-lg font-semibold text-nidaan-ink">Generation Verifier</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-nidaan-border bg-white p-3">
                      <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">score</p>
                      <p className="mt-1 text-xl font-bold text-nidaan-ink">{result.verifier.score.toFixed(3)}</p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-white p-3">
                      <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">source</p>
                      <p className="mt-1 text-sm font-semibold text-nidaan-ink">{result.generation_metadata.source}</p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-white p-3">
                      <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">evidence overlap</p>
                      <p className="mt-1 text-sm font-semibold text-nidaan-ink">{result.verifier.evidence_overlap}</p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-white p-3">
                      <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">telemetry overlap</p>
                      <p className="mt-1 text-sm font-semibold text-nidaan-ink">{result.verifier.telemetry_overlap}</p>
                    </div>
                  </div>
                  <div className="mt-3 rounded-xl border border-nidaan-border bg-white p-3">
                    <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">candidate selection</p>
                    <p className="mt-1 text-xs text-nidaan-ink">
                      {result.generation_metadata.selected_candidate_index >= 0
                        ? `${result.generation_metadata.selected_candidate_index + 1} / ${result.generation_metadata.candidate_count}`
                        : `fallback / ${result.generation_metadata.candidate_count}`}
                    </p>
                    <p className="mt-2 text-xs text-nidaan-muted">
                      {result.verifier.reasons.join(" · ")}
                    </p>
                  </div>
                </div>

                {expertMode && refutation && (
                  <div className="nidaan-card p-5">
                    <h3 className="nidaan-display mb-3 text-lg font-semibold text-nidaan-ink">Refutation Monitor</h3>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="rounded-xl border border-nidaan-border bg-white p-3">
                        <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">test type</p>
                        <p className="mt-1 text-xs font-semibold text-nidaan-ink">{String(refutation.test_type || "-")}</p>
                      </div>
                      <div className="rounded-xl border border-nidaan-border bg-white p-3">
                        <p className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">verdict</p>
                        <p className={`mt-1 text-xs font-semibold ${refutation.is_robust ? "text-nidaan-success" : "text-nidaan-warning"}`}>
                          {refutation.is_robust ? "PASS" : "WARN"}
                        </p>
                      </div>
                    </div>
                    <p className="mt-3 text-xs text-nidaan-muted">
                      Confounder injected: {String(refutation.injected_confounder || "-")}
                    </p>
                  </div>
                )}
              </div>
            </article>
          )}

          {result && (
            <article className={`nidaan-card border-2 p-5 transition ${
              authorized ? "border-nidaan-success/45" : "border-nidaan-danger/40"
            }`}>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">Intervention Engine</h3>
                <span className={`rounded-full border px-2 py-1 nidaan-mono text-[10px] uppercase ${
                  authorized
                    ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                    : "border-nidaan-danger/30 bg-nidaan-danger/10 text-nidaan-danger"
                }`}>
                  {authorized ? "authorized" : "awaiting human approval"}
                </span>
              </div>
              <div className="rounded-2xl border border-nidaan-border bg-white p-4">
                <p className="nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">recommended action</p>
                <p className="mt-2 text-sm leading-relaxed text-nidaan-ink">{result.analysis.recommended_action}</p>
              </div>
              <div className="mt-4">
                {!authorized ? (
                  <div className="space-y-3">
                    <textarea
                      value={authorizationReason}
                      onChange={(event) => setAuthorizationReason(event.target.value)}
                      rows={2}
                      className="w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
                      placeholder="Approval reason (required for audit log)"
                    />
                    <button
                      id="authorize-intervention-btn"
                      onClick={() => {
                        void authorizeIntervention();
                      }}
                      disabled={authorizationSubmitting}
                      className="w-full rounded-xl bg-gradient-to-r from-nidaan-danger to-[#e1664b] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-nidaan-danger/30 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {authorizationSubmitting ? "Authorizing..." : "Human-in-the-loop: Authorize Intervention"}
                    </button>
                    {authorizationStatus && (
                      <p className="text-xs text-nidaan-ink">{authorizationStatus}</p>
                    )}
                  </div>
                ) : (
                  <div className="w-full rounded-xl border border-nidaan-success/30 bg-nidaan-success/15 px-5 py-3 text-center text-sm font-semibold text-nidaan-success">
                    Intervention Authorized. Execution can proceed through operator controls.
                  </div>
                )}
              </div>
              {result.requires_human_approval && !authorized && (
                <p className="mt-3 text-center text-xs text-nidaan-muted">
                  Safety Plane <span className="nidaan-mono text-nidaan-ink">{result.safety_plane}</span> blocks action until explicit human authorization.
                </p>
              )}
            </article>
          )}

          {result && (
            <article id="analyst-feedback-loop" className="nidaan-card p-5">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">Analyst Feedback Loop</h3>
                <span className="nidaan-mono text-[10px] uppercase tracking-wider text-nidaan-muted">{result.analysis_id}</span>
              </div>
              <p className="mb-3 text-sm text-nidaan-muted">
                Capture real analyst judgment now. These labels are the fastest path to improving production outputs.
              </p>
              <textarea
                value={feedbackNote}
                onChange={(event) => setFeedbackNote(event.target.value)}
                rows={3}
                className="w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
                placeholder="Optional correction or missing operational context..."
              />
              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  onClick={() => void submitFeedback("useful")}
                  disabled={feedbackSubmitting}
                  className="rounded-full border border-nidaan-success/35 bg-nidaan-success/10 px-4 py-2 text-sm font-semibold text-nidaan-success transition hover:bg-nidaan-success/15 disabled:opacity-45"
                >
                  Mark Useful
                </button>
                <button
                  onClick={() => void submitFeedback("needs_correction")}
                  disabled={feedbackSubmitting}
                  className="rounded-full border border-nidaan-warning/35 bg-nidaan-warning/10 px-4 py-2 text-sm font-semibold text-nidaan-warning transition hover:bg-nidaan-warning/15 disabled:opacity-45"
                >
                  Needs Correction
                </button>
              </div>
              {feedbackStatus && <p className="mt-3 text-sm text-nidaan-ink">{feedbackStatus}</p>}
            </article>
          )}

        </section>
      </main>

      <footer className="border-t border-nidaan-border/80 bg-white/70 px-5 py-4 lg:px-8">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-1 text-xs text-nidaan-muted md:flex-row md:items-center md:justify-between">
          <span>SRE-Nidaan · Stable SFT baseline + grounding + verifier</span>
          <span className="nidaan-mono">Safety Plane: read-only-copilot · requires_human_approval=true</span>
        </div>
      </footer>
    </div>
  );
}
