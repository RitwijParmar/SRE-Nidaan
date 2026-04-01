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
  source?: string;
  used_fallback?: boolean;
}

interface IncidentResult {
  analysis_id: string;
  analysis: CausalAnalysis;
  grounding_evidence: GroundingEvidence[];
  verifier: VerifierResult;
  generation_metadata: GenerationMetadata;
  requires_human_approval: boolean;
  telemetry_snapshot: Record<string, Record<string, string | number>>;
  timestamp: string;
}

interface HealthPayload {
  status: string;
  service: string;
}

interface IntegrationCheck {
  face: "online" | "offline";
  body: "online" | "offline";
  telemetry: "online" | "online_simulated" | "offline";
  brain: "ready" | "warming" | "error" | "unknown";
  checked_at: string;
}

interface IntegrationSnapshot {
  status: string;
  checked_at: string;
  services: {
    face: string;
    body: string;
    telemetry_api: string;
    brain: string;
  };
}

interface InterventionAuthorizationAck {
  status: string;
  authorization_id: string;
  analysis_id: string;
  tenant_id: string;
  timestamp: string;
}

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
const ANALYZE_TIMEOUT_MS = 40000;

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

function normalizeTelemetryState(
  value: string | undefined,
  fallback: IntegrationCheck["telemetry"] = "offline"
): IntegrationCheck["telemetry"] {
  if (!value) {
    return fallback;
  }
  const compact = value.toLowerCase();
  if (compact.includes("simulated")) {
    return "online_simulated";
  }
  if (compact.includes("online")) {
    return "online";
  }
  return "offline";
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
  const BODY_BASE = "/body";
  const API_BASE = "/api";
  const DIRECT_API_BASE = API_BASE;
  const ANALYSIS_CANDIDATE_COUNT = 1;
  const TENANT_ID = (process.env.NEXT_PUBLIC_TENANT_ID || "default-tenant").trim();
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [telemetry, setTelemetry] = useState<Record<string, Record<string, string | number>> | null>(null);
  const [incidentSummary, setIncidentSummary] = useState(INCIDENT_PRESETS[0].summary);
  const [selectedPreset, setSelectedPreset] = useState(INCIDENT_PRESETS[0].id);
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
  const [integrationCheck, setIntegrationCheck] = useState<IntegrationCheck | null>(null);
  const [checkingIntegration, setCheckingIntegration] = useState(false);
  const [integrationCheckMessage, setIntegrationCheckMessage] = useState<string | null>(null);
  const analysisAbortRef = useRef<AbortController | null>(null);
  const analysisRunRef = useRef(0);
  const latestHealthRef = useRef<HealthPayload | null>(null);
  const didBootstrapRef = useRef(false);

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
    setStatusMessage("Checking system readiness...");
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

      setHealth(healthPayload);
      latestHealthRef.current = healthPayload;
      setTelemetry(telemetryPayload);
      setStatusTone("success");
      setStatusMessage("System check completed.");
      return healthPayload;
    } catch (error) {
      setStatusTone("error");
      setStatusMessage("System check failed. Verify backend connectivity.");
      throw error;
    } finally {
      setRefreshingRuntime(false);
    }
  }, [API_BASE, BODY_BASE, withTenantHeaders]);

  const runIntegrationCheck = useCallback(
    async (seedHealth: HealthPayload | null) => {
      setCheckingIntegration(true);
      setIntegrationCheckMessage("Running connectivity checks...");

      try {
        if (!seedHealth && !latestHealthRef.current) {
          await refreshRuntime();
        }
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

        let faceStatus: IntegrationCheck["face"] = "online";
        let bodyStatus: IntegrationCheck["body"] = "online";
        let telemetryStatus: IntegrationCheck["telemetry"] = telemetryRes.ok ? "online" : "offline";
        let brainStatus: IntegrationCheck["brain"] = "unknown";
        let checkedAt = new Date().toISOString();

        if (integrationRes.ok) {
          const integrationPayload = (await integrationRes.json()) as IntegrationSnapshot;
          checkedAt = integrationPayload.checked_at || checkedAt;
          faceStatus = normalizeOnlineState(integrationPayload.services?.face, "online");
          bodyStatus = normalizeOnlineState(integrationPayload.services?.body, "online");
          telemetryStatus = normalizeTelemetryState(
            integrationPayload.services?.telemetry_api,
            telemetryStatus
          );
          brainStatus = normalizeBrainState(integrationPayload.services?.brain);
          setIntegrationCheckMessage(
            `System ${integrationPayload.status} · checked ${shortTimestamp(checkedAt)}.`
          );
        } else {
          setIntegrationCheckMessage(
            `Checked ${shortTimestamp(checkedAt)} · diagnostics unavailable (${integrationRes.status}).`
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
        setIntegrationCheck({
          face: "online",
          body: "offline",
          telemetry: "offline",
          brain: "unknown",
          checked_at: checkedAt,
        });
        setIntegrationCheckMessage(
          `Checked ${shortTimestamp(checkedAt)} · connectivity checks failed.`
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
          setStatusMessage("Could not connect to runtime. Check deployment health.");
          console.error(err);
        }
      }
    }

    void bootstrap();

    return () => {
      disposed = true;
    };
  }, [refreshRuntime, runIntegrationCheck]);

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
    setAnalysisStage("Collecting telemetry and generating response...");
    setAuthorized(false);
    setAuthorizationStatus(null);
    setResult(null);
    setFeedbackStatus(null);
    setAnalysisError(null);
    setStatusTone("info");
    setStatusMessage("Running incident analysis...");

    try {
      const response = await fetchWithTimeout(`${DIRECT_API_BASE}/analyze-incident`, {
        method: "POST",
        headers: withTenantHeaders({ "Content-Type": "application/json" }),
        signal: abortController.signal,
        body: JSON.stringify({
          incident_summary: incidentSummary,
          candidate_count: ANALYSIS_CANDIDATE_COUNT,
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
      setStatusMessage("Analysis complete.");
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
          : "Analysis failed. Please retry."
      );
      if (aborted) {
        setStatusTone("error");
        setStatusMessage(
          stopReason === "timeout"
            ? "Analysis timed out before completion."
            : "Analysis stopped by operator."
        );
      } else {
        setStatusTone("error");
        setStatusMessage("Analysis failed. Please retry.");
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
  }, [
    ANALYSIS_CANDIDATE_COUNT,
    DIRECT_API_BASE,
    incidentSummary,
    loading,
    withTenantHeaders,
  ]);

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

  const telemetryState = integrationCheck?.telemetry ?? "offline";
  const telemetryLabel =
    telemetryState === "online"
      ? "live"
      : telemetryState === "online_simulated"
        ? "simulated"
        : "offline";
  const runtimeToneClass =
    statusTone === "success"
      ? "border-nidaan-success/35"
      : statusTone === "error"
        ? "border-nidaan-danger/35"
        : "border-nidaan-accent/25";
  const telemetryServiceCount = Object.keys(telemetry ?? {}).length;

  return (
    <div className="nidaan-shell min-h-screen text-nidaan-ink">
      <div className="nidaan-glow nidaan-glow-left" />
      <div className="nidaan-glow nidaan-glow-right" />

      <header className="sticky top-0 z-50 border-b border-nidaan-border/80 bg-nidaan-paper">
        <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-4 px-4 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-6">
          <div className="flex items-start gap-3">
            <img
              src="/sre-nidaan-mark.svg"
              alt="SRE निदान logo"
              className="h-12 w-12 rounded-2xl border border-white/80 shadow-md md:h-14 md:w-14"
            />
            <div>
              <p className="nidaan-mono text-[10px] uppercase tracking-[0.16em] text-nidaan-muted">SRE NIDAAN</p>
              <h1 className="nidaan-display text-2xl font-semibold text-nidaan-ink md:text-[30px]">
                SRE निदान Command Deck
              </h1>
              <p className="text-sm text-nidaan-muted">
                Clear incident diagnosis, safer approvals, and a guided operator workflow.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              id="analyze-incident-btn"
              onClick={analyzeIncident}
              disabled={loading}
              className="rounded-xl bg-gradient-to-r from-nidaan-accent to-nidaan-accent-strong px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-nidaan-accent/25 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-55"
            >
              {loading ? "Analyzing..." : "Analyze Incident"}
            </button>
            {loading && (
              <button
                onClick={stopAnalysis}
                className="rounded-xl border border-nidaan-danger/35 bg-nidaan-danger/10 px-3 py-2 text-sm font-semibold text-nidaan-danger transition hover:bg-nidaan-danger/15"
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
              className="rounded-xl border border-nidaan-border bg-white px-3 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {refreshingRuntime || checkingIntegration ? "Checking..." : "Check System"}
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto w-full max-w-[1400px] px-4 pt-4 lg:px-6">
        <div className={`nidaan-card flex flex-col gap-3 border ${runtimeToneClass} p-4 lg:flex-row lg:items-center lg:justify-between`}>
          <p className="text-sm text-nidaan-ink">{statusMessage}</p>
          <div className="flex flex-wrap items-center gap-2">
            <span className="nidaan-status-pill">
              <span className={`nidaan-status-dot ${integrationCheck?.face === "offline" ? "bg-nidaan-danger" : "bg-nidaan-success"}`} />
              frontend
            </span>
            <span className="nidaan-status-pill">
              <span className={`nidaan-status-dot ${integrationCheck?.body === "online" || health ? "bg-nidaan-success" : "bg-nidaan-danger"}`} />
              backend
            </span>
            <span className="nidaan-status-pill">
              <span className={`nidaan-status-dot ${
                telemetryState === "online" ? "bg-nidaan-success" : telemetryState === "online_simulated" ? "bg-nidaan-warning" : "bg-nidaan-danger"
              }`} />
              data: {telemetryLabel}
            </span>
            <span className="nidaan-status-pill">
              <span className={`nidaan-status-dot ${
                (integrationCheck?.brain ?? "unknown") === "ready"
                  ? "bg-nidaan-success"
                  : (integrationCheck?.brain ?? "unknown") === "warming"
                    ? "bg-nidaan-warning"
                    : "bg-nidaan-danger"
              }`} />
              ai: {integrationCheck?.brain ?? "unknown"}
            </span>
          </div>
        </div>
        {integrationCheckMessage && (
          <p className="mt-2 px-1 text-xs text-nidaan-muted">{integrationCheckMessage}</p>
        )}
      </div>

      <main className="mx-auto grid w-full max-w-[1400px] grid-cols-1 gap-5 px-4 py-5 lg:px-6 xl:grid-cols-12">
        <section className="space-y-5 xl:col-span-4">
          <article id="incident-command" className="nidaan-card p-5">
            <div className="mb-2 flex items-center justify-between gap-2">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">1. Describe Incident</h2>
              <span className="nidaan-chip">required</span>
            </div>
            <p className="text-sm text-nidaan-muted">
              Mention symptoms, customer impact, and what changed before the incident.
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
                      ? "border-nidaan-accent/35 bg-nidaan-accent/10 text-nidaan-accent-strong"
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
              rows={6}
              className="mt-4 w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm leading-relaxed text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
              placeholder="Example: Login requests are retrying. Error rate jumped to 18%. DB connections are near max."
            />
          </article>

          <article className="nidaan-card p-5">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">2. System Status</h2>
              <span className="nidaan-chip">live status</span>
            </div>
            <div className="space-y-2 text-sm text-nidaan-muted">
              <p>Platform: <span className="font-semibold text-nidaan-ink">{integrationCheck?.body === "online" ? "Connected" : "Checking..."}</span></p>
              <p>Data Feed: <span className={`font-semibold ${telemetryState === "online" ? "text-nidaan-success" : telemetryState === "online_simulated" ? "text-nidaan-warning" : "text-nidaan-danger"}`}>{telemetryLabel}</span></p>
              <p>AI Engine: <span className="font-semibold text-nidaan-ink">{integrationCheck?.brain ?? "checking..."}</span></p>
              <p>Observed Services: <span className="font-semibold text-nidaan-ink">{telemetryServiceCount || "not yet"}</span></p>
              <p>Last Check: <span className="font-semibold text-nidaan-ink">{integrationCheck?.checked_at ? shortTimestamp(integrationCheck.checked_at) : "not yet"}</span></p>
            </div>
            {telemetryState === "online_simulated" && (
              <p className="mt-3 rounded-xl border border-nidaan-warning/30 bg-nidaan-warning/10 px-3 py-2 text-xs text-nidaan-warning">
                Data is currently simulated. Connect live telemetry before production escalation actions.
              </p>
            )}
          </article>

          <article className="nidaan-card p-5">
            <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">3. Guided Flow</h2>
            <div className="mt-3 space-y-2 text-sm text-nidaan-muted">
              <p><span className="nidaan-mono text-nidaan-ink">A.</span> Click <strong>Analyze Incident</strong>.</p>
              <p><span className="nidaan-mono text-nidaan-ink">B.</span> Review cause, graph, and evidence summary.</p>
              <p><span className="nidaan-mono text-nidaan-ink">C.</span> Authorize intervention only with clear safety reason.</p>
              <p><span className="nidaan-mono text-nidaan-ink">D.</span> Submit operator feedback to improve future analyses.</p>
            </div>
          </article>
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
                    {graphResult.verifier.accepted ? "confidence: good" : "confidence: review"}
                  </span>
                  <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-nidaan-muted">
                    {graphResult.analysis.dag_nodes.length} nodes · {graphResult.analysis.dag_edges.length} edges
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
                  <p className="text-sm font-semibold text-nidaan-ink">{analysisStage || "Running grounded analysis..."}</p>
                  <p className="max-w-md text-xs text-nidaan-muted">
                    First response may take up to 40 seconds during warm-up.
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
                <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
                  <p className="text-sm font-semibold text-nidaan-danger">{analysisError}</p>
                  <p className="max-w-md text-xs text-nidaan-muted">
                    Check system status and retry once platform health is green.
                  </p>
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                  <div className="nidaan-display text-5xl text-nidaan-accent/30">DAG</div>
                  <p className="max-w-md text-sm text-nidaan-muted">
                    Analysis output will appear here with root cause and intervention path.
                  </p>
                </div>
              )}
            </div>
          </article>

          {!result && (
            <article className="nidaan-card p-5">
              <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">What You’ll Get</h3>
              <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
                <div className="rounded-xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-muted">
                  <p className="font-semibold text-nidaan-ink">Root Cause</p>
                  <p className="mt-1">Structural explanation, not just symptoms.</p>
                </div>
                <div className="rounded-xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-muted">
                  <p className="font-semibold text-nidaan-ink">Intervention</p>
                  <p className="mt-1">Safe remediation path with human approval gate.</p>
                </div>
                <div className="rounded-xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-muted">
                  <p className="font-semibold text-nidaan-ink">Evidence</p>
                  <p className="mt-1">Grounding notes and verifier signals.</p>
                </div>
              </div>
            </article>
          )}

          {result && (
            <article className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <div className="nidaan-card p-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-nidaan-muted">Root Cause</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.root_cause}</p>
              </div>
              <div className="nidaan-card p-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-nidaan-muted">Intervention Logic</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.intervention_simulation}</p>
              </div>
              <div className="nidaan-card p-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-nidaan-muted">Recommended Next Action</p>
                <p className="text-sm leading-relaxed text-nidaan-ink">{result.analysis.recommended_action}</p>
              </div>
            </article>
          )}

          {result && (
            <article className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <div className={`nidaan-card border-2 p-5 transition ${
                authorized ? "border-nidaan-success/45" : "border-nidaan-danger/40"
              }`}>
                <div className="mb-4 flex items-center justify-between gap-2">
                  <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">4. Safety Approval</h3>
                  <span className={`rounded-full border px-2 py-1 nidaan-mono text-[10px] uppercase ${
                    authorized
                      ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                      : "border-nidaan-danger/30 bg-nidaan-danger/10 text-nidaan-danger"
                  }`}>
                    {authorized ? "authorized" : "pending"}
                  </span>
                </div>
                {!authorized ? (
                  <div className="space-y-3">
                    <textarea
                      value={authorizationReason}
                      onChange={(event) => setAuthorizationReason(event.target.value)}
                      rows={2}
                      className="w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
                      placeholder="Why is this safe to approve?"
                    />
                    <button
                      id="authorize-intervention-btn"
                      onClick={() => void authorizeIntervention()}
                      disabled={authorizationSubmitting}
                      className="w-full rounded-xl bg-gradient-to-r from-nidaan-danger to-[#e1664b] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-nidaan-danger/30 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {authorizationSubmitting ? "Authorizing..." : "Authorize Intervention"}
                    </button>
                    {authorizationStatus && <p className="text-xs text-nidaan-ink">{authorizationStatus}</p>}
                    <p className="text-xs text-nidaan-muted">
                      Human approval is mandatory before executing incident interventions.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-xl border border-nidaan-success/30 bg-nidaan-success/10 p-3 text-sm font-semibold text-nidaan-success">
                    Intervention authorized. You can proceed with operator runbooks.
                  </div>
                )}
              </div>

              <div className="nidaan-card p-5">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">Evidence Summary</h3>
                  <span className="nidaan-chip">{result.grounding_evidence.length} sources</span>
                </div>
                <p className="mb-4 text-xs text-nidaan-muted">
                  Confidence score: <span className="font-semibold text-nidaan-ink">{result.verifier.score.toFixed(3)}</span>
                </p>
                <div className="space-y-2">
                  {result.grounding_evidence.slice(0, 3).map((evidence) => (
                    <div key={evidence.id} className="rounded-xl border border-nidaan-border bg-white p-3">
                      <p className="text-sm font-semibold text-nidaan-ink">{evidence.title}</p>
                      <p className="mt-1 text-xs text-nidaan-muted">{evidence.summary}</p>
                    </div>
                  ))}
                </div>
              </div>
            </article>
          )}

          {result && (
            <article id="analyst-feedback-loop" className="nidaan-card p-5">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="nidaan-display text-lg font-semibold text-nidaan-ink">5. Analyst Feedback</h3>
              </div>
              <p className="mb-3 text-sm text-nidaan-muted">
                Mark whether this response was useful, and add corrections if needed.
              </p>
              <textarea
                value={feedbackNote}
                onChange={(event) => setFeedbackNote(event.target.value)}
                rows={3}
                className="w-full rounded-2xl border border-nidaan-border bg-white p-3 text-sm text-nidaan-ink outline-none transition focus:border-nidaan-accent/45 focus:ring-2 focus:ring-nidaan-accent/15"
                placeholder="Optional correction or missing context..."
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

      <footer className="border-t border-nidaan-border/80 bg-white/80 px-4 py-4 lg:px-6">
        <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-1 text-xs text-nidaan-muted md:flex-row md:items-center md:justify-between">
          <span>SRE निदान · Causal incident assistant with operator safety gates</span>
          <span className="nidaan-mono">Designed for incident command workflows</span>
        </div>
      </footer>
    </div>
  );
}
