"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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

function toBrainHealthUrl(vllmEndpoint: string): string {
  return vllmEndpoint.replace(/\/v1\/?$/, "/health");
}

function buildLocalFallbackResult(
  incidentSummary: string,
  telemetrySnapshot: Record<string, Record<string, string | number>>
): IncidentResult {
  const nowIso = new Date().toISOString();
  return {
    analysis_id: `local-${Date.now()}`,
    analysis: {
      root_cause:
        "Database connection pool saturation amplified by upstream auth retry pressure under frontend 503 load.",
      intervention_simulation:
        "Scaling auth-service directly can increase connection churn and worsen lock contention at the database layer.",
      recommended_action:
        "Throttle retries, apply circuit breaking on auth upstream calls, drain lock-heavy sessions, and require human approval before capacity changes.",
      dag_nodes: [
        { id: "frontend_503", label: "Frontend 503 Spike" },
        { id: "auth_retry", label: "Auth Retry Storm" },
        { id: "db_pool", label: "DB Pool Saturation" },
        { id: "user_impact", label: "Customer Login Failure" },
      ],
      dag_edges: [
        { id: "e1", source: "frontend_503", target: "auth_retry", animated: true },
        { id: "e2", source: "auth_retry", target: "db_pool", animated: true },
        { id: "e3", source: "db_pool", target: "user_impact", animated: true },
      ],
    },
    grounding_evidence: [
      {
        id: "local-doc-1",
        kind: "doc",
        title: "Fallback Runbook Context",
        summary:
          "Using deterministic fallback due temporary inference outage. Validate with live backend once available.",
        matched_terms: ["retry", "db", "503"],
        score: 0.7,
      },
    ],
    verifier: {
      accepted: true,
      score: 0.66,
      evidence_overlap: 2,
      telemetry_overlap: 3,
      generic_penalty: 0.0,
      panic_scaling_penalty: 0.0,
      reasons: ["local fallback generated to keep operator workflow available"],
    },
    generation_metadata: {
      artifact_label: "ui-fallback",
      source: "local-ui-fallback",
      model_id: "unavailable",
      llm_reachable: false,
      candidate_count: 1,
      selected_candidate_index: -1,
      used_fallback: true,
      knowledge_base_path: "unavailable",
    },
    requires_human_approval: true,
    safety_plane: "read-only-copilot",
    telemetry_snapshot: telemetrySnapshot,
    refutation_status: "pending",
    timestamp: nowIso,
  };
}

export default function DashboardPage() {
  const configuredBodyBase = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");
  const BODY_BASE = configuredBodyBase || "/body";
  const API_BASE = configuredBodyBase ? `${configuredBodyBase}/api` : "/api";
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [brainHealth, setBrainHealth] = useState<BrainHealthPayload | null>(null);
  const [telemetry, setTelemetry] = useState<Record<string, Record<string, string | number>> | null>(null);
  const [incidentSummary, setIncidentSummary] = useState(INCIDENT_PRESETS[0].summary);
  const [selectedPreset, setSelectedPreset] = useState(INCIDENT_PRESETS[0].id);
  const [candidateCount, setCandidateCount] = useState(3);
  const [authorized, setAuthorized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackNote, setFeedbackNote] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Connecting to SRE-Nidaan runtime...");
  const [statusTone, setStatusTone] = useState<"info" | "success" | "error">("info");
  const [refutation, setRefutation] = useState<Record<string, unknown> | null>(null);
  const [integrationCheck, setIntegrationCheck] = useState<IntegrationCheck | null>(null);
  const [checkingIntegration, setCheckingIntegration] = useState(false);

  const serviceLinks = useMemo(
    () => [
      { label: "Body Health", href: `${BODY_BASE}/health`, hint: "Runtime health and model metadata" },
      { label: "Body API Docs", href: `${BODY_BASE}/docs`, hint: "Try requests live with OpenAPI UI" },
      { label: "Analyze Endpoint", href: `${BODY_BASE}/docs#/default/analyze_incident_api_analyze_incident_post`, hint: "Primary inference entrypoint" },
      { label: "Telemetry Feed", href: `${API_BASE}/telemetry`, hint: "Current incident telemetry payload" },
      { label: "Feedback Sink", href: `${BODY_BASE}/docs#/default/analysis_feedback_api_analysis_feedback_post`, hint: "Analyst rating submission endpoint" },
    ],
    [API_BASE, BODY_BASE]
  );

  const telemetryView = result?.telemetry_snapshot ?? telemetry ?? STATIC_TELEMETRY;

  const runIntegrationCheck = useCallback(
    async (seedHealth: HealthPayload | null) => {
      setCheckingIntegration(true);
      const sourceHealth = seedHealth;
      const brainHealthUrl = sourceHealth ? toBrainHealthUrl(sourceHealth.vllm_endpoint) : null;

      try {
        const [bodyRes, telemetryRes, brainRes] = await Promise.all([
          fetch(`${BODY_BASE}/health`),
          fetch(`${API_BASE}/telemetry`),
          brainHealthUrl ? fetch(brainHealthUrl) : Promise.resolve(null),
        ]);

        let brainStatus: IntegrationCheck["brain"] = "unknown";
        if (brainRes) {
          if (brainRes.ok) {
            const payload = (await brainRes.json()) as BrainHealthPayload;
            setBrainHealth(payload);
            if (payload.status === "ready") {
              brainStatus = "ready";
            } else if (payload.status === "warming") {
              brainStatus = "warming";
            } else {
              brainStatus = "error";
            }
          } else {
            brainStatus = "error";
          }
        }

        setIntegrationCheck({
          face: "online",
          body: bodyRes.ok ? "online" : "offline",
          telemetry: telemetryRes.ok ? "online" : "offline",
          brain: brainStatus,
          checked_at: new Date().toISOString(),
        });
      } catch {
        setIntegrationCheck({
          face: "online",
          body: "offline",
          telemetry: "offline",
          brain: "unknown",
          checked_at: new Date().toISOString(),
        });
      } finally {
        setCheckingIntegration(false);
      }
    },
    [API_BASE, BODY_BASE]
  );

  useEffect(() => {
    let disposed = false;

    async function bootstrap() {
      try {
        const [healthRes, telemetryRes] = await Promise.all([
          fetch(`${BODY_BASE}/health`),
          fetch(`${API_BASE}/telemetry`),
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
          const brainHealthResponse = await fetch(toBrainHealthUrl(healthPayload.vllm_endpoint));
          if (brainHealthResponse.ok) {
            brainPayload = (await brainHealthResponse.json()) as BrainHealthPayload;
          }
        } catch {
          // Best effort only.
        }

        if (!disposed) {
          setHealth(healthPayload);
          setBrainHealth(brainPayload);
          setTelemetry(telemetryPayload);
          setCandidateCount(Math.max(1, Math.min(8, healthPayload.default_candidate_count || 3)));
          setStatusTone("success");
          setStatusMessage("Runtime connected. Pick a scenario and run causal analysis.");
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
  }, [API_BASE, BODY_BASE, runIntegrationCheck]);

  const pollRefutation = useCallback(async () => {
    for (let attempt = 0; attempt < 6; attempt += 1) {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 1200);
      });
      try {
        const response = await fetch(`${API_BASE}/refutation-result`);
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
  }, [API_BASE]);

  const analyzeIncident = useCallback(async () => {
    setLoading(true);
    setAuthorized(false);
    setFeedbackStatus(null);
    setRefutation(null);
    setStatusTone("info");
    setStatusMessage("Running grounded causal analysis...");

    try {
      const response = await fetch(`${API_BASE}/analyze-incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          incident_summary: incidentSummary,
          candidate_count: candidateCount,
        }),
      });

      if (!response.ok) {
        throw new Error(`analysis failed (${response.status})`);
      }

      const payload: IncidentResult = await response.json();
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
      const fallback = buildLocalFallbackResult(
        incidentSummary,
        telemetry ?? STATIC_TELEMETRY
      );
      setResult(fallback);
      setTelemetry(fallback.telemetry_snapshot);
      setStatusTone("error");
      setStatusMessage("Live inference failed. Showing deterministic fallback analysis.");
    } finally {
      setLoading(false);
    }
  }, [API_BASE, candidateCount, incidentSummary, pollRefutation, telemetry]);

  const submitFeedback = useCallback(
    async (rating: "useful" | "needs_correction") => {
      if (!result) {
        return;
      }
      setFeedbackSubmitting(true);
      setFeedbackStatus(null);
      try {
        const response = await fetch(`${API_BASE}/analysis-feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            analysis_id: result.analysis_id,
            rating,
            correction: feedbackNote,
            incident_summary: incidentSummary,
            analysis: result.analysis,
            verifier: result.verifier,
            generation_metadata: result.generation_metadata,
          }),
        });

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
    [API_BASE, feedbackNote, incidentSummary, result]
  );

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
              id="analyze-incident-btn"
              onClick={analyzeIncident}
              disabled={loading}
              className="rounded-full bg-gradient-to-r from-nidaan-accent to-nidaan-accent-strong px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-nidaan-accent/25 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-55"
            >
              {loading ? "Analyzing..." : "Analyze Incident"}
            </button>
            <a
              href={`${BODY_BASE}/health`}
              className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent"
            >
              Runtime Health
            </a>
            <a
              href={`${BODY_BASE}/docs`}
              className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent"
            >
              API Docs
            </a>
            <button
              onClick={() => void runIntegrationCheck(health)}
              disabled={checkingIntegration}
              className="rounded-full border border-nidaan-border bg-white px-4 py-2 text-sm font-medium text-nidaan-ink transition hover:border-nidaan-accent/40 hover:text-nidaan-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {checkingIntegration ? "Checking..." : "Integration Check"}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1600px] grid-cols-1 gap-6 px-5 py-6 lg:px-8 xl:grid-cols-12">
        <section className="space-y-5 xl:col-span-4">
          <article className="nidaan-card p-5">
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
          </article>

          <article className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Runtime Pulse</h2>
              <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] ${
                statusTone === "success"
                  ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                  : statusTone === "error"
                    ? "border-nidaan-danger/30 bg-nidaan-danger/10 text-nidaan-danger"
                    : "border-nidaan-accent/30 bg-nidaan-accent/10 text-nidaan-accent-strong"
              }`}>
                <span className={`h-2 w-2 rounded-full ${statusTone === "error" ? "bg-nidaan-danger" : "animate-beacon bg-nidaan-success"}`} />
                {statusTone}
              </span>
            </div>
            <p className="text-sm text-nidaan-muted">{statusMessage}</p>

            {health && (
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
          </article>

          <article className="nidaan-card p-5">
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
          </article>

          <article className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Live Telemetry</h2>
              <span className="nidaan-mono text-[10px] uppercase tracking-[0.1em] text-nidaan-muted">streaming snapshot</span>
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
                          <span className={`nidaan-mono text-xs font-semibold ${metricLooksCritical(value) ? "metric-alert text-nidaan-danger" : "text-nidaan-ink"}`}>
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
        </section>

        <section className="space-y-5 xl:col-span-8">
          <article className="nidaan-card overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-nidaan-border/80 bg-white/80 px-5 py-3">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Causal Graph Workspace</h2>
              {result ? (
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className={`rounded-full border px-2 py-1 nidaan-mono ${
                    result.verifier.accepted
                      ? "border-nidaan-success/30 bg-nidaan-success/10 text-nidaan-success"
                      : "border-nidaan-warning/30 bg-nidaan-warning/10 text-nidaan-warning"
                  }`}>
                    verifier {result.verifier.accepted ? "accepted" : "fallback"}
                  </span>
                  <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-nidaan-muted">
                    {result.analysis.dag_nodes.length} nodes · {result.analysis.dag_edges.length} edges
                  </span>
                  <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-nidaan-muted">
                    {shortTimestamp(result.timestamp)}
                  </span>
                </div>
              ) : (
                <span className="rounded-full border border-nidaan-border bg-white px-2 py-1 nidaan-mono text-[11px] text-nidaan-muted">
                  waiting for first analysis
                </span>
              )}
            </div>
            <div className="h-[460px]">
              {result ? (
                <CausalGraph nodes={result.analysis.dag_nodes} edges={result.analysis.dag_edges} />
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

                {refutation && (
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
                  <button
                    id="authorize-intervention-btn"
                    onClick={() => setAuthorized(true)}
                    className="w-full rounded-xl bg-gradient-to-r from-nidaan-danger to-[#e1664b] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-nidaan-danger/30 transition hover:brightness-105"
                  >
                    Human-in-the-loop: Authorize Intervention
                  </button>
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
            <article className="nidaan-card p-5">
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

          <article className="nidaan-card p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="nidaan-display text-lg font-semibold text-nidaan-ink">Demo Links</h2>
              <span className="nidaan-chip">Everything connected</span>
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {serviceLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className="group rounded-2xl border border-nidaan-border bg-white p-3 transition hover:border-nidaan-accent/35 hover:shadow-md"
                >
                  <p className="text-sm font-semibold text-nidaan-ink group-hover:text-nidaan-accent-strong">{link.label}</p>
                  <p className="mt-1 text-xs text-nidaan-muted">{link.hint}</p>
                  <p className="mt-2 nidaan-mono text-[10px] text-nidaan-muted">{link.href}</p>
                </a>
              ))}
            </div>
          </article>
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
