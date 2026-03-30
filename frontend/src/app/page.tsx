"use client";

import { useCallback, useState } from "react";
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

const STATIC_TELEMETRY = {
  frontend: { status: "503 Gateway Timeout", error_rate: "Spiking" },
  auth_service: { cpu_utilization: "96%", latency_ms: 4500, replicas: 5 },
  database: {
    connections: "990/1000 (99%)",
    wait_event: "ClientRead (Locked)",
  },
};

function getSeverityColor(service: string): string {
  const map: Record<string, string> = {
    frontend: "text-red-400",
    auth_service: "text-amber-400",
    database: "text-red-500",
  };
  return map[service] || "text-blue-400";
}

function getSeverityBorder(service: string): string {
  const map: Record<string, string> = {
    frontend: "border-red-500/40",
    auth_service: "border-amber-500/40",
    database: "border-red-600/40",
  };
  return map[service] || "border-blue-500/40";
}

function getSeverityBadge(service: string): { text: string; color: string } {
  const map: Record<string, { text: string; color: string }> = {
    frontend: {
      text: "CRITICAL",
      color: "bg-red-500/20 text-red-400 border-red-500/30",
    },
    auth_service: {
      text: "WARNING",
      color: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    },
    database: {
      text: "CRITICAL",
      color: "bg-red-500/20 text-red-400 border-red-500/30",
    },
  };
  return map[service] || {
    text: "INFO",
    color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  };
}

function prettifyLabel(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function DashboardPage() {
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [authorized, setAuthorized] = useState(false);
  const [incidentSummary, setIncidentSummary] = useState(
    "Users report 503s during login bursts. Auth latency spiked before the database saturated."
  );
  const [refutation, setRefutation] = useState<Record<string, unknown> | null>(null);
  const [feedbackNote, setFeedbackNote] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const telemetryView = result?.telemetry_snapshot ?? STATIC_TELEMETRY;

  const analyzeIncident = useCallback(async () => {
    setLoading(true);
    setAuthorized(false);
    setRefutation(null);
    setFeedbackStatus(null);
    try {
      const res = await fetch(`${API_BASE}/api/analyze-incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          incident_summary: incidentSummary,
        }),
      });
      if (!res.ok) {
        throw new Error(`Analyze request failed: ${res.status}`);
      }

      const data: IncidentResult = await res.json();
      setResult(data);

      window.setTimeout(async () => {
        try {
          const refRes = await fetch(`${API_BASE}/api/refutation-result`);
          if (!refRes.ok) {
            return;
          }
          const refData = await refRes.json();
          setRefutation(refData);
        } catch {
          // Non-critical background poll.
        }
      }, 3000);
    } catch (err) {
      console.error("Analysis failed:", err);
      setFeedbackStatus("Analysis failed. Check the backend logs and try again.");
    } finally {
      setLoading(false);
    }
  }, [API_BASE, incidentSummary]);

  const submitFeedback = useCallback(
    async (rating: "useful" | "needs_correction") => {
      if (!result) {
        return;
      }
      setFeedbackSubmitting(true);
      setFeedbackStatus(null);
      try {
        const res = await fetch(`${API_BASE}/api/analysis-feedback`, {
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
        if (!res.ok) {
          throw new Error(`Feedback request failed: ${res.status}`);
        }
        setFeedbackStatus(
          rating === "useful"
            ? "Analyst feedback saved: marked useful."
            : "Analyst feedback saved: correction queued for review."
        );
        if (rating === "needs_correction") {
          setFeedbackNote("");
        }
      } catch (err) {
        console.error("Feedback failed:", err);
        setFeedbackStatus("Feedback submission failed. Please retry.");
      } finally {
        setFeedbackSubmitting(false);
      }
    },
    [API_BASE, feedbackNote, incidentSummary, result]
  );

  const handleAuthorize = () => {
    setAuthorized(true);
  };

  return (
    <div className="min-h-screen bg-nidaan-bg bg-grid-pattern bg-grid">
      <header className="sticky top-0 z-50 border-b border-nidaan-border bg-nidaan-surface/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1920px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 shadow-lg shadow-blue-500/20">
              <span className="text-lg font-bold text-white">N</span>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">
                SRE-Nidaan
              </h1>
              <p className="font-mono text-xs text-nidaan-muted">
                Production Causal Copilot · Grounded Baseline Path
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-1.5">
              <span className="status-critical h-2 w-2 rounded-full bg-red-500" />
              <span className="text-xs font-medium text-red-400">
                INCIDENT ACTIVE
              </span>
            </div>
            <button
              id="analyze-incident-btn"
              onClick={analyzeIncident}
              disabled={loading}
              className="rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition-all duration-300 hover:from-blue-500 hover:to-blue-400 hover:shadow-blue-500/40 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                  Analyzing...
                </span>
              ) : (
                "Analyze Incident"
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1920px] p-6">
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
          <div className="space-y-5 xl:col-span-4">
            <div className="glass-card border border-nidaan-border p-5">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-semibold uppercase tracking-wider text-nidaan-text-dim">
                  Incident Brief
                </span>
                {result && (
                  <span className="rounded-full border border-blue-500/25 bg-blue-500/15 px-2 py-0.5 font-mono text-[10px] text-blue-300">
                    {result.generation_metadata.artifact_label}
                  </span>
                )}
              </div>
              <textarea
                value={incidentSummary}
                onChange={(event) => setIncidentSummary(event.target.value)}
                rows={5}
                className="w-full rounded-xl border border-nidaan-border bg-nidaan-bg/70 p-3 text-sm leading-relaxed text-nidaan-text outline-none transition focus:border-blue-500/40"
                placeholder="Summarize the operator report, blast radius, or customer symptoms."
              />
              <p className="mt-3 text-xs text-nidaan-muted">
                The backend combines this note with live telemetry and runbook evidence before it picks a candidate.
              </p>
            </div>

            <div className="mb-1 flex items-center gap-2">
              <span className="text-sm font-semibold uppercase tracking-wider text-nidaan-text-dim">
                Live Telemetry
              </span>
              <span className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
            </div>

            {Object.entries(telemetryView).map(([service, metrics]) => {
              const badge = getSeverityBadge(service);
              return (
                <div
                  key={service}
                  id={`telemetry-${service}`}
                  className={`glass-card animate-slide-up border p-5 ${getSeverityBorder(service)}`}
                >
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className={`text-base font-bold ${getSeverityColor(service)}`}>
                      {prettifyLabel(service)}
                    </h3>
                    <span
                      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] font-bold ${badge.color}`}
                    >
                      {badge.text}
                    </span>
                  </div>
                  <div className="space-y-2.5">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="font-mono text-xs text-nidaan-muted">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span
                          className={`font-mono text-sm font-semibold ${
                            typeof value === "string" &&
                            (value.includes("503") ||
                              value.includes("99%") ||
                              value.includes("96%") ||
                              value.includes("Spiking") ||
                              value.includes("Locked"))
                              ? "metric-flash text-red-400"
                              : "text-nidaan-text"
                          }`}
                        >
                          {String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            {refutation && (
              <div className="glass-card animate-slide-up border border-cyan-500/30 p-5">
                <h3 className="mb-3 flex items-center gap-2 text-sm font-bold text-cyan-400">
                  Refutation Test
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-mono text-xs text-nidaan-muted">type</span>
                    <span className="font-mono text-xs text-nidaan-text">
                      {String(refutation.test_type || "-")}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="font-mono text-xs text-nidaan-muted">confounder</span>
                    <span className="text-right font-mono text-xs text-nidaan-text">
                      {String(refutation.injected_confounder || "-")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono text-xs text-nidaan-muted">verdict</span>
                    <span
                      className={`font-mono text-xs font-bold ${
                        refutation.is_robust ? "text-green-400" : "text-amber-400"
                      }`}
                    >
                      {refutation.is_robust ? "PASS" : "WARN"}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-5 xl:col-span-8">
            <div className="glass-card border border-nidaan-border">
              <div className="flex items-center justify-between border-b border-nidaan-border px-5 py-3">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-nidaan-text-dim">
                  Causal DAG Visualizer
                </h2>
                {result && (
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] ${
                        result.verifier.accepted
                          ? "border-green-500/25 bg-green-500/15 text-green-300"
                          : "border-amber-500/25 bg-amber-500/15 text-amber-300"
                      }`}
                    >
                      verifier {result.verifier.accepted ? "accepted" : "fallback"}
                    </span>
                    <span className="rounded-full border border-blue-500/25 bg-blue-500/15 px-2 py-0.5 font-mono text-[10px] text-blue-400">
                      {result.analysis.dag_nodes.length} nodes · {result.analysis.dag_edges.length} edges
                    </span>
                  </div>
                )}
              </div>
              <div className="h-[450px]">
                {result ? (
                  <CausalGraph
                    nodes={result.analysis.dag_nodes}
                    edges={result.analysis.dag_edges}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-center">
                      <div className="mb-3 text-4xl opacity-30">DAG</div>
                      <p className="text-sm text-nidaan-muted">
                        Click <span className="font-semibold text-blue-400">Analyze Incident</span> to generate the causal DAG.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {result && (
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div className="glass-card border border-nidaan-border p-5">
                  <h3 className="mb-3 text-sm font-bold text-blue-400">
                    Root Cause
                  </h3>
                  <p className="text-sm leading-relaxed text-nidaan-text">
                    {result.analysis.root_cause}
                  </p>
                </div>

                <div className="glass-card border border-amber-500/30 p-5">
                  <h3 className="mb-3 text-sm font-bold text-amber-400">
                    Intervention Simulation
                  </h3>
                  <p className="text-sm leading-relaxed text-nidaan-text">
                    {result.analysis.intervention_simulation}
                  </p>
                </div>
              </div>
            )}

            {result && (
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div className="glass-card border border-nidaan-border p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-bold text-cyan-400">
                      Grounding Evidence
                    </h3>
                    <span className="font-mono text-[10px] text-nidaan-muted">
                      {result.grounding_evidence.length} docs
                    </span>
                  </div>
                  <div className="space-y-3">
                    {result.grounding_evidence.map((evidence) => (
                      <div
                        key={evidence.id}
                        className="rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3"
                      >
                        <div className="mb-1 flex items-center justify-between gap-3">
                          <span className="text-xs font-semibold text-nidaan-text">
                            {evidence.title}
                          </span>
                          <span className="font-mono text-[10px] uppercase text-cyan-300">
                            {evidence.kind}
                          </span>
                        </div>
                        <p className="text-xs leading-relaxed text-nidaan-muted">
                          {evidence.summary}
                        </p>
                        {evidence.matched_terms.length > 0 && (
                          <p className="mt-2 font-mono text-[10px] text-cyan-300">
                            matched: {evidence.matched_terms.join(", ")}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-card border border-nidaan-border p-5">
                  <h3 className="mb-3 text-sm font-bold text-emerald-400">
                    Generation Verifier
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3">
                      <p className="font-mono text-[10px] text-nidaan-muted">score</p>
                      <p className="text-xl font-bold text-white">
                        {result.verifier.score.toFixed(3)}
                      </p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3">
                      <p className="font-mono text-[10px] text-nidaan-muted">source</p>
                      <p className="text-sm font-semibold text-white">
                        {result.generation_metadata.source}
                      </p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3">
                      <p className="font-mono text-[10px] text-nidaan-muted">evidence overlap</p>
                      <p className="text-sm font-semibold text-white">
                        {result.verifier.evidence_overlap}
                      </p>
                    </div>
                    <div className="rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3">
                      <p className="font-mono text-[10px] text-nidaan-muted">telemetry overlap</p>
                      <p className="text-sm font-semibold text-white">
                        {result.verifier.telemetry_overlap}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 rounded-xl border border-nidaan-border bg-nidaan-bg/50 p-3">
                    <p className="mb-2 font-mono text-[10px] text-nidaan-muted">
                      generation metadata
                    </p>
                    <div className="space-y-1 text-xs text-nidaan-text">
                      <p>artifact: {result.generation_metadata.artifact_label}</p>
                      <p>model: {result.generation_metadata.model_id}</p>
                      <p>
                        candidate selection:{" "}
                        {result.generation_metadata.selected_candidate_index >= 0
                          ? `${result.generation_metadata.selected_candidate_index + 1} / ${result.generation_metadata.candidate_count}`
                          : `fallback / ${result.generation_metadata.candidate_count}`}
                      </p>
                    </div>
                    <p className="mt-3 text-xs text-nidaan-muted">
                      {result.verifier.reasons.join(" · ")}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {result && (
              <div
                id="intervention-engine"
                className={`glass-card animate-slide-up border-2 p-6 transition-colors duration-500 ${
                  authorized ? "border-green-500/50" : "border-red-500/40"
                }`}
              >
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-nidaan-text-dim">
                    Intervention Engine
                  </h2>
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        authorized ? "bg-green-500" : "status-critical bg-red-500"
                      }`}
                    />
                    <span
                      className={`font-mono text-[10px] font-bold ${
                        authorized ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {authorized ? "AUTHORIZED" : "AWAITING HUMAN APPROVAL"}
                    </span>
                  </div>
                </div>

                <div className="mb-5 rounded-lg border border-nidaan-border bg-nidaan-bg/50 p-4">
                  <p className="mb-1 font-mono text-xs text-nidaan-muted">
                    RECOMMENDED ACTION
                  </p>
                  <p className="text-sm font-medium leading-relaxed text-nidaan-text">
                    {result.analysis.recommended_action}
                  </p>
                </div>

                <div className="flex items-center gap-4">
                  {!authorized ? (
                    <button
                      id="authorize-intervention-btn"
                      onClick={handleAuthorize}
                      className="btn-danger-glow flex-1 rounded-xl bg-gradient-to-r from-red-600 to-red-500 py-3.5 text-sm font-bold text-white shadow-lg transition-all duration-300 hover:from-red-500 hover:to-red-400 active:scale-[0.98]"
                    >
                      Human-in-the-Loop: Authorize Intervention
                    </button>
                  ) : (
                    <div className="flex-1 rounded-xl border border-green-500/30 bg-gradient-to-r from-green-600/80 to-emerald-500/80 py-3.5 text-center text-sm font-bold text-white shadow-lg shadow-green-500/15">
                      Intervention Authorized - Executing Action
                    </div>
                  )}
                </div>

                {result.requires_human_approval && !authorized && (
                  <p className="mt-3 text-center font-mono text-[11px] text-nidaan-muted">
                    Safety Plane: <span className="text-amber-400">{result.safety_plane}</span>
                    {" · "}Execution remains blocked until a human operator authorizes the intervention.
                  </p>
                )}
              </div>
            )}

            {result && (
              <div className="glass-card border border-nidaan-border p-5">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-bold text-violet-300">
                    Analyst Feedback Loop
                  </h3>
                  <span className="font-mono text-[10px] text-nidaan-muted">
                    analysis {result.analysis_id}
                  </span>
                </div>
                <p className="mb-3 text-sm text-nidaan-muted">
                  Mark strong outputs immediately, or leave a correction note so we can promote real analyst preferences into the next tuning pass.
                </p>
                <textarea
                  value={feedbackNote}
                  onChange={(event) => setFeedbackNote(event.target.value)}
                  rows={3}
                  className="w-full rounded-xl border border-nidaan-border bg-nidaan-bg/70 p-3 text-sm text-nidaan-text outline-none transition focus:border-violet-500/40"
                  placeholder="Optional correction or missing evidence..."
                />
                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    onClick={() => submitFeedback("useful")}
                    disabled={feedbackSubmitting}
                    className="rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-2 text-sm font-semibold text-green-300 transition hover:bg-green-500/20 disabled:opacity-50"
                  >
                    Mark Useful
                  </button>
                  <button
                    onClick={() => submitFeedback("needs_correction")}
                    disabled={feedbackSubmitting}
                    className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-sm font-semibold text-amber-300 transition hover:bg-amber-500/20 disabled:opacity-50"
                  >
                    Needs Correction
                  </button>
                </div>
                {feedbackStatus && (
                  <p className="mt-3 text-sm text-violet-200">{feedbackStatus}</p>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-8 border-t border-nidaan-border px-6 py-4">
        <div className="mx-auto flex max-w-[1920px] items-center justify-between font-mono text-xs text-nidaan-muted">
          <span>SRE-Nidaan · Stable SFT baseline + grounding + verifier</span>
          <span>Safety: Read-Only Copilot · Blast Radius: Contained</span>
        </div>
      </footer>
    </div>
  );
}
