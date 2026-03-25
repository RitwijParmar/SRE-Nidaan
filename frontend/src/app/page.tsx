"use client";

import { useState, useCallback } from "react";
import CausalGraph from "@/components/CausalGraph";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

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

interface IncidentResult {
  analysis: CausalAnalysis;
  requires_human_approval: boolean;
  safety_plane: string;
  telemetry_snapshot: Record<string, Record<string, string | number>>;
  refutation_status: string;
  timestamp: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Static telemetry (mirrors backend mock for immediate display)
// ─────────────────────────────────────────────────────────────────────────────

const STATIC_TELEMETRY = {
  frontend: { status: "503 Gateway Timeout", error_rate: "Spiking" },
  auth_service: { cpu_utilization: "96%", latency_ms: 4500, replicas: 5 },
  database: {
    connections: "990/1000 (99%)",
    wait_event: "ClientRead (Locked)",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Severity helpers
// ─────────────────────────────────────────────────────────────────────────────

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
    frontend: { text: "CRITICAL", color: "bg-red-500/20 text-red-400 border-red-500/30" },
    auth_service: { text: "WARNING", color: "bg-amber-500/20 text-amber-400 border-amber-500/30" },
    database: { text: "CRITICAL", color: "bg-red-500/20 text-red-400 border-red-500/30" },
  };
  return map[service] || { text: "INFO", color: "bg-blue-500/20 text-blue-400 border-blue-500/30" };
}

// ─────────────────────────────────────────────────────────────────────────────
// Page Component
// ─────────────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [authorized, setAuthorized] = useState(false);
  const [refutation, setRefutation] = useState<Record<string, unknown> | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const analyzeIncident = useCallback(async () => {
    setLoading(true);
    setAuthorized(false);
    setRefutation(null);
    try {
      const res = await fetch(`${API_BASE}/api/analyze-incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data: IncidentResult = await res.json();
      setResult(data);

      // Poll for refutation result after a delay
      setTimeout(async () => {
        try {
          const refRes = await fetch(`${API_BASE}/api/refutation-result`);
          const refData = await refRes.json();
          setRefutation(refData);
        } catch {
          /* refutation poll failed — non-critical */
        }
      }, 3000);
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  const handleAuthorize = () => {
    setAuthorized(true);
  };

  return (
    <div className="min-h-screen bg-nidaan-bg bg-grid-pattern bg-grid">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="border-b border-nidaan-border bg-nidaan-surface/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <span className="text-white font-bold text-lg">N</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                SRE-Nidaan
              </h1>
              <p className="text-xs text-nidaan-muted font-mono">
                NEXUS-CAUSAL v3.1 · Pearl&apos;s Causal Hierarchy
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20">
              <span className="w-2 h-2 rounded-full bg-red-500 status-critical" />
              <span className="text-xs text-red-400 font-medium">
                INCIDENT ACTIVE
              </span>
            </div>
            <button
              id="analyze-incident-btn"
              onClick={analyzeIncident}
              disabled={loading}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 
                         text-white text-sm font-semibold shadow-lg shadow-blue-500/25 
                         hover:shadow-blue-500/40 hover:from-blue-500 hover:to-blue-400
                         disabled:opacity-50 disabled:cursor-not-allowed 
                         transition-all duration-300 active:scale-95"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  Analyzing…
                </span>
              ) : (
                "⚡ Analyze Incident"
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Grid ───────────────────────────────────────────────── */}
      <main className="max-w-[1920px] mx-auto p-6">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* ── LEFT PANEL: Live Telemetry ────────────────────────── */}
          <div className="xl:col-span-4 space-y-5 animate-fade-in">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-semibold text-nidaan-text-dim uppercase tracking-wider">
                Live Telemetry
              </span>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            </div>

            {Object.entries(STATIC_TELEMETRY).map(([service, metrics]) => {
              const badge = getSeverityBadge(service);
              return (
                <div
                  key={service}
                  id={`telemetry-${service}`}
                  className={`glass-card p-5 ${getSeverityBorder(service)} border animate-slide-up`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className={`text-base font-bold ${getSeverityColor(service)}`}>
                      {service.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </h3>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono font-bold ${badge.color}`}>
                      {badge.text}
                    </span>
                  </div>
                  <div className="space-y-2.5">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-xs text-nidaan-muted font-mono">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span
                          className={`text-sm font-semibold font-mono ${
                            typeof value === "string" &&
                            (value.includes("503") ||
                              value.includes("99%") ||
                              value.includes("96%") ||
                              value.includes("Spiking") ||
                              value.includes("Locked"))
                              ? "text-red-400 metric-flash"
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

            {/* Refutation Test Result */}
            {refutation && (
              <div className="glass-card p-5 border border-cyan-500/30 animate-slide-up">
                <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center gap-2">
                  <span>🧪</span> Refutation Test
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-xs text-nidaan-muted font-mono">type</span>
                    <span className="text-xs text-nidaan-text font-mono">
                      {String(refutation.test_type || "—")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-nidaan-muted font-mono">confounder</span>
                    <span className="text-xs text-nidaan-text font-mono">
                      {String(refutation.injected_confounder || "—")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-nidaan-muted font-mono">verdict</span>
                    <span
                      className={`text-xs font-mono font-bold ${
                        refutation.is_robust ? "text-green-400" : "text-amber-400"
                      }`}
                    >
                      {refutation.is_robust ? "PASS ✓" : "WARN ⚠"}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── RIGHT PANEL: DAG + Intervention ──────────────────── */}
          <div className="xl:col-span-8 space-y-5 animate-fade-in">
            {/* ── Causal DAG Visualizer ────────────────────────────── */}
            <div className="glass-card border border-nidaan-border">
              <div className="px-5 py-3 border-b border-nidaan-border flex items-center justify-between">
                <h2 className="text-sm font-semibold text-nidaan-text-dim uppercase tracking-wider">
                  Causal DAG Visualizer
                </h2>
                {result && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/25 font-mono">
                    {result.analysis.dag_nodes.length} nodes · {result.analysis.dag_edges.length} edges
                  </span>
                )}
              </div>
              <div className="h-[450px]">
                {result ? (
                  <CausalGraph
                    nodes={result.analysis.dag_nodes}
                    edges={result.analysis.dag_edges}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <div className="text-4xl mb-3 opacity-30">🔬</div>
                      <p className="text-nidaan-muted text-sm">
                        Click <span className="text-blue-400 font-semibold">Analyze Incident</span> to generate the causal DAG
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── Root Cause & Intervention Simulation ────────────── */}
            {result && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 animate-slide-up">
                <div className="glass-card p-5 border border-nidaan-border">
                  <h3 className="text-sm font-bold text-blue-400 mb-3 flex items-center gap-2">
                    <span>🎯</span> Root Cause
                  </h3>
                  <p className="text-sm text-nidaan-text leading-relaxed">
                    {result.analysis.root_cause}
                  </p>
                </div>

                <div className="glass-card p-5 border border-amber-500/30">
                  <h3 className="text-sm font-bold text-amber-400 mb-3 flex items-center gap-2">
                    <span>⚠️</span> Intervention Simulation
                  </h3>
                  <p className="text-sm text-nidaan-text leading-relaxed">
                    {result.analysis.intervention_simulation}
                  </p>
                </div>
              </div>
            )}

            {/* ── Intervention Engine (Safety Plane) ──────────────── */}
            {result && (
              <div
                id="intervention-engine"
                className={`glass-card p-6 border-2 transition-colors duration-500 ${
                  authorized ? "border-green-500/50" : "border-red-500/40"
                } animate-slide-up`}
              >
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold text-nidaan-text-dim uppercase tracking-wider">
                    Intervention Engine
                  </h2>
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        authorized ? "bg-green-500" : "bg-red-500 status-critical"
                      }`}
                    />
                    <span
                      className={`text-[10px] font-mono font-bold ${
                        authorized ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {authorized ? "AUTHORIZED" : "AWAITING HUMAN APPROVAL"}
                    </span>
                  </div>
                </div>

                <div className="bg-nidaan-bg/50 rounded-lg p-4 mb-5 border border-nidaan-border">
                  <p className="text-xs text-nidaan-muted font-mono mb-1">
                    RECOMMENDED ACTION
                  </p>
                  <p className="text-sm text-nidaan-text font-medium leading-relaxed">
                    {result.analysis.recommended_action}
                  </p>
                </div>

                <div className="flex items-center gap-4">
                  {!authorized ? (
                    <button
                      id="authorize-intervention-btn"
                      onClick={handleAuthorize}
                      className="flex-1 py-3.5 rounded-xl bg-gradient-to-r from-red-600 to-red-500 
                                 text-white font-bold text-sm shadow-lg btn-danger-glow
                                 hover:from-red-500 hover:to-red-400 
                                 active:scale-[0.98] transition-all duration-300"
                    >
                      🛡️ Human-in-the-Loop: Authorize Intervention
                    </button>
                  ) : (
                    <div className="flex-1 py-3.5 rounded-xl bg-gradient-to-r from-green-600/80 to-emerald-500/80 
                                    text-center text-white font-bold text-sm border border-green-500/30
                                    shadow-lg shadow-green-500/15"
                    >
                      ✅ Intervention Authorized — Executing Action
                    </div>
                  )}
                </div>

                {result.requires_human_approval && !authorized && (
                  <p className="text-[11px] text-nidaan-muted mt-3 text-center font-mono">
                    Safety Plane: <span className="text-amber-400">{result.safety_plane}</span>
                    {" · "}Execution is blocked until a human operator authorizes the intervention.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="border-t border-nidaan-border mt-8 py-4 px-6">
        <div className="max-w-[1920px] mx-auto flex items-center justify-between text-xs text-nidaan-muted font-mono">
          <span>SRE-Nidaan · NEXUS-CAUSAL v3.1 · Pearl&apos;s do-calculus</span>
          <span>
            Safety: Read-Only Copilot · Blast Radius: Contained
          </span>
        </div>
      </footer>
    </div>
  );
}
