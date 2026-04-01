# SRE-Nidaan Product Completion Roadmap (April 1, 2026)

## 1) Current Product Maturity (Codebase Audit)

The system already has a strong technical base:
- split microservices architecture (Face + Body + Brain)
- structured causal output with human-approval gating
- grounding + verifier + fallback strategy in backend
- Cloud Run deployment scripts and basic tests

Main blockers to call this a complete production product:
- no authentication or role-based access control
- no persistent operational datastore (feedback is JSONL only)
- no tenant/workspace isolation and no audit trail
- no real telemetry ingestion pipeline (static telemetry payload)
- no alerting/on-call integration with SLO/error-budget workflow
- permissive runtime security defaults (`allow_origins=["*"]`, public Brain)
- limited release gates (no automated canary verification pipeline)

## 2) What Is Needed for a Complete Product

### A. Product and UX completeness
- Login + team/workspace onboarding flow (OIDC SSO)
- Incident lifecycle model: open, triage, investigate, mitigate, resolved, postmortem
- Analyst collaboration: comments, ownership, approvals, handoff notes
- Shareable incident report export (JSON + markdown/PDF)
- Operator action console: explicit "proposed action", "approved action", "executed action", "rollback"

### B. Data and platform completeness
- Replace JSONL feedback with managed database (Postgres preferred)
- Add entity model: organizations, users, incidents, analyses, approvals, feedback, runbook citations
- Add idempotent API contracts and pagination for incident history
- Introduce async work queue for long jobs (refutation/evaluation) and status polling
- Add retention policy and backup/restore procedures

### C. Reliability and SRE completeness
- Define SLIs/SLOs and error budgets for each service (Face, Body, Brain)
- Add production telemetry via OpenTelemetry collector and standardized traces/logs/metrics
- Alerting policy with symptom-based, actionable alerts and escalation paths
- On-call integration (PagerDuty/Opsgenie/Slack) for incident routing
- Release strategy: canary revisions with automated rollback thresholds

### D. Security and compliance completeness
- OIDC-based authN and role-based authZ (viewer/analyst/approver/admin)
- Service-to-service auth on Cloud Run (ID token audience checks)
- Private ingress for Brain; expose only required endpoints
- Secrets hardening and rotation (HF token and all external credentials)
- API security controls aligned to OWASP API Top 10 (2023)
- Supply chain controls: signed provenance, dependency scanning, build policy gates
- Full audit logging for approvals, overrides, and incident state transitions

### E. Model and AI operations completeness
- Keep checkpoint-1064 as serving baseline unless challenger beats it on gated eval
- Shadow evaluation of challenger models before traffic exposure
- Offline + online eval suite with safety and schema adherence as hard gates
- Feedback-to-training loop with prompt-matched preferences and clear dataset lineage
- Model registry with artifact provenance, rollback metadata, and promotion policy

## 3) Phased Execution Plan

### Phase 0 (3-5 days): Demo reliability hardening
- finalize reliable clickable endpoint actions in Face
- remove aggressive cache on home route to prevent stale UI after deploy
- unify health checks and integration diagnostics panel behavior
- add structured incident report export from existing response payload

Exit criteria:
- all top action links and endpoint cards open reliably
- zero stale-homepage issues after redeploy
- 95%+ success in repeated analyze + feedback demo loop

### Phase 1 (1-2 weeks): Production MVP
- add OIDC login (Google identity first), session handling, role checks
- migrate feedback + incident records to Postgres
- implement incident CRUD + timeline + ownership API
- wire OpenTelemetry traces/metrics/logs from all services
- set first SLO set and alerting routes

Exit criteria:
- authenticated multi-user operation with role enforcement
- incident history survives restarts and deploys
- paging/alerts triggered for real SLO breaches

### Phase 2 (2-4 weeks): Enterprise-ready operations
- tenant/workspace isolation model and access boundaries
- canary + rollback automation in CI/CD
- policy engine for approval rules by risk level
- audit trail and compliance report endpoints
- supply-chain verification gates (provenance + signature checks)

Exit criteria:
- controlled progressive delivery with automated rollback
- complete auditability for high-risk decisions
- repeatable compliance-ready operational evidence

## 4) Immediate P0 Work Items (Recommended Next 7 Items)

1. Implement OIDC auth and role middleware in Body.
2. Move feedback/incident persistence to Postgres.
3. Add incident timeline APIs and UI workflow states.
4. Integrate OpenTelemetry collector and correlation IDs end-to-end.
5. Add SLO dashboards and actionable alerts.
6. Restrict Brain ingress and enforce service-to-service authentication.
7. Add canary rollout verification + rollback automation on Cloud Run.

## 5) Success Metrics to Track

- Product reliability: p95 analyze latency, request success rate, UI action success rate
- Incident quality: analyst useful-rate, correction-rate, approval turnaround time
- Safety quality: unauthorized action attempts blocked, manual-approval compliance
- Delivery health: deployment frequency, lead time, failure rate, restoration time
- Platform trust: security findings age, secret rotation compliance, audit completeness

## 6) Standards and References Used

- Google SRE Book: Service Level Objectives  
  https://sre.google/sre-book/service-level-objectives/
- Google SRE Book: Emergency Response  
  https://sre.google/sre-book/emergency-response/
- DORA metrics guidance  
  https://dora.dev/guides/dora-metrics/
- NIST SP 800-61r3 (final, April 2025) incident response recommendations  
  https://csrc.nist.gov/pubs/sp/800/61/r3/final
- NIST CSF 2.0 (final, Feb 26, 2024)  
  https://doi.org/10.6028/NIST.CSWP.29
- NIST AI RMF 1.0  
  https://doi.org/10.6028/NIST.AI.100-1
- OpenTelemetry collector/docs  
  https://opentelemetry.io/docs/collector/
- OWASP API Security Top 10 (2023)  
  https://owasp.org/API-Security/editions/2023/en/0x11-t10/
- OpenID Connect Core 1.0  
  https://openid.net/specs/openid-connect-core-1_0.html
- SLSA build security levels  
  https://slsa.dev/spec/v1.0/levels
