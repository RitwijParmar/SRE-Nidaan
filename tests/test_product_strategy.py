from pathlib import Path
import unittest

from src.runtime.product_strategy import (
    build_grounded_fallback_analysis,
    retrieve_grounding_evidence,
    score_candidate_analysis,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = ROOT_DIR / "ops" / "knowledge_base.json"
TELEMETRY = {
    "frontend": {"status": "503 Gateway Timeout", "error_rate": "Spiking"},
    "auth_service": {"cpu_utilization": "96%", "latency_ms": 4500, "replicas": 5},
    "database": {"connections": "990/1000 (99%)", "wait_event": "ClientRead (Locked)"},
}


class ProductStrategyTests(unittest.TestCase):
    def test_retrieve_grounding_evidence_returns_runbook_and_policy_docs(self) -> None:
        evidence = retrieve_grounding_evidence(
            TELEMETRY,
            incident_summary="Auth requests are timing out and DB connections are pinned near 100%",
            knowledge_base_path=KNOWLEDGE_BASE_PATH,
            limit=6,
        )
        evidence_ids = {item["id"] for item in evidence}
        self.assertIn("runbook-db-connection-pool", evidence_ids)
        self.assertIn("postmortem-panic-scaling-auth", evidence_ids)

    def test_generic_candidate_is_rejected(self) -> None:
        evidence = retrieve_grounding_evidence(
            TELEMETRY,
            incident_summary="Users see login timeouts.",
            knowledge_base_path=KNOWLEDGE_BASE_PATH,
            limit=4,
        )
        generic_candidate = {
            "root_cause": "There is insufficient evidence and we need more data.",
            "intervention_simulation": "Scaling may help, but we cannot determine the cause yet.",
            "recommended_action": "Scale up the service.",
            "dag_nodes": [{"id": "a", "label": "Unknown issue"}],
            "dag_edges": [],
        }
        verdict = score_candidate_analysis(
            generic_candidate,
            telemetry=TELEMETRY,
            grounding_evidence=evidence,
        )
        self.assertFalse(verdict["accepted"])
        self.assertGreater(verdict["generic_penalty"], 0)

    def test_grounded_fallback_is_specific_and_safe(self) -> None:
        evidence = retrieve_grounding_evidence(
            TELEMETRY,
            incident_summary="Frontend login requests are failing with 503s",
            knowledge_base_path=KNOWLEDGE_BASE_PATH,
            limit=4,
        )
        fallback = build_grounded_fallback_analysis(
            TELEMETRY,
            grounding_evidence=evidence,
        )
        verdict = score_candidate_analysis(
            fallback,
            telemetry=TELEMETRY,
            grounding_evidence=evidence,
        )
        self.assertIn("retry storm", fallback["root_cause"].lower())
        self.assertIn("manual review", fallback["recommended_action"].lower())
        self.assertTrue(verdict["accepted"])


if __name__ == "__main__":
    unittest.main()
