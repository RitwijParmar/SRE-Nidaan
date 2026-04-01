import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

try:
    from fastapi.testclient import TestClient
    import backend.main as backend_main
except ImportError:  # pragma: no cover - dependency optional in local lightweight envs
    TestClient = None
    backend_main = None


@unittest.skipUnless(TestClient is not None and backend_main is not None, "fastapi runtime dependencies not installed")
class BackendRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(backend_main.app)
        backend_main._analysis_cache.clear()
        backend_main._analysis_tenants.clear()

    def test_analyze_incident_returns_grounded_metadata(self) -> None:
        generic_candidate = backend_main.CausalAnalysisResponse(
            root_cause="There is insufficient evidence to know the issue.",
            intervention_simulation="Scaling might help.",
            recommended_action="Scale up the service.",
            dag_nodes=[backend_main.DAGNode(id="node", label="Unknown")],
            dag_edges=[],
        )

        with patch.object(
            backend_main,
            "_generate_candidate_analyses",
            new=AsyncMock(return_value=[generic_candidate]),
        ), patch.object(
            backend_main,
            "_run_refutation_background",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post(
                "/api/analyze-incident",
                json={"incident_summary": "Login failures and auth latency spike."},
                headers={"x-tenant-id": "test-tenant"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generation_metadata"]["artifact_label"], "checkpoint-1064")
        self.assertEqual(payload["generation_metadata"]["source"], "grounded-fallback")
        self.assertGreaterEqual(len(payload["grounding_evidence"]), 1)
        self.assertIn("manual review", payload["analysis"]["recommended_action"].lower())

    def test_feedback_endpoint_appends_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "analyst_feedback.jsonl"
            feedback_db_path = Path(temp_dir) / "analyst_feedback.db"
            with patch.object(
                backend_main,
                "FEEDBACK_LOG_PATH",
                str(feedback_path),
            ), patch.object(
                backend_main,
                "FEEDBACK_DB_PATH",
                str(feedback_db_path),
            ):
                response = self.client.post(
                    "/api/analysis-feedback",
                    json={
                        "analysis_id": "analysis-test",
                        "rating": "useful",
                        "correction": "",
                        "incident_summary": "Synthetic incident for testing.",
                    },
                    headers={"x-tenant-id": "test-tenant"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(feedback_path.exists())
            log_lines = feedback_path.read_text().strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            self.assertIn("analysis-test", log_lines[0])
            self.assertTrue(feedback_db_path.exists())

    def test_mcp_tools_endpoint_lists_telemetry_tool(self) -> None:
        response = self.client.get(
            "/api/mcp/tools",
            headers={"x-tenant-id": "test-tenant"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        tool_names = {item["name"] for item in payload}
        self.assertIn("sre.telemetry.get_snapshot", tool_names)

    def test_authorize_intervention_requires_existing_analysis(self) -> None:
        response = self.client.post(
            "/api/interventions/authorize",
            json={
                "analysis_id": "analysis-missing",
                "operator_id": "alice",
                "tenant_id": "tenant-a",
                "reason": "Manual incident commander approval.",
            },
            headers={"x-tenant-id": "tenant-a"},
        )
        self.assertEqual(response.status_code, 404)

    def test_authorize_intervention_rejects_tenant_mismatch(self) -> None:
        backend_main._analysis_cache["analysis-tenant-test"] = {
            "requires_human_approval": True,
            "analysis": {"recommended_action": "Do safe rollout"},
        }
        backend_main._analysis_tenants["analysis-tenant-test"] = "tenant-a"
        response = self.client.post(
            "/api/interventions/authorize",
            json={
                "analysis_id": "analysis-tenant-test",
                "operator_id": "alice",
                "tenant_id": "tenant-b",
                "reason": "Cross-tenant attempt should fail.",
            },
            headers={"x-tenant-id": "tenant-b"},
        )
        self.assertEqual(response.status_code, 403)

    def test_api_endpoints_require_tenant_header(self) -> None:
        response = self.client.get("/api/mcp/tools")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
