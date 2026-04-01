import asyncio
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

    def test_analyze_incident_skips_llm_when_brain_not_ready(self) -> None:
        mocked_generate = AsyncMock(return_value=[])
        with patch.object(
            backend_main,
            "_brain_ready_for_inference",
            new=AsyncMock(return_value=False),
        ), patch.object(
            backend_main,
            "_generate_candidate_analyses",
            new=mocked_generate,
        ), patch.object(
            backend_main,
            "_run_refutation_background",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post(
                "/api/analyze-incident",
                json={"incident_summary": "Auth retries with DB saturation."},
                headers={"x-tenant-id": "test-tenant"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generation_metadata"]["source"], "grounded-fallback")
        self.assertFalse(payload["generation_metadata"]["llm_reachable"])
        self.assertTrue(payload["generation_metadata"]["used_fallback"])
        self.assertEqual(mocked_generate.await_count, 0)

    def test_analyze_incident_falls_back_when_selected_candidate_has_invalid_dag(self) -> None:
        weak_candidate = backend_main.CausalAnalysisResponse(
            root_cause="auth_service:96% CPU; database:990/1000",
            intervention_simulation="scale_up_auth_service:5->20 replicas",
            recommended_action="rate_limit_frontend:1000 req/s",
            dag_nodes=[],
            dag_edges=[],
        )
        forced_verifier = {
            "accepted": True,
            "score": 0.9,
            "evidence_overlap": 4,
            "telemetry_overlap": 4,
            "generic_penalty": 0.0,
            "panic_scaling_penalty": 0.0,
            "reasons": ["accepted"],
        }
        with patch.object(
            backend_main,
            "_brain_ready_for_inference",
            new=AsyncMock(return_value=True),
        ), patch.object(
            backend_main,
            "_generate_candidate_analyses",
            new=AsyncMock(return_value=[weak_candidate]),
        ), patch.object(
            backend_main,
            "select_best_candidate",
            return_value=(weak_candidate.model_dump(), forced_verifier, 0),
        ), patch.object(
            backend_main,
            "_run_refutation_background",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post(
                "/api/analyze-incident",
                json={"incident_summary": "Auth retry storm with DB saturation."},
                headers={"x-tenant-id": "test-tenant"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generation_metadata"]["source"], "grounded-fallback")
        self.assertTrue(payload["generation_metadata"]["used_fallback"])
        self.assertGreaterEqual(len(payload["analysis"]["dag_nodes"]), 3)
        self.assertGreaterEqual(len(payload["analysis"]["dag_edges"]), 2)

    def test_analyze_incident_falls_back_when_live_generation_times_out(self) -> None:
        with patch.object(
            backend_main,
            "_brain_ready_for_inference",
            new=AsyncMock(return_value=True),
        ), patch.object(
            backend_main,
            "_generate_candidate_analyses",
            new=AsyncMock(side_effect=asyncio.TimeoutError()),
        ), patch.object(
            backend_main,
            "_run_refutation_background",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post(
                "/api/analyze-incident",
                json={"incident_summary": "Auth retries surge while DB reaches saturation."},
                headers={"x-tenant-id": "test-tenant"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generation_metadata"]["source"], "grounded-fallback")
        self.assertTrue(payload["generation_metadata"]["used_fallback"])
        self.assertFalse(payload["generation_metadata"]["llm_reachable"])
        self.assertGreaterEqual(len(payload["analysis"]["dag_nodes"]), 3)
        self.assertGreaterEqual(len(payload["analysis"]["dag_edges"]), 2)

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

    def test_authorize_intervention_loads_analysis_from_persistent_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_db_path = Path(temp_dir) / "analyst_feedback.db"
            feedback_log_path = Path(temp_dir) / "analyst_feedback.jsonl"
            with patch.object(
                backend_main,
                "FEEDBACK_DB_PATH",
                str(feedback_db_path),
            ), patch.object(
                backend_main,
                "FEEDBACK_LOG_PATH",
                str(feedback_log_path),
            ), patch.object(
                backend_main,
                "_brain_ready_for_inference",
                new=AsyncMock(return_value=False),
            ), patch.object(
                backend_main,
                "_run_refutation_background",
                new=AsyncMock(return_value=None),
            ):
                analyze_response = self.client.post(
                    "/api/analyze-incident",
                    json={"incident_summary": "Auth retries + DB saturation"},
                    headers={"x-tenant-id": "tenant-a"},
                )
                self.assertEqual(analyze_response.status_code, 200)
                analysis_id = analyze_response.json()["analysis_id"]

                backend_main._analysis_cache.clear()
                backend_main._analysis_tenants.clear()

                authorize_response = self.client.post(
                    "/api/interventions/authorize",
                    json={
                        "analysis_id": analysis_id,
                        "operator_id": "alice",
                        "tenant_id": "tenant-a",
                        "reason": "Incident commander approves safe intervention path.",
                    },
                    headers={"x-tenant-id": "tenant-a"},
                )

            self.assertEqual(authorize_response.status_code, 200)
            payload = authorize_response.json()
            self.assertEqual(payload["status"], "authorized")
            self.assertEqual(payload["analysis_id"], analysis_id)

    def test_api_key_enforcement_mode_requires_valid_key(self) -> None:
        with patch.object(backend_main, "REQUIRE_API_AUTH", True), patch.object(
            backend_main, "API_AUTH_TOKEN", "super-secret"
        ):
            without_key = self.client.get(
                "/api/mcp/tools",
                headers={"x-tenant-id": "tenant-a"},
            )
            with_key = self.client.get(
                "/api/mcp/tools",
                headers={"x-tenant-id": "tenant-a", "x-api-key": "super-secret"},
            )

        self.assertEqual(without_key.status_code, 401)
        self.assertEqual(with_key.status_code, 200)

    def test_api_key_enforcement_mode_returns_503_when_misconfigured(self) -> None:
        with patch.object(backend_main, "REQUIRE_API_AUTH", True), patch.object(
            backend_main, "API_AUTH_TOKEN", ""
        ):
            response = self.client.get(
                "/api/mcp/tools",
                headers={"x-tenant-id": "tenant-a"},
            )

        self.assertEqual(response.status_code, 503)

    def test_api_endpoints_require_tenant_header(self) -> None:
        response = self.client.get("/api/mcp/tools")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
