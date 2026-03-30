"""
SRE-Nidaan: Evaluation Framework
==================================
Evaluates the trained model across all three levels of Pearl's Causal Hierarchy
with SRE-specific test cases spanning 12 infrastructure domains.

Mirrors NEXUS-CAUSAL v3.1 src/evaluation/evaluator.py with comprehensive
SRE incident test suites and automated scoring.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List
from tqdm import tqdm

from src.utils.model_utils import build_chat_prompt
from src.utils.sre_schema import (
    coerce_structured_response,
    generate_and_rerank_structured_response,
)


class SREEvaluationFramework:
    """
    Comprehensive evaluation framework for SRE causal reasoning models.

    Tests across:
      - Pearl's Hierarchy L1 (Association), L2 (Intervention), L3 (Counterfactual)
      - 12 SRE infrastructure domains
      - Structural DAG accuracy, root cause precision, safety compliance
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        reward_model=None,
        strict_schema: bool = False,
        num_candidates: int = 1,
        report_path: str = "results/final_evaluation_report.json",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reward_model = reward_model
        self.strict_schema = strict_schema
        self.num_candidates = max(1, int(num_candidates))
        self.report_path = report_path
        self.model_name = getattr(tokenizer, "name_or_path", "")
        self.model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

    def create_test_suite(self) -> Dict[str, List[Dict]]:
        """
        Create a comprehensive SRE test suite covering all Pearl's levels
        and multiple infrastructure domains.
        """
        return {
            "L1_Association": [
                {
                    "id": "assoc_k8s_mem",
                    "domain": "kubernetes",
                    "premise": (
                        "A Kubernetes cluster shows high memory usage at 92% "
                        "and pod restart count increased 5x. CPU remains normal at 45%."
                    ),
                    "keywords": [
                        "memory", "correlation", "pod", "restart",
                        "oom", "resource", "utilization",
                    ],
                },
                {
                    "id": "assoc_net_latency",
                    "domain": "networking",
                    "premise": (
                        "Network latency between services A and B increased "
                        "from 2ms to 200ms. Packet loss is 0%. Bandwidth "
                        "utilization is 30% of capacity."
                    ),
                    "keywords": [
                        "latency", "correlation", "network", "conntrack",
                        "connection", "tcp", "throughput",
                    ],
                },
                {
                    "id": "assoc_monitoring_cardinality",
                    "domain": "monitoring",
                    "premise": (
                        "Prometheus memory usage doubled in 24 hours. "
                        "Scrape duration now exceeds the scrape interval. "
                        "Active time series count: 10M."
                    ),
                    "keywords": [
                        "cardinality", "time series", "label",
                        "memory", "scrape", "prometheus",
                    ],
                },
                {
                    "id": "assoc_dns_spike",
                    "domain": "dns",
                    "premise": (
                        "DNS resolution latency spiked to 500ms. CoreDNS "
                        "cache hit ratio dropped from 95% to 25%. No config "
                        "changes detected."
                    ),
                    "keywords": [
                        "dns", "cache", "resolution", "ndots",
                        "query", "coredns", "latency",
                    ],
                },
            ],
            "L2_Intervention": [
                {
                    "id": "interv_db_conn",
                    "domain": "database",
                    "premise": (
                        "PostgreSQL connection pool at 990/1000. Auth service "
                        "CPU at 96% with retry storms. Naive plan: scale up "
                        "auth service replicas."
                    ),
                    "keywords": [
                        "connection pool", "root cause", "retry",
                        "do(", "intervention", "circuit breaker",
                        "rate limit", "confound",
                    ],
                },
                {
                    "id": "interv_k8s_hpa",
                    "domain": "kubernetes",
                    "premise": (
                        "HPA scaled pods from 5 to 20 but latency worsened. "
                        "Node memory at 95%. Naive plan: increase HPA max replicas."
                    ),
                    "keywords": [
                        "memory leak", "connection pool", "do(",
                        "contention", "root cause", "bound",
                        "eviction", "intervention",
                    ],
                },
                {
                    "id": "interv_lb_health",
                    "domain": "load_balancer",
                    "premise": (
                        "ALB shows 60% unhealthy targets. Backend response "
                        "time 10s. Surge queue length 500. Naive plan: add "
                        "more backend instances."
                    ),
                    "keywords": [
                        "downstream", "thread", "health check",
                        "cascade", "circuit breaker", "root cause",
                        "do(", "intervention",
                    ],
                },
                {
                    "id": "interv_cache_eviction",
                    "domain": "cache",
                    "premise": (
                        "Redis at 96% memory with 2000 evictions/sec. "
                        "Cache hit ratio dropped from 98% to 30%. "
                        "Naive plan: double Redis memory."
                    ),
                    "keywords": [
                        "ttl", "eviction", "root cause", "no ttl",
                        "monotonic", "do(", "intervention",
                        "preference", "session",
                    ],
                },
            ],
            "L3_Counterfactual": [
                {
                    "id": "counter_auth_jwks",
                    "domain": "authentication",
                    "premise": (
                        "JWKS key rotation caused a thundering herd of "
                        "token validation requests. 60% of authentications "
                        "timed out. What if stale-while-revalidate caching "
                        "had been implemented?"
                    ),
                    "keywords": [
                        "counterfactual", "would not", "had",
                        "stale-while-revalidate", "thundering herd",
                        "background refresh", "cache",
                    ],
                },
                {
                    "id": "counter_cicd_cache",
                    "domain": "ci_cd",
                    "premise": (
                        "A dependency bump invalidated shared build cache "
                        "for all 50 concurrent pipelines. Build time 6x. "
                        "What if the cache key used lock file hash?"
                    ),
                    "keywords": [
                        "counterfactual", "would not", "had",
                        "cache key", "lock file", "digest",
                        "invalidation",
                    ],
                },
                {
                    "id": "counter_mq_rebalance",
                    "domain": "message_queue",
                    "premise": (
                        "Kafka consumer group experiences constant rebalances "
                        "causing 1M message lag. What if max.poll.records "
                        "had been reduced and poll interval increased?"
                    ),
                    "keywords": [
                        "counterfactual", "would not", "had",
                        "poll", "rebalance", "cooperative",
                        "batch", "consumer",
                    ],
                },
                {
                    "id": "counter_storage_burst",
                    "domain": "storage",
                    "premise": (
                        "EBS burst credits depleted during compaction job. "
                        "I/O latency jumped 100x. What if the volume had "
                        "been provisioned as io2 with guaranteed IOPS?"
                    ),
                    "keywords": [
                        "counterfactual", "would not", "had",
                        "burst", "provisioned", "io2",
                        "iops", "compaction",
                    ],
                },
            ],
        }

    def evaluate_case(self, test_case: Dict) -> Dict:
        """
        Evaluate a single test case and return detailed scores.
        """
        premise = test_case["premise"]
        if self.strict_schema or self.num_candidates > 1 or self.reward_model is not None:
            generated = generate_and_rerank_structured_response(
                self.model,
                self.tokenizer,
                premise=premise,
                model_name=self.model_name,
                device=self.device,
                reward_model=self.reward_model,
                num_candidates=self.num_candidates,
                max_new_tokens=256,
            )
            raw_response = generated["best_response"]
        else:
            prompt = build_chat_prompt(
                [
                    {
                        "role": "user",
                        "content": (
                            "You are an SRE causal reasoning agent. "
                            "Analyze the following incident using Pearl's Causal Hierarchy. "
                            "Identify root cause, explain confounding interventions, "
                            "and recommend safe remediation.\n\n"
                            f"Incident: {premise}"
                        ),
                    }
                ],
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                add_generation_prompt=True,
            )

            original_padding_side = getattr(self.tokenizer, "padding_side", "right")
            self.tokenizer.padding_side = "left"
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            finally:
                self.tokenizer.padding_side = original_padding_side

            prompt_len = inputs["input_ids"].shape[-1]
            raw_response = self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

        normalized_response = coerce_structured_response(raw_response)
        response = normalized_response.lower()

        # Keyword coverage scoring
        keywords = test_case["keywords"]
        hits = [kw for kw in keywords if kw.lower() in response]
        coverage = len(hits) / len(keywords) if keywords else 0.0

        # Safety compliance check
        safety_compliant = any(
            term in response
            for term in [
                "human", "approval", "safety", "review",
                "authorize", "manual", "verify",
            ]
        )

        # DAG structure check
        has_dag = any(
            term in response
            for term in ["dag", "graph", "node", "edge", "→", "->"]
        )

        return {
            "id": test_case["id"],
            "domain": test_case.get("domain", "unknown"),
            "response": normalized_response,
            "keyword_coverage": round(coverage, 3),
            "keywords_hit": hits,
            "keywords_missed": [k for k in keywords if k not in hits],
            "safety_compliant": safety_compliant,
            "has_dag_structure": has_dag,
            "composite_score": round(
                coverage * 0.6
                + (0.2 if safety_compliant else 0.0)
                + (0.2 if has_dag else 0.0),
                3,
            ),
        }

    def conduct_evaluation(self) -> Dict:
        """
        Run the full evaluation suite across all Pearl's levels.
        """
        print("\n🔬 Conducting SRE Causal Reasoning Evaluation...")
        test_suite = self.create_test_suite()

        all_results = []
        category_scores = {}

        for category, cases in test_suite.items():
            print(f"\n📊 Evaluating {category} ({len(cases)} cases)...")
            cat_scores = []

            for case in tqdm(cases, desc=f"  {category}"):
                result = self.evaluate_case(case)
                all_results.append(result)
                cat_scores.append(result["composite_score"])
                print(
                    f"   {result['id']}: "
                    f"coverage={result['keyword_coverage']:.3f} "
                    f"safety={'✓' if result['safety_compliant'] else '✗'} "
                    f"dag={'✓' if result['has_dag_structure'] else '✗'} "
                    f"→ {result['composite_score']:.3f}"
                )

            category_scores[category] = round(np.mean(cat_scores), 3)

        # Overall assessment
        overall_score = round(np.mean([r["composite_score"] for r in all_results]), 3)

        if overall_score > 0.75:
            assessment = "STATE-OF-THE-ART"
        elif overall_score > 0.55:
            assessment = "PRODUCTION-READY"
        elif overall_score > 0.35:
            assessment = "PROMISING"
        else:
            assessment = "NEEDS-IMPROVEMENT"

        # Safety compliance rate
        safety_rate = round(
            sum(1 for r in all_results if r["safety_compliant"]) / len(all_results),
            3,
        )

        print(f"\n{'='*60}")
        print(f"🏆 EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"   Overall Score:    {overall_score:.3f}")
        print(f"   Assessment:       {assessment}")
        print(f"   Safety Rate:      {safety_rate:.1%}")
        for cat, score in category_scores.items():
            print(f"   {cat}: {score:.3f}")

        # Save results
        evaluation_report = {
            "overall_score": overall_score,
            "assessment": assessment,
            "safety_compliance_rate": safety_rate,
            "category_scores": category_scores,
            "detailed_results": all_results,
        }

        os.makedirs(os.path.dirname(self.report_path) or "results", exist_ok=True)
        with open(self.report_path, "w") as f:
            json.dump(evaluation_report, f, indent=2)

        print(f"\n💾 Evaluation report saved to {self.report_path}")
        return evaluation_report
