import unittest

from src.utils.sre_schema import (
    REQUIRED_RESPONSE_TAGS,
    build_curated_continuation_subset,
    build_structured_training_response,
    coerce_structured_response,
)


class SRESchemaTests(unittest.TestCase):
    def test_coerce_structured_response_fills_all_required_tags(self):
        response = coerce_structured_response("[ROOT_CAUSE] DNS cache churn")
        for tag in REQUIRED_RESPONSE_TAGS:
            self.assertIn(tag, response)
        self.assertIn("requires_human_approval=true", response)

    def test_structured_training_response_adds_lexical_cues(self):
        example = {
            "domain": "dns",
            "pearl_level": 2,
            "root_cause": "DNS cache churn",
            "confounding_action": "Scaling replicas masks the cache invalidation pattern.",
            "counterfactual": "A longer TTL would have reduced the churn.",
            "correct_action": "Increase TTL and smooth invalidations.",
            "dag_nodes": [{"id": "dns", "label": "DNS Cache"}],
            "dag_edges": [{"id": "e1", "source": "dns", "target": "latency"}],
        }
        response = build_structured_training_response(example, lexical_cues=True)
        self.assertIn("do(", response)
        self.assertIn("Counterfactual:", response)
        self.assertIn("human approval", response.lower())

    def test_curated_continuation_subset_balances_levels(self):
        dataset = []
        for level in (1, 2, 3):
            for score in (0.9, 0.8, 0.7):
                dataset.append(
                    {
                        "pearl_level": level,
                        "quality_score": score,
                    }
                )
        curated = build_curated_continuation_subset(dataset, max_examples=6)
        counts = {1: 0, 2: 0, 3: 0}
        for example in curated:
            counts[int(example["pearl_level"])] += 1
        self.assertEqual(len(curated), 6)
        self.assertTrue(all(count >= 1 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
