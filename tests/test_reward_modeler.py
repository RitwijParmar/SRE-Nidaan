import unittest

from src.training.reward_modeler import create_preference_pairs


class RewardModelerTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "domain": "dns",
            "pearl_level": 2,
            "premise": "DNS resolution latency spiked to 500ms. Cache hit ratio fell from 95% to 25%.",
            "root_cause": "A cache invalidation storm forced recursive lookups for most queries.",
            "confounding_action": "do(Scale CoreDNS) hides the invalidation storm without restoring cache locality.",
            "counterfactual": "Counterfactual: a staggered invalidation rollout would have reduced the latency spike.",
            "correct_action": "Throttle invalidations, restore TTLs, and verify cache warmup before scaling.",
            "dag_nodes": [{"id": "cache", "label": "DNS Cache"}],
            "dag_edges": [{"id": "e1", "source": "cache", "target": "latency"}],
            "quality_score": 0.95,
            "reasoning": "[CAUSE] Cache invalidation storm\n[CONCLUSION] Restore TTLs",
        }

    def test_prompt_matched_preferences_share_same_incident_prompt(self):
        pairs = create_preference_pairs(
            [self.example],
            tokenizer=None,
            model_name="",
            mode="prompt_matched",
            negative_variants=3,
        )
        self.assertGreaterEqual(len(pairs), 2)
        prompts = {pair["prompt"] for pair in pairs}
        self.assertEqual(len(prompts), 1)
        for pair in pairs:
            self.assertIn(self.example["premise"], pair["prompt"])
            self.assertIn(pair["prompt"], pair["preferred"])
            self.assertIn(pair["prompt"], pair["rejected"])
            self.assertIn("[SAFETY_CHECK]", pair["preferred"])
            self.assertNotEqual(pair["preferred"], pair["rejected"])

    def test_sorted_quality_mode_still_supported(self):
        lower_quality = dict(self.example)
        lower_quality["quality_score"] = 0.3
        lower_quality["premise"] = "CoreDNS CPU rose to 95% while cache hit ratio stayed stable."
        pairs = create_preference_pairs(
            [self.example, lower_quality],
            tokenizer=None,
            model_name="",
            mode="sorted_quality",
        )
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
