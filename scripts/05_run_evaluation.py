"""
SRE-Nidaan Pipeline — Script 05: Final Evaluation
===================================================
Evaluates the RLHF-trained model across Pearl's Causal Hierarchy
with SRE-specific test cases.
"""

import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.evaluator import SREEvaluationFramework
import config


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 5: Final Model Evaluation")
    print("=" * 60)

    # Load final RLHF model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    final_model = PeftModel.from_pretrained(
        base_model, "./results/final_rlhf_model"
    )
    final_tokenizer = AutoTokenizer.from_pretrained(
        "./results/final_rlhf_model"
    )

    os.makedirs("results", exist_ok=True)

    # Run evaluation
    evaluator = SREEvaluationFramework(
        final_model, final_tokenizer, config.DEVICE
    )
    evaluator.conduct_evaluation()

    print("\n--- Evaluation Complete ---\n")


if __name__ == "__main__":
    main()
