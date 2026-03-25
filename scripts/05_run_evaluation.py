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

    # Local test memory limits cause RLHF to fail, so evaluate the SFT model instead
    model_dir = config.SFT_TRAINING_ARGS.output_dir if getattr(config, "FAST_LOCAL_TEST", False) else "./results/final_rlhf_model"
    
    # Load model
    final_tokenizer = AutoTokenizer.from_pretrained(
        model_dir if os.path.exists(model_dir) else config.SFT_TRAINING_ARGS.output_dir
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(final_tokenizer))
    final_model = PeftModel.from_pretrained(
        base_model, model_dir
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
