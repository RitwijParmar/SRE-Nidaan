"""
SRE-Nidaan Pipeline — Script 05: Final Evaluation
===================================================
Evaluates the RLHF-trained model across Pearl's Causal Hierarchy
with SRE-specific test cases.
"""

import sys
import os

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.evaluator import SREEvaluationFramework
from src.training.reward_modeler import SRE7DRewardModel
from src.utils.model_utils import load_peft_checkpoint
import config


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 5: Final Model Evaluation")
    print("=" * 60)

    # Local test memory limits cause RLHF to fail, so evaluate the SFT model instead.
    default_model_dir = (
        config.SFT_TRAINING_ARGS.output_dir
        if getattr(config, "FAST_LOCAL_TEST", False)
        else "./results/final_rlhf_model"
    )
    model_dir = os.environ.get("MODEL_DIR", default_model_dir)
    if not os.path.exists(model_dir):
        model_dir = config.SFT_TRAINING_ARGS.output_dir

    final_model, final_tokenizer = load_peft_checkpoint(
        model_dir,
        config.MODEL_ID,
        config.BNB_CONFIG,
    )

    os.makedirs("results", exist_ok=True)
    report_path = os.environ.get(
        "EVAL_REPORT_PATH",
        "results/final_evaluation_report.json",
    )
    strict_schema = _env_flag("STRICT_SCHEMA_EVAL", False)
    num_candidates = int(os.environ.get("EVAL_NUM_CANDIDATES", "1"))

    reward_model = None
    reward_head_path = os.environ.get("REWARD_HEAD_PATH")
    if reward_head_path and os.path.exists(reward_head_path):
        reward_model = SRE7DRewardModel(final_model, final_tokenizer)
        reward_model.reward_head.load_state_dict(
            torch.load(reward_head_path, map_location=config.DEVICE)
        )
        reward_model.to(config.DEVICE)

    # Run evaluation
    evaluator = SREEvaluationFramework(
        final_model,
        final_tokenizer,
        config.DEVICE,
        reward_model=reward_model,
        strict_schema=strict_schema,
        num_candidates=num_candidates,
        report_path=report_path,
    )
    evaluator.conduct_evaluation()

    print("\n--- Evaluation Complete ---\n")


if __name__ == "__main__":
    main()
