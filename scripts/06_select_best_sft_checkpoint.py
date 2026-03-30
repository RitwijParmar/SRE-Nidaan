"""
SRE-Nidaan Pipeline — Script 06: Select Best Saved SFT Checkpoint
=================================================================
Evaluate every saved SFT adapter checkpoint and choose the strongest one.
"""

import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.evaluator import SREEvaluationFramework
from src.training.reward_modeler import SRE7DRewardModel
from src.utils.model_utils import (
    discover_adapter_checkpoints,
    load_peft_checkpoint,
)
import config


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _checkpoint_label(adapter_dir: str, adapter_root: str) -> str:
    path = Path(adapter_dir)
    root = Path(adapter_root)
    if path.resolve() == root.resolve():
        return "final-adapter"
    return path.name


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 6: SFT Checkpoint Sweep")
    print("=" * 60)

    adapter_root = os.environ.get(
        "SFT_ADAPTER_ROOT",
        config.SFT_TRAINING_ARGS.output_dir,
    )
    reward_head_path = os.environ.get("REWARD_HEAD_PATH")
    strict_schema = _env_flag("STRICT_SCHEMA_EVAL", True)
    num_candidates = int(os.environ.get("EVAL_NUM_CANDIDATES", "4"))
    summary_path = os.environ.get(
        "SWEEP_SUMMARY_PATH",
        "results/sft_checkpoint_sweep.json",
    )
    report_dir = os.environ.get(
        "SWEEP_REPORT_DIR",
        "results/checkpoint_sweep",
    )

    checkpoints = discover_adapter_checkpoints(adapter_root)
    if not checkpoints:
        raise FileNotFoundError(
            f"No adapter checkpoints found under {adapter_root}"
        )

    os.makedirs(report_dir, exist_ok=True)
    sweep_results = []

    for adapter_dir in checkpoints:
        label = _checkpoint_label(adapter_dir, adapter_root)
        print(f"\n🔎 Evaluating {label} ({adapter_dir})")

        model, tokenizer = load_peft_checkpoint(
            adapter_dir,
            config.MODEL_ID,
            config.BNB_CONFIG,
        )

        reward_model = None
        if reward_head_path and os.path.exists(reward_head_path):
            reward_model = SRE7DRewardModel(model, tokenizer)
            reward_model.reward_head.load_state_dict(
                torch.load(reward_head_path, map_location=config.DEVICE)
            )
            reward_model.to(config.DEVICE)

        report_path = os.path.join(report_dir, f"{label}.json")
        evaluator = SREEvaluationFramework(
            model,
            tokenizer,
            config.DEVICE,
            reward_model=reward_model,
            strict_schema=strict_schema,
            num_candidates=num_candidates,
            report_path=report_path,
        )
        report = evaluator.conduct_evaluation()
        sweep_results.append(
            {
                "label": label,
                "adapter_dir": adapter_dir,
                "report_path": report_path,
                "overall_score": report["overall_score"],
                "assessment": report["assessment"],
                "safety_compliance_rate": report["safety_compliance_rate"],
            }
        )

        del evaluator
        del reward_model
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_result = max(sweep_results, key=lambda result: result["overall_score"])
    summary = {
        "best": best_result,
        "best_checkpoint": best_result,
        "all_results": sweep_results,
    }

    os.makedirs(os.path.dirname(summary_path) or "results", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n🏁 Best checkpoint selected:")
    print(f"   Label:        {best_result['label']}")
    print(f"   Adapter Dir:  {best_result['adapter_dir']}")
    print(f"   Score:        {best_result['overall_score']:.3f}")
    print(f"   Assessment:   {best_result['assessment']}")
    print(f"\n💾 Sweep summary saved to {summary_path}")


if __name__ == "__main__":
    main()
