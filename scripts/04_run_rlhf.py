"""
SRE-Nidaan Pipeline — Script 04: RLHF with Pearl's Ladder
===========================================================
Aligns the model using RLHF with Pearl's Causal Hierarchy curriculum.
"""

import sys
import os
import json
import shutil
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.evaluator import SREEvaluationFramework
from src.training.rlhf_trainer import execute_rlhf_with_pearls_ladder
from src.training.reward_modeler import SRE7DRewardModel
from src.utils.model_utils import load_peft_checkpoint
import config


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _replace_dir(src: str, dst: str):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _build_eval_callback(
    *,
    reward_model,
    output_dir: str,
    strict_schema: bool,
    num_candidates: int,
):
    os.makedirs(output_dir, exist_ok=True)

    def _evaluate(model, tokenizer, iteration: int):
        report_path = os.path.join(output_dir, f"iter_{iteration:04d}.json")
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
        report["report_path"] = report_path
        return report

    return _evaluate


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 4: RLHF Training")
    print("=" * 60)

    # Load dataset
    dataset_path = os.environ.get("RLHF_DATASET_PATH", "data/sre_nidaan_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    if getattr(config, "FAST_LOCAL_TEST", False):
        dataset = dataset[:10]

    sft_adapter_dir = os.environ.get(
        "SFT_ADAPTER_DIR",
        config.SFT_TRAINING_ARGS.output_dir,
    )

    # Load policy model (SFT checkpoint)
    policy_model, tokenizer = load_peft_checkpoint(
        sft_adapter_dir,
        config.MODEL_ID,
        config.BNB_CONFIG,
        is_trainable=True,
    )

    # Load reward model
    reward_sft, _ = load_peft_checkpoint(
        sft_adapter_dir,
        config.MODEL_ID,
        config.BNB_CONFIG,
    )
    reward_model = SRE7DRewardModel(reward_sft, tokenizer)
    reward_head_path = os.environ.get(
        "REWARD_HEAD_PATH",
        "results/reward_model_head.pt",
    )
    reward_model.reward_head.load_state_dict(
        torch.load(reward_head_path, map_location=config.DEVICE)
    )
    reward_model.to(config.DEVICE)
    reward_model.eval()

    iterations = int(os.environ.get("RLHF_ITERATIONS_OVERRIDE", config.RLHF_ITERATIONS))
    output_dir = os.environ.get("RLHF_OUTPUT_DIR", "./results/final_rlhf_model")
    short_stage_iterations = min(
        iterations,
        int(
            os.environ.get(
                "RLHF_SHORT_STAGE_ITERATIONS",
                str(config.RLHF_SHORT_STAGE_ITERATIONS),
            )
        ),
    )
    eval_interval = int(
        os.environ.get(
            "RLHF_EVAL_INTERVAL",
            str(config.RLHF_EVAL_INTERVAL),
        )
    )
    schema_bonus_weight = float(
        os.environ.get(
            "RLHF_SCHEMA_BONUS_WEIGHT",
            str(config.RLHF_SCHEMA_BONUS_WEIGHT),
        )
    )
    reference_kl_coef = float(
        os.environ.get(
            "RLHF_REFERENCE_KL_COEF",
            str(config.RLHF_REFERENCE_KL_COEF),
        )
    )
    min_improvement = float(
        os.environ.get(
            "RLHF_MIN_IMPROVEMENT",
            str(config.RLHF_MIN_IMPROVEMENT),
        )
    )
    strict_schema = _env_flag("STRICT_SCHEMA_EVAL", True)
    num_candidates = int(os.environ.get("EVAL_NUM_CANDIDATES", "4"))
    eval_root = os.environ.get("RLHF_EVAL_ROOT", f"{output_dir}_evals")
    stage1_output_dir = f"{output_dir}_stage1_best"

    baseline_report_path = os.environ.get(
        "RLHF_BASELINE_REPORT_PATH",
        os.path.join(eval_root, "baseline.json"),
    )
    baseline_score_env = os.environ.get("RLHF_BASELINE_SCORE")
    if baseline_score_env is not None:
        baseline_score = float(baseline_score_env)
    else:
        os.makedirs(os.path.dirname(baseline_report_path) or ".", exist_ok=True)
        baseline_evaluator = SREEvaluationFramework(
            policy_model,
            tokenizer,
            config.DEVICE,
            reward_model=reward_model,
            strict_schema=strict_schema,
            num_candidates=num_candidates,
            report_path=baseline_report_path,
        )
        baseline_report = baseline_evaluator.conduct_evaluation()
        baseline_score = baseline_report["overall_score"]
        policy_model.train()
        print(
            f"📏 Baseline checkpoint score: {baseline_score:.3f} "
            f"({baseline_report['assessment']})"
        )

    stage1_eval_callback = _build_eval_callback(
        reward_model=reward_model,
        output_dir=os.path.join(eval_root, "stage1"),
        strict_schema=strict_schema,
        num_candidates=num_candidates,
    )

    policy_model, tokenizer, stage1_summary = execute_rlhf_with_pearls_ladder(
        dataset,
        policy_model,
        reward_model,
        tokenizer,
        short_stage_iterations,
        config.RLHF_LR,
        config.DEVICE,
        model_name=config.MODEL_ID,
        output_dir=stage1_output_dir,
        reference_model=reward_model.base_model,
        schema_bonus_weight=schema_bonus_weight,
        reference_kl_coef=reference_kl_coef,
        eval_interval=eval_interval,
        eval_callback=stage1_eval_callback,
        baseline_score=baseline_score,
        min_improvement=min_improvement,
        gate_after_iterations=short_stage_iterations,
        total_planned_iterations=short_stage_iterations,
    )

    overall_summary = {
        "baseline_score": baseline_score,
        "output_dir": output_dir,
        "stage1": stage1_summary,
    }

    stage1_best_score = stage1_summary.get("best_eval_score")
    if (
        stage1_best_score is None
        or stage1_best_score <= (baseline_score + min_improvement)
    ):
        print(
            "⏹ Short RLHF did not beat the baseline checkpoint. "
            "Restoring the original SFT adapter as the selected artifact."
        )
        _replace_dir(sft_adapter_dir, output_dir)
        overall_summary["selected_artifact"] = {
            "type": "baseline_sft",
            "adapter_dir": sft_adapter_dir,
            "score": baseline_score,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "safe_rlhf_summary.json"), "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"📝 Safe RLHF summary saved to {output_dir}/safe_rlhf_summary.json")
        print("\n--- RLHF Phase Complete ---\n")
        return

    remaining_iterations = max(iterations - short_stage_iterations, 0)
    if remaining_iterations > 0:
        _replace_dir(stage1_output_dir, output_dir)
        del policy_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        policy_model, tokenizer = load_peft_checkpoint(
            output_dir,
            config.MODEL_ID,
            config.BNB_CONFIG,
            is_trainable=True,
        )

        stage2_eval_callback = _build_eval_callback(
            reward_model=reward_model,
            output_dir=os.path.join(eval_root, "stage2"),
            strict_schema=strict_schema,
            num_candidates=num_candidates,
        )
        policy_model, tokenizer, stage2_summary = execute_rlhf_with_pearls_ladder(
            dataset,
            policy_model,
            reward_model,
            tokenizer,
            remaining_iterations,
            config.RLHF_LR,
            config.DEVICE,
            model_name=config.MODEL_ID,
            output_dir=output_dir,
            reference_model=reward_model.base_model,
            schema_bonus_weight=schema_bonus_weight,
            reference_kl_coef=reference_kl_coef,
            eval_interval=eval_interval,
            eval_callback=stage2_eval_callback,
            best_score_so_far=stage1_best_score,
            total_planned_iterations=remaining_iterations,
        )
        overall_summary["stage2"] = stage2_summary
        final_score = stage2_summary.get("best_eval_score", stage1_best_score)
    else:
        _replace_dir(stage1_output_dir, output_dir)
        final_score = stage1_best_score

    overall_summary["selected_artifact"] = {
        "type": "rlhf_best",
        "adapter_dir": output_dir,
        "score": final_score,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "safe_rlhf_summary.json"), "w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"📝 Safe RLHF summary saved to {output_dir}/safe_rlhf_summary.json")

    print("\n--- RLHF Phase Complete ---\n")


if __name__ == "__main__":
    main()
