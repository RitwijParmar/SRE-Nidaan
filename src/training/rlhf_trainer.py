"""
SRE-Nidaan: Pearl's Ladder RLHF Trainer
=========================================
Phase 3 of the NEXUS-CAUSAL SRE training pipeline.
Implements Reinforcement Learning from Human Feedback with a curriculum
learning strategy aligned to Pearl's Causal Hierarchy.

Mirrors NEXUS-CAUSAL v3.1 src/training/rlhf_trainer.py with SRE-specific
prompt engineering and safety-aware reward shaping.
"""

import json
import os
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from typing import Callable, Dict, List, Optional

from src.utils.model_utils import build_chat_prompt
from src.utils.sre_schema import coerce_structured_response, heuristic_response_bonus


# ─────────────────────────────────────────────────────────────────────────────
# Pearl's Ladder Curriculum for SRE
# ─────────────────────────────────────────────────────────────────────────────
# Level 1 (Association):    "What correlates with this outage?"
# Level 2 (Intervention):   "What happens if we do(Scale Up Auth_Service)?"
# Level 3 (Counterfactual): "Would this outage have occurred if we had
#                            implemented circuit breakers?"
# ─────────────────────────────────────────────────────────────────────────────

PEARL_LEVEL_PROMPTS = {
    1: (
        "You are an SRE agent analyzing telemetry correlations. "
        "Identify which metrics are correlated with the incident below. "
        "Do NOT infer causation — only report observational associations.\n\n"
        "Incident: {premise}"
    ),
    2: (
        "You are an SRE agent performing causal intervention analysis. "
        "Given the incident below, use Pearl's do-calculus to determine: "
        "(1) the structural root cause, (2) why the naive intervention "
        "do(Scale_Up) is a confounding error, and (3) the correct intervention.\n\n"
        "Incident: {premise}"
    ),
    3: (
        "You are an SRE agent performing counterfactual reasoning. "
        "Given the incident below, answer: Would this failure have occurred "
        "if the preventive measure had been in place? Construct a causal DAG "
        "showing both the actual and counterfactual worlds.\n\n"
        "Incident: {premise}"
    ),
}

SCHEMA_SUFFIX = (
    "\n\nReturn the answer using [ROOT_CAUSE], [INTERVENTION], [COUNTERFACTUAL], "
    "[DAG], [REMEDIATION], and [SAFETY_CHECK]. The intervention section must "
    "mention do(...), the counterfactual section must say counterfactual, the "
    "DAG section must describe the graph with node and edge wording, and the "
    "safety section must include requires_human_approval=true plus human approval "
    "or manual review."
)


def _build_curriculum_prompts(
    dataset: List[Dict],
    tokenizer=None,
    model_name: str = "",
) -> Dict[int, List[str]]:
    """
    Organize dataset prompts by Pearl's Hierarchy level for curriculum learning.
    """
    prompts = {1: [], 2: [], 3: []}

    for ex in dataset:
        level = ex.get("pearl_level", 1)
        if level not in prompts:
            continue

        premise = ex.get("premise", "")
        prompt_template = PEARL_LEVEL_PROMPTS[level]
        formatted = build_chat_prompt(
            [
                {
                    "role": "user",
                    "content": prompt_template.format(premise=premise) + SCHEMA_SUFFIX,
                }
            ],
            model_name=model_name,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )
        prompts[level].append(formatted)

    for level, p_list in prompts.items():
        print(f"   Pearl Level {level}: {len(p_list)} prompts")

    return prompts


def _compute_response_log_probs(
    policy_model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    """
    Compute mean token log-probabilities only for generated continuation tokens.
    """
    outputs = policy_model(
        input_ids=sequences,
        attention_mask=attention_mask,
    )
    logits = outputs.logits[:, :-1, :]
    labels = sequences[:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    response_mask = attention_mask[:, 1:].clone()
    for row_idx in range(response_mask.shape[0]):
        response_start = max(int(prompt_width) - 1, 0)
        response_mask[row_idx, :response_start] = 0

    token_counts = response_mask.sum(dim=-1).clamp_min(1)
    return (token_log_probs * response_mask).sum(dim=-1) / token_counts


def _compute_response_kl(
    policy_model,
    reference_model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    """Approximate KL(policy || reference) over generated response tokens."""
    policy_outputs = policy_model(
        input_ids=sequences,
        attention_mask=attention_mask,
    )
    policy_logits = policy_outputs.logits[:, :-1, :]
    labels = sequences[:, 1:]
    policy_token_log_probs = F.log_softmax(policy_logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        reference_outputs = reference_model(
            input_ids=sequences,
            attention_mask=attention_mask,
        )
        reference_logits = reference_outputs.logits[:, :-1, :]
        reference_token_log_probs = F.log_softmax(
            reference_logits,
            dim=-1,
        ).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    response_mask = attention_mask[:, 1:].clone()
    for row_idx in range(response_mask.shape[0]):
        response_start = max(int(prompt_width) - 1, 0)
        response_mask[row_idx, :response_start] = 0

    token_counts = response_mask.sum(dim=-1).clamp_min(1)
    kl_terms = (policy_token_log_probs - reference_token_log_probs) * response_mask
    return kl_terms.sum(dim=-1) / token_counts


def _get_curriculum_level(iteration: int, total_iterations: int) -> int:
    """
    Determine the Pearl's Hierarchy level for the current iteration.
    Progressive curriculum: L1 → L2 → L3.
    """
    progress = iteration / total_iterations
    if progress < 0.33:
        return 1  # Association
    elif progress < 0.66:
        return 2  # Intervention
    else:
        return 3  # Counterfactual


def _save_model_snapshot(policy_model, tokenizer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def execute_rlhf_with_pearls_ladder(
    dataset: List[Dict],
    policy_model,
    reward_model,
    tokenizer,
    iterations: int,
    lr: float,
    device: str,
    model_name: str = "",
    output_dir: str = "./results/final_rlhf_model",
    reference_model=None,
    schema_bonus_weight: float = 0.0,
    reference_kl_coef: float = 0.02,
    eval_interval: int = 0,
    eval_callback: Optional[Callable[[object, object, int], Dict]] = None,
    best_score_so_far: Optional[float] = None,
    baseline_score: Optional[float] = None,
    min_improvement: float = 0.0,
    gate_after_iterations: int = 0,
    total_planned_iterations: Optional[int] = None,
):
    """
    Execute RLHF training with Pearl's Ladder curriculum.

    The model progresses through three stages:
    1. Iterations 1–5:  Association (L1) — learn to identify correlations
    2. Iterations 6–10: Intervention (L2) — learn do-calculus reasoning
    3. Iterations 11–15: Counterfactual (L3) — learn "what if" reasoning

    Args:
        dataset: Full SRE incident dataset
        policy_model: SFT model to refine
        reward_model: Trained 7D reward model
        tokenizer: Tokenizer from SFT phase
        iterations: Number of RLHF iterations
        lr: Learning rate
        device: Compute device
    """
    print("⚡ Starting RLHF with Pearl's Ladder Curriculum...")
    print(f"   Total iterations: {iterations}")
    print(f"   Learning rate: {lr}")
    print(f"   Curriculum: L1(Association) → L2(Intervention) → L3(Counterfactual)")
    print(f"   Schema bonus weight: {schema_bonus_weight:.3f}")
    if reference_model is not None:
        print(f"   Reference KL coefficient: {reference_kl_coef:.4f}")
    if baseline_score is not None:
        print(f"   Baseline score to beat: {baseline_score:.3f}")

    reward_model.eval()
    if reference_model is not None:
        reference_model.eval()

    # Build curriculum prompts
    prompts = _build_curriculum_prompts(
        dataset,
        tokenizer=tokenizer,
        model_name=model_name or getattr(tokenizer, "name_or_path", ""),
    )

    optimizer = AdamW(policy_model.parameters(), lr=lr)
    policy_model.train()

    level_losses = {1: [], 2: [], 3: []}
    evaluation_history = []
    best_eval_score = best_score_so_far
    best_iteration = None
    stop_reason = None
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    planned_iterations = max(total_planned_iterations or iterations, 1)
    batch_size_limit = max(1, int(os.environ.get("RLHF_BATCH_SIZE", "2")))
    prompt_max_length = max(64, int(os.environ.get("RLHF_PROMPT_MAX_LENGTH", "192")))
    max_new_tokens = max(32, int(os.environ.get("RLHF_MAX_NEW_TOKENS", "96")))

    print(f"   RLHF batch size limit: {batch_size_limit}")
    print(f"   RLHF prompt max length: {prompt_max_length}")
    print(f"   RLHF max new tokens: {max_new_tokens}")

    try:
        for i in tqdm(range(iterations), desc="RLHF Iterations"):
            # Determine current Pearl's level
            level = _get_curriculum_level(i, planned_iterations)
            level_prompts = prompts.get(level, prompts[1])

            if not level_prompts:
                print(f"   ⚠ No prompts for level {level}, falling back to L1")
                level_prompts = prompts[1]

            # Sample a mini-batch of prompts
            batch_size = min(batch_size_limit, len(level_prompts))
            batch_prompts = random.sample(level_prompts, batch_size)

            # Tokenize prompts with left padding for decoder-only generation.
            batch = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                max_length=prompt_max_length,
                truncation=True,
            ).to(device)
            prompt_width = batch["input_ids"].shape[1]

            # Generate responses from policy model.
            with torch.no_grad():
                gen_ids = policy_model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_attention_mask = gen_ids.ne(tokenizer.pad_token_id).long()
                rewards = reward_model(
                    gen_ids,
                    attention_mask=gen_attention_mask,
                ).mean(dim=-1)

            response_log_probs = _compute_response_log_probs(
                policy_model,
                sequences=gen_ids,
                attention_mask=gen_attention_mask,
                prompt_width=prompt_width,
            )

            heuristic_bonus = []
            for gen_id in gen_ids:
                continuation = gen_id[int(prompt_width):]
                decoded = tokenizer.decode(continuation, skip_special_tokens=True)
                normalized = coerce_structured_response(decoded)
                heuristic_bonus.append(
                    schema_bonus_weight * heuristic_response_bonus(normalized)
                )

            total_rewards = rewards
            if schema_bonus_weight:
                total_rewards = total_rewards + torch.tensor(
                    heuristic_bonus,
                    device=rewards.device,
                    dtype=rewards.dtype,
                )

            if total_rewards.shape[0] > 1:
                advantages = total_rewards - total_rewards.mean()
            else:
                advantages = total_rewards

            policy_loss = -(response_log_probs * advantages.detach()).mean()
            kl_penalty = torch.zeros((), device=response_log_probs.device)
            if reference_model is not None and reference_kl_coef > 0:
                kl_penalty = _compute_response_kl(
                    policy_model,
                    reference_model,
                    sequences=gen_ids,
                    attention_mask=gen_attention_mask,
                    prompt_width=prompt_width,
                ).mean()

            # REINFORCE-style loss on generated response tokens with a
            # reference-model anchor to keep RLHF close to the best SFT policy.
            loss = policy_loss + reference_kl_coef * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            level_losses[level].append(loss.item())

            if (i + 1) % 5 == 0:
                print(
                    f"   Iter {i+1}/{iterations} | Level: L{level} | "
                    f"Loss: {loss.item():.4f} | Reward: {total_rewards.mean().item():.4f} "
                    f"| KL: {kl_penalty.item():.4f}"
                )

            should_evaluate = (
                eval_callback is not None
                and (
                    (eval_interval > 0 and (i + 1) % eval_interval == 0)
                    or (i + 1) == iterations
                    or (gate_after_iterations > 0 and (i + 1) == gate_after_iterations)
                )
            )
            if should_evaluate:
                report = eval_callback(policy_model, tokenizer, i + 1) or {}
                policy_model.train()
                score = report.get("overall_score")
                evaluation_event = {
                    "iteration": i + 1,
                    "overall_score": score,
                    "assessment": report.get("assessment"),
                    "report_path": report.get("report_path"),
                }
                evaluation_history.append(evaluation_event)
                if score is not None:
                    print(
                        f"   Eval @ iter {i+1}: score={score:.3f} "
                        f"assessment={report.get('assessment', 'unknown')}"
                    )
                    if best_eval_score is None or score > best_eval_score:
                        best_eval_score = score
                        best_iteration = i + 1
                        _save_model_snapshot(policy_model, tokenizer, output_dir)
                        print(
                            f"   ↳ New best RLHF checkpoint at iter {i+1}; saved to {output_dir}"
                        )

                if (
                    gate_after_iterations > 0
                    and (i + 1) >= gate_after_iterations
                    and baseline_score is not None
                    and (
                        best_eval_score is None
                        or best_eval_score <= (baseline_score + min_improvement)
                    )
                ):
                    stop_reason = (
                        "short_stage_did_not_beat_baseline"
                    )
                    print(
                        "   ⏹ Short RLHF gate failed to beat the baseline; "
                        "stopping before longer continuation."
                    )
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    # Print level-wise summary
    print("\n📊 Training Summary:")
    for level in [1, 2, 3]:
        losses = level_losses[level]
        if losses:
            avg = sum(losses) / len(losses)
            labels = ["Association", "Intervention", "Counterfactual"]
            print(f"   L{level} ({labels[level-1]}): {len(losses)} iters, avg_loss={avg:.4f}")

    if best_eval_score is None or not os.path.exists(output_dir):
        _save_model_snapshot(policy_model, tokenizer, output_dir)
        print(f"💾 Final RLHF model saved to {output_dir}")
    else:
        print(
            f"💾 Best RLHF checkpoint retained from iter {best_iteration or 'existing'} "
            f"at {output_dir}"
        )

    summary = {
        "iterations": iterations,
        "planned_iterations": planned_iterations,
        "schema_bonus_weight": schema_bonus_weight,
        "reference_kl_coef": reference_kl_coef,
        "baseline_score": baseline_score,
        "min_improvement": min_improvement,
        "gate_after_iterations": gate_after_iterations,
        "best_eval_score": best_eval_score,
        "best_iteration": best_iteration,
        "stop_reason": stop_reason,
        "evaluation_history": evaluation_history,
    }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "rlhf_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 RLHF training summary saved to {summary_path}")

    return policy_model, tokenizer, summary
