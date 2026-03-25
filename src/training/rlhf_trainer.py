"""
SRE-Nidaan: Pearl's Ladder RLHF Trainer
=========================================
Phase 3 of the NEXUS-CAUSAL SRE training pipeline.
Implements Reinforcement Learning from Human Feedback with a curriculum
learning strategy aligned to Pearl's Causal Hierarchy.

Mirrors NEXUS-CAUSAL v3.1 src/training/rlhf_trainer.py with SRE-specific
prompt engineering and safety-aware reward shaping.
"""

import torch
import torch.nn.functional as F
import random
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict


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


def _build_curriculum_prompts(dataset: List[Dict]) -> Dict[int, List[str]]:
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
        formatted = f"<s>[INST] {prompt_template.format(premise=premise)} [/INST]"
        prompts[level].append(formatted)

    for level, p_list in prompts.items():
        print(f"   Pearl Level {level}: {len(p_list)} prompts")

    return prompts


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


def execute_rlhf_with_pearls_ladder(
    dataset: List[Dict],
    policy_model,
    reward_model,
    tokenizer,
    iterations: int,
    lr: float,
    device: str,
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

    reward_model.eval()

    # Build curriculum prompts
    prompts = _build_curriculum_prompts(dataset)

    optimizer = AdamW(policy_model.parameters(), lr=lr)
    policy_model.train()

    level_losses = {1: [], 2: [], 3: []}

    for i in tqdm(range(iterations), desc="RLHF Iterations"):
        # Determine current Pearl's level
        level = _get_curriculum_level(i, iterations)
        level_prompts = prompts.get(level, prompts[1])

        if not level_prompts:
            print(f"   ⚠ No prompts for level {level}, falling back to L1")
            level_prompts = prompts[1]

        # Sample a mini-batch of prompts
        batch_size = min(8, len(level_prompts))
        batch_prompts = random.sample(level_prompts, batch_size)

        # Tokenize
        batch = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            max_length=256,
            truncation=True,
        ).to(device)

        # Generate responses from policy model
        with torch.no_grad():
            gen_ids = policy_model.generate(
                **batch,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Score with reward model (7D rewards, averaged)
            rewards = reward_model(gen_ids).mean()

        # Policy gradient with reward signal
        log_probs = F.log_softmax(
            policy_model(**batch).logits, dim=-1
        ).mean()

        # REINFORCE-style loss (negative because we want to maximize reward)
        loss = -log_probs * rewards

        # Add safety bonus: penalize responses that don't include safety tokens
        # This encourages the model to always flag requires_human_approval
        safety_bonus = 0.0
        for gen_id in gen_ids:
            decoded = tokenizer.decode(gen_id, skip_special_tokens=False)
            if "[SAFETY_CHECK]" in decoded or "requires_human_approval" in decoded:
                safety_bonus += 0.1
        loss = loss - safety_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        level_losses[level].append(loss.item())

        if (i + 1) % 5 == 0:
            print(
                f"   Iter {i+1}/{iterations} | Level: L{level} | "
                f"Loss: {loss.item():.4f} | Reward: {rewards.item():.4f}"
            )

    # Print level-wise summary
    print("\n📊 Training Summary:")
    for level in [1, 2, 3]:
        losses = level_losses[level]
        if losses:
            avg = sum(losses) / len(losses)
            labels = ["Association", "Intervention", "Counterfactual"]
            print(f"   L{level} ({labels[level-1]}): {len(losses)} iters, avg_loss={avg:.4f}")

    # Save final model
    policy_model.save_pretrained("./results/final_rlhf_model")
    tokenizer.save_pretrained("./results/final_rlhf_model")
    print("💾 Final RLHF model saved to ./results/final_rlhf_model")
    return policy_model, tokenizer
