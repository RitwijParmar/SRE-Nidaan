"""
SRE-Nidaan: 7-Dimensional Reward Model
========================================
Phase 2 of the NEXUS-CAUSAL SRE training pipeline.
Trains a multi-dimensional reward function that scores SRE causal analyses
across seven quality dimensions specific to incident response.

Mirrors NEXUS-CAUSAL v3.1 src/training/reward_modeler.py with SRE-specific
reward dimensions (blast radius awareness, safety compliance, etc.).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Dict
from tqdm import tqdm

from src.utils.model_utils import build_training_example
from src.utils.sre_schema import (
    REQUIRED_RESPONSE_TAGS,
    build_structured_training_response,
    coerce_structured_response,
    extract_tagged_sections,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Dimensions for SRE Causal Analysis
# ─────────────────────────────────────────────────────────────────────────────
# 1. Structural Accuracy    — Is the causal graph structurally correct?
# 2. Root Cause Precision   — Does it identify the true root cause vs symptoms?
# 3. Confounder Detection   — Does it explain why naive interventions fail?
# 4. Counterfactual Quality — Is the counterfactual reasoning sound?
# 5. DAG Completeness       — Are all relevant nodes and edges present?
# 6. Blast Radius Awareness — Does it assess cascading failure impact?
# 7. Safety Compliance      — Does the response enforce human-in-the-loop?

REWARD_DIMENSIONS = [
    "structural_accuracy",
    "root_cause_precision",
    "confounder_detection",
    "counterfactual_quality",
    "dag_completeness",
    "blast_radius_awareness",
    "safety_compliance",
]

REWARD_RESPONSE_TAG_ORDER = [
    "[DOMAIN]",
    "[PEARL_LEVEL]",
    *REQUIRED_RESPONSE_TAGS,
]


class SRE7DRewardModel(nn.Module):
    """
    7-Dimensional Reward Model for SRE Causal Reasoning.

    Extends the NEXUS-CAUSAL Frontier7DRewardModel with SRE-specific
    reward dimensions (blast radius, safety compliance).

    Architecture:
        base_model (frozen) → last hidden state → reward_head → 7D score
    """

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        hidden_size = self.base_model.config.hidden_size

        # 7-dimensional reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 7),  # 7 reward dimensions
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass: extract last-token hidden state, project to 7D reward.
        """
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Get the last hidden state
        hidden = out.hidden_states[-1]

        # Extract the last non-padding token's representation
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(-1) - 1
        else:
            seq_lengths = torch.ne(
                input_ids, self.tokenizer.pad_token_id
            ).sum(-1) - 1

        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), seq_lengths
        ]

        return self.reward_head(pooled.to(self.reward_head[0].weight.dtype))


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _build_reward_instruction(example: Dict) -> str:
    """Use one prompt per incident so reward comparisons are prompt-matched."""
    domain = _clean_text(example.get("domain", "unknown"))
    pearl_level = example.get("pearl_level", 1)
    premise = _clean_text(example.get("premise", ""))
    return (
        "You are an SRE causal reasoning agent. "
        f"Analyze the following incident using Pearl's Causal Hierarchy (Level {pearl_level}).\n"
        f"- Domain: {domain}\n"
        f"- Incident: {premise}\n"
        "Return the answer using [ROOT_CAUSE], [INTERVENTION], [COUNTERFACTUAL], "
        "[DAG], [REMEDIATION], and [SAFETY_CHECK]."
    )


def _render_tagged_response(sections: Dict[str, str]) -> str:
    ordered_tags = list(REWARD_RESPONSE_TAG_ORDER)
    ordered_tags.extend(
        tag for tag in sections
        if tag not in ordered_tags
    )
    lines = []
    for tag in ordered_tags:
        value = sections.get(tag, "").strip()
        if value:
            lines.append(f"{tag} {value}")
    return "\n".join(lines)


def _build_prompt_matched_rejections(
    example: Dict,
    chosen_response: str,
    *,
    max_variants: int,
) -> List[str]:
    """
    Create weaker answer candidates for the same incident prompt.

    These negatives are intentionally prompt-matched so the reward model learns
    to rank better-vs-worse analyses for one incident, instead of learning a
    noisy global ordering across different incidents.
    """
    sections = extract_tagged_sections(chosen_response)
    premise = _clean_text(example.get("premise", ""))
    symptom_sentence = premise.split(".", 1)[0].strip() or "The observed symptom."

    variants = []

    symptom_variant = dict(sections)
    symptom_variant["[ROOT_CAUSE]"] = (
        f"The visible symptom appears to be the root cause: {symptom_sentence}"
    )
    symptom_variant["[INTERVENTION]"] = (
        "Increase replicas immediately and suppress alerts; the structural bottleneck can be investigated later."
    )
    symptom_variant["[COUNTERFACTUAL]"] = (
        "The incident likely would have behaved the same way either way."
    )
    symptom_variant["[DAG]"] = "Several systems were involved, but the graph is omitted."
    symptom_variant["[SAFETY_CHECK]"] = (
        "requires_human_approval=false; automated remediation can proceed immediately."
    )
    variants.append(_render_tagged_response(symptom_variant))

    generic_variant = dict(sections)
    generic_variant["[ROOT_CAUSE]"] = (
        "Insufficient evidence, but the impacted service was degraded under load."
    )
    generic_variant["[INTERVENTION]"] = "Scale up the impacted service immediately."
    generic_variant["[COUNTERFACTUAL]"] = (
        "Counterfactual: this might still have happened even with the preventive change."
    )
    generic_variant["[DAG]"] = "The issue propagated through multiple components."
    generic_variant["[REMEDIATION]"] = "Restart the service and continue monitoring."
    generic_variant["[SAFETY_CHECK]"] = "requires_human_approval=true."
    variants.append(_render_tagged_response(generic_variant))

    lexical_variant = dict(sections)
    lexical_variant["[INTERVENTION]"] = (
        "Scale the service and adjust the setting immediately."
    )
    lexical_variant["[COUNTERFACTUAL]"] = (
        "With a preventive change in place, the impact may have been lower."
    )
    lexical_variant["[DAG]"] = "This path connects the issue to the outcome."
    lexical_variant["[SAFETY_CHECK]"] = (
        "Manual review can happen after the first automated mitigation."
    )
    variants.append(_render_tagged_response(lexical_variant))

    if example.get("reasoning"):
        legacy_variant = {"[ROOT_CAUSE]": _clean_text(example["reasoning"])}
        variants.append(coerce_structured_response(_render_tagged_response(legacy_variant)))

    deduped = []
    seen = set()
    for response in variants:
        normalized = coerce_structured_response(response)
        if normalized != chosen_response and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
        if len(deduped) >= max(1, max_variants):
            break
    return deduped


def _create_prompt_matched_preference_pairs(
    dataset: List[Dict],
    tokenizer=None,
    model_name: str = "",
    lexical_cues: bool = True,
    negative_variants: int = 3,
) -> List[Dict]:
    pref_data = []
    for example in dataset:
        instruction = _build_reward_instruction(example)
        chosen_response = build_structured_training_response(
            example,
            lexical_cues=lexical_cues,
        )
        rejected_candidates = _build_prompt_matched_rejections(
            example,
            chosen_response,
            max_variants=negative_variants,
        )
        for rejected_response in rejected_candidates:
            pref_data.append({
                "prompt": instruction,
                "preferred": build_training_example(
                    instruction,
                    chosen_response,
                    model_name=model_name,
                    tokenizer=tokenizer,
                ),
                "rejected": build_training_example(
                    instruction,
                    rejected_response,
                    model_name=model_name,
                    tokenizer=tokenizer,
                ),
            })
    return pref_data


def _create_sorted_quality_preference_pairs(
    dataset: List[Dict],
    tokenizer=None,
    model_name: str = "",
    lexical_cues: bool = True,
) -> List[Dict]:
    pref_data = []
    sorted_data = sorted(dataset, key=lambda x: x.get("quality_score", 0))

    for i in range(0, len(sorted_data) - 1, 2):
        s1 = sorted_data[i].get("quality_score", 0)
        s2 = sorted_data[i + 1].get("quality_score", 0)

        if s1 > s2:
            preferred, rejected = sorted_data[i], sorted_data[i + 1]
        else:
            preferred, rejected = sorted_data[i + 1], sorted_data[i]

        preferred_response = build_structured_training_response(
            preferred,
            lexical_cues=lexical_cues,
        )
        rejected_response = build_structured_training_response(
            rejected,
            lexical_cues=lexical_cues,
        )

        pref_data.append({
            "prompt": _build_reward_instruction(preferred),
            "preferred": build_training_example(
                f"Analyze incident: {preferred['premise']}",
                preferred_response,
                model_name=model_name,
                tokenizer=tokenizer,
            ),
            "rejected": build_training_example(
                f"Analyze incident: {rejected['premise']}",
                rejected_response,
                model_name=model_name,
                tokenizer=tokenizer,
            ),
        })

    return pref_data


def create_preference_pairs(
    dataset: List[Dict],
    tokenizer=None,
    model_name: str = "",
    lexical_cues: bool = True,
    mode: str = "prompt_matched",
    negative_variants: int = 3,
) -> List[Dict]:
    """
    Create preference pairs from the SRE dataset.

    The default prompt-matched mode creates multiple better-vs-worse answers for
    the same incident prompt so the reward model learns relative answer quality
    instead of cross-incident dataset ordering.
    """
    if mode == "sorted_quality":
        return _create_sorted_quality_preference_pairs(
            dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            lexical_cues=lexical_cues,
        )
    return _create_prompt_matched_preference_pairs(
        dataset,
        tokenizer=tokenizer,
        model_name=model_name,
        lexical_cues=lexical_cues,
        negative_variants=negative_variants,
    )


def train_reward_model(
    dataset: List[Dict],
    sft_model,
    sft_tokenizer,
    epochs: int,
    lr: float,
    device: str,
    model_name: str = "",
    output_path: str = "results/reward_model_head.pt",
    preference_mode: str = "prompt_matched",
    negative_variants: int = 3,
):
    """
    Train the 7-dimensional SRE reward model.

    Args:
        dataset: Full SRE incident dataset
        sft_model: Frozen SFT model as the base
        sft_tokenizer: Tokenizer from SFT phase
        epochs: Number of training epochs
        lr: Learning rate for reward head
        device: Compute device
    """
    print("🎯 Training 7D SRE Reward Model...")
    print(f"   Dimensions: {', '.join(REWARD_DIMENSIONS)}")

    # Create preference pairs
    pref_data = create_preference_pairs(
        dataset,
        tokenizer=sft_tokenizer,
        model_name=model_name or getattr(sft_tokenizer, "name_or_path", ""),
        lexical_cues=True,
        mode=preference_mode,
        negative_variants=negative_variants,
    )
    print(f"📊 Created {len(pref_data)} preference pairs.")

    # Initialize reward model with frozen base
    reward_model = SRE7DRewardModel(sft_model, sft_tokenizer).to(device)
    for param in reward_model.base_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(reward_model.reward_head.parameters(), lr=lr)

    for epoch in range(epochs):
        reward_model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(pref_data, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Tokenize preferred and rejected responses
            p_in = sft_tokenizer(
                batch["preferred"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            r_in = sft_tokenizer(
                batch["rejected"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Get 7D rewards for both
            p_reward = reward_model(**p_in)  # [batch, 7]
            r_reward = reward_model(**r_in)  # [batch, 7]

            # Bradley-Terry loss: preferred should score higher across all dims
            loss = -torch.nn.functional.logsigmoid(
                p_reward.mean() - r_reward.mean()
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"   Epoch {epoch + 1} — Avg Loss: {avg_loss:.4f}")

    # Save reward head weights
    torch.save(reward_model.reward_head.state_dict(), output_path)
    print("🏆 Reward Model training complete!")
    print(f"   Saved reward head to {output_path}")
    return reward_model
