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


def create_preference_pairs(dataset: List[Dict]) -> List[Dict]:
    """
    Create preference pairs from the SRE dataset.
    Higher quality_score examples are preferred, mirroring RLHF preference data.
    """
    pref_data = []
    sorted_data = sorted(dataset, key=lambda x: x.get("quality_score", 0))

    for i in range(0, len(sorted_data) - 1, 2):
        s1 = sorted_data[i].get("quality_score", 0)
        s2 = sorted_data[i + 1].get("quality_score", 0)

        if s1 > s2:
            preferred, rejected = sorted_data[i], sorted_data[i + 1]
        else:
            preferred, rejected = sorted_data[i + 1], sorted_data[i]

        pref_data.append({
            "preferred": (
                f"[INST] Analyze incident: {preferred['premise']} [/INST]"
                f"[ROOT_CAUSE] {preferred.get('root_cause', '')}\n"
                f"[INTERVENTION] {preferred.get('confounding_action', '')}\n"
                f"[REMEDIATION] {preferred.get('correct_action', '')}"
            ),
            "rejected": (
                f"[INST] Analyze incident: {rejected['premise']} [/INST]"
                f"[ROOT_CAUSE] {rejected.get('root_cause', '')}\n"
                f"[INTERVENTION] {rejected.get('confounding_action', '')}\n"
                f"[REMEDIATION] {rejected.get('correct_action', '')}"
            ),
        })

    return pref_data


def train_reward_model(
    dataset: List[Dict],
    sft_model,
    sft_tokenizer,
    epochs: int,
    lr: float,
    device: str,
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
    pref_data = create_preference_pairs(dataset)
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
    torch.save(
        reward_model.reward_head.state_dict(),
        "results/reward_model_head.pt",
    )
    print("🏆 Reward Model training complete!")
    print(f"   Saved reward head to results/reward_model_head.pt")
    return reward_model
