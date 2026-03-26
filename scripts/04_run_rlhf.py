"""
SRE-Nidaan Pipeline — Script 04: RLHF with Pearl's Ladder
===========================================================
Aligns the model using RLHF with Pearl's Causal Hierarchy curriculum.
"""

import sys
import os
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.rlhf_trainer import execute_rlhf_with_pearls_ladder
from src.training.reward_modeler import SRE7DRewardModel
import config


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 4: RLHF Training")
    print("=" * 60)

    # Load dataset
    with open("data/sre_nidaan_dataset.json", "r") as f:
        dataset = json.load(f)
    if getattr(config, "FAST_LOCAL_TEST", False):
        dataset = dataset[:10]

    # Load policy model (SFT checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        config.SFT_TRAINING_ARGS.output_dir
    )
    policy_base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    policy_base.resize_token_embeddings(len(tokenizer))
    policy_model = PeftModel.from_pretrained(
        policy_base, config.SFT_TRAINING_ARGS.output_dir, is_trainable=True
    )

    # Load reward model
    reward_base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    reward_base.resize_token_embeddings(len(tokenizer))
    reward_sft = PeftModel.from_pretrained(
        reward_base, config.SFT_TRAINING_ARGS.output_dir
    )
    reward_model = SRE7DRewardModel(reward_sft, tokenizer)
    reward_model.reward_head.load_state_dict(
        torch.load("results/reward_model_head.pt")
    )
    reward_model.to(config.DEVICE)

    # Execute RLHF
    execute_rlhf_with_pearls_ladder(
        dataset,
        policy_model,
        reward_model,
        tokenizer,
        config.RLHF_ITERATIONS,
        config.RLHF_LR,
        config.DEVICE,
    )

    print("\n--- RLHF Phase Complete ---\n")


if __name__ == "__main__":
    main()
