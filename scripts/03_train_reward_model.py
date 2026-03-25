"""
SRE-Nidaan Pipeline — Script 03: Train Reward Model
=====================================================
Trains the 7-dimensional SRE reward model on preference pairs.
"""

import sys
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.reward_modeler import train_reward_model
import config


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 3: Reward Model Training")
    print("=" * 60)

    # Load dataset
    with open("data/sre_nidaan_dataset.json", "r") as f:
        dataset = json.load(f)

    # Load SFT model as base for reward model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    sft_model = PeftModel.from_pretrained(
        base_model, config.SFT_TRAINING_ARGS.output_dir
    )
    sft_tokenizer = AutoTokenizer.from_pretrained(
        config.SFT_TRAINING_ARGS.output_dir
    )

    os.makedirs("results", exist_ok=True)

    # Train reward model
    train_reward_model(
        dataset,
        sft_model,
        sft_tokenizer,
        config.REWARD_MODEL_EPOCHS,
        config.REWARD_MODEL_LR,
        config.DEVICE,
    )

    print("\n--- Reward Model Training Complete ---\n")


if __name__ == "__main__":
    main()
