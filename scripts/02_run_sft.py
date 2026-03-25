"""
SRE-Nidaan Pipeline — Script 02: Supervised Fine-Tuning (SFT)
==============================================================
Fine-tunes Mistral-7B with QLoRA on SRE causal incident data.
"""

import sys
import os
import json

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.model_utils import load_frontier_model_and_tokenizer
from src.training.sft_trainer import SRENexusSFT
import config


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 2: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    # Load dataset
    with open("data/sre_nidaan_dataset.json", "r") as f:
        dataset = json.load(f)
    
    if getattr(config, "FAST_LOCAL_TEST", False):
        dataset = dataset[:10]

    train_data, _ = train_test_split(dataset, test_size=0.15, random_state=42)
    print(f"📊 Using {len(train_data)} examples for SFT.")

    # Load model & tokenizer
    model, tokenizer = load_frontier_model_and_tokenizer(
        config.MODEL_ID, config.BNB_CONFIG
    )

    # Run SFT
    sft = SRENexusSFT(model, tokenizer, config.SFT_TRAINING_ARGS, config.LORA_CONFIG)
    sft.setup_special_tokens()
    sft.setup_frontier_lora()
    sft.train(train_data)

    print("\n--- SFT Phase Complete ---\n")


if __name__ == "__main__":
    main()
