"""
SRE-Nidaan Pipeline — Script 03: Train Reward Model
=====================================================
Trains the 7-dimensional SRE reward model on preference pairs.
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.reward_modeler import train_reward_model
from src.utils.model_utils import load_peft_checkpoint
import config


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 3: Reward Model Training")
    print("=" * 60)

    # Load dataset
    dataset_path = os.environ.get("REWARD_DATASET_PATH", "data/sre_nidaan_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    if getattr(config, "FAST_LOCAL_TEST", False):
        dataset = dataset[:10]

    # Load SFT model as base for reward model
    sft_adapter_dir = os.environ.get(
        "SFT_ADAPTER_DIR",
        config.SFT_TRAINING_ARGS.output_dir,
    )
    sft_model, sft_tokenizer = load_peft_checkpoint(
        sft_adapter_dir,
        config.MODEL_ID,
        config.BNB_CONFIG,
    )

    os.makedirs("results", exist_ok=True)
    output_path = os.environ.get(
        "REWARD_HEAD_OUTPUT_PATH",
        "results/reward_model_head.pt",
    )

    # Train reward model
    train_reward_model(
        dataset,
        sft_model,
        sft_tokenizer,
        config.REWARD_MODEL_EPOCHS,
        config.REWARD_MODEL_LR,
        config.DEVICE,
        model_name=config.MODEL_ID,
        output_path=output_path,
        preference_mode=os.environ.get(
            "REWARD_PREFERENCE_MODE",
            config.REWARD_PREFERENCE_MODE,
        ),
        negative_variants=int(
            os.environ.get(
                "REWARD_NEGATIVE_VARIANTS",
                str(config.REWARD_NEGATIVE_VARIANTS),
            )
        ),
    )

    print("\n--- Reward Model Training Complete ---\n")


if __name__ == "__main__":
    main()
