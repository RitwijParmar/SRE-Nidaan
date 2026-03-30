"""
SRE-Nidaan Pipeline — Script 02: Supervised Fine-Tuning (SFT)
==============================================================
Fine-tunes the configured instruct model with QLoRA on SRE causal incident data.
"""

import sys
import os
import json
import copy

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.model_utils import (
    load_frontier_model_and_tokenizer,
    load_peft_checkpoint,
)
from src.training.sft_trainer import SRENexusSFT
from src.utils.sre_schema import build_curated_continuation_subset
import config


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 2: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    # Load dataset
    dataset_path = os.environ.get("SFT_DATASET_PATH", "data/sre_nidaan_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    if getattr(config, "FAST_LOCAL_TEST", False):
        dataset = dataset[:10]

    if _env_flag("SFT_CURATED_CONTINUATION", False):
        max_examples = int(os.environ.get("SFT_CONTINUATION_SIZE", "384"))
        dataset = build_curated_continuation_subset(dataset, max_examples=max_examples)
        print(f"🧪 Using curated continuation subset with {len(dataset)} examples.")

    train_data, _ = train_test_split(dataset, test_size=0.15, random_state=42)
    print(f"📊 Using {len(train_data)} examples for SFT.")

    output_dir = os.environ.get("SFT_OUTPUT_DIR", config.SFT_TRAINING_ARGS.output_dir)
    training_args = copy.deepcopy(config.SFT_TRAINING_ARGS)
    training_args.output_dir = output_dir
    epoch_override = os.environ.get("SFT_EPOCHS_OVERRIDE")
    if epoch_override:
        training_args.num_train_epochs = float(epoch_override)

    base_adapter_dir = os.environ.get("SFT_BASE_ADAPTER_DIR")
    strict_lexical_cues = _env_flag("SFT_STRICT_LEXICAL_CUES", False)

    # Load model & tokenizer
    if base_adapter_dir:
        print(f"♻️ Continuing from saved adapter: {base_adapter_dir}")
        model, tokenizer = load_peft_checkpoint(
            base_adapter_dir,
            config.MODEL_ID,
            config.BNB_CONFIG,
            is_trainable=True,
        )
    else:
        model, tokenizer = load_frontier_model_and_tokenizer(
            config.MODEL_ID, config.BNB_CONFIG
        )

    # Run SFT
    sft = SRENexusSFT(
        model,
        tokenizer,
        training_args,
        config.LORA_CONFIG,
        model_name=config.MODEL_ID,
        strict_lexical_cues=strict_lexical_cues,
    )
    if not base_adapter_dir:
        sft.setup_special_tokens()
        sft.setup_frontier_lora()
    sft.train(train_data)

    print("\n--- SFT Phase Complete ---\n")


if __name__ == "__main__":
    main()
