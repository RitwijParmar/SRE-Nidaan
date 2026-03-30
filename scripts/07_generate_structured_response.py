"""
SRE-Nidaan Pipeline — Script 07: Structured Inference from Saved Adapter
=========================================================================
Load a saved SFT/RLHF adapter, apply the strict schema wrapper, and optionally
rerank candidates with the saved reward head.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.reward_modeler import SRE7DRewardModel
from src.utils.model_utils import load_peft_checkpoint
from src.utils.sre_schema import generate_and_rerank_structured_response
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incident",
        required=True,
        help="Incident premise to analyze.",
    )
    parser.add_argument(
        "--model-dir",
        default=config.SFT_TRAINING_ARGS.output_dir,
        help="Adapter directory to load.",
    )
    parser.add_argument(
        "--reward-head",
        default="",
        help="Optional reward head path for candidate reranking.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="Number of candidate responses to sample before reranking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_peft_checkpoint(
        args.model_dir,
        config.MODEL_ID,
        config.BNB_CONFIG,
    )

    reward_model = None
    if args.reward_head and os.path.exists(args.reward_head):
        reward_model = SRE7DRewardModel(model, tokenizer)
        reward_model.reward_head.load_state_dict(
            torch.load(args.reward_head, map_location=config.DEVICE)
        )
        reward_model.to(config.DEVICE)

    result = generate_and_rerank_structured_response(
        model,
        tokenizer,
        premise=args.incident,
        model_name=config.MODEL_ID,
        device=config.DEVICE,
        reward_model=reward_model,
        num_candidates=args.num_candidates,
    )

    print("\n=== Best Structured Response ===\n")
    print(result["best_response"])
    print("\n=== Candidate Scores ===\n")
    for idx, candidate in enumerate(result["candidates"], start=1):
        print(
            f"{idx}. total={candidate['total_score']:.4f} "
            f"reward={candidate['reward_score']:.4f} "
            f"heuristic={candidate['heuristic_bonus']:.4f}"
        )


if __name__ == "__main__":
    main()
