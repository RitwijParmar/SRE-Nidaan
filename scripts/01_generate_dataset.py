"""
SRE-Nidaan Pipeline — Script 01: Generate Dataset
===================================================
Generates a massive SRE causal incident training dataset.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.dataset_generator import SREDatasetGenerator, save_dataset


def main():
    print("=" * 60)
    print("  SRE-Nidaan — Phase 1: Data Generation")
    print("=" * 60)

    generator = SREDatasetGenerator()
    dataset = generator.create_sre_dataset(num_examples=2500)
    save_dataset(dataset, "data/sre_nidaan_dataset.json")

    print("\n--- Data Generation Complete ---\n")


if __name__ == "__main__":
    main()
