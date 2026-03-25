"""
SRE-Nidaan: Complete Training Pipeline for Google Colab
========================================================
Run this script end-to-end on Google Colab Premium (A100/V100).

Usage in Colab:
    !git clone https://github.com/RitwijParmar/SRE-Nidaan.git
    %cd SRE-Nidaan
    !pip install -r requirements.txt
    !python notebooks/colab_train_full_pipeline.py

This script runs all 5 phases sequentially:
  Phase 1: Dataset is pre-generated (data/sre_nidaan_dataset.json)
  Phase 2: QLoRA Supervised Fine-Tuning
  Phase 3: 7D Reward Model Training
  Phase 4: Pearl's Ladder RLHF
  Phase 5: Final Evaluation
"""

import os
import sys
import json
import time

# ── HuggingFace Authentication ───────────────────────────────────────────────
# Required for gated model: mistralai/Mistral-7B-Instruct-v0.2
# Set this in Colab: Secrets panel → Add HF_TOKEN, or:
#   import os; os.environ["HF_TOKEN"] = "your_token_here"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("=" * 70)
print("  SRE-Nidaan — Full Training Pipeline (Colab Premium)")
print("=" * 70)

# Login to HuggingFace
print("\n🔐 Authenticating with HuggingFace Hub...")
try:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HuggingFace authentication successful")
except Exception as e:
    print(f"⚠️ HuggingFace login warning: {e}")
    print("   Continuing — model may already be cached or public")

# ── System Path Setup ────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
print(f"\n🖥️  Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (training will be extremely slow)'}")
if torch.cuda.is_available():
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")

# ── Ensure dataset exists ────────────────────────────────────────────────────
DATASET_PATH = "data/sre_nidaan_dataset.json"

if not os.path.exists(DATASET_PATH):
    print("\n📊 Dataset not found — generating 2,500 SRE incident examples...")
    from src.data.dataset_generator import SREDatasetGenerator, save_dataset
    generator = SREDatasetGenerator()
    dataset = generator.create_sre_dataset(num_examples=2500)
    save_dataset(dataset, DATASET_PATH)
else:
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
    print(f"\n📊 Dataset loaded: {len(dataset)} examples ({os.path.getsize(DATASET_PATH) / 1e6:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Supervised Fine-Tuning (SFT)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Phase 2: Supervised Fine-Tuning (SFT)")
print("=" * 70)

from sklearn.model_selection import train_test_split
from src.utils.model_utils import load_frontier_model_and_tokenizer
from src.training.sft_trainer import SRENexusSFT
import config

# Split dataset
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)
train_data, eval_data = train_test_split(dataset, test_size=0.15, random_state=42)
print(f"📊 Training: {len(train_data)} | Evaluation: {len(eval_data)}")

# Load model
t0 = time.time()
model, tokenizer = load_frontier_model_and_tokenizer(config.MODEL_ID, config.BNB_CONFIG)
print(f"⏱️  Model loaded in {time.time() - t0:.1f}s")

# Run SFT
sft = SRENexusSFT(model, tokenizer, config.SFT_TRAINING_ARGS, config.LORA_CONFIG)
sft.setup_special_tokens()
sft.setup_frontier_lora()

t0 = time.time()
sft.train(train_data)
sft_time = time.time() - t0
print(f"⏱️  SFT completed in {sft_time / 60:.1f} minutes")

# Free memory
del model
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Reward Model Training
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Phase 3: 7-Dimensional Reward Model Training")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.training.reward_modeler import train_reward_model

# Load SFT model as base
base_model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
)
sft_model = PeftModel.from_pretrained(base_model, config.SFT_TRAINING_ARGS.output_dir)
sft_tokenizer = AutoTokenizer.from_pretrained(config.SFT_TRAINING_ARGS.output_dir)

os.makedirs("results", exist_ok=True)

t0 = time.time()
reward_model = train_reward_model(
    dataset, sft_model, sft_tokenizer,
    config.REWARD_MODEL_EPOCHS, config.REWARD_MODEL_LR, config.DEVICE,
)
rm_time = time.time() - t0
print(f"⏱️  Reward model trained in {rm_time / 60:.1f} minutes")

# Free memory
del base_model, sft_model, reward_model
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: RLHF with Pearl's Ladder Curriculum
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Phase 4: RLHF with Pearl's Ladder Curriculum")
print("=" * 70)

from src.training.rlhf_trainer import execute_rlhf_with_pearls_ladder
from src.training.reward_modeler import SRE7DRewardModel

# Load policy model
policy_base = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
)
policy_model = PeftModel.from_pretrained(policy_base, config.SFT_TRAINING_ARGS.output_dir)
tokenizer = AutoTokenizer.from_pretrained(config.SFT_TRAINING_ARGS.output_dir)

# Load reward model
reward_base = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
)
reward_sft = PeftModel.from_pretrained(reward_base, config.SFT_TRAINING_ARGS.output_dir)
reward_model = SRE7DRewardModel(reward_sft, tokenizer)
reward_model.reward_head.load_state_dict(torch.load("results/reward_model_head.pt"))
reward_model.to(config.DEVICE)

t0 = time.time()
execute_rlhf_with_pearls_ladder(
    dataset, policy_model, reward_model, tokenizer,
    config.RLHF_ITERATIONS, config.RLHF_LR, config.DEVICE,
)
rlhf_time = time.time() - t0
print(f"⏱️  RLHF completed in {rlhf_time / 60:.1f} minutes")

# Free memory
del policy_base, policy_model, reward_base, reward_sft, reward_model
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Final Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Phase 5: Final Model Evaluation")
print("=" * 70)

from src.evaluation.evaluator import SREEvaluationFramework

# Load final RLHF model
final_base = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
)
final_model = PeftModel.from_pretrained(final_base, "./results/final_rlhf_model")
final_tokenizer = AutoTokenizer.from_pretrained("./results/final_rlhf_model")

evaluator = SREEvaluationFramework(final_model, final_tokenizer, config.DEVICE)
results = evaluator.conduct_evaluation()


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total_time = sft_time + rm_time + rlhf_time

print("\n" + "=" * 70)
print("  🏆 SRE-Nidaan Training Complete!")
print("=" * 70)
print(f"   SFT:             {sft_time / 60:.1f} min")
print(f"   Reward Model:    {rm_time / 60:.1f} min")
print(f"   RLHF:            {rlhf_time / 60:.1f} min")
print(f"   Total:           {total_time / 60:.1f} min")
print(f"\n   Final Score:     {results['overall_score']:.3f}")
print(f"   Assessment:      {results['assessment']}")
print(f"   Safety Rate:     {results['safety_compliance_rate']:.1%}")
print(f"\n   Artifacts saved to ./results/")
print(f"   - results/sft_model/           (SFT LoRA weights)")
print(f"   - results/reward_model_head.pt (Reward head)")
print(f"   - results/final_rlhf_model/    (Final model)")
print(f"   - results/final_evaluation_report.json")
print("=" * 70)
