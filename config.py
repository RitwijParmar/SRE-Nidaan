"""
SRE-Nidaan Configuration
========================
Centralized configuration for the entire NEXUS-CAUSAL SRE pipeline.
Mirrors the NEXUS-CAUSAL v3.1 config pattern with SRE-specific parameters.
"""

import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType

import os

# ── Model & Tokenizer ────────────────────────────────────────────────────────
FAST_LOCAL_TEST = os.environ.get("FAST_LOCAL_TEST", "1") == "1"
USE_FREE_LLM = os.environ.get("USE_FREE_LLM", "0")

DEFAULT_PRODUCTION_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
FREE_MODEL_ID_BY_FLAG = {
    "1": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "2": "HuggingFaceH4/zephyr-7b-beta",
}

# Explicit MODEL_ID wins. Otherwise keep the production default unless
# USE_FREE_LLM is set for lower-resource Colab experiments.
MODEL_ID = os.environ.get(
    "MODEL_ID",
    FREE_MODEL_ID_BY_FLAG.get(USE_FREE_LLM, DEFAULT_PRODUCTION_MODEL_ID),
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Quantization (4-bit QLoRA) ───────────────────────────────────────────────
# BitsAndBytes (4-bit) is only supported on NVIDIA CUDA GPUs. 
# We disable it for Mac (MPS) or CPU execution.
if DEVICE == "cuda":
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
else:
    BNB_CONFIG = None

# ── LoRA Configuration ──────────────────────────────────────────────────────
# r=64 matches NEXUS-CAUSAL v3.1 and caps max_lora_rank for vLLM serving
LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── SFT Training Arguments ──────────────────────────────────────────────────
SFT_TRAINING_ARGS = TrainingArguments(
    output_dir="./results/sft_model",
    num_train_epochs=1 if FAST_LOCAL_TEST else 10,
    per_device_train_batch_size=1 if FAST_LOCAL_TEST else 2,
    gradient_accumulation_steps=1 if FAST_LOCAL_TEST else 4,
    optim="adamw_torch" if FAST_LOCAL_TEST else "paged_adamw_32bit", # Paged optimizers are CUDA only
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5 if FAST_LOCAL_TEST else 10,
    save_strategy="epoch",
    fp16=False,
    bf16=False if FAST_LOCAL_TEST else True, # Apple MPS often doesn't support bfloat16 properly yet
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",
    max_steps=2 if FAST_LOCAL_TEST else -1, # Run only 2 steps for lightning fast test
)

# ── Reward Model Training ───────────────────────────────────────────────────
REWARD_MODEL_EPOCHS = 1 if FAST_LOCAL_TEST else 3
REWARD_MODEL_LR = 5e-5
REWARD_PREFERENCE_MODE = os.environ.get("REWARD_PREFERENCE_MODE", "prompt_matched")
REWARD_NEGATIVE_VARIANTS = int(
    os.environ.get("REWARD_NEGATIVE_VARIANTS", "2" if FAST_LOCAL_TEST else "3")
)

# ── RLHF Training ───────────────────────────────────────────────────────────
RLHF_ITERATIONS = 2 if FAST_LOCAL_TEST else 500
RLHF_LR = 1e-5
RLHF_SCHEMA_BONUS_WEIGHT = float(
    os.environ.get("RLHF_SCHEMA_BONUS_WEIGHT", "0.0")
)
RLHF_REFERENCE_KL_COEF = float(
    os.environ.get("RLHF_REFERENCE_KL_COEF", "0.02")
)
RLHF_EVAL_INTERVAL = int(
    os.environ.get("RLHF_EVAL_INTERVAL", "1" if FAST_LOCAL_TEST else "10")
)
RLHF_SHORT_STAGE_ITERATIONS = int(
    os.environ.get("RLHF_SHORT_STAGE_ITERATIONS", "2" if FAST_LOCAL_TEST else "30")
)
RLHF_MIN_IMPROVEMENT = float(
    os.environ.get("RLHF_MIN_IMPROVEMENT", "0.0")
)

# ── SRE-Specific Configuration ──────────────────────────────────────────────
SRE_DATASET_SIZE = 10000          # Scaled up for production SRE causal scenarios
SRE_DOMAINS = [
    "kubernetes", "database", "networking",
    "load_balancer", "dns", "storage",
    "authentication", "message_queue", "cache",
    "ci_cd", "monitoring", "api_gateway",
]
if FAST_LOCAL_TEST:
    SRE_DOMAINS = SRE_DOMAINS[:1] # Just test Kubernetes to skip waiting

# ── Inference Server ────────────────────────────────────────────────────────
VLLM_PORT = 8000
BACKEND_PORT = 8001
MAX_MODEL_LEN = 2048
MAX_LORA_RANK = 64

# ── Safety Plane ────────────────────────────────────────────────────────────
SAFETY_MODE = "read-only-copilot"
BLAST_RADIUS_LIMIT = 3  # max services affected before requiring escalation
