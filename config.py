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
# If you hit HuggingFace quotas, gated access limits, or are on a FREE Colab tier,
# set USE_FREE_LLM="1" to use a completely open, ungated model.
USE_FREE_LLM = os.environ.get("USE_FREE_LLM", "1") # Defaulting to TinyLlama for local Mac runs
FAST_LOCAL_TEST = os.environ.get("FAST_LOCAL_TEST", "1") == "1"

if USE_FREE_LLM == "1":
    # TinyLlama fits perfectly on free Colab T4 GPUs and needs NO HuggingFace token
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
elif USE_FREE_LLM == "2":
    # Zephyr is an ungated Mistral-7B equivalent (needs NO token, but requires 16GB VRAM)
    MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
else:
    # Requires HF token and gated access approval on HuggingFace
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

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
    num_train_epochs=1 if FAST_LOCAL_TEST else 4,
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

# ── RLHF Training ───────────────────────────────────────────────────────────
RLHF_ITERATIONS = 2 if FAST_LOCAL_TEST else 100
RLHF_LR = 1e-5

# ── SRE-Specific Configuration ──────────────────────────────────────────────
SRE_DATASET_SIZE = 2500           # Massive dataset for SRE causal scenarios
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
