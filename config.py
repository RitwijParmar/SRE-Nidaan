"""
SRE-Nidaan Configuration
========================
Centralized configuration for the entire NEXUS-CAUSAL SRE pipeline.
Mirrors the NEXUS-CAUSAL v3.1 config pattern with SRE-specific parameters.
"""

import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType

# ── Model & Tokenizer ────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Quantization (4-bit QLoRA) ───────────────────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

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
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",
)

# ── Reward Model Training ───────────────────────────────────────────────────
REWARD_MODEL_EPOCHS = 2
REWARD_MODEL_LR = 5e-5

# ── RLHF Training ───────────────────────────────────────────────────────────
RLHF_ITERATIONS = 15
RLHF_LR = 1e-5

# ── SRE-Specific Configuration ──────────────────────────────────────────────
SRE_DATASET_SIZE = 2500           # Massive dataset for SRE causal scenarios
SRE_DOMAINS = [
    "kubernetes", "database", "networking",
    "load_balancer", "dns", "storage",
    "authentication", "message_queue", "cache",
    "ci_cd", "monitoring", "api_gateway",
]

# ── Inference Server ────────────────────────────────────────────────────────
VLLM_PORT = 8000
BACKEND_PORT = 8001
MAX_MODEL_LEN = 2048
MAX_LORA_RANK = 64

# ── Safety Plane ────────────────────────────────────────────────────────────
SAFETY_MODE = "read-only-copilot"
BLAST_RADIUS_LIMIT = 3  # max services affected before requiring escalation
