"""
SRE-Nidaan: Model Utilities
============================
Shared model loading functions with quantization support.
Mirrors NEXUS-CAUSAL v3.1 src/utils/model_utils.py pattern.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_frontier_model_and_tokenizer(model_name: str, quantization_config: BitsAndBytesConfig):
    """Loads a model with a specified quantization config and its tokenizer."""
    print(f"🤖 Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def format_mistral_prompt(system: str, user: str) -> str:
    """Format a prompt following Mistral-7B-Instruct-v0.2 token conventions."""
    return f"<s>[INST] {system}\n\n{user} [/INST]"


def format_mistral_training_example(instruction: str, response: str) -> str:
    """Format a training example in Mistral instruction format."""
    return f"<s>[INST] {instruction} [/INST]{response}</s>"
