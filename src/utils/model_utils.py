"""
SRE-Nidaan: Model Utilities
============================
Shared model loading functions with quantization support.
Mirrors NEXUS-CAUSAL v3.1 src/utils/model_utils.py pattern.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from peft import PeftModel
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def load_peft_checkpoint(
    adapter_dir: str,
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig],
    *,
    is_trainable: bool = False,
    tokenizer_padding_side: Optional[str] = None,
):
    """
    Load a saved LoRA adapter together with the matching tokenizer.

    The adapter directory owns the tokenizer state because SFT adds special
    schema tokens that must exist before the adapter weights are loaded.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_padding_side or "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
    return peft_model, tokenizer


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("checkpoint-"):
        step = name.split("checkpoint-", 1)[1]
        if step.isdigit():
            return (0, f"{int(step):012d}")
    return (1, name)


def discover_adapter_checkpoints(adapter_root: str) -> list[str]:
    """
    Return all adapter checkpoints under an SFT output directory.

    The root adapter directory is appended when it contains a final adapter save,
    so callers can compare intermediate checkpoints with the final export.
    """
    root = Path(adapter_root)
    if not root.exists():
        return []

    checkpoints = sorted(
        [
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ],
        key=_checkpoint_sort_key,
    )
    checkpoint_paths = [str(path) for path in checkpoints]

    if (root / "adapter_config.json").exists():
        checkpoint_paths.append(str(root))

    return checkpoint_paths


def get_model_family(model_name: str) -> str:
    """Return the chat-format family used by the supplied model name."""
    lowered = (model_name or "").lower()
    if "llama-3" in lowered or "meta-llama-3" in lowered:
        return "llama3"
    if "mistral" in lowered:
        return "mistral"
    return "generic"


def _apply_chat_template(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
):
    if tokenizer is None:
        return None

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None

    try:
        return apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return None


def _format_mistral_chat(
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
) -> str:
    prompt_parts: list[str] = []
    pending_system: list[str] = []

    for raw_message in messages:
        role = raw_message["role"]
        content = raw_message["content"].strip()

        if role == "system":
            pending_system.append(content)
            continue

        if role == "user":
            if pending_system:
                content = "\n\n".join(pending_system + [content])
                pending_system.clear()
            prompt_parts.append(f"<s>[INST] {content} [/INST]")
            continue

        if role == "assistant":
            prompt_parts.append(f" {content}</s>")

    prompt = "".join(prompt_parts)
    if add_generation_prompt and messages and messages[-1]["role"] == "assistant":
        return prompt
    return prompt


def _format_llama3_chat(
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
) -> str:
    prompt_parts = ["<|begin_of_text|>"]

    for raw_message in messages:
        role = raw_message["role"]
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = raw_message["content"].strip()
        prompt_parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )

    if add_generation_prompt:
        prompt_parts.append(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    return "".join(prompt_parts)


def _format_plain_chat(
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
) -> str:
    prompt = "\n\n".join(
        f"{message['role'].upper()}: {message['content'].strip()}"
        for message in messages
    )
    if add_generation_prompt:
        prompt = f"{prompt}\n\nASSISTANT:"
    return prompt


def build_chat_prompt(
    messages: Sequence[Mapping[str, str]],
    model_name: str,
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Build a chat prompt that matches the active model family.
    Prefer tokenizer chat templates when available, then fall back to
    model-family-specific manual templates.
    """
    prompt = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    if prompt is not None:
        return prompt

    model_family = get_model_family(model_name)
    if model_family == "llama3":
        return _format_llama3_chat(messages, add_generation_prompt)
    if model_family == "mistral":
        return _format_mistral_chat(messages, add_generation_prompt)
    return _format_plain_chat(messages, add_generation_prompt)


def build_training_example(
    instruction: str,
    response: str,
    model_name: str,
    tokenizer=None,
    system: Optional[str] = None,
) -> str:
    """Build a full supervised training example for the active chat model."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})

    return build_chat_prompt(
        messages,
        model_name=model_name,
        tokenizer=tokenizer,
        add_generation_prompt=False,
    )
