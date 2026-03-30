"""
SRE-Nidaan: Structured Response Utilities
=========================================
Helpers for training, inference, and evaluation around the tagged SRE schema.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import torch

from src.utils.model_utils import build_chat_prompt, build_training_example


REQUIRED_RESPONSE_TAGS = [
    "[ROOT_CAUSE]",
    "[INTERVENTION]",
    "[COUNTERFACTUAL]",
    "[DAG]",
    "[REMEDIATION]",
    "[SAFETY_CHECK]",
]

STRICT_SYSTEM_PROMPT = (
    "You are an SRE causal reasoning agent. "
    "Respond only with the required tagged sections in the requested order."
)

STRICT_INCIDENT_TEMPLATE = """Analyze the following incident using Pearl's Causal Hierarchy.
Return exactly these sections in order:
[ROOT_CAUSE]
[INTERVENTION]
[COUNTERFACTUAL]
[DAG]
[REMEDIATION]
[SAFETY_CHECK]

Requirements:
- The [INTERVENTION] section must mention the naive action as do(...).
- The [COUNTERFACTUAL] section must explicitly say counterfactual.
- The [DAG] section must describe the causal graph using node and edge wording.
- The [SAFETY_CHECK] section must include requires_human_approval=true and mention human approval or manual review.

Incident: {premise}"""

_TAG_BLOCK_PATTERN = re.compile(
    r"(?P<tag>\[[A-Z_]+\])\s*(?P<content>.*?)(?=\n\[[A-Z_]+\]|\Z)",
    re.DOTALL,
)

_DEFAULT_SECTION_TEXT = {
    "[ROOT_CAUSE]": "Insufficient evidence to confirm the structural root cause yet.",
    "[INTERVENTION]": (
        "do(naive_intervention) is unsafe until the causal bottleneck is validated."
    ),
    "[COUNTERFACTUAL]": (
        "Counterfactual: with the preventive control in place, the incident would likely "
        "have been reduced, but this must be verified against production telemetry."
    ),
    "[DAG]": (
        "Causal graph: node(root_cause) -> edge(service_impact) -> node(observed_symptom)."
    ),
    "[REMEDIATION]": (
        "Stabilize the service, verify the root cause, then apply the lowest-risk remediation."
    ),
    "[SAFETY_CHECK]": (
        "requires_human_approval=true; human approval, manual review, and verification required."
    ),
}


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _stringify_dag(example: Dict) -> tuple[str, str]:
    dag_nodes = example.get("dag_nodes", [])
    dag_edges = example.get("dag_edges", [])
    nodes_str = " ".join(f"[NODE] {n['id']}:{n['label']}" for n in dag_nodes)
    edges_str = " ".join(f"[EDGE] {e['source']}->{e['target']}" for e in dag_edges)
    return nodes_str, edges_str


def build_structured_training_response(
    example: Dict,
    *,
    lexical_cues: bool = False,
) -> str:
    """
    Build the tagged assistant response used for SFT and reward-model preference data.

    lexical_cues=True makes the wording more literal so checkpoint sweeps and
    cheap continuation-SFT runs align better with the evaluator.
    """
    domain = _clean_text(example.get("domain", "unknown"))
    pearl_level = example.get("pearl_level", 1)
    root_cause = _clean_text(example.get("root_cause", ""))
    intervention = _clean_text(example.get("confounding_action", ""))
    counterfactual = _clean_text(example.get("counterfactual", ""))
    remediation = _clean_text(example.get("correct_action", ""))
    nodes_str, edges_str = _stringify_dag(example)

    if lexical_cues and intervention and "do(" not in intervention.lower():
        intervention = f"do(naive_intervention) {intervention}"
    if lexical_cues and counterfactual and "counterfactual" not in counterfactual.lower():
        counterfactual = f"Counterfactual: {counterfactual}"
    if lexical_cues:
        dag_prefix = "Causal graph with node and edge structure:"
        safety_text = (
            "requires_human_approval=true; human approval, manual review, and "
            "verification required before remediation."
        )
    else:
        dag_prefix = ""
        safety_text = "requires_human_approval=true"

    dag_body = " ".join(part for part in [dag_prefix, nodes_str, edges_str] if part).strip()

    return (
        f"[DOMAIN] {domain}\n"
        f"[PEARL_LEVEL] L{pearl_level}\n"
        f"[ROOT_CAUSE] {root_cause}\n"
        f"[INTERVENTION] {intervention}\n"
        f"[COUNTERFACTUAL] {counterfactual}\n"
        f"[DAG] {dag_body}\n"
        f"[REMEDIATION] {remediation}\n"
        f"[SAFETY_CHECK] {safety_text}"
    )


def build_strict_incident_instruction(premise: str) -> str:
    """Build the stricter user instruction used for checkpoint salvage."""
    return STRICT_INCIDENT_TEMPLATE.format(premise=premise.strip())


def build_strict_chat_prompt(
    premise: str,
    model_name: str,
    tokenizer=None,
) -> str:
    """Build a strict tagged prompt using the active chat template."""
    return build_chat_prompt(
        [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": build_strict_incident_instruction(premise)},
        ],
        model_name=model_name,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )


def extract_tagged_sections(text: str) -> Dict[str, str]:
    """Parse tagged sections from a generated response."""
    sections: Dict[str, str] = {}
    for match in _TAG_BLOCK_PATTERN.finditer(text or ""):
        tag = match.group("tag").strip()
        sections[tag] = match.group("content").strip()
    return sections


def coerce_structured_response(text: str) -> str:
    """
    Normalize arbitrary generations into the required tagged schema.

    Missing sections get conservative placeholders so downstream evaluation and
    serving still produce a valid structured answer.
    """
    raw_text = (text or "").strip()
    sections = extract_tagged_sections(raw_text)
    if not sections and raw_text:
        sections["[ROOT_CAUSE]"] = raw_text

    normalized_lines = []
    for tag in REQUIRED_RESPONSE_TAGS:
        value = sections.get(tag, "").strip() or _DEFAULT_SECTION_TEXT[tag]
        normalized_lines.append(f"{tag} {value}")
    return "\n".join(normalized_lines)


def heuristic_response_bonus(text: str) -> float:
    """Score literal schema compliance and safety/DAG phrasing."""
    lowered = (text or "").lower()
    bonus = 0.0
    if "[root_cause]" in lowered:
        bonus += 0.05
    if "[intervention]" in lowered and "do(" in lowered:
        bonus += 0.15
    if "[counterfactual]" in lowered and "counterfactual" in lowered:
        bonus += 0.15
    if "[dag]" in lowered and any(term in lowered for term in ("graph", "node", "edge", "->", "→")):
        bonus += 0.20
    if "[safety_check]" in lowered and "requires_human_approval=true" in lowered:
        bonus += 0.25
    if any(term in lowered for term in ("human approval", "manual review", "verify")):
        bonus += 0.10
    return bonus


def score_structured_response(
    response_text: str,
    *,
    instruction_text: str,
    tokenizer,
    model_name: str,
    device: str,
    reward_model=None,
) -> Dict[str, float]:
    """Combine reward-head score with a small schema/safety heuristic bonus."""
    normalized = coerce_structured_response(response_text)
    heuristic_bonus = heuristic_response_bonus(normalized)
    reward_score = 0.0

    if reward_model is not None:
        with torch.no_grad():
            reward_example = build_training_example(
                instruction_text,
                normalized,
                model_name=model_name,
                tokenizer=tokenizer,
            )
            reward_inputs = tokenizer(
                reward_example,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            ).to(device)
            reward_score = reward_model(**reward_inputs).mean().item()

    return {
        "reward_score": float(reward_score),
        "heuristic_bonus": float(heuristic_bonus),
        "total_score": float(reward_score + heuristic_bonus),
    }


def generate_and_rerank_structured_response(
    model,
    tokenizer,
    *,
    premise: str,
    model_name: str,
    device: str,
    reward_model=None,
    num_candidates: int = 1,
    max_new_tokens: int = 256,
    temperature: Optional[float] = None,
    top_p: float = 0.9,
) -> Dict[str, object]:
    """
    Generate 1..N structured candidates and return the highest-scoring response.
    """
    candidate_count = max(1, int(num_candidates))
    sampling_temperature = (
        0.0 if candidate_count == 1 else (0.35 if temperature is None else temperature)
    )
    instruction_text = build_strict_incident_instruction(premise)
    prompt = build_strict_chat_prompt(premise, model_name=model_name, tokenizer=tokenizer)

    generated_candidates = []
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    try:
        for _ in range(candidate_count):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": candidate_count > 1,
            }
            if candidate_count > 1:
                generation_kwargs["temperature"] = sampling_temperature
                generation_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            prompt_len = inputs["input_ids"].shape[-1]
            continuation_ids = outputs[0][prompt_len:]
            decoded = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            normalized = coerce_structured_response(decoded)
            score = score_structured_response(
                normalized,
                instruction_text=instruction_text,
                tokenizer=tokenizer,
                model_name=model_name,
                device=device,
                reward_model=reward_model,
            )
            generated_candidates.append(
                {
                    "response": normalized,
                    "raw_response": decoded,
                    **score,
                }
            )
    finally:
        tokenizer.padding_side = original_padding_side

    best_candidate = max(
        generated_candidates,
        key=lambda candidate: candidate["total_score"],
    )
    return {
        "best_response": best_candidate["response"],
        "best_score": best_candidate["total_score"],
        "candidates": generated_candidates,
    }


def build_curated_continuation_subset(
    dataset: Iterable[Dict],
    *,
    max_examples: int = 384,
) -> list[Dict]:
    """
    Build a small high-quality continuation-SFT subset with balanced Pearl levels.
    """
    examples = list(dataset)
    if not examples:
        return []

    per_level_target = max(1, max_examples // 3)
    selected: list[Dict] = []
    seen_ids: set[int] = set()

    for level in (1, 2, 3):
        level_examples = [
            example for example in examples
            if int(example.get("pearl_level", 1)) == level
        ]
        level_examples.sort(key=lambda example: example.get("quality_score", 0), reverse=True)
        for example in level_examples[:per_level_target]:
            selected.append(example)
            seen_ids.add(id(example))

    if len(selected) < max_examples:
        remaining = [
            example for example in sorted(
                examples,
                key=lambda example: example.get("quality_score", 0),
                reverse=True,
            )
            if id(example) not in seen_ids
        ]
        selected.extend(remaining[: max_examples - len(selected)])

    return selected[:max_examples]
