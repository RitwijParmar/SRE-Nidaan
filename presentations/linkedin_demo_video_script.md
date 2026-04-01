# SRE Nidaan LinkedIn Demo Script (90 sec)

## Voiceover
When incidents hit, teams usually ask three questions: what broke first, what should we do now, and what should we definitely not touch.

I built SRE Nidaan to make that reasoning faster and safer.

This system runs as three services: a command interface, an orchestration and safety layer, and an LLM inference layer.

The model stack uses QLoRA SFT, reward modeling, and RLHF, with an MCP-inspired tool interface so incident reasoning can pull structured telemetry through controlled calls.

Here is a live run: we describe the incident context, run analysis, inspect the causal graph, review intervention logic, and keep human approval mandatory before actions.

The goal is not autopilot. The goal is a practical copilot that helps teams think clearly under pressure.

Early, imperfect but seems useful in practice.

Product and code are linked below.
