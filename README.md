# SRE-Nidaan — NEXUS-CAUSAL v3.1

[![Model](https://img.shields.io/badge/Model-Mistral--7B--Instruct--v0.2-orange.svg)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-NEXUS--CAUSAL_v3.1-green.svg)](https://github.com/RitwijParmar/NEXUS-CAUSAL-v3.1)

**Causal AI Site Reliability Engineering System**

An end-to-end distributed system that prevents catastrophic **"panic scaling"** in cloud-native environments by replacing standard correlational AI with formal causal reasoning (Pearl's *do*-calculus). Powered by the [NEXUS-CAUSAL v3.1](https://github.com/RitwijParmar/NEXUS-CAUSAL-v3.1.git) fine-tuned Mistral-7B model.

The system achieves **76% exact match accuracy** on the CLADDER benchmark with optimized prompting, outperforming GPT-4 (62%) using a 7B parameter model.

---

## 🎯 The Problem

Standard AI-driven SRE tools use correlational analysis to generate alerts and auto-scaling decisions. When a database connection pool fills up, they naively scale upstream services — which **worsens** the outage by opening more connections. This "panic scaling" is a confounding error that Pearl's Causal Hierarchy can structurally prevent.

---

## ✨ Our Solution: The NEXUS-CAUSAL SRE Pipeline

We apply the [NEXUS-CAUSAL v3.1](https://github.com/RitwijParmar/NEXUS-CAUSAL-v3.1.git) three-phase training pipeline to the domain of Site Reliability Engineering:

1. **🧠 Phase 1: Supervised Fine-Tuning (SFT)**
   - QLoRA fine-tuning on **2,500+ SRE causal incident examples** across 12 infrastructure domains
   - Custom SRE special tokens: `[ROOT_CAUSE]`, `[BLAST_RADIUS]`, `[SAFETY_CHECK]`, `[DAG]`, etc.

2. **🏆 Phase 2: Multi-Dimensional Reward Modeling**
   - 7-dimensional reward model scoring: structural accuracy, root cause precision, confounder detection, counterfactual quality, DAG completeness, **blast radius awareness**, **safety compliance**

3. **🚀 Phase 3: Pearl's Ladder RLHF**
   - Curriculum learning: L1 (Association) → L2 (Intervention) → L3 (Counterfactual)
   - Safety-aware reward shaping: bonus for enforcing `requires_human_approval`

---

## 📊 Dataset: 12 SRE Infrastructure Domains

| Domain | Pearl Level | Example Incident |
|---|---|---|
| Kubernetes | L2 | Pod eviction cascade from HPA memory contention |
| Database | L2 | Connection pool exhaustion from auth retry storm |
| Networking | L1 | Conntrack table overflow causing TCP RSTs |
| Load Balancer | L2 | Thread exhaustion from slow downstream dependency |
| DNS | L1 | ndots misconfiguration causing query amplification |
| Storage | L2 | EBS burst credit depletion during compaction |
| Authentication | L3 | JWKS thundering herd from key rotation |
| Message Queue | L2 | Kafka rebalance storm from large poll batches |
| Cache | L2 | Redis eviction from no-TTL preference objects |
| CI/CD | L3 | Build cache invalidation from dependency bump |
| Monitoring | L1 | Prometheus cardinality explosion from unbounded labels |
| API Gateway | L2 | Noisy client consuming global rate limit quota |

---

## 🏗️ Architecture (Split-Compute)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SRE-Nidaan Architecture                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐  │
│  │  THE FACE   │────▶│    THE BODY      │────▶│   THE BRAIN      │  │
│  │  Next.js 14 │◀────│    FastAPI       │◀────│   vLLM + LoRA    │  │
│  │  Port 3000  │     │    Port 8001     │     │   Port 8000      │  │
│  └─────────────┘     └──────────────────┘     └──────────────────┘  │
│   React Flow +        Pydantic Schema         Mistral-7B-Instruct  │
│   Dagre Layout        Safety Plane             + NEXUS-CAUSAL LoRA  │
│   Tailwind CSS        MCP Router               4-bit Quantization   │
│                       Refutation Tests         max_lora_rank=64     │
│                                                                      │
│  ═══════════════════ DATA FLOW ═════════════════════════════════════ │
│                                                                      │
│  1. Frontend sends POST /api/analyze-incident                        │
│  2. Backend fetches telemetry → builds Mistral [INST] prompt         │
│  3. Prompt sent to vLLM with Pydantic guided_json decoding           │
│  4. LLM returns causal DAG (root cause + intervention simulation)    │
│  5. Backend returns JSON with requires_human_approval: true          │
│  6. Frontend renders DAG via Dagre + shows safety gate button        │
│  7. Human operator must click "Authorize Intervention" to proceed    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
SRE-Nidaan/
├── config.py                       # Centralized pipeline configuration
├── requirements.txt
├── scripts/                        # Sequential pipeline execution
│   ├── 01_generate_dataset.py      # Generate 2,500 SRE incident examples
│   ├── 02_run_sft.py               # QLoRA supervised fine-tuning
│   ├── 03_train_reward_model.py    # 7D reward model training
│   ├── 04_run_rlhf.py             # Pearl's Ladder RLHF
│   └── 05_run_evaluation.py       # Final model evaluation
├── src/                            # Core library
│   ├── data/
│   │   └── dataset_generator.py    # Massive SRE dataset generator
│   ├── training/
│   │   ├── sft_trainer.py          # SFT with SRE special tokens
│   │   ├── reward_modeler.py       # 7D reward model (blast radius + safety)
│   │   └── rlhf_trainer.py         # Pearl's Ladder curriculum RLHF
│   ├── evaluation/
│   │   └── evaluator.py            # Multi-level evaluation framework
│   └── utils/
│       └── model_utils.py          # Model loading & prompt formatting
├── inference_server.py             # The Brain: vLLM + LoRA inference
├── backend/
│   └── main.py                     # The Body: FastAPI + Safety Plane
├── frontend/                       # The Face: Next.js 14 Dashboard
│   ├── src/app/page.tsx
│   ├── src/components/CausalGraph.tsx
│   ├── tailwind.config.ts
│   └── package.json
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.9+** with CUDA support
- **NVIDIA GPU with ≥16 GB VRAM** (for training; Phases 2–3 backend/frontend work on CPU)
- **Node.js 18+** (for frontend)

### 1. Clone & Install

```bash
git clone https://github.com/RitwijParmar/SRE-Nidaan.git
cd SRE-Nidaan
pip install -r requirements.txt
```

### 2. Training Pipeline (Colab / GPU)

Run the five phases sequentially:

```bash
# Phase 1: Generate 2,500 SRE incident training examples
python scripts/01_generate_dataset.py

# Phase 2: QLoRA Supervised Fine-Tuning (~2-4 hours)
python scripts/02_run_sft.py

# Phase 3: Train 7D Reward Model (~30-60 minutes)
python scripts/03_train_reward_model.py

# Phase 4: Pearl's Ladder RLHF (~45-90 minutes)
python scripts/04_run_rlhf.py

# Phase 5: Evaluate on SRE test suite
python scripts/05_run_evaluation.py
```

### 3. Inference Server (The Brain)

```bash
export NGROK_AUTHTOKEN="your-token"
python inference_server.py
```

### 4. Backend (The Body)

```bash
export VLLM_ENDPOINT="http://localhost:8000/v1"
cd backend && uvicorn main:app --port 8001 --reload
```

### 5. Frontend (The Face)

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000** ← Dark-mode SRE dashboard with causal DAG visualization.

---

## 🛠️ Key Design Decisions

| Decision | Rationale |
|---|---|
| `max_lora_rank=64` | Caps GPU memory during adapter swaps; prevents `cudaMemcpyAsync` latency spikes |
| Pydantic `guided_json` | Deterministic JSON generation — never relies on prompt alone |
| Mistral `[INST]`/`[/INST]` | Required by Mistral-7B-Instruct-v0.2 tokenizer |
| Concise prompts (no CausalCoT) | Optimized Prompting: 76% CLADDER vs CausalCoT 41% |
| Read-only copilot pattern | Backend never auto-executes — human must authorize |
| 7D reward model | Adds blast_radius_awareness + safety_compliance to NEXUS-CAUSAL's 7 dimensions |
| Background refutation tests | Validates causal estimates with placebo confounders |
| 2,500 training examples | 12 SRE domains × parameterized templates with Pearl's level labels |

---

## 🛡️ Hardware Requirements

| Phase | GPU | Time |
|---|---|---|
| Data Generation | CPU only | ~1 minute |
| SFT Training | ≥16 GB VRAM | ~2-4 hours |
| Reward Model | ≥16 GB VRAM | ~30-60 min |
| RLHF Training | ≥16 GB VRAM | ~45-90 min |
| Inference (vLLM) | ≥16 GB VRAM | Continuous |
| Backend (FastAPI) | CPU | Continuous |
| Frontend (Next.js) | CPU | Continuous |

---

## 📖 NEXUS-CAUSAL v3.1

**Repository:** https://github.com/RitwijParmar/NEXUS-CAUSAL-v3.1.git

- **Base Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Training:** 3-phase pipeline — QLoRA SFT → Reward Modeling → Pearl's Ladder RLHF
- **Performance:** 76.0% accuracy on CLADDER benchmark (Optimized Prompting)

---

## License

Apache 2.0
