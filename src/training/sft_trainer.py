"""
SRE-Nidaan: Supervised Fine-Tuning (SFT) Trainer
==================================================
Phase 1 of the NEXUS-CAUSAL SRE training pipeline.
Teaches the model the structure, vocabulary, and format of SRE causal analysis
using QLoRA on the configured instruct model.

Mirrors NEXUS-CAUSAL v3.1 src/training/sft_trainer.py with SRE-specific
special tokens and incident-domain instruction formatting.
"""

from typing import List, Dict
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from src.utils.model_utils import build_training_example
from src.utils.sre_schema import build_structured_training_response


# SRE-specific causal reasoning tokens
SRE_SPECIAL_TOKENS = [
    "[CAUSE]", "[EFFECT]", "[MECHANISM]", "[EVIDENCE]", "[REASONING]",
    "[CONCLUSION]", "[DOMAIN]", "[GRAPH]", "[COUNTERFACTUAL]",
    "[CONFOUNDERS]", "[PEARL_LEVEL]", "[INTERVENTION]", "[ASSOCIATION]",
    "[SPURIOUS]", "[MEDIATION]", "[NODE]", "[EDGE]", "[REVERSE_CAUSATION]",
    # SRE-Nidaan specific tokens
    "[INCIDENT]", "[TELEMETRY]", "[ROOT_CAUSE]", "[BLAST_RADIUS]",
    "[SAFETY_CHECK]", "[DAG]", "[REFUTATION]", "[REMEDIATION]",
    "[ESCALATION]", "[SLO_IMPACT]", "[RUNBOOK]",
]


class SRENexusSFT:
    """
    Handles the Supervised Fine-Tuning (SFT) phase for SRE causal reasoning.

    Extends the NEXUS-CAUSAL DataScaledNexusSFT with SRE-domain-specific
    instruction formatting and incident analysis tokens.
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        lora_config,
        model_name=None,
        strict_lexical_cues: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.lora_config = lora_config
        self.model_name = model_name or getattr(tokenizer, "name_or_path", "")
        self.strict_lexical_cues = strict_lexical_cues

    def setup_special_tokens(self):
        """Adds SRE causal reasoning tokens to the tokenizer and resizes embeddings."""
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": SRE_SPECIAL_TOKENS
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"🗣️  Added {len(SRE_SPECIAL_TOKENS)} special SRE causal tokens.")

    def setup_frontier_lora(self):
        """Applies LoRA configuration for efficient fine-tuning."""
        print("🔧 Setting up Frontier LoRA Configuration...")
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def format_instruction_data(self, dataset: List[Dict]) -> List[Dict]:
        """
        Formats raw SRE incident data into the active model's chat format.
        """
        formatted_data = []
        for ex in dataset:
            # Build the instruction (concise — no CausalCoT)
            domain = ex.get("domain", "unknown")
            pearl_level = ex.get("pearl_level", 1)
            premise = ex.get("premise", "")

            instruction = (
                f"You are an SRE causal reasoning agent. "
                f"Analyze the following incident using Pearl's Causal Hierarchy "
                f"(Level {pearl_level}).\n"
                f"- Domain: {domain}\n"
                f"- Incident: {premise}\n"
                f"Identify the structural root cause, explain why the naive "
                f"intervention is a confounding error, and produce a causal DAG."
            )
            if self.strict_lexical_cues:
                instruction += (
                    "\nReturn the answer using [ROOT_CAUSE], [INTERVENTION], "
                    "[COUNTERFACTUAL], [DAG], [REMEDIATION], and [SAFETY_CHECK]. "
                    "Use explicit do(...), counterfactual, graph/node/edge, and "
                    "human approval wording."
                )

            response = build_structured_training_response(
                ex,
                lexical_cues=self.strict_lexical_cues,
            )

            formatted_text = build_training_example(
                instruction,
                response,
                model_name=self.model_name,
                tokenizer=self.tokenizer,
            )
            formatted_data.append({"text": formatted_text})

        return formatted_data

    def train(self, train_data: List[Dict]):
        """Executes the SFT training process."""
        print("🚀 Starting SRE-Nidaan SFT Training...")
        train_formatted = self.format_instruction_data(train_data)
        train_dataset = Dataset.from_list(train_formatted)

        def tokenize(ex):
            return self.tokenizer(
                ex["text"],
                truncation=True,
                max_length=512,
                return_attention_mask=True,
            )

        tokenized_train_dataset = train_dataset.map(
            tokenize,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing SFT dataset",
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        print(f"🔥 Training on {len(train_formatted)} examples...")
        trainer.train()

        trainer.save_model(self.training_args.output_dir)
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        print(f"✅ SFT model saved to {self.training_args.output_dir}")
        return self.model, self.tokenizer
