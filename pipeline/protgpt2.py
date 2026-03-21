"""ProtGPT2 sequence generation and perplexity filtering.

ProtGPT2 is a 738M parameter GPT-2 decoder trained on UniRef50.
Perplexity under ProtGPT2 correlates with AF3 pLDDT (~0.6 Spearman),
making it a cheap pre-filter before the expensive gradient attack.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from .config import PipelineConfig
from .trick_sequences import AA_TO_IDX



class ProtGPT2Generator:
    """Wraps ProtGPT2 for sequence generation and perplexity scoring."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        model_path = cfg.protgpt2_model_path
        local = "/" in model_path and not model_path.startswith("nferruz/")
        print(f"[ProtGPT2] Loading from {'local path' if local else 'HuggingFace'}: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=local
        ).to(cfg.device)

    def generate(self, n: int = None) -> List[str]:
        """Generate n raw amino acid sequences.

        Uses the <|endoftext|> BOS token to start fresh proteins.
        Recommended sampling params from the authors:
            top_k=950, repetition_penalty=1.2
        """
        n = n or self.cfg.n_generated
        bos = self.tokenizer.encode("<|endoftext|>", return_tensors="pt").to(self.cfg.device)

        with torch.no_grad():
            outputs = self.model.generate(
                bos,
                max_length=self.cfg.max_seq_len + 10,
                do_sample=True,
                top_k=self.cfg.top_k_generation,
                repetition_penalty=self.cfg.rep_penalty,
                num_return_sequences=n,
                eos_token_id=0,
                pad_token_id=0,
            )

        seqs = []
        for ids in outputs:
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            clean = "".join(c for c in decoded.upper() if c in AA_TO_IDX)
            if 50 <= len(clean) <= self.cfg.max_seq_len:
                seqs.append(clean)
        return seqs

    def perplexity(self, seq: str) -> float:
        """Compute ProtGPT2 perplexity for a single sequence.

        Lower perplexity = more natural-looking to the model.
        Correlates with ESMFold pLDDT (~0.6 Spearman) per the original paper.
        """
        tokens = self.tokenizer(seq, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            loss = self.model(**tokens, labels=tokens["input_ids"]).loss
        return torch.exp(loss).item()

    def filter_by_perplexity(
        self, seqs: List[str], threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Return (seq, perplexity) pairs that pass the threshold, sorted ascending."""
        threshold = threshold or self.cfg.perplexity_threshold
        scored = [(s, self.perplexity(s)) for s in seqs]
        passed = [(s, p) for s, p in scored if p < threshold]
        passed.sort(key=lambda x: x[1])
        print(
            f"[ProtGPT2] Perplexity filter: {len(seqs)} -> {len(passed)} "
            f"(threshold={threshold:.1f})"
        )
        return passed

    def generate_and_filter(self) -> List[Tuple[str, float]]:
        """Full Stage 1+2: generate sequences then filter by perplexity."""
        print(f"[ProtGPT2] Generating {self.cfg.n_generated} sequences...")
        raw = self.generate(self.cfg.n_generated)
        print(f"[ProtGPT2] {len(raw)} valid sequences generated")
        return self.filter_by_perplexity(raw)
