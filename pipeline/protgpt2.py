"""ProtGPT2 sequence generation and perplexity filtering.

ProtGPT2 is a 738M parameter GPT-2 decoder trained on UniRef50.
Perplexity under ProtGPT2 correlates with AF3 pLDDT (~0.6 Spearman),
making it a cheap pre-filter before the expensive gradient attack.
"""
import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer
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
        # ProtGPT2 uses the standard GPT2 tokenizer.
        # Use GPT2Tokenizer directly to avoid AutoTokenizer reading a
        # potentially empty/corrupted tokenizer_config.json from the weights dir.
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_path, local_files_only=local
            )
        except Exception:
            # Fallback: fetch tokenizer config from HuggingFace (tiny, no model weights)
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "nferruz/ProtGPT2", local_files_only=False
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=local
        ).to(cfg.device)
        self.model.eval()
        print(f"[ProtGPT2] Ready on {cfg.device}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, n: int = None, seed_text: str = "<|endoftext|>") -> List[str]:
        """Generate n protein sequences using ProtGPT2.

        Args:
            n: Number of sequences to generate (default: cfg.n_generated)
            seed_text: Prompt token to start generation
        Returns:
            List of raw generated sequences
        """
        n = n or self.cfg.n_generated
        inputs = self.tokenizer(seed_text, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.cfg.max_seq_len,
                do_sample=True,
                top_k=self.cfg.top_k_generation,
                repetition_penalty=self.cfg.rep_penalty,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        seqs = []
        for out in outputs:
            text = self.tokenizer.decode(out, skip_special_tokens=True)
            # Keep only uppercase amino acid characters
            cleaned = "".join(c for c in text.upper() if c in AA_TO_IDX)
            if cleaned:
                seqs.append(cleaned)
        return seqs

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    def perplexity(self, seq: str) -> float:
        """Compute ProtGPT2 perplexity for a sequence. Lower = more protein-like."""
        enc = self.tokenizer(seq, return_tensors="pt").to(self.cfg.device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            loss = self.model(input_ids, labels=input_ids).loss
        return torch.exp(loss).item()

    def filter_by_perplexity(
        self, seqs: List[str], threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Keep sequences with perplexity below threshold.

        Args:
            seqs: List of amino acid sequences
            threshold: Max allowed perplexity (default: cfg.perplexity_threshold)
        Returns:
            List of (sequence, perplexity) tuples, sorted by perplexity ascending
        """
        threshold = threshold or self.cfg.perplexity_threshold
        results = []
        for seq in seqs:
            ppl = self.perplexity(seq)
            if ppl < threshold:
                results.append((seq, ppl))
        return sorted(results, key=lambda x: x[1])

    def generate_and_filter(
        self, n: int = None, threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Generate sequences and filter by perplexity.

        Args:
            n: Number of sequences to generate
            threshold: Max perplexity to keep
        Returns:
            List of (sequence, perplexity) tuples that passed the filter
        """
        seqs = self.generate(n)
        return self.filter_by_perplexity(seqs, threshold)
