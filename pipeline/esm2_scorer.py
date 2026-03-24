"""ESM-2 OFS pseudo-perplexity scorer.

Uses the One-Forward-Sweep (OFS) approximation to compute masked-token
pseudo-perplexity from a single bidirectional forward pass over each
sequence position masked once:

    PPL_OFS(s) = exp( -1/L * sum_i log p(s_i | s\\i) )

This is a pure CPU, forward-only operation — no backpropagation, no GPU.
With facebook/esm2_t6_8M_UR50D (~35 MB) each sequence takes ~7 ms on a
modern laptop CPU.

Higher PPL  => sequence is evolutionarily implausible to the model
             => correlates with low predicted pLDDT (disrupted fold).
Attacking toward higher PPL is therefore a proxy for disrupting AF3 confidence.

Reference:
    Ferruz et al. 2025  "Pseudo-perplexity in one fell swoop"
    Schreiber 2024  HuggingFace blog: In-silico directed evolution with ESM-2
"""

import math
from typing import List

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from .config import PipelineConfig


class ESM2OracleScorer:
    """CPU-friendly ESM-2 OFS pseudo-perplexity oracle.

    Usage::

        scorer = ESM2OracleScorer(cfg)
        ppl = scorer.score("MKTAYIAKQRQ...")
        scores = scorer.score_batch([seq1, seq2, ...])
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        print(f"[ESM2] Loading {self.cfg.esm2_model_name} on {self.cfg.device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.esm2_model_name)
        self._model = EsmForMaskedLM.from_pretrained(self.cfg.esm2_model_name)
        self._model.eval().to(self.cfg.device)
        print("[ESM2] Model loaded")

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, seq: str) -> float:
        """Return OFS pseudo-perplexity for a single sequence (higher = more adversarial)."""
        self._load()
        tok = self._tokenizer
        model = self._model
        device = self.cfg.device

        encoding = tok(seq, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]  # (1, L+2)  includes [CLS] and [EOS]

        # Token positions that correspond to actual residues (exclude special tokens)
        # ESM tokenizer: position 0 = <cls>, positions 1..L = residues, L+1 = <eos>
        L = input_ids.shape[1] - 2  # number of residue tokens
        mask_token_id = tok.mask_token_id

        log_prob_sum = 0.0
        for i in range(1, L + 1):  # iterate over residue positions
            masked_ids = input_ids.clone()
            true_token = masked_ids[0, i].item()
            masked_ids[0, i] = mask_token_id

            logits = model(input_ids=masked_ids).logits  # (1, L+2, vocab)
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            log_prob_sum += log_probs[true_token].item()

        ppl = math.exp(-log_prob_sum / L)
        return ppl

    def score_batch(self, seqs: List[str]) -> List[float]:
        """Score a list of sequences; returns list of OFS pseudo-perplexity values."""
        scores = []
        for i, seq in enumerate(seqs):
            ppl = self.score(seq)
            if (i + 1) % 10 == 0:
                print(f"  [ESM2] Scored {i+1}/{len(seqs)}")
            scores.append(ppl)
        return scores

    # ------------------------------------------------------------------
    # Position sensitivity (replaces gradient_sensitivity)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def position_sensitivity(self, seq: str) -> List[float]:
        """Per-position contribution to PPL.

        Returns a list of length L where entry i is:
            -log p(s_i | s\\i)

        Higher values flag positions where the model is most "surprised" —
        these are the most evolutionarily unusual sites and the best targets
        for DE mutation vectors. Used to bias initial DE population.
        """
        self._load()
        tok = self._tokenizer
        model = self._model
        device = self.cfg.device

        encoding = tok(seq, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        L = input_ids.shape[1] - 2
        mask_token_id = tok.mask_token_id

        surprisals = []
        for i in range(1, L + 1):
            masked_ids = input_ids.clone()
            true_token = masked_ids[0, i].item()
            masked_ids[0, i] = mask_token_id
            logits = model(input_ids=masked_ids).logits
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            surprisals.append(-log_probs[true_token].item())

        return surprisals  # length L, higher = more sensitive / unusual
