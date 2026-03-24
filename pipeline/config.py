"""PipelineConfig: all hyperparameters for the gradient-free DE pipeline.

GPU is optional but accelerates ESM-2 OFS scoring and ProtGPT2 generation.
Defaults to CUDA if available, falls back to CPU automatically.

20% mutation cap:
    n_mutations=None triggers automatic budget: b = floor(max_mutation_fraction * L)
    where L is the sequence length. This caps the number of mutated sites to
    at most 20% of the chain length, preventing over-disruption while maximising
    attack coverage.
"""
from dataclasses import dataclass
import os
import torch


@dataclass
class PipelineConfig:
    # ── ESM-2 oracle (forward-pass only — no backprop) ─────────────────
    esm2_model_name: str = os.environ.get(
        "ESM2_MODEL_NAME", "facebook/esm2_t6_8M_UR50D"
    )  # 8 M params ~35 MB; swap for esm2_t12_35M_UR50D for stronger oracle

    # ── ProtGPT2 generation ──────────────────────────────────────────────
    protgpt2_model_path: str = os.environ.get("PROTGPT2_MODEL_PATH", "nferruz/ProtGPT2")
    n_generated: int = 50
    max_seq_len: int = 150
    top_k_generation: int = 950
    rep_penalty: float = 1.2
    perplexity_threshold: float = 15.0

    # ── Mutation budget ────────────────────────────────────────────────
    # n_mutations=None: auto-compute as floor(max_mutation_fraction * L) per sequence.
    # Set to a fixed int to override (e.g. n_mutations=5 always mutates exactly 5 sites).
    n_mutations: int = None
    max_mutation_fraction: float = 0.20   # hard cap: at most 20% of residues changed
    n_blosum_variants: int = 15           # more seed variants for broader coverage

    # ── Differential Evolution (higher intensity) ────────────────────────
    de_pop_size: int = 32       # doubled from 16 — wider search
    de_generations: int = 10    # more generations for deeper optimisation
    de_CR: float = 0.9          # crossover probability
    de_W: float = 0.8           # differential weight
    de_seed: int = 42

    # ── Attack target ────────────────────────────────────────────────────
    ppl_target: float = 25.0    # adversarial success: PPL >= this threshold
    top_k_attack: int = 5

    # ── General ──────────────────────────────────────────────────────────
    output_dir: str = "af3_attack_jobs"
    # Auto-detect: cuda if available, else cpu. Override with --device.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_budget(cfg: PipelineConfig, seq_len: int) -> int:
    """Compute the effective mutation budget b for a sequence of length seq_len.

    If cfg.n_mutations is set (not None), use it directly (still capped at 20%).
    Otherwise compute floor(max_mutation_fraction * seq_len), minimum 1.
    """
    cap = max(1, int(cfg.max_mutation_fraction * seq_len))
    if cfg.n_mutations is not None:
        return min(cfg.n_mutations, cap)
    return cap
