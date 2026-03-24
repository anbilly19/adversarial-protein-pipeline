"""PipelineConfig: all hyperparameters for the gradient-free DE pipeline.

GPU is optional but accelerates ESM-2 OFS scoring and ProtGPT2 generation.
Defaults to CUDA if available, falls back to CPU automatically.
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

    # ── BLOSUM seed candidates ────────────────────────────────────────────
    n_mutations: int = 3        # DE budget b: mutations per individual
    n_blosum_variants: int = 10

    # ── Differential Evolution ───────────────────────────────────────────
    de_pop_size: int = 16       # population size (must be >= 4)
    de_generations: int = 6     # number of generations
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
