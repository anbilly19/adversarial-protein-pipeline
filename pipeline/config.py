"""PipelineConfig: all hyperparameters for the gradient-free DE pipeline.

No GPU required. ESM-2 8M runs comfortably on CPU (~7 ms / sequence).
"""
from dataclasses import dataclass, field
import os


@dataclass
class PipelineConfig:
    # ── ESM-2 oracle (CPU, forward-pass only) ───────────────────────────
    esm2_model_name: str = os.environ.get(
        "ESM2_MODEL_NAME", "facebook/esm2_t6_8M_UR50D"
    )  # 8 M params, ~35 MB, runs on CPU in ~7 ms/seq

    # ── ProtGPT2 generation ──────────────────────────────────────────────
    protgpt2_model_path: str = os.environ.get("PROTGPT2_MODEL_PATH", "nferruz/ProtGPT2")
    n_generated: int = 50
    max_seq_len: int = 150
    top_k_generation: int = 950
    rep_penalty: float = 1.2
    perplexity_threshold: float = 15.0

    # ── BLOSUM random mutations (seed candidates) ────────────────────────
    n_mutations: int = 3        # DE budget b (mutations per individual)
    n_blosum_variants: int = 10

    # ── Differential Evolution ───────────────────────────────────────────
    de_pop_size: int = 16       # population size (must be >= 4)
    de_generations: int = 6     # number of generations
    de_CR: float = 0.9          # crossover probability
    de_W: float = 0.8           # differential weight
    de_seed: int = 42

    # ── Attack target ────────────────────────────────────────────────────
    ppl_target: float = 25.0    # adversarial success: PPL >= this threshold
    top_k_attack: int = 5       # number of seed candidates to attack

    # ── General ──────────────────────────────────────────────────────────
    output_dir: str = "af3_attack_jobs"
    device: str = "cpu"         # intentionally CPU-only
