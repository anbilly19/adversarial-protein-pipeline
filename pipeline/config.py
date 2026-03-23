from dataclasses import dataclass
import os
import torch


@dataclass
class PipelineConfig:
    # ── Model paths (set for air-gapped / HPC use) ──────────────────────
    esmfold_model_path: str = os.environ.get("ESMFOLD_MODEL_PATH", "facebook/esmfold_v1")
    protgpt2_model_path: str = os.environ.get("PROTGPT2_MODEL_PATH", "nferruz/ProtGPT2")
    esm_if1_checkpoint: str = os.environ.get("ESM_IF1_CHECKPOINT", None)  # path to .pt file

    # ProtGPT2 generation
    n_generated: int = 50
    max_seq_len: int = 150
    top_k_generation: int = 950
    rep_penalty: float = 1.2
    perplexity_threshold: float = 15.0

    # ESM-Design gradient attack
    esm_design_steps: int = 300
    esm_lr: float = 0.01
    esm_temp: float = 1.0
    esm_temp_final: float = 0.1
    plddt_target: float = 90.0

    # BLOSUM adversarial mutations
    n_mutations: int = 5
    n_blosum_variants: int = 10

    # ESM-IF1 inverse folding
    n_if_sequences: int = 20
    if_temperature: float = 1.5

    # General
    batch_size: int = 4
    top_k_attack: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "af3_attack_jobs"
        all_chains: bool = False  # Process all chains in PDB file
