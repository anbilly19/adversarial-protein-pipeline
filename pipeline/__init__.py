"""Adversarial Protein Pipeline package (gradient-free edition).

Modules:
    config          - PipelineConfig dataclass with all hyperparameters
    trick_sequences - Known adversarial sequences + BLOSUM62 substitution table
    protgpt2        - ProtGPT2 sequence generation and perplexity filtering
    esm2_scorer     - ESM-2 OFS pseudo-perplexity oracle (CPU, forward-only)
    de_attacker     - Differential Evolution adversarial attacker
    inverse_fold    - BLOSUMAttack (random + sensitivity-guided mutations)
    pdb_utils       - PDB structure reading utilities

Removed (required GPU / backprop):
    esmfold.py          - ESMFoldScorer, ESM-Design gradient attack
    InverseFoldingModule - ESM-IF1 inverse folding (requires biotite + torch_scatter)
"""

from .config import PipelineConfig
from .trick_sequences import (
    TRICK_SEQUENCES,
    BLOSUM62_SIMILAR,
    AA_VOCAB,
    AA_TO_IDX,
    get_all_trick_sequences,
    get_trick_sequences_by_category,
)
from .protgpt2 import ProtGPT2Generator
from .esm2_scorer import ESM2OracleScorer
from .de_attacker import DEAttacker
from .inverse_fold import BLOSUMAttack

__all__ = [
    "PipelineConfig",
    "TRICK_SEQUENCES",
    "BLOSUM62_SIMILAR",
    "AA_VOCAB",
    "AA_TO_IDX",
    "get_all_trick_sequences",
    "get_trick_sequences_by_category",
    "ProtGPT2Generator",
    "ESM2OracleScorer",
    "DEAttacker",
    "BLOSUMAttack",
]
