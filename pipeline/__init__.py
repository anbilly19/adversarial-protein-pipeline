"""Adversarial Protein Pipeline package.

Modules:
    config               - PipelineConfig dataclass with all hyperparameters
    trick_sequences      - Known adversarial sequences + BLOSUM62 substitution table
    protgpt2             - ProtGPT2 sequence generation and perplexity filtering
    esmfold              - ESMFold scoring and ESM-Design gradient attack
    inverse_fold         - ESM-IF1 inverse folding and BLOSUM62 adversarial mutations
    evolutionary_attack  - Gradient-free evolutionary attack (AF2-Mutation-inspired)
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
from .esmfold import ESMFoldScorer
from .inverse_fold import InverseFoldingModule, BLOSUMAttack
from .evolutionary_attack import EvolutionaryAttack

__all__ = [
    "PipelineConfig",
    "TRICK_SEQUENCES",
    "BLOSUM62_SIMILAR",
    "AA_VOCAB",
    "AA_TO_IDX",
    "get_all_trick_sequences",
    "get_trick_sequences_by_category",
    "ProtGPT2Generator",
    "ESMFoldScorer",
    "InverseFoldingModule",
    "BLOSUMAttack",
    "EvolutionaryAttack",
]
