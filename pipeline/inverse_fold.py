"""BLOSUM62 adversarial mutations (gradient-free).

InverseFoldingModule (ESM-IF1) has been removed — it required GPU,
biotite, and torch_scatter. This module now only exposes BLOSUMAttack
with its random and exhaustive mutation strategies. Position ranking
is done externally via ESM2OracleScorer.position_sensitivity().

Reference:
    Alkhouri et al. 2024 (gradient-guided BLOSUM attack — we adapt
    to gradient-free sensitivity proxy from ESM-2 OFS surprisal)
"""

from typing import List, Dict, Optional
import numpy as np

from .config import PipelineConfig
from .trick_sequences import BLOSUM62_SIMILAR


class BLOSUMAttack:
    """BLOSUM62-conservative adversarial mutations.

    Two strategies:
        random:    pick n_mutations random substitutable positions
        exhaustive: all possible single conservative point mutations

    Position ranking (previously done by gradient_sensitivity) is now
    provided externally as a surprisal list from ESM2OracleScorer.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def random_mutations(
        self,
        seq: str,
        n_mutations: int = None,
        n_variants: int = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate n_variants by applying n_mutations random BLOSUM62 substitutions."""
        n_mutations = n_mutations or self.cfg.n_mutations
        n_variants = n_variants or self.cfg.n_blosum_variants
        rng = np.random.default_rng(seed)

        variants = []
        seq_list = list(seq)
        mutable = [i for i, aa in enumerate(seq_list) if aa in BLOSUM62_SIMILAR]

        for _ in range(n_variants):
            mutant = seq_list.copy()
            positions = rng.choice(
                mutable, size=min(n_mutations, len(mutable)), replace=False
            )
            for pos in positions:
                aa = mutant[pos]
                mutant[pos] = rng.choice(BLOSUM62_SIMILAR[aa])
            variants.append("".join(mutant))
        return variants

    def sensitivity_guided_mutations(
        self,
        seq: str,
        surprisals: List[float],
        n_mutations: int = None,
        n_variants: int = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Apply BLOSUM62 mutations at positions with highest ESM-2 surprisal.

        Replaces gradient_guided_mutations: position importance is derived
        from ESM2OracleScorer.position_sensitivity() (OFS surprisal), which
        is a forward-only CPU operation instead of gradient backpropagation.

        Args:
            seq:        Amino acid sequence string
            surprisals: Per-position surprisal list from ESM2OracleScorer.position_sensitivity()
            n_mutations: Number of positions to mutate
            n_variants:  Number of variant sequences to generate
            seed:        Random seed

        Returns:
            List of mutant sequences
        """
        n_mutations = n_mutations or self.cfg.n_mutations
        n_variants = n_variants or self.cfg.n_blosum_variants
        rng = np.random.default_rng(seed)

        seq_list = list(seq)
        ranked = np.argsort(surprisals)[::-1]
        top_positions = [
            pos for pos in ranked if seq_list[pos] in BLOSUM62_SIMILAR
        ][:n_mutations]

        print(
            f"[BLOSUM] Top {len(top_positions)} high-surprisal positions: "
            f"{top_positions} | surprisal: {[round(surprisals[p], 3) for p in top_positions]}"
        )

        variants = []
        for _ in range(n_variants):
            mutant = seq_list.copy()
            for pos in top_positions:
                aa = mutant[pos]
                mutant[pos] = rng.choice(BLOSUM62_SIMILAR[aa])
            variants.append("".join(mutant))
        return variants

    def exhaustive_single_mutations(self, seq: str) -> List[Dict]:
        """Generate all possible single BLOSUM62-conservative point mutations."""
        mutations = []
        for i, aa in enumerate(seq):
            if aa in BLOSUM62_SIMILAR:
                for sub in BLOSUM62_SIMILAR[aa]:
                    mutant = list(seq)
                    mutant[i] = sub
                    mutations.append({
                        "seq": "".join(mutant),
                        "name": f"blosum_{i}{aa}{sub}",
                        "source": "blosum_exhaustive",
                        "position": i,
                        "original": aa,
                        "mutation": sub,
                    })
        return mutations
