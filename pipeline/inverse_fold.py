"""BLOSUM62 adversarial mutations (gradient-free).

BLOSUMAttack now respects the same 20% mutation budget as DEAttacker
via resolve_budget(), so seed candidates generated here are consistent
with the DE search space.
"""

from typing import List, Dict, Optional
import numpy as np

from .config import PipelineConfig, resolve_budget
from .trick_sequences import BLOSUM62_SIMILAR


class BLOSUMAttack:
    """BLOSUM62-conservative adversarial mutations.

    All methods respect the 20% mutation budget via resolve_budget().
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
        """Generate n_variants by applying up to 20%-capped random BLOSUM62 substitutions."""
        b = resolve_budget(self.cfg, len(seq)) if n_mutations is None else min(
            n_mutations, resolve_budget(self.cfg, len(seq))
        )
        n_variants = n_variants or self.cfg.n_blosum_variants
        rng = np.random.default_rng(seed)

        variants = []
        seq_list = list(seq)
        mutable = [i for i, aa in enumerate(seq_list) if aa in BLOSUM62_SIMILAR]

        for _ in range(n_variants):
            mutant = seq_list.copy()
            positions = rng.choice(
                mutable, size=min(b, len(mutable)), replace=False
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
        """Apply BLOSUM62 mutations at highest ESM-2 surprisal positions.

        Budget b = floor(20% * L), so at most 20% of residues are changed.
        Position importance from ESM2OracleScorer.position_sensitivity() (OFS surprisal).

        Args:
            seq:        Amino acid sequence string
            surprisals: Per-position surprisal list from ESM2OracleScorer
            n_mutations: Override budget (still capped at 20%)
            n_variants:  Number of variant sequences to generate
            seed:        Random seed
        """
        b = resolve_budget(self.cfg, len(seq)) if n_mutations is None else min(
            n_mutations, resolve_budget(self.cfg, len(seq))
        )
        n_variants = n_variants or self.cfg.n_blosum_variants
        rng = np.random.default_rng(seed)

        seq_list = list(seq)
        ranked = np.argsort(surprisals)[::-1]
        top_positions = [
            pos for pos in ranked if seq_list[pos] in BLOSUM62_SIMILAR
        ][:b]

        print(
            f"[BLOSUM] b={b} sites ({b/len(seq)*100:.1f}% of chain) | "
            f"top positions: {top_positions[:10]}{'...' if len(top_positions)>10 else ''} | "
            f"surprisal: {[round(surprisals[p], 3) for p in top_positions[:5]]}"
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
