"""Gradient-free Differential Evolution (DE) adversarial attacker.

Replaces the ESM-Design gradient attack with a black-box DE loop that
uses ESM-2 OFS pseudo-perplexity as its fitness signal.

DE strategy: DE/rand/1/bin  (standard, robust, no gradients)
  - Mutation:   v = x_a + W * (x_b - x_c)
  - Crossover:  trial_j = v_j if rand < CR else x_j
  - Selection:  keep trial if fitness(trial) > fitness(parent)

Mutation budget:
    b = floor(max_mutation_fraction * L)  per sequence at runtime.
    This hard-caps changed residues at 20% of chain length.
    Overridden by cfg.n_mutations if explicitly set (still capped at 20%).

Fitness = ESM-2 OFS pseudo-perplexity (higher = more adversarial).

Reference:
    AF2-Mutation §3.2 (Xu et al. 2023, arXiv:2305.08929)
    DE: Storn & Price 1997, J. Global Optimization
"""

import random
from typing import List, Dict, Optional

import numpy as np

from .config import PipelineConfig, resolve_budget
from .trick_sequences import BLOSUM62_SIMILAR, AA_VOCAB
from .esm2_scorer import ESM2OracleScorer


def _apply_mutations(seq: str, mutations: List[tuple]) -> str:
    """Apply a list of (position, new_aa) mutations to a sequence string."""
    s = list(seq)
    for pos, aa in mutations:
        if 0 <= pos < len(s):
            s[pos] = aa
    return "".join(s)


def _count_changes(seq_a: str, seq_b: str) -> int:
    """Count number of positions that differ between two equal-length sequences."""
    return sum(a != b for a, b in zip(seq_a, seq_b))


def _random_blosum_mutation(seq: str, rng: np.random.Generator) -> tuple:
    """Sample a single random BLOSUM62-conservative (position, substitution) pair."""
    mutable = [i for i, aa in enumerate(seq) if aa in BLOSUM62_SIMILAR]
    if not mutable:
        pos = rng.integers(0, len(seq))
        aa = rng.choice(AA_VOCAB)
        return (int(pos), aa)
    pos = int(rng.choice(mutable))
    aa = rng.choice(BLOSUM62_SIMILAR[seq[pos]])
    return (pos, aa)


class Individual:
    """One DE individual: a list of b (position, aa) mutation tuples + its mutated sequence.

    Enforces deduplication of positions so b unique sites are always mutated.
    """

    def __init__(self, seq: str, mutations: List[tuple]):
        self.base_seq = seq
        # Deduplicate: last mutation wins for any repeated position
        seen = {}
        for pos, aa in mutations:
            seen[pos] = aa
        self.mutations = list(seen.items())  # list of (int pos, str aa)
        self.seq = _apply_mutations(seq, self.mutations)
        self.fitness: Optional[float] = None

    def score(self, scorer: ESM2OracleScorer) -> float:
        if self.fitness is None:
            self.fitness = scorer.score(self.seq)
        return self.fitness


class DEAttacker:
    """Differential Evolution adversarial attacker with 20% mutation budget cap.

    Args:
        cfg:         PipelineConfig
        scorer:      ESM2OracleScorer instance (shared across calls)
        sensitivity: Optional per-position surprisal list from ESM2OracleScorer.
                     Biases population initialisation toward high-surprisal sites.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        scorer: ESM2OracleScorer,
        sensitivity: Optional[List[float]] = None,
    ):
        self.cfg = cfg
        self.scorer = scorer
        self.sensitivity = sensitivity
        self.rng = np.random.default_rng(cfg.de_seed)

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _init_individual(self, seq: str, b: int) -> Individual:
        """Sample b unique mutations to form one individual.

        Positions are sampled proportional to ESM-2 surprisal when available,
        otherwise uniformly over BLOSUM62-mutable sites.
        Budget b is pre-computed per sequence (20% cap enforced by caller).
        """
        L = len(seq)

        if self.sensitivity is not None and len(self.sensitivity) == L:
            surp = np.array(self.sensitivity, dtype=float)
            surp = np.clip(surp, 0, None)
            total = surp.sum()
            probs = surp / total if total > 0 else None
        else:
            probs = None

        mutations = []
        positions_used = set()
        attempts = 0
        while len(mutations) < b and attempts < b * 20:
            attempts += 1
            if probs is not None:
                pos = int(self.rng.choice(L, p=probs))
            else:
                mutable = [i for i, aa in enumerate(seq) if aa in BLOSUM62_SIMILAR]
                pos = int(self.rng.choice(mutable)) if mutable else int(self.rng.integers(0, L))
            if pos in positions_used:
                continue
            original_aa = seq[pos]
            if original_aa in BLOSUM62_SIMILAR:
                new_aa = self.rng.choice(BLOSUM62_SIMILAR[original_aa])
            else:
                new_aa = self.rng.choice(AA_VOCAB)
            mutations.append((pos, new_aa))
            positions_used.add(pos)

        return Individual(seq, mutations)

    def _init_population(self, seq: str, b: int) -> List[Individual]:
        pop = [self._init_individual(seq, b) for _ in range(self.cfg.de_pop_size)]
        print(f"  [DE] Scoring initial population ({len(pop)} individuals, b={b} sites)...")
        for ind in pop:
            ind.score(self.scorer)
        return pop

    # ------------------------------------------------------------------
    # DE operators (discrete sequence space)
    # ------------------------------------------------------------------

    def _mutate_crossover(
        self,
        target: Individual,
        pop: List[Individual],
        seq: str,
        b: int,
    ) -> Individual:
        """DE/rand/1/bin mutation + crossover in discrete (pos, AA) space.

        For each mutation slot j:
          - With prob CR or j==j_rand: inherit from donor a;
            if b_ind and c_ind disagree at slot j, resample a new BLOSUM mutation
            (this is the discrete analogue of W*(x_b - x_c)).
          - Otherwise: inherit from target.
        New individual is capped at b unique positions (20% cap maintained).
        """
        CR = self.cfg.de_CR
        W = self.cfg.de_W

        candidates = [ind for ind in pop if ind is not target]
        a, b_ind, c_ind = random.sample(candidates, 3)

        trial_muts = list(target.mutations)
        j_rand = self.rng.integers(0, max(len(trial_muts), 1))

        for j in range(len(trial_muts)):
            if self.rng.random() < CR or j == j_rand:
                if j < len(a.mutations):
                    pos_a, aa_a = a.mutations[j]
                    if j < len(b_ind.mutations) and j < len(c_ind.mutations):
                        _, aa_b = b_ind.mutations[j]
                        _, aa_c = c_ind.mutations[j]
                        if aa_b != aa_c and self.rng.random() < abs(W):
                            trial_muts[j] = _random_blosum_mutation(seq, self.rng)
                            continue
                    trial_muts[j] = (pos_a, aa_a)
                else:
                    trial_muts[j] = _random_blosum_mutation(seq, self.rng)

        # Enforce b-slot cap: deduplicate and truncate to b
        seen = {}
        for pos, aa in trial_muts:
            seen[pos] = aa
        trial_muts = list(seen.items())[:b]

        # Pad back to b if deduplication shrunk the list
        while len(trial_muts) < b:
            trial_muts.append(_random_blosum_mutation(seq, self.rng))

        return Individual(seq, trial_muts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def attack(self, seq: str) -> Dict:
        """Run DE adversarial attack on a single sequence.

        The mutation budget b = floor(20% * L) is computed here and
        propagated to all population and crossover operations.

        Returns a result dict with keys:
            orig_seq, attacked_seq, orig_ppl, final_ppl,
            attack_success, mutations, n_mutations, mutation_fraction
        """
        L = len(seq)
        b = resolve_budget(self.cfg, L)
        print(f"  [DE] Sequence length: {L} | budget b={b} sites ({b/L*100:.1f}% of chain)")

        orig_ppl = self.scorer.score(seq)
        print(f"  [DE] Base PPL: {orig_ppl:.2f}")

        pop = self._init_population(seq, b)
        best = max(pop, key=lambda x: x.fitness)

        for gen in range(self.cfg.de_generations):
            new_pop = []
            for target in pop:
                trial = self._mutate_crossover(target, pop, seq, b)
                trial.score(self.scorer)
                new_pop.append(trial if trial.fitness > target.fitness else target)

            pop = new_pop
            best = max(pop, key=lambda x: x.fitness)
            n_changed = _count_changes(seq, best.seq)
            print(
                f"  [DE] Gen {gen+1:>2}/{self.cfg.de_generations} "
                f"| best PPL: {best.fitness:.2f} "
                f"| sites changed: {n_changed}/{L} ({n_changed/L*100:.1f}%) "
                f"| target PPL: {self.cfg.ppl_target:.1f}"
            )

            if best.fitness >= self.cfg.ppl_target:
                print(f"  [DE] ✓ Target PPL reached at generation {gen+1}!")
                break

        n_changed_final = _count_changes(seq, best.seq)
        return {
            "orig_seq": seq,
            "attacked_seq": best.seq,
            "orig_ppl": orig_ppl,
            "final_ppl": best.fitness,
            "attack_success": best.fitness >= self.cfg.ppl_target,
            "mutations": best.mutations,
            "n_mutations": n_changed_final,
            "mutation_fraction": n_changed_final / L,
        }
