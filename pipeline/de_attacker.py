"""Gradient-free Differential Evolution (DE) adversarial attacker.

Replaces the ESM-Design gradient attack with a black-box DE loop that
uses ESM-2 OFS pseudo-perplexity as its fitness signal.

DE strategy: DE/rand/1/bin  (standard, robust, no gradients)
  - Mutation:   v = x_a + W * (x_b - x_c)
  - Crossover:  trial_j = v_j if rand < CR else x_j
  - Selection:  keep trial if fitness(trial) > fitness(parent)

The solution vector encodes a set of b (position, substitution) pairs
for a given protein sequence. Crossover operates directly over these
discrete (position, AA) choices, adapted to discrete sequence space.

Fitness = ESM-2 OFS pseudo-perplexity (higher = more adversarial).

Reference:
    AF2-Mutation §3.2 (Xu et al. 2023, arXiv:2305.08929)
    DE: Storn & Price 1997, J. Global Optimization
"""

import random
from typing import List, Dict, Optional

import numpy as np

from .config import PipelineConfig
from .trick_sequences import BLOSUM62_SIMILAR, AA_VOCAB
from .esm2_scorer import ESM2OracleScorer


def _apply_mutations(seq: str, mutations: List[tuple]) -> str:
    """Apply a list of (position, new_aa) mutations to a sequence string."""
    s = list(seq)
    for pos, aa in mutations:
        if 0 <= pos < len(s):
            s[pos] = aa
    return "".join(s)


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
    """One DE individual: a list of b (position, aa) mutation tuples + its mutated sequence."""

    def __init__(self, seq: str, mutations: List[tuple]):
        self.base_seq = seq
        self.mutations = mutations  # list of (int pos, str aa)
        self.seq = _apply_mutations(seq, mutations)
        self.fitness: Optional[float] = None  # set after scoring

    def score(self, scorer: ESM2OracleScorer) -> float:
        if self.fitness is None:
            self.fitness = scorer.score(self.seq)
        return self.fitness


class DEAttacker:
    """Differential Evolution adversarial attacker.

    Args:
        cfg:            PipelineConfig (uses de_pop_size, de_generations, de_CR, de_W,
                        n_mutations, ppl_target, de_seed)
        scorer:         ESM2OracleScorer instance (shared across calls)
        sensitivity:    Optional per-position surprisal list from ESM2OracleScorer.
                        If provided, biases initial population toward high-sensitivity sites.
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

    def _init_individual(self, seq: str) -> Individual:
        """Sample b mutations to form one individual.

        If sensitivity scores are available, positions are sampled with
        probability proportional to their surprisal (higher surprisal =
        more unusual site = better attack target).
        """
        b = self.cfg.n_mutations
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
        while len(mutations) < b and attempts < b * 10:
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

    def _init_population(self, seq: str) -> List[Individual]:
        pop = [self._init_individual(seq) for _ in range(self.cfg.de_pop_size)]
        print(f"  [DE] Scoring initial population ({len(pop)} individuals)...")
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
    ) -> Individual:
        """DE/rand/1/bin mutation + crossover in discrete (pos, AA) space.

        Interpretation in discrete space:
            - Select 3 distinct individuals a, b, c != target
            - For each mutation slot j in [0, b):
                * With prob CR (or if j == j_rand): take mutation from individual_a
                  perturbed by the "difference" of (b - c), i.e. if b and c differ
                  at slot j, resample a new random BLOSUM mutation at that slot;
                  otherwise inherit individual_a's mutation.
                * Otherwise: inherit target's mutation at slot j.
        """
        b = self.cfg.n_mutations
        CR = self.cfg.de_CR
        W = self.cfg.de_W

        candidates = [ind for ind in pop if ind is not target]
        a, b_ind, c_ind = random.sample(candidates, 3)

        trial_muts = list(target.mutations)  # start from target
        j_rand = self.rng.integers(0, max(b, 1))

        for j in range(len(trial_muts)):
            if self.rng.random() < CR or j == j_rand:
                # Take from a; if b_ind and c_ind differ at slot j, perturb
                if j < len(a.mutations):
                    pos_a, aa_a = a.mutations[j]
                    if j < len(b_ind.mutations) and j < len(c_ind.mutations):
                        _, aa_b = b_ind.mutations[j]
                        _, aa_c = c_ind.mutations[j]
                        if aa_b != aa_c:  # "difference" is non-zero: perturb
                            if self.rng.random() < abs(W):
                                new_mut = _random_blosum_mutation(seq, self.rng)
                                trial_muts[j] = new_mut
                                continue
                    trial_muts[j] = (pos_a, aa_a)
                else:
                    trial_muts[j] = _random_blosum_mutation(seq, self.rng)

        return Individual(seq, trial_muts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def attack(self, seq: str) -> Dict:
        """Run DE adversarial attack on a single sequence.

        Returns a result dict compatible with the run_pipeline.py summary
        printer (keys: orig_seq, attacked_seq, orig_ppl, final_ppl,
        attack_success, source, name).
        """
        orig_ppl = self.scorer.score(seq)
        print(f"  [DE] Base PPL: {orig_ppl:.2f}")

        pop = self._init_population(seq)
        best = max(pop, key=lambda x: x.fitness)

        for gen in range(self.cfg.de_generations):
            new_pop = []
            for target in pop:
                trial = self._mutate_crossover(target, pop, seq)
                trial.score(self.scorer)
                # Selection: maximise PPL (adversarial direction)
                new_pop.append(trial if trial.fitness > target.fitness else target)

            pop = new_pop
            best = max(pop, key=lambda x: x.fitness)
            print(
                f"  [DE] Gen {gen+1}/{self.cfg.de_generations} "
                f"| best PPL: {best.fitness:.2f} "
                f"| target: {self.cfg.ppl_target:.1f}"
            )

            if best.fitness >= self.cfg.ppl_target:
                print(f"  [DE] Target PPL reached at generation {gen+1}!")
                break

        return {
            "orig_seq": seq,
            "attacked_seq": best.seq,
            "orig_ppl": orig_ppl,
            "final_ppl": best.fitness,
            "attack_success": best.fitness >= self.cfg.ppl_target,
            "mutations": best.mutations,
        }
