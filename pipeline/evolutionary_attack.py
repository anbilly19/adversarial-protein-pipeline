"""Evolutionary attack for adversarial protein design.

Implements AF2-Mutation strategies (arXiv:2305.08929):
- Differential Evolution (DE) algorithm
- Mixed mutation operators: replacement, deletion, insertion
- Fitness based on pLDDT score (proxy for lDDT)

Gradient-free black-box optimization using evolutionary strategies.
Inspired by AF2-Mutation and genetic algorithms for protein engineering.

Strategy:
    1. Initialize population with mutation vectors
    2. Evaluate fitness (pLDDT score) for all candidates
    3. DE mutation: V = P_r1 + F × (P_r2 - P_r3)
    4. DE crossover with CR (crossover rate)
    5. Repeat until convergence or max generations

This avoids overfitting to ESMFold's gradient landscape and can discover
non-local sequence changes that gradients cannot reach.
"""
import numpy as np
from typing import Dict, List, Optional
from .config import PipelineConfig


class EvolutionaryAttack:
    """Evolutionary/genetic algorithm with AF2-Mutation strategies."""

    def __init__(self, cfg: PipelineConfig, esm_scorer):
        """Initialize evolutionary attack.

        Args:
            cfg: Pipeline configuration
            esm_scorer: ESMFoldScorer instance for fitness evaluation
        """
        self.cfg = cfg
        self.esm_scorer = esm_scorer

        # Differential Evolution hyperparameters (from AF2-Mutation paper)
        self.pop_size = 20  # Population size
        self.n_generations = 50  # Number of generations
        self.F = 0.8  # Differential weight (mutation factor)
        self.CR = 0.9  # Crossover rate

        # Mutation strategy probabilities
        self.p_replacement = 0.7  # Probability of replacement mutation
        self.p_deletion = 0.15  # Probability of deletion mutation
        self.p_insertion = 0.15  # Probability of insertion mutation

        # Standard amino acids
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def attack(self, seed_seq: str) -> Dict:
        """Run evolutionary attack on seed sequence.

        Args:
            seed_seq: Starting sequence

        Returns:
            Dict with attack results (same format as ESM-Design attack)
        """
        print(f"[Evolutionary] Starting AF2-Mutation DE attack with pop_size={self.pop_size}, "
              f"generations={self.n_generations}")
        print(f"[Evolutionary] Mutation mix: replacement={self.p_replacement:.0%}, "
              f"deletion={self.p_deletion:.0%}, insertion={self.p_insertion:.0%}")

        # Evaluate seed sequence
        orig_plddt = self.esm_scorer.score_batch([seed_seq])[0]
        print(f"[Evolutionary] Seed pLDDT: {orig_plddt:.2f}")

        # Initialize population with mixed mutations
        population = self._initialize_population(seed_seq)
        best_seq = seed_seq
        best_score = orig_plddt

        # Differential Evolution loop
        for gen in range(self.n_generations):
            # Evaluate fitness
            scores = self.esm_scorer.score_batch(population)

            # Track best
            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > best_score:
                best_seq = population[gen_best_idx]
                best_score = gen_best_score
                print(f"[Evolutionary] Gen {gen+1}/{self.n_generations}: "
                      f"New best pLDDT = {best_score:.2f} "
                      f"(+{best_score - orig_plddt:.2f})")
            elif (gen + 1) % 10 == 0:
                print(f"[Evolutionary] Gen {gen+1}/{self.n_generations}: "
                      f"Best pLDDT = {best_score:.2f}")

            # Differential Evolution: mutation and crossover
            new_population = []
            for i in range(self.pop_size):
                # DE mutation: V = P_r1 + F × (P_r2 - P_r3)
                # Select three distinct random indices
                indices = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                # Create mutant through differential mutation
                mutant = self._differential_mutate(
                    population[r1], population[r2], population[r3], seed_seq
                )

                # DE crossover with target
                trial = self._de_crossover(population[i], mutant)

                new_population.append(trial)

            population = new_population

        # Final evaluation
        final_score = self.esm_scorer.score_batch([best_seq])[0]
        success = final_score > self.cfg.plddt_target
        n_mutations = sum(1 for a, b in zip(seed_seq, best_seq[:len(seed_seq)]) if a != b)
        n_mutations += abs(len(best_seq) - len(seed_seq))  # Account for indels

        print(f"[Evolutionary] Final: pLDDT {orig_plddt:.2f} -> {final_score:.2f} "
              f"(+{final_score - orig_plddt:.2f})")
        print(f"[Evolutionary] Length: {len(seed_seq)} -> {len(best_seq)}, "
              f"mutations: {n_mutations}")
        print(f"[Evolutionary] Attack {'SUCCESS' if success else 'FAILED'} "
              f"(target: {self.cfg.plddt_target})")

        return {
            "orig_seq": seed_seq,
            "attacked_seq": best_seq,
            "orig_plddt": orig_plddt,
            "final_plddt": final_score,
            "attack_success": success,
            "n_mutations": n_mutations,
        }

    def _initialize_population(self, seed_seq: str) -> List[str]:
        """Initialize population with mixed mutation strategies."""
        population = [seed_seq]  # Keep seed in population

        for _ in range(self.pop_size - 1):
            # Apply random mixed mutations
            mutated = self._apply_mixed_mutations(seed_seq, rate=0.15)
            population.append(mutated)

        return population

    def _apply_mixed_mutations(self, seq: str, rate: float = 0.15) -> str:
        """Apply mixed mutation operators: replacement, deletion, insertion."""
        seq_list = list(seq)
        positions = list(range(len(seq_list)))

        for pos in positions:
            if np.random.rand() < rate:
                # Choose mutation type based on probabilities
                mut_type = np.random.choice(
                    ['replacement', 'deletion', 'insertion'],
                    p=[self.p_replacement, self.p_deletion, self.p_insertion]
                )

                if mut_type == 'replacement' and pos < len(seq_list):
                    seq_list[pos] = np.random.choice(self.amino_acids)
                elif mut_type == 'deletion' and pos < len(seq_list) and len(seq_list) > 10:
                    seq_list.pop(pos)
                elif mut_type == 'insertion' and len(seq_list) < len(seq) * 1.5:
                    aa = np.random.choice(self.amino_acids)
                    seq_list.insert(pos, aa)

        return "".join(seq_list)

    def _differential_mutate(self, p_r1: str, p_r2: str, p_r3: str, seed: str) -> str:
        """Differential Evolution mutation: V = P_r1 + F × (P_r2 - P_r3).
        
        Adapts DE for discrete sequence space using alignment-based differences.
        """
        # Use p_r1 as base and incorporate differences from p_r2 and p_r3
        result = list(p_r1)
        max_len = max(len(p_r1), len(p_r2), len(p_r3))

        # Extend sequences to same length for comparison
        s1 = list(p_r1) + ['-'] * (max_len - len(p_r1))
        s2 = list(p_r2) + ['-'] * (max_len - len(p_r2))
        s3 = list(p_r3) + ['-'] * (max_len - len(p_r3))

        result = []
        for i in range(max_len):
            # If p_r2 and p_r3 differ, apply scaled difference to p_r1
            if s2[i] != s3[i] and np.random.rand() < self.F:
                # Use p_r2's amino acid with probability F
                if s2[i] != '-':
                    result.append(s2[i])
                # Otherwise skip (implicit deletion)
            else:
                # Keep p_r1's amino acid
                if s1[i] != '-':
                    result.append(s1[i])

        # Ensure minimum length
        if len(result) < max(10, len(seed) // 2):
            result = list(seed)

        return "".join(result)

    def _de_crossover(self, target: str, mutant: str) -> str:
        """Differential Evolution crossover with CR rate."""
        max_len = max(len(target), len(mutant))
        t_list = list(target) + ['-'] * (max_len - len(target))
        m_list = list(mutant) + ['-'] * (max_len - len(mutant))

        result = []
        for i in range(max_len):
            # Crossover with probability CR
            if np.random.rand() < self.CR:
                if m_list[i] != '-':
                    result.append(m_list[i])
            else:
                if t_list[i] != '-':
                    result.append(t_list[i])

        # Ensure non-empty result
        if len(result) == 0:
            result = list(target)

        return "".join(result)
