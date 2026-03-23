"""Evolutionary attack for adversarial protein design.

Gradient-free black-box optimization using evolutionary strategies.
Inspired by AF2-Mutation and genetic algorithms for protein engineering.

Strategy:
    1. Initialize population with random mutations of seed sequence
    2. Evaluate fitness (pLDDT score) for all candidates
    3. Select top performers (elitism)
    4. Generate new population via mutation and crossover
    5. Repeat until convergence or max generations

This avoids overfitting to ESMFold's gradient landscape and can discover
non-local sequence changes that gradients cannot reach.
"""
import numpy as np
from typing import Dict, List, Optional
from .config import PipelineConfig


class EvolutionaryAttack:
    """Evolutionary/genetic algorithm for adversarial sequence design."""
    
    def __init__(self, cfg: PipelineConfig, esm_scorer):
        """Initialize evolutionary attack.
        
        Args:
            cfg: Pipeline configuration
            esm_scorer: ESMFoldScorer instance for fitness evaluation
        """
        self.cfg = cfg
        self.esm_scorer = esm_scorer
        
        # Evolutionary hyperparameters
        self.pop_size = 20  # Population size per generation
        self.n_generations = 50  # Number of generations
        self.elite_frac = 0.2  # Top fraction to keep
        self.mutation_rate = 0.15  # Probability of mutating each position
        self.crossover_rate = 0.3  # Probability of crossover
        
        # Standard amino acids
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    def attack(self, seed_seq: str) -> Dict:
        """Run evolutionary attack on seed sequence.
        
        Args:
            seed_seq: Starting sequence
            
        Returns:
            Dict with attack results (same format as ESM-Design attack)
        """
        print(f"[Evolutionary] Starting attack with pop_size={self.pop_size}, "
              f"generations={self.n_generations}")
        
        # Evaluate seed sequence
        orig_plddt = self.esm_scorer.score_batch([seed_seq])[0]
        print(f"[Evolutionary] Seed pLDDT: {orig_plddt:.2f}")
        
        # Initialize population
        population = self._initialize_population(seed_seq)
        best_seq = seed_seq
        best_score = orig_plddt
        
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
            
            # Selection: keep elites
            n_elite = max(2, int(self.pop_size * self.elite_frac))
            elite_indices = np.argsort(scores)[-n_elite:]
            elites = [population[i] for i in elite_indices]
            
            # Generate new population
            new_population = elites.copy()
            
            while len(new_population) < self.pop_size:
                # Tournament selection
                parent1 = self._tournament_select(population, scores)
                parent2 = self._tournament_select(population, scores)
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 if np.random.rand() < 0.5 else parent2
                
                # Mutation
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population[:self.pop_size]
        
        # Final evaluation
        final_score = self.esm_scorer.score_batch([best_seq])[0]
        success = final_score > self.cfg.plddt_target
        
        print(f"[Evolutionary] Final: pLDDT {orig_plddt:.2f} -> {final_score:.2f} "
              f"(+{final_score - orig_plddt:.2f})")
        print(f"[Evolutionary] Attack {'SUCCESS' if success else 'FAILED'} "
              f"(target: {self.cfg.plddt_target})")
        
        return {
            "orig_seq": seed_seq,
            "attacked_seq": best_seq,
            "orig_plddt": orig_plddt,
            "final_plddt": final_score,
            "attack_success": success,
            "n_mutations": sum(a != b for a, b in zip(seed_seq, best_seq)),
        }
    
    def _initialize_population(self, seed_seq: str) -> List[str]:
        """Create initial population with random mutations of seed."""
        population = [seed_seq]  # Include seed
        
        for _ in range(self.pop_size - 1):
            mutated = self._mutate(seed_seq, rate=0.1)
            population.append(mutated)
        
        return population
    
    def _mutate(self, seq: str, rate: Optional[float] = None) -> str:
        """Apply random point mutations to sequence."""
        if rate is None:
            rate = self.mutation_rate
        
        seq_list = list(seq)
        for i in range(len(seq_list)):
            if np.random.rand() < rate:
                seq_list[i] = np.random.choice(self.amino_acids)
        
        return "".join(seq_list)
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Single-point crossover between two parent sequences."""
        assert len(parent1) == len(parent2), "Parents must have same length"
        
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child
    
    def _tournament_select(self, population: List[str], scores: np.ndarray, 
                          k: int = 3) -> str:
        """Tournament selection: pick best from k random candidates."""
        indices = np.random.choice(len(population), size=k, replace=False)
        tournament_scores = scores[indices]
        winner_idx = indices[np.argmax(tournament_scores)]
        
        return population[winner_idx]
