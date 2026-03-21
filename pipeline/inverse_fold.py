"""ESM-IF1 inverse folding and BLOSUM62 adversarial mutations.

Inverse folding: given a 3D backbone structure, generate amino acid
sequences that fold back to that structure. Using high temperature
sampling produces diverse sequences that are plausible structure-wise
but deviate from the native sequence -- ideal adversarial seeds.

BLOSUM62 adversarial mutations: apply n conservative substitutions
at positions identified by ESMFold gradient sensitivity. These look
biologically valid to sequence filters but cause large structural shifts.

Reference:
    Hsu et al. 2022 (ESM-IF1 inverse folding)
    Alkhouri et al. 2024 (gradient-guided BLOSUM attack)
"""

import numpy as np
from typing import List, Dict, Optional

from .config import PipelineConfig
from .trick_sequences import BLOSUM62_SIMILAR, AA_TO_IDX


class InverseFoldingModule:
    """ESM-IF1 inverse folding: structure -> sequences."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._model = None
        self._alphabet = None

    def _load(self):
        """Lazy-load ESM-IF1 (requires fair-esm package)."""
        if self._model is None:
            try:
                import esm
                if self.cfg.esm_if1_checkpoint:
                print(f"[ESM-IF1] Loading from local checkpoint: {self.cfg.esm_if1_checkpoint}")
                self._model, self._alphabet = esm.pretrained.load_model_and_alphabet_core(
                    "esm_if1_gvp4_t16_142M_UR50",
                    self.cfg.esm_if1_checkpoint,
                )
            else:
                print("[ESM-IF1] No local checkpoint set, downloading from HuggingFace...")
                self._model, self._alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            self._model = self._model.eval().to(self.cfg.device)
                print("[ESM-IF1] Model loaded")
            except ImportError:
                raise ImportError(
                    "fair-esm is required for inverse folding. "
                    "Install with: pip install fair-esm"
                )

    def from_pdb(
        self,
        pdb_path: str,
        chain_id: str = "A",
        n_sequences: int = None,
        temperature: float = None,
    ) -> List[Dict]:
        """Sample n_sequences from ESM-IF1 given a PDB backbone.

        Higher temperature = more diverse sequences (better for adversarial seeding).
        The authors recommend temperature=1.0 for native-like sequences;
        temperature=1.5 produces structurally plausible but sequence-diverse outputs.

        Args:
            pdb_path: Path to .pdb or .cif file
            chain_id: Which chain to use
            n_sequences: Number of samples (default: cfg.n_if_sequences)
            temperature: Sampling temperature (default: cfg.if_temperature)

        Returns:
            List of candidate dicts sorted by log-likelihood descending
        """
        import esm
        self._load()

        n_sequences = n_sequences or self.cfg.n_if_sequences
        temperature = temperature or self.cfg.if_temperature

        # Load backbone coordinates
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(
            structure
        )
        print(f"[ESM-IF1] PDB={pdb_path} chain={chain_id} length={len(native_seq)}")

        results = []
        for i in range(n_sequences):
            import torch
            with torch.no_grad():
                sampled_seq = self._model.sample(
                    coords, temperature=temperature, device=self.cfg.device
                )
                ll, _ = esm.inverse_folding.util.score_sequence(
                    self._model, self._alphabet, coords, sampled_seq
                )
            results.append({
                "seq": sampled_seq,
                "name": f"if_{i:02d}_ll{ll:.2f}",
                "source": "esm_if1",
                "log_likelihood": ll,
                "native_seq": native_seq,
                "pdb": pdb_path,
                "chain": chain_id,
            })
            if (i + 1) % 5 == 0:
                print(f"  [ESM-IF1] {i+1}/{n_sequences} sampled")

        results.sort(key=lambda x: x["log_likelihood"], reverse=True)
        return results


class BLOSUMAttack:
    """BLOSUM62-conservative adversarial mutations.

    Two strategies:
        random: pick n_mutations random substitutable positions
        gradient: pick positions with highest ESMFold gradient magnitude
                  (requires ESMFoldScorer.gradient_sensitivity)
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
        # Only mutate positions that have valid BLOSUM substitutes
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

    def gradient_guided_mutations(
        self,
        seq: str,
        grad_magnitude: np.ndarray,
        n_mutations: int = None,
        n_variants: int = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Apply BLOSUM62 mutations at the most gradient-sensitive positions.

        Position sensitivity = ||d(pLDDT)/d(embedding)||_2
        Mutating high-sensitivity positions maximally disrupts confidence
        while keeping the sequence chemically conservative.

        Args:
            seq: Amino acid sequence string
            grad_magnitude: Per-position gradient norms from ESMFoldScorer.gradient_sensitivity
            n_mutations: Number of positions to mutate (default: cfg.n_mutations)
            n_variants: Number of variant sequences to generate
            seed: Random seed for reproducibility

        Returns:
            List of mutant sequences
        """
        n_mutations = n_mutations or self.cfg.n_mutations
        n_variants = n_variants or self.cfg.n_blosum_variants
        rng = np.random.default_rng(seed)

        seq_list = list(seq)
        # Rank positions by gradient magnitude (descending)
        ranked = np.argsort(grad_magnitude)[::-1]
        # Filter to mutable positions (have BLOSUM substitutes)
        top_positions = [
            pos for pos in ranked if seq_list[pos] in BLOSUM62_SIMILAR
        ][:n_mutations]

        print(
            f"[BLOSUM] Top {len(top_positions)} sensitive positions: "
            f"{top_positions} | grad_mag: {grad_magnitude[top_positions].tolist()}"
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
        """Generate all possible single BLOSUM62-conservative point mutations.

        Useful for identifying the most adversarial single mutation.
        Returns list of (mutant_seq, position, original_aa, mutated_aa).
        """
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
