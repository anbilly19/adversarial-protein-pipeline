"""ESMFold scoring and ESM-Design adversarial gradient attack.

ESMFold is fully differentiable end-to-end, enabling white-box
gradient-based attacks on the confidence score (pLDDT).

The ESM-Design attack replaces the discrete tokenizer lookup
with a soft weighted sum over the ESM-2 embedding matrix,
making the entire forward pass differentiable w.r.t. a continuous
sequence representation. Gumbel-softmax with temperature annealing
bridges the continuous optimization back to a discrete sequence.

Reference:
    Verkuil et al. 2022 (ESMFold)
    Chu et al. 2024 (ESM-Design hallucination)
    Alkhouri et al. 2024 (Adversarial attacks via red-teaming)
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import EsmForProteinFolding, EsmTokenizer
from typing import List, Dict, Optional

from .config import PipelineConfig
from .trick_sequences import AA_VOCAB, AA_TO_IDX


class ESMFoldScorer:
    """ESMFold wrapper for batch pLDDT scoring and the ESM-Design gradient attack."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        model_path = cfg.esmfold_model_path
        local = "/" in model_path  # local path vs HuggingFace Hub ID
        print(f"[ESMFold] Loading from {'local path' if local else 'HuggingFace'}: {model_path}...")
        # EsmTokenizer uses a fixed amino acid vocabulary - no protobuf/sentencepiece needed.
        # We load from the model path if local, but fall back to the built-in ESM vocab
        # to avoid the AutoTokenizer -> tokenizer.model -> protobuf dependency chain.
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(
                model_path, local_files_only=local
            )
        except Exception:
            # Fallback: construct tokenizer directly from the canonical ESM amino acid vocab
            self.tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esmfold_v1", local_files_only=False
            )
        self.model = EsmForProteinFolding.from_pretrained(
            model_path, low_cpu_mem_usage=True, local_files_only=local
        ).to(cfg.device)
        self.model.eval()
        # fp16 ESM-2 backbone to save VRAM (~8GB -> ~5GB)
        self.model.esm = self.model.esm.half()
        # Chunk attention for long sequences (> 500 residues)
        self.model.trunk.set_chunk_size(64)
        print(f"[ESMFold] Ready on {cfg.device}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, seq: str) -> float:
        """Return mean pLDDT (0-100) for a single sequence."""
        tokens = self.tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        ).to(self.cfg.device)
        with torch.no_grad():
            out = self.model(**tokens)
        return out["plddt"][0].mean().item() * 100

    def score_batch(self, seqs: List[str]) -> List[float]:
        """Batch scoring. Sequences are padded to the same length within each batch."""
        scores = []
        bs = self.cfg.batch_size
        for i in range(0, len(seqs), bs):
            batch = seqs[i : i + bs]
            tokens = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.cfg.device)
            with torch.no_grad():
                out = self.model(**tokens)
            for j, seq in enumerate(batch):
                scores.append(out["plddt"][j, : len(seq)].mean().item() * 100)
        return scores

    def score_with_pae(self, seq: str) -> Dict:
        """Return pLDDT + PAE matrix for a single sequence."""
        tokens = self.tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        ).to(self.cfg.device)
        with torch.no_grad():
            out = self.model(**tokens)
        return {
            "plddt": out["plddt"][0].cpu().numpy() * 100,  # [L]
            "pae": out["predicted_aligned_error"][0].cpu().numpy(),  # [L, L]
            "mean_plddt": out["plddt"][0].mean().item() * 100,
            "ptm": out.get("ptm", torch.tensor(0.0)).item(),
        }

    # ------------------------------------------------------------------
    # ESM-Design Adversarial Attack
    # ------------------------------------------------------------------
    def _seq_to_logits(self, seq: str) -> torch.Tensor:
        """Convert AA string to sharp initial logits [L, 20]."""
        idxs = [AA_TO_IDX[aa] for aa in seq if aa in AA_TO_IDX]
        onehot = F.one_hot(torch.tensor(idxs), num_classes=20).float()
        return onehot * 10.0  # sharp initial distribution

    def _logits_to_seq(self, logits: torch.Tensor) -> str:
        """Argmax over the 20 AA classes -> sequence string."""
        return "".join(AA_VOCAB[i] for i in logits.argmax(dim=-1).tolist())

    def _soft_embed(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """Differentiable embedding: weighted sum over ESM-2 embedding matrix.

        This replaces the non-differentiable tokenizer lookup, making the
        entire forward pass differentiable w.r.t. soft_tokens.

        Args:
            soft_tokens: [L, 20] float tensor (Gumbel-softmax output)
        Returns:
            [L, 1280] float tensor in ESM-2 embedding space
        """
        # ESM-2 embedding matrix: [vocab_size, 1280]
        embed_w = self.model.esm.embeddings.word_embeddings.weight
        # AA tokens start at index 6 in ESM vocabulary
        aa_embeds = embed_w[6:26].float()  # [20, 1280]
        return soft_tokens @ aa_embeds  # [L, 1280]

    def _forward_with_soft_embed(
        self, soft_tokens: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Run ESMFold trunk with injected soft embeddings. Returns pLDDT [L]."""
        soft_embed = self._soft_embed(soft_tokens)  # [L, 1280]
        # Prepend CLS, append EOS tokens
        cls = self.model.esm.embeddings.word_embeddings.weight[3].float()
        eos = self.model.esm.embeddings.word_embeddings.weight[1].float()
        full_embed = torch.cat(
            [
                cls.unsqueeze(0).unsqueeze(0),
                soft_embed.unsqueeze(0),
                eos.unsqueeze(0).unsqueeze(0),
            ],
            dim=1,
        )  # [1, L+2, 1280]
        attn_mask = torch.ones(1, seq_len + 2, device=self.cfg.device)
        out = self.model.trunk(full_embed, attention_mask=attn_mask)
        return out["plddt"][0, 1:-1]  # strip CLS/EOS -> [L]

    def esm_design_attack(
        self,
        seq: str,
        steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """Gradient descent through ESMFold to maximize mean pLDDT.

        Algorithm:
            1. Initialize logits from seed sequence (sharp one-hot * 10)
            2. At each step, sample via Gumbel-softmax at temperature tau
            3. Forward pass through ESMFold trunk using soft embeddings
            4. Loss = -mean(pLDDT) -> backprop -> Adam step on logits
            5. Anneal tau: 1.0 -> 0.1 (encourages discrete convergence)
            6. Track best sequence = argmax of logits at highest pLDDT step

        Args:
            seq: Seed amino acid sequence string
            steps: Number of gradient steps (default: cfg.esm_design_steps)
            verbose: Print progress every 50 steps
        Returns:
            Dict with original_seq, attacked_seq, orig_plddt, final_plddt,
            attack_success, plddt_curve
        """
        steps = steps or self.cfg.esm_design_steps
        L = len(seq)

        # Baseline
        orig_plddt = self.score(seq)

        # Initialize learnable logits
        logits = torch.nn.Parameter(self._seq_to_logits(seq).to(self.cfg.device))
        optimizer = torch.optim.Adam([logits], lr=self.cfg.esm_lr)

        best_plddt = orig_plddt
        best_seq = seq
        plddt_curve = [orig_plddt]

        for step in range(steps):
            optimizer.zero_grad()
            # Cosine temperature annealing: tau_init -> tau_final
            progress = step / steps
            tau = self.cfg.esm_temp * (
                self.cfg.esm_temp_final / self.cfg.esm_temp
            ) ** progress
            # Gumbel-softmax: differentiable discrete approximation [L, 20]
            soft_tokens = F.gumbel_softmax(logits, tau=tau, hard=False)
            # Forward pass through ESMFold trunk
            plddt_per_residue = self._forward_with_soft_embed(soft_tokens, L)
            mean_plddt = plddt_per_residue.mean()
            loss = -mean_plddt
            loss.backward()
            optimizer.step()

            cur_plddt = mean_plddt.item() * 100
            plddt_curve.append(cur_plddt)
            if cur_plddt > best_plddt:
                best_plddt = cur_plddt
                best_seq = self._logits_to_seq(logits)
            if verbose and step % 50 == 0:
                print(
                    f"  [ESM-Design] step {step:3d}/{steps} "
                    f"tau={tau:.3f} pLDDT={cur_plddt:.1f}"
                )

        return {
            "original_seq": seq,
            "attacked_seq": best_seq,
            "orig_plddt": orig_plddt,
            "final_plddt": best_plddt,
            "attack_success": best_plddt > self.cfg.plddt_target,
            "plddt_curve": plddt_curve,
        }

    def gradient_sensitivity(
        self, seq: str
    ) -> np.ndarray:
        """Compute gradient magnitude per residue position.

        Returns per-position gradient norm of pLDDT w.r.t. the ESM-2 embedding.
        High values = positions where small perturbations cause large pLDDT changes.
        Used to guide BLOSUM mutation site selection.

        Args:
            seq: Amino acid sequence string
        Returns:
            grad_magnitude: np.ndarray of shape [L]
        """
        tokens = self.tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        ).to(self.cfg.device)
        embed = self.model.esm.embeddings.word_embeddings(
            tokens["input_ids"]
        ).float()  # [1, L, 1280]
        embed.requires_grad_(True)
        out = self.model.trunk(
            embed, attention_mask=tokens["attention_mask"]
        )
        plddt = out["plddt"][0].mean()
        plddt.backward()
        grad_mag = embed.grad[0].norm(dim=-1).detach().cpu().numpy()  # [L]
        return grad_mag
