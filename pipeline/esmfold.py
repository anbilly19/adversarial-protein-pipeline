"""ESMFold scoring and ESM-Design adversarial gradient attack.

ESMFold is fully differentiable end-to-end, enabling white-box
gradient-based attacks on the confidence score (pLDDT).

The ESM-Design attack optimizes over the ESM-2 token embedding space.
Instead of running a discrete forward pass, we inject differentiable
"soft" embeddings directly into the ESM-2 language model, bypassing
the non-differentiable tokenizer lookup. The full pipeline:
  soft_embed (via ESM-2 word_embeddings) -> ESM-2 LM -> esm_s_mlp ->
  trunk (seq_feats, pair_feats) -> structure_module -> pLDDT

The gradient flows back through the entire chain to the soft token
logits, which are optimized via Adam. Gumbel-softmax with temperature
annealing bridges continuous optimization back to a discrete sequence.

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

# ESM vocabulary: AA tokens start at index 6 (after <cls>=0,<pad>=1,<eos>=2,<unk>=3,X=4,special=5)
# The 20 standard AAs map to indices 6..25 in the ESM-2 vocabulary.
_ESM_AA_START = 6
_ESM_AA_END = 26  # exclusive


class ESMFoldScorer:
    """ESMFold wrapper for batch pLDDT scoring and the ESM-Design gradient attack."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        model_path = cfg.esmfold_model_path
        local = "/" in model_path
        print(f"[ESMFold] Loading from {'local path' if local else 'HuggingFace'}: {model_path}...")
        # EsmTokenizer uses a fixed amino acid vocabulary - no protobuf needed.
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(
                model_path, local_files_only=local
            )
        except Exception:
            self.tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esmfold_v1", local_files_only=False
            )
        self.model = EsmForProteinFolding.from_pretrained(
            model_path, low_cpu_mem_usage=True, local_files_only=local,
            ignore_mismatched_sizes=True
        ).to(cfg.device)
        self.model.eval()
        # fp16 ESM-2 backbone to save VRAM
        self.model.esm = self.model.esm.half()
        # Chunk attention for long sequences
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
            "plddt": out["plddt"][0].cpu().numpy() * 100,
            "pae": out["predicted_aligned_error"][0].cpu().numpy(),
            "mean_plddt": out["plddt"][0].mean().item() * 100,
            "ptm": out.get("ptm", torch.tensor(0.0)).item(),
        }

    # ------------------------------------------------------------------
    # ESM-Design Adversarial Attack (corrected injection point)
    # ------------------------------------------------------------------
    def _seq_to_logits(self, seq: str) -> torch.Tensor:
        """Convert AA string to sharp initial logits over 20 AA classes [L, 20]."""
        idxs = [AA_TO_IDX[aa] for aa in seq if aa in AA_TO_IDX]
        onehot = F.one_hot(torch.tensor(idxs), num_classes=20).float()
        return onehot * 10.0

    def _logits_to_seq(self, logits: torch.Tensor) -> str:
        """Argmax over 20 AA classes -> sequence string."""
        return "".join(AA_VOCAB[i] for i in logits.argmax(dim=-1).tolist())

    def _soft_embed_to_esm_hidden(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """Map soft [L, 20] token distribution -> ESM-2 word embeddings [1, L+2, D].

        Injects a soft weighted combination of the 20 AA token embeddings
        directly into the ESM-2 embedding layer, replacing the discrete lookup.
        Prepends CLS and appends EOS to match the ESM-2 expected input format.

        Args:
            soft_tokens: [L, 20] Gumbel-softmax output
        Returns:
            [1, L+2, esm_feats] float tensor with CLS prepended and EOS appended
        """
        embed_w = self.model.esm.embeddings.word_embeddings.weight  # [vocab, D]
        # The 20 standard AA embeddings at indices 6..25
        aa_embeds = embed_w[_ESM_AA_START:_ESM_AA_END].float()  # [20, D]
        seq_embed = soft_tokens @ aa_embeds  # [L, D]

        cls_embed = embed_w[self.model.esm_dict_cls_idx].float().unsqueeze(0)  # [1, D]
        eos_embed = embed_w[self.model.esm_dict_eos_idx].float().unsqueeze(0)  # [1, D]
        # [1, L+2, D]
        full_embed = torch.cat([cls_embed, seq_embed, eos_embed], dim=0).unsqueeze(0)
        return full_embed

    def _forward_from_esm_embed(
        self, full_embed: torch.Tensor, L: int
    ) -> torch.Tensor:
        """Run the full ESMFold pipeline from pre-computed ESM-2 embeddings.

        Bypasses the ESM-2 token lookup and feeds embeddings directly through
        the ESM-2 transformer, then through esm_s_mlp and the folding trunk.
        Returns per-residue pLDDT [L].

        This correctly matches EsmForProteinFolding.forward() which calls:
            esm hidden states -> esm_s_mlp -> trunk(seq_feats, pair_feats, aa, pos_ids, mask)
        """
        B = 1
        device = self.cfg.device

        # Build attention mask: all 1s (CLS + L residues + EOS)
        attn_mask = torch.ones(B, L + 2, device=device, dtype=torch.bool)

        # Run ESM-2 transformer with injected embeddings
        # We use inputs_embeds to bypass the token embedding lookup
        esm_out = self.model.esm(
            inputs_embeds=full_embed.half(),  # ESM-2 runs in fp16
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        # Stack all hidden states: [B, L+2, n_layers+1, D]
        esm_hidden = torch.stack(esm_out["hidden_states"], dim=2)  # [B, L+2, n_layers+1, D]
        esm_s = esm_hidden[:, 1:-1]  # strip CLS/EOS: [B, L, n_layers+1, D]

        # Project to trunk sequence features
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)  # [B, L, D_esm]
        s_s_0 = self.model.esm_s_mlp(esm_s)  # [B, L, c_s]

        # Build pair features (zeros, same as normal forward)
        c_z = self.model.config.esmfold_config.trunk.pairwise_state_dim
        s_z_0 = s_s_0.new_zeros(B, L, L, c_z)

        # Build aa indices (argmax of logits for true_aa)
        # EsmFoldingTrunk needs true_aa in AF2 index space (0-based, 20 classes)
        # We pass zeros here since we don't have discrete tokens; this only affects
        # the structure module's angle resnet, not the pLDDT head.
        true_aa = torch.zeros(B, L, device=device, dtype=torch.long)
        position_ids = torch.arange(L, device=device).unsqueeze(0)  # [B, L]
        mask = torch.ones(B, L, device=device, dtype=torch.float)

        # Run folding trunk
        structure = self.model.trunk(
            s_s_0, s_z_0, true_aa, position_ids, mask, no_recycles=0
        )

        # Compute pLDDT from lddt_head
        from transformers.models.esm.modeling_esmfold import categorical_lddt
        lddt_head = self.model.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        plddt = categorical_lddt(lddt_head[-1], bins=self.model.lddt_bins)  # [B, L]
        return plddt[0]  # [L]

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
            3. Compute soft ESM-2 embeddings and run full ESMFold pipeline
            4. Loss = -mean(pLDDT) -> backprop -> Adam step on logits
            5. Anneal tau: 1.0 -> 0.1 (encourages discrete convergence)
            6. Track best sequence at highest pLDDT step
        """
        steps = steps or self.cfg.esm_design_steps
        L = len(seq)

        orig_plddt = self.score(seq)
        logits = torch.nn.Parameter(self._seq_to_logits(seq).to(self.cfg.device))
        optimizer = torch.optim.Adam([logits], lr=self.cfg.esm_lr)

        best_plddt = orig_plddt
        best_seq = seq
        plddt_curve = [orig_plddt]

        for step in range(steps):
            optimizer.zero_grad()
            progress = step / steps
            tau = self.cfg.esm_temp * (
                self.cfg.esm_temp_final / self.cfg.esm_temp
            ) ** progress
            soft_tokens = F.gumbel_softmax(logits, tau=tau, hard=False)  # [L, 20]
            full_embed = self._soft_embed_to_esm_hidden(soft_tokens)     # [1, L+2, D]
            plddt_per_residue = self._forward_from_esm_embed(full_embed, L)  # [L]
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

    def gradient_sensitivity(self, seq: str) -> np.ndarray:
        """Compute gradient magnitude per residue position.

        Returns per-position gradient norm of pLDDT w.r.t. ESM-2 word embeddings.
        High values = positions where small perturbations cause large pLDDT changes.
        Used to guide BLOSUM mutation site selection.

        The ESM-2 backbone runs in fp16 but gradients must accumulate in fp32.
        We keep a float32 proxy tensor that receives gradients via a fp32->fp16
        cast using a retain_grad hook, avoiding the embed.grad=None issue caused
        by .half() breaking the autograd graph back to the original float tensor.
        """
        tokens = self.tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        ).to(self.cfg.device)
        input_ids = tokens["input_ids"]
        B, L = input_ids.shape

        # Build ESM input with CLS/EOS
        bos = input_ids.new_full((B, 1), self.model.esm_dict_cls_idx)
        eos = input_ids.new_full((B, 1), self.model.esm_dict_padding_idx)
        esmaa = torch.cat([bos, input_ids, eos], dim=1)
        esmaa[range(B), (esmaa != 1).sum(1)] = self.model.esm_dict_eos_idx

        # float32 proxy: gradients accumulate here
        embed_f32 = self.model.esm.embeddings.word_embeddings(esmaa).float()
        embed_f32.requires_grad_(True)

        # fp16 view for ESM-2 (which runs in half precision)
        # We register retain_grad on the fp16 tensor so backward can flow through it,
        # then manually pull gradients back to embed_f32 via the chain rule.
        embed_f16 = embed_f32.half()
        embed_f16.retain_grad()  # keep grad on the non-leaf fp16 tensor

        attn_mask = torch.ones(B, L + 2, device=self.cfg.device, dtype=torch.bool)
        esm_out = self.model.esm(
            inputs_embeds=embed_f16,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        esm_hidden = torch.stack(esm_out["hidden_states"], dim=2)  # [B, L+2, layers+1, D]
        esm_s = esm_hidden[:, 1:-1]  # [B, L, layers+1, D]
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)
        c_z = self.model.config.esmfold_config.trunk.pairwise_state_dim
        s_z_0 = s_s_0.new_zeros(B, L, L, c_z)
        true_aa = torch.zeros(B, L, device=self.cfg.device, dtype=torch.long)
        position_ids = torch.arange(L, device=self.cfg.device).unsqueeze(0)
        mask = torch.ones(B, L, device=self.cfg.device, dtype=torch.float)

        structure = self.model.trunk(
            s_s_0, s_z_0, true_aa, position_ids, mask, no_recycles=0
        )
        from transformers.models.esm.modeling_esmfold import categorical_lddt
        lddt_head = self.model.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        plddt = categorical_lddt(lddt_head[-1], bins=self.model.lddt_bins)
        plddt.mean().backward()

        # Gradient w.r.t. residue embeddings: use fp16 grad upcast to fp32,
        # strip CLS/EOS tokens, compute per-position L2 norm.
        grad = embed_f16.grad.float()          # [B, L+2, D] in fp32
        grad_mag = grad[0, 1:-1].norm(dim=-1).detach().cpu().numpy()  # [L]
        return grad_mag
