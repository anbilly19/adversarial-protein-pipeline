"""Full adversarial protein pipeline orchestration.

Usage:
    # Mode 1: ProtGPT2 seed + ESM-Design attack
    python run_pipeline.py

    # Mode 2: Start from PDB structure (inverse folding)
    python run_pipeline.py --pdb rfah.pdb --chain B

    # Mode 3: Only run trick sequences
    python run_pipeline.py --tricks-only

    # Mode 4: Full pipeline with all stages
    python run_pipeline.py --pdb rfah.pdb --chain B --use-tricks --n-generated 30
"""

import argparse
from html import parser
import json
import os
from typing import List, Dict

from pipeline import (
    PipelineConfig,
    ProtGPT2Generator,
    ESMFoldScorer,
    InverseFoldingModule,
    BLOSUMAttack,
    get_all_trick_sequences,
)


def make_af3_job(seq: str, name: str, seed: int = 42) -> List[Dict]:
    """Format a sequence as an AlphaFold3 server input JSON.

    AlphaFold Server only allows a single seed per job.
    Output format matches the alphafoldserver dialect v1.
    Returns a list containing one job object.
    """
    return [
        {
            "name": name,
            "modelSeeds": [seed],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": seq,
                        "count": 1,
                    }
                }
            ],
            "dialect": "alphafoldserver",
            "version": 1,
        }
    ]


def print_summary(results: List[Dict], cfg: PipelineConfig) -> None:
    """Print a formatted summary table of attack results."""
    successful = [r for r in results if r.get("attack_success", False)]
    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print(f"  Candidates attacked : {len(results)}")
    print(f"  Successful hacks    : {len(successful)} (pLDDT > {cfg.plddt_target})")
    print(f"  Success rate        : {len(successful)/max(len(results),1)*100:.0f}%")
    print("-" * 72)
    print(f"{'Source':<22} {'Name':<26} {'Init':>6} {'Final':>6} {'Delta':>6} OK")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["final_plddt"], reverse=True):
        delta = r["final_plddt"] - r["orig_plddt"]
        ok = "v" if r.get("attack_success") else "x"
        print(
            f"{r.get('source','?'):<22} {r.get('name','?'):<26} "
            f"{r['orig_plddt']:>6.1f} {r['final_plddt']:>6.1f} "
            f"{delta:>+6.1f} {ok}"
        )
    print("=" * 72)

def run_complex_attack(
    cfg: PipelineConfig,
    pdb_path: str,
    chain_ids: list = None,          # None = auto-detect
) -> Dict[str, List[Dict]]:
    """
    Adversarial attack mode for protein complexes.

    For each chain independently:
        1. ESM-IF1 inverse folding (n_if_sequences_per_chain samples)
        2. Gradient-guided BLOSUM mutations on native chain sequence
        3. ESMFold batch scoring of candidates
        4. ESM-Design gradient attack on top-k per chain
        5. Export a single multi-chain AF3 JSON per attack combination

    Args:
        cfg: PipelineConfig
        pdb_path: Path to PDB/CIF of the complex
        chain_ids: Specific chains to attack; None = all chains

    Returns:
        Dict mapping chain_id -> list of attack result dicts
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    esm_scorer = ESMFoldScorer(cfg)
    if_module = InverseFoldingModule(cfg)
    blosum = BLOSUMAttack(cfg)

    # Stage 1: per-chain inverse folding
    if_results = if_module.from_pdb_all_chains(pdb_path, chain_ids)
    all_chain_ids = list(if_results.keys())

    chain_attacked_seqs = {}   # chain_id -> best attacked sequence
    chain_results = {}

    for chain_id, if_candidates in if_results.items():
        print(f"\n{'='*60}")
        print(f"[Complex] Attacking chain {chain_id} ({len(if_candidates)} IF seeds)")
        print(f"{'='*60}")

        # Gradient-guided BLOSUM mutations on native sequence
        native_seq = if_candidates[0]["native_seq"]
        grad_mag = esm_scorer.gradient_sensitivity(native_seq)
        mutants = blosum.gradient_guided_mutations(
            native_seq, grad_mag,
            n_mutations=cfg.n_mutations,
            n_variants=cfg.n_blosum_variants,
        )
        blosum_cands = [
            {"seq": s, "name": f"chain{chain_id}_blosum_{i:02d}",
             "source": "blosum_adversarial"}
            for i, s in enumerate(mutants)
        ]

        candidates = list(if_candidates) + blosum_cands

        # Score all candidates with ESMFold
        seqs = [c["seq"] for c in candidates]
        scores = esm_scorer.score_batch(seqs)
        for cand, score in zip(candidates, scores):
            cand["init_plddt"] = score
        candidates.sort(key=lambda x: x["init_plddt"], reverse=True)

        print(f"  Top candidate before attack: {candidates[0]['name']} "
              f"pLDDT={candidates[0]['init_plddt']:.1f}")

        # ESM-Design gradient attack on top-k
        results = []
        for cand in candidates[:cfg.top_k_attack]:
            print(f"\n  Attacking {cand['name']} "
                  f"(len={len(cand['seq'])}, pLDDT={cand['init_plddt']:.1f})")
            result = esm_scorer.esm_design_attack(cand["seq"])
            result["source"] = cand["source"]
            result["name"] = cand["name"]
            result["chain_id"] = chain_id
            results.append(result)

        chain_results[chain_id] = results

        # Best attacked sequence for this chain = highest final pLDDT
        best = max(results, key=lambda r: r["final_plddt"])
        chain_attacked_seqs[chain_id] = best["attacked_seq"]
        print(f"  [chain {chain_id}] best pLDDT: {best['orig_plddt']:.1f} "
              f"-> {best['final_plddt']:.1f}")

    # Export a single multi-chain AF3 JSON with all attacked chains
    _export_complex_af3_job(chain_attacked_seqs, all_chain_ids, cfg)
    _print_complex_summary(chain_results, cfg)

    return chain_results

def _export_complex_af3_job(
    chain_seqs: Dict[str, str],
    chain_order: list,
    cfg: PipelineConfig,
    seed: int = 42,
) -> None:
    """Write one AF3 JSON with all attacked chains as separate proteinChain entries."""
    sequences = [
        {"proteinChain": {"sequence": chain_seqs[c], "count": 1}}
        for c in chain_order
    ]
    job = [{
        "name": "complex_attack_" + "_".join(chain_order),
        "modelSeeds": [seed],
        "sequences": sequences,
        "dialect": "alphafoldserver",
        "version": 1,
    }]
    fname = os.path.join(cfg.output_dir, f"complex_{''.join(chain_order)}_attacked.json")
    with open(fname, "w") as f:
        json.dump(job, f, indent=2)
    print(f"\n[Complex] Multi-chain AF3 job written -> {fname}")


def _print_complex_summary(
    chain_results: Dict[str, List[Dict]],
    cfg: PipelineConfig,
) -> None:
    print("\n" + "=" * 72)
    print("COMPLEX ATTACK SUMMARY")
    print("-" * 72)
    print(f"{'Chain':<8} {'Source':<22} {'Name':<24} {'Init':>6} {'Final':>6} {'Delta':>6} OK")
    print("-" * 72)
    for chain_id, results in chain_results.items():
        for r in sorted(results, key=lambda x: x["final_plddt"], reverse=True):
            delta = r["final_plddt"] - r["orig_plddt"]
            ok = "v" if r.get("attack_success") else "x"
            print(f"{chain_id:<8} {r.get('source','?'):<22} {r.get('name','?'):<24} "
                  f"{r['orig_plddt']:>6.1f} {r['final_plddt']:>6.1f} {delta:>+6.1f} {ok}")
    print("=" * 72)

def run(
    cfg: PipelineConfig,
    pdb_path: str = None,
    chain_id: str = "A",
    use_tricks: bool = True,
    n_generated: int = None,
) -> List[Dict]:
    """Run the full adversarial protein pipeline.

    Stage 0: Load hardcoded trick sequences (fold-switchers, chameleons, BLOSUM)
    Stage 1: ESM-IF1 inverse folding from PDB (optional)
    Stage 2: ProtGPT2 generation + perplexity filter (optional)
    Stage 3: BLOSUM gradient-guided mutations on native PDB sequence (if PDB given)
    Stage 4: Score all candidates with ESMFold
    Stage 5: ESM-Design gradient attack on top-k candidates
    Stage 6: Export AF3 JSON files

    Args:
        cfg: PipelineConfig with all hyperparameters
        pdb_path: Optional path to PDB file for inverse folding
        chain_id: Chain to use from PDB
        use_tricks: Whether to include trick sequences
        n_generated: Override cfg.n_generated

    Returns:
        List of result dicts with attack outcomes
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_candidates: List[Dict] = []

    # -- Stage 0: Trick sequences ------------------------------------------------
    if use_tricks:
        tricks = get_all_trick_sequences()
        print(f"[Stage 0] Loaded {len(tricks)} trick sequences")
        all_candidates.extend(tricks)

    # -- Stage 1: Inverse folding + BLOSUM mutations from PDB --------------------
    esm_scorer = ESMFoldScorer(cfg)
    if pdb_path:
        print(f"[Stage 1] Inverse folding from {pdb_path} (chain={chain_id})...")
        if_module = InverseFoldingModule(cfg)
        if_seqs = if_module.from_pdb(pdb_path, chain_id)
        all_candidates.extend(if_seqs)
        print(f"  -> {len(if_seqs)} inverse-folded sequences")

        # Gradient-guided BLOSUM mutations on native sequence
        native_seq = if_seqs[0]["native_seq"]
        print("  -> Computing gradient sensitivity for native sequence...")
        grad_mag = esm_scorer.gradient_sensitivity(native_seq)
        blosum = BLOSUMAttack(cfg)
        mutants = blosum.gradient_guided_mutations(
            native_seq, grad_mag, n_mutations=cfg.n_mutations, n_variants=15
        )
        blosum_cands = [
            {"seq": s, "name": f"blosum_{i:02d}", "source": "blosum_adversarial"}
            for i, s in enumerate(mutants)
        ]
        all_candidates.extend(blosum_cands)
        print(f"  -> {len(blosum_cands)} BLOSUM gradient-guided variants")

    # -- Stage 2: ProtGPT2 generation --------------------------------------------
    if n_generated is None:
        n_generated = cfg.n_generated
    if n_generated > 0:
        print(f"[Stage 2] Generating {n_generated} sequences with ProtGPT2...")
        pg2 = ProtGPT2Generator(cfg)
        filtered = pg2.generate_and_filter()
        pg2_cands = [
            {"seq": s, "name": f"protgpt2_{i:02d}_ppl{p:.1f}", "source": "protgpt2", "perplexity": p}
            for i, (s, p) in enumerate(filtered)
        ]
        all_candidates.extend(pg2_cands)
        print(f"  -> {len(pg2_cands)} ProtGPT2 sequences passed perplexity filter")

    # -- Stage 3: Score all candidates with ESMFold ------------------------------
    print(f"[Stage 3] Scoring {len(all_candidates)} candidates with ESMFold...")
    seqs = [c["seq"] for c in all_candidates]
    plddt_scores = esm_scorer.score_batch(seqs)
    for cand, score in zip(all_candidates, plddt_scores):
        cand["init_plddt"] = score

    # Sort by initial pLDDT descending - best seeds for gradient attack
    all_candidates.sort(key=lambda x: x["init_plddt"], reverse=True)
    print("  Top 5 candidates before attack:")
    for c in all_candidates[:5]:
        print(f"    [{c['source']:20s}] {c['name']:30s} pLDDT={c['init_plddt']:.1f}")

    # -- Stage 4: ESM-Design gradient attack on top-k candidates ----------------
    print(f"\n[Stage 4] Running ESM-Design attack on top {cfg.top_k_attack} candidates...")
    results = []
    for cand in all_candidates[:cfg.top_k_attack]:
        print(
            f"\n  Attacking [{cand['source']}] {cand['name']} "
            f"(len={len(cand['seq'])}, init_pLDDT={cand['init_plddt']:.1f})"
        )
        result = esm_scorer.esm_design_attack(cand["seq"])
        result["source"] = cand["source"]
        result["name"] = cand["name"]
        results.append(result)

    # -- Stage 5: Export AF3 JSON files -----------------------------------------
    print(f"\n[Stage 5] Exporting AF3 job JSONs to {cfg.output_dir}/...")
    exported = 0
    for r in results:
        job = make_af3_job(r["attacked_seq"], r["name"])
        fname = os.path.join(
            cfg.output_dir,
            f"{r['name']}_plddt{r['final_plddt']:.0f}.json"
        )
        with open(fname, "w") as f:
            json.dump(job, f, indent=2)
        exported += 1
    print(f"  -> {exported} AF3 job files written")

    print_summary(results, cfg)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial Protein Pipeline: hack AlphaFold3 confidence scores"
    )
    parser.add_argument("--pdb", type=str, default=None,
                        help="Path to PDB file for ESM-IF1 inverse folding")
    parser.add_argument("--chain", type=str, default="A",
                        help="Chain ID to use from PDB file")
    parser.add_argument("--use-tricks", action="store_true", default=True,
                        help="Include known adversarial trick sequences")
    parser.add_argument("--tricks-only", action="store_true",
                        help="Only run trick sequences (skip ProtGPT2 generation)")
    parser.add_argument("--n-generated", type=int, default=None,
                        help="Number of ProtGPT2 sequences to generate")
    parser.add_argument("--steps", type=int, default=300,
                        help="ESM-Design gradient attack steps")
    parser.add_argument("--plddt-target", type=float, default=90.0,
                        help="pLDDT threshold for attack success")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top candidates to attack")
    parser.add_argument("--output-dir", type=str, default="af3_attack_jobs",
                        help="Output directory for AF3 JSON files")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (auto-detect if not set)")
    parser.add_argument("--esmfold-path", type=str, default=None,
                        help="Local path to pre-downloaded ESMFold weights")
    parser.add_argument("--complex", action="store_true",
                    help="Attack all chains of a protein complex (requires --pdb)")
    parser.add_argument("--chains", type=str, default=None,
                    help="Comma-separated chain IDs e.g. 'A,B,C' (default: auto-detect all)")
    parser.add_argument("--protgpt2-path", type=str, default=None,
                        help="Local path to pre-downloaded ProtGPT2 weights")
    parser.add_argument("--esm-if1-checkpoint", type=str, default=None,
                        help="Local path to esm_if1_gvp4_t16_142M_UR50.pt")
    parser.add_argument("--af3-seed", type=int, default=42,
                        help="Single integer seed for AF3 modelSeeds (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(
        esm_design_steps=args.steps,
        plddt_target=args.plddt_target,
        top_k_attack=args.top_k,
        output_dir=args.output_dir,
        **({"esmfold_model_path": args.esmfold_path} if args.esmfold_path else {}),
        **({"protgpt2_model_path": args.protgpt2_path} if args.protgpt2_path else {}),
        **({"esm_if1_checkpoint": args.esm_if1_checkpoint} if args.esm_if1_checkpoint else {}),
    )
    if args.device:
        cfg.device = args.device

    if args.complex:
        if not args.pdb:
            parser.error("--complex requires --pdb")
        chain_ids = [c.strip() for c in args.chains.split(",")] if args.chains else None
        run_complex_attack(cfg=cfg, pdb_path=args.pdb, chain_ids=chain_ids)
    else:
        n_generated = 0 if args.tricks_only else args.n_generated
        results = run(cfg=cfg, pdb_path=args.pdb, chain_id=args.chain,
                    use_tricks=args.use_tricks, n_generated=n_generated)
