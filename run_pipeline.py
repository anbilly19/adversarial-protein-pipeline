"""Full adversarial protein pipeline — gradient-free edition.

Usage:
    # Mode 1: ProtGPT2 seed + DE attack (no PDB needed)
    python run_pipeline.py

    # Mode 2: Start from PDB structure (native sequence extraction)
    python run_pipeline.py --pdb rfah.pdb --chain B

    # Mode 3: Only run trick sequences through DE
    python run_pipeline.py --tricks-only

    # Mode 4: Full pipeline
    python run_pipeline.py --pdb rfah.pdb --chain B --use-tricks --n-generated 30

    # Mode 5: Attack all chains of a protein complex
    python run_pipeline.py --complex --pdb 7k3g.pdb
    python run_pipeline.py --complex --pdb 7k3g.pdb --chains A,B

Changes from stable-gradient-attack branch:
    - ESMFold / ESM-IF1 / ESM-Design gradient attack removed entirely
    - Replaced by ESM-2 OFS pseudo-perplexity oracle (CPU, ~7 ms/seq)
    - Differential Evolution (DE/rand/1/bin) replaces gradient-guided attack
    - No GPU required; no torch_scatter; no biotite
    - AF3 JSON export unchanged
"""

import argparse
import json
import os
from typing import List, Dict, Optional

from pipeline import (
    PipelineConfig,
    ProtGPT2Generator,
    ESM2OracleScorer,
    DEAttacker,
    BLOSUMAttack,
    get_all_trick_sequences,
)
from pipeline.pdb_utils import read_pdb_sequences, get_chain_ids as _pdb_chain_ids


# ---------------------------------------------------------------------------
# AF3 JSON helpers  (unchanged from stable-gradient-attack)
# ---------------------------------------------------------------------------

def make_af3_job(seq: str, name: str, seed: int = 42) -> List[Dict]:
    """Format a single-chain sequence as an AlphaFold3 server input JSON."""
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


def make_af3_job_single_chain(seq: str, chain_id: str, name: str, seed: int = 42) -> List[Dict]:
    """Format one chain as a standalone single-chain AF3 job."""
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


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict], cfg: PipelineConfig) -> None:
    successful = [r for r in results if r.get("attack_success", False)]
    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print(f"  Candidates attacked : {len(results)}")
    print(f"  Successful attacks  : {len(successful)} (PPL >= {cfg.ppl_target})")
    print(f"  Success rate        : {len(successful)/max(len(results),1)*100:.0f}%")
    print("-" * 72)
    print(f"{'Source':<22} {'Name':<26} {'Init PPL':>9} {'Final PPL':>9} {'Delta':>7} OK")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["final_ppl"], reverse=True):
        delta = r["final_ppl"] - r["orig_ppl"]
        ok = "v" if r.get("attack_success") else "x"
        print(
            f"{r.get('source','?'):<22} {r.get('name','?'):<26} "
            f"{r['orig_ppl']:>9.2f} {r['final_ppl']:>9.2f} "
            f"{delta:>+7.2f} {ok}"
        )
    print("=" * 72)


def _print_complex_summary(chain_results: Dict[str, List[Dict]], cfg: PipelineConfig) -> None:
    print("\n" + "=" * 80)
    print("COMPLEX ATTACK SUMMARY")
    print("-" * 80)
    print(f"{'Chain':<8} {'Source':<22} {'Name':<24} {'Init PPL':>9} {'Final PPL':>9} {'Delta':>7} OK")
    print("-" * 80)
    for chain_id, results in chain_results.items():
        for r in sorted(results, key=lambda x: x["final_ppl"], reverse=True):
            delta = r["final_ppl"] - r["orig_ppl"]
            ok = "v" if r.get("attack_success") else "x"
            print(
                f"{chain_id:<8} {r.get('source','?'):<22} {r.get('name','?'):<24} "
                f"{r['orig_ppl']:>9.2f} {r['final_ppl']:>9.2f} {delta:>+7.2f} {ok}"
            )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Single-chain pipeline
# ---------------------------------------------------------------------------

def run(
    cfg: PipelineConfig,
    pdb_path: str = None,
    chain_id: str = "A",
    use_tricks: bool = True,
    n_generated: int = None,
) -> List[Dict]:
    """Run the full gradient-free adversarial protein pipeline.

    Stage 0: Load hardcoded trick sequences
    Stage 1: Read native sequence from PDB (if provided)
             + BLOSUM sensitivity-guided seed variants
    Stage 2: ProtGPT2 generation + perplexity filter (optional)
    Stage 3: Score all candidates with ESM-2 OFS PPL
    Stage 4: DE attack on top-k candidates
    Stage 5: Export AF3 JSON files

    Args:
        cfg:         PipelineConfig
        pdb_path:    Optional path to PDB file
        chain_id:    Chain to use from PDB
        use_tricks:  Whether to include trick sequences
        n_generated: Override cfg.n_generated

    Returns:
        List of result dicts with attack outcomes
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_candidates: List[Dict] = []

    scorer = ESM2OracleScorer(cfg)

    # -- Stage 0: Trick sequences --------------------------------------------
    if use_tricks:
        tricks = get_all_trick_sequences()
        print(f"[Stage 0] Loaded {len(tricks)} trick sequences")
        all_candidates.extend(tricks)

    # -- Stage 1: Native sequence + BLOSUM seed variants from PDB ------------
    native_seq = None
    if pdb_path:
        print(f"[Stage 1] Reading native sequence from {pdb_path} (chain={chain_id})")
        chain_seqs = read_pdb_sequences(pdb_path)
        if chain_id not in chain_seqs:
            raise ValueError(
                f"Chain '{chain_id}' not found in {pdb_path}. "
                f"Available: {list(chain_seqs.keys())}"
            )
        native_seq = chain_seqs[chain_id]
        print(f"  -> Native sequence length: {len(native_seq)}")

        # Use ESM-2 OFS surprisal for position sensitivity (replaces gradient)
        print("  -> Computing position surprisal (ESM-2 OFS) for native sequence...")
        surprisals = scorer.position_sensitivity(native_seq)

        blosum = BLOSUMAttack(cfg)
        mutants = blosum.sensitivity_guided_mutations(
            native_seq, surprisals,
            n_mutations=cfg.n_mutations,
            n_variants=cfg.n_blosum_variants,
        )
        blosum_cands = [
            {"seq": s, "name": f"blosum_{i:02d}", "source": "blosum_sensitivity"}
            for i, s in enumerate(mutants)
        ]
        all_candidates.extend(blosum_cands)
        print(f"  -> {len(blosum_cands)} BLOSUM sensitivity-guided variants")

    # -- Stage 2: ProtGPT2 generation ----------------------------------------
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

    # -- Stage 3: Score all candidates with ESM-2 OFS PPL --------------------
    print(f"[Stage 3] Scoring {len(all_candidates)} candidates with ESM-2 OFS PPL...")
    seqs = [c["seq"] for c in all_candidates]
    ppl_scores = scorer.score_batch(seqs)
    for cand, score in zip(all_candidates, ppl_scores):
        cand["init_ppl"] = score

    # Sort descending: highest PPL = most unusual = best adversarial seeds
    all_candidates.sort(key=lambda x: x["init_ppl"], reverse=True)
    print("  Top 5 candidates before DE attack:")
    for c in all_candidates[:5]:
        print(f"    [{c['source']:20s}] {c['name']:30s} PPL={c['init_ppl']:.2f}")

    # -- Stage 4: DE attack on top-k candidates ------------------------------
    print(f"\n[Stage 4] Running DE attack on top {cfg.top_k_attack} candidates...")
    results = []
    for cand in all_candidates[:cfg.top_k_attack]:
        print(
            f"\n  Attacking [{cand['source']}] {cand['name']} "
            f"(len={len(cand['seq'])}, init_PPL={cand['init_ppl']:.2f})"
        )
        # Reuse per-candidate sensitivity if native_seq is available
        sens = None
        if native_seq and cand["seq"] == native_seq:
            sens = surprisals
        de = DEAttacker(cfg, scorer, sensitivity=sens)
        result = de.attack(cand["seq"])
        result["source"] = cand["source"]
        result["name"] = cand["name"]
        result["orig_ppl"] = cand["init_ppl"]
        results.append(result)

    # -- Stage 5: Export AF3 JSON files --------------------------------------
    print(f"\n[Stage 5] Exporting AF3 job JSONs to {cfg.output_dir}/...")
    exported = 0
    for r in results:
        job = make_af3_job(r["attacked_seq"], r["name"])
        fname = os.path.join(
            cfg.output_dir,
            f"{r['name']}_ppl{r['final_ppl']:.0f}.json"
        )
        with open(fname, "w") as f:
            json.dump(job, f, indent=2)
        exported += 1
    print(f"  -> {exported} AF3 job files written")

    print_summary(results, cfg)
    return results


# ---------------------------------------------------------------------------
# Complex attack pipeline
# ---------------------------------------------------------------------------

def run_complex_attack(
    cfg: PipelineConfig,
    pdb_path: str,
    chain_ids: List[str] = None,
    af3_seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Adversarial attack mode for protein complexes.

    Attacks each chain independently using DE + ESM-2 OFS PPL.
    No ESM-IF1, no GPU, no torch_scatter required.

    Pipeline per chain:
        1. Read native sequence from PDB
        2. ESM-2 position surprisal -> BLOSUM sensitivity-guided seeds
        3. ESM-2 OFS PPL scoring of all candidates
        4. DE attack on top-k per chain
        5. Export one AF3 JSON per chain + combined multi-chain JSON
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    scorer = ESM2OracleScorer(cfg)
    blosum = BLOSUMAttack(cfg)

    all_native_seqs = read_pdb_sequences(pdb_path)
    if chain_ids is None:
        chain_ids = list(all_native_seqs.keys())
        print(f"[Complex] Auto-detected chains: {chain_ids}")
    else:
        print(f"[Complex] Using specified chains: {chain_ids}")

    chain_results: Dict[str, List[Dict]] = {}
    best_per_chain: Dict[str, str] = {}

    for chain_id in chain_ids:
        print(f"\n{'='*64}")
        print(f"[Complex] Chain {chain_id}")
        print(f"{'='*64}")

        native_seq = all_native_seqs[chain_id]
        print(f"  Native sequence length: {len(native_seq)}")

        print("  Computing position surprisal (ESM-2 OFS)...")
        surprisals = scorer.position_sensitivity(native_seq)

        mutants = blosum.sensitivity_guided_mutations(
            native_seq, surprisals,
            n_mutations=cfg.n_mutations,
            n_variants=cfg.n_blosum_variants,
        )
        candidates = [
            {
                "seq": s,
                "name": f"chain{chain_id}_blosum_{i:02d}",
                "source": "blosum_sensitivity",
            }
            for i, s in enumerate(mutants)
        ]

        # Score candidates
        print(f"  Scoring {len(candidates)} candidates with ESM-2 OFS PPL...")
        seqs = [c["seq"] for c in candidates]
        scores = scorer.score_batch(seqs)
        for cand, score in zip(candidates, scores):
            cand["init_ppl"] = score
        candidates.sort(key=lambda x: x["init_ppl"], reverse=True)

        print(f"  Top candidate: {candidates[0]['name']} PPL={candidates[0]['init_ppl']:.2f}")

        # DE attack on top-k
        results = []
        for cand in candidates[:cfg.top_k_attack]:
            print(
                f"\n  Attacking {cand['name']} "
                f"(len={len(cand['seq'])}, PPL={cand['init_ppl']:.2f})"
            )
            de = DEAttacker(cfg, scorer, sensitivity=surprisals)
            result = de.attack(cand["seq"])
            result["source"] = cand["source"]
            result["name"] = cand["name"]
            result["chain_id"] = chain_id
            result["orig_ppl"] = cand["init_ppl"]
            results.append(result)

        chain_results[chain_id] = results

        best = max(results, key=lambda r: r["final_ppl"])
        best_per_chain[chain_id] = best["attacked_seq"]
        print(
            f"\n  [Chain {chain_id}] best PPL: {best['orig_ppl']:.2f} "
            f"-> {best['final_ppl']:.2f} "
            f"({'SUCCESS' if best.get('attack_success') else 'below target'})"
        )

        _export_chain_af3_job(
            seq=best["attacked_seq"],
            chain_id=chain_id,
            result=best,
            cfg=cfg,
            seed=af3_seed,
        )

    _export_combined_af3_job(best_per_chain, chain_ids, cfg, seed=af3_seed)
    _print_complex_summary(chain_results, cfg)
    return chain_results


def _export_chain_af3_job(
    seq: str,
    chain_id: str,
    result: Dict,
    cfg: PipelineConfig,
    seed: int = 42,
) -> None:
    job_name = f"complex_chain{chain_id}_{result.get('source', 'attacked')}"
    job = make_af3_job_single_chain(seq, chain_id, job_name, seed)
    fname = os.path.join(
        cfg.output_dir,
        f"chain_{chain_id}_{result.get('source', 'attacked')}_ppl{result['final_ppl']:.0f}.json",
    )
    with open(fname, "w") as f:
        json.dump(job, f, indent=2)
    print(f"  [Export] Chain {chain_id} -> {fname}")


def _export_combined_af3_job(
    chain_seqs: Dict[str, str],
    chain_order: List[str],
    cfg: PipelineConfig,
    seed: int = 42,
) -> None:
    sequences = [
        {"proteinChain": {"sequence": chain_seqs[c], "count": 1}}
        for c in chain_order
    ]
    job = [
        {
            "name": "complex_attacked_" + "_".join(chain_order),
            "modelSeeds": [seed],
            "sequences": sequences,
            "dialect": "alphafoldserver",
            "version": 1,
        }
    ]
    fname = os.path.join(
        cfg.output_dir,
        f"complex_{''.join(chain_order)}_combined.json",
    )
    with open(fname, "w") as f:
        json.dump(job, f, indent=2)
    print(f"\n[Export] Combined multi-chain AF3 job -> {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradient-free Adversarial Protein Pipeline: DE + ESM-2 OFS PPL oracle"
    )
    parser.add_argument("--pdb", type=str, default=None, help="Path to PDB file")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID for single-chain PDB mode")
    parser.add_argument("--ppl-target", type=float, default=25.0,
                        help="ESM-2 PPL threshold for attack success (default: 25.0)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top candidates to attack")
    parser.add_argument("--de-pop", type=int, default=16, help="DE population size")
    parser.add_argument("--de-gen", type=int, default=6, help="DE number of generations")
    parser.add_argument("--output-dir", type=str, default="af3_attack_jobs",
                        help="Output directory for AF3 JSON files")
    parser.add_argument("--esm2-model", type=str, default=None,
                        help="ESM-2 model name or local path (default: facebook/esm2_t6_8M_UR50D)")
    parser.add_argument("--protgpt2-path", type=str, default=None,
                        help="Local path to pre-downloaded ProtGPT2 weights")
    parser.add_argument("--af3-seed", type=int, default=42,
                        help="Single integer seed for AF3 modelSeeds")
    parser.add_argument("--use-tricks", action="store_true", default=True,
                        help="Include known adversarial trick sequences")
    parser.add_argument("--tricks-only", action="store_true",
                        help="Only run trick sequences (skip ProtGPT2 generation)")
    parser.add_argument("--n-generated", type=int, default=None,
                        help="Number of ProtGPT2 sequences to generate")
    parser.add_argument("--complex", action="store_true",
                        help="Attack all chains of a protein complex (requires --pdb)")
    parser.add_argument("--chains", type=str, default=None,
                        help="Comma-separated chain IDs for complex mode, e.g. 'A,B,C'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_kwargs = {}
    if args.protgpt2_path:
        cfg_kwargs["protgpt2_model_path"] = args.protgpt2_path
    if args.esm2_model:
        cfg_kwargs["esm2_model_name"] = args.esm2_model

    cfg = PipelineConfig(
        ppl_target=args.ppl_target,
        top_k_attack=args.top_k,
        de_pop_size=args.de_pop,
        de_generations=args.de_gen,
        output_dir=args.output_dir,
        **cfg_kwargs,
    )

    if args.complex:
        if not args.pdb:
            raise ValueError("--complex requires --pdb")
        chain_ids = (
            [c.strip() for c in args.chains.split(",")]
            if args.chains else None
        )
        run_complex_attack(
            cfg=cfg,
            pdb_path=args.pdb,
            chain_ids=chain_ids,
            af3_seed=args.af3_seed,
        )
    else:
        n_generated = 0 if args.tricks_only else args.n_generated
        results = run(
            cfg=cfg,
            pdb_path=args.pdb,
            chain_id=args.chain,
            use_tricks=args.use_tricks,
            n_generated=n_generated,
        )
        if args.af3_seed != 42:
            print(f"\n[Re-export] Writing AF3 JSONs with seed={args.af3_seed}...")
            for r in results:
                job = make_af3_job(r["attacked_seq"], r["name"], seed=args.af3_seed)
                fname = os.path.join(
                    cfg.output_dir,
                    f"{r['name']}_ppl{r['final_ppl']:.0f}.json"
                )
                with open(fname, "w") as f:
                    json.dump(job, f, indent=2)
