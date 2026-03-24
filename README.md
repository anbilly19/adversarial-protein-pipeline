# Adversarial Protein Pipeline

A gradient-free pipeline for generating adversarial protein sequences that confuse AlphaFold3's confidence scoring.

## Branches

| Branch | Oracle | GPU Required | Attack Method |
|---|---|---|---|
| `stable-gradient-attack` | ESMFold (local) | ✅ Yes | ESM-Design gradient attack |
| `gradient-free-de-attack` | ESM-2 OFS PPL (CPU) | ❌ No | Differential Evolution |

---

## `gradient-free-de-attack` — Overview

This branch replaces all gradient-based components with a **black-box Differential Evolution (DE)** loop. The fitness oracle is **ESM-2 OFS pseudo-perplexity** — a single forward pass per sequence, no backpropagation, runs on CPU with the 8M parameter model (~35 MB).

### Key idea

Higher ESM-2 pseudo-perplexity (PPL) means the model considers the sequence evolutionarily implausible, which correlates with low predicted pLDDT. Attacking toward higher PPL therefore acts as a proxy for disrupting AlphaFold3's confidence.

### Hardware requirements

- **CPU only** — no CUDA, no GPU needed
- ~4 GB system RAM
- ESM-2 8M model: ~35 MB download
- ProtGPT2 (optional): ~1.5 GB download

### Install

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Trick sequences only (no PDB, fastest)
python run_pipeline.py --tricks-only

# From PDB (single chain)
python run_pipeline.py --pdb rfah.pdb --chain A

# From PDB (protein complex, all chains)
python run_pipeline.py --complex --pdb 7k3g.pdb

# From PDB (specific chains)
python run_pipeline.py --complex --pdb 7k3g.pdb --chains A,B

# Tune DE budget
python run_pipeline.py --pdb rfah.pdb --de-pop 32 --de-gen 8 --ppl-target 30.0

# Use a larger ESM-2 model for a stronger oracle
python run_pipeline.py --pdb rfah.pdb --esm2-model facebook/esm2_t12_35M_UR50D
```

### Pipeline stages

```
Stage 0  Load hardcoded trick sequences (fold-switchers, chameleons, repeats)
Stage 1  Read native sequence from PDB + ESM-2 surprisal-guided BLOSUM seeds
Stage 2  ProtGPT2 generation + perplexity filter          (optional)
Stage 3  Score all candidates: ESM-2 OFS pseudo-perplexity (CPU, ~7 ms/seq)
Stage 4  Differential Evolution attack on top-k candidates
Stage 5  Export attacked sequences as AlphaFold3 Server JSON files
```

### DE configuration

| Parameter | Default | Description |
|---|---|---|
| `--de-pop` | 16 | Population size (≥ 4) |
| `--de-gen` | 6 | Generations |
| `--ppl-target` | 25.0 | PPL threshold for success |
| `--top-k` | 5 | Candidates to attack |
| `n_mutations` | 3 | Mutations per individual (DE budget b) |

### Oracle sizing

| ESM-2 model | Params | Size | Speed (CPU) | Accuracy proxy |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 8M | ~35 MB | ~7 ms/seq | Fast, good enough for DE |
| `esm2_t12_35M_UR50D` | 35M | ~140 MB | ~30 ms/seq | Better sensitivity |
| `esm2_t30_150M_UR50D` | 150M | ~600 MB | ~120 ms/seq | Strong oracle |

### Architecture

```
pipeline/
  config.py          PipelineConfig (no torch.cuda dependency)
  esm2_scorer.py     ESM-2 OFS pseudo-perplexity oracle  [NEW]
  de_attacker.py     DE/rand/1/bin attacker               [NEW]
  inverse_fold.py    BLOSUMAttack (random + sensitivity-guided, no ESM-IF1)
  trick_sequences.py Hardcoded adversarial sequences      [unchanged]
  protgpt2.py        ProtGPT2 generator                   [unchanged]
  pdb_utils.py       PDB sequence reading                 [unchanged]
  __init__.py        Package exports
```

### References

- Xu et al. 2023 — AF2-Mutation: Adversarial Sequence Mutations against AlphaFold2 ([arXiv:2305.08929](https://arxiv.org/abs/2305.08929))
- Ferruz et al. 2025 — Pseudo-perplexity in one fell swoop
- Alkhouri et al. 2024 — Gradient-guided BLOSUM attack (IEEE PST)
- Storn & Price 1997 — Differential Evolution (J. Global Optimization)
