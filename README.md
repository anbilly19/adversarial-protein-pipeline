# Adversarial Protein Pipeline

A research pipeline for probing the robustness of AlphaFold3 and ESMFold confidence scores via white-box adversarial attacks. Combines three complementary attack strategies into a unified, modular codebase.

## Overview

AlphaFold3's confidence metrics (pLDDT, PAE, ipTM) measure internal model consistency, not physical reality. This pipeline exploits that gap through:

| Strategy | Method | Key Insight |
|---|---|---|
| **ESM-Design attack** | Gradient descent through ESMFold | Replaces discrete tokenizer with soft embeddings — makes the entire forward 
| **Evolutionary attack** | Differential Evolution with mixed mutation operators (replacement, deletion, insertion) | Black-box optimization via AF2-Mutation strategies — naturally avoids overfitting to ESMFold || **BLOSUM62 mutations** | Conservative substitutions at gradient-sensitive positions | Looks biologically valid but maximally disrupts co-evolutionary signal |
| **Inverse folding** | ESM-IF1 at high temperature | Generates structurally plausible sequences that deviate from the native — ideal adversarial seeds |
| **Trick sequences** | Known fold-switchers + chameleons |
Hard-coded adversarial seeds from fold-switching literature |

## Repository Structure

```
adversarial-protein-pipeline/
  pipeline/
    __init__.py          # Package exports
    config.py            # PipelineConfig dataclass
    trick_sequences.py   # Fold-switchers, chameleons, BLOSUM baselines + BLOSUM62 table
    protgpt2.py          # ProtGPT2 generation + perplexity filtering
    esmfold.py           # ESMFold scoring + ESM-Design gradient attack
    inverse_fold.py      # ESM-IF1 inverse folding + BLOSUM adversarial mutations
  run_pipeline.py        # Main orchestration script (CLI)
  requirements.txt
```

## Installation

```bash
git clone https://github.com/anbilly19/adversarial-protein-pipeline
cd adversarial-protein-pipeline
pip install -r requirements.txt
```

> **Note:** `fair-esm` requires a separate install step for ESM-IF1:
> ```bash
> pip install fair-esm
> # For ESMFold via HuggingFace (alternative):
> pip install transformers accelerate
> ```

## Usage

### Mode 1 — Trick sequences only (fastest, no GPU required for generation)
```bash
python run_pipeline.py --tricks-only --steps 300 --plddt-target 88
```

### Mode 2 — ProtGPT2 seed + ESM-Design attack
```bash
python run_pipeline.py --n-generated 30 --steps 300
```

### Mode 3 — Start from PDB structure
```bash
# Download RfaH (known fold-switcher) from PDB: 2LCL
python run_pipeline.py --pdb rfah.pdb --chain B --steps 500 --top-k 5
```

### Mode 4 — Full pipeline
```bash
python run_pipeline.py \
  --pdb rfah.pdb --chain B \
  --use-tricks \
  --n-generated 50 \
  --steps 300 \
  --plddt-target 90 \
  --top-k 5 \
  --output-dir af3_jobs
```

### Mode 5 — Evolutionary attack only (gradient-free)

```bash
python run_pipeline.py --pdb protein.pdb --chain A --attack-method evolutionary --n-generated 0
```

### Mode 6 — Multi-chain crystalline structures

```bash
# Process all chains in a PDB file (e.g., for crystalline structures)
python run_pipeline.py --pdb protein.pdb --all-chains --attack-method gradient --top-k 3
```

Each chain's mutations will be exported as separate AF3 JSON files with chain IDs in filenames:
- `if_00_chainA_plddt92.json`
- `if_00_chainB_plddt88.json`
- etc.

### Mode 7 — Both attack methods (gradient + evolutionary)

```bash
python run_pipeline.py --pdb protein.pdb --chain A --attack-method both --top-k 5
```

Outputs are AlphaFold3-compatible JSON files in `af3_jobs/` ready to submit to the AF3 server or local inference.

## Pipeline Stages

```
Stage 0  Trick sequences     Hardcoded fold-switchers, chameleons, BLOSUM variants
Stage 1  Inverse folding     ESM-IF1 from PDB backbone (high-temperature sampling)
         BLOSUM mutations    Gradient-guided conservative substitutions on native seq
Stage 2  ProtGPT2            Generate + perplexity-filter (low PPL -> high pLDDT proxy)
Stage 3  ESMFold scoring     Batch pLDDT pre-score all candidates; sort descending
Stage 4  ESM-Design attack   Gradient descent on top-k; Gumbel-softmax + temp annealing
Stage 5  AF3 export          Write JSON jobs for each successful attack
```

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `esm_design_steps` | 300 | Gradient steps — 300 sufficient for pLDDT maximization |
| `esm_lr` | 0.01 | Adam LR — keep <= 0.05 |
| `esm_temp` / `esm_temp_final` | 1.0 / 0.1 | Gumbel-softmax temperature annealing |
| `plddt_target` | 90.0 | Attack success threshold |
| `n_mutations` | 5 | BLOSUM positions to mutate |
| `if_temperature` | 1.5 | ESM-IF1 sampling temp (higher = more diverse) |
| `top_k_attack` | 5 | Candidates to run gradient attack on |
| `attack_method` | "gradient" | "gradient" (ESM-Design), "evolutionary" (AF2-Mutation), or "both" |

## Why It Works

All three attack strategies share the same root cause: **AF3's confidence heads are trained jointly with structure prediction on PDB data**, so the gradient of confidence w.r.t. the input is just as informative as the gradient of coordinate error. The model has no mechanism to distinguish a genuinely stable protein from a hallucination that happens to match its training distribution.

## References

- Alkhouri et al. 2024 — *Probing AlphaFold's Input Attack Surface via Red-Teaming* (IEEE PST)
- Chu et al. 2024 — ESM-Design hallucination via differentiable embedding
- Hsu et al. 2022 — ESM-IF1 inverse folding (NeurIPS)
- Porter et al. 2022 — RfaH fold-switching (PNAS)
- Verkuil et al. 2022 — ESMFold (Science)
- Pak et al. 2023 — Protein design via deep learning guided coevolution (AF2-Mutation)
