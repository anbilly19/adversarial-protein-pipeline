#!/bin/bash

#SBATCH --output=/netscratch/billimoria/slurm/alpha/%x-2025-09-13-20-22-45-%j-%N.out
#SBATCH --partition=A100-80GB,A100-RP,H100,H100-RP,H200,H200-SDS
#SBATCH --job-name="basesegform_esm_protgpt2"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=0-01:10:00
#SBATCH --exclude=serv-3338

srun -K \
  --export=ALL,NLTK_DATA=/netscratch/$USER/NLTK_DATA/,TQDM_DISABLE=1,HF_HOME=/fscratch/billimoria/HF_HOME/ \
  --container-image=/netscratch/billimoria/alphahack_v2.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/billimoria:/netscratch/billimoria,/fscratch/billimoria:/fscratch/billimoria,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python "$@"
  # bash -c 'source /miniconda/etc/profile.d/conda.sh && conda activate py39-esmfold && python "$@"' -- "$@"


# sbatch run_script.sh \
#   run_pipeline.py \
#   --protgpt2-path /netscratch/billimoria/weights/ProtGPT2 \
#   --esmfold-path /netscratch/billimoria/weights/esmfold_v1 \
#   --esm-if1-checkpoint /netscratch/billimoria/weights/esm_if1_gvp4_t16_142M_UR50.pt \
#   --steps 300 \
#   --plddt-target 90.0 \
#   --top-k 5 \
#   --n-generated 20 \
#   --output-dir /netscratch/billimoria/adversarial_output \
#   --use-tricks \
#   --device cuda
