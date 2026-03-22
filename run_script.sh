#!/bin/bash

#SBATCH --output=/netscratch/billimoria/slurm/%x-2025-09-13-20-22-45-%j-%N.out
#SBATCH --partition=A100-80GB,A100-RP,H100,H100-RP,H200,H200-SDS
#SBATCH --job-name="basesegform_vit_crossattn"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=0-06:10:00
#SBATCH --exclude=serv-3338

srun -K \
  --export=ALL,NLTK_DATA=/netscratch/$USER/NLTK_DATA/,TQDM_DISABLE=1,HF_HOME=/fscratch/billimoria/HF_HOME/ \
  --container-image=/netscratch/billimoria/esmfold_root_runtime.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/billimoria:/netscratch/billimoria,/fscratch/billimoria:/fscratch/billimoria,/ds-sds:/ds-sds:ro,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python "$@"
