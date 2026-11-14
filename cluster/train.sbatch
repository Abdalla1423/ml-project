#!/bin/bash
# Pre-training job for the Hydra cluster (owner: B).
#
# Run from the repo root on hydra:
#   mkdir -p logs
#   sbatch cluster/train.sbatch                                   # default config, gpu-5h
#   sbatch --partition=gpu-test --time=00:15:00 cluster/train.sbatch configs/train/debug.yaml --max-minutes 10
# The job resumes from checkpoint_latest.pt in the output directory if it exists, so a run that
# hits the time limit continues when the same command is submitted again.
#SBATCH --job-name=dopfn-train
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb|40gb"
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/train-%j.out

set -euo pipefail

CONFIG=${1:-configs/train/default.yaml}
shift || true
SIF=${DOPFN_SIF:-$HOME/containers/dopfn.sif}

export PYTHONPATH="$PWD/src:$PWD/Do-PFN-prior"
echo "job $SLURM_JOB_ID on $(hostname), config $CONFIG, commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stop cleanly 15 min before the 5 h limit; later arguments override earlier ones.
apptainer exec --nv "$SIF" python scripts/train.py --config "$CONFIG" --device cuda --resume --max-minutes 285 "$@"
