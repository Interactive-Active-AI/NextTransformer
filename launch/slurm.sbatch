#!/usr/bin/env bash
# Multi-node training on a SLURM cluster: one task per node, eight GPUs per node.
#
#   sbatch --nodes=4 launch/slurm.sbatch --config configs/linear/gdn.json --output runs/gdn --formal --resume auto
#
# Adjust the partition, account, time limit and environment activation for your cluster. With
# --resume auto, resubmitting the same command continues from the newest complete checkpoint, and
# --max-wall-hours ends a segment cleanly before the scheduler's limit.
#SBATCH --job-name=nexttransformer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=%x-%j.log
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29500}" NNODES="$SLURM_NNODES" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NEXTTRANSFORMER_JOB_ID="$SLURM_JOB_ID"

srun --ntasks-per-node=1 bash -c 'NODE_RANK="$SLURM_NODEID" exec "$0/launch/train.sh" "$@"' "$REPO" "$@"
