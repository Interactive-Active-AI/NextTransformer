#!/usr/bin/env bash
# Submit from the shared checkout: sbatch launch/slurm.sbatch
# With no arguments this runs the GDN example. Pass trainer arguments for a different run.
#SBATCH --job-name=nexttransformer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=%x-%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if (( $# == 0 )); then
  set -- --config configs/linear/gdn.json --site-config site.local.json \
    --output runs/gdn --micro-batch 16 --formal
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export NEXTTRANSFORMER_JOB_ID="$SLURM_JOB_ID"

# One launcher per node; Accelerate starts one training process per allocated GPU.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'exec accelerate launch --config_file launch/accelerate.yaml \
    --num_machines "$SLURM_NNODES" --num_processes "$((SLURM_NNODES * NPROC_PER_NODE))" \
    --machine_rank "$SLURM_NODEID" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
    --module nexttransformer.train "$@"' _ "$@"
