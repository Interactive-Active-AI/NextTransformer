#!/usr/bin/env bash
# Launch nexttransformer.train with torchrun, on one node or on each node of a multi-node job.
#
#   One node with eight GPUs:
#     launch/train.sh --config configs/baseline/dense_full.json --output runs/dense_full --formal
#
#   Submit a multi-node job with Slurm (four nodes, eight GPUs each):
#     sbatch --nodes=4 launch/slurm.sbatch --config configs/linear/gdn.json --site-config site.local.json --output runs/gdn --micro-batch 16 --formal --resume auto
#
# Environment: NNODES (default 1), NPROC_PER_NODE (8), NODE_RANK (0), MASTER_ADDR (127.0.0.1),
# MASTER_PORT (29500), PYTHON (python), FLA_DISABLE_BACKEND_DISPATCH (1). DRY_RUN=1 prints the command instead of running it.
# Data, caches and tracking come from --site-config or NEXTTRANSFORMER_* variables; see the
# configuration guide. Formal runs use one of the registered layouts: 8, 16 or 32 GPUs in total.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
WORLD=$((NNODES * NPROC_PER_NODE))
# Chunked linear backends (fla_chunk64) run the registered FLA Triton path; the trainer refuses to start without it.
export FLA_DISABLE_BACKEND_DISPATCH="${FLA_DISABLE_BACKEND_DISPATCH:-1}"

smoke=0
for arg in "$@"; do
  [[ "$arg" == --smoke-steps* ]] && smoke=1
done
if [[ "$smoke" == 0 && "$WORLD" != 8 && "$WORLD" != 16 && "$WORLD" != 32 ]]; then
  echo "train.sh: $NNODES node(s) x $NPROC_PER_NODE GPU(s) = $WORLD ranks; formal and research runs use 8, 16 or 32" >&2
  exit 2
fi

if [[ "$NNODES" == 1 ]]; then
  LAUNCH=(--standalone --nnodes 1 --nproc-per-node "$NPROC_PER_NODE")
else
  LAUNCH=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
          --nproc-per-node "$NPROC_PER_NODE")
fi

COMMAND=("$PYTHON" -m torch.distributed.run "${LAUNCH[@]}" -m nexttransformer.train "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"; echo
  exit 0
fi
cd "$ROOT"
exec "${COMMAND[@]}"
