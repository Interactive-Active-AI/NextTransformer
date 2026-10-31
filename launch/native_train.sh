#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ " $* " == *" --help "* || $# == 0 ]]; then
  echo 'Usage: launch/native_train.sh dense|mor|loopformer|circle --site-config FILE [--micro-batch N] [--dry-run] [--preflight] [training options]'
  exit 0
fi
MODEL="$1"; shift
case "$MODEL" in dense|mor|loopformer|circle) ;; *) echo 'Unsupported native model' >&2; exit 2;; esac
source "$ROOT/launch/env.sh" "$@"
cd "$ROOT"
MICRO_BATCH="${MICRO_BATCH:-8}"
WORLD="${GPUS:-8}"
RUN_MODEL="$MODEL"
if [[ "$MODEL" == circle ]]; then RUN_MODEL=circle_sparse; fi
if [[ "$MODEL" == mor || "$MODEL" == loopformer ]]; then RUN_MODEL="${MODEL}_core393"; fi
OUTPUT="${OUTPUT_DIR:-$NANOCHAT_RUNTIME/runs/${NEXTTRANSFORMER_RUN_NAME:-native_${RUN_MODEL}_seed42}}"
ARGS=(--config "$ROOT/configs/native/$MODEL.json" --micro-batch "$MICRO_BATCH" --world-size "$WORLD" --output "$OUTPUT")
if [[ " $* " == *" --dry-run "* ]]; then
  exec "$NANOCHAT_VENV/bin/python" -m nexttransformer.native.train "${ARGS[@]}" "$@"
fi
if [[ " $* " == *" --preflight "* ]]; then
  EXTRA=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preflight) shift ;;
      --output) OUTPUT="$2"; shift 2 ;;
      --output=*) OUTPUT="${1#*=}"; shift ;;
      *) EXTRA+=("$1"); shift ;;
    esac
  done
  exec "$NANOCHAT_VENV/bin/python" -m nexttransformer.native.preflight --config "$ROOT/configs/native/$MODEL.json" --micro-batch "$MICRO_BATCH" --output "$OUTPUT/preflight.json" "${EXTRA[@]}"
fi
exec "$NANOCHAT_VENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0 --nproc_per_node="$WORLD" -m nexttransformer.native.train "${ARGS[@]}" "$@"
