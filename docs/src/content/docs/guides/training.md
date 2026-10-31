---
title: Training
description: The trainer, its modes and options, resume, re-auditing configurations, and validation checks.
---

## Entry points

| Entry point | Use |
|---|---|
| `python -m nexttransformer.train` (`launch/train.sh`) | Every registered method on the 8-, 16- and 32-rank layouts: chunked linear backends, `--sparse-backend kernel` for sparse methods (CORE always uses the reference path), Ouro's gate-only stage, segmented resume |
| `python -m nexttransformer.native.train` (`launch/native_train.sh`) | The historical native track: SSSL, MoR, LoopFormer and Circle Loop, with matrix-FLOPs reporting |

The trainer refuses a prepared tokenizer that differs from the pinned one. By default it trains in **research mode**:
stale cost evidence (changed source, another PyTorch build) or a cost outside the budget is printed and recorded, and
the run can never be exported as a result. **`--formal`** enforces cost admission and is required for results; a run
keeps its mode across segments.

```bash
# a bounded smoke run on prepared resources
launch/train.sh --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn-smoke --micro-batch 8 --smoke-steps 2

# a sparse method with the kernel backend
launch/train.sh --config configs/sparse/qsa.json --site-config site.local.json --sparse-backend kernel \
  --output runs/qsa-smoke --micro-batch 8 --smoke-steps 2
```

| Option | Meaning |
|---|---|
| `--config` | The admitted scientific configuration |
| `--output` | A fresh run directory for formal runs |
| `--micro-batch` | 2,048-token rows per GPU; accumulation fills the fixed global batch |
| `--formal` / `--research` | Enforce admission, or record findings and never publish (default) |
| `--resume auto` | Continue from the newest complete checkpoint, or start if none |
| `--max-wall-hours` | End a segment at a checkpoint before a time limit |
| `--sparse-backend {reference,kernel}` | Sparse attention implementation for training |
| `--deterministic-attention` | Reject FlashAttention-3 and disable SDPA flash, for deterministic-resume checks |
| `--smoke-steps`, `--smoke-val-tokens`, `--smoke-core-per-task` | Bounded tests; smoke runs never become results |
| `--log-every` | Training log interval (default 10 updates) |

Linear methods on CUDA with the chunked backend need the hash-pinned `requirements/linear.txt` and run with
`NANOCHAT_DTYPE=bfloat16` and `FLA_DISABLE_BACKEND_DISPATCH=1`. The optimizer compiler also needs `setuptools`.

All six registered hybrids use the same entry point with their files under `configs/hybrid/`.

## Resume and recovery

`--resume auto` checks the source and software (including the sparse backend), configuration, random number
generators, training stage and budget continuity. Resume a run from the source it started with.

Recovery makes three separate claims: the training state is restored, data and counters continue exactly, and the
numbers agree with an uninterrupted run. A checkpoint that loads proves only the first. Bounded checks show
bitwise-identical continuation on specific tested paths; long real continuations can agree only to rounding. To claim a
recovery guarantee for a new path, record the exact revision, configuration and proof, tokenizer and data identity,
PyTorch/CUDA/NCCL and kernel selection, rank and microbatch layout, compiler and cache policy, and compare a bounded
uninterrupted trajectory with a save, exit and fresh-process continuation, under tolerances registered in advance.

## Re-audit configurations

Run from the repository root in the target PyTorch environment, with a new output directory for each audit:

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 \
python -m nexttransformer.protocol.calibrate --workers 4 --reuse-configs configs \
  --cache .agent-work/cost-audit/cache --output .agent-work/cost-audit/configs \
  --results .agent-work/cost-audit/proofs
```

The audit runs on CPU, without data or GPU training. `--reuse-configs` keeps the existing structures fixed; omitting it
runs the registered candidate search. Never overwrite historical proof directories.

## Validation checks

```bash
# fast suite
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m pytest -q -n auto

# full-size BF16 architecture checks on one node; ranks test disjoint models
NANOCHAT_DTYPE=bfloat16 python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m nexttransformer.checks.validate --mode target --config-dir configs --output .agent-work/full-context.json
```

`nexttransformer.checks.distributed` checks a synchronized global optimizer update, `nexttransformer.checks.integration`
checks resources, data and the evaluator together, and `nexttransformer.checks.fla` compares the linear backends with
numerical oracles; see each command's `--help`. Scores from randomly initialized models in these checks are not
quality results.
