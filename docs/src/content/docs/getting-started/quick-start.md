---
title: Quick start
description: Train a registered method under the protocol on one node or several.
---

This page trains a registered method, for example the Dense Baseline, under the full protocol: 49.152B tokens and
46,875 updates. On one node of eight current data-center GPUs this takes about two to four days for a model of
roughly 1.4B parameters, plus a few hours of evaluation.

Before you start, [install](../installation/) the GPU environment and [prepare the data](../data/).

## 1. Smoke test

A run of two updates checks that the environment, data, tokenizer and model fit together:

```bash
source launch/env.sh --site-config site.local.json
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full-smoke --micro-batch 8 --smoke-steps 2
```

Smoke runs accept any number of GPUs (for example `NPROC_PER_NODE=1`) and can never become formal results.

## 2. One node, eight GPUs

```bash
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full --micro-batch 8 --formal --resume auto
```

`launch/train.sh` wraps `torchrun` and `python -m nexttransformer.train`. `--micro-batch` is the number of 2,048-token
rows per GPU; gradient accumulation fills the fixed global batch of 512 rows, so pick the largest microbatch that fits
your GPUs' memory. `--resume auto` makes the command restartable: run it again after an interruption and it continues
from the newest complete checkpoint.

## 3. Several nodes

The protocol registers 16- and 32-GPU layouts that keep the data order of an eight-GPU run. Run the same command on
every node, changing only `NODE_RANK`:

```bash
# node 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 launch/train.sh \
  --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto
# node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=node0 launch/train.sh \
  --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto
```

`MASTER_ADDR` is a hostname or address of node 0 that every node can reach; `MASTER_PORT` defaults to 29500. With four
nodes, set `NNODES=4` and `NODE_RANK=0..3`. The output directory must be on a filesystem that all nodes share.

On SLURM, `launch/slurm.sbatch` does the same with one task per node:

```bash
sbatch --nodes=4 launch/slurm.sbatch --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto --max-wall-hours 23.5
```

`--max-wall-hours` ends a segment cleanly at a checkpoint before the scheduler's time limit; resubmit the same command
to continue.

## 4. Evaluate, audit and export

Training writes validation and CORE results at the scheduled steps, and an independent final BPB at the end. To
evaluate any checkpoint separately:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m nexttransformer.evaluate \
  --checkpoint runs/dense_full/checkpoint_00046875 --output runs/dense_full/evaluation.json \
  --micro-batch 8 --eval bpb,core
```

A finished formal run is audited and exported in two commands, then submitted by pull request:

```bash
python -m nexttransformer.audit_run runs/dense_full --method dense_full
python -m nexttransformer.export_run runs/dense_full --method dense_full --name "Dense Baseline" \
  --audit evidence/audits/dense_full.json
```

See [submit a result](../../guides/submit-a-result/) for the pull request, and the [training guide](../../guides/training/)
for every trainer option.
