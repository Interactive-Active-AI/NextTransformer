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

## 3. Submit multi-node training

<span id="multi-node"></span>

Install Accelerate once in the shared training environment (`uv pip install accelerate==1.15.0`). The checkout, `site.local.json`, data and output directory must be accessible at the same paths on every node. From the repository root on the Slurm login node, submit:

```bash
sbatch launch/slurm.sbatch
```

The example requests two eight-GPU nodes and trains GDN under the formal protocol with a per-GPU microbatch of 16, writing to `runs/gdn`. Slurm runs one Accelerate launcher on each node via `srun`. Each launcher starts one training process per local GPU. Slurm supplies the host address and node ranks; no hostfile or worker-to-worker SSH is needed. Set partition, account and time limit with `sbatch` as required by your cluster.

To change the model or node count, pass the full trainer arguments after the script name. For example, to train a hybrid on four eight-GPU nodes:

```bash
sbatch --nodes=4 launch/slurm.sbatch --config configs/hybrid/gdn_hybrid.json --site-config site.local.json \
  --output runs/gdn-hybrid --micro-batch 8 --formal
```

### Other managed training platforms

If your platform runs a job entry point once per node, submit **one** multi-node job with this command as its entry point. Map the platform's node count, GPUs per node, zero-based node rank, and main node address and port to the variables below. The `launch/accelerate.yaml` configuration uses PyTorch DDP and defaults to two eight-GPU nodes. The launch arguments below override that topology; `--num_processes` is the total GPU count across the job. The platform dispatches the command to each node.

```bash
accelerate launch --config_file launch/accelerate.yaml \
  --num_machines "$NNODES" --num_processes "$((NNODES * GPUS_PER_NODE))" \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --module nexttransformer.train --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal
```

Accelerate launches the processes. The trainer uses native PyTorch DDP and the registered Muon + AdamW optimizer. Resubmission resumes from the latest complete checkpoint in the same output directory by default.

<span id="hostfile"></span>

For independently managed SSH hosts, the [DeepSpeed hostfile launcher](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) is an alternative. The scheduler path above does not require DeepSpeed.

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
