---
title: Installation
description: Install NextTransformer for GPU training or CPU development.
---

NextTransformer is a Python package (`nexttransformer`) plus the nanochat runtime it builds on. It uses
[uv](https://docs.astral.sh/uv/) and a locked environment (`uv.lock`), so every machine resolves the same versions.

**Requirements:** Linux with NVIDIA GPUs for training (one node of 8 GPUs, or 16 or 32 GPUs across nodes), Python
3.10–3.13, and uv. CPU-only machines can build and test models but not train them.

## GPU training environment

```bash
git clone https://github.com/Interactive-Active-AI/NextTransformer.git
cd NextTransformer
uv sync --frozen --extra gpu --extra upstream --extra tracking
source .venv/bin/activate
# Linear-attention methods on CUDA use the hash-pinned flash-linear-attention kernels:
uv pip install --no-deps --require-hashes -r requirements/linear.txt
```

The `gpu` extra installs PyTorch **2.9.1** with CUDA 12.8, the build every cost proof records. Formal runs need it;
research runs work with any build. `upstream` adds the optional FlashAttention-3 kernel hub and dataset tooling, and
`tracking` adds Weights & Biases (disabled unless you turn it on).

On a cluster where each job needs a self-contained environment, `launch/setup_environment.sh` creates one under a
persistent runtime directory from the same lock file:

```bash
bash launch/setup_environment.sh --site-config site.local.json --tracking --upstream
```

## CPU development environment

```bash
uv sync --frozen --extra cpu
export CUDA_VISIBLE_DEVICES= NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1
uv run python -m pytest -q -n auto
```

The test suite builds every registered method at debug size, checks causality, gradients, checkpoint round trips, the
cost ledger and the published records. Slow tests (full-size cost audits, multi-rank data layouts) run with
`pytest -m slow`.

## Next

- [Prepare the data](../data/) once per machine or shared filesystem.
- Follow the [quick start](../quick-start/) to train on one node or several.
