---
title: Optimizer, schedule and layouts
description: The shared optimization recipe and the registered distributed data layouts.
---

## Optimizer

Muon updates the ordinary block matrices; the existing AdamW groups update embeddings, the output head, value
embeddings and native scalars. Auxiliary gates, indexers, halters, residual maps and pseudo-queries are registered
explicitly; every trainable parameter appears in exactly one group.

| Group | Base learning rate |
|---|---|
| Muon | **0.02** |
| Embedding | **0.3** |
| Unembedding | **0.008** |
| Scalars | **0.5** |
| Auxiliary AdamW | **0.0006** |

The existing square-root global-batch scaling and native width scaling apply on top; these are base values, not
necessarily the final group rates. Warmup is **40** updates; the final **65%** of training decays linearly to **5%** of
the peak rate. Weight decay is **0.28** with the common d24 scaling and the native Muon schedule. There is no default
global gradient clipping; any method-specific auxiliary learning-rate or clipping exception is preregistered before
pilots.

## Distributed layouts

The data always follows eight fixed logical streams. Three layouts are registered:

| World size | Logical streams | Ranks per stream | Per-rank microbatch × accumulation |
|---:|---:|---:|---:|
| 8 | 8 | 1 | 64 |
| 16 | 8 | 2 | 32 |
| 32 | 8 | 4 | 16 |

Adjacent ranks that share a stream pack all of its rows and keep their contiguous share of each update, so the global
row set, ordering, row IDs and stream state at every update boundary equal an eight-rank run. Validation uses the same
sharing; CORE distributes question IDs without changing the question set. Other world sizes are for smoke tests only.
Four independent eight-GPU jobs are not one 32-GPU job.

Different GPU types may change the per-rank microbatch and accumulation after a preflight check, without changing the
global tokens per update. Kernel, BF16 and reduction-order differences are recorded with the run. Auxiliary losses
that depend nonlinearly on the microbatch need tests and disclosure: averaging squared microbatch means is not the
square of a global mean.

The [quick start](../../getting-started/quick-start/) shows how to launch one node or several.
