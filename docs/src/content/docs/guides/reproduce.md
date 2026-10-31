---
title: Reproduce a published result
description: Re-run a published method under the protocol and compare with its record.
---

Every published record names its method, configuration and fingerprints, so a published result can be reproduced on
any GPUs that fit the model.

1. [Install](../../getting-started/installation/) the GPU environment with PyTorch 2.9.1 and
   [prepare the data](../../getting-started/data/); the trainer verifies the tokenizer and validation shard against the
   pinned hashes.
2. Check that the method still matches its proof: `python -m nexttransformer.protocol.fingerprint --methods <method>`.
3. Train with the admitted configuration, `--formal` and a fresh output directory, as in the
   [quick start](../../getting-started/quick-start/). Any registered layout (8, 16 or 32 GPUs) keeps the data order.
4. Audit the run with `python -m nexttransformer.audit_run`, then compare its validation BPB and CORE at update 46,875
   with `results/runs/<method>.json`.

Runs are single-seed; expect agreement in data, compute and evaluation schedule, and numerical rather than bitwise
agreement across different GPUs, kernels and rank counts. A few published runs were trained from a frozen source
revision; their records say so in `comparisonNote`, and the [run notes](../../results/runs/) explain what differs.
A reproduction that confirms or contradicts a record is welcome as a run-notes contribution.
