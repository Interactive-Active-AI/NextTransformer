---
title: Protocol FAQ
description: Common questions about why the protocol is the way it is.
---

### Why match training compute instead of parameters?

Many next-generation architectures trade parameters for computation: loops reuse weights, sparse attention skips
work, linear attention changes how cost grows with length. Matching parameters would hide those trade-offs. Matching
the theoretical training FLOPs per token asks the question practitioners face: with this much compute, what should a
model spend it on?

### Why theoretical FLOPs and not measured time?

Wall time depends on hardware, kernels and engineering effort that differ across methods and change quickly. The
operator ledger is reproducible on any machine, counts what the algorithm requires, and keeps kernel padding and
redundancy out of the budget.

### Why one seed?

A 49B-token run per method is expensive. One fixed seed keeps every method on the same data order and initialization
procedure; results are reported as observations, with single-seed limits stated. Seed noise estimates are planned when
margins call for them.

### Why is SSSL the anchor but Dense the baseline?

SSSL was the first audited configuration, and every earlier budget is defined against it, so it stays the frozen
compute anchor. Full causal attention in every layer is the natural quality reference, so `dense_full` was calibrated
to the same budget and is the only baseline for Δ values.

### Can I use the validation shard or CORE to choose my design?

No. They are reserved for the formal run. Use training loss or the dev split.

### Do I need the exact GPU type or a container?

No. Any GPU type can train under the protocol; the per-rank microbatch adapts while the global batch stays fixed.
Formal runs need the admitted configuration and the pinned PyTorch build for the cost proof, not a specific machine.

### My method needs a change to shared code. What now?

Open an issue first. Changes that keep every published method bitwise identical can be merged after a maintainer's
full audit; changes that alter published methods become a new method version. See
[changing shared code](../../guides/changing-shared-code/).
