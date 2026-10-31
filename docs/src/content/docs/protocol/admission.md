---
title: Calibration and admission
description: How a method's structure is chosen to fit the shared budget, and what admission checks before a formal run.
---

## Thresholds

| Check | Threshold |
|---|---|
| Preflight (the audited configuration) | within **±1%** of the SSSL anchor |
| Final cumulative compute of the run | within **±2%** |
| Rolling deviation during training | recorded as an anomaly above **3%** |

The thresholds are frozen before any run and never relaxed for one method.

## Calibration

```bash
python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/cost-cache \
    --output configs --results evidence/costs/<proof-set>
```

New stacks start at 24 layers and FFN width 6144 and search **16–32 layers** within the registered layout. If needed,
the FFN width is adjusted within **[3d, 5d]** in multiples of **128**. Among candidates within 1%, the preferred one has
the FFN closest to 6144, then the smallest absolute cost error, then the depth closest to 24, then the shallower depth.
A full candidate table is published if monotonic assumptions fail. Existing configurations keep their recorded
selection provenance.

Padding with useless computation, training on fewer input tokens, early stopping, seed shopping and selection on
formal CORE are not allowed. If the grid cannot meet the budget, the configuration is recorded as
`calibration_pending` and a revised grid is preregistered.

Calibration writes the configuration under `configs/<family>/<method>.json` and its proof under `evidence/costs/`, next
to the SSSL anchor proof. Report parameters by component (embeddings and value embeddings, mixer, FFN, routers and
residual maps), physical blocks and unfolded calls, the forward, backward, auxiliary and optimizer costs, and stage
weights.

## Admission

`nexttransformer.protocol.admission.admit_config` refuses a configuration whose proof is stale or unmatched, was made
with a different PyTorch build, has changed model or protocol hashes, contains unknown operators or missing stages, or
lies more than 1% from the anchor. Formal runs (`python -m nexttransformer.train --formal`) require admission;
research mode records the findings instead and its runs can never be published.

During training, `BudgetState` tracks optimizer steps, tokens, cumulative compute and the completed evaluation events;
an incomplete run cannot be `matched`.

The PyTorch build the proofs record is **2.9.1+cu128**. The audit runs on CPU, so any machine with that build can
reproduce a proof; maintainers re-audit a new method in that environment before admitting it.
