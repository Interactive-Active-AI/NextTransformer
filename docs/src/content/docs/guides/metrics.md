---
title: Training metrics
description: What the trainer writes during a run, and optional experiment tracking.
---

Every run directory is self-contained. The trainer writes JSON files that the audit and the exporter read; nothing
depends on a tracking service.

| File | Contents |
|---|---|
| `metrics.jsonl` | One event per line: `train`, `validation`, `core`, `final_bpb`, `budget_anomaly` |
| `provenance.json`, `resume_<step>.json` | The configuration, fingerprints, software and batch layout of each training segment |
| `run_status.json`, `budget.json` | Phase, final status and the cumulative compute ledger |
| `val_<step>.json`, `core_<step>.json`, `core_final.json`, `bpb_final.json` | Evaluation results at the scheduled points |
| `checkpoint_<step>/`, `latest.json` | Checkpoints every 5,000 updates and at the end |

`train` events carry the optimizer step, input tokens, loss and auxiliary loss, gradient norm, learning-rate
multiplier and the cumulative theoretical FLOPs with their deviation from the admitted proof; methods add their own
diagnostics (sampled depth, router statistics, sparse selection). Training logs every 10 updates by default
(`--log-every`), plus the first and the last.

## Optional tracking

Weights & Biases logging is off by default (`--wandb-mode disabled`). `offline` and `online` modes log numbers,
the scientific configuration and provenance from rank 0; models, data, source and console text are never uploaded.
`--wandb-metrics core` uploads a compact profile and `full` every numerical field. Keys carry event prefixes such as
`train/loss`, `validation/val_bpb`, `core/core_metric` and `final_bpb/val_bpb`. Credentials come from the usual
`WANDB_API_KEY` or a mounted netrc file, never from configuration files.
