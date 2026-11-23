---
title: MoE results
description: Published formal runs on the MoE tracks, with MoE-CORE, validation BPB and the task breakdown.
---

Every formal run published on a MoE track is listed here. This page is generated from `moe/results/` by
`python docs/scripts/generate.py`.

MoE-CORE, its short and long halves and every task score are shown ×100. Δ columns are differences from the track's
baseline. Each method links to its run record, which names the audit that admitted it. Relative throughput is the ratio
`moe.throughput` measures against the baseline back to back on one node, so the baseline itself is the reference.

Validation BPB is the full evaluation that follows the last update, over every row of the six sources, where lower is
better. The validation every 500 updates reads a fixed sample per source instead and lands lower, so the two are not
comparable. The last scheduled MoE-CORE evaluation and the final one are separate passes, which is why the last row of
the schedule below differs from the headline by a few hundredths.

## d768 · 240 TPP

53,686 updates · 14.07B input tokens · 240 tokens per active parameter · 2.00B total and 213M active parameters.

| Method | MoE-CORE ↑ | Δ | Val BPB ↓ | Δ | Short | Long | Relative throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| **[MoE Baseline](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/results/d768-240tpp/baseline.json)** | 36.82 | — | 0.7708 | — | 20.40 | 77.87 | — |

### MoE-CORE by task

| Task | Group | MoE Baseline |
|---|---|---:|
| ARC-E | short | 47.76 |
| ARC-C | short | 9.22 |
| HellaSwag | short | 22.62 |
| PIQA | short | 37.65 |
| WinoGrande | short | 1.97 |
| CSQA | short | 7.04 |
| OBQA | short | 12.53 |
| LAMBADA | short | 43.82 |
| SQuAD | short | 15.81 |
| TriviaQA | short | 5.59 |
| RULER S-NIAH-1 2K | long | 100.00 |
| RULER S-NIAH-1 4K | long | 84.20 |
| FDA | long | 66.61 |
| SWDE | long | 60.67 |

### MoE-CORE over training

| Update | MoE Baseline |
|---:|---:|
| 12,726 | 29.41 |
| 25,451 | 31.52 |
| 38,177 | 33.96 |
| 53,686 | 36.83 |

## d1024 · 240 TPP

No published run yet.

## d768

No published run yet.

## d1024

No published run yet.
