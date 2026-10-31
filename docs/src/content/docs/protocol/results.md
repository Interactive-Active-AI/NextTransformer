---
title: Results and statuses
description: What a published run contains, and the statuses a method or run can have.
---

## What a published run contains

A run joins the leaderboard when it is **complete and matched**: 46,875 updates and 49.152B input tokens, all 189
validation events, all six full CORE-22 events, and a cumulative compute audit within 2% of the anchor. Every segment
must have been trained in formal mode with the admitted configuration.

The record (`results/runs/<method>.json`) publishes:

- the periodic validation BPB at every scheduled step and the independent final BPB, separately;
- CORE at the six events, with all 22 task scores;
- the input tokens and the average and cumulative theoretical training compute, with its deviation from the anchor;
- the protocol, model, tokenizer, data and source fingerprints;
- notes on provenance whenever the run was trained from a frozen source revision.

Its audit (`evidence/audits/<method>.json`) records the checks and the data layout (number of ranks, microbatch and
accumulation). Records describe the protocol, not the machines: they contain no host names, paths or device names.
Missing measurements are `null`, never zero.

## Statuses

| Status | Meaning |
|---|---|
| `planned` | Registered for future work |
| `implementation_pending` | Being implemented |
| `calibration_pending` | Implemented; the grid has not met the budget yet |
| `preflight_passed` | Admitted within 1%; ready for a formal run |
| `running` | A formal run is in progress |
| `matched` | Complete, with every event and an accepted cumulative audit |
| `budget_mismatch` | Cumulative compute outside 2% |
| `protocol_mismatch` | A controlled input or event differs from the protocol |
| `incomplete_eval` | Training finished but evaluation events are missing |
| `failed` | The attempt ended unsuccessfully; it stays on record |

Only a complete, accepted run is `matched`. Historical SSSL results, failed runs and earlier accounting formulas are
preserved without rewriting their scores or fingerprints.

## Reading comparisons

All runs are single-seed and were trained on different machines and rank counts; orderings are observations, not
statistical results. Compute values are comparable only under the same accounting: the leaderboard marks the
historical matrix-FLOPs estimate and the frozen replay accounting separately.
