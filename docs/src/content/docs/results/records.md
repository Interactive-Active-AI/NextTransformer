---
title: Run records and model cards
description: The fields of results/runs and results/methods.
---

```text
results/
  runs/<id>.json        one record per formal run, named after its model card
  methods/<id>.json     one model card per method
  arena/<entry>.json    one disclosure card per arena entry
```

## Run records

`python -m nexttransformer.export_run` writes a record from a finished run directory and its audit;
`python -m nexttransformer.export_run --check` validates every record in CI. The exporter refuses a run unless all 189
validation and six full CORE-22 events are present, the per-task CORE scores reproduce the aggregate, the cumulative
audit is `matched`, every segment was formal, and the run used the pinned tokenizer and the frozen protocol.

| Field | Meaning |
|---|---|
| `id`, `methodId`, `cohort`, `name` | Run id, registered method (SSSL keeps `dense`), accounting group for the checks, display name |
| `status`, `step`, `complete` | `finished` or `partial`; last update; whether the full protocol was met and accepted |
| `val`, `core` | `{step, value}` points: validation BPB at every scheduled step; CORE on its 0–1 scale |
| `taskEvaluations` | Per CORE event: `raw` accuracy and `centered` score for each of the 22 tasks |
| `finalBpb` | The independent final validation BPB, reported separately from the curve |
| `trainingCompute` | Cumulative training FLOPs, average GFLOPs per input token, and the accounting used |
| `proof` | Protocol, model, tokenizer, data and source fingerprints of the run |
| `source`, `url` | Provenance of the record and its evidence in the repository |
| `comparisonNote` | A provenance note shown with the run, when needed |
| `baselineAcceptance` | Only on the accepted `dense_full` run |

`cohort` names the accounting the checks apply: `reference-v1` (the operator-level ledger), `native-v3` (the historical
matrix-FLOPs estimate) or `frozen-replay-20261019` (two runs with their original replay accounting). It does not select
a separate leaderboard. The file name is the model card id; `--file NAME` distinguishes a second run of a method.

## Model cards

`methods/<id>.json` holds the name, family, tagline, description, paper and code links, mechanism, limitations,
parameter and block counts, configured compute, and the architecture diagram the observatory draws. Its `id` is the
`methodId` of run records. `"status": "pending"` marks a registered method without a published run; the observatory
lists it with a "Results coming soon" badge and never ranks it.
