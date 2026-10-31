---
title: Validation, pilots and resume
description: What must be checked before a formal run, how pilots are limited, and what a resume must restore.
---

## Before admission

1. Fingerprints and manifests for the source, model, protocol, cost formula, data, tokenizer and evaluation resources.
2. Small numerical oracles, forward and backward checks, causality, boundaries and method-specific semantic checks.
3. Full-size BF16 checks and a complete global optimizer update on the intended backend; verified parameter grouping
   and distributed synchronization.
4. Identical data rows and scheduled validation and CORE events across the supported batch layouts.
5. Resume checks covering the model, optimizer, scheduler, all random number generators, stream buffers and cursors,
   training stage, budget controller and cumulative compute. Counters are never reset, and a data prefix is never
   replayed as if training had been continuous.

## Pilots

A method may run at most **two preregistered candidates × 512 updates**, from the same prefix and seed (up to about
1.074B input tokens). Pilot cost is published separately. Pilots check stability, mechanism, cost and resources; they
are never tuned on formal CORE. Every attempt is recorded, including fixes to initialization, learning rate or
clipping, and the formal run is reinitialized from scratch.

## Resume

`--resume auto` continues from the newest complete checkpoint, moves incomplete checkpoint directories aside, never
starts a new run over an existing one, and refuses a changed source, configuration or software. `--max-wall-hours` ends
a segment at an update boundary after a checkpoint, so a cluster time limit loses at most the update in progress.

Bounded checks show bitwise-identical continuation for the tested layouts; real long continuations can differ at
rounding level across segments, and each published run discloses its segments. A guarantee proven for one path
(source, kernel, rank count, optimizer layout) is not generalized to others.
