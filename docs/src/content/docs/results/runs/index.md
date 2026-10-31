---
title: Run notes
description: What each published run is, how it was audited, and the limitations a reader should know.
sidebar:
  label: Overview
  order: 0
---

Each published run has a record in `results/runs/` and, when audited, an audit in `evidence/audits/`. These notes
summarize what the records cannot say in numbers. New results add a page here next to their record.

## Dense Baseline (`dense_full`)

Full causal attention in every layer, 23 layers, FFN 6144, 1.356B parameters. One segment; cumulative compute
**+0.048%** versus the SSSL anchor, equal to its proof on every update, no budget anomaly. Independent final BPB
0.6645 (validation) and 0.6666 (training). The preregistered pilot was skipped by the maintainers' decision; the formal
run started from random initialization and is the only `dense_full` run. Accepted as the Dense baseline: every Δ on the
leaderboard is measured against it.

## Residual family (`residual_standard`, `attnres`)

Both share the Dense shell and were trained as independent single-segment runs: cumulative compute **+0.041%**
(Standard Residual) and **+0.160%** (Block AttnRes). Read-only audits of both run directories found no problems. Read
AttnRes as a paired difference from Standard Residual. Hyper-Connections and mHC were interrupted and resumed from
checkpoints; their results will be published with their resume provenance (mHC continues under a registered source
change to log-space Sinkhorn-Knopp).

## Linear attention

- **KDA** is the restarted second version; its first attempt, aborted at update 2,340, is not part of this run. Three
  segments (0–21,681, 21,681–32,000, 32,000–46,875) with matching provenance; cumulative **+0.429%**.
- **ELU+1 with q/k normalization** was registered after the plain ELU+1 attempts failed (invalid numerics, then
  divergence); two segments on 16 ranks; cumulative **−0.431%**.
- **Gated DeltaNet** ran in one segment on 32 ranks; cumulative **+0.317%**.
- **RetNet** resumed once, at update 25,736; cumulative **−0.906%**.

Resumed segments are consistent to rounding, not bitwise; a guarantee is never claimed beyond what was tested.

## DeltaNet and GLA (frozen replay accounting)

Both completed the full protocol (independent final BPB 0.6760 and 0.6663; CORE 28.34 and 31.11) from a frozen source
revision whose public file hashes are listed in `evidence/audits/deltanet-gla-source-manifest.json`. Their compute uses the
operator-level ledger with backward replay against a theoretical anchor of 4.71661506444785 GFLOPs per token, so their
compute values carry mark ² and are not compared with other rows; their BPB and CORE use the standard rankings.

## Others (MQA, GQA, MLA + QK-norm, MLA-NoPE + QK-norm)

Trained from a frozen source revision of the Others implementation, which is not yet ported to the current code. Each
run has a read-only cumulative audit; per-task CORE scores, centering and aggregates are reproduced, and the MLA runs
record deterministic mode. MQA resumed at updates 5,000 and 35,000, GQA at update 5,000; the MLA runs ran in one segment each.

## Hybrids (GDN + MHA, GDN + SWA)

Trained from a frozen source revision; the GDN + SWA run added a launch-environment patch whose source diff was not
archived. Read-only metadata audits pass: cumulative **+0.672%** and **−0.542%**. A 512-update pilot and distributed
fresh-process resume evidence on the training cluster were not supplied, so final acceptance remains pending. At the
2,048-token context the SWA window covers every causal key, so GDN + SWA is a 1:1 GDN/full-attention hybrid.

## Looped and sparse

UT-ACT, Fixed Recursive, Huginn, Ouro, DeepLoop, DSA, QSA, NSA, Seer and MoBA report complete training and every
scheduled evaluation, with the trainer's cumulative compute audit. Their records keep `complete: false` until
independent read-only audits of the original run directories are done. Sparse methods trained with the kernel backend;
CORE always uses the reference path.

## Historical native track (SSSL, MoR, LoopFormer, Circle Loop)

The first runs of the project, trained with the original native implementations and reported with the historical
matrix-FLOPs estimate (mark ¹) from maintainer-verified final training logs. They are kept for reference; entering them
into the strict matched comparison would require independent full-cost evidence.

## Coming soon

Hyper-Connections, mHC, the DeltaNet, KDA + MLA-NoPE, QSA and MoBA hybrids, and the controls KDA without VE, MLA-NoPE
and MHA without VE are registered and admitted; their model cards are already in the observatory, marked "Results
coming soon".
