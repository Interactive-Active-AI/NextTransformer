---
title: Hyper-Connections (HC)
description: Four training segments with unchanged source, and the comparison across the Residual family.
---

At 46,875 updates (49.152B tokens), HC reached **0.664400 periodic validation BPB**, **0.661683 independent final validation BPB** and **35.0943 CORE ×100**, second on CORE after mHC. Compared with Standard Residual, the periodic BPB was 0.00956 lower and CORE was 3.14 points higher. Compared with the full-attention Dense Baseline, BPB was 0.00277 lower and CORE was 2.00 points higher. All comparisons use seed 42, and HC keeps the 6144-wide FFN of both controls.

| Updates retained | Resumed from | Training source |
|---|---|---|
| 1–29,000 | — | `4dd6ebb` |
| 29,001–33,000 | checkpoint 29,000 | `4dd6ebb` |
| 33,001–38,500 | checkpoint 33,000 | `4dd6ebb` |
| 38,501–46,875 | checkpoint 38,500 | `4dd6ebb` |

The first segment stopped when the shared training job ended after mHC's non-finite gradient. Two later interruptions restored the newest complete checkpoint. Every segment kept the same source, configuration, cost proof and batch layout: eight ranks, microbatch four and 16 accumulation steps. The published curve excludes the superseded tails.

The cumulative training charge is **4.698483 GFLOPs per token** (+0.558% versus the SSSL anchor), equal to the cost proof at every logged update. The audit covers 189 validations and six full CORE-22 evaluations.

Gradient norms stayed finite but were larger and spikier than Standard Residual's: after update 500 the median was 0.90, against 0.25. One isolated spike reached 1,824 at update 1,900, and spikes of 30–72 occurred around updates 24,400–24,750, without changing the loss trend.

With HC, the Residual family is complete. Δ is periodic BPB / CORE points versus Standard Residual:

| Run | Periodic val BPB | Final val BPB | CORE ×100 | Δ |
|---|---:|---:|---:|---:|
| Standard Residual | 0.673959 | 0.671214 | 31.9513 | — |
| HC | 0.664400 | 0.661683 | 35.0943 | -0.00956 / +3.14 |
| mHC | 0.661665 | 0.658908 | 35.3278 | -0.01229 / +3.38 |
| Block AttnRes | 0.662963 | 0.660217 | 33.9971 | -0.01100 / +2.05 |

mHC's FFN is 6016 wide versus 6144 for the others. These are single-seed observations, not statistical rankings.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/hc.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/hc.json)
