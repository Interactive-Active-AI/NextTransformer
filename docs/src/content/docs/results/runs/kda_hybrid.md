---
title: Hybrid KDA + MLA-NoPE
description: Single-segment hybrid training without value embeddings, and the comparison with KDA and MLA-NoPE.
---

At 46,875 updates (49.152B tokens), Hybrid KDA + MLA-NoPE (`kda_hybrid`) reached **0.673751 periodic validation BPB**, **0.671160 independent final validation BPB** (0.672881 on the training split) and **31.5418 CORE ×100** at seed 42. All 24 layers omit value embeddings and their gates, leaving **793,542,986 parameters** while keeping the 6016-wide FFN of standalone KDA.

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,688,180 | 0.667172 | 0.664460 | 33.0952 | — |
| KDA (`kda`) | 24 | 6016 | 1,412,126,344 | 0.667123 | 0.664387 | 31.0606 | -0.00005 / -2.03 |
| MLA-NoPE + QK-norm (`mla_nope_qknorm`) | 24 | 6272 | 1,303,914,612 | 0.685620 | 0.682963 | 27.6536 | +0.01845 / -5.44 |
| Hybrid KDA + MLA-NoPE (`kda_hybrid`) | 24 | 6016 | 793,542,986 | 0.673751 | 0.671160 | 31.5418 | +0.00658 / -1.55 |

Compared with standalone KDA, replacing every fourth KDA block (one-based layers 4, 8, 12, 16, 20 and 24) with full causal MLA-NoPE while removing all value embeddings reduced the parameter count by 618.6M (43.8%), raised periodic validation BPB by 0.00663, and increased CORE ×100 by 0.48 points. On CORE tasks, the hybrid improved reading comprehension and algorithmic reasoning over standalone KDA (SQuAD +6.92, BIG-bench CS Algorithms +5.91, CoQA +5.75, AGIEval LSAT-AR +4.89, Winograd +4.40, CommonsenseQA +24.26 centered points) while trailing on BoolQ (−11.67) and WinoGrande (−6.63). Compared with standalone MLA-NoPE + QK-norm, periodic validation BPB was 0.01187 lower and CORE was 3.89 points higher.

The formal run completed in one uninterrupted segment on eight ranks with microbatch eight and eight accumulation steps (`fab8c032` parent source with launch-environment patch `08668a33` and science hash `e1b5a94c`). KDA layers used the FLA `chunk64` backend and MLA-NoPE layers used deterministic SDPA without FlashAttention.

The cumulative training charge is **4.692309 GFLOPs per token** (+0.426% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The audit covers all 189 scheduled validations, six full CORE-22 evaluations and 94 verified checkpoints. Gradient norms stayed smooth and finite throughout training: after update 500 the median was 0.27 and the maximum was 0.71.

Distributed eight-rank 12-update target-batch smoke and fresh-process resume acceptance passed (`weights_equal`, `losses_equal`, `all_rank_states_equal`). A separate 512-update pilot and the launch-patch diff were not archived, so final acceptance remains subject to maintainer review.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/kda_hybrid.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/kda_hybrid.json)
