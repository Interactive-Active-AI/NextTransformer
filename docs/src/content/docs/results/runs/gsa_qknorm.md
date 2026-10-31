---
title: GSA + QK-norm
description: The frozen Gated Slot Attention run and its submitted evidence.
---

This run used 24 layers with 64 slots, per-head QK RMSNorm scaled by 1.2 before SiLU, and a chunk-128 FP64 recurrence. It trained for 46,875 updates (49.152B tokens) at seed 42. Final periodic validation BPB was **0.685242**, independent final validation BPB was **0.682638**, and CORE ×100 was **28.1720**. Its cumulative charge of 4.689438 GFLOPs per input token is 0.36468% above the SSSL anchor.

The submission includes all 189 validation points, six CORE-22 evaluations with per-task scores, and the contributor's cumulative audit. The audit identifies the frozen source and metrics archive by hash. The submitted record contains no segment-by-segment resume log or trainable implementation for this variant.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gsa_qknorm.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gsa_qknorm.json)
