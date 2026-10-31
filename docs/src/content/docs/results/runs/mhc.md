---
title: Manifold-Constrained Hyper-Connections (mHC)
description: Four training segments, the registered Sinkhorn change, and the final comparison with Standard Residual.
---

At 46,875 updates (49.152B tokens), mHC reached **0.661665 periodic validation BPB**, **0.658908 independent final validation BPB** and **35.3278 CORE ×100**. Compared with Standard Residual, the periodic BPB was 0.01229 lower and CORE was 3.38 points higher. Compared with the full-attention Dense Baseline, BPB was 0.00551 lower and CORE was 2.23 points higher. All comparisons use seed 42. The calibrated mHC FFN width was 6016, versus 6144 in both controls.

| Updates retained | Sinkhorn implementation | Training source |
|---|---|---|
| 1–31,500 | FP32 multiplicative normalization | `4dd6ebb` |
| 31,501–35,000 | Log-space normalization | `51f4560` |
| 35,001–45,000 | Log-space normalization | `51f4560` |
| 45,001–46,875 | Log-space normalization | `51f4560` |

After the first segment logged update 31,750, a non-finite gradient stopped training. The continuation restored checkpoint 31,500 and applied the preregistered switch to log-space Sinkhorn. Two later interruptions restored checkpoints 35,000 and 45,000. The published curve uses the retained updates from all four segments, excluding superseded tails. Each segment used eight ranks, microbatch eight and eight accumulation steps.

The two frozen proofs differ by 6,072 FLOPs per input token. Weighting them by the retained updates reproduces the logged cumulative training charge of 4.674359 GFLOPs per token (+0.04197% versus the SSSL anchor). All 189 validations and six full CORE-22 evaluations were recovered from the canonical ledger.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mhc.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/mhc.json) · [Registered source and cost transition](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/registrations/mhc-sinkhorn.json)
