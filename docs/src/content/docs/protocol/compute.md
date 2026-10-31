---
title: Compute accounting
description: What counts as training compute, how it is counted, and the SSSL anchor every method is matched to.
---

The controlled quantity is **average theoretical training FLOPs per input token, weighted across all training
stages**. It is not the parameter count, the inference depth, wall time or one forward pass.

```text
N_input = 49,152,000,000 input tokens
G       = 1,048,576 input tokens per update
S       = 46,875 optimizer updates

F_step  = F_forward + F_backward + F_auxiliary + F_recompute + F_optimizer
C_train = sum(F_step) / N_input

C_ref   = 4,672,398,325.44785 FLOPs per input token   (the SSSL anchor)
F_ref   = C_ref × N_input ≈ 2.2965772249241273 × 10^20 FLOPs
relative error = C_train / C_ref − 1
```

The anchor is the audited SSSL configuration; its proof is
[`evidence/costs/plugin-interface-20261031/dense.json`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/costs/plugin-interface-20261031/dense.json).
The historical matrix-only estimate of 4.775 GFLOPs per token, used by the first native runs, is a different
accounting and is never mixed with this one.

## How counting works

The auditor (`nexttransformer/protocol/compute.py`) runs the model with fake tensors at the formal dimensions and a
logical **512 × 2,048** global batch. It executes the forward pass, autograd and any custom backward, and the unfused
mathematical body of the optimizer, and counts every floating-point operator. It allocates no model storage and reads
no training data. Validation and CORE are counted separately and never charged to the training budget.

Repeated loops, teachers and readouts increase the numerator, not the token denominator.

- A candidate that is computed fully and masked afterwards is charged fully. Shared weights do not make repeated calls
  free. No-gradient forward passes and multi-exit heads are charged.
- Method-specific backward accounting is never replaced by `6ND`, the parameter count, the inference exit depth or a
  universal "3 × forward".
- Stages are weighted by their actual input tokens. DSA, QSA and Seer use 512 dense burn-in updates and 46,363 sparse
  updates; Ouro uses 45,851 joint and 1,024 gate-only updates.
- Algorithmic compute is kept separate from kernel padding, tiles and redundant execution. A dense masked kernel does
  not demonstrate sparse training savings. Necessary algorithmic recomputation is charged.
- Activation checkpointing is off. Changing it, the model semantics, the source, the software or the kernels requires
  a new audit; earlier proofs and frozen runs are preserved.

## The operator ledger (`reference-aten-math-v2`)

A fused multiply-add counts as two operations. Floating math is counted consistently across dtypes. Casts, data
movement, indexing, integer comparisons and top-k, and communication are outside the FLOP account. Nonlinear
unit-operation conventions are reproducible accounting choices, not equal runtime costs.

| Operation | Count |
|---|---|
| matmul / bmm | `2 × batch × M × N × K`; backward counts the actual multiplications |
| Conv1d | `2 × output elements × input channels per group × kernel size`; input and weight gradients when needed |
| add, sub, mul, div, exp, log, sqrt, rsqrt, sin, cos, tanh, pow, clamp, ReLU | One per output element; a non-unit add alpha adds a multiplication |
| sigmoid / SiLU / softplus / ELU | 4 / 5 / 4 / 3 per element |
| backward of sigmoid / tanh / SiLU / softplus / ELU | 3 / 3 / 8 / 5 / 3 per element |
| sum / mean / max reduction | Inputs minus outputs; mean also divides each output |
| softmax / log-softmax | Forward 5 per score; backward 4 |
| RMSNorm | The actual expanded pow, mean, rsqrt and multiply, and their backward; no double counting |
| Fused LayerNorm | Forward `7N + 5R`; input gradient `10N`; each optional affine gradient `2N` |
| L2 norm | `2N + outputs` |
| Embedding | Lookup zero; a dense-gradient scatter costs one accumulation per gradient element |
| scatter_add | One per floating source element |
| NLL / BCE-with-logits | NLL forward and backward 2 per target, mean adds 1; BCE forward 10 per logit, actual backward |
| cumsum | `n − n/L` additions per scan |
| Triangular solve | Unit lower triangle `n(n − 1)` per column; non-unit adds `n` divisions |
| tril / triu / eye | Zero floating operations |

An unregistered floating operator is reported as `unknown` and fails calibration; it is never silently charged zero.

## Attention

A left window `w` includes the current key, so it allows at most `w + 1` keys; `causal_edges(T, w)` handles the
triangular boundary exactly (SSSL's short window of 512 allows at most 513 keys). For `C` required edges, Q/K dimension
`d` and V dimension `v`, the forward matrices cost `2C(d + v)` and the backward with all Q/K/V gradients costs
`4C(d + v)`, plus scaling and softmax. Frozen inputs remove only the gradient paths that are actually absent.

DSA counts `min(K, i + 1)` keys; QSA counts the selected complete blocks plus the unfinished tail; MoBA includes the
mandatory current block in K; NSA counts its compression, selection and local branches separately. Indexer, teacher
and compression score matrices are charged at their full shapes before masking. Tests compare the analytical edge
counts with the actual masks.

## Optimizer

The count executes the pinned AdamW and Muon mathematics, including momentum, five Polar Express iterations,
MuonEq/Muon+, variance reduction and weight decay, for one logical update of the real trainable groups. DDP and shard
padding and extra microbatch reductions are execution overhead, not algorithmic compute.

## Chunked and scan backends

Linear attention is counted as the **chunk training algorithm**, not as the recurrent scan or the instructions of a
fused kernel. The pure-PyTorch `chunk_reference` in `nexttransformer/models/linear.py` implements the scalar-decay and
WY/UT gated-delta algorithms, and autograd supplies the actual backward.

- A C × C chunk tile counts only the causal pairs `i ≥ j`, `C(C + 1)/2`; dense padding is reported separately.
- The audit runs T = 2,048 directly and cross-checks every operator against 3, 4 and 5 chunks; the structure is affine
  from three chunks onward.
- Kernel-internal replay, tile padding and BF16/TF32 arithmetic are execution details; required algorithmic
  recomputation is still charged.
- Scan-based counts are checked at T = 16, 32 and 48: every operator must satisfy the affine relation before
  extrapolation to 2,048.

Residual-family maps are charged as executed: Hyper-Connections and mHC stream norms, dynamic projections, gates, all
20 Sinkhorn-Knopp iterations and their backward, stream mixing and the final stream sum; Attention Residuals' key
norms, depth logits, softmax and weighted sums.

## Fingerprints

Each proof fingerprints the source its method runs (`nexttransformer/protocol/fingerprint.py`). Calibration records
the repository files executed while the method's cost is audited, plus the repository modules they import; the
protocol, cost ledger, calibration, fingerprint and optimizer sources are always included. Files are hashed without
comments, docstrings or layout, and module-level tables of model files keep only the entries of the method and its
layer mixers. A plugin's whole package belongs to its method's fingerprint.

A change to another method's code, or to comments only, keeps a proof valid; a change to anything the method runs
requires a new audit. `python -m nexttransformer.protocol.fingerprint` compares every configuration with the current
source, and CI fails when a published method no longer matches its proof.
