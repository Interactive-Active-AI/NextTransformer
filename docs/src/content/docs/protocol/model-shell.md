---
title: Model shell
description: The shared architecture every method starts from, and what a method may change.
---

Every method is implemented inside one shared shell derived from nanochat's d24 GPT, so that differences between
results come from the mechanism under study rather than from incidental choices.

## Defaults

External width **1536**, **12** query heads of dimension **128**, vocabulary **32,768**, FFN width **6144**.
The shell keeps untied input and output embeddings, the ReLU² FFN, the native RMSNorm, QK normalization and RoPE
policy where applicable, value embeddings (VE), smear, backout, and learned residual and x0 coefficients, unless a
registered mechanism requires an explicit, documented adaptation.

## What a method may change

Layer count, shared-core size, loop distribution, sparse selection, state and latent dimensions, required gates,
convolutions and normalizations, and the registered FFN width are **architecture variables**. Freeze the method's
semantics before cost calibration, document every necessary change to the shell, and provide paired controls when a
change could explain a result on its own (for example, removing value embeddings).

## Attention layouts

SSSL's causal short-window tuple `(512, 0)` includes the current token, so its maximum visible span is **513 tokens**.
Looped models schedule windows by unfolded execution position. Pure sparse and linear mixers replace attention in
every layer; a leftover `window_pattern=SSSL` field does not mean those mixers use SSSL. Hybrids serialize their
actual per-layer layout (`layer_mixers`), and the layout is verified on load.

## Parameters are outcomes

Independent parameters, inference cost, memory, throughput and GPU-hours are reported, not controlled. The budget is
training compute; two methods with the same budget may differ widely in parameter count.

The [source notes](../../methods/sources/) document, method by method, what each implementation retains from its
paper and what it adapts.
