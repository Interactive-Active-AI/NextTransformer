"""Gated Attention (Qiu et al., arXiv 2505.06708): a query-dependent sigmoid gate on each head's SDPA output.

The Dense baseline's causal attention with the paper's best gate: position G1 (after scaled dot-product
attention, before the output projection), multiplicative, head-specific and sigmoid,

    Y' = Y ⊙ σ(X W_θ)

where Y is the per-head attention output and X the block's pre-normalized input, the same input as q, k and v.
`granularity` "elementwise" gives one score per head and channel (the paper's best row and Qwen3-Next's layout),
"headwise" one score per head. As in the official code (github.com/qiuzh20/gated_attention, modeling_qwen3.py
at f4c2a5f), the gate has no bias and no extra scale.

Adaptation to the shell: the official code fuses W_θ into q_proj; here it is a separate linear layer `c_gate`,
which computes the same function. It is initialized with c_q's uniform rule and, like c_q, trained by Muon.
Value embeddings, RoPE, QK norm, the attention scale and the zero-initialized c_proj are the Dense baseline's.
"""
import torch
from torch import nn

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm

GRANULARITY = ("elementwise", "headwise")


class GatedAttention(CausalSelfAttention):
    """CausalSelfAttention whose per-head output is multiplied by σ(c_gate(x)) before c_proj.

    c_gate: width -> heads * head_dim (elementwise) or width -> heads (headwise)
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.gate_channels = self.head_dim if config.option("granularity") == "elementwise" else 1
        self.c_gate = Linear(self.n_embd, self.n_head * self.gate_channels, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Identical to CausalSelfAttention.forward up to the gate before c_proj.
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v, cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        # G1: the query-dependent gate scales each head's attention output, then the heads are projected.
        y = y * torch.sigmoid(self.c_gate(x)).view(B, T, self.n_head, self.gate_channels)
        return self.c_proj(y.contiguous().view(B, T, -1))


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    """c_gate takes c_q's rule (uniform, standard deviation 1/sqrt(width)), as the gate rows of a fused q_proj."""
    bound = 3**.5 * model.config.width**-.5
    for block in model.transformer.h:
        nn.init.uniform_(block.attn.c_gate.weight, -bound, bound)


def check(config):
    if config.option("granularity") not in GRANULARITY:
        raise ValueError(f"granularity must be one of {', '.join(GRANULARITY)}")
