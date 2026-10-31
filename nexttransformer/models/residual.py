"""Residual organizations of the v2 Residual family on the native full-attention shell.

Every attention or FFN sublayer is one residual layer. ``residual_standard`` is the
single-stream ``h + F(norm(h))`` control. Hyper-Connections, mHC and Block AttnRes replace
that same path; the native learned residual/x0 scalars and backout are disabled for all
four rows by ``ReferenceLM``. Branch inputs stay ``width``-dimensional.
"""
import torch
from torch import nn

from nexttransformer.models.backbone import norm


def coefficient_precision(x):
    """Residual coefficients use FP32 for BF16/FP32 activations; FP64 inputs stay FP64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def sinkhorn_knopp(logits, iterations):
    """Approximate doubly stochastic projection of ``exp(logits)`` over the last two dimensions.

    Each iteration rescales columns, then rows (mHC Eq. 9), so rows sum to one exactly and
    columns approximately. The iterates are kept as logarithms: subtracting a log-sum-exp is
    the division by the sum, so a row whose entries all underflow under FP32 ``exp`` (column
    logit spreads beyond about 104) never becomes 0/0 or a division by a vanishing sum.
    """
    def normalize(log, dim):
        # log - logsumexp(log), from the operators the cost ledger counts; the shift's gradient cancels.
        centered = log - log.amax(dim, keepdim=True).detach()
        return centered - centered.exp().sum(dim, keepdim=True).log()

    log = logits
    for _ in range(iterations):
        log = normalize(normalize(log, -2), -1)
    return log.exp()


class HyperConnection(nn.Module):
    """Dynamic hyper-connection around one sublayer (Zhu et al., Eqs. 10-14 and Algorithm 2).

    Streams have shape [B, T, n, D]. ``read`` returns the sublayer input ``A_m^T H``, the
    mixed residual ``A_r^T H`` and the write weights ``B``; ``write`` adds ``B^T F`` to every
    stream. The dynamic-mapping normalization is the native parameter-free RMSNorm.
    """

    def __init__(self, width, streams, index, gate_init):
        super().__init__()
        self.streams, self.index, self.gate_init = streams, index, gate_init
        self.static_alpha = nn.Parameter(torch.empty(streams, streams + 1))
        self.static_beta = nn.Parameter(torch.empty(streams))
        self.dynamic_alpha = nn.Parameter(torch.empty(width, streams + 1))
        self.dynamic_beta = nn.Parameter(torch.empty(width, 1))
        self.alpha_scale = nn.Parameter(torch.empty(1))
        self.beta_scale = nn.Parameter(torch.empty(1))

    @torch.no_grad()
    def reset_parameters(self):
        """Pre-Norm-equivalent start: read stream ``index mod n``, identity residual, unit writes."""
        self.static_alpha.zero_()
        self.static_alpha[self.index % self.streams, 0] = 1.
        self.static_alpha[:, 1:] = torch.eye(self.streams, device=self.static_alpha.device)
        self.static_beta.fill_(1.)
        self.dynamic_alpha.zero_()
        self.dynamic_beta.zero_()
        self.alpha_scale.fill_(self.gate_init)
        self.beta_scale.fill_(self.gate_init)

    def read(self, streams):
        normed = norm(streams)
        alpha = self.alpha_scale * torch.tanh(coefficient_precision(normed @ self.dynamic_alpha.to(normed.dtype))) + self.static_alpha
        beta = self.beta_scale * torch.tanh(coefficient_precision(normed @ self.dynamic_beta.to(normed.dtype)).squeeze(-1)) + self.static_beta
        mixed = torch.einsum("btjk,btjd->btkd", alpha.to(streams.dtype), streams)
        return mixed[:, :, 0], mixed[:, :, 1:], beta

    @staticmethod
    def write(residual, output, weights):
        return residual + weights.to(output.dtype)[..., None] * output[:, :, None]


class ManifoldHyperConnection(nn.Module):
    """Manifold-constrained hyper-connection around one sublayer (Xie et al., Eqs. 7-9).

    The n x D stream is flattened and RMS-normalized (its weight is absorbed into the
    projection, as in the paper's fused kernels). One projection yields the pre, post and
    residual logits: ``H_pre = sigmoid``, ``H_post = 2 sigmoid``, ``H_res = Sinkhorn-Knopp``.
    Coefficients are computed in at least FP32 from the activation-precision projection.
    """

    def __init__(self, width, streams, index, gate_init, iterations):
        super().__init__()
        self.streams, self.index, self.gate_init, self.iterations = streams, index, gate_init, iterations
        self.projection = nn.Parameter(torch.empty(streams * width, streams * streams + 2 * streams))
        self.bias = nn.Parameter(torch.empty(streams * streams + 2 * streams))
        self.gates = nn.Parameter(torch.empty(3))

    @torch.no_grad()
    def reset_parameters(self):
        """The paper fixes only the gate scales; static biases start at HC's Eq. 14 values."""
        n = self.streams
        self.projection.zero_()
        self.bias.zero_()
        self.bias[self.index % n] = 1.
        self.bias[n:2 * n] = 1.
        self.bias[2 * n:] = torch.eye(n, device=self.bias.device).flatten()
        self.gates.fill_(self.gate_init)

    def maps(self, streams):
        """(H_pre [B,T,n], H_post [B,T,n], H_res [B,T,n,n]) in coefficient precision."""
        b, t, n, d = streams.shape
        flat = norm(streams.reshape(b, t, n * d))
        logits = coefficient_precision(flat @ self.projection.to(flat.dtype))
        pre, post, res = logits.split((n, n, n * n), -1)
        bias_pre, bias_post, bias_res = self.bias.split((n, n, n * n))
        pre = torch.sigmoid(self.gates[0] * pre + bias_pre)
        post = 2 * torch.sigmoid(self.gates[1] * post + bias_post)
        res = sinkhorn_knopp((self.gates[2] * res + bias_res).unflatten(-1, (n, n)), self.iterations)
        return pre, post, res

    def read(self, streams):
        pre, post, res = self.maps(streams)
        layer_input = torch.einsum("btj,btjd->btd", pre.to(streams.dtype), streams)
        mixed = torch.einsum("btij,btjd->btid", res.to(streams.dtype), streams)
        return layer_input, mixed, post

    write = staticmethod(HyperConnection.write)


def depth_attention(query, sources, keys):
    """Block AttnRes read (Kimi Team, Eqs. 2-4 and 6): softmax attention over depth sources.

    ``keys`` are the RMS-normalized ``sources`` ([B, T, D] each); ``query`` is the sublayer's
    learned pseudo-query, initialized to zero so the first reads are uniform averages.
    Sources are combined one at a time so no stacked copy of the depth history is stored.
    """
    q = query.to(keys[0].dtype)[:, None]
    logits = coefficient_precision(torch.stack([(key @ q).squeeze(-1) for key in keys]))
    weights = logits.softmax(0).to(sources[0].dtype)
    out = weights[0, ..., None] * sources[0]
    for weight, value in zip(weights[1:], sources[1:]):
        out = out + weight[..., None] * value
    return out
