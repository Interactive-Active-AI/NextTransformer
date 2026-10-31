"""Regression tests for deviations from the papers / official code found in the 20261018 audit.

Each test pins one behaviour to its source (see side-research doc paper_check_20261018.md):
indexer detach for DSA / QSA / Seer (DeepSeek-V3.2 report; SeerAttention), Huginn SandwichBlock
(recurrent-pretraining), NSA per-head gates (FLA native-sparse-attention), Ouro gate input (Ouro paper §3.2),
DeepLoop init (lszshu/DeepLoop). The sparse `kernel` backend is checked against the reference path.
"""
import pytest
import torch

from nexttransformer.models.backbone import norm
from nexttransformer.models.config import debug_config
from nexttransformer.models.looped import SandwichBlock
from nexttransformer.models.sparse import SparseMixer, safe_softmax
from nexttransformer.models.sparse_kernels import head_mean_teacher
from nexttransformer.models.lm import build_model


def _mixer_aux_grad(method):
    c = debug_config(method, sequence_len=16)
    m = build_model(c).train()
    mixer = next(b.attn for b in m.transformer.h if isinstance(b.attn, SparseMixer))
    mixer.in_burnin = False                    # sparse stage: L_I over the selected set
    x = torch.randn(1, 16, c.width, requires_grad=True)
    t = x.shape[1]
    mixer(x, None, (m.cos[:, :t], m.sin[:, :t]), (t, 0), None)
    assert mixer.last_aux is not None and float(mixer.last_aux) > 0
    grad, = torch.autograd.grad(mixer.last_aux, x, allow_unused=True)
    return grad


@pytest.mark.parametrize("method", ["dsa", "qsa", "seer"])
def test_indexer_loss_does_not_reach_the_backbone(method):
    grad = _mixer_aux_grad(method)
    assert grad is None or float(grad.abs().sum()) == 0.0   # indexer input detached: L_I trains only the indexer


def test_sparse_kernel_backend_refuses_instead_of_falling_back():
    c = debug_config("moba", sequence_len=16)
    m = build_model(c).train()
    t = c.sequence_len
    with pytest.raises(ValueError, match="unknown sparse backend"):
        m.set_sparse_backend("masked")
    m.set_sparse_backend("kernel")
    mixer = next(b.attn for b in m.transformer.h if isinstance(b.attn, SparseMixer))
    with pytest.raises(ValueError, match="sparse kernel backend unavailable"):   # CPU tensors, T not a multiple of 128
        mixer(torch.randn(1, t, c.width), None, (m.cos[:, :t], m.sin[:, :t]), (t, 0), None)


def test_huginn_blocks_are_true_sandwich_blocks():
    c = debug_config("huginn")
    m = build_model(c)
    assert all(isinstance(b, SandwichBlock) for b in m.transformer.h)
    x = torch.randn(2, c.sequence_len, c.width) * 5
    cos_sin = (m.cos[:, :c.sequence_len], m.sin[:, :c.sequence_len])
    y = m.transformer.h[0](x, None, cos_sin, (c.sequence_len, 0), None)
    rms = y.float().pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)  # norm_4 is the last op


def test_nsa_gates_are_per_head_and_per_branch():
    c = debug_config("nsa", sequence_len=16)
    m = build_model(c).train()
    mixer = next(b.attn for b in m.transformer.h if isinstance(b.attn, SparseMixer))
    assert mixer.branch_gate.weight.shape == (3 * c.heads, c.width)
    with torch.no_grad():
        mixer.c_proj.weight.normal_(std=.05)   # native init zeroes c_proj; activate it so the mixer output is non-trivial
    x = torch.randn(1, 16, c.width)
    t = x.shape[1]
    base = mixer(x, None, (m.cos[:, :t], m.sin[:, :t]), (t, 0), None)
    with torch.no_grad():  # make head 0 and head 1 gates differ: outputs must change per head
        mixer.branch_gate.weight[:3].add_(1.0)
    changed = mixer(x, None, (m.cos[:, :t], m.sin[:, :t]), (t, 0), None)
    assert not torch.allclose(base, changed)


def test_ouro_gate_reads_the_post_coda_hidden_state():
    c = debug_config("ouro", sequence_len=8)
    m = build_model(c).train()
    ys, halter_inputs = [], []
    readout = m._readout

    def spy(y, backout):
        ys.append(y.detach().clone())
        return readout(y, backout)

    m._readout = spy
    m.halter.register_forward_pre_hook(lambda mod, args: halter_inputs.append(args[0].detach().clone()))
    x = torch.randint(c.vocab_size, (1, c.sequence_len))
    m(x, x)
    assert len(ys) == c.max_loops and len(halter_inputs) == c.max_loops - 1
    for gate_in, y in zip(halter_inputs, ys):
        torch.testing.assert_close(gate_in, norm(y.float()), rtol=1e-5, atol=1e-5)


def test_deeploop_unrolled_depth_matches_alpha_beta_definition():
    c = debug_config("deeploop")
    m = build_model(c)
    n = c.pre_layers + c.core_layers * c.loops + c.post_layers   # N = K * R unrolled blocks
    for block in m.transformer.h:
        assert block.alpha == (2 * n) ** .5 and block.beta == (8 * n) ** -.5



@pytest.mark.skipif(not torch.cuda.is_available(), reason="sparse kernel backend needs CUDA")
@pytest.mark.parametrize("method", ["dsa", "qsa", "seer", "nsa", "moba", "qsa_hybrid", "moba_hybrid"])
@pytest.mark.parametrize("stage", ["burnin", "sparse"])
def test_kernel_backend_equals_reference(method, stage):
    """Block-sparse attention + fused teacher must compute the reference path's function (float32 on GPU).

    A different key set or teacher changes outputs at O(1); the tolerance only absorbs float32 accumulation order.
    """
    t = 256                                   # two 128-long kernel tiles
    c = debug_config(method, width=64, sequence_len=t)    # head_dim 32: FlexAttention needs a power of two >= 16
    torch.manual_seed(0)
    m = build_model(c).cuda().train()
    mixer = next(b.attn for b in m.transformer.h if isinstance(b.attn, SparseMixer))
    with torch.no_grad():
        mixer.c_proj.weight.normal_(std=.2)
        for p in mixer.parameters():
            if p.ndim == 1: p.add_(0.01 * torch.randn_like(p))
    mixer.in_burnin = stage == "burnin"
    x = torch.randn(2, t, c.width, device="cuda", requires_grad=True)
    cos_sin = (m.cos[:, :t].float(), m.sin[:, :t].float())
    w = torch.randn(2, t, c.width, device="cuda")

    def run(backend):
        mixer.backend = backend
        mixer.zero_grad(set_to_none=True); x.grad = None
        y = mixer(x, None, cos_sin, (t, 0), None)
        aux = mixer.last_aux
        ((y * w).sum() + aux).backward()
        grads = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}
        return y.detach(), aux.detach(), x.grad.detach().clone(), grads, mixer.last_stats["backend"]

    def rel_err(got, want):
        return float((got - want).norm() / want.norm().clamp_min(1e-12))

    ref, new = run("reference"), run("kernel")
    assert (ref[4], new[4]) == ("reference", "kernel")
    for label, got, want in zip(("output", "aux", "input grad"), new[:3], ref[:3]):
        assert rel_err(got, want) < 1e-4, (label, rel_err(got, want))
    assert new[3].keys() == ref[3].keys()
    # Some gradients (e.g. NSA's zero-initialised compress_position) are cancellation-dominated sums of size ~1e-6,
    # where float32 noise is of the same size; judge those against the scale of all parameter gradients.
    scale = max(float(g.abs().max()) for g in ref[3].values())
    for name in ref[3]:
        err = float((new[3][name] - ref[3][name]).abs().max())
        assert rel_err(new[3][name], ref[3][name]) < 1e-4 or err <= 1e-6 * scale, (name, err, scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused teacher kernel needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("domain", ["causal", "selected"])
def test_fused_teacher_equals_dense_softmax_mean(dtype, domain):
    """head_mean_teacher is safe_softmax(q k^T / sqrt(d), domain).mean(1): same scores, float32 reduction order only."""
    torch.manual_seed(0)
    b, h, t, d, start, chunk = 2, 3, 320, 64, 192, 96     # a query chunk that starts mid-sequence, sizes not multiples of 64
    q = torch.randn(b, h, chunk, d, device="cuda").to(dtype)
    k = torch.randn(b, h, t, d, device="cuda").to(dtype)
    positions = torch.arange(start, start + chunk, device="cuda")
    causal = torch.arange(t, device="cuda")[None, :] <= positions[:, None]
    mask = None
    full = causal[None, None].expand(b, h, -1, -1)
    if domain == "selected":
        mask = causal[None] & (torch.rand(b, chunk, t, device="cuda") < 0.2)
        mask[:, 5] = False                                   # an empty row must come out as zeros
        full = mask[:, None].expand(-1, h, -1, -1)
    want = safe_softmax(torch.matmul(q.double(), k.double().transpose(-2, -1)) * d**-.5, full).mean(1)
    got = head_mean_teacher(q, k, mask, start, d**-.5)
    torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-7)
