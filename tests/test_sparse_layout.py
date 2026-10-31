"""Exact sparse layout and empty-row regressions for fused execution metadata."""
import pytest
import torch

from nexttransformer.models import sparse_kernels
from nexttransformer.models.sparse import block_indices, indices_to_mask, select_tokens


@pytest.mark.parametrize("length,topk", [(256, 0), (256, 1), (384, 3)])
def test_selected_block_layout_preserves_every_key_and_edge(length, topk):
    torch.manual_seed(42)
    positions = torch.arange(length)
    blocks = length // 128
    scores = torch.randn(2, 3, length, blocks)
    complete = (torch.arange(blocks) + 1) * 128 - 1 <= positions[:, None]
    eligible = complete & (torch.arange(blocks)[None] != (positions // 128)[:, None])
    chosen, valid = select_tokens(scores, eligible[None, None], topk)
    selected, full, partial, edges = sparse_kernels._selected_block_layout(chosen, valid, positions, blocks)
    indices, reference_valid = block_indices(scores, complete, positions, 128, length, topk)
    expected = indices_to_mask(indices, reference_valid, length)
    actual = selected[..., torch.arange(length) // 128] & (positions[None, :] <= positions[:, None])
    assert torch.equal(actual, expected)
    assert edges == reference_valid.sum()
    for query in range(length // 128):
        for key in range(blocks):
            tile = expected[..., query * 128:(query + 1) * 128, key * 128:(key + 1) * 128]
            assert torch.equal(full[..., query, key], tile.flatten(-2).all(-1))
            assert torch.equal(partial[..., query, key], tile.flatten(-2).any(-1) & ~full[..., query, key])


@pytest.mark.parametrize("shape", [(1, 1, 128, 256), (2, 1, 256, 256), (1, 3, 256, 384), (2, 3, 256, 256)])
def test_mask_packing_preserves_broadcasts_and_empty_rows(shape):
    torch.manual_seed(42)
    mask = torch.rand(shape) < .3
    mask[..., 5, :] = False
    has_key, full, partial, packed = sparse_kernels._pack_block_mask(mask)
    key_positions = torch.arange(shape[-1])
    unpacked = ((packed[..., key_positions // 8].to(torch.int32) >> (key_positions % 8)) & 1) != 0
    expected = mask.clone()
    expected[..., 0] |= ~mask.any(-1)
    assert torch.equal(unpacked, expected)
    assert torch.equal(has_key, mask.any(-1, keepdim=True))
    for query in range(shape[-2] // 128):
        for key in range(shape[-1] // 128):
            tile = expected[..., query * 128:(query + 1) * 128, key * 128:(key + 1) * 128]
            assert torch.equal(full[..., query, key], tile.flatten(-2).all(-1))
            assert torch.equal(partial[..., query, key], tile.flatten(-2).any(-1) & ~full[..., query, key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sparse metadata execution")
@pytest.mark.parametrize("length,topk", [(256, 0), (256, 1), (2048, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_block_attention_matches_existing_kernel_and_gradients(length, topk, dtype):
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 2, length, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    positions = torch.arange(length, device="cuda")
    blocks = length // 128
    scores = torch.randn(1, 2, length, blocks, device="cuda", dtype=dtype)
    complete = (torch.arange(blocks, device="cuda") + 1) * 128 - 1 <= positions[:, None]
    eligible = complete & (torch.arange(blocks, device="cuda")[None] != (positions // 128)[:, None])
    chosen, valid = select_tokens(scores, eligible[None, None], topk)
    indices, reference_valid = block_indices(scores, complete, positions, 128, length, topk)
    expected = sparse_kernels.block_sparse_attention(q, k, v, indices_to_mask(indices, reference_valid, length))
    actual, edges = sparse_kernels.selected_block_attention(q, k, v, chosen, valid, positions)
    assert edges == reference_valid.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    weight = torch.randn_like(actual)
    old_gradients = torch.autograd.grad(expected, (q, k, v), weight)
    new_gradients = torch.autograd.grad(actual, (q, k, v), weight)
    for actual_gradient, expected_gradient in zip(new_gradients, old_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA QSA mask packing")
@pytest.mark.parametrize("dtype,length,burnin", [(torch.float32, 256, False), (torch.bfloat16, 256, False),
                                               (torch.bfloat16, 2048, False), (torch.bfloat16, 256, True)])
def test_qsa_mask_packing_preserves_native_indexer_and_all_gradients(dtype, length, burnin, monkeypatch):
    from nexttransformer.models import sparse
    from nexttransformer.models.config import candidate_config

    torch.manual_seed(42)
    config = candidate_config("qsa_hybrid")
    mixer = sparse.SparseMixer(config, 3, kind="qsa").cuda().train()
    mixer.backend, mixer.in_burnin = "kernel", burnin
    with torch.no_grad():
        mixer.c_proj.weight.normal_(std=.02)
    x = torch.randn(1, length, config.width, device="cuda", dtype=dtype)
    ve = torch.randn_like(x) * .1
    angle = torch.arange(length, device="cuda")[:, None] * 100000.**(
        -torch.arange(0, config.head_dim, 2, device="cuda") / config.head_dim)
    cos_sin = (angle.cos().to(dtype)[None, :, None], angle.sin().to(dtype)[None, :, None])
    weight = torch.randn_like(x)
    compiled = sparse_kernels._metadata_kernel

    def run(factory):
        monkeypatch.setattr(sparse_kernels, "_metadata_kernel", factory)
        xx, vv = x.clone().requires_grad_(), ve.clone().requires_grad_()
        out = mixer(xx, vv, cos_sin, (-1, 0), None)
        aux = mixer.last_aux
        gradients = torch.autograd.grad((out * weight).sum() + aux, (xx, vv, *mixer.parameters()))
        return out.detach(), aux.detach(), gradients, mixer.last_stats["selected_edges"]

    expected = run(lambda function: function)
    actual = run(compiled)
    assert torch.equal(actual[3], expected[3])
    tolerance = 1e-4 if dtype == torch.float32 else .01
    relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))
    assert relative(actual[0], expected[0]) < tolerance
    assert relative(actual[1], expected[1]) < tolerance
    scale = max(float(g.abs().max()) for g in expected[2])
    for got, want in zip(actual[2], expected[2]):
        assert relative(got, want) < (1e-4 if dtype == torch.float32 else .03) or float((got-want).abs().max()) <= 1e-6*scale
    mixer.eval()
    with torch.no_grad():
        monkeypatch.setattr(sparse_kernels, "_metadata_kernel", lambda function: function)
        expected_eval = mixer(x, ve, cos_sin, (-1, 0), None)
        monkeypatch.setattr(sparse_kernels, "_metadata_kernel", compiled)
        actual_eval = mixer(x, ve, cos_sin, (-1, 0), None)
    torch.testing.assert_close(actual_eval, expected_eval, rtol=0, atol=0)
