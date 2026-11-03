"""The Gated Attention plugin (gated_attention).

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here:
with a zero gate (σ = 1/2 everywhere) the plugin is exactly `dense_full` with halved attention outputs, and the halving
happens on the input of c_proj (position G1, not after the output projection); the gate shapes, initialization and
optimizer group; and the option check.
"""
from dataclasses import replace

import pytest
import torch

from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def debug_plugin(granularity, seed):
    return build_model(replace(debug_config("gated_attention"), options={"granularity": granularity}), "cpu", seed=seed)


def c_proj_inputs(model, x):
    seen = []
    hooks = [block.attn.c_proj.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
             for block in model.transformer.h]
    try:
        out = model(x)
    finally:
        for hook in hooks: hook.remove()
    return out, seen


@pytest.mark.parametrize("granularity", ["elementwise", "headwise"])
def test_zero_gate_is_dense_full_with_halved_attention_outputs(granularity):
    reference = build_model(debug_config("dense_full"), "cpu", seed=3)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        # The shell starts every projection into the residual stream at zero; make them non-trivial.
        for block in reference.transformer.h:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=generator) * .2)
    plugin = debug_plugin(granularity, seed=11)
    missing, unexpected = plugin.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and missing and all(".attn.c_gate." in name for name in missing)
    with torch.no_grad():
        for block in plugin.transformer.h: block.attn.c_gate.weight.zero_()
    x = torch.randint(0, plugin.config.vocab_size, (2, plugin.config.sequence_len))
    for model in (reference, plugin): model.eval()
    dense_logits = reference(x)
    with torch.no_grad():
        for block in reference.transformer.h: block.attn.c_proj.weight.mul_(.5)
    halved_logits, ungated_inputs = c_proj_inputs(reference, x)
    gated_logits, gated_inputs = c_proj_inputs(plugin, x)
    assert torch.equal(gated_logits, halved_logits) and not torch.equal(gated_logits, dense_logits)
    # The gate acts on the heads before the output projection: c_proj reads half of Dense's attention output.
    assert len(gated_inputs) == len(ungated_inputs) == plugin.config.layers
    assert all(torch.equal(gated, .5*ungated) for gated, ungated in zip(gated_inputs, ungated_inputs))


@pytest.mark.parametrize("granularity, rows", [("elementwise", 12*128), ("headwise", 12)])
def test_gate_shape_at_the_formal_size(granularity, rows):
    config = replace(candidate_config("gated_attention"), options={"granularity": granularity})
    model = build_model(config, "meta")
    assert config.width == 1536 and config.heads == 12
    assert all(block.attn.c_gate.weight.shape == (rows, 1536) for block in model.transformer.h)


@pytest.mark.parametrize("granularity", ["elementwise", "headwise"])
def test_gate_is_initialized_like_c_q_and_trained_by_muon(granularity):
    model = debug_plugin(granularity, seed=1)
    bound = 3**.5 * model.config.width**-.5
    for block in model.transformer.h:
        weight = block.attn.c_gate.weight
        assert torch.count_nonzero(weight) == weight.numel() and weight.abs().max() <= bound
    owner = {name: group["kind"] for group in parameter_groups(model) for name in group["parameter_names"]}
    gates = [name for name in owner if ".attn.c_gate." in name]
    assert len(gates) == model.config.layers and all(owner[name] == "muon" for name in gates)


@pytest.mark.parametrize("granularity", ["sigmoid", "", 1, True])
def test_granularity_option_is_checked(granularity):
    with pytest.raises(ValueError, match="granularity"):
        replace(candidate_config("gated_attention"), options={"granularity": granularity})
