"""The public long-budget profiles, their admissions, and the first d768 run schema."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from moe import audit, compute, data, plugins, spec, train
from moe.budgets import ACTIVE_TRUNK_PARAMS, D768_LAUNCH_PROTOCOL
from moe.model import build_model


@pytest.mark.parametrize("name,steps,tokens,warmup,core", [
    ("d768", 53_686, 14_073_462_784, 536, (12_726, 25_451, 38_177, 53_686)),
    ("d1024", 64_973, 34_064_564_224, 649, (16_162, 32_325, 48_487, 64_973)),
])
def test_240tpp_budgets(name, steps, tokens, warmup, core):
    parent, track = spec.track(name), spec.track(name + "-240tpp")
    params = compute.parameters(plugins.config_for("baseline", track.name))["active_non_embedding"]
    assert params == ACTIVE_TRUNK_PARAMS[name]
    assert abs(track.total_tokens - 240 * params) <= track.tokens_per_update // 2
    assert (track.updates, track.total_tokens, track.warmup_steps, track.core_steps) == (steps, tokens, warmup, core)
    assert data.needed_tokens([track.name]) == tokens + 1
    assert track.marin_budget_flops is None and track.marin_tokens_per_second is None
    changed = {k for k, v in asdict(parent).items() if asdict(track)[k] != v}
    assert changed == {"name", "marin_run", "updates", "core_steps", "marin_budget_flops", "marin_tokens_per_second"}


@pytest.mark.parametrize("track_name,method", [
    ("d768-240tpp", "baseline"), ("d768-240tpp", "hc"), ("d768-240tpp", "mhc"),
    ("d768-240tpp", "simplehc"), ("d1024-240tpp", "baseline"),
])
def test_240tpp_config_is_admitted(track_name, method):
    track = spec.track(track_name)
    config, calibration = spec.load_config(spec.config_path(method, track_name))
    assert config == plugins.config_for(method, track_name)
    record = train.formal_checks(config, calibration, None, SimpleNamespace(smoke_steps=None), 8, track)
    assert (record["model_hash"], record["track"], record["method"]) == (config.fingerprint, track_name, method)
    assert calibration["source_sha256"] == record["source"]["sha256"]
    assert record["training_flops"] == calibration["flops_per_token"] * track.total_tokens
    assert abs(calibration["relative_gap"]) <= spec.PROTOCOL.preflight_tolerance


@pytest.mark.parametrize("method,parameter", [
    ("hc", "attn_connection.dynamic_alpha"),
    ("mhc", "attn_connection.projection"),
    ("simplehc", "gates.projection"),
])
def test_residual_plugins_train(method, parameter):
    config = plugins.config_for(method, "d768-240tpp", debug=True)
    model = build_model(config, "cpu")
    tokens = torch.randint(0, config.vocab_size, (config.qb_group_rows, 16))
    loss = model(tokens, tokens.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss)
    weights = model.blocks[0].get_parameter(parameter)
    assert weights.grad is not None and torch.isfinite(weights.grad).all() and torch.count_nonzero(weights.grad)


def test_audit_preserves_the_initial_d768_run_identity(tmp_path):
    track = spec.track("d768-240tpp")
    launch_hash = "c2d1a34d01c39433efcc5cc0cd8e9003854abe425920d91be218504ec286678e"
    assert D768_LAUNCH_PROTOCOL.fingerprint == launch_hash
    run = dict(track=track.name, track_hash=track.fingerprint, protocol_hash=launch_hash, formal=True)
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "metrics.jsonl").write_text("")
    problems, summary = audit.audit(tmp_path)
    assert summary["protocol_hash"] == launch_hash
    assert not any("fingerprint" in p for p in problems)
    assert not summary["passed"]  # The run has no evaluation or ledger.
    (tmp_path / "run-0001.json").write_text(json.dumps(dict(run, protocol_hash=spec.PROTOCOL.fingerprint)))
    assert any("protocol fingerprint" in p for p in audit.audit(tmp_path)[0])
    (tmp_path / "run-0001.json").unlink()
    other = spec.track("d1024-240tpp")
    (tmp_path / "run.json").write_text(json.dumps(dict(run, track=other.name, track_hash=other.fingerprint)))
    assert any("protocol fingerprint" in p for p in audit.audit(tmp_path)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="checks pinned host storage and CUDA streams")
def test_activation_offload_preserves_values_strides_and_gradients():
    from moe.offload import _pack, _unpack, activation_storage

    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    stream = torch.cuda.Stream()
    saved = _pack(x)
    assert saved[1].is_pinned()
    with torch.cuda.stream(stream):
        restored = _unpack(saved)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(restored, x) and restored.stride() == x.stride()

    view = x.transpose(0, 1)
    restored_view = _unpack(_pack(view))
    assert torch.equal(restored_view, view) and restored_view.stride() == view.stride()
    with activation_storage(True):
        loss = (x * x).float().sum()
    loss.backward()
    assert torch.equal(x.grad, x.detach() * 2)
