"""Opt-in test against the installed HuggingArch dependency, with a temporary cache.

NEXTT_HUGGINGARCH_PYTHON=tools/huggingarch/.venv/bin/python \
python -m pytest -q tests/test_huggingarch_upstream.py

Set NEXTT_HUGGINGARCH_ROOT only to test a source checkout override.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def live_check(upstream, temporary):
    if upstream:
        sys.path.insert(0, str(upstream))
    from backend import custom_id
    worker_path = Path(__file__).resolve().parents[1] / "nexttransformer/integrations/huggingarch_worker.py"
    spec = importlib.util.spec_from_file_location("worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    custom_id.CACHE_DIR = temporary / "cache"
    directory = temporary / "bundle"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(dict(
        model_type="bridge_smoke", hidden_size=8, vocab_size=16, torch_dtype="bfloat16")))
    (directory / "forward.py").write_text("# Synthetic embedding and linear head; integration test only.\n")
    request = dict(action="register", name="bridge-smoke", directory=str(directory),
                   batch_size=1, seq_len=4, dtype="bfloat16")
    assert worker.perform(request)["status"] == "registered"
    assert worker.perform(request)["reused"]
    request["action"] = "analyze"
    assert worker.perform(request)["status"] == "needs_spec"
    spec_path = custom_id.model_path("bridge-smoke") / "spec.yaml"
    spec_path.write_text('''match:
  model_id: ["custom:bridge-smoke"]
components:
  ffn_dense: swiglu_ffn
dims:
  d_model: config.hidden_size
  n_vocab: config.vocab_size
model_inputs:
  input_ids: {kind: tokens}
model:
  - name: embed
    op: embedding
    inputs: [input_ids]
    boundary_role: input_embedding
    dims: {n_vocab: n_vocab, d_model: d_model}
    weight_axes: [{replicated: true}, {replicated: true}]
  - name: lm_head
    op: linear
    inputs: [embed]
    boundary_role: output_head
    dims: {in: d_model, out: n_vocab}
    weight_axes: [{replicated: true}, {replicated: true}]
''')
    result = worker.perform(request)
    assert result["status"] == "analyzed", result
    summary = result["analysis"]["summary"]
    assert summary["total_params"] == 256  # two 16 x 8 matrices
    assert summary["weight_bytes"] == 512  # explicit BF16 storage assumption
    assert summary["total_flops"] == 1024  # 4 positions x 2 x 8 x 16
    assert "resolved" not in result["analysis"]
    # Both upstream cache files are called spec.yaml. Their graphs must not
    # collide in the pinned loader's filename-keyed registry.
    other = temporary / "other"
    other.mkdir()
    (other / "config.json").write_text((directory / "config.json").read_text())
    (other / "forward.py").write_text("# Synthetic model with an additional projection.\n")
    other_request = dict(request, name="bridge-other", directory=str(other))
    assert worker.perform(dict(other_request, action="register"))["status"] == "registered"
    other_spec = spec_path.read_text().replace("custom:bridge-smoke", "custom:bridge-other")
    other_spec = other_spec.replace("  - name: lm_head", '''  - name: extra
    op: linear
    inputs: [embed]
    dims: {in: d_model, out: d_model}
    weight_axes: [{replicated: true}, {replicated: true}]
  - name: lm_head''').replace("    inputs: [embed]\n    boundary_role: output_head", "    inputs: [extra]\n    boundary_role: output_head")
    (custom_id.model_path("bridge-other") / "spec.yaml").write_text(other_spec)
    assert worker.perform(other_request)["analysis"]["summary"]["total_params"] == 320
    assert worker.perform(request)["analysis"]["summary"]["total_params"] == 256
    assert worker.perform(other_request)["analysis"]["summary"]["total_params"] == 320
    bundled = temporary / "bundled"
    bundled.mkdir()
    for filename in ("config.json", "forward.py"):
        (bundled / filename).write_text((directory / filename).read_text())
    bundled_spec = spec_path.read_text().replace("custom:bridge-smoke", "custom:bridge-bundled")
    (bundled / "spec.yaml").write_text(bundled_spec)
    bundled_request = dict(request, name="bridge-bundled", directory=str(bundled))
    assert worker.perform(dict(bundled_request, action="register"))["spec_source"] == "bundle"
    assert worker.perform(bundled_request)["analysis"]["summary"]["total_params"] == 256
    (bundled / "spec-review.json").write_text(json.dumps({"supported_settings": {"min_seq_len": 2, "max_seq_len": 16}}))
    assert worker.perform(dict(bundled_request, seq_len=1))["status"] == "unsupported_settings"
    assert worker.perform(dict(bundled_request, seq_len=17))["status"] == "unsupported_settings"
    (bundled / "spec.yaml").write_text("# conflicting spec")
    try:
        worker.perform(dict(bundled_request, action="register"))
    except ValueError as exc:
        assert "existing spec differs" in str(exc)
    else:
        raise AssertionError("Bundled specs must not overwrite a different cached spec")
    assert (custom_id.model_path("bridge-bundled") / "spec.yaml").read_text() == bundled_spec
    # Reusing a serving spec must not need an agent binary or provider credentials.
    from backend import custom_model
    original_generate = custom_model.run_spec_generation
    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Existing spec must not start an agent")
    custom_model.run_spec_generation = unexpected_generation
    try:
        ready = worker.perform(dict(request, action="generate"))
        assert ready["status"] == "spec_ready" and ready["reused"]
        assert (directory / "spec.yaml").read_text() == spec_path.read_text()
        assert ready["cached_spec_sha256"] == result["cached_spec_sha256"]
    finally:
        custom_model.run_spec_generation = original_generate
    request["batch_size"] = 2
    larger = worker.perform(request)
    assert larger["analysis"]["summary"]["total_flops"] == 2048
    request["action"] = "register"
    assert worker.perform(request)["reused"] and spec_path.exists()
    (directory / "forward.py").write_text("# changed source")
    try:
        worker.perform(request)
    except ValueError as exc:
        assert "differs from the bundle" in str(exc)
    else:
        raise AssertionError("Changed source must not reuse an existing spec")
    print("PASS: real HuggingArch registration, spec reuse without an agent, missing spec, analysis, units and stale-source refusal")


def test_real_huggingarch(tmp_path):
    import pytest
    upstream = os.environ.get("NEXTT_HUGGINGARCH_ROOT")
    python = os.environ.get("NEXTT_HUGGINGARCH_PYTHON")
    if not python:
        pytest.skip("set NEXTT_HUGGINGARCH_PYTHON for the upstream integration test")
    result = subprocess.run([python, str(Path(__file__).resolve()), upstream or "-", str(tmp_path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: real HuggingArch" in result.stdout


if __name__ == "__main__":
    live_check(Path(sys.argv[1]).resolve() if sys.argv[1] != "-" else None, Path(sys.argv[2]).resolve())
