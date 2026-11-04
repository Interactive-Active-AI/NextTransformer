"""Opt-in test against an actual HuggingArch checkout, with a temporary custom cache.

NEXTT_HUGGINGARCH_ROOT=/path/to/HuggingArch \
NEXTT_HUGGINGARCH_PYTHON=/path/to/HuggingArch/.venv/bin/python \
python -m pytest -q tests/test_huggingarch_upstream.py
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def live_check(upstream, temporary):
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
    print("PASS: real HuggingArch registration, reuse, missing spec, analysis, units and stale-source refusal")


def test_real_huggingarch(tmp_path):
    import pytest
    upstream = os.environ.get("NEXTT_HUGGINGARCH_ROOT")
    python = os.environ.get("NEXTT_HUGGINGARCH_PYTHON")
    if not upstream or not python:
        pytest.skip("set NEXTT_HUGGINGARCH_ROOT and NEXTT_HUGGINGARCH_PYTHON for the upstream integration test")
    result = subprocess.run([python, str(Path(__file__).resolve()), upstream, str(tmp_path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: real HuggingArch" in result.stdout


if __name__ == "__main__":
    live_check(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
