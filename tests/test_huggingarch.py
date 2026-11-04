"""The optional bridge must preserve identities and distinguish estimates from evidence."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nexttransformer.integrations import huggingarch as bridge


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "bundle with spaces"
    bridge.export_bundle(output, ["dense_full", "native-circle"])
    return output


def test_export_all_without_torch_or_huggingarch(tmp_path):
    output = tmp_path / "all"
    # -S removes site-packages, including torch, pytest and any HuggingArch installation.
    proc = subprocess.run([sys.executable, "-S", "-m", "nexttransformer.integrations.huggingarch",
                           "export", "--output", str(output)], cwd=bridge.ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    models = bridge.load_bundle(output)["models"]
    from nexttransformer.models.registry import METHODS
    assert {m["method"] for m in models} == set(METHODS) | {
        "native-circle", "native-dense", "native-mor", "native-loopformer"}
    dense = next(m for m in models if m["method"] == "dense_full")
    assert dense["parameters"] == 1355810568
    assert dense["training_flops_per_token"] == pytest.approx(4674661693.46476)
    assert not dense["checkpoint_validated"]
    assert "torch" not in proc.stderr


def test_looped_export_preserves_weight_sharing(tmp_path):
    bridge.export_bundle(tmp_path, ["ut_act", "gdn_hybrid"])
    looped = bridge.read_json(tmp_path / "ut_act/config.json")
    assert looped["num_hidden_layers"] == looped["pre_layers"] + looped["core_layers"] + looped["post_layers"]
    assert looped["max_loops"] == 3
    assert looped["model_type"] == "nexttransformer_ut_act"
    assert looped["torch_dtype"] == "bfloat16"
    assert bridge.read_json(tmp_path / "gdn_hybrid/config.json")["layer_mixers"]
    source = (tmp_path / "ut_act/forward.py").read_text()
    assert "# Source: nexttransformer/models/looped.py" in source
    assert "# Source: nexttransformer/models/backbone.py" in source


def test_identity_is_reproducible_and_content_bound(bundle, tmp_path):
    other = tmp_path / "other"
    bridge.export_bundle(other, ["dense_full", "native-circle"])
    assert bridge.load_bundle(other) == bridge.load_bundle(bundle)
    with (other / "dense_full/forward.py").open("a") as stream:
        stream.write("# modified source\n")
    with pytest.raises(ValueError, match="Bundle changed"):
        bridge.load_bundle(other)
    with pytest.raises(ValueError, match="must be empty"):
        bridge.export_bundle(bundle)


def test_unknown_method_does_not_create_output(tmp_path):
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="Unknown methods"):
        bridge.export_bundle(output, ["not-a-model"])
    assert not output.exists()


def test_weight_precision_is_explicit_and_changes_identity(tmp_path):
    bf16, fp32 = tmp_path / "bf16", tmp_path / "fp32"
    bridge.export_bundle(bf16, ["dense_full"])
    bridge.export_bundle(fp32, ["dense_full"], weight_dtype="float32")
    first = bridge.load_bundle(bf16)["models"][0]
    second = bridge.load_bundle(fp32)["models"][0]
    assert first["model_id"] != second["model_id"]
    assert second["weight_dtype_assumption"] == "float32"
    assert "Assumed unquantized weight dtype: float32" in (fp32 / "index.html").read_text()


def test_report_separates_costs_and_escapes_external_text(bundle):
    model = bridge.load_bundle(bundle)["models"][0]
    bridge.write_json(bundle / "dense_full/analyze.json", dict(
        status="analyzed", content_sha256=model["content_sha256"], settings={"batch_size": 2, "seq_len": 64},
        analysis={"summary": {"total_params": 123, "total_flops_gflops": 7, "weight_bytes": 1024**3},
                  "layer_groups": [{"name": "<script>alert(1)</script>", "total_params": 22, "total_flops": 2e9}]}))
    bridge.render_report(bundle, "http://localhost:49217")
    html = (bundle / "index.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Training GFLOPs/token" in html and "Prefill GFLOPs" in html
    assert "4.6747" in html and "1,355,810,568" in html
    assert "Checkpoint validation: unavailable" in html
    assert "custom%3Anextt-dense_full-" in html
    assert "native-circle" in html and "exported" in html
    assert "<td>—</td>" in html
    assert "1.0000" in html  # weight bytes converted to GiB


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/x", "https://u:p@host", "https://host/?token=x"])
def test_unsafe_inspector_urls_are_rejected(url):
    with pytest.raises(ValueError):
        bridge.inspector_url(url, "custom:a")


def test_timeout_is_a_model_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = bridge.run_worker({"huggingarch_root": "."}, Path(sys.executable), 1)
    assert result["status"] == "error" and "exceeded" in result["error"]


def test_failed_model_does_not_hide_other_results(bundle, tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    (upstream / "backend").mkdir(parents=True)
    (upstream / "backend/custom_model.py").write_text("")
    calls = []
    def worker(request, python, timeout):
        calls.append(request)
        if request["name"].startswith("nextt-dense_full-"):
            return {"status": "error", "error": "bad spec"}
        return {"status": "needs_spec", "error": "Generate a spec"}
    monkeypatch.setattr(bridge, "run_worker", worker)
    args = SimpleNamespace(bundle=bundle, huggingarch_root=upstream, huggingarch_python=Path(sys.executable),
                           methods=None, command="analyze", batch_size=2, seq_len=64,
                           dtype="bfloat16", timeout=1, base_url=None)
    assert bridge.run_bundle(args) == 1
    assert len(calls) == 2
    assert bridge.read_json(bundle / "dense_full/analyze.json")["status"] == "error"
    report = bridge.read_json(bundle / "report.json")
    assert [m["result"]["status"] for m in report["models"]] == ["error", "needs_spec"]
    assert report["models"][1]["result"]["settings"]["seq_len"] == 64


def test_default_uses_installed_dependency_without_checkout(bundle, monkeypatch):
    calls = []
    def worker(request, python, timeout):
        assert request["huggingarch_root"] is None
        assert python == Path(sys.executable).absolute()
        calls.append(request)
        return {"status": "registered", "huggingarch_version": "0.2.0", "huggingarch_revision": "test-commit"}
    monkeypatch.setattr(bridge, "run_worker", worker)
    assert bridge.main(["register", "--bundle", str(bundle), "--methods", "dense_full"]) == 0
    assert len(calls) == 1
    result = bridge.read_json(bundle / "dense_full/register.json")
    assert result["huggingarch_revision"] == "test-commit"
    assert result["huggingarch_version"] == "0.2.0"


def test_worker_real_subprocess_contract(bundle, tmp_path):
    # A tiny upstream double exercises process isolation, not the training interpreter's imports.
    upstream = tmp_path / "upstream checkout"
    backend = upstream / "backend"
    backend.mkdir(parents=True)
    (backend / "__init__.py").write_text("")
    (backend / "custom_id.py").write_text("from pathlib import Path\ndef model_path(name): return Path('.')\n")
    config = {"model_type": "test"}
    bridge.write_json(bundle / "config.json", config)
    (bundle / "forward.py").write_text("# test source")
    (backend / "custom_model.py").write_text(
        "def load(name): return {'config': {'model_type': 'test'}, "
        "'forward_source': '# ======== forward.py ========\\n# test source', 'spec_yaml': ''}\n"
        "def register(**kwargs): pass\n")
    (backend / "analyzer.py").write_text(
        "RESOLVED_KEY = 'resolved'\nclass NoSpecMatch(ValueError): pass\n"
        "def analyze_model(**kwargs):\n"
        "    assert kwargs['trust_remote_code'] is False\n"
        "    print('upstream diagnostic')\n"
        "    return {'success': True, 'summary': {'total_params': 123}, 'resolved': object()}\n")
    request = dict(action="analyze", name="test", directory=str(bundle), huggingarch_root=str(upstream),
                   batch_size=1, seq_len=64, dtype="bfloat16")
    result = bridge.run_worker(request, Path(sys.executable), 10)
    assert result == {"status": "analyzed", "cached_spec_sha256": None,
                      "analysis": {"success": True, "summary": {"total_params": 123}}}
