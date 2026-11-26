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
    assert {m["method"] for m in models if not (output / m["directory"] / "spec.yaml").exists()} == {
        "native-circle", "native-mor"}


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


def test_names_are_stable_and_export_requires_empty_directory(bundle, tmp_path):
    other = tmp_path / "other"
    bridge.export_bundle(other, ["dense_full", "native-circle"])
    assert bridge.load_bundle(other) == bridge.load_bundle(bundle)
    assert bridge.load_bundle(other)["models"][0]["model_id"] == "custom:nextt-dense_full"
    with pytest.raises(ValueError, match="must be empty"):
        bridge.export_bundle(bundle)


def test_unknown_method_does_not_create_output(tmp_path):
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="Unknown methods"):
        bridge.export_bundle(output, ["not-a-model"])
    assert not output.exists()


def test_fixed_spec_requires_matching_configuration(tmp_path):
    directory = tmp_path / "tools/huggingarch/specs"
    directory.mkdir(parents=True)
    text = "# fixed artifact\n"
    (directory / "test.yaml").write_text(text)
    entry = dict(method="test", config={"hidden_size": 16}, spec_file="test.yaml")
    bridge.write_json(directory / "catalog.json", {"models": [entry]})
    assert bridge.reviewed_spec(tmp_path, entry, entry["config"]) == (text, entry)
    assert bridge.reviewed_spec(tmp_path, entry, {"hidden_size": 8}) is None
    assert bridge.reviewed_spec(tmp_path, entry, dict(entry["config"], torch_dtype="float32")) == (text, entry)


def test_export_copies_spec_and_rejects_incompatible_configuration(tmp_path):
    bridge.export_bundle(tmp_path, ["mha_no_ve"])
    assert (tmp_path / "mha_no_ve/spec.yaml").is_file()
    assert bridge.read_json(tmp_path / "mha_no_ve/spec-review.json")["status"] == "analysis_checked"
    path = tmp_path / "mha_no_ve/config.json"
    config = bridge.read_json(path)
    config["hidden_size"] = 48
    bridge.write_json(path, config)
    with pytest.raises(ValueError, match="does not support the configuration"):
        bridge.load_bundle(tmp_path)


def test_weight_precision_is_explicit_and_reuses_graph(tmp_path):
    bf16, fp32 = tmp_path / "bf16", tmp_path / "fp32"
    bridge.export_bundle(bf16, ["dense_full"])
    bridge.export_bundle(fp32, ["dense_full"], weight_dtype="float32")
    first = bridge.load_bundle(bf16)["models"][0]
    second = bridge.load_bundle(fp32)["models"][0]
    assert first["model_id"] == second["model_id"]
    assert second["weight_dtype_assumption"] == "float32"
    assert (fp32 / "dense_full/spec.yaml").read_text() == (bf16 / "dense_full/spec.yaml").read_text()


def test_json_report_preserves_upstream_results_and_separates_training_costs(bundle):
    result = dict(status="analyzed", settings={"batch_size": 2, "seq_len": 64},
                  analysis={"summary": {"total_params": 123, "total_flops_gflops": 7, "weight_bytes": 1024**3},
                            "layer_groups": [{"name": "projection", "total_params": 22, "total_flops": 2e9}]})
    bridge.write_json(bundle / "dense_full/analyze.json", result)
    report = bridge.write_report(bundle)
    assert bridge.read_json(bundle / "report.json") == report
    assert report["models"][0]["result"] == result
    assert report["models"][0]["model"]["parameters"] == 1355810568
    assert report["models"][0]["model"]["training_flops_per_token"] == pytest.approx(4674661693.46476)
    assert report["models"][1]["result"]["status"] == "exported"
    assert report["models"][1]["model"]["parameters"] is None
    assert not (bundle / "index.html").exists()
    assert "sha256" not in (bundle / "report.json").read_text()


def test_timeout_is_a_model_error(monkeypatch):
    killed = []
    class Process:
        pid = 42
        calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def communicate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            return "", ""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(bridge.os, "killpg", lambda pid, sig: killed.append(pid))
    result = bridge.run_worker({"huggingarch_root": "."}, Path(sys.executable), 1)
    assert result["status"] == "error" and "exceeded" in result["error"]
    assert killed == [42]


def test_smoke_export_has_no_formal_training_evidence(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        bridge.export_bundle(tmp_path / "all", smoke=True)
    manifest = bridge.export_bundle(tmp_path / "small", ["mha_no_ve"], smoke=True)
    model = manifest["models"][0]
    assert model["profile"] == "smoke"
    assert model["parameters"] is None and model["training_flops_per_token"] is None
    config = bridge.read_json(tmp_path / "small/mha_no_ve/config.json")
    assert config["hidden_size"] == 48 and config["max_position_embeddings"] == 12
    assert model["name"] == "nextt-mha_no_ve-smoke"
    assert not (tmp_path / "small/mha_no_ve/spec.yaml").exists()


@pytest.mark.parametrize("exit_code,publish", [(0, True), (1, False), (0, False)])
def test_generate_delegates_once_and_preserves_failed_status(tmp_path, monkeypatch, exit_code, publish):
    from nexttransformer.integrations import huggingarch_worker as worker
    from types import ModuleType

    directory = tmp_path / "bundle"
    directory.mkdir()
    bridge.write_json(directory / "config.json", {"model_type": "test"})
    (directory / "forward.py").write_text("# architecture")
    cache = tmp_path / "cache"
    calls = []
    def load(name):
        path = cache / "spec.yaml"
        return {"config": {"model_type": "test"},
                "forward_source": "# ======== forward.py ========\n# architecture",
                "spec_yaml": path.read_text() if path.exists() else ""}
    def register(**kwargs):
        cache.mkdir()
        (cache / "prompt.md").write_text("upstream prompt")
        return {"status": {}}
    def generate(name, agent_cli=None):
        calls.append((name, agent_cli))
        print("upstream generation and validation")
        if publish:
            (cache / "spec.yaml").write_text("# published upstream spec")
        return exit_code
    monkeypatch.setitem(sys.modules, "backend", ModuleType("backend"))
    monkeypatch.setitem(sys.modules, "backend.custom_id", SimpleNamespace(model_path=lambda name: cache))
    monkeypatch.setitem(sys.modules, "backend.custom_model", SimpleNamespace(
        load=load, register=register, run_spec_generation=generate))
    request = dict(action="generate", name="test", directory=str(directory), agent_cli="codex")
    result = worker.perform(request)
    assert calls == [("test", "codex")]
    assert "validation" in (directory / "generate.log").read_text()
    if exit_code or not publish:
        assert result["status"] == "generation_failed"
        assert not (directory / "spec.yaml").exists()
    else:
        assert result["status"] == "spec_ready" and not result["reused"]
        assert result["validation"] == "upstream_generation_gate"
        again = worker.perform(request)
        assert again["reused"] and len(calls) == 1
        assert (directory / "spec.yaml").read_text() == "# published upstream spec"


def test_ready_spec_replaces_obsolete_missing_spec_in_report(bundle, monkeypatch):
    bridge.write_json(bundle / "dense_full/analyze.json", {"status": "needs_spec"})
    monkeypatch.setattr(bridge, "run_worker", lambda *args: {"status": "spec_ready", "reused": False})
    assert bridge.main(["generate", "--bundle", str(bundle), "--methods", "dense_full"]) == 0
    report = bridge.read_json(bundle / "report.json")
    assert report["models"][0]["result"]["status"] == "spec_ready"


def test_failed_model_does_not_hide_other_results(bundle, tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    (upstream / "backend").mkdir(parents=True)
    (upstream / "backend/custom_model.py").write_text("")
    calls = []
    def worker(request, python, timeout):
        calls.append(request)
        if request["name"] == "nextt-dense_full":
            return {"status": "error", "error": "bad spec"}
        return {"status": "needs_spec", "error": "Generate a spec"}
    monkeypatch.setattr(bridge, "run_worker", worker)
    args = SimpleNamespace(bundle=bundle, huggingarch_root=upstream, huggingarch_python=Path(sys.executable),
                           methods=None, command="analyze", batch_size=2, seq_len=64,
                           dtype="bfloat16", timeout=1)
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
    (backend / "arch").mkdir()
    (backend / "arch/__init__.py").write_text("")
    (backend / "arch/loader.py").write_text("from pathlib import Path\nCACHE_DIR = Path('.')\n")
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
    assert result == {"status": "analyzed",
                      "analysis": {"success": True, "summary": {"total_params": 123}}}
