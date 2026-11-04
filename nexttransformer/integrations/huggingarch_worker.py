"""Subprocess adapter importing the installed HuggingArch dependency.

Keep imports of NextTransformer out of this file. The JSON protocol is private to
integrations.huggingarch; upstream diagnostics go to stderr.
"""
import contextlib
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys


def perform(request):
    from backend.custom_model import load, register
    from backend.custom_id import model_path

    name = request["name"]
    directory = Path(request["directory"])
    cached = None
    if model_path(name).exists():
        cached = load(name)
        config = json.loads((directory / "config.json").read_text())
        forward = (directory / "forward.py").read_text()
        # Upstream prepends the filename when concatenating forward sources.
        expected = "# ======== forward.py ========\n" + forward
        if cached["config"] != config or cached["forward_source"] != expected:
            raise ValueError("Existing custom model differs from the bundle; refusing to reuse or overwrite it")
    if request["action"] == "generate":
        registered = perform(dict(request, action="register"))
        if registered["status"] != "registered":
            return registered
        cached = load(name)
        reused = bool(cached["spec_yaml"])
        if not reused:
            from backend.custom_model import run_spec_generation

            # Keep upstream's agent, validation, locking and publication together.
            # A failed run's overlay is a draft, never a serving spec.
            with (directory / "generate.log").open("w", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    code = run_spec_generation(name, agent_cli=request.get("agent_cli"))
            if code != 0:
                return {"status": "generation_failed", "error": "Upstream generation/validation failed; see generate.log"}
            cached = load(name)
        if not cached["spec_yaml"]:
            return {"status": "generation_failed", "error": "Upstream did not publish a spec; see generate.log"}
        (directory / "spec.yaml").write_text(cached["spec_yaml"], encoding="utf-8")
        return {"status": "spec_ready", "reused": reused,
                "cached_spec_sha256": hashlib.sha256(cached["spec_yaml"].encode()).hexdigest(),
                "validation": "existing_spec" if reused else "upstream_generation_gate"}
    if request["action"] == "register":
        if cached is not None:
            if not cached["spec_yaml"] and not (model_path(name) / "prompt.md").is_file():
                return {"status": "registration_warning", "error": "No prompt or spec exists; repair upstream dependencies and generate a spec in Inspector"}
            return {"status": "registered", "reused": True}
        result = register(name=name, config_path=str(directory / "config.json"),
                          forward_paths=[str(directory / "forward.py")])
        if result["status"].get("error"):
            return {"status": "registration_warning", "error": result["status"]["error"]}
        return {"status": "registered", "reused": False}

    if not model_path(name).is_dir():
        return {"status": "not_registered", "error": "Run register first"}
    from backend.analyzer import RESOLVED_KEY, NoSpecMatch, analyze_model
    try:
        result = analyze_model(model_id="custom:" + name, trust_remote_code=False,
                               batch_size=request["batch_size"], seq_len=request["seq_len"],
                               dtype=request["dtype"])
    except NoSpecMatch:
        if cached["spec_yaml"]:
            return {"status": "error", "error": "The cached spec could not be loaded/matched; repair it in HuggingArch and retry"}
        return {"status": "needs_spec", "error": "Generate and review a spec in HuggingArch's Inspector, then analyze again"}
    if not result.get("success"):
        return {"status": "error", "error": result.get("error", "HuggingArch analysis failed")}
    spec = cached["spec_yaml"]
    return {"status": "analyzed", "cached_spec_sha256": hashlib.sha256(spec.encode()).hexdigest() if spec else None,
            "analysis": {k: v for k, v in result.items() if k != RESOLVED_KEY}}


def main():
    request = json.loads(sys.stdin.read())
    if not request.get("huggingarch_root") and sys.version_info < (3, 13):
        print(json.dumps({"status": "error", "error": "HuggingArch requires Python 3.13+. Run with uv run --project tools/huggingarch python -m nexttransformer.integrations.huggingarch"}))
        return
    # Normal operation imports the dependency. A checkout is only a developer override.
    if request.get("huggingarch_root"):
        sys.path.insert(0, request["huggingarch_root"])
    try:
        package = None
        if not request.get("huggingarch_root"):
            package = metadata.distribution("huggingarch")
        with contextlib.redirect_stdout(sys.stderr):
            result = perform(request)
        if package:
            result["huggingarch_version"] = package.version
            direct = json.loads(package.read_text("direct_url.json") or "{}")
            result["huggingarch_revision"] = direct.get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        result = {"status": "error", "error": "HuggingArch is not installed. Run with uv run --project tools/huggingarch python -m nexttransformer.integrations.huggingarch"}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
