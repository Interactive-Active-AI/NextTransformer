"""Subprocess adapter, executed with HuggingArch's Python, not the training environment.

Keep imports of NextTransformer out of this file. The JSON protocol is private to
integrations.huggingarch; upstream diagnostics go to stderr.
"""
import contextlib
import hashlib
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
    # The checkout is explicit; never import an unrelated installed `backend`.
    sys.path.insert(0, request["huggingarch_root"])
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = perform(request)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
