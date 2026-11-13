"""Export local architectures to HuggingArch and write per-model JSON analysis reports.

    python -m nexttransformer.integrations.huggingarch export --output .agent-work/huggingarch
    python -m nexttransformer.integrations.huggingarch --help

Export and report use only the standard library. HuggingArch runs in its own
Python environment; neither weights nor training data are required.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "facbbbc96856e558b9084a94a5070b2e2c86cf0b"
NOTICE = ("HuggingArch results are spec-derived inference estimates, not measured benchmarks or "
          "NextTransformer training cost proofs. Exports contain no checkpoint: tensor/storage "
          "validation is unavailable. Review each generated spec against forward.py, especially "
          "shared weights, value embeddings, residual scalars, sparse routing and dynamic depth.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def git_revision(root):
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def configurations(root):
    from nexttransformer.models.registry import METHODS
    from nexttransformer.protocol.spec import config_path
    paths = {method: config_path(method, root) for method in METHODS}
    paths.update({"native-" + path.stem: path for path in (root / "configs/native").glob("*.json")
                  if "model" in read_json(path)})
    return paths


def source_bundle(root, method):
    # Include the complete model package: lm.py imports family implementations,
    # and native/recurrent models depend on helpers beyond their entry point.
    paths = set((root / "nexttransformer/models").rglob("*.py"))
    paths.update(root / "nanochat" / name for name in ("common.py", "flash_attention.py", "optim.py"))
    paths.add(root / "nexttransformer/protocol/spec.py")
    from nexttransformer.models.registry import PLUGINS
    if method in PLUGINS:
        plugin = Path(PLUGINS[method]["path"]).resolve()
        if not plugin.is_relative_to(root.resolve()):
            raise ValueError("Export requires the plugin source inside the repository")
        paths.update(plugin.rglob("*.py"))
    sources = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted(paths)}
    return sources, "\n\n".join(f"# Source: {name}\n{text}" for name, text in sources.items())


def reviewed_spec(root, model, config):
    """Select a fixed spec by method and its explicitly supported configuration.

    Architecture changes require updating the corresponding spec in review.
    Weight precision is an analysis assumption, not part of the graph layout.
    """
    directory = root / "tools/huggingarch/specs"
    catalog_path = directory / "catalog.json"
    if not catalog_path.exists():
        return None
    architecture = {key: value for key, value in config.items() if key != "torch_dtype"}
    for entry in read_json(catalog_path)["models"]:
        if entry["method"] != model["method"] or entry["config"] != architecture:
            continue
        if not entry.get("spec_file"):
            return None, entry
        path = directory / entry["spec_file"]
        if path.name != model["method"] + ".yaml" or path.parent != directory:
            raise ValueError("Invalid reviewed spec filename")
        return path.read_text(encoding="utf-8"), entry
    return None


def export_bundle(output, methods=None, root=ROOT, weight_dtype="bfloat16", smoke=False):
    if smoke and not methods:
        raise ValueError("Smoke export requires an explicit --methods selection")
    available = configurations(root)
    selected = sorted(available) if not methods else list(dict.fromkeys(methods))
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError("Unknown methods: " + ", ".join(sorted(unknown)))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty; reuse the existing bundle or choose a new directory")
    entries = []
    # Prepare everything before writing so a missing configuration cannot leave a partial bundle.
    prepared = []
    for method in selected:
        path = available[method]
        raw = read_json(path)
        native = method.startswith("native-")
        if smoke and native:
            raise ValueError("Smoke export currently supports registered methods, not historical native configurations")
        if native:
            from nexttransformer.models.native.config import NativeModelConfig
            config = NativeModelConfig(**raw["model"])
            model = asdict(config)
        else:
            from nexttransformer.protocol.spec import load_config
            config = load_config(path)
            model = config.serialized()
        if smoke:
            from nexttransformer.models.config import debug_config
            config = debug_config(method)
            model = config.serialized()
            raw = dict(model=model, model_hash=config.fingerprint, profile="smoke",
                       parent_config=path.relative_to(root).as_posix(), parent_model_hash=raw.get("model_hash"))
        _, forward = source_bundle(root, method)
        # model_type must never accidentally select a stock Llama/Qwen spec.
        exported = dict(model, model_type="nexttransformer_" + method.replace("-", "_"),
                        hidden_size=config.width, num_attention_heads=config.heads,
                        num_hidden_layers=config.n_layer, intermediate_size=config.mlp_width,
                        max_position_embeddings=config.sequence_len,
                        tie_word_embeddings=False, torch_dtype=weight_dtype)
        # Native dataclasses contain tuples; compare their serialized JSON form.
        exported = json.loads(json.dumps(exported))
        name = f"nextt-{method}" + ("-smoke" if smoke else "")
        proof_path = raw.get("calibration", {}).get("cost_record")
        proof = read_json(path.parent / proof_path) if proof_path else {}
        if proof and proof.get("model_hash") != raw.get("model_hash"):
            raise ValueError(f"Cost proof model hash mismatch: {method}")
        entry = dict(method=method, family="native" if native else path.parent.name, profile="smoke" if smoke else "configured",
                     name=name, model_id="custom:" + name, directory=method,
                     config_path=path.relative_to(root).as_posix(),
                     parameters=proof.get("parameters"),
                     training_flops_per_token=proof.get("flops_per_token"),
                     training_formula=proof.get("formula_version"),
                     cost_record=(path.parent / proof_path).resolve().relative_to(root.resolve()).as_posix() if proof_path else None,
                     physical_layers=config.n_layer, weight_dtype_assumption=weight_dtype,
                     checkpoint_validated=False)
        entries.append(entry)
        prepared.append((method, exported, forward, raw, reviewed_spec(root, entry, exported)))
    output.mkdir(parents=True, exist_ok=True)
    for method, config, forward, raw, spec in prepared:
        directory = output / method
        directory.mkdir()
        write_json(directory / "config.json", config)
        write_json(directory / "nexttransformer-config.json", raw)
        (directory / "forward.py").write_text(forward, encoding="utf-8")
        if spec:
            if spec[0] is not None:
                (directory / "spec.yaml").write_text(spec[0], encoding="utf-8")
            write_json(directory / "spec-review.json", spec[1])
    manifest = dict(schema_version=1, source_revision=git_revision(root),
                    huggingarch_tested_revision=UPSTREAM_REVISION, notice=NOTICE, models=entries)
    write_json(output / "manifest.json", manifest)
    write_report(output)
    return manifest


def load_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    seen = set()
    for model in manifest["models"]:
        name = model["method"]
        if not re.fullmatch(r"[a-z0-9_-]+", name) or model["directory"] != name or name in seen:
            raise ValueError("Invalid or duplicate model directory")
        seen.add(name)
        directory = bundle / name
        if not directory.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Model directory escapes bundle")
        expected_name = f"nextt-{name}" + ("-smoke" if model.get("profile") == "smoke" else "")
        if model["name"] != expected_name or model["model_id"] != "custom:" + expected_name:
            raise ValueError("Invalid custom model name")
        review_path = directory / "spec-review.json"
        if review_path.exists():
            config = read_json(directory / "config.json")
            architecture = {key: value for key, value in config.items() if key != "torch_dtype"}
            if read_json(review_path)["config"] != architecture:
                raise ValueError(f"Bundled spec does not support the configuration for {name}")
    return manifest


def run_worker(request, python, timeout):
    worker = Path(__file__).with_name("huggingarch_worker.py")
    try:
        with subprocess.Popen([str(python), str(worker)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=request.get("huggingarch_root"), text=True,
                              start_new_session=True) as proc:
            try:
                stdout, stderr = proc.communicate(json.dumps(request), timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Generation spawns an agent and validator; stop the whole job.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                raise
        if proc.returncode:
            return {"status": "error", "error": f"Worker exited {proc.returncode}: {stderr[-2000:]}"}
        result = json.loads(stdout)
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError("Missing worker status")
        return result
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"HuggingArch exceeded {timeout:g} seconds"}
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}


def run_bundle(args):
    bundle = args.bundle.resolve()
    manifest = load_bundle(bundle)
    upstream = args.huggingarch_root.resolve() if args.huggingarch_root else None
    if upstream and not (upstream / "backend/custom_model.py").is_file():
        raise ValueError("--huggingarch-root must be a HuggingArch checkout")
    python = args.huggingarch_python.absolute()  # Preserve venv symlinks.
    if not python.is_file():
        raise ValueError("--huggingarch-python must name the Python executable in its environment")
    models = manifest["models"]
    if args.methods:
        unknown = set(args.methods) - {m["method"] for m in models}
        if unknown:
            raise ValueError("Methods not in bundle: " + ", ".join(sorted(unknown)))
        models = [m for m in models if m["method"] in args.methods]
    failures = 0
    for model in models:
        request = dict(action=args.command, name=model["name"], directory=str(bundle / model["directory"]),
                       huggingarch_root=str(upstream) if upstream else None, batch_size=args.batch_size,
                       seq_len=args.seq_len or read_json(bundle / model["directory"] / "config.json")["max_position_embeddings"],
                       dtype=args.dtype, agent_cli=getattr(args, "agent_cli", None))
        result = run_worker(request, python, args.timeout)
        if upstream:
            result["huggingarch_revision"] = git_revision(upstream)
        result.update(model_id=model["model_id"], checkpoint_validated=False)
        if args.command == "analyze":
            result["settings"] = {key: request[key] for key in ("batch_size", "seq_len", "dtype")}
        write_json(bundle / model["directory"] / (args.command + ".json"), result)
        if args.command == "generate" and result["status"] == "spec_ready":
            analysis_path = bundle / model["directory"] / "analyze.json"
            if analysis_path.exists() and read_json(analysis_path)["status"] != "analyzed":
                analysis_path.unlink()  # Clear an obsolete needs_spec/error result.
        print(f"{model['method']}: {result['status']}")
        failures += result["status"] not in ("registered", "analyzed", "spec_ready")
    write_report(bundle)
    return 1 if failures else 0


def write_report(bundle):
    """Aggregate per-model results without importing HuggingArch."""
    manifest = load_bundle(bundle)
    records = []
    for model in manifest["models"]:
        directory = bundle / model["directory"]
        result_path = next((directory / filename for filename in
                            ("analyze.json", "generate.json", "register.json")
                            if (directory / filename).exists()), None)
        result = read_json(result_path) if result_path else {"status": "exported"}
        review_path = directory / "spec-review.json"
        review = read_json(review_path) if review_path.exists() else {}
        records.append(dict(model=model, result=result, spec_review=review))
    report = dict(schema_version=1, notice=NOTICE, models=records)
    write_json(bundle / "report.json", report)
    return report


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export all registered and historical native configurations")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--methods", nargs="+")
    export.add_argument("--smoke", action="store_true", help="export explicitly selected debug-sized models without formal cost evidence")
    export.add_argument("--weight-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="explicit unquantized weight precision assumption (no checkpoint is exported)")
    report = commands.add_parser("report", help="aggregate local results into report.json without HuggingArch")
    report.add_argument("--bundle", type=Path, required=True)
    for name in ("register", "generate", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--methods", nargs="+")
        command.add_argument("--huggingarch-root", type=Path, help="optional source checkout override")
        command.add_argument("--huggingarch-python", type=Path, default=Path(sys.executable),
                             help="optional interpreter override; defaults to this Python's installed dependency")
        command.add_argument("--batch-size", type=positive_int, default=1)
        command.add_argument("--seq-len", type=positive_int, help="defaults to each exported model's context length")
        command.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        command.add_argument("--timeout", type=positive_int, default=600 if name == "generate" else 120,
                             help="per-model timeout in seconds (generate: 600; others: 120)")
        if name == "generate":
            command.add_argument("--agent-cli", choices=("codex", "claude", "kimi"),
                                 help="upstream agent; defaults to HuggingArch's configured agent")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            manifest = export_bundle(args.output.resolve(), args.methods, weight_dtype=args.weight_dtype, smoke=args.smoke)
            print(f"Exported {len(manifest['models'])} models: {args.output / 'report.json'}")
        elif args.command == "report":
            write_report(args.bundle.resolve())
            print(args.bundle / "report.json")
        else:
            return run_bundle(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
