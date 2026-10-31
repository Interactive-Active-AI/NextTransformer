"""Bitwise comparison of every registered method at debug size, between two source trees.

    python -m nexttransformer.checks.golden --base origin/main
    python -m nexttransformer.checks.golden --dump OUT.json [--methods dense_full gdn]

`--dump` builds each method at debug size with a fixed seed, runs one forward and backward pass on
fixed tokens, and records the loss, the parameter names, SHA-256 digests of all parameters and all
gradients, and the model hash. `--base REF` writes that dump for REF (in a temporary git worktree)
and for the working tree, then lists every method whose values differ. A change to shared code that
should not change any published method must leave every method identical. The comparison covers
the CPU code paths; kernels that only run on GPUs need their own bounded check.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV = dict(CUDA_VISIBLE_DEVICES="", NANOCHAT_DTYPE="float32", TORCH_COMPILE_DISABLE="1", OMP_NUM_THREADS="1")


def digest(tensors):
    h = hashlib.sha256()
    for name, tensor in tensors:
        h.update(name.encode())
        h.update(b"none" if tensor is None else tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dump(methods=None):
    """{method: values} for the source tree this interpreter imports."""
    import torch
    from nexttransformer.models.config import METHODS, debug_config
    from nexttransformer.models.lm import build_model
    torch.use_deterministic_algorithms(True)
    out = {}
    for method in methods or METHODS:
        config = debug_config(method)
        torch.manual_seed(0)
        model = build_model(config)
        if hasattr(model, "set_step"):
            model.set_step(getattr(config, "burnin_steps", 0)+1)
        generator = torch.Generator().manual_seed(1)
        idx = torch.randint(config.vocab_size, (2, config.sequence_len), generator=generator)
        labels = torch.randint(config.vocab_size, idx.shape, generator=generator)
        loss = model(idx, labels)
        loss.backward()
        named = sorted(model.named_parameters())
        out[method] = dict(loss=float(loss.detach()).hex(), names=[n for n, _ in named], parameters=digest(named),
                           gradients=digest([(n, p.grad) for n, p in named]), model_hash=config.fingerprint)
    return out


def dump_tree(root, path, methods):
    env = dict(os.environ, **ENV, PYTHONPATH=str(root))
    command = [sys.executable, "-m", "nexttransformer.checks.golden", "--dump", str(path)]
    if methods:
        command += ["--methods", *methods]
    subprocess.run(command, cwd=root, env=env, check=True)
    return json.loads(Path(path).read_text())


def compare(base, methods=None):
    """Methods whose debug-size values differ between `base` and the working tree."""
    git = lambda *args: subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)/"base"
        git("worktree", "add", "--detach", str(tree), git("merge-base", base, "HEAD").strip())
        try:
            before = dump_tree(tree, Path(tmp)/"base.json", methods)
        finally:
            git("worktree", "remove", "--force", str(tree))
        after = dump_tree(REPO, Path(tmp)/"head.json", methods)
    differences = {}
    for method in sorted(set(before) | set(after)):
        a, b = before.get(method), after.get(method)
        if a != b:
            differences[method] = "added" if a is None else "removed" if b is None else \
                ", ".join(k for k in a if a[k] != b.get(k))
    return differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--base", help="git revision to compare the working tree with")
    action.add_argument("--dump", type=Path, help="write this tree's values to a JSON file")
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args(argv)
    if args.dump:
        for key, value in ENV.items():
            os.environ.setdefault(key, value)
        args.dump.write_text(json.dumps(dump(args.methods), indent=1)+"\n")
        return
    differences = compare(args.base, args.methods)
    for method, what in differences.items():
        print(f"{method:20} differs: {what}")
    print("all methods are bitwise identical" if not differences else f"{len(differences)} method(s) differ")
    raise SystemExit(1 if differences else 0)


if __name__ == "__main__":
    main()
