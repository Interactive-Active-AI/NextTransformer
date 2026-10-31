"""Per-method source fingerprints of cost proofs.

A cost proof covers the source its method runs, not the whole repository. While a method's cost is
audited, calibration records the repository files whose code executes and adds the repository modules
those files import, so kernels that the CPU audit does not run are covered as well. Imports of the
shared files are not followed: the hubs `lm.py` and the models package import every family, and the
core files below are always included. The fingerprint hashes the files without comments, docstrings
or layout, and reduces module-level data tables of model files to the entries of the method and its
layer mixers. Registry families are display metadata and are never fingerprinted.

Editing another method's module or table entries, or only comments, therefore keeps a proof valid;
any change to code the method runs, to the protocol, cost ledger, calibration or optimizer requires a
new audit.

    python -m nexttransformer.protocol.fingerprint [--all]

compares every registered configuration with the current source. It fails when a published
(leaderboard) method no longer matches its proof, or with --all when any configuration does not.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[2]
VERSION = "method-source-v1"
CORE = ("nanochat/common.py", "nanochat/flash_attention.py", "nanochat/optim.py",
        "nexttransformer/protocol/calibrate.py", "nexttransformer/protocol/compute.py",
        "nexttransformer/protocol/fingerprint.py", "nexttransformer/protocol/spec.py",
        "nexttransformer/training/optimizer.py")
DISPLAY = frozenset(("nexttransformer/models/registry.py",))
HUBS = frozenset(("nexttransformer/models/__init__.py", "nexttransformer/models/lm.py"))
# The plugin interface counts only for the methods that run it, never as an import of shared code.
EXECUTED_ONLY = frozenset(("nexttransformer/models/plugins.py",))
PACKAGES = ("nanochat", "nexttransformer")
SCALARS = (str, int, float, bool, type(None))
DATA = (dict, frozenset, set, tuple, list, *SCALARS)


def plugin_roots():
    """Repository paths of the plugin roots inside the repository (see models/plugins.py)."""
    from nexttransformer.models.plugins import roots
    out = []
    for root in roots():
        try: out.append(root.resolve().relative_to(ROOT).as_posix())
        except ValueError: pass
    return tuple(out)


def relative(filename, prefixes=()):
    """Repository path of a Python source file of the packages or of a plugin root, else None."""
    try: path = Path(filename).resolve().relative_to(ROOT)
    except (OSError, ValueError): return None
    posix = path.as_posix()
    inside = path.parts[0] in PACKAGES or any(posix.startswith(prefix + "/") for prefix in prefixes)
    return posix if inside and path.suffix == ".py" else None


@contextmanager
def traced():
    """Repository files whose Python code runs inside the block, on this thread."""
    files, known, prefixes = set(), {}, plugin_roots()

    def trace(frame, event, arg):
        name = frame.f_code.co_filename
        if name not in known: known[name] = relative(name, prefixes)
        if known[name]: files.add(known[name])

    previous = sys.gettrace()
    sys.settrace(trace)
    try: yield files
    finally: sys.settrace(previous)


def imports(path):
    """Repository modules imported anywhere in the file at `path`."""
    out = set()
    for node in ast.walk(ast.parse((ROOT/path).read_text())):
        if isinstance(node, ast.Import): modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        else: continue
        for module in modules:
            if module.split(".")[0] not in PACKAGES: continue
            base = ROOT/Path(*module.split("."))
            for candidate in (base.with_suffix(".py"), base/"__init__.py"):
                if candidate.is_file(): out.add(candidate.relative_to(ROOT).as_posix()); break
    return out


def method_files(executed):
    """Executed files plus, transitively, the repository modules imported by the non-shared ones.

    A plugin's whole package belongs to its method: every Python file of a plugin directory that ran is added."""
    executed = set(executed)
    for root in plugin_roots():
        for path in list(executed):
            package = ROOT/root/path[len(root)+1:].split("/")[0]
            if path.startswith(root + "/") and package.is_dir():
                executed |= {p.relative_to(ROOT).as_posix() for p in package.rglob("*.py")}
    files, pending = set(), [path for path in executed if path not in DISPLAY]
    while pending:
        path = pending.pop()
        if path in files: continue
        files.add(path)
        if path not in HUBS and path not in CORE:
            pending += [other for other in imports(path) if other not in DISPLAY | EXECUTED_ONLY and other not in files]
    return sorted(files)


def normalized(source, tables=None):
    """Tokens of `source` without comments, docstrings or layout, identical on Python 3.10 to 3.13.

    `tables` maps a statement's ((line, column), (end line, end column)) to the text replacing it."""
    tree = ast.parse(source)
    docstrings = {(n.body[0].lineno, n.body[0].col_offset) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                  and isinstance(n.body[0].value.value, str)}
    spans = sorted((tables or {}).items())
    lines = source.splitlines(keepends=True)
    out, after_docstring, depth, start, span = [], False, 0, None, 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = tokenize.tok_name[token.type]
        while span < len(spans) and token.start >= spans[span][0][1]: span += 1
        if span < len(spans) and token.start >= spans[span][0][0]:
            if spans[span][1] is not None: out.append(spans[span][1]); spans[span] = (spans[span][0], None)
            continue
        if kind == "FSTRING_START":  # Python 3.12+ splits f-strings; keep them one token as before
            if not depth: start = token.start
            depth += 1; continue
        if depth:
            if kind == "FSTRING_END":
                depth -= 1
                if not depth: out.append("STRING " + segment(lines, start, token.end))
            continue
        if kind in ("COMMENT", "NL", "ENCODING", "ENDMARKER"): continue
        if kind == "STRING" and token.start in docstrings: after_docstring = True; continue
        if kind == "NEWLINE" and after_docstring: after_docstring = False; continue
        after_docstring = False
        out.append(kind if kind in ("NEWLINE", "INDENT", "DEDENT") else f"{kind} {token.string}")
    return out


def segment(lines, start, end):
    (row, column), (end_row, end_column) = start, end
    if row == end_row: return lines[row-1][column:end_column]
    return lines[row-1][column:] + "".join(lines[row:end_row-1]) + lines[end_row-1][:end_column]


def canonical(value):
    """A JSON-serializable, deterministic form of a module-level value."""
    if isinstance(value, dict): return [[canonical(k), canonical(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(value, (frozenset, set)): return sorted((canonical(v) for v in value), key=repr)
    if isinstance(value, (tuple, list)): return [canonical(v) for v in value]
    if isinstance(value, SCALARS): return value
    if hasattr(value, "__qualname__"): return f"{getattr(value, '__module__', '')}.{value.__qualname__}"
    text = repr(value)
    return f"{type(value).__module__}.{type(value).__qualname__}" if " at 0x" in text else text


def projected(value, relevant, names):
    """`value` reduced to the `relevant` entries where it lists, or is keyed by, method and mixer `names`."""
    if isinstance(value, dict) and not value: return {}  # the same form as a per-method table without relevant entries
    if isinstance(value, dict) and all(isinstance(k, str) and k in names for k in value):
        return {k: canonical(value[k]) for k in sorted(relevant & value.keys())}
    if isinstance(value, dict):
        return [[canonical(k), projected(v, relevant, names)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(value, (frozenset, set, tuple, list)) and value and all(isinstance(x, str) and x in names for x in value):
        return sorted(relevant & set(value))
    if isinstance(value, (frozenset, set)): return sorted((projected(v, relevant, names) for v in value), key=repr)
    if isinstance(value, (tuple, list)): return [projected(v, relevant, names) for v in value]
    return canonical(value)


def tables(source, module, relevant, names):
    """Spans of the module-level data assignments of `source`, with their projected values."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign): targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None: targets = [node.target]
        else: continue
        if not all(isinstance(t, ast.Name) and isinstance(getattr(module, t.id, None), DATA) for t in targets): continue
        values = {t.id: projected(getattr(module, t.id), relevant, names) for t in targets}
        out[((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset))] = \
            "TABLE " + json.dumps(values, sort_keys=True)
    return out


def file_text(path, relevant=frozenset(), names=frozenset()):
    """Normalized source of a repository file; model files keep only the `relevant` table entries."""
    file = ROOT/path
    if not file.is_file(): return "MISSING"
    source = file.read_text()
    spans = None
    if path.startswith("nexttransformer/models/"):
        name = path[:-3].replace("/", ".").removesuffix(".__init__")
        spans = tables(source, importlib.import_module(name), relevant, names)
    return "\n".join(normalized(source, spans))


def mixers(config):
    from nexttransformer.models.config import HYBRID
    return tuple(config.layer_mixers) if config.method in HYBRID else ()


def method_names():
    """Registered methods and the mixer kinds of their layouts: the keys of per-method tables."""
    from nexttransformer.models.config import candidate_config
    from nexttransformer.models.registry import METHODS
    return frozenset(METHODS).union(*(mixers(candidate_config(method)) for method in METHODS))


def method_source(method, files, config):
    """Fingerprint of the source `method` runs: its recorded files plus the core, normalized and projected."""
    relevant, names = frozenset((method, *mixers(config))), method_names()
    digest = hashlib.sha256(VERSION.encode())
    for path in sorted(set(files) | set(CORE)):
        if path in DISPLAY: continue
        digest.update(f"\0{path}\0".encode())
        digest.update(file_text(path, relevant, names).encode())
    return digest.hexdigest()


def record(method, executed, config):
    """The `method_source` entry of a cost proof, from the files executed while it was audited."""
    files = method_files(executed)
    return dict(version=VERSION, files=files, hash=method_source(method, files, config))


def published():
    """Registered methods with a published run under results/runs/."""
    from nexttransformer.models.registry import METHODS
    ids = {json.loads(path.read_text())["methodId"] for path in (ROOT/"results"/"runs").glob("*.json")}
    return [method for method in METHODS if method in ids]


def status(method):
    """(recorded, current) fingerprints of a registered configuration; current is None without a record."""
    from nexttransformer.protocol.spec import config_path, load_config
    path = config_path(method)
    calibration = json.loads(path.read_text())["calibration"]
    source = json.loads((path.parent/calibration["cost_record"]).read_text()).get("method_source")
    current = method_source(method, source["files"], load_config(path)) if source else None
    return calibration.get("method_source_hash"), current


def main(argv=None):
    from nexttransformer.models.registry import METHODS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--all", action="store_true", help="fail on any configuration whose proof no longer matches")
    args = parser.parse_args(argv)
    board, stale = set(published()), []
    for method in args.methods:
        recorded, current = status(method)
        matched = current is not None and recorded == current
        if not matched: stale.append(method)
        print(f"{method:20} {'published' if method in board else 'registered':10} "
              f"{'matches its proof' if matched else 'needs a new audit' if current else 'no per-method fingerprint'}")
    affected = [method for method in stale if method in board]
    if affected: print("published methods whose code changed since their audit: " + ", ".join(affected))
    raise SystemExit(1 if affected or (args.all and stale) else 0)


if __name__ == "__main__":
    main()
