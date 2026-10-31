"""Per-method source fingerprints: which changes invalidate a cost proof, and which do not."""
import types

import torch

from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.fingerprint import (DISPLAY, method_files, normalized, projected, published, status,
                                                  tables, traced)


def test_published_methods_match_their_audited_source():
    """Code that a published (leaderboard) method runs cannot change without a new audit."""
    stale = [method for method in published() if len(set(status(method))) != 1 or status(method)[1] is None]
    assert not stale, f"re-audit with nexttransformer.protocol.calibrate: {stale}"


def test_comments_docstrings_and_layout_do_not_count():
    plain = 'def f(x):\n    """Doc."""\n    return x+1  # one\n'
    restyled = 'def f( x ):\n\n    # a comment\n    return x + 1\n'
    assert normalized(plain) == normalized(restyled) != normalized('def f(x):\n    return x+2\n')
    assert normalized('x = f"{a!r:>{w}}"\n') == ["NAME x", "OP =", 'STRING f"{a!r:>{w}}"', "NEWLINE"]


def test_tables_keep_only_the_entries_of_the_method_and_its_mixers():
    names = frozenset(("gdn", "kda", "moba", "gdn_hybrid", "attention"))
    relevant = frozenset(("gdn_hybrid", "gdn", "attention"))
    source = 'A = {"gdn": 4, "kda": 2}\nB = frozenset(("kda",))\nC = {"field": ("gdn", "kda")}\nD = {"x": 1}\n'
    module = types.SimpleNamespace(A={"gdn": 4, "kda": 2}, B=frozenset(("kda",)), C={"field": ("gdn", "kda")}, D={"x": 1})
    before = normalized(source, tables(source, module, relevant, names))
    module.A, module.B = {"gdn": 4, "kda": 3, "moba": 1}, frozenset(("kda", "moba"))  # other methods' entries
    assert normalized(source, tables(source, module, relevant, names)) == before
    module.C = {"field": ("kda",)}  # a mixer of this hybrid leaves a group
    assert normalized(source, tables(source, module, relevant, names)) != before
    module.C, module.D = {"field": ("gdn", "kda")}, {"x": 2}  # a table that is not per method
    assert normalized(source, tables(source, module, relevant, names)) != before
    assert projected({"gdn": [1, 2], "kda": 0}, relevant, names) == {"gdn": [1, 2]}


def test_family_modules_belong_only_to_the_methods_that_run_them():
    files = {}
    for method in ("gdn", "qsa"):
        config = debug_config(method)
        with traced() as executed:
            model = build_model(config, "cpu")
            x = torch.zeros((1, config.sequence_len), dtype=torch.long)
            model(x, x).backward()
        files[method] = set(method_files(executed))
    assert "nexttransformer/models/linear.py" in files["gdn"] - files["qsa"]
    assert {"nexttransformer/models/sparse.py", "nexttransformer/models/sparse_kernels.py"} <= files["qsa"] - files["gdn"]


def test_registry_families_are_display_metadata():
    assert "nexttransformer/models/registry.py" in DISPLAY
    assert "nexttransformer/models/registry.py" not in method_files(["nexttransformer/models/config.py"])
