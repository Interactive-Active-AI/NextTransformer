"""Historical records retain their cumulative budgets and accounting identity."""
from copy import deepcopy
import hashlib
import json

import pytest

from nexttransformer.export_frozen import REPO, check_audit


@pytest.mark.parametrize("job", ["deltanet", "gla"])
def test_historical_run_matches_its_cumulative_audit(job):
    run = json.loads((REPO / "results/runs" / f"{job}.json").read_text())
    path = REPO / f"evidence/audits/{job}.json"
    audit = json.loads(path.read_text())
    check_audit(run, audit)
    assert run["auditSha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ("theory_flops", "formula_version", "periodic_final_val_bpb"):
        changed = deepcopy(audit)
        changed[key] = 0 if key != "formula_version" else "reference-aten-math-v2"
        with pytest.raises(ValueError):
            check_audit(run, changed)
