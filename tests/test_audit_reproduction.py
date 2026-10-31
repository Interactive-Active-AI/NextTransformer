"""Full-size cost audits reproduce the recorded proofs exactly (slow; run with -m slow).

The representative set covers each family and execution path: the SSSL anchor, sparse burn-in,
sampled loop depth, chunked linear attention, a linear hybrid, a sparse hybrid, MLA and the VE-off
control. A full re-audit of every configuration is the maintainer's full-audit workflow.
"""
import json

import pytest

from nexttransformer.protocol.calibrate import Auditor
from nexttransformer.protocol.spec import config_path, load_config

REPRESENTATIVE = ("dense", "dsa", "huginn", "deltanet", "kda", "gdn_hybrid_swa", "moba_hybrid", "mla_nope", "mha_no_ve")


@pytest.mark.slow
@pytest.mark.parametrize("method", REPRESENTATIVE)
def test_full_size_audit_reproduces_the_proof(method, tmp_path):
    path = config_path(method)
    proof = json.loads((path.parent/json.loads(path.read_text())["calibration"]["cost_record"]).read_text())
    audited = Auditor(tmp_path).aggregate(load_config(path))
    assert audited["total_training_flops"] == proof["total_training_flops"]
    assert [(s["stage"], s["optimizer_steps"], s["cost"]["total"]) for s in audited["stages"]] == \
           [(s["stage"], s["optimizer_steps"], s["cost"]["total"]) for s in proof["stages"]]
    assert audited["method_source"]["hash"] == proof["method_source"]["hash"]
