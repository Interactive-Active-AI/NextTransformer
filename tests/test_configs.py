"""Every registered configuration is admitted by the current source.

Cost proofs are tied to the PyTorch build they were audited with, so this runs only in that environment.
"""
import json

import pytest
import torch

from nexttransformer.models.registry import METHODS
from nexttransformer.models.config import HYBRID
from nexttransformer.protocol.admission import admit_config, review_config
from nexttransformer.protocol.spec import config_path

AUDITED_TORCH = json.loads(config_path("dense").read_text())["calibration"]["torch"]


@pytest.mark.skipif(torch.__version__ != AUDITED_TORCH, reason=f"cost proofs were audited with PyTorch {AUDITED_TORCH}")
@pytest.mark.parametrize("method", METHODS)
def test_configuration_is_admitted(method):
    admit_config(config_path(method))


@pytest.mark.skipif(torch.__version__ != AUDITED_TORCH, reason=f"cost proofs were audited with PyTorch {AUDITED_TORCH}")
@pytest.mark.parametrize("method", sorted(HYBRID))
@pytest.mark.parametrize("mutation", ["tokens", "layout", "source", "unknown", "steps", "total"])
def test_hybrids_use_the_shared_fail_closed_admission(method, mutation, tmp_path):
    path = config_path(method)
    raw = json.loads(path.read_text())
    original = path.parent/raw["calibration"]["cost_record"]
    proof = json.loads(original.read_text())
    (tmp_path/"dense.json").write_text(original.with_name("dense.json").read_text())
    raw["calibration"]["cost_record"] = "proof.json"
    if mutation == "tokens": raw["protocol"]["total_tokens"] -= 1
    elif mutation == "layout": raw["model"]["layer_mixers"][0] = "attention"
    elif mutation == "source": proof["method_source"] = dict(proof["method_source"], hash="stale")
    elif mutation == "unknown": proof["unknown"] = {"uncovered": 1}
    elif mutation == "steps": proof["stages"][0]["optimizer_steps"] -= 1
    elif mutation == "total": proof["total_training_flops"] += 1
    (tmp_path/"proof.json").write_text(json.dumps(proof))
    (tmp_path/"config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        admit_config(tmp_path/"config.json")


def test_research_review_reports_stale_evidence_but_refuses_a_proof_of_another_model(tmp_path):
    config, proof, findings = review_config(config_path("gdn"))
    assert proof["model_hash"] == config.fingerprint
    assert (not findings) if torch.__version__ == AUDITED_TORCH else any("PyTorch version" in f for f in findings)
    path = config_path("gdn")
    raw = json.loads(path.read_text())
    original = path.parent/raw["calibration"]["cost_record"]
    (tmp_path/"dense.json").write_text(original.with_name("dense.json").read_text())
    raw["calibration"]["cost_record"] = "proof.json"
    (tmp_path/"config.json").write_text(json.dumps(raw))
    (tmp_path/"proof.json").write_text(json.dumps(dict(proof, method_source=dict(proof["method_source"], hash="stale"))))
    findings = review_config(tmp_path/"config.json")[2]
    assert "cost proof source/software mismatch" in findings and any("stale" in f for f in findings)
    with pytest.raises(ValueError):
        admit_config(tmp_path/"config.json")
    (tmp_path/"proof.json").write_text(json.dumps(dict(proof, unknown={"uncovered": 1})))
    with pytest.raises(ValueError, match="does not cover"):
        review_config(tmp_path/"config.json")
