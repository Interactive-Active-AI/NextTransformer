"""Pull-request classification and the public-content scan."""
from nexttransformer.checks.changes import classify
from nexttransformer.checks.public import scan, tracked


def test_results_and_documentation_skip_the_fast_tests():
    result = classify(["results/runs/hc.json", "evidence/audits/hc.json", "docs/src/content/docs/results/runs/hc.md"])
    assert (result["tests"], result["site"], result["contract"], result["affected"]) == (False, True, [], [])


def test_a_new_plugin_runs_the_tests_and_affects_no_published_method():
    result = classify(["nexttransformer/methods/ring/__init__.py", "nexttransformer/methods/ring/method.json"])
    assert result["tests"] and not result["contract"] and result["affected"] == []


def test_the_cost_ledger_is_contract_and_affects_published_methods():
    result = classify(["nexttransformer/protocol/compute.py"])
    assert result["contract"] == ["nexttransformer/protocol/compute.py"]
    assert {"dense_full", "gdn", "attnres"} <= set(result["affected"])


def test_the_repository_has_no_findings():
    assert scan(tracked()) == []
