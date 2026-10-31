"""The run audit accepts a complete formal run directory and names what is wrong with an incomplete one."""
import json

from nexttransformer.audit_run import SSSL_ANCHOR, audit
from nexttransformer.export_run import RESULTS
from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256

P = Protocol()


def write(path, value):
    path.write_text(json.dumps(value))


def run_directory(root):
    """A synthetic finished run with the published Dense Baseline's evaluation results."""
    r = json.loads((RESULTS/"runs/dense_full.json").read_text())
    flops = r["trainingCompute"]["trainingFlopsEstimate"]
    ledger = dict(optimizer_step=P.steps, input_tokens=P.total_tokens, protocol_hash=P.fingerprint,
                  val_steps=list(P.validation_steps), core_steps=list(P.core_steps), theory_flops=flops)
    write(root/"run_status.json", dict(phase="training_complete", step=P.steps, final_status="matched", budget=ledger,
                                       evaluation_forward_flops=1.0))
    write(root/"budget.json", dict(ledger, charged_theory_flops=flops))
    cost = dict(flops_per_token=flops/P.total_tokens, relative_error=flops/P.total_tokens/SSSL_ANCHOR-1,
                dense_flops_per_token=SSSL_ANCHOR)
    write(root/"provenance.json", dict(model_hash="m"*64, protocol_hash=P.fingerprint, data_manifest_sha256="d"*64,
                                       tokenizer_sha256=TOKENIZER_SHA256, parameters=1, model={"layers": 23}, cost=cost,
                                       software=dict(science_hash="s"*64, reference_source_hash="r"*64, torch="2.9.1"),
                                       smoke=False, admission=dict(mode="formal"), world_size=8, micro_batch=8,
                                       accumulation=8, resumed_from=None))
    events = [dict(event="train", step=s, input_tokens=s*P.global_batch_tokens, loss=2.5, grad_norm=1.0,
                   theory_flops=flops*s/P.steps, theory_relative_error=cost["relative_error"])
              for s in range(250, P.steps+1, 250)] + [dict(event="train", step=P.steps, input_tokens=P.total_tokens, loss=2.2,
                                                         grad_norm=1.0, theory_flops=flops, theory_relative_error=0.0)]
    events = sorted({e["step"]: e for e in events}.values(), key=lambda e: e["step"])
    for point in r["val"]:
        events.append(dict(event="validation", step=point["step"], val_bpb=point["value"], input_tokens=P.val_tokens,
                           smoke=False, protocol_hash=P.fingerprint))
        write(root/f"val_{point['step']:08d}.json", dict(step=point["step"], val_bpb=point["value"]))
    for e, c in zip(r["taskEvaluations"], r["core"]):
        events.append(dict(event="core", step=c["step"], core_metric=c["value"], results=e["raw"],
                           centered_results=e["centered"], full_core=True))
        name = "core_final.json" if c["step"] == P.steps else f"core_{c['step']:08d}.json"
        write(root/name, dict(core_metric=c["value"]))
    events.append(dict(event="final_bpb", val_bpb=r["finalBpb"], train_bpb=r["finalBpb"]+.002))
    write(root/"bpb_final.json", dict(val=dict(bpb=r["finalBpb"], input_tokens=P.final_bpb_tokens),
                                      train=dict(bpb=r["finalBpb"]+.002, input_tokens=P.final_bpb_tokens)))
    write(root/"latest.json", dict(step=P.steps, checkpoint=f"checkpoint_{P.steps:08d}"))
    (root/"metrics.jsonl").write_text("\n".join(map(json.dumps, events))+"\n")
    return r


def test_a_complete_run_passes_and_the_report_names_nothing_local(tmp_path):
    r = run_directory(tmp_path)
    report, problems = audit(tmp_path, "dense_full")
    assert problems == [] and report["ok"]
    assert (report["validation_events"], report["core_events"]) == (189, 6)
    assert report["core_final"] == r["core"][-1]["value"] and report["final_bpb"]["val"] == r["finalBpb"]
    text = json.dumps(report)
    assert str(tmp_path) not in text and "host" not in text and "device" not in text


def test_an_incomplete_run_is_reported(tmp_path):
    run_directory(tmp_path)
    (tmp_path/"val_00046875.json").unlink()
    provenance = json.loads((tmp_path/"provenance.json").read_text())
    write(tmp_path/"resume_00020000.json", dict(provenance, admission=dict(mode="research"), resumed_from="ckpt"))
    report, problems = audit(tmp_path, "dense_full")
    assert not report["ok"]
    assert any("val_*.json files" in p for p in problems)
    assert any("research-mode segment" in p for p in problems)
    assert report["segments"][1]["step"] == 20000
