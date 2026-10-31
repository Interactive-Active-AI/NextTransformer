"""Read-only cumulative audit of a finished formal run directory.

    python -m nexttransformer.audit_run RUN_DIR --method gdn
    python -m nexttransformer.export_run RUN_DIR --method gdn --name "Gated DeltaNet · GDN" --audit evidence/audits/gdn.json

The audit re-reads everything the trainer wrote and checks it against the protocol: the ledger
(46,875 updates, 49.152B input tokens, charged == theory FLOPs, cumulative error within 2% of the
SSSL anchor), the training events, all 189 scheduled validation events and their files, the six
full 22-task CORE events with their centering and aggregates, the independent final BPB, the final
checkpoint, and the provenance of every training segment (identical model, protocol, data,
tokenizer, configuration, cost proof and source hashes; formal mode; a registered batch layout).

It writes evidence/audits/<method>.json in the public run-audit-v1 format. The file records
scientific quantities, fingerprints and the data layout only; it never contains host names,
paths, device names, timings or tracking-service details. The command exits non-zero when any
check fails, and the file then lists the problems.
"""
import argparse
import json
import math
from datetime import date
from pathlib import Path

from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256

REPO = Path(__file__).resolve().parents[1]
P = Protocol()
SSSL_ANCHOR = 4672398325.44785
CHECKS = [
    "run_status.json: training_complete at the final update with final_status matched",
    "ledger (run_status.json, budget.json): protocol updates and input tokens, protocol validation/CORE steps, charged == theory FLOPs",
    "cumulative theory FLOPs within 2% of input tokens x SSSL anchor; budget_anomaly events counted",
    "train events: strictly increasing steps, input_tokens == step x global batch, finite loss and gradient norm, final cumulative FLOPs equal the ledger",
    "189 validation events at the protocol steps with the protocol token count, equal to the val_*.json files",
    "6 CORE events, each the full 22-task evaluation with reproduced centering and aggregate, equal to the core_*.json files",
    "one independent final BPB event equal to bpb_final.json; latest.json names the final checkpoint",
    "every segment: identical fingerprints, parameter count, configuration, cost proof and source hashes; formal mode; non-smoke; registered batch layout",
]


def audit(run, method):
    """(report, problems) for the run directory `run`."""
    run = Path(run)
    problems = []
    need = lambda ok, message: ok or problems.append(message)
    load = lambda name: json.loads((run/name).read_text())
    tasks = {t["id"]: t["randomBaseline"] for t in json.loads((REPO/"website/data/core-tasks.json").read_text())["tasks"]}
    val_steps, core_steps = list(P.validation_steps), list(P.core_steps)

    status, budget = load("run_status.json"), load("budget.json")
    ledger = status["budget"]
    need(status["phase"] == "training_complete" and status["step"] == P.steps, "run_status: not training_complete at the final update")
    need(status["final_status"] == "matched", f"run_status: final_status {status['final_status']}")
    for source, b in (("run_status", ledger), ("budget.json", budget)):
        need(b["optimizer_step"] == P.steps and b["input_tokens"] == P.total_tokens, f"{source}: updates or tokens differ from the protocol")
        need(b["protocol_hash"] == P.fingerprint, f"{source}: protocol hash differs")
        need(b["val_steps"] == val_steps and b["core_steps"] == core_steps, f"{source}: validation/CORE steps differ from the protocol")
    need(budget["charged_theory_flops"] == budget["theory_flops"] == ledger["theory_flops"], "budget: charged and theory FLOPs differ")

    names = ["provenance.json", *sorted(f.name for f in run.glob("resume_*.json"))]
    first = load(names[0])
    anchor = first.get("cost", {}).get("dense_flops_per_token", SSSL_ANCHOR)
    cumulative = ledger["theory_flops"]/(P.total_tokens*anchor)-1
    need(abs(cumulative) <= P.cumulative_tolerance, f"cumulative error {cumulative:.4%} exceeds {P.cumulative_tolerance:.0%}")

    events = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines() if line.strip()]
    train = [e for e in events if e["event"] == "train"]
    steps = [e["step"] for e in train]
    need(bool(train) and steps[-1] == P.steps, "train: the last event is not the final update")
    need(all(a < b for a, b in zip(steps, steps[1:])), "train: steps are not strictly increasing")
    need(all(e["input_tokens"] == e["step"]*P.global_batch_tokens for e in train), "train: input_tokens != step x global batch")
    need(all(math.isfinite(e["loss"]) and math.isfinite(e["grad_norm"]) for e in train), "train: non-finite loss or gradient norm")
    need(bool(train) and train[-1]["theory_flops"] == ledger["theory_flops"], "train: final cumulative FLOPs differ from the ledger")
    relative = [e["theory_relative_error"] for e in train if "theory_relative_error" in e]

    val = [e for e in events if e["event"] == "validation"]
    need([e["step"] for e in val] == val_steps, "validation: schedule differs from the protocol")
    need(all(e["input_tokens"] == P.val_tokens and not e.get("smoke") and e["protocol_hash"] == P.fingerprint for e in val),
         "validation: tokens, smoke flag or protocol hash")
    files = sorted(run.glob("val_*.json"))
    need(len(files) == len(val_steps), f"validation: {len(files)} val_*.json files")
    by_step = {e["step"]: e for e in val}
    for f in files:
        v = json.loads(f.read_text())
        need(by_step.get(v["step"], {}).get("val_bpb") == v["val_bpb"], f"{f.name}: differs from metrics.jsonl")

    core = [e for e in events if e["event"] == "core"]
    need([e["step"] for e in core] == core_steps, "CORE: schedule differs from the protocol")
    for e in core:
        full = e.get("full_core") and set(e["results"]) == set(tasks) == set(e["centered_results"])
        need(full, f"CORE {e['step']}: not the full {len(tasks)}-task evaluation")
        if not full:
            continue
        for task, base in tasks.items():
            need(abs((e["results"][task]-base)/(1-base)-e["centered_results"][task]) <= 1e-7, f"CORE {e['step']}: {task} centering")
        need(abs(sum(e["centered_results"].values())/len(tasks)-e["core_metric"]) <= 1e-7, f"CORE {e['step']}: aggregate")
        name = "core_final.json" if e["step"] == P.steps else f"core_{e['step']:08d}.json"
        need(load(name)["core_metric"] == e["core_metric"], f"{name}: differs from metrics.jsonl")

    finals = [e for e in events if e["event"] == "final_bpb"]
    need(len(finals) == 1, f"{len(finals)} final_bpb events")
    bpb = load("bpb_final.json")
    need(bpb["val"]["input_tokens"] == bpb["train"]["input_tokens"] == P.final_bpb_tokens, "bpb_final: tokens per split")
    need(bool(finals) and finals[0]["val_bpb"] == bpb["val"]["bpb"] and finals[0]["train_bpb"] == bpb["train"]["bpb"],
         "bpb_final: differs from metrics.jsonl")
    need(load("latest.json") == {"step": P.steps, "checkpoint": f"checkpoint_{P.steps:08d}"}, "latest.json does not name the final checkpoint")

    keys = ("model_hash", "protocol_hash", "data_manifest_sha256", "tokenizer_sha256", "parameters")
    segments = []
    for name in names:
        p = load(name)
        for k in keys:
            need(p[k] == first[k], f"{name}: {k} differs from the first segment")
        need(p["model"] == first["model"] and p["cost"] == first["cost"], f"{name}: configuration or cost proof differs")
        for k in ("science_hash", "reference_source_hash"):
            need(p["software"][k] == first["software"][k], f"{name}: {k} differs from the first segment")
        need(not p.get("smoke"), f"{name}: smoke segment")
        need(p.get("admission", {}).get("mode", "formal") == "formal", f"{name}: research-mode segment")
        need(p["world_size"]*p["micro_batch"]*p["accumulation"]*P.sequence_len == P.global_batch_tokens, f"{name}: batch layout")
        need(p["world_size"] in (8, 16, 32), f"{name}: unregistered world size {p['world_size']}")
        step = 0 if name == "provenance.json" else int(name.removeprefix("resume_").removesuffix(".json"))
        segments.append(dict(file=name, step=step, world=p["world_size"], micro=p["micro_batch"], accumulation=p["accumulation"]))
    need(first["protocol_hash"] == P.fingerprint, "provenance: protocol hash differs from the frozen protocol")
    need(first["tokenizer_sha256"] == TOKENIZER_SHA256, "provenance: tokenizer differs from the pinned tokenizer")
    need(abs(first["cost"]["relative_error"]) <= P.preflight_tolerance, "provenance: cost proof outside the 1% preflight")

    periodic = {e["step"]: e["val_bpb"] for e in val}
    best = min(val, key=lambda e: e["val_bpb"]) if val else None
    report = dict(
        schema="run-audit-v1", run=method, method=method, date=date.today().isoformat(),
        scope="Read-only cumulative audit of the run directory against the reference protocol.",
        checks=CHECKS, ok=not problems, problems=problems, final_status=status["final_status"],
        optimizer_steps=ledger["optimizer_step"], input_tokens=ledger["input_tokens"], theory_flops=ledger["theory_flops"],
        dense_flops_per_token=anchor, flops_per_token=first["cost"]["flops_per_token"],
        proof_relative_error=first["cost"]["relative_error"], cumulative_relative_error=cumulative,
        train_event_relative_error=dict(min=min(relative), max=max(relative)) if relative else None,
        budget_anomalies=sum(e["event"] == "budget_anomaly" for e in events),
        train_events=len(train), validation_events=len(val), core_events=len(core),
        final_bpb=dict(train=bpb["train"]["bpb"], val=bpb["val"]["bpb"], tokens_per_split=bpb["val"]["input_tokens"]),
        periodic_final_val_bpb=periodic.get(P.steps),
        core={e["step"]: e["core_metric"] for e in core}, core_final=core[-1]["core_metric"] if core else None,
        last_loss=train[-1]["loss"] if train else None,
        grad_norm_max_after_500=max((e["grad_norm"] for e in train if e["step"] > 500), default=None),
        evaluation_forward_flops=status.get("evaluation_forward_flops"),
        segments=segments,
        fingerprints=dict(model_hash=first["model_hash"], protocol_hash=first["protocol_hash"],
                          data_manifest_sha256=first["data_manifest_sha256"], tokenizer_sha256=first["tokenizer_sha256"],
                          reference_source_hash=first["software"]["reference_source_hash"],
                          science_hash=first["software"]["science_hash"]),
        parameters=first["parameters"],
        software={k: first["software"].get(k) for k in ("torch", "cuda", "nccl") if first["software"].get(k) is not None},
        periodic_val_bpb={s: periodic[s] for s in (0, 8000, 16000, 24000, 32000, 40000, P.steps) if s in periodic},
        periodic_val_bpb_minimum=dict(val_bpb=best["val_bpb"], step=best["step"]) if best else None,
    )
    return report, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--method", required=True, help="registered method id; also the name of the audit file")
    parser.add_argument("--output", type=Path, help="default: evidence/audits/<method>.json")
    args = parser.parse_args(argv)
    report, problems = audit(args.run_dir, args.method)
    out = args.output or REPO/"evidence/audits"/f"{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1)+"\n")
    print(f"wrote {out}: {'all checks passed' if not problems else f'{len(problems)} problem(s)'}")
    for problem in problems:
        print("  "+problem)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
