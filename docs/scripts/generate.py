"""Regenerate the documentation pages and README tables derived from results/ and configs/.

    python docs/scripts/generate.py           # rewrite the generated files
    python docs/scripts/generate.py --check   # exit 1 when a generated file is out of date (CI)

Writes docs/src/content/docs/methods/zoo.md and methods/configurations.md, and the tables between the
`<!-- generated:leaderboard -->` markers in results/leaderboard.md, README.md and README.zh-CN.md.
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nexttransformer.export_run import summary  # noqa: E402

DOCS = ROOT/"docs/src/content/docs"
REPO_FILE = "https://github.com/Interactive-Active-AI/NextTransformer/blob/main/"
SITE = "https://interactive-active-ai.github.io/NextTransformer/"
FAMILIES = ["baseline", "looped", "sparse", "linear", "residual", "hybrid", "others"]
TITLE = {"baseline": "Baseline", "looped": "Looped", "sparse": "Sparse", "linear": "Linear", "residual": "Residual",
         "hybrid": "Hybrid", "others": "Others"}
SNAPSHOT = 8
START, END = "<!-- generated:leaderboard -->", "<!-- /generated:leaderboard -->"


def load(path): return json.loads(path.read_text())


def zoo():
    cards = [load(p) for p in (ROOT/"results/methods").glob("*.json")]
    published = {p.stem for p in (ROOT/"results/runs").glob("*.json")}
    configs = {p.stem for f in FAMILIES for p in (ROOT/"configs"/f).glob("*.json")}
    uncarded = sorted(configs - {c["id"] for c in cards} - {"dense"})
    rows = []
    for c in sorted(cards, key=lambda c: (FAMILIES.index(c["family"]), c.get("order", 99), c["id"])):
        paper = f"[{c['paperTitle']}]({c['paperUrl']})" if c.get("paperUrl") else c.get("paperTitle", "")
        status = "published" if c["id"] in published else "coming soon" if c.get("status") == "pending" else "registered"
        rows.append(f"| {TITLE[c['family']]} | {c['name']} (`{c['id']}`) | {paper} | {status} |")
    note = (f"Registered configurations without a model card ({', '.join(f'`{m}`' for m in uncarded)}) are superseded "
            "variants kept on record. " if uncarded else "")
    return f"""---
title: Model zoo
description: Every method in NextTransformer, by family.
---

NextTransformer has {len(cards)} model cards in seven families. Every method runs in the same shell and is calibrated to
the same training compute. Open a method in the [observatory]({SITE}#atlas) for its architecture diagram, curves and
model card; "registered" methods have a card and a configuration but no published run.

| Family | Method | Paper | Result |
|---|---|---|---|
{chr(10).join(rows)}

{note}Add your own as a [plugin](../../guides/plugins/).
"""


def configurations():
    rows = []
    for family in FAMILIES:
        for path in sorted((ROOT/"configs"/family).glob("*.json")):
            config = load(path)
            calibration = config["calibration"]
            proof = load((path.parent/calibration["cost_record"]).resolve())
            stages = []
            for stage in proof["stages"]:
                if not stages or stages[-1] != stage["stage"]: stages.append(stage["stage"])
            method = "`dense` (SSSL)" if path.stem == "dense" else f"`{path.stem}`"
            rows.append(f"| {family} | {method} | {proof['physical_layers']} | {proof['selected_ffn']:,} | "
                        f"{proof['parameters']:,} | {calibration['flops_per_token']/1e9:.6f} | "
                        f"{100*calibration['relative_error']:+.3f}% | {', '.join(stages)} |")
    return f"""---
title: Configurations
description: The admitted structure, parameters and audited compute of every registered configuration.
---

Every registered method has one admitted configuration, `configs/<family>/<method>.json`, and a cost proof in the current
proof set. The table is generated from those files: physical blocks, FFN width, independent parameters, audited training
GFLOPs per input token and the deviation from the SSSL anchor (4.67239832544785). Parameters are outcomes, not controls.

| Family | Method | Blocks | FFN | Parameters | GFLOPs / token | vs SSSL | Stages |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Blocks count physical layers; looped methods reuse them across loops, so their unfolded depth is larger (see the
[source notes](../sources/)). Multi-stage methods weight each stage by its updates: DSA, QSA, Seer and the QSA hybrid run
512 dense burn-in updates before sparse training; Ouro trains its exit gate alone for the final 1,024 updates; Huginn
samples a depth per update from a fixed seed, and its proof enumerates all 46,875 samples.

## Reading a configuration

A configuration holds the model fields (`model`), the protocol it is admitted under (`protocol`, with its fingerprint)
and the calibration record (`calibration`: the audited FLOPs per token and their deviation, the method's source
fingerprint, the PyTorch build of the audit and the path of the cost proof). Configurations are written by
`python -m nexttransformer.protocol.calibrate` and checked by `admit_config`; never edit them by hand.

The historical native track (SSSL, MoR, LoopFormer, Circle Loop) keeps its own configurations under `configs/native/`,
with the historical matrix-FLOPs accounting.
"""


def snapshot(table, chinese):
    """The README excerpt: the Dense Baseline and the leading runs by CORE, without links or the compute column."""
    rows = [line for line in table.splitlines()[2:]]
    marks, out = set(), []
    for line in rows[:SNAPSHOT]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cells[0])
        mark = re.search(r"[¹²]$", cells[-1])
        if mark: marks.add(mark.group(0)); name = name.replace("**", "") + mark.group(0)
        if cells[0].startswith("**"): name = f"**{name.strip('*')}**"
        deltas = [c.replace("-", "−") for c in (cells[3], cells[5])]
        out.append(f"| {name} | {cells[1]} | {cells[2]} | {deltas[0]} | {cells[4]} | {deltas[1]} |")
    if chinese:
        head = "| 模型 | 模型族 | 验证 BPB ↓ | Δ BPB | CORE ↑ | Δ CORE |"
        notes = {"¹": "¹ 使用历史算力核算", "²": "² 使用冻结的重放核算"}
        foot = (f"单种子运行；{'；'.join(notes[m] for m in sorted(marks))}。" if marks else "单种子运行。") + \
               f"全部 {len(rows)} 个运行及其说明见[排行榜]({SITE}docs/zh-cn/results/leaderboard/)。"
    else:
        head = "| Model | Family | Val BPB ↓ | Δ BPB | CORE ↑ | Δ CORE |"
        notes = {"¹": "¹ historical compute accounting", "²": "² frozen replay accounting"}
        foot = (f"Single-seed runs; {'; '.join(notes[m] for m in sorted(marks))}." if marks else "Single-seed runs.") + \
               f" All {len(rows)} runs, with notes on each, are on the [leaderboard]({SITE}docs/results/leaderboard/)."
    return "\n".join([head, "|---|---|---:|---:|---:|---:|", *out, "", foot])


def splice(text, block):
    before, rest = text.split(START, 1)
    return before + START + "\n" + block + "\n" + END + rest.split(END, 1)[1]


def outputs():
    table = summary()
    docs_table = table.replace("](results/runs/", f"]({REPO_FILE}results/runs/")
    leaderboard, readme, readme_zh = DOCS/"results/leaderboard.md", ROOT/"README.md", ROOT/"README.zh-CN.md"
    return {DOCS/"methods/zoo.md": zoo(), DOCS/"methods/configurations.md": configurations(),
            leaderboard: splice(leaderboard.read_text(), docs_table),
            readme: splice(readme.read_text(), snapshot(table, False)),
            readme_zh: splice(readme_zh.read_text(), snapshot(table, True))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail when a generated file is out of date")
    args = ap.parse_args()
    stale = [path for path, text in outputs().items() if path.read_text() != text]
    if args.check:
        for path in stale: print(f"out of date: {path.relative_to(ROOT)} (run python docs/scripts/generate.py)")
        sys.exit(1 if stale else 0)
    for path, text in outputs().items(): path.write_text(text)
    print(f"updated {len(stale)} file(s)" if stale else "all generated files are up to date")


if __name__ == "__main__":
    main()
