"""Regenerate the documentation pages and README charts derived from results/ and configs/.

    python docs/scripts/generate.py           # rewrite the generated files
    python docs/scripts/generate.py --check   # exit 1 when a generated file is out of date (CI)

Writes docs/src/content/docs/methods/zoo.md and methods/configurations.md, the results charts in
.github/assets/readme/, and the blocks between the `<!-- generated:leaderboard -->` markers in results/leaderboard.md,
README.md and README.zh-CN.md.
"""
import argparse, html, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nexttransformer.export_run import CARD_ALIASES, ledger_mark, summary  # noqa: E402

DOCS = ROOT/"docs/src/content/docs"
REPO_FILE = "https://github.com/Interactive-Active-AI/NextTransformer/blob/main/"
SITE = "https://interactive-active-ai.github.io/NextTransformer/"
FAMILIES = ["baseline", "looped", "sparse", "linear", "residual", "hybrid", "others"]
TITLE = {"baseline": "Baseline", "looped": "Looped", "sparse": "Sparse", "linear": "Linear", "residual": "Residual",
         "hybrid": "Hybrid", "others": "Others"}
ASSETS = ROOT/".github/assets/readme"
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


COLOR = {"baseline": "#7A8288", "looped": "#8B5CF6", "sparse": "#E5607E", "linear": "#3F7FD9", "residual": "#33456E",
         "hybrid": "#F07856", "others": "#D99A2B"}
FAMILY_ZH = {"baseline": "基线", "looped": "循环", "sparse": "稀疏", "linear": "线性", "residual": "残差", "hybrid": "混合", "others": "其他"}
SHORT = {"Universal Transformer · ACT": "UT · ACT", "RetNet · Retention": "RetNet", "Ouro · LoopLM": "Ouro",
         "Huginn · Recurrent-depth": "Huginn", "ELU+1 · q/k-normalized": "ELU+1 · q/k-norm", "LoopFormer-Fixed2": "LoopFormer"}
NOTES = {"en": {"¹": "¹ historical compute accounting", "²": "² frozen replay accounting"},
         "zh": {"¹": "¹ 历史算力核算", "²": "² 冻结的重放核算"}}


def finished():
    """Finished runs with their card, final validation BPB and CORE (x100), and compute-accounting mark."""
    cards = {c["id"]: c for c in (load(p) for p in (ROOT/"results/methods").glob("*.json"))}
    rows = []
    for path in sorted((ROOT/"results/runs").glob("*.json")):
        r = load(path)
        if r["status"] != "finished": continue
        card = cards[CARD_ALIASES.get(r["methodId"], r["methodId"])]
        rows.append({"method": r["methodId"], "id": r["id"], "name": card["name"], "family": card["family"], "bpb": r["val"][-1]["value"],
                     "core": 100*r["core"][-1]["value"], "base": bool(r.get("baselineAcceptance")),
                     "mark": ledger_mark((r.get("trainingCompute") or {}).get("accounting", ""))})
    rows.sort(key=lambda r: (-r["core"], r["method"], r["id"]))
    previous_core = None
    for position, row in enumerate(rows, 1):
        # The observatory ranks unrounded CORE and gives equal scores the same rank.
        if previous_core is None or abs(row["core"] - previous_core) > 1e-10:
            rank = position
        row["rank"] = rank
        previous_core = row["core"]
    return rows


def chart(lang):
    """SVG bar chart of CORE at the final update for every finished run, with the Dense Baseline marked."""
    zh, rows = lang == "zh", finished()
    base = next(r for r in rows if r["base"])
    width, top, step, left, right = 980, 172, 24, 300, 760
    lo, hi = 5*math.floor(min(r["core"] for r in rows)/5 - 1), 5*math.ceil(max(r["core"] for r in rows)/5)
    x = lambda v: left + (v-lo)/(hi-lo)*(right-left)
    height = top + step*len(rows) + 64
    font = "'Avenir Next','Segoe UI',-apple-system,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif"
    title = "第 46,875 次更新时的 CORE" if zh else "CORE at update 46,875"
    subtitle = (f"{len(rows)} 个运行 · 相同的数据、token 与训练算力 · 单种子 · 越高越好" if zh
                else f"{len(rows)} runs · same data, tokens and training compute · single seed · higher is better")
    heads = ("CORE ×100", "差值", "验证 BPB") if zh else ("CORE ×100", "vs Dense", "Val BPB")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="{font}">',
           f'<rect x="1" y="1" width="{width-2}" height="{height-2}" rx="26" fill="#FFFFFF" stroke="#E6E9EE" stroke-width="1.5"/>',
           f'<text x="36" y="56" font-size="25" font-weight="700" fill="#262C33">{title}</text>',
           f'<text x="36" y="84" font-size="14.5" fill="#6B7480">{subtitle}</text>']
    lx = 36
    for family in ("baseline", "looped", "sparse", "linear", "residual", "hybrid", "others"):
        label = FAMILY_ZH[family] if zh else family.title()
        out.append(f'<rect x="{lx}" y="104" width="11" height="11" rx="3" fill="{COLOR[family]}"/>'
                   f'<text x="{lx+17}" y="114" font-size="13" font-weight="600" fill="#5A646E">{label}</text>')
        lx += 17 + 8*len(label) + 26 if not zh else 17 + 14*len(label) + 26
    for text, cx in zip(heads, (right+52, right+132, width-36)):
        out.append(f'<text x="{cx}" y="{top-12}" font-size="12.5" font-weight="700" fill="#8A939C" text-anchor="end">{text}</text>')
    for i, r in enumerate(rows):
        y = top + i*step
        if r["base"]: out.append(f'<rect x="20" y="{y}" width="{width-40}" height="{step}" rx="8" fill="#FBF2F3"/>')
        elif i % 2: out.append(f'<rect x="20" y="{y}" width="{width-40}" height="{step}" rx="8" fill="#F7F8FA"/>')
        weight = "700" if r["base"] else "500"
        name = html.escape(SHORT.get(r["name"], r["name"].split(" · ")[-1] if " · " in r["name"] else r["name"])) + r["mark"]
        delta = r["core"] - base["core"]
        dtext = "—" if r["base"] else ("±0.00" if abs(delta) < 0.005 else f"{delta:+.2f}".replace("-", "−"))
        dcolor = "#8A939C" if r["base"] or delta < 0.005 else "#3F7FD9"
        out.append(f'<text x="44" y="{y+16.5}" font-size="12" font-weight="600" fill="#A2AAB3" text-anchor="end">{r["rank"]}</text>'
                   f'<circle cx="58" cy="{y+12}" r="4.5" fill="{COLOR[r["family"]]}"/>'
                   f'<text x="70" y="{y+16.5}" font-size="14" font-weight="{weight}" fill="#2F363D">{name}</text>'
                   f'<rect x="{left}" y="{y+6}" width="{x(r["core"])-left:.1f}" height="12" rx="6" fill="{COLOR[r["family"]]}" opacity="{1 if r["base"] else .9}"/>'
                   f'<text x="{right+52}" y="{y+16.5}" font-size="14" font-weight="{weight}" fill="#2F363D" text-anchor="end">{r["core"]:.2f}</text>'
                   f'<text x="{right+132}" y="{y+16.5}" font-size="13.5" font-weight="700" fill="{dcolor}" text-anchor="end">{dtext}</text>'
                   f'<text x="{width-36}" y="{y+16.5}" font-size="13.5" fill="#5A646E" text-anchor="end">{r["bpb"]:.4f}</text>')
    bx, bottom = x(base["core"]), top + step*len(rows)
    out.append(f'<line x1="{bx:.1f}" y1="{top-28}" x2="{bx:.1f}" y2="{bottom+4}" stroke="#CA6F6F" stroke-width="1.5" stroke-dasharray="4 4"/>'
               f'<text x="{bx:.1f}" y="{top-34}" font-size="12.5" font-weight="700" fill="#CA6F6F" text-anchor="middle">Dense {base["core"]:.2f}</text>')
    for v in range(lo, hi+1, 5):
        out.append(f'<text x="{x(v):.1f}" y="{bottom+22}" font-size="12" fill="#A2AAB3" text-anchor="middle">{v}</text>')
    marks = [m for m in ("¹", "²") if any(r["mark"] == m for r in rows)]
    foot = ("验证 BPB 与 CORE 取最后一次更新的测量值。" if zh else "Validation BPB and CORE measured at the final update. ") + \
           ("；".join(NOTES[lang][m] for m in marks) if zh else "; ".join(NOTES[lang][m] for m in marks))
    out.append(f'<text x="36" y="{bottom+48}" font-size="12.5" fill="#8A939C">{html.escape(foot)}</text></svg>')
    return "\n".join(out) + "\n"


def snapshot(chinese):
    """The README block: the results chart and its caption."""
    n = len(finished())
    if chinese:
        return (f'<a href="{SITE}#leaderboard"><img src=".github/assets/readme/results-zh.svg" '
                f'alt="全部 {n} 个运行在第 46,875 次更新时的 CORE，与 Dense Baseline 对比" width="100%"></a>\n\n'
                f"每个运行的数值和说明见[排行榜]({SITE}docs/zh-cn/results/leaderboard/)；交互式回放见[观测站]({SITE}#leaderboard)。")
    return (f'<a href="{SITE}#leaderboard"><img src=".github/assets/readme/results-en.svg" '
            f'alt="CORE at update 46,875 for all {n} runs, against the Dense Baseline" width="100%"></a>\n\n'
            f"Numbers and notes for every run are on the [leaderboard]({SITE}docs/results/leaderboard/); replay training in the "
            f"[observatory]({SITE}#leaderboard).")


def splice(text, block):
    before, rest = text.split(START, 1)
    return before + START + "\n" + block + "\n" + END + rest.split(END, 1)[1]


def outputs():
    table = summary()
    docs_table = table.replace("](results/runs/", f"]({REPO_FILE}results/runs/")
    leaderboard, readme, readme_zh = DOCS/"results/leaderboard.md", ROOT/"README.md", ROOT/"README.zh-CN.md"
    return {DOCS/"methods/zoo.md": zoo(), DOCS/"methods/configurations.md": configurations(),
            leaderboard: splice(leaderboard.read_text(), docs_table),
            readme: splice(readme.read_text(), snapshot(False)),
            readme_zh: splice(readme_zh.read_text(), snapshot(True)),
            ASSETS/"results-en.svg": chart("en"), ASSETS/"results-zh.svg": chart("zh")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail when a generated file is out of date")
    args = ap.parse_args()
    files = outputs()
    stale = [path for path, text in files.items() if not path.exists() or path.read_text() != text]
    if args.check:
        for path in stale: print(f"out of date: {path.relative_to(ROOT)} (run python docs/scripts/generate.py)")
        sys.exit(1 if stale else 0)
    for path in stale:
        path.write_text(files[path])
    print(f"updated {len(stale)} file(s)" if stale else "all generated files are up to date")


if __name__ == "__main__":
    main()
