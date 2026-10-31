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
from nexttransformer.export_run import CARD_ALIASES, summary  # noqa: E402

DOCS = ROOT/"docs/src/content/docs"
REPO_FILE = "https://github.com/Interactive-Active-AI/NextTransformer/blob/main/"
SITE = "https://interactive-active-ai.github.io/NextTransformer/"
FAMILIES = ["baseline", "looped", "sparse", "linear", "residual", "hybrid", "others"]
TITLE = {"baseline": "Baseline", "looped": "Looped", "sparse": "Sparse", "linear": "Linear", "residual": "Residual",
         "hybrid": "Hybrid", "others": "Others"}
ASSETS = ROOT/".github/assets/readme"
START, END = "<!-- generated:leaderboard -->", "<!-- /generated:leaderboard -->"


def load(path): return json.loads(path.read_text())


def zoo(chinese=False):
    cards = [load(p) for p in (ROOT/"results/methods").glob("*.json")]
    published = {p.stem for p in (ROOT/"results/runs").glob("*.json")}
    rows = []
    for c in sorted(cards, key=lambda c: (FAMILIES.index(c["family"]), c.get("order", 99), c["id"])):
        paper = f"[{c['paperTitle']}]({c['paperUrl']})" if c.get("paperUrl") else c.get("paperTitle", "")
        status = "published" if c["id"] in published else "coming soon" if c.get("status") == "pending" else "registered"
        family = FAMILY_ZH[c["family"]] if chinese else TITLE[c["family"]]
        if chinese:
            status = {"published": "已发布", "coming soon": "待发布", "registered": "已注册"}[status]
        rows.append(f"| {family} | {c['name']} (`{c['id']}`) | {paper} | {status} |")
    if chinese:
        return f"""---
title: 模型库
description: 按模型族浏览已注册的架构、论文和结果状态。
---

当前共有 {len(cards)} 张模型卡。各方法在同一骨干上校准至相同训练算力。在[观测站]({SITE}#atlas)选择模型可查看结构图、曲线和模型卡。“已注册”表示已有配置与模型卡，尚无公开运行结果。

| 模型族 | 方法 | 论文 | 结果 |
|---|---|---|---|
{chr(10).join(rows)}

训练配置见[模型配置](../configurations/)。新增架构请参见[编写插件](../../guides/plugins/)。
"""
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

See the [configuration reference](../configurations/) for training configurations. Add your own as a [plugin](../../guides/plugins/).
"""


def configurations(chinese=False):
    # The sources guide also documents the output-normalization reference used by ELU-Norm.
    documented = {p.stem for p in (ROOT/"results/methods").glob("*.json")} | {"elu_norm"}
    rows = []
    for family in FAMILIES:
        for path in sorted((ROOT/"configs"/family).glob("*.json")):
            if CARD_ALIASES.get(path.stem, path.stem) not in documented:
                continue
            config = load(path)
            calibration = config["calibration"]
            proof = load((path.parent/calibration["cost_record"]).resolve())
            stages = []
            for stage in proof["stages"]:
                if not stages or stages[-1] != stage["stage"]: stages.append(stage["stage"])
            method = "`dense` (SSSL)" if path.stem == "dense" else f"`{path.stem}`"
            label = FAMILY_ZH[family] if chinese else family
            rows.append(f"| {label} | {method} | {proof['physical_layers']} | {proof['selected_ffn']:,} | "
                        f"{proof['parameters']:,} | {calibration['flops_per_token']/1e9:.6f} | "
                        f"{100*calibration['relative_error']:+.3f}% | {', '.join(stages)} |")
    if chinese:
        return f"""---
title: 模型配置
description: 公开模型及其文档所列参考配置的结构、参数量与审计计算量。
---

表中方法的准入配置位于 `configs/<family>/<method>.json`，并指向当前凭证集中的对应文件。下表直接从配置与凭证生成。每个输入 token 的训练 GFLOPs 相对 SSSL 锚点 4.67239832544785 计算偏差；参数量是校准的结果。

| 模型族 | 方法 | 物理块数 | FFN 宽度 | 参数量 | GFLOPs/token | 相对 SSSL | 训练阶段 ID |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

循环方法跨轮复用物理块，展开后的深度会大于表内块数，详见[方法来源与实现](../sources/)。各阶段按更新次数加权。DSA、QSA、Seer 和 QSA 混合架构先做 512 步稠密预热，Ouro 在最后 1,024 步仅训练退出门。Huginn 按固定种子逐步采样深度，凭证枚举了全部 46,875 次采样。

## 阅读配置

`model` 定义模型结构，`protocol` 给出准入协议及其指纹。`calibration` 保存每 token 的审计算力及偏差、方法的源码指纹、用于审计的 PyTorch 构建和凭证路径。配置由 `python -m nexttransformer.protocol.calibrate` 写入，再由 `admit_config` 检查；请通过校准流程更新。

SSSL、MoR、LoopFormer 和 Circle Loop 的历史原生配置位于 `configs/native/`，采用历史矩阵 FLOPs 核算。
"""
    return f"""---
title: Configurations
description: Admitted shapes, parameters and audited compute for the documented models and their reference configurations.
---

Each method in the table has an admitted configuration, `configs/<family>/<method>.json`, and a cost proof in the current
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
METHOD_LABELS = load(ROOT / "website/data/method-labels.json")


def finished():
    """Finished runs with their card, final validation BPB and CORE (x100)."""
    cards = {c["id"]: c for c in (load(p) for p in (ROOT/"results/methods").glob("*.json"))}
    rows = []
    for path in sorted((ROOT/"results/runs").glob("*.json")):
        r = load(path)
        if r["status"] != "finished": continue
        card = cards[CARD_ALIASES.get(r["methodId"], r["methodId"])]
        rows.append({"method": r["methodId"], "id": r["id"], "name": METHOD_LABELS.get(card["id"], card["name"]), "family": card["family"], "bpb": r["val"][-1]["value"],
                     "core": 100*r["core"][-1]["value"], "base": bool(r.get("baselineAcceptance"))})
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
    height = top + step*len(rows) + 40
    font = "'Avenir Next','Segoe UI',-apple-system,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif"
    title = "第 46,875 次更新时的 CORE" if zh else "CORE at update 46,875"
    subtitle = (f"{len(rows)} 个已完成运行 · 单种子 · CORE 越高越好" if zh
                else f"{len(rows)} completed runs · single seed · higher CORE is better")
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
        name = html.escape(r["name"])
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
    out.append("</svg>")
    return "\n".join(out) + "\n"


def snapshot(chinese):
    """The README block: the results chart and its caption."""
    n = len(finished())
    if chinese:
        return (f'<a href="{SITE}#leaderboard"><img src=".github/assets/readme/results-zh.svg" '
                f'alt="全部 {n} 个运行在第 46,875 次更新时的 CORE，与 Dense Baseline 对比" width="100%"></a>\n\n'
                f"[完整排行榜]({SITE}docs/zh-cn/results/leaderboard/) · [训练回放]({SITE}#leaderboard)")
    return (f'<a href="{SITE}#leaderboard"><img src=".github/assets/readme/results-en.svg" '
            f'alt="CORE at update 46,875 for all {n} runs, against the Dense Baseline" width="100%"></a>\n\n'
            f"[Full leaderboard]({SITE}docs/results/leaderboard/) · [Training replay]({SITE}#leaderboard)")


def splice(text, block, start=START, end=END):
    before, rest = text.split(start, 1)
    return before + start + "\n" + block + "\n" + end + rest.split(end, 1)[1]


def readme_sections(path, chinese):
    cards = sorted((load(p) for p in (ROOT / "results/methods").glob("*.json")),
                   key=lambda card: (card.get("order", 99), card["id"]))
    table = ["| 模型族 | 模型 |" if chinese else "| Family | Models |", "|---|---|"]
    for family in ("baseline", "residual", "looped", "sparse", "linear", "hybrid", "others"):
        label = TITLE[family]
        alt = FAMILY_ZH[family] if chinese else label
        badge = f'<img src="https://img.shields.io/badge/{label}-{COLOR[family][1:]}?style=flat-square" alt="{alt}">'
        names = [METHOD_LABELS.get(card["id"], card["name"]) + (r"\*" if card.get("status") == "pending" else "")
                 for card in cards if card["family"] == family]
        table.append(f"| {badge} | {' · '.join(names)} |")
    text = splice(path.read_text(), snapshot(chinese))
    return splice(text, "\n".join(table), "<!-- generated:model-zoo -->", "<!-- /generated:model-zoo -->")


def chinese_leaderboard(table):
    lines = table.splitlines()
    lines[0] = "| 模型 | 模型族 | 验证 BPB ↓ | Δ BPB | CORE ↑ | Δ CORE | 训练 GFLOPs/token |"
    for i in range(2, len(lines)):
        cells = lines[i].split("|")
        cells[2] = " " + FAMILY_ZH[cells[2].strip()] + " "
        lines[i] = "|".join(cells)
    return "\n".join(lines)


def outputs():
    table = summary()
    docs_table = table.replace("](results/runs/", f"]({REPO_FILE}results/runs/")
    leaderboard, readme, readme_zh = DOCS/"results/leaderboard.md", ROOT/"README.md", ROOT/"README.zh-CN.md"
    zh = DOCS / "zh-cn"
    zh_leaderboard = zh / "results/leaderboard.md"
    return {DOCS/"methods/zoo.md": zoo(), DOCS/"methods/configurations.md": configurations(),
            zh/"methods/zoo.md": zoo(True), zh/"methods/configurations.md": configurations(True),
            zh_leaderboard: splice(zh_leaderboard.read_text(), chinese_leaderboard(docs_table)),
            leaderboard: splice(leaderboard.read_text(), docs_table),
            readme: readme_sections(readme, False),
            readme_zh: readme_sections(readme_zh, True),
            ASSETS/"results-en.svg": chart("en"), ASSETS/"results-zh.svg": chart("zh")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail when a generated file is out of date")
    args = ap.parse_args()
    files = outputs()
    stale = [path for path, text in files.items() if not path.exists() or path.read_text() != text]
    if args.check:
        pages = {p.relative_to(DOCS).with_suffix("") for p in DOCS.rglob("*") if p.suffix in {".md", ".mdx"}}
        missing = [p for p in sorted(pages) if p.parts[0] != "zh-cn" and Path("zh-cn") / p not in pages]
        if missing:
            sys.exit("Missing Chinese documentation: " + ", ".join(map(str, missing)))
        for path in stale: print(f"out of date: {path.relative_to(ROOT)} (run python docs/scripts/generate.py)")
        sys.exit(1 if stale else 0)
    for path in stale:
        path.write_text(files[path])
    print(f"updated {len(stale)} file(s)" if stale else "all generated files are up to date")


if __name__ == "__main__":
    main()
