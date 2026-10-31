<div align="center">

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/banner.jpg" alt="NextTransformer: toward the next generation of Transformers" width="100%"></a>

<h3>An open framework and a fair benchmark for next-generation Transformer architectures</h3>

<p>
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/"><img alt="Documentation" src="https://img.shields.io/badge/docs-read-3F7FD9?style=flat-square"></a>
<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-live-8B5CF6?style=flat-square"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-F2A93B?style=flat-square"></a>
<img alt="Python 3.10-3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="PyTorch 2.9.1" src="https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
<br>
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/community/#wechat"><img alt="WeChat group" src="https://img.shields.io/badge/WeChat-group-E5607E?style=flat-square&logo=wechat&logoColor=white"></a>
<a href="https://github.com/Interactive-Active-AI/NextTransformer/discussions"><img alt="GitHub Discussions" src="https://img.shields.io/badge/discussions-ask-7A5AF0?style=flat-square&logo=github&logoColor=white"></a>
</p>

<b>
<a href="https://interactive-active-ai.github.io/NextTransformer/">Observatory</a> ·
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/">Docs</a> ·
<a href="#quick-start">Quick start</a> ·
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/">Protocol</a> ·
<a href="#auto-research-arena">Arena</a> ·
<a href="#community">Community</a> ·
<a href="README.zh-CN.md">简体中文</a>
</b>

</div>

<br>

**When the training budget is fixed, how should a language model spend its computation?** NextTransformer trains
recurrent depth, sparse and linear attention, residual-stream variants and hybrids on the same data, for the same
49.152B tokens, at the same theoretical training FLOPs per token, and evaluates them at the same checkpoints. Every
result is an audited, file-backed record you can inspect in the [observatory](https://interactive-active-ai.github.io/NextTransformer/).

## News

- **2026-11-01** · NextTransformer **v0.1.0** is open source: 29 finished runs, 39 model cards, the full protocol, the
  documentation and the observatory.
- **By 2026-11-14** · **Arena v2**: full support for autonomous agent research, with a research gym and metered compute.
- **November 2026** · Training recipes for **400B tokens and larger models**.

## Highlights

<table>
<tr>
<td width="50%" valign="top">

**Frontier architectures, one framework**<br>
39 models across full attention, residual connections, linear and sparse attention, looped computation and hybrids,
in one codebase built on nanochat. A new method is [one plugin folder](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/).

</td>
<td width="50%" valign="top">

**A fair, explicit protocol**<br>
Same data, tokens, optimizer, schedule and evaluation for every model. Training compute is matched to ±1% by an
operator-level ledger and audited over the whole run. [Read the protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/).

</td>
</tr>
<tr>
<td width="50%" valign="top">

**1B-scale pretraining on one machine**<br>
Train a ~1.4B-parameter model on 49B tokens on a single 8-GPU node in 2–4 days, or spread it over 16 or 32 GPUs.
Recipes for 400B tokens and larger models arrive in November.

</td>
<td width="50%" valign="top">

**Agent-native research**<br>
Hand the [arena task](https://interactive-active-ai.github.io/NextTransformer/docs/arena/task/) to your agent and let it
research the next architecture, and improve how it researches. The full version ships by November 14, 2026.

</td>
</tr>
</table>

## The observatory

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/observatory.jpg" alt="The NextTransformer observatory: learning curves, leaderboard, model atlas and arena" width="100%"></a>

Learning curves of every run, a replayable leaderboard, all 22 CORE tasks, architecture diagrams and model cards. The
leading runs at update 46,875, against the full-attention Dense Baseline:

<!-- generated:leaderboard -->
| Model | Family | Val BPB ↓ | Δ BPB | CORE ↑ | Δ CORE |
|---|---|---:|---:|---:|---:|
| **Dense Baseline** | baseline | 0.6672 | — | 33.10 | — |
| Block Attention Residuals · AttnRes | residual | 0.6630 | −0.0042 | 34.00 | +0.90 |
| SSSL¹ | others | 0.6671 | ±0.0000 | 33.78 | +0.69 |
| Native Sparse Attention · NSA | sparse | 0.6678 | +0.0006 | 33.78 | +0.69 |
| Hybrid GDN + SWA | hybrid | 0.6656 | −0.0015 | 33.66 | +0.57 |
| Hybrid GDN + MHA | hybrid | 0.6647 | −0.0025 | 33.37 | +0.28 |
| Grouped-Query Attention · GQA | others | 0.6741 | +0.0069 | 32.62 | −0.47 |
| Multi-head Latent Attention · MLA + QK-norm | others | 0.6759 | +0.0087 | 32.53 | −0.56 |

Single-seed runs; ¹ historical compute accounting. All 29 runs, with notes on each, are on the [leaderboard](https://interactive-active-ai.github.io/NextTransformer/docs/results/leaderboard/).
<!-- /generated:leaderboard -->

## Quick start

**1. Install** (Linux, NVIDIA GPUs, [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Interactive-Active-AI/NextTransformer.git && cd NextTransformer
uv sync --frozen --extra gpu --extra upstream && source .venv/bin/activate
uv pip install --no-deps --require-hashes -r requirements/linear.txt   # linear-attention kernels
```

**2. Prepare the data** (ClimbMix at a pinned revision, about 600 GB, plus the tokenizer and CORE bundle):

```bash
cp site.native.example.json site.local.json        # set data_dir and resources
hf download karpathy/climbmix-400b-shuffle --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 --local-dir /path/to/climbmix-400b-shuffle
source launch/env.sh --site-config site.local.json
python -m nexttransformer.prepare --site-config site.local.json
```

**3. Train on one node with 8 GPUs:**

```bash
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full --micro-batch 8 --formal --resume auto
```

**4. Or on several nodes**, running the same command on each node, or with SLURM:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 launch/train.sh --config configs/linear/gdn.json \
  --site-config site.local.json --output runs/gdn --micro-batch 16 --formal --resume auto   # NODE_RANK=1 on node 1

sbatch --nodes=4 launch/slurm.sbatch --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto --max-wall-hours 23.5
```

Evaluation runs on schedule during training. The [quick start guide](https://interactive-active-ai.github.io/NextTransformer/docs/getting-started/quick-start/)
covers smoke tests, evaluation, auditing and exporting a result.

## Model zoo

<details>
<summary><b>39 models in seven families</b> (click to expand)</summary>

| Family | Models |
|---|---|
| Baseline | Dense (full causal attention) |
| Residual | Standard Residual · Block Attention Residuals · Hyper-Connections\* · mHC\* |
| Looped | Universal Transformer · ACT · Fixed Recursive · Huginn · Ouro · DeepLoop · Mixture-of-Recursions · LoopFormer · Circle Loop |
| Sparse | DeepSeek Sparse Attention · Qwen Sparse Attention · Native Sparse Attention · SeerAttention-style · MoBA |
| Linear | DeltaNet · Gated DeltaNet · Gated Linear Attention · Kimi Delta Attention · RetNet · Linear Transformer (ELU+1)† · ELU+1 with q/k-norm · KDA without VE\* |
| Hybrid | GDN + MHA · GDN + SWA · DeltaNet + MHA\* · KDA + MLA-NoPE\* · GDN + QSA\* · MoBA + MHA\* |
| Others | SSSL · MQA · GQA · MLA + QK-norm · MLA-NoPE + QK-norm · MLA-NoPE\* · MHA without VE\* |

\* Results coming soon. † Did not train stably under the protocol; the q/k-normalized variant replaces it. Papers, adaptations and configurations are in the [model zoo](https://interactive-active-ai.github.io/NextTransformer/docs/methods/zoo/).

</details>

## The protocol at a glance

| | |
|---|---|
| **Data** | ClimbMix at a pinned revision · one 32,768-token tokenizer · 2,048-token context · one global sample order |
| **Budget** | 49.152B input tokens · 46,875 updates of 1,048,576 tokens · seed 42 |
| **Compute** | 4.672 GFLOPs per input token (forward, backward, auxiliary and optimizer work); ±1% at admission, ±2% over the run |
| **Evaluation** | Validation BPB at 189 points · full CORE-22 at six checkpoints · an independent final BPB |
| **Baseline** | Dense: full causal attention in every layer, trained under the same budget |

Parameters, memory and speed are outcomes, not controls. The [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/)
explains every rule and the [compute accounting](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/compute/)
what is counted.

## Add your method

1. **Write a plugin**: a folder `nexttransformer/methods/<name>/` with a `method.json` and a few hooks. No shared file changes.
2. **Calibrate** it to the shared budget: `python -m nexttransformer.protocol.calibrate --methods <name> ...`
3. **Open a pull request.** Once admitted, train it with `--formal` and submit the result in a second pull request.

Start with the [plugin guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Auto-research arena

> [!NOTE]
> **The arena is in active development.** Arena v2, with full support for autonomous agent research (a research gym
> with MCP tools, metered compute, research traces and self-improving research loops), ships by **November 14, 2026**.

Any LLM, with or without a research framework, can already enter an architecture today: it is trained under the same
protocol, ranked on the same leaderboard, and disclosed with how it was found. See the
[arena](https://interactive-active-ai.github.io/NextTransformer/docs/arena/overview/).

## Roadmap

- **By November 14, 2026:** Arena v2 and agent-native research.
- **November 2026:** recipes for 400B tokens and larger models.
- **Upcoming results:** Hyper-Connections, mHC, four more hybrids and three controls.
- **Planned:** checkpoints and the tokenizer on the Hugging Face Hub.

More on the [roadmap](https://interactive-active-ai.github.io/NextTransformer/docs/roadmap/).

## Community

- [GitHub Discussions](https://github.com/Interactive-Active-AI/NextTransformer/discussions) for questions and ideas
- [WeChat group](https://interactive-active-ai.github.io/NextTransformer/docs/community/#wechat) (QR code coming soon)

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation

```bibtex
@software{nexttransformer2026,
  title   = {NextTransformer: An Open Framework and Fair Benchmark for Next-Generation Transformer Architectures},
  author  = {{NextTransformer contributors}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Interactive-Active-AI/NextTransformer}
}
```

## Acknowledgments

NextTransformer builds on [nanochat](https://github.com/karpathy/nanochat) and ideas from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), uses
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) kernels, and trains on
[ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle). We thank the authors of every method in the
zoo. Released under the [MIT License](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md).
