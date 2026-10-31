<div align="center">

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/banner.jpg" alt="NextTransformer: toward the next generation of Transformers" width="100%"></a>

<h3>The comprehensive, cutting-edge open-source framework, benchmark<br>and auto-research platform for next-generation Transformers</h3>

<p>
<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/readme/button-observatory-en.webp" alt="Explore the observatory" height="60"></a>
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/"><img src=".github/assets/readme/button-docs-en.webp" alt="Read the docs" height="60"></a>
<a href="#quick-start"><img src=".github/assets/readme/button-quickstart-en.webp" alt="Quick start" height="60"></a>
</p>

<p>
<a href="https://github.com/Interactive-Active-AI/NextTransformer/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Interactive-Active-AI/NextTransformer/ci.yml?branch=main&style=flat-square&label=CI&color=3F7FD9"></a>
<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-live-8B5CF6?style=flat-square"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-F2A93B?style=flat-square"></a>
<img alt="Python 3.10-3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="PyTorch 2.9.1" src="https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
<a href="https://interactive-active-ai.github.io/NextTransformer/docs/community/#wechat"><img alt="WeChat group" src="https://img.shields.io/badge/WeChat-group-E5607E?style=flat-square&logo=wechat&logoColor=white"></a>
<a href="https://github.com/Interactive-Active-AI/NextTransformer/discussions"><img alt="GitHub Discussions" src="https://img.shields.io/badge/discussions-ask-7A5AF0?style=flat-square&logo=github&logoColor=white"></a>
</p>

<b>English</b> · <a href="README.zh-CN.md">简体中文</a>

</div>

<br>

**When the training budget is fixed, how should a language model spend its computation?** NextTransformer trains
recurrent depth, sparse and linear attention, residual-stream variants and hybrids on the same data, for the same
49.152B tokens, at the same theoretical training FLOPs per token, and evaluates them at the same checkpoints. Every
result is an audited, file-backed record, and AI agents can research new architectures under the very same rules.

## News

- <img src="https://img.shields.io/badge/NEW-E5607E?style=flat-square" alt="New" height="18"> **2026-10-31** · NextTransformer is now open source: 29 finished runs, 39 model cards, the full protocol, the documentation and the observatory.
- <img src="https://img.shields.io/badge/SOON-8B5CF6?style=flat-square" alt="Soon" height="18"> **By 2026-11-14** · **Arena v2**, with full support for autonomous agent research.
- <img src="https://img.shields.io/badge/SOON-8B5CF6?style=flat-square" alt="Soon" height="18"> **November 2026** · Training recipes for **400B tokens and larger models**.

## Highlights

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/methods/zoo/"><img src=".github/assets/readme/card-framework-en.webp" alt="Frontier architectures, one framework: 39 models in seven families, in one codebase; a new method is one plugin folder." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/"><img src=".github/assets/readme/card-protocol-en.webp" alt="A fair, explicit protocol: same data, tokens, optimizer, schedule and evaluation for every model; training compute matched to within 1% and audited over the whole run." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/getting-started/quick-start/"><img src=".github/assets/readme/card-scale-en.webp" alt="1B-scale pretraining on one machine: a ~1.4B-parameter model on 49B tokens in 2–4 days on one 8-GPU node; 400B tokens and larger models in November." width="100%"></a>

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/arena/task/"><img src=".github/assets/readme/card-agent-en.webp" alt="Agent-native research: hand the arena task to your agent; the full version, Arena v2, ships by November 14, 2026." width="100%"></a>

## The observatory

<a href="https://interactive-active-ai.github.io/NextTransformer/"><img src=".github/assets/readme/observatory.webp" alt="The NextTransformer observatory: learning curves, leaderboard, model atlas and arena" width="100%"></a>

Learning curves of every run, a replayable leaderboard, all 22 CORE tasks, architecture diagrams and model cards, in
your browser.

## Results

<!-- generated:leaderboard -->
<a href="https://interactive-active-ai.github.io/NextTransformer/#leaderboard"><img src=".github/assets/readme/results-en.svg" alt="CORE at update 46,875 for all 29 runs, against the Dense Baseline" width="100%"></a>

Numbers and notes for every run are on the [leaderboard](https://interactive-active-ai.github.io/NextTransformer/docs/results/leaderboard/); replay training in the [observatory](https://interactive-active-ai.github.io/NextTransformer/#leaderboard).
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

<details open>
<summary><b>39 models in seven families</b></summary>

| Family | Models |
|---|---|
| <img src="https://img.shields.io/badge/Baseline-7A8288?style=flat-square" alt="Baseline"> | Dense (full causal attention) |
| <img src="https://img.shields.io/badge/Residual-33456E?style=flat-square" alt="Residual"> | Standard Residual · Block Attention Residuals · Hyper-Connections\* · mHC\* |
| <img src="https://img.shields.io/badge/Looped-8B5CF6?style=flat-square" alt="Looped"> | Universal Transformer · ACT · Fixed Recursive · Huginn · Ouro · DeepLoop · Mixture-of-Recursions · LoopFormer · Circle Loop |
| <img src="https://img.shields.io/badge/Sparse-E5607E?style=flat-square" alt="Sparse"> | DeepSeek Sparse Attention · Qwen Sparse Attention · Native Sparse Attention · SeerAttention-style · MoBA |
| <img src="https://img.shields.io/badge/Linear-3F7FD9?style=flat-square" alt="Linear"> | DeltaNet · Gated DeltaNet · Gated Linear Attention · Kimi Delta Attention · RetNet · Linear Transformer (ELU+1)† · ELU+1 with q/k-norm · KDA without VE\* |
| <img src="https://img.shields.io/badge/Hybrid-F07856?style=flat-square" alt="Hybrid"> | GDN + MHA · GDN + SWA · DeltaNet + MHA\* · KDA + MLA-NoPE\* · GDN + QSA\* · MoBA + MHA\* |
| <img src="https://img.shields.io/badge/Others-D99A2B?style=flat-square" alt="Others"> | SSSL · MQA · GQA · MLA + QK-norm · MLA-NoPE + QK-norm · MLA-NoPE\* · MHA without VE\* |

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

<a href="https://interactive-active-ai.github.io/NextTransformer/docs/arena/overview/"><img src=".github/assets/readme/arena-en.webp" alt="Arena v2, by November 14, 2026: agent-native research, end to end" width="100%"></a>

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
