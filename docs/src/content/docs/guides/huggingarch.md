---
title: HuggingArch architecture analysis
description: Analyze fixed model specs with an optional dependency and write per-model JSON results.
---

[HuggingArch](https://github.com/shenh10/HuggingArch) evaluates a declarative architecture spec to estimate
parameters, inference FLOPs, weight bytes and layer costs. This integration exports NextTransformer's actual
configuration and forward source, supplies a fixed YAML spec, and writes JSON results. No server is needed.

## Dependency

`tools/huggingarch/pyproject.toml` declares HuggingArch directly as a Git dependency, pinned to the tested
revision `facbbbc96856e558b9084a94a5070b2e2c86cf0b`; `uv.lock` fixes its transitive dependencies.
Run from the repository root with `uv run --project tools/huggingarch --frozen` to install and use it.

The separate tool environment uses Python 3.13 and CPU PyTorch 2.9.1. The training environment supports
Python 3.10–3.13 and pins `huggingface-hub==1.26.0`; the upstream Transformers dependency requires
`huggingface-hub>=1.31.0`. Keeping these environments separate avoids changing training dependencies.
The adapter imports the installed upstream package; no manual clone is necessary. For upstream development,
`--huggingarch-root` and `--huggingarch-python` can override the checkout and interpreter.

## Interface

The adapter has three normal operations:

| Operation | Input | Output |
| --- | --- | --- |
| `export` | Registered method names; repository configurations and source | A local bundle containing config, source, fixed spec and manifest |
| `register` | Bundle | HuggingArch custom-model registration and installation of the bundled spec |
| `analyze` | Bundle, batch size, sequence length, dtype | `<method>/analyze.json` and aggregate `report.json` |

```bash
python -m nexttransformer.integrations.huggingarch export \
  --methods dense gdn ut_act --output .agent-work/huggingarch
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch register \
  --bundle .agent-work/huggingarch
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch analyze \
  --bundle .agent-work/huggingarch --batch-size 1 --seq-len 2048 --dtype bfloat16
```

Omit `--methods` during export to include all 33 registered methods and four historical native models.
`register` and `analyze` also accept `--methods` to select a subset of an existing bundle. Export requires an
empty output directory. Subsequent operations reuse that bundle. Analysis defaults to batch size 1, each
model's configured context length, and BF16. `--timeout` limits each analysis to 120 seconds by default.
One model's failure does not prevent the other results from being written; any incomplete selected model
makes the command exit nonzero.

The upstream boundary is deliberately small:

```python
from backend.custom_model import register
from backend.analyzer import analyze_model

register(name="nextt-dense", config_path="config.json", forward_paths=["forward.py"])
# The adapter installs the fixed spec.yaml in this custom model's cache.
result = analyze_model(
    model_id="custom:nextt-dense", batch_size=1, seq_len=2048,
    dtype="bfloat16", trust_remote_code=False,
)
```

The adapter executes this boundary in an isolated worker and removes the upstream non-JSON resolved object
before serialization. Its loader workaround scopes the custom-spec scan to one model: the pinned upstream
otherwise keys multiple custom `spec.yaml` files by the same filename and can select another model's graph.
Call the CLI for this complete behavior; the snippet illustrates the dependency API only.

## JSON output

```text
manifest.json                       # model names, configuration paths and training evidence
report.json                         # aggregate results for all exported models
<method>/config.json                # full configuration with HuggingArch dimension aliases
<method>/nexttransformer-config.json # original repository configuration
<method>/forward.py                 # source material for spec authoring
<method>/spec.yaml                  # fixed architecture graph, when supported
<method>/spec-review.json           # supported configuration, scope, checks and limitations
<method>/register.json              # registration outcome
<method>/analyze.json               # analysis outcome, inputs and upstream JSON response
```

`report.json` has `schema_version`, `notice` and a `models` array. Each record contains:

- `model`: method name, stable `model_id` such as `custom:nextt-dense`, configuration path, weight precision
  assumption and any existing training cost evidence.
- `result`: `status`, analysis `settings`, `checkpoint_validated: false`, dependency version/revision, and
  the upstream `analysis` object on success or `error` on failure. Before analysis, this is the latest
  registration/generation outcome, or simply `{"status": "exported"}`.
- `spec_review`: supported configuration, analysis scope, checks and limitations, if a fixed spec exists.

Read `result.analysis.summary` for `total_params`, `total_flops`, `total_flops_gflops` and `weight_bytes`;
`result.analysis.layer_groups` retains upstream layer breakdowns. These inference estimates are separate
from `model.training_flops_per_token` and the existing cost proof. Missing results remain missing.
Use `report --bundle .agent-work/huggingarch` to rebuild the aggregate from local JSON files without
HuggingArch. No HTML report or content/spec/prompt hashes are generated by this integration. Original
training configurations and cost proofs retain their existing protocol fields.

## Fixed specs and updates

`tools/huggingarch/specs/` contains 35 fixed YAML graphs: all 33 registered methods, `native-dense` and
`native-loopformer`. GPT-6.1 Sol agents authored them using HuggingArch's own per-model `prompt.md`, shared
schema and bidirectional source review instructions. Normal registration and analysis do not invoke an agent.

The catalog maps each method to its spec and explicit supported configuration. Export compares configuration
values directly; changed dimensions or architecture options do not silently receive an incompatible spec.
Weight dtype is an estimate and does not change graph selection. Names are stable, without content hashes.
Update and review the corresponding spec whenever the model implementation changes; source changes are not
automatically tracked by the catalog. Registration compares cached config, source and spec text directly and
refuses conflicting material. To intentionally replace a cache entry, remove that specific custom model
through HuggingArch's custom-model management, then register the reviewed bundle again.

`export --weight-dtype float32` changes the assumed stored-weight precision (default BF16).
`analyze --dtype` controls the upstream analysis request; it does not rewrite the stored-weight assumption.
A different exported configuration under an already registered name requires explicit cache replacement.

`native-mor` and `native-circle` remain `unsupported_architecture`: their compacted token axes depend on
runtime token survival and the maximum live count across a batch. Their catalog records explain the missing
representation. They are not substituted with dense graphs. Authored specs support full-sequence prefill at
sequence lengths 2–2048; other lengths report `unsupported_settings`.

## Optional authoring

For a model without a fixed spec, register first to obtain HuggingArch's per-model prompt. An authoring agent
must read that prompt, the upstream schema/review instructions, configuration and actual source. Review and
check its YAML before committing it with its supported configuration and limitations.

Alternatively, the adapter can call upstream's own agent generation and publication workflow:

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch --methods dense --agent-cli codex
```

This reuses an existing spec without an agent call. Otherwise it calls `backend.custom_model.run_spec_generation`
with the installed/authenticated `codex`, `claude` or `kimi` CLI. It records `generate.json` and `generate.log`,
with a default timeout of 600 seconds. Failed validation or a missing publication stays `generation_failed`;
a draft is never promoted by the adapter. Upstream publication may require checkpoint headers that these
architecture-only exports do not supply. A debug-sized `export --smoke --methods mha_no_ve` uses a separate
`-smoke` name, attaches no formal cost evidence, and requires its own spec.

## Validation scope

The supplied specs passed parameter-count, tensor-name/shape/multiplicity and graph-geometry checks against
configured PyTorch meta models at `(batch, sequence)` values `(1,16)`, `(2,16)` and `(1,2048)`. No weights or
training data are required for ordinary analysis. HuggingArch evaluates the spec; it does not accept an
arbitrary `torch.nn.Module` and automatically recover an exact graph from it.

These checks do not establish numerical output equivalence, checkpoint validity, measured latency or peak
memory. No checkpoint was provided; publication validation reports `CheckpointUnavailable`. Per-model
limitations describe dynamic execution assumptions, custom operation costs, precision and cache estimates.
The training protocol, model implementations and calibration proofs are unchanged.
