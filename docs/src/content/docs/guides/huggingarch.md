---
title: HuggingArch architecture analysis
description: Analyze fixed model specs and write per-model JSON results.
---

[HuggingArch](https://github.com/shenh10/HuggingArch) evaluates YAML architecture specs to estimate
parameters, inference FLOPs, weight bytes and layer costs. This integration supplies NextTransformer's
configuration, forward source and fixed specs. Ordinary analysis needs no agent, weights or training data.

## Run an analysis

Run from the repository root:

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

| Command | Input | Output |
| --- | --- | --- |
| `export` | Method names, repository configurations and source | Local bundle with fixed specs where available |
| `register` | Bundle | HuggingArch custom models with the bundled specs installed |
| `analyze` | Bundle, batch size, sequence length, dtype | `<method>/analyze.json` and aggregate `report.json` |

Export requires an empty directory. Omit `--methods` to export all 37 configurations; register and analyze
also accept this option to select a subset. Analysis defaults to batch size 1, the configured context length
and BF16, with a per-model timeout of 120 seconds (`--timeout`). Failures are recorded per model; any
incomplete selected model makes the command exit nonzero after writing the other results.

`tools/huggingarch/pyproject.toml` and `uv.lock` pin the upstream dependency. The tool uses a separate
Python 3.13 environment with CPU PyTorch because its Python and Hugging Face Hub requirements conflict
with training dependencies. `uv run` installs it automatically. For upstream development,
`--huggingarch-root` and `--huggingarch-python` override the checkout and interpreter.

The adapter calls `backend.custom_model.register` and `backend.analyzer.analyze_model` in a worker.
It scopes the upstream custom-spec loader to one model to prevent collisions between files named `spec.yaml`,
and removes the non-JSON resolved object before serialization.

## Read the output

The complete batch-1, 2048-token, BF16 snapshot is checked in at
`tools/huggingarch/reports/b1-s2048-bf16.json` (35 analyzed, 2 unsupported).
To change batch size, rerun `analyze --bundle <directory> --batch-size 8` using the
same registered specs; no agent is called. Reruns replace the selected models'
results, so save `report.json` separately for each setting. Each record retains
its own `result.settings`.

The bundle contains `manifest.json`, `report.json`, and a directory per method with `config.json`,
`nexttransformer-config.json`, `forward.py`, any available `spec.yaml` and `spec-review.json`, and the
operation results (`register.json`, `analyze.json`).

`report.json` has `schema_version`, `notice` and a `models` array. Each model record contains:

| Field | Contents |
| --- | --- |
| `model` | Method, stable ID such as `custom:nextt-dense`, configuration path, weight precision and training cost evidence |
| `result` | Status, analysis settings and dependency version; `analysis` on success or `error` on failure |
| `spec_review` | Supported configuration, checks, scope and limitations, when available |

`result.analysis.summary` contains `total_params`, `total_flops`, `total_flops_gflops` and `weight_bytes`;
`result.analysis.layer_groups` contains the layer breakdown. Inference FLOPs are separate from
`model.training_flops_per_token`. Before analysis, `result` contains the registration/generation outcome
or `{"status": "exported"}`. Use `report --bundle .agent-work/huggingarch` to rebuild the aggregate locally.

`export --weight-dtype` sets the stored-weight precision assumption (default BF16). `analyze --dtype`
controls the analysis request and does not change that assumption.

## Spec coverage and maintenance

There are 35 fixed specs: all 33 registered methods, `native-dense` and `native-loopformer`.
`native-mor` and `native-circle` return `unsupported_architecture` because their compacted token axes depend
on runtime token survival and the maximum live count across a batch. Fixed specs cover full-sequence prefill
at 2–2048 tokens; other lengths return `unsupported_settings`.

`tools/huggingarch/specs/catalog.json` records supported configurations and per-model limitations. Export compares configuration values directly.
Source changes require reviewing and updating the spec; the catalog does not automatically detect them.
Registration refuses conflicting cached config, source or spec text. To replace an entry, remove that custom
model through HuggingArch's model management and register the reviewed bundle again.

The specs passed parameter-count, tensor-name/shape/multiplicity and graph-geometry checks against configured
PyTorch meta models at `(batch, sequence) = (1,16), (2,16), (1,2048)`. These checks do not establish numerical
output equivalence, measured latency or peak memory. No checkpoint was supplied: `checkpoint_validated`
remains false and upstream publication checks report `CheckpointUnavailable`.

## Author a new spec

Register the model to obtain HuggingArch's per-model `prompt.md`. The authoring agent should read that prompt,
the upstream schema/review instructions, configuration and forward source. Review and check the YAML before
committing it with its supported configuration and limitations. HuggingArch evaluates the spec; it does not
extract an exact graph from an arbitrary `torch.nn.Module`.

To use upstream's agent generation and publication workflow:

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch --methods dense --agent-cli codex
```

`generate` reuses an existing spec, otherwise calls `backend.custom_model.run_spec_generation` using an
installed/authenticated `codex`, `claude` or `kimi` CLI. It records `generate.json` and `generate.log`, with a
600-second default timeout. Validation failure or missing publication returns `generation_failed`; drafts
are not promoted. The checkpoint requirement above may block publication for architecture-only exports.

For a debug-sized configuration, use `export --smoke --methods mha_no_ve`. It gets a separate `-smoke` name,
no formal training cost evidence, and requires its own spec.
