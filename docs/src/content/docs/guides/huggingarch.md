---
title: HuggingArch architecture analysis
description: Export every local architecture to HuggingArch and browse per-model analysis without changing training.
---

[HuggingArch](https://github.com/shenh10/HuggingArch) is an optional architecture and inference-cost inspector.
NextTransformer methods are local research implementations, not the similarly named Hugging Face models.
The bridge registers their actual configurations and forward sources using HuggingArch's
[`custom:` workflow](https://github.com/shenh10/HuggingArch/blob/main/docs/custom_models.md).

The tool creates a portable HTML report with one row and detail section per model. It shows the existing
training cost evidence alongside available HuggingArch parameter counts, prefill FLOPs, weight sizes and
layer groups. Full analysis JSON is retained. No service, GPU, checkpoint, dataset or model execution is
needed to export or browse the report.

## What the dependency takes as input

HuggingArch's Python entry point is `backend.analyzer.analyze_model(model_id, batch_size, seq_len, dtype)`.
For a local model, `model_id` resolves to the following registered inputs:

| Input | Purpose | Required? |
| --- | --- | --- |
| `config.json` | Model dimensions, architecture-specific options and explicit weight dtype | Yes |
| Forward source and helpers | Ground truth for authoring/reviewing the architecture spec | For custom registration/spec generation |
| `spec.yaml` and any custom components | The declarative graph the analysis engine evaluates | Yes, for analysis |
| `batch_size`, `seq_len`, `dtype` | Requested analysis conditions | Defaults supplied; configurable |
| Safetensors tensor headers | Reconcile shapes, dtypes and stored bytes against a real checkpoint | Optional for estimates; required for checkpoint validation |

The analyzer does not accept an arbitrary `torch.nn.Module` or infer its exact computation from config alone.
Source registration prepares the material for a spec; it is not a replacement for that spec. Once a spec exists,
the installed Python dependency computes the analysis without a web server or a spec-generation agent.

## Fixed specs authored with the upstream prompts

`tools/huggingarch/specs/catalog.json` records the authored specs and the outcome for every exported
configuration. GPT-6.1 Sol agents read HuggingArch's registered per-model `prompt.md`, its shared schema
and review instructions, and the actual forward source. Each entry records the prompt hash, configuration/
source identity, spec hash, checks and limitations. Authoring helpers are not runtime dependencies.

Export automatically includes a checked-in spec only when its complete configuration/source identity
matches the catalog. Registration installs that YAML into HuggingArch's cache; a different existing spec
is never overwritten. The usual register/analyze path therefore needs no agent for these fixed artifacts.
Changed architectures get new identities and need new spec authoring. An unsupported entry has no YAML;
its catalog entry explains what remains unrepresented.

These checks establish loadability, graph geometry and parameter agreement with the corresponding meta
model. They do not establish numerical equivalence, measured performance, or checkpoint validation.
The scope is full-sequence prefill: cache layouts are architectural estimates where the source has no
decode implementation. Read each entry's limitations, including upstream attribution fields that may
report zero for work already included in total FLOPs. The report links the spec and its review record.
Authored specs require sequence lengths from 2 through the exported context length; the bridge rejects
requests outside that range. The source's single-token smear bypass is not represented by these graphs.

## 1. Export configurations and source

Run from the NextTransformer checkout:

```bash
python -m nexttransformer.integrations.huggingarch export \
  --output .agent-work/huggingarch
```

By default this exports all registered methods (33 built-ins today) plus four historical native configurations.
Native names have a `native-` prefix so historical Dense and the calibrated SSSL anchor cannot collide.
To select methods, add `--methods dense_full gdn kda native-circle`. In-repository plugins with an admitted
configuration are included automatically; missing configs fail export explicitly. External plugin roots must
first be moved into the repository. Each output directory must be empty; registration and analysis reuse it.

Open `.agent-work/huggingarch/index.html` directly in a browser. At this point the rows say `exported`:
**exporting source does not generate a HuggingArch spec or an analysis**. Existing training evidence is
visible immediately; unavailable inference values are shown as dashes.
Matching authored specs are copied into the bundle at export time but are analyzed only when requested.

The bundle contains:

```text
manifest.json                    # source revision, identities and hashes
index.html / report.json          # per-model report
<method>/config.json              # complete method options plus inspector dimension aliases
<method>/nexttransformer-config.json
<method>/forward.py               # source text with repository-relative filename banners
<method>/register.json            # written by register
<method>/generate.json / generate.log # upstream generation outcome and diagnostics
<method>/spec.yaml                # copy of the published spec, written by generate
<method>/spec-review.json         # identity, prompt provenance and limitations for a bundled spec
<method>/analyze.json             # written by analyze, including failures and analysis inputs
```

`forward.py` is input text for spec generation, not a standalone executable. It contains the model package,
shared backbone, kernel helpers and the selected plugin's source. Looped models retain physical layers and
loop/halting settings; hybrids retain their layer layout. No stock Llama/Qwen spec is substituted.

Weights-less analysis needs a precision assumption: export writes `torch_dtype: bfloat16` by default.
Use `export --weight-dtype float32` (or `float16`) to change it. This is an explicit estimate of
unquantized weight storage, not evidence of the training checkpoint's actual tensor dtypes.
The assumption is recorded in the report and participates in the model identity.

## 2. Register using the declared dependency

The tool declares HuggingArch as a Git-pinned Python dependency in `tools/huggingarch/pyproject.toml`,
with transitive versions in its `uv.lock`. No manual clone, source path or interpreter path is needed:

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch register \
  --bundle .agent-work/huggingarch
```

Run this from the NextTransformer repository root. `uv` installs the dependency and selects Python 3.13.
The normal adapter imports the installed package, and records its version and Git revision in the result.
The pinned upstream revision is `facbbbc96856e558b9084a94a5070b2e2c86cf0b`.

The tool project is separate because the dependency constraints conflict: NextTransformer pins
`huggingface-hub==1.26.0`, while HuggingArch's `transformers>=5.20.0` requires `huggingface-hub>=1.31.0`.
HuggingArch also requires Python 3.13+, while training supports Python 3.10–3.13. The tool uses CPU PyTorch;
installing it does not upgrade the training environment or download CUDA runtimes.

The pinned upstream loader keys custom files by their common filename `spec.yaml`. The bridge scopes
each worker's custom-model scan to the requested model, preventing one model's graph from being used
for another when several custom specs are installed.

If HuggingArch is already installed in a compatible Python environment, simply run
`python -m nexttransformer.integrations.huggingarch register --bundle .agent-work/huggingarch` there.
For upstream development only, `--huggingarch-root` and `--huggingarch-python` remain optional overrides.

Registration calls HuggingArch's `backend.custom_model.register`. Each model has a content-addressed name
such as `custom:nextt-gdn-<hash>` and is stored in HuggingArch's local custom-model cache. Repeating registration
reuses identical material and preserves an existing spec. Changed configs or source produce new identities;
edited bundles and conflicting cached files are refused. Registration does not launch an agent.

## 3. Generate each spec once using the upstream agent

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch --agent-cli codex
```

`generate` registers missing models and calls HuggingArch's `backend.custom_model.run_spec_generation`.
The upstream agent reads the configuration and forward source, authors the spec, and runs its validation
and publication pipeline. The bridge contains no per-architecture conversion rules and never constructs
a PyTorch model. It accepts every exported method; support for its operations remains an upstream concern.

An existing cached spec is reused without starting an agent or rewriting it. Each model identity is fixed
by its configuration and source hash, so changing either produces a new identity and a separate spec.
Batch size and analysis context do not change the spec identity. `generate.json` records whether generation
reused a spec and copies its SHA-256; `validation: existing_spec` does not claim a fresh validation run.
The serving spec and any components remain in HuggingArch's cache; the bundle's YAML copy is for inspection.

Install and authenticate your chosen upstream agent (`codex`, `claude` or `kimi`) before first generation.
Omitting `--agent-cli` uses HuggingArch's configured agent. Generation can incur the agent provider's usage
charges; subsequent analysis uses the cached spec without an LLM call. `--timeout` bounds generation to
600 seconds per model by default. Generation logs are kept in `<method>/generate.log`, including agent or
validation failures. Use `--methods mha_no_ve` to generate just one model before processing the full bundle.

An upstream generation failure stays `generation_failed`; a draft YAML is not promoted by this bridge.
In particular, these exports have no checkpoint headers: upstream's checkpoint validation gates may block
publication even if a draft can be evaluated. We do not bypass those gates or claim checkpoint validation.
An already reviewed spec registered through HuggingArch remains usable for weights-less estimates.

You can alternatively use your own HuggingArch Inspector's generation workflow. The public hosted instance
cannot read the local files. Both paths use the same upstream custom-model identity and serving cache.

Review the spec against the exported source, particularly value embeddings, ReLU-squared MLPs, QK norm,
residual scalars/backout, sparse selection, shared recurrent weights and input-dependent halting. A familiar
method name does not establish equivalence with the paper's original model. Dynamic routes need explicit
assumptions in the spec; this bridge does not infer a measured average depth.

This integration delegates generation and validation; **it does not ship validated specs for all methods**.
Some methods may require new HuggingArch components before their full graph can be represented. Missing
or invalid specs remain visibly incomplete instead of being replaced by a dense approximation.

## 4. Analyze and browse

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch analyze \
  --bundle .agent-work/huggingarch \
  --batch-size 1 --seq-len 2048 --dtype bfloat16
```

Open the report again. Each model runs in a separate subprocess with a per-model timeout (`--timeout`,
default 120 seconds for analysis). An analysis never starts an agent; it only evaluates the existing spec.
One failing model does not prevent the other results or the report from being written.
Use `--methods` to retry a subset. The command exits nonzero if any selected model remains incomplete.
Rerunning analysis replaces that model's prior result, including replacing old success with a new failure.

| Status | Meaning / next step |
| --- | --- |
| `exported` | Local bundle exists; register it. |
| `registered` | Config and source are available to HuggingArch; generate/review the spec. |
| `spec_ready` | An existing spec was reused, or upstream generation completed and published a spec. |
| `generation_failed` | Upstream agent/validation failed or published no spec; inspect `generate.log`. |
| `registration_warning` | Registration could not prepare its prompt; inspect the recorded error and upstream dependencies. |
| `not_registered` | Register in the same HuggingArch environment/user account. |
| `needs_spec` | Generate and review the missing spec, then rerun analysis. |
| `analyzed` | A spec-derived result is available; this does **not** mean checkpoint validation passed. |
| `unsupported_settings` | Requested context is outside the authored spec's reviewed range. |
| `unsupported_architecture` | The catalog records a source/engine gap; read the per-model review for evidence. |
| `error` | Timeout, invalid spec, incompatible upstream API or another failure; inspect the error and retry. |

To add Inspector links, pass `--base-url "$HUGGINGARCH_URL"` to register, generate, analyze or report, where
`HUGGINGARCH_URL` is the URL of your existing local instance. The bridge does not start or expose a server.
The HTML and its relative file links also work without any server or external JavaScript:

```bash
python -m nexttransformer.integrations.huggingarch report \
  --bundle .agent-work/huggingarch --base-url "$HUGGINGARCH_URL"
```

For a bounded smoke with the repository's small debug configuration:

```bash
python -m nexttransformer.integrations.huggingarch export \
  --methods mha_no_ve --smoke --output .agent-work/huggingarch-smoke
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch-smoke --agent-cli codex --timeout 180
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch analyze \
  --bundle .agent-work/huggingarch-smoke
```

`--smoke` requires explicit methods, gives the debug configuration its own identity, and attaches no formal
training cost evidence. If generation fails, analysis remains incomplete. Without `--seq-len`, analysis
uses each exported configuration's context length (12 in this smoke), not a hard-coded 2,048.

## Reading the numbers

- **Training GFLOPs/input token and proof parameters** are copied from the configuration's cost record,
  with its path and formula version retained. They are not recomputed, revalidated or used for admission.
  Historical native configs have no equivalent proof attached here and display dashes.
- **HuggingArch counts and prefill FLOPs** are the upstream spec-derived summary for the recorded batch,
  context and dtype request. The report does not convert them to training FLOPs. Weight bytes are shown
  in GiB under the exported `--weight-dtype` assumption; `analyze --dtype` is passed as an upstream
  analysis request and does not rewrite that stored-weight assumption. The complete upstream response
  remains available in `analyze.json`.
- **No checkpoint is exported.** HuggingArch's tensor/storage reconciliation cannot establish checkpoint
  validity from this bundle. `checkpoint_validated` stays false, even when analysis succeeds.
- Latency estimates available in HuggingArch's own Inspector/Planner are analytical estimates, not measured
  benchmark results. The bridge does not add fabricated latency, throughput, KV-cache or evaluation scores.

The training protocol, model implementations, published results and calibration proofs are untouched.
