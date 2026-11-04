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

The bundle contains:

```text
manifest.json                    # source revision, identities and hashes
index.html / report.json          # per-model report
<method>/config.json              # complete method options plus inspector dimension aliases
<method>/nexttransformer-config.json
<method>/forward.py               # source text with repository-relative filename banners
<method>/register.json            # written by register
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

If HuggingArch is already installed in a compatible Python environment, simply run
`python -m nexttransformer.integrations.huggingarch register --bundle .agent-work/huggingarch` there.
For upstream development only, `--huggingarch-root` and `--huggingarch-python` remain optional overrides.

Registration calls HuggingArch's `backend.custom_model.register`. Each model has a content-addressed name
such as `custom:nextt-gdn-<hash>` and is stored in HuggingArch's local custom-model cache. Repeating registration
reuses identical material and preserves an existing spec. Changed configs or source produce new identities;
edited bundles and conflicting cached files are refused. No agent is launched by the bridge.

## 3. Generate and review the specs

Use your own HuggingArch instance; the public hosted instance cannot read these local files.
Copy the model ID from the report into its Inspector. For a registered model without a spec, HuggingArch
offers its normal spec-generation workflow. Configure its agent as described in the upstream documentation,
or use the generated `prompt.md` to author `spec.yaml` manually in the custom-model cache.

Review the spec against the exported source, particularly value embeddings, ReLU-squared MLPs, QK norm,
residual scalars/backout, sparse selection, shared recurrent weights and input-dependent halting. A familiar
method name does not establish equivalence with the paper's original model. Dynamic routes need explicit
assumptions in the spec; this bridge does not infer a measured average depth.

This PR provides the registration/reporting bridge; **it does not ship validated specs for all methods**.
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
default 120 seconds). One failing model does not prevent the other results or the report from being written.
Use `--methods` to retry a subset. The command exits nonzero if any selected model remains incomplete.
Rerunning analysis replaces that model's prior result, including replacing old success with a new failure.

| Status | Meaning / next step |
| --- | --- |
| `exported` | Local bundle exists; register it. |
| `registered` | Config and source are available to HuggingArch; generate/review the spec. |
| `registration_warning` | Registration could not prepare its prompt; inspect the recorded error and upstream dependencies. |
| `not_registered` | Register in the same HuggingArch environment/user account. |
| `needs_spec` | Generate and review the missing spec, then rerun analysis. |
| `analyzed` | A spec-derived result is available; this does **not** mean checkpoint validation passed. |
| `error` | Timeout, invalid spec, incompatible upstream API or another failure; inspect the error and retry. |

To add Inspector links, pass `--base-url "$HUGGINGARCH_URL"` to register, analyze or report, where
`HUGGINGARCH_URL` is the URL of your existing local instance. The bridge does not start or expose a server.
The HTML and its relative file links also work without any server or external JavaScript:

```bash
python -m nexttransformer.integrations.huggingarch report \
  --bundle .agent-work/huggingarch --base-url "$HUGGINGARCH_URL"
```

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
