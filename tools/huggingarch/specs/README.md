# Authored architecture specs

These fixed YAML artifacts cover all 33 registered methods, plus `native-dense` and
`native-loopformer`. The catalog also records why `native-mor` and `native-circle`
have no static spec: their compacted query/key lengths depend on learned token
survival and the maximum active-token count across the batch. A dense substitute
would misrepresent their actual execution.

GPT-6.1 Sol subagents authored the specs using each model's **HuggingArch-generated
`prompt.md`**, including its schema, source material and bidirectional review
instructions. Identical shared prompt sections were reviewed once per group;
model-specific configuration and task sections were reviewed individually. Prompt
hashes and section records are in [catalog.json](catalog.json).

Each supplied spec has passed independent checks with the pinned installed
HuggingArch engine:

- Complete parameter counts match an actual configured PyTorch meta model.
- Materialized tensor names, every shape axis and multiplicities match that model;
  no duplicate tensor names remain.
- Both upstream graph-geometry checks pass at batch/context combinations `(1,16)`,
  `(2,16)` and `(1,2048)`.
- Full-context analysis results are available through the bridge. Parameter counts
  agree with formal cost records where those records exist; inference FLOPs remain
  a different metric from training cost proofs.

**These are analytical checks, not checkpoint validation or numerical output
equivalence.** No weights were loaded or training run. Upstream publication
validation reports `CheckpointUnavailable`. Per-model limitations qualify custom
operation cost conventions, precision assumptions, cache estimates, summary-field
attribution and unverified activation-memory/parallel-deployment behavior. In
particular, sparse reference kernels operate in chunks; a full-sequence logical
graph does not establish peak allocation or measured traffic.

## Reuse

From the repository root:

```bash
python -m nexttransformer.integrations.huggingarch export --output .agent-work/models
uv run --project tools/huggingarch --frozen python -m nexttransformer.integrations.huggingarch register --bundle .agent-work/models
uv run --project tools/huggingarch --frozen python -m nexttransformer.integrations.huggingarch analyze --bundle .agent-work/models
```

Export copies a spec and its review only when the full configuration/source hash
matches. Registration installs it without an agent call and refuses to overwrite
a different cached spec. Analysis also rejects mismatched specs and contexts
outside the authored range (2–2,048 tokens). The all-model command reports the two
unsupported native models and exits nonzero while retaining the other results;
use `--methods` to select supported models only.

Changing source or configuration creates a new model identity. Re-author and
recheck the corresponding spec before updating the catalog. Do not change a spec
hash to suppress a review mismatch without repeating its checks. Authoring scripts,
prompts, temporary caches and full analysis logs are deliberately not runtime
dependencies or checked-in artifacts.
