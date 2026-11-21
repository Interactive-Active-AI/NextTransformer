# Analysis snapshots

[b1-s2048-bf16.json](b1-s2048-bf16.json) contains the complete report for all 37
configurations: 35 analyzed and 2 unsupported (`native-mor`, `native-circle`).
Conditions: batch size 1, sequence length 2048, BF16 analysis and stored weights.
Each model record includes its analysis settings, upstream result, spec scope and
validation limits. These are architecture estimates, not measured performance.

## Reproduce or change batch size

From the repository root, export to a new directory and register the fixed specs:

```bash
python -m nexttransformer.integrations.huggingarch export --output .agent-work/huggingarch
uv run --project tools/huggingarch --frozen python -m nexttransformer.integrations.huggingarch register --bundle .agent-work/huggingarch
uv run --project tools/huggingarch --frozen python -m nexttransformer.integrations.huggingarch analyze --bundle .agent-work/huggingarch --batch-size 1 --seq-len 2048 --dtype bfloat16
```

The all-model command records both unsupported models and exits nonzero after
writing the other results. To analyze a supported subset with another batch size:

```bash
uv run --project tools/huggingarch --frozen python -m nexttransformer.integrations.huggingarch analyze --bundle .agent-work/huggingarch --methods dense gdn ut_act --batch-size 8 --seq-len 2048
```

This evaluates the same specs without invoking an agent, allocating model weights
or running inference kernels. Each run replaces the selected models' results in
that bundle; copy `report.json` to a separate snapshot path to retain each setting.
Only selected models are recomputed, so use each record's `result.settings` when
reading a bundle containing results from several runs. Fixed specs currently
support sequence lengths 2–2048.
