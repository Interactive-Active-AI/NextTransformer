---
title: Data and tokenizer
description: Download the pinned ClimbMix revision and prepare the tokenizer and evaluation resources.
---

Every run reads the same data: [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) at revision
`915333b4f8b8684f39aeaafea600fea6f43fb703`, 6,543 Parquet shards, about **600 GB**. Training reads shards in sorted
order from the first; `shard_06542.parquet` is the validation shard and `shard_06541.parquet` is the dev shard for
research.

## 1. Point the tools at your storage

Paths never go into scientific configurations. Copy the example site file and set the persistent paths:

```bash
cp site.native.example.json site.local.json
```

```json
{
  "runtime_dir": "/path/to/runtime",
  "resources": "/path/to/runtime/resources",
  "upstream_base_dir": "/path/to/runtime/resources",
  "data_dir": "/path/to/datasets/climbmix-400b-shuffle",
  "wandb_mode": "disabled"
}
```

Files named `*.local.json` are ignored by git. Every field can also be given on the command line
(`--data-dir`, `--resources`, ...) or as an environment variable; see [configuration](../../guides/configuration/).

## 2. Download the pinned revision

```bash
hf download karpathy/climbmix-400b-shuffle --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 --local-dir /path/to/datasets/climbmix-400b-shuffle
```

## 3. Prepare resources

```bash
source launch/env.sh --site-config site.local.json
python -m nexttransformer.prepare --site-config site.local.json
```

Preparation checks that the directory holds exactly the pinned files and sizes and that the validation shard matches
the Hub checksum, trains the 32,768-token tokenizer and its byte table from the original Parquet, and prepares the
pinned CORE-22 evaluation bundle. The trainer refuses a tokenizer whose SHA-256 differs from the pinned one, so every
machine ends up with identical resources. Reuse a verified CORE bundle with `--existing-core-resources PATH`.

Prepare once per shared filesystem and point every run at the same `resources` directory. Moving resources later
requires unchanged Parquet contents, tokenizer, byte table and CORE bundle.
