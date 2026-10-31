---
title: Configuration
description: Site settings for paths, caches, mirrors and tracking, and how they are resolved.
---

Scientific configurations (`configs/`) never contain machine paths. Paths, caches, mirrors and tracking come from
command-line options, a local site file, or environment variables, resolved in this order:

**command line > local JSON (`--site-config`) > environment > defaults derived from `runtime_dir`.**

Paths in the JSON file resolve against the file; command-line paths resolve against the working directory. A JSON
field `venv_dir` is the option `--venv-dir`. Resolving settings never creates input files, and changing paths never
relaxes the source, configuration, tokenizer, data or world-size checks.

## Paths

`R` is the persistent runtime root.

| Option / JSON | Environment | Default and purpose |
|---|---|---|
| `--runtime-dir` / `runtime_dir` | `NANOCHAT_RUNTIME` | Required persistent root |
| `--venv-dir` / `venv_dir` | `NANOCHAT_VENV` | `R/venv` |
| `--resources` / `resources` | `NEXTTRANSFORMER_RESOURCES` | `R/resources`: tokenizer, CORE bundle, fingerprints |
| `--data-dir` / `data_dir` | `NEXTTRANSFORMER_DATA_DIR` | The ClimbMix directory; required for preparation |
| `--source-manifest` / `source_manifest` | `NEXTTRANSFORMER_SOURCE_MANIFEST` | Optional official file manifest; the pinned revision is checked |
| `--core-archive` / `core_archive` | `NEXTTRANSFORMER_CORE_ARCHIVE` | Optional local CORE ZIP; the pinned SHA-256 is checked |
| `--upstream-base-dir` / `upstream_base_dir` | `NANOCHAT_BASE_DIR` | `R/upstream`; set it to the same directory as `resources` |
| `--output` | | Exact training or evaluation destination |
| `--run-name` / `run_name` | `NEXTTRANSFORMER_RUN_NAME` | Optional run directory and tracking display name |

## Caches

Every cache has a JSON field, a hyphenated option and its usual environment variable: `hf_home` (`HF_HOME`),
`hf_hub_cache`, `tiktoken_cache_dir`, `torch_home`, `triton_cache_dir`, `torchinductor_cache_dir`, `cuda_cache_path`,
`xdg_cache_home`, `pythonpycacheprefix`, `uv_python_install_dir`, `uv_cache_dir`, `pip_cache_dir`, `tmpdir` and the
W&B directories. Each defaults to a directory under `R`, so a job keeps all caches on persistent storage. Each local
rank gets its own Triton and Inductor cache directory.

## Mirrors and offline use

| Option / JSON | Environment | Purpose |
|---|---|---|
| `--index-url` / `index_url` | `NEXTTRANSFORMER_INDEX_URL` | Package index for environment setup |
| `--hf-endpoint` / `hf_endpoint` | `HF_ENDPOINT` | Hugging Face Hub mirror |
| `--core-url` / `core_url` | `NEXTTRANSFORMER_CORE_URL` | CORE bundle mirror; the fixed SHA-256 is still checked |
| `--python-version` / `python_version` | `NEXTTRANSFORMER_PYTHON_VERSION` | Python for `setup_environment.sh` (default 3.10.19) |

## Tracking

| Option / JSON | Environment | Default |
|---|---|---|
| `--wandb-mode` / `wandb_mode` | `WANDB_MODE` | `disabled`; also `offline`, `online` |
| `--wandb-metrics` / `wandb_metrics` | `NEXTTRANSFORMER_WANDB_METRICS` | `core` (compact profile); `full` |
| `--wandb-project`, `--wandb-entity`, `--wandb-group` | `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_GROUP` | none |

Credentials come from `WANDB_API_KEY` or a mounted netrc file (`--wandb-netrc`); JSON files never accept keys. Local
JSON, `.env*` files and tracking logs stay out of git. A forward online resume in the same output directory reuses the
run ID; tracking choices never relax training-state checks. See [training metrics](../metrics/).

## Other environment variables

Use the environment for `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, NCCL settings and proxies. The launchers read
`NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` (see the
[quick start](../../getting-started/quick-start/)). `NEXTTRANSFORMER_JOB_ID` (or `SLURM_JOB_ID`) is recorded in a
run's provenance when set. `NEXTTRANSFORMER_METHOD_PATHS` lists extra plugin roots for research.
