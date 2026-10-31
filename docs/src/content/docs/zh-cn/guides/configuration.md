---
title: 运行环境配置
description: 配置本地路径、缓存、镜像和实验追踪，并了解各设置的优先级。
---

模型配置保存在 `configs/` 中。机器路径等运行环境设置按以下顺序解析，左侧优先：

**命令行 → 本地 JSON（`--site-config`）→ 环境变量 → 根据 `runtime_dir` 推导的默认值。**

JSON 中的相对路径从该文件所在目录解析，命令行中的相对路径从当前工作目录解析。例如 `venv_dir` 对应 `--venv-dir`。解析设置不会创建输入文件；无论路径如何改变，源码、配置、分词器、数据和进程总数仍需通过原有检查。

## 路径

下表以 `R` 表示持久化运行根目录。

| 命令行 / JSON 字段 | 环境变量 | 默认值与用途 |
|---|---|---|
| `--runtime-dir` / `runtime_dir` | `NANOCHAT_RUNTIME` | 必须指定的持久化根目录 |
| `--venv-dir` / `venv_dir` | `NANOCHAT_VENV` | `R/venv` |
| `--resources` / `resources` | `NEXTTRANSFORMER_RESOURCES` | `R/resources`，包含分词器、CORE 评测包和指纹 |
| `--data-dir` / `data_dir` | `NEXTTRANSFORMER_DATA_DIR` | ClimbMix 目录，准备数据时必填 |
| `--source-manifest` / `source_manifest` | `NEXTTRANSFORMER_SOURCE_MANIFEST` | 可选的官方文件清单，仍检查锁定的版本 |
| `--core-archive` / `core_archive` | `NEXTTRANSFORMER_CORE_ARCHIVE` | 可选的本地 CORE ZIP，仍校验 SHA-256 |
| `--upstream-base-dir` / `upstream_base_dir` | `NANOCHAT_BASE_DIR` | 默认 `R/upstream`；请设为与 `resources` 相同的目录 |
| `--output` | | 本次训练或评测的输出目录 |
| `--run-name` / `run_name` | `NEXTTRANSFORMER_RUN_NAME` | 可选的运行目录名和追踪面板显示名 |

## 缓存

缓存均可通过 JSON、带连字符的命令行参数或对应的环境变量配置。包括 `hf_home`（`HF_HOME`）、`hf_hub_cache`、`tiktoken_cache_dir`、`torch_home`、`triton_cache_dir`、`torchinductor_cache_dir`、`cuda_cache_path`、`xdg_cache_home`、`pythonpycacheprefix`、`uv_python_install_dir`、`uv_cache_dir`、`pip_cache_dir`、`tmpdir` 和 W&B 的目录。默认都位于持久存储 `R` 下。每个本地 rank 有独立的 Triton 与 Inductor 缓存。

## 镜像与离线运行

| 命令行 / JSON 字段 | 环境变量 | 用途 |
|---|---|---|
| `--index-url` / `index_url` | `NEXTTRANSFORMER_INDEX_URL` | 安装环境时使用的包索引 |
| `--hf-endpoint` / `hf_endpoint` | `HF_ENDPOINT` | Hugging Face Hub 镜像 |
| `--core-url` / `core_url` | `NEXTTRANSFORMER_CORE_URL` | CORE 评测包镜像，下载后仍校验固定的 SHA-256 |
| `--python-version` / `python_version` | `NEXTTRANSFORMER_PYTHON_VERSION` | `setup_environment.sh` 使用的 Python，默认 3.10.19 |

## 实验追踪

| 命令行 / JSON 字段 | 环境变量 | 默认值 |
|---|---|---|
| `--wandb-mode` / `wandb_mode` | `WANDB_MODE` | `disabled`，也支持 `offline` 和 `online` |
| `--wandb-metrics` / `wandb_metrics` | `NEXTTRANSFORMER_WANDB_METRICS` | `core` 记录主要指标，`full` 记录全部数值字段 |
| `--wandb-project`、`--wandb-entity`、`--wandb-group` | `WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_RUN_GROUP` | 无 |

凭据从 `WANDB_API_KEY` 或挂载的 netrc 文件（`--wandb-netrc`）读取。本地 JSON 不接受密钥。请将本地配置、`.env*` 和追踪日志留在 Git 之外。沿用输出目录在线续训时会复用原运行 ID，训练状态仍按同样的规则验证。详见[训练指标](../metrics/)。

## 其他环境变量

使用环境变量设置 `CUDA_VISIBLE_DEVICES`、`OMP_NUM_THREADS`、NCCL 和代理。节点启动脚本读取 `NNODES`、`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`，从主机发起多机任务的命令见[快速开始](../../getting-started/quick-start/)。若设有 `NEXTTRANSFORMER_JOB_ID` 或 `SLURM_JOB_ID`，运行记录会保存作业 ID。`NEXTTRANSFORMER_METHOD_PATHS` 可为研究运行指定额外的插件目录。
