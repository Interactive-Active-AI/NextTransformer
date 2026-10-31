---
title: 仓库结构
description: 代码、配置、结果和证据分别放在哪里。
---

| 路径 | 内容 |
|---|---|
| `nexttransformer/models/` | 内置架构、注册表和共享外壳 |
| `nexttransformer/methods/` | 贡献的方法，每个方法一个插件包 |
| `nexttransformer/protocol/` | 协议规范、成本账本、校准、准入和指纹 |
| `nexttransformer/train.py`、`evaluate.py`、`generate.py`、`prepare.py` | 命令行入口 |
| `nexttransformer/audit_run.py`、`export_run.py` | 审计完成的运行；将其导出为结果记录 |
| `nexttransformer/checks/` | 有界的 GPU 检查、改动分类、逐位比对、公开内容扫描 |
| `nexttransformer/arena.py` | 竞技场披露卡片，以及竞技场 pull request 的改动范围 |
| `nanochat/` | 锁定的上游运行时（数据加载、评测、优化器） |
| `configs/<family>/<method>.json` | 已准入的科学配置 |
| `evidence/costs/` | 成本证明；`evidence/audits/` 存放已发布运行的累计审计 |
| `results/` | 运行记录、模型卡片和竞技场卡片；网站由它们构建 |
| `launch/` | 启动器：`train.sh`（torchrun，单机或多机）、`slurm.sbatch`、环境搭建 |
| `tests/` | 测试套件；`tests/plugins/` 中有示例插件 |
| `website/` | 观测站（Vite + React） |
| `docs/` | 本文档（Starlight），中英文 |

科学配置中永远不包含机器路径：数据、缓存和输出来自命令行选项、本地站点文件（`*.local.json`，被 git 忽略）或环境变量。
