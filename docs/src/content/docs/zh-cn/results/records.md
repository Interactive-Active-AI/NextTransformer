---
title: 运行记录与模型卡
description: 查询 results/runs 与 results/methods 中各字段的含义。
---

```text
results/
  runs/<id>.json       每个正式运行一份记录，默认以模型卡 ID 命名
  methods/<id>.json    每个方法一张模型卡
  arena/<entry>.json   每个竞技场条目一张披露卡
```

## 运行记录

`python -m nexttransformer.export_run` 从完成的运行目录及其审计生成记录。CI 用 `python -m nexttransformer.export_run --check` 验证全部记录。新结果需要满足：189 次验证与 6 次完整 CORE-22 评测无缺失，各任务得分能复算出 CORE 汇总值，累计审计为 `matched`，所有训练段均为正式模式，并且使用了锁定的分词器和协议。

| 字段 | 含义 |
|---|---|
| `id`、`methodId`、`cohort`、`name` | 运行 ID、注册方法（SSSL 保留 `dense`）、核算分组、显示名称 |
| `status`、`step`、`complete` | `finished` 或 `partial`、最终步数、是否完成协议并通过验收 |
| `val`、`core` | 形如 `{step, value}` 的测量点，包括定期验证 BPB，以及使用 0–1 量纲的 CORE |
| `taskEvaluations` | 每次 CORE 评测中 22 个任务各自的原始准确率 `raw` 和中心化得分 `centered` |
| `finalBpb` | 独立测得的最终验证 BPB，与定期曲线分开报告 |
| `trainingCompute` | 累计训练 FLOPs、每个输入 token 的平均 GFLOPs 及核算口径 |
| `proof` | 协议、模型、分词器、数据和源码的指纹 |
| `source`、`url` | 记录来源及其在仓库中的证据 |
| `comparisonNote` | 有必要说明来源差异时，连同运行一起展示的说明 |
| `baselineAcceptance` | 仅出现在已验收的 `dense_full` 基线记录 |

`cohort` 决定检查采用的核算口径。`reference-v1` 使用算子级账本，`native-v3` 使用历史矩阵 FLOPs 估算，`frozen-replay-20261019` 保留两个运行原有的反向重放核算。这些运行在同一个排行榜展示。文件名默认为模型卡 ID，同一方法的第二次运行可用 `--file NAME` 区分。

## 模型卡

`methods/<id>.json` 记录名称、模型族、简介、论文与代码链接、机制、限制、参数量、块数和配置算力，也指定观测站所画的结构图。模型卡的 `id` 对应运行记录的 `methodId`。`"status": "pending"` 表示已注册且尚未发布运行结果，观测站会标明状态，不纳入排名。
