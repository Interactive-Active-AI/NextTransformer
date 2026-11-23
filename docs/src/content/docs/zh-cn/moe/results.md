---
title: MoE 结果
description: MoE 各赛道上已发布的正式运行，含 MoE-CORE、验证 BPB 和逐任务得分。
---

MoE 赛道上每个已发布的正式运行都在这里。本页由 `python docs/scripts/generate.py` 从 `moe/results/` 生成。

MoE-CORE 及其短任务、长任务两半和每一项任务的得分都按 ×100 显示。Δ 列是与本赛道 baseline 的差值。方法名链接到运行记录，记录里写明了准入它的审计文件。相对吞吐是 `moe.throughput` 在同一节点上与 baseline 前后脚测得的比值，baseline 自身是基准。

验证 BPB 取的是训练结束后那次完整验证，读六个数据源的全部行，越低越好；训练途中每 500 步的那次验证只读每个数据源的固定抽样，数值更低，两者不能直接比较。最后那次 MoE-CORE 评测和收尾的完整评测是两趟独立的评测，所以下面表格的最后一行与上面的总分会有零点零几的出入。

## d768 · 240 TPP

53,686 次更新 · 14.07B 输入 token · 每个激活参数 240 个 token · 总参数 2.00B，激活参数 213M。

| 方法 | MoE-CORE ↑ | Δ | 验证 BPB ↓ | Δ | 短任务 | 长任务 | 相对吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **[MoE Baseline](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/results/d768-240tpp/baseline.json)** | 36.82 | — | 0.7708 | — | 20.40 | 77.87 | — |

### 各任务得分

| 任务 | 类别 | MoE Baseline |
|---|---|---:|
| ARC-E | 短 | 47.76 |
| ARC-C | 短 | 9.22 |
| HellaSwag | 短 | 22.62 |
| PIQA | 短 | 37.65 |
| WinoGrande | 短 | 1.97 |
| CSQA | 短 | 7.04 |
| OBQA | 短 | 12.53 |
| LAMBADA | 短 | 43.82 |
| SQuAD | 短 | 15.81 |
| TriviaQA | 短 | 5.59 |
| RULER S-NIAH-1 2K | 长 | 100.00 |
| RULER S-NIAH-1 4K | 长 | 84.20 |
| FDA | 长 | 66.61 |
| SWDE | 长 | 60.67 |

### 训练过程中的 MoE-CORE

| 更新 | MoE Baseline |
|---:|---:|
| 12,726 | 29.41 |
| 25,451 | 31.52 |
| 38,177 | 33.96 |
| 53,686 | 36.83 |

## d1024 · 240 TPP

本赛道尚无已发布的运行。

## d768

本赛道尚无已发布的运行。

## d1024

本赛道尚无已发布的运行。
