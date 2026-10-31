---
title: 校准与准入
description: 如何选择方法的结构以符合共同预算，以及正式运行前准入要检查什么。
---

:::note
协议以[英文版](../../../protocol/admission/)为准；中文版仅便于阅读。
:::

## 阈值

| 检查 | 阈值 |
|---|---|
| 预检（经过审计的配置） | 与 SSSL 锚点相差 **±1%** 以内 |
| 运行结束时的累计算力 | **±2%** 以内 |
| 训练中的滚动偏差 | 超过 **3%** 时记为异常 |

阈值在任何运行开始之前就已冻结，不会为某个方法放宽。

## 校准

```bash
python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/cost-cache \
    --output configs --results evidence/costs/<proof-set>
```

新的结构从 24 层、FFN 宽度 6144 开始，在登记的布局内搜索 **16–32 层**。如有需要，FFN 宽度在 **[3d, 5d]** 范围内以 **128**
的倍数调整。在相差 1% 以内的候选中，优先选择 FFN 最接近 6144 的，其次是绝对成本误差最小的，再次是深度最接近 24 的，最后是更浅的。
若单调性假设不成立，会公开完整的候选表。已有配置保留其记录在案的选择依据。

不允许用无用计算填充、用更少的输入 token 训练、提前停止、挑选随机种子，或用正式 CORE 做选择。如果网格无法满足预算，该配置记为
`calibration_pending`，并预先登记修订后的网格。

校准把配置写入 `configs/<family>/<method>.json`，把证明写入 `evidence/costs/`，与 SSSL 锚点证明放在一起。需按组件报告参数量（嵌入和
value embedding、mixer、FFN、路由器和残差映射）、物理块数和展开后的调用次数、前向、反向、辅助和优化器成本，以及各阶段权重。

## 准入

`nexttransformer.protocol.admission.admit_config` 会拒绝以下配置：证明已过期或不匹配、由不同的 PyTorch 构建生成、模型或协议哈希已改变、
包含未知算子或缺失阶段，或与锚点相差超过 1%。正式运行（`python -m nexttransformer.train --formal`）需要准入；研究模式只记录这些问题，其运行永远不能发布。

训练过程中，`BudgetState` 跟踪优化器步数、token 数、累计算力和已完成的评测事件；不完整的运行不可能是 `matched`。

证明记录的 PyTorch 构建是 **2.9.1+cu128**。审计在 CPU 上运行，因此任何装有该构建的机器都能复现证明；维护者会在该环境中重新审计新方法，然后再准入。
