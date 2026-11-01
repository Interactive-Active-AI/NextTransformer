---
title: 超连接（HC）
description: 源码不变的四段训练，以及残差族的整体比较。
---

HC 在第 46,875 步完成 49.152B tokens 的训练。终点周期验证 BPB 为 **0.664400**，独立终验 BPB 为 **0.661683**，CORE ×100 为 **35.0943**，CORE 仅次于 mHC。与 Standard Residual 相比，周期 BPB 降低 0.00956，CORE 提高 3.14 分。与全注意力 Dense 相比，BPB 降低 0.00277，CORE 提高 2.00 分。三者均使用种子 42，HC 与两个对照的 FFN 宽度均为 6144。

| 纳入曲线的更新 | 恢复自 | 训练源码 |
|---|---|---|
| 1–29,000 | — | `4dd6ebb` |
| 29,001–33,000 | 第 29,000 步检查点 | `4dd6ebb` |
| 33,001–38,500 | 第 33,000 步检查点 | `4dd6ebb` |
| 38,501–46,875 | 第 38,500 步检查点 | `4dd6ebb` |

第一段因 mHC 出现非有限梯度、共享训练作业被终止而停止。之后又中断两次，每次都从最新的完整检查点恢复。四段的源码、配置、成本凭证和批量设置完全相同：8 个 rank、每卡 microbatch 4、梯度累积 16 次。曲线剔除了被回滚的尾段。

累计训练计算量为 **4.698483 GFLOPs/token**，高于 SSSL 锚点 0.558%，每个记录步都与成本凭证一致。审计覆盖 189 次验证和六次完整 CORE-22。

梯度范数始终为有限值，但比 Standard Residual 更大、尖峰更多：第 500 步之后的中位数为 0.90，Standard Residual 为 0.25。第 1,900 步出现一次孤立尖峰 1,824，第 24,400–24,750 步之间有几次 30–72 的尖峰，loss 走势未受影响。

HC 发布后，残差族已全部完成。Δ 为相对 Standard Residual 的周期 BPB / CORE 分差：

| 运行 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | Δ |
|---|---:|---:|---:|---:|
| Standard Residual | 0.673959 | 0.671214 | 31.9513 | — |
| HC | 0.664400 | 0.661683 | 35.0943 | -0.00956 / +3.14 |
| mHC | 0.661665 | 0.658908 | 35.3278 | -0.01229 / +3.38 |
| Block AttnRes | 0.662963 | 0.660217 | 33.9971 | -0.01100 / +2.05 |

mHC 的 FFN 宽度为 6016，其余为 6144。以上均为单种子观测，不构成统计意义上的排名。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/hc.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/hc.json)
