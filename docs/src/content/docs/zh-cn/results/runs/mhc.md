---
title: 流形约束超连接（mHC）
description: 四段训练、登记过的 Sinkhorn 数值实现变更，以及与 Standard Residual 的配对比较。
---

mHC 在第 46,875 步完成 49.152B tokens 的训练。终点周期验证 BPB 为 **0.661665**，独立终验 BPB 为 **0.658908**，CORE ×100 为 **35.3278**。与 Standard Residual 相比，周期 BPB 降低 0.01229，CORE 提高 3.38 分。与全注意力 Dense 相比，BPB 降低 0.00551，CORE 提高 2.23 分。三者均使用种子 42，mHC 的 FFN 宽度为 6016，两个对照均为 6144。

| 纳入曲线的更新 | Sinkhorn 实现 | 训练源码 |
|---|---|---|
| 1–31,500 | FP32 乘法归一化 | `4dd6ebb` |
| 31,501–35,000 | 对数域归一化 | `51f4560` |
| 35,001–45,000 | 对数域归一化 | `51f4560` |
| 45,001–46,875 | 对数域归一化 | `51f4560` |

首次运行在记录第 31,750 步后因梯度范数非有限而停止。从第 31,500 步检查点恢复时，采用了事先登记的对数域实现。后两次分别从 35,000 和 45,000 步恢复。曲线保留每次恢复后的有效更新，剔除被回滚的尾段。四段的设置均为 8 个 rank、每卡 microbatch 8、梯度累积 8 次。

两版实现每个输入 token 的理论成本相差 6,072 FLOPs。按各段实际步数加总后为 **4.674359 GFLOPs/token**，高于 SSSL 锚点 0.04197%。审计覆盖 189 次验证和六次完整 CORE-22。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mhc.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/mhc.json) · [源码和成本变更登记](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/registrations/mhc-sinkhorn.json)
