---
title: GSA + QK-norm
description: Gated Slot Attention 冻结版本的正式运行和提交证据。
---

本次运行采用 24 层、64 个槽位，在 SiLU 前执行逐头 QK RMSNorm 并乘以 1.2，递归计算使用长度为 128 的分块和 FP64。种子 42，训练 46,875 步，共 49.152B tokens。终点周期验证 BPB 为 **0.685242**，独立终验 BPB 为 **0.682638**，CORE ×100 为 **28.1720**。累计计算量为 4.689438 GFLOPs/token，高于 SSSL 锚点 0.36468%。

提交材料含 189 个验证点、六次 CORE-22 的逐任务得分，以及贡献者提供的累计审计。可用审计中的哈希标识训练源码版本和指标归档。本次发布范围为实验记录和审计，材料未提供逐段续训日志或该变体的可训练实现。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gsa_qknorm.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/gsa_qknorm.json)
