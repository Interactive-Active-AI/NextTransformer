---
title: 复现已发布结果
description: 按原协议重新训练，使用运行记录核对结果。
---

每条已发布记录都包含方法、配置和指纹。选用显存足以容纳该模型的 GPU，然后按以下步骤复现：

1. [安装](../../getting-started/installation/) PyTorch 2.9.1 环境并[准备数据](../../getting-started/data/)。训练器会核对分词器和验证分片的哈希。
2. 运行 `python -m nexttransformer.protocol.fingerprint --methods <method>`，确认实现与算力凭证一致。
3. 参照[快速开始](../../getting-started/quick-start/)，使用已准入的配置、`--formal` 和新的输出目录训练。登记的 8、16、32 卡布局使用相同的数据顺序。
4. 使用 `python -m nexttransformer.audit_run` 审计运行，随后将第 46,875 次更新的验证 BPB 和 CORE 与 `results/runs/<method>.json` 比较。

这些运行只使用单个随机种子。数据、算力和评测时间表应保持一致；不同 GPU、内核及 rank 数之间通常比较数值误差，不要求逐位相同。部分结果使用冻结的源码版本，差异记录在 `comparisonNote` 和[运行说明](../../results/runs/)中。欢迎提交能验证或质疑已发布结果的复现记录。
