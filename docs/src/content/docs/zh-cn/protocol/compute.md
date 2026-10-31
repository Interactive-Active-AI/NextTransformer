---
title: 算力核算
description: 什么算作训练算力、如何计数，以及每个方法都要对齐的 SSSL 锚点。
---

:::note
协议以[英文版](../../../protocol/compute/)为准；中文版仅便于阅读。
:::

受控的量是**按所有训练阶段加权的、每输入 token 平均理论训练 FLOPs**。它不是参数量、推理深度、墙钟时间，也不是一次前向。

```text
N_input = 49,152,000,000 个输入 token
G       = 每次更新 1,048,576 个输入 token
S       = 46,875 次优化器更新

F_step  = F_forward + F_backward + F_auxiliary + F_recompute + F_optimizer
C_train = sum(F_step) / N_input

C_ref   = 每输入 token 4,672,398,325.44785 FLOPs   （SSSL 锚点）
F_ref   = C_ref × N_input ≈ 2.2965772249241273 × 10^20 FLOPs
相对误差 = C_train / C_ref − 1
```

锚点是经过审计的 SSSL 配置，其证明见
[`evidence/costs/plugin-interface-20261031/dense.json`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/costs/plugin-interface-20261031/dense.json)。
最初几次 native 运行使用的历史“仅矩阵”估计（每 token 4.775 GFLOPs）是另一套核算方式，永远不与本核算混用。

## 计数方式

审计器（`nexttransformer/protocol/compute.py`）以正式尺寸、逻辑上 **512 × 2,048** 的全局批量，用 fake tensor 运行模型。它执行前向、autograd
和任何自定义反向，以及优化器未融合的数学主体，并统计每一个浮点算子。它不分配模型存储，也不读取训练数据。验证和 CORE 单独统计，永远不计入训练预算。

重复的循环、teacher 和 readout 会增加分子，而不会增加 token 分母。

- 先完整计算、再做掩码的候选会被全额计费。权重共享不会让重复调用变成免费。无梯度的前向和多出口的输出头都要计费。
- 方法特有的反向核算永远不能用 `6ND`、参数量、推理退出深度或通用的“3 × 前向”替代。
- 各阶段按实际输入 token 加权。DSA、QSA 和 Seer 使用 512 次 dense burn-in 更新和 46,363 次稀疏更新；Ouro 使用 45,851 次联合更新和 1,024 次仅门控更新。
- 算法算力与 kernel 填充、tile 和冗余执行分开。dense 的掩码 kernel 不能证明稀疏训练节省了算力。必要的算法重算要计费。
- 激活检查点关闭。改变它、模型语义、源码、软件或 kernel 都需要新的审计；之前的证明和已冻结的运行保持不变。

## 算子账本（`reference-aten-math-v2`）

一次融合乘加计为两次运算。浮点数学在各种 dtype 下统一计数。类型转换、数据搬运、索引、整数比较和 top-k，以及通信，都不计入 FLOPs。
非线性函数的单位运算约定是可复现的核算选择，并不代表运行时成本相等。

| 运算 | 计数 |
|---|---|
| matmul / bmm | `2 × batch × M × N × K`；反向统计实际执行的乘法 |
| Conv1d | `2 × 输出元素数 × 每组输入通道数 × kernel 大小`；需要时计入输入梯度和权重梯度 |
| add、sub、mul、div、exp、log、sqrt、rsqrt、sin、cos、tanh、pow、clamp、ReLU | 每个输出元素 1 次；add 的 alpha 不为 1 时多计一次乘法 |
| sigmoid / SiLU / softplus / ELU | 每元素 4 / 5 / 4 / 3 |
| sigmoid / tanh / SiLU / softplus / ELU 的反向 | 每元素 3 / 3 / 8 / 5 / 3 |
| sum / mean / max 归约 | 输入数减输出数；mean 还要对每个输出做一次除法 |
| softmax / log-softmax | 前向每个分数 5 次；反向 4 次 |
| RMSNorm | 实际展开的 pow、mean、rsqrt 和乘法，以及它们的反向；不重复计数 |
| 融合 LayerNorm | 前向 `7N + 5R`；输入梯度 `10N`；每个可选仿射梯度 `2N` |
| L2 范数 | `2N + 输出数` |
| Embedding | 查表为 0；稠密梯度的 scatter 每个梯度元素计一次累加 |
| scatter_add | 每个浮点源元素 1 次 |
| NLL / BCE-with-logits | NLL 前向和反向每个目标 2 次，mean 再加 1；BCE 前向每个 logit 10 次，反向按实际计数 |
| cumsum | 每次扫描 `n − n/L` 次加法 |
| 三角求解 | 单位下三角每列 `n(n − 1)`；非单位再加 `n` 次除法 |
| tril / triu / eye | 0 次浮点运算 |

未登记的浮点算子会被报告为 `unknown` 并导致校准失败；永远不会被悄悄计为 0。

## 注意力

左窗口 `w` 包含当前 key，因此最多允许 `w + 1` 个 key；`causal_edges(T, w)` 精确处理三角形边界（SSSL 的 512 短窗口最多允许 513 个 key）。对于
`C` 条必需的边、Q/K 维度 `d` 和 V 维度 `v`，前向矩阵成本为 `2C(d + v)`，计算全部 Q/K/V 梯度的反向成本为 `4C(d + v)`，另加缩放和 softmax。
冻结的输入只去掉实际不存在的梯度路径。

DSA 统计 `min(K, i + 1)` 个 key；QSA 统计被选中的完整块加上未完成的尾部；MoBA 的 K 包含必选的当前块；NSA 分别统计压缩、选择和局部三个分支。
indexer、teacher 和压缩打分矩阵在掩码之前按完整形状计费。测试会把解析的边数与实际掩码进行比较。

## 优化器

计数会执行锁定版本的 AdamW 和 Muon 数学，包括动量、五次 Polar Express 迭代、MuonEq/Muon+、方差缩减和权重衰减，针对真实可训练参数组的一次逻辑更新。
DDP、分片填充和额外的 microbatch 归约属于执行开销，而不是算法算力。

## 分块与扫描后端

线性注意力按**分块训练算法**计数，而不是按循环扫描或融合 kernel 的指令计数。`nexttransformer/models/linear.py` 中纯 PyTorch 的
`chunk_reference` 实现了标量衰减和 WY/UT gated-delta 算法，反向由 autograd 给出实际计算。

- C × C 的分块 tile 只统计满足 `i ≥ j` 的因果对，即 `C(C + 1)/2`；稠密填充单独报告。
- 审计直接运行 T = 2,048，并用 3、4、5 个分块交叉检查每个算子；从三个分块起结构是仿射的。
- kernel 内部的重放、tile 填充和 BF16/TF32 算术属于执行细节；必需的算法重算仍要计费。
- 基于扫描的计数在 T = 16、32、48 上检查：每个算子都必须满足仿射关系，才能外推到 2,048。

残差族的映射按实际执行计费：Hyper-Connections 和 mHC 的流归一化、动态投影、门控、全部 20 次 Sinkhorn-Knopp 迭代及其反向、流混合和最终的流求和；
Attention Residuals 的 key 归一化、深度 logits、softmax 和加权求和。

## 指纹

每份证明都会对方法运行的源码取指纹（`nexttransformer/protocol/fingerprint.py`）。校准会记录审计方法成本时执行到的仓库文件，以及这些文件导入的仓库模块；
协议、成本账本、校准、指纹和优化器的源码总是包含在内。文件在去掉注释、docstring 和排版后再计算哈希，模型文件中的模块级表只保留该方法及其层
mixer 的条目。插件的整个包都属于该方法的指纹。

修改其他方法的代码，或只修改注释，证明依然有效；修改方法实际运行的任何内容都需要新的审计。`python -m nexttransformer.protocol.fingerprint`
会把每个配置与当前源码比较；已发布的方法不再与其证明一致时，CI 会失败。
