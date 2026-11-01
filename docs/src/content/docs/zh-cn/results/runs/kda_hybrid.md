---
title: Hybrid KDA + MLA-NoPE
description: 去除值嵌入的单段混合架构训练，以及与 KDA 和 MLA-NoPE 的对比。
---

Hybrid KDA + MLA-NoPE（`kda_hybrid`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.673751**，独立终验 BPB 为 **0.671160**（训练集为 0.672881），CORE ×100 为 **31.5418**。全部 24 层均移除值嵌入（VE）及其门控，FFN 宽度保持与独立 KDA 相同的 6016，总参数量为 **793,542,986**。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Dense Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,688,180 | 0.667172 | 0.664460 | 33.0952 | — |
| KDA（`kda`） | 24 | 6016 | 1,412,126,344 | 0.667123 | 0.664387 | 31.0606 | -0.00005 / -2.03 |
| MLA-NoPE + QK-norm（`mla_nope_qknorm`） | 24 | 6272 | 1,303,914,612 | 0.685620 | 0.682963 | 27.6536 | +0.01845 / -5.44 |
| Hybrid KDA + MLA-NoPE（`kda_hybrid`） | 24 | 6016 | 793,542,986 | 0.673751 | 0.671160 | 31.5418 | +0.00658 / -1.55 |

与独立 KDA 相比，将每四个块中的最后一个（1-based 第 4、8、12、16、20、24 层）替换为完整因果 MLA-NoPE 并移除全部值嵌入后，参数量减少 618.6M（43.8%），周期验证 BPB 高 0.00663，CORE ×100 提高 0.48 分。在 CORE 子任务上，该混合架构在阅读理解与算法推理任务上高于独立 KDA（SQuAD +6.92、BIG-bench CS Algorithms +5.91、CoQA +5.75、AGIEval LSAT-AR +4.89、Winograd +4.40、CommonsenseQA +24.26 中心化分），而在 BoolQ（−11.67）和 WinoGrande（−6.63）上较低。与独立 MLA-NoPE + QK-norm 相比，周期验证 BPB 降低 0.01187，CORE 提高 3.89 分。

正式运行在单段内连续完成，采用 8 个 rank、每卡 microbatch 8、梯度累积 8 次（父提交 `fab8c032`，含启动环境补丁 `08668a33`，科学指纹 `e1b5a94c`）。KDA 层使用 FLA `chunk64` 后端，MLA-NoPE 层使用禁用 FlashAttention 的确定性 SDPA。

累计训练计算量为 **4.692309 GFLOPs/token**，高于 SSSL 锚点 0.426%，每个记录步均与成本凭证一致，无预算异常。只读审计覆盖全部 189 次周期验证、六次完整 CORE-22 评测和 94 个已校验检查点。训练全程梯度范数平稳且为有限值：第 500 步之后的中位数为 0.27，最大值为 0.71。

8 个 rank 下 12 步目标全局批量的 smoke 与新进程续训一致性验收已通过（`weights_equal`、`losses_equal`、`all_rank_states_equal`）。独立的 512 步 pilot 与启动补丁源码差异未留存归档，最终验收仍待维护者复核。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/kda_hybrid.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/kda_hybrid.json)
