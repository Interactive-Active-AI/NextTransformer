---
title: 训练
description: 选择训练入口，设置运行模式和批量，恢复检查点，并重新核验配置与算力。
---

## 选择训练入口

| 入口 | 适用范围 |
|---|---|
| `python -m nexttransformer.train`（`launch/train.sh`） | 已注册方法，支持 8、16、32 个 rank。包含分块线性注意力、稀疏训练内核（`--sparse-backend kernel`）、Ouro 的门控微调阶段和分段续训。CORE 评测始终走参考实现。 |
| `python -m nexttransformer.native.train`（`launch/native_train.sh`） | SSSL、MoR、LoopFormer、Circle Loop 的历史原生实现，使用矩阵 FLOPs 估算。 |

训练器会校验分词器的指纹。默认的**研究模式**会记录算力超出预算、代码或 PyTorch 版本与凭证不符等情况，研究运行不能导出为正式结果。发布结果请用 **`--formal`** 开启准入检查。续训时沿用首次启动的模式。多机启动见[快速开始](../../getting-started/quick-start/)。

```bash
# 使用准备好的资源进行两步冒烟测试
launch/train.sh --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn-smoke --micro-batch 8 --smoke-steps 2

# 使用 kernel 后端验证稀疏方法
launch/train.sh --config configs/sparse/qsa.json --site-config site.local.json --sparse-backend kernel \
  --output runs/qsa-smoke --micro-batch 8 --smoke-steps 2
```

| 参数 | 含义 |
|---|---|
| `--config` | 已通过准入的模型配置 |
| `--output` | 正式运行首次启动使用新目录，续训沿用原目录 |
| `--micro-batch` | 每张 GPU 每次处理多少行，每行 2,048 tokens。梯度累积补足固定的全局批量 |
| `--formal` / `--research` | 强制执行准入检查，或只记录检查结果（默认） |
| `--resume auto` | 从最近的完整检查点继续，无检查点时重新开始 |
| `--max-wall-hours` | 在作业时限之前保存检查点并结束本段运行 |
| `--sparse-backend {reference,kernel}` | 训练时的稀疏注意力实现 |
| `--deterministic-attention` | 在确定性续训检查中禁用 FlashAttention-3 和 SDPA flash |
| `--smoke-steps`、`--smoke-val-tokens`、`--smoke-core-per-task` | 限定冒烟测试规模；测试不计入正式结果 |
| `--log-every` | 训练日志间隔，默认每 10 次更新 |

CUDA 上的分块线性后端使用 `requirements/linear.txt` 锁定的依赖，并设置 `NANOCHAT_DTYPE=bfloat16`、`FLA_DISABLE_BACKEND_DISPATCH=1`。`launch/train.sh` 与 `launch/slurm.sbatch` 会设置后者；用其他方式（例如 `accelerate launch`）启动训练器时需要自行 export。优化器编译还需要 `setuptools`。六种已注册的混合架构共用同一训练入口，配置位于 `configs/hybrid/`。

## 续训与恢复

`--resume auto` 会检查源码、软件和稀疏后端、配置、随机数状态、训练阶段与累计预算是否连续。应在启动该运行时使用的源码版本上续训。

恢复包含三个层次：能否读回训练状态，数据顺序和计数是否准确衔接，数值轨迹是否与不中断训练一致。成功加载检查点只能验证第一层。部分路径通过了短程逐位一致性检查，真实长程续训可能只在舍入精度内一致。验证新路径时，记录源码版本、配置与凭证、分词器和数据指纹、PyTorch/CUDA/NCCL、内核、rank 和微批量布局、编译器及缓存策略。在预先约定的容差下，将连续运行与“保存、退出、新进程续训”两条短程轨迹进行比较。

## 重新审计算力

在目标 PyTorch 环境中，从仓库根目录运行，每次审计使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 \
python -m nexttransformer.protocol.calibrate --workers 4 --reuse-configs configs \
  --cache .agent-work/cost-audit/cache --output .agent-work/cost-audit/configs \
  --results .agent-work/cost-audit/proofs
```

此检查在 CPU 上执行，不需要训练数据。`--reuse-configs` 固定已有结构；省略它会重新执行已注册的候选配置搜索。历史算力凭证应保留在原目录。

## 验证训练实现

```bash
# 快速测试
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m pytest -q -n auto

# 单节点验证完整尺寸的 BF16 架构，各 rank 分别测试不同模型
NANOCHAT_DTYPE=bfloat16 python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m nexttransformer.checks.validate --mode target --config-dir configs --output .agent-work/full-context.json
```

`nexttransformer.checks.distributed` 检查同步的全局优化器更新，`nexttransformer.checks.integration` 联合检查资源、数据与评测器，`nexttransformer.checks.fla` 用独立的数值参考实现校验线性后端。各命令的 `--help` 列有参数。这里使用随机初始化模型，分数只用于实现检查。
