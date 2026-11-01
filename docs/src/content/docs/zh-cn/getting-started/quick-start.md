---
title: 快速开始
description: 在单机或多机上，按协议训练一个已注册的方法。
---

本页按完整协议训练一个已注册的方法（例如 Dense Baseline）：49.152B tokens、46,875 次更新。在一台 8 卡的当前主流数据中心
GPU 机器上，约 1.4B 参数的模型需要两到四天，另加几个小时的评测。

开始之前，请先[安装](../installation/) GPU 环境并[准备数据](../data/)。

## 1. 冒烟测试

训练两次更新，检查环境、数据、分词器和模型能否一起工作：

```bash
source launch/env.sh --site-config site.local.json
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full-smoke --micro-batch 8 --smoke-steps 2
```

冒烟运行可以使用任意 GPU 数量（例如 `NPROC_PER_NODE=1`），并且永远不会成为正式结果。

## 2. 单机 8 卡

```bash
launch/train.sh --config configs/baseline/dense_full.json --site-config site.local.json \
  --output runs/dense_full --micro-batch 8 --formal --resume auto
```

`launch/train.sh` 封装了 `torchrun` 和 `python -m nexttransformer.train`。`--micro-batch` 是每张 GPU 上 2,048-token
样本的行数；梯度累积会补足固定的 512 行全局批量，所以选显存放得下的最大 microbatch 即可。`--resume auto`
让命令可以重复执行：中断后再运行一次，就会从最新的完整检查点继续。

## 3. 提交多节点训练

<span id="multi-node"></span>

在共享训练环境中运行一次 `uv pip install accelerate==1.15.0`。各节点需要访问相同路径下的仓库、`site.local.json`、数据与输出目录。在 Slurm 登录节点上，从仓库根目录提交：

```bash
sbatch launch/slurm.sbatch
```

示例默认申请两台各 8 卡的节点，按正式协议训练 GDN，每卡微批量 16，输出到 `runs/gdn`。作业使用 `srun` 在每个节点启动一个 Accelerate 启动器，每个启动器为本机的每张 GPU 启动一个训练进程。节点地址和节点序号由 Slurm 提供，不需要 hostfile 或节点之间的 SSH。分区、账户和时限按集群要求传给 `sbatch`。

要更换模型或节点数，在脚本名之后传入完整的训练参数。例如在四台各 8 卡的节点上训练混合模型：

```bash
sbatch --nodes=4 launch/slurm.sbatch --config configs/hybrid/gdn_hybrid.json --site-config site.local.json \
  --output runs/gdn-hybrid --micro-batch 8 --formal
```

### 其他托管训练平台

若平台会在每个节点执行同一份任务入口，提交**一个**多节点作业，将下面的命令填为入口。把平台提供的节点数、每节点 GPU 数、从 0 开始的节点序号和主节点地址、端口对应到这些变量。示例配置 `launch/accelerate.yaml` 使用 PyTorch DDP，默认两台各 8 卡。以下参数可按实际资源覆盖默认值；`--num_processes` 是整个作业的 GPU 总数。平台负责分发命令，用户无需登录各节点。

```bash
export FLA_DISABLE_BACKEND_DISPATCH=1   # 线性与混合方法走注册的 FLA Triton 路径
accelerate launch --config_file launch/accelerate.yaml \
  --num_machines "$NNODES" --num_processes "$((NNODES * GPUS_PER_NODE))" \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --module nexttransformer.train --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal
```

Accelerate 负责启动进程，训练器继续使用原生 PyTorch DDP 和协议规定的 Muon + AdamW。重新提交时，默认从同一输出目录下最近的完整检查点恢复。

<span id="hostfile"></span>

单独管理 SSH 主机列表时，也可使用 [DeepSpeed 的 hostfile 启动方式](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。上面的调度器模式不依赖 DeepSpeed。

## 4. 评测、审计与导出

训练会在预定步数写出验证集和 CORE 结果，并在结束时给出独立的最终 BPB。单独评测任意检查点：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m nexttransformer.evaluate \
  --checkpoint runs/dense_full/checkpoint_00046875 --output runs/dense_full/evaluation.json \
  --micro-batch 8 --eval bpb,core
```

完成的正式运行用两条命令审计和导出，然后通过 pull request 提交：

```bash
python -m nexttransformer.audit_run runs/dense_full --method dense_full
python -m nexttransformer.export_run runs/dense_full --method dense_full --name "Dense Baseline" \
  --audit evidence/audits/dense_full.json
```

pull request 的流程见[提交结果](../../guides/submit-a-result/)，训练器的全部选项见[训练指南](../../guides/training/)。
