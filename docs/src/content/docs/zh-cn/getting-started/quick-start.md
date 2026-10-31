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

## 3. 多机

协议登记了 16 卡和 32 卡的布局，它们与 8 卡运行的数据顺序完全一致。在每台机器上运行同一条命令，只改 `NODE_RANK`：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 launch/train.sh \
  --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto
# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=node0 launch/train.sh \
  --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto
```

`MASTER_ADDR` 是所有节点都能访问到的节点 0 的主机名或地址；`MASTER_PORT` 默认为 29500。四台机器时，设置
`NNODES=4`、`NODE_RANK=0..3`。输出目录必须位于所有节点共享的文件系统上。

在 SLURM 上，`launch/slurm.sbatch` 以每节点一个任务的方式完成同样的事：

```bash
sbatch --nodes=4 launch/slurm.sbatch --config configs/linear/gdn.json --site-config site.local.json \
  --output runs/gdn --micro-batch 16 --formal --resume auto --max-wall-hours 23.5
```

`--max-wall-hours` 会在调度器的时间限制之前，于某个检查点处干净地结束当前分段；重新提交同一条命令即可继续。

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
