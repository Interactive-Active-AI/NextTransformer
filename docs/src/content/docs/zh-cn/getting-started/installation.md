---
title: 安装
description: 安装用于 GPU 训练或 CPU 开发的 NextTransformer。
---

NextTransformer 是一个 Python 包（`nexttransformer`），以及它所基于的 nanochat 运行时。项目使用
[uv](https://docs.astral.sh/uv/) 和锁定的环境（`uv.lock`），因此每台机器解析出的版本都相同。

**要求：** 训练需要带 NVIDIA GPU 的 Linux（单机 8 卡，或跨节点 16 / 32 卡）、Python 3.10–3.13 和 uv。纯 CPU
机器可以构建和测试模型，但不能训练。

## GPU 训练环境

```bash
git clone https://github.com/Interactive-Active-AI/NextTransformer.git
cd NextTransformer
uv sync --frozen --extra gpu --extra upstream --extra tracking
source .venv/bin/activate
# CUDA 上的线性注意力方法使用按哈希锁定的 flash-linear-attention kernel：
uv pip install --no-deps --require-hashes -r requirements/linear.txt
```

`gpu` extra 安装 PyTorch **2.9.1**（CUDA 12.8），这是所有成本证明记录的构建版本。正式运行需要它；研究运行可以使用任意构建。
`upstream` 增加可选的 FlashAttention-3 kernel hub 和数据集工具，`tracking` 增加 Weights & Biases（默认关闭，需要时再开启）。

如果集群上的每个作业都需要自包含的环境，`launch/setup_environment.sh` 会用同一个锁文件在持久化的运行时目录下创建环境：

```bash
bash launch/setup_environment.sh --site-config site.local.json --tracking --upstream
```

## CPU 开发环境

```bash
uv sync --frozen --extra cpu
export CUDA_VISIBLE_DEVICES= NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1
uv run python -m pytest -q -n auto
```

测试套件会以调试尺寸构建每个已注册的方法，检查因果性、梯度、检查点往返、成本账本和已发布的记录。慢速测试（全尺寸成本审计、多
rank 数据布局）用 `pytest -m slow` 运行。

## 下一步

- 在每台机器或每个共享文件系统上[准备一次数据](../data/)。
- 按照[快速开始](../quick-start/)在单机或多机上训练。
