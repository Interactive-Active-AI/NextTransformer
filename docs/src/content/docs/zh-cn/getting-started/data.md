---
title: 数据与分词器
description: 下载锁定版本的 ClimbMix，并准备分词器和评测资源。
---

所有运行读取同一份数据：[ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) 的
`915333b4f8b8684f39aeaafea600fea6f43fb703` 版本，共 6,543 个 Parquet 分片，约 **600 GB**。训练从第一个分片开始按排序读取；
`shard_06542.parquet` 是验证分片，`shard_06541.parquet` 是供研究使用的 dev 分片。

## 1. 指定存储路径

科学配置中永远不包含路径。复制示例站点文件并设置持久化路径：

```bash
cp site.native.example.json site.local.json
```

```json
{
  "runtime_dir": "/path/to/runtime",
  "resources": "/path/to/runtime/resources",
  "upstream_base_dir": "/path/to/runtime/resources",
  "data_dir": "/path/to/datasets/climbmix-400b-shuffle",
  "wandb_mode": "disabled"
}
```

名为 `*.local.json` 的文件会被 git 忽略。每个字段也可以通过命令行（`--data-dir`、`--resources` 等）或环境变量给出；
详见[配置](../../guides/configuration/)。

## 2. 下载锁定版本

```bash
hf download karpathy/climbmix-400b-shuffle --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 --local-dir /path/to/datasets/climbmix-400b-shuffle
```

## 3. 准备资源

```bash
source launch/env.sh --site-config site.local.json
python -m nexttransformer.prepare --site-config site.local.json
```

准备步骤会检查目录中的文件和大小与锁定版本完全一致、验证分片与 Hub 上的校验和一致，然后从原始 Parquet 训练 32,768-token
的分词器及其字节表，并准备锁定的 CORE-22 评测包。训练器会拒绝 SHA-256 与锁定值不同的分词器，因此每台机器最终得到的资源完全相同。
已验证过的 CORE 包可以用 `--existing-core-resources PATH` 复用。

每个共享文件系统只需准备一次，所有运行都指向同一个 `resources` 目录。之后迁移资源时，Parquet 内容、分词器、字节表和 CORE
包都必须保持不变。
