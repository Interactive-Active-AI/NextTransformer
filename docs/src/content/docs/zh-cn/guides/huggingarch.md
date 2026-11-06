---
title: HuggingArch 架构分析
description: 分析固定模型 spec，输出每个模型的 JSON 结果。
---

[HuggingArch](https://github.com/shenh10/HuggingArch) 根据 YAML 架构 spec 估算参数量、推理 FLOPs、
权重字节数和分层开销。本接口提供 NextTransformer 的配置、forward 源码和固定 spec。
日常分析不需要 agent、权重或训练数据。

## 运行分析

在仓库根目录执行：

```bash
python -m nexttransformer.integrations.huggingarch export \
  --methods dense gdn ut_act --output .agent-work/huggingarch
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch register \
  --bundle .agent-work/huggingarch
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch analyze \
  --bundle .agent-work/huggingarch --batch-size 1 --seq-len 2048 --dtype bfloat16
```

| 命令 | 输入 | 输出 |
| --- | --- | --- |
| `export` | 方法名、仓库配置和源码 | 本地导出目录，包含适用的固定 spec |
| `register` | 导出目录 | 注册 HuggingArch 自定义模型并安装 spec |
| `analyze` | 导出目录、batch size、序列长度、dtype | 每个模型的 `analyze.json` 和汇总 `report.json` |

导出目录必须为空。省略 `--methods` 会导出全部 37 个配置；注册和分析也可以用此选项选择子集。
分析默认 batch size 为 1，序列长度取模型配置，精度为 BF16，每模型超时 120 秒（`--timeout`）。
失败按模型记录；只要有选中的模型未完成，命令就会在写入其他结果后以非零状态退出。

`tools/huggingarch/pyproject.toml` 和 `uv.lock` 固定上游依赖。由于 Python 和 Hugging Face Hub
版本要求与训练依赖冲突，工具使用独立的 Python 3.13 环境和 CPU PyTorch，由 `uv run` 自动安装。
开发上游接口时，可用 `--huggingarch-root` 和 `--huggingarch-python` 覆盖源码目录及解释器。

适配器在 worker 内调用 `backend.custom_model.register` 和 `backend.analyzer.analyze_model`，
将自定义 spec 扫描限制到当前模型，避免同名 `spec.yaml` 导致选错架构图；序列化时移除非 JSON 解析对象。

## 读取结果

导出目录包含 `manifest.json`、`report.json`，以及各模型的子目录。子目录中有 `config.json`、
`nexttransformer-config.json`、`forward.py`、适用的 `spec.yaml` 和 `spec-review.json`，
以及操作结果 `register.json`、`analyze.json`。

`report.json` 顶层为 `schema_version`、`notice`、`models`。每个模型记录包含：

| 字段 | 内容 |
| --- | --- |
| `model` | 方法名、固定 ID（如 `custom:nextt-dense`）、配置路径、权重精度和训练成本证据 |
| `result` | 状态、分析输入和依赖版本；成功时有 `analysis`，失败时有 `error` |
| `spec_review` | 适用配置、检查结果、范围和限制；无对应记录时为空 |

`result.analysis.summary` 包含 `total_params`、`total_flops`、`total_flops_gflops`、`weight_bytes`；
`result.analysis.layer_groups` 包含分层结果。推理 FLOPs 与 `model.training_flops_per_token`
分开统计。分析前，`result` 是注册/生成结果，或 `{"status": "exported"}`。
运行 `report --bundle .agent-work/huggingarch` 可从本地文件重建汇总。

`export --weight-dtype` 设置存储权重精度假设，默认 BF16；`analyze --dtype` 控制分析请求，
不会改写存储权重精度假设。

## Spec 覆盖与维护

已有 35 份固定 spec：全部 33 个注册方法，以及 `native-dense`、`native-loopformer`。
`native-mor` 和 `native-circle` 返回 `unsupported_architecture`，因为压缩 token 轴依赖运行时
存活 token 和 batch 内最大存活数量。固定 spec 覆盖 2–2048 token 的完整序列 prefill；
超出范围返回 `unsupported_settings`。

`tools/huggingarch/specs/catalog.json` 记录适用配置和逐模型限制。导出时直接比较配置值；源码修改需要同步审阅、更新 spec，catalog 不会
自动检测源码变化。注册拒绝与缓存中的配置、源码或 spec 冲突的内容。需要替换时，通过 HuggingArch
模型管理删除对应自定义模型，再注册已审阅的新目录。

已有 spec 对照配置的 PyTorch meta 模型，在 `(batch, sequence)` 为 `(1,16)`、`(2,16)`、`(1,2048)`
时通过参数量、张量名称/形状/重复次数及图几何检查。这不代表数值输出等价、实测延迟或峰值显存。
未提供 checkpoint，因此 `checkpoint_validated` 保持 false，上游发布校验返回 `CheckpointUnavailable`。

## 编写新 spec

先注册模型以获得 HuggingArch 的逐模型 `prompt.md`。编写 agent 应读取该 prompt、上游 schema 和
审阅要求、模型配置及 forward 源码。审阅和检查 YAML 后，与适用配置和限制一同提交。
HuggingArch 计算的是 spec，不会自动从任意 `torch.nn.Module` 提取精确计算图。

也可以调用上游自带的 agent 生成和发布流程：

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch --methods dense --agent-cli codex
```

`generate` 复用已有 spec；否则通过已安装并认证的 `codex`、`claude` 或 `kimi` CLI 调用
`backend.custom_model.run_spec_generation`，记录 `generate.json`、`generate.log`，默认超时 600 秒。
校验失败或未发布 spec 时返回 `generation_failed`，草稿不会升级为正式 spec。
上面的 checkpoint 要求可能阻止纯架构导出的发布。

`export --smoke --methods mha_no_ve` 可导出小型 debug 配置，使用独立的 `-smoke` 名称，
不附带正式训练成本证据，需要单独编写 spec。
