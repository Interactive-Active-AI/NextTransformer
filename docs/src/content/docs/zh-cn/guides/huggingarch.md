---
title: HuggingArch 架构分析
description: 通过可选依赖分析固定模型 spec，输出每个模型的 JSON 结果。
---

[HuggingArch](https://github.com/shenh10/HuggingArch) 根据声明式架构 spec 估算参数量、推理 FLOPs、
权重字节数和分层开销。本接口导出 NextTransformer 的实际配置和 forward 源码，提供固定 YAML spec，
输出 JSON 分析结果，无需启动服务。

## 依赖

`tools/huggingarch/pyproject.toml` 直接声明 HuggingArch Git 依赖，固定到已测试的版本
`facbbbc96856e558b9084a94a5070b2e2c86cf0b`，传递依赖由 `uv.lock` 固定。
在仓库根目录通过 `uv run --project tools/huggingarch --frozen` 安装并调用即可，无需手动克隆上游。

工具环境使用 Python 3.13 和 CPU PyTorch 2.9.1。训练环境支持 Python 3.10–3.13，并固定
`huggingface-hub==1.26.0`；上游的 Transformers 依赖要求 `huggingface-hub>=1.31.0`。
因此工具使用独立环境，避免改变训练依赖。开发上游接口时，可以通过 `--huggingarch-root`
和 `--huggingarch-python` 覆盖源码目录及解释器。

## 接口

| 操作 | 输入 | 输出 |
| --- | --- | --- |
| `export` | 方法名、仓库中的模型配置和源码 | 包含配置、源码、固定 spec 和清单的本地目录 |
| `register` | 导出目录 | 注册 HuggingArch 自定义模型并安装已有 spec |
| `analyze` | 导出目录、batch size、序列长度、dtype | 每个模型的 `analyze.json` 和汇总 `report.json` |

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

导出时省略 `--methods` 会包含全部 33 个注册方法和 4 个历史 native 配置。
`register` 和 `analyze` 也可以用 `--methods` 选择子集。导出目录必须为空，后续操作复用该目录。
默认 batch size 为 1，序列长度取各模型的配置，dtype 为 BF16。
每个模型分析默认超时 120 秒，可通过 `--timeout` 设置。某个模型失败不会阻止其他模型输出结果；
只要本次选择中有未完成的模型，命令就会以非零状态退出。

实际依赖边界是上游的两个函数：

```python
from backend.custom_model import register
from backend.analyzer import analyze_model

register(name="nextt-dense", config_path="config.json", forward_paths=["forward.py"])
# The adapter installs spec.yaml in this custom model's cache.
result = analyze_model(
    model_id="custom:nextt-dense", batch_size=1, seq_len=2048,
    dtype="bfloat16", trust_remote_code=False,
)
```

上面的代码用于说明 API。完整调用应走本仓库 CLI：它在独立 worker 内调用上游，并移除返回值中
不能序列化为 JSON 的解析对象。它还将自定义 spec 扫描限制到当前模型，解决固定上游版本中
多个文件都叫 `spec.yaml` 时可能选错模型图的问题。

## JSON 输出

导出目录包含 `manifest.json` 和汇总 `report.json`。每个模型子目录包含 `config.json`、
原始 `nexttransformer-config.json`、供 spec 编写使用的 `forward.py`，以及已有的 `spec.yaml`
和 `spec-review.json`。注册及分析后分别写入 `register.json`、`analyze.json`。

`report.json` 顶层为 `schema_version`、`notice`、`models`。每个模型记录包含：

- `model`：方法名、固定名称（例如 `custom:nextt-dense`）、配置路径、权重精度假设和已有训练成本证据。
- `result`：状态 `status`、分析输入 `settings`、`checkpoint_validated: false`、依赖版本信息；成功时
  包含上游 `analysis`，失败时包含 `error`。分析前为最近的注册或生成结果，或者 `{"status": "exported"}`。
- `spec_review`：固定 spec 的适用配置、分析范围、检查结果及限制；没有对应记录时为空。

`result.analysis.summary` 中保留上游的 `total_params`、`total_flops`、`total_flops_gflops`、
`weight_bytes`；`result.analysis.layer_groups` 保留分层结果。推理 FLOPs 与
`model.training_flops_per_token` 是不同指标，缺失结果不会被填成零。

运行 `report --bundle .agent-work/huggingarch` 可以仅使用本地 JSON 重建汇总，不需要 HuggingArch。
本接口不生成 HTML，也不计算源码、prompt 或 spec 的内容哈希。原始训练配置和成本证明保留已有协议字段。

## 固定 spec 与更新

`tools/huggingarch/specs/` 提供 35 份固定 YAML：全部 33 个注册方法，以及 `native-dense`、
`native-loopformer`。它们由 GPT-6.1 Sol 子 agent 按照 HuggingArch 自己的逐模型 `prompt.md`、
共享 schema 和双向源码检查要求编写。日常注册和分析不会调用 agent。

catalog 按方法名记录 spec 及明确的适用配置。导出时直接比较配置值，修改层数、宽度或其他架构选项后，
不会误用原 spec。权重 dtype 属于估算假设，不改变架构图选择。模型使用固定名称，不带内容哈希。
源码兼容性由代码 review 维护；修改实现时应同步更新、检查 spec。注册会直接比较已有缓存的配置、
源码和 spec 文本，并拒绝冲突内容。如果确实需要替换，请通过 HuggingArch 的自定义模型管理删除
对应缓存模型，再注册已审阅的新目录。

`export --weight-dtype float32` 可以修改存储权重精度假设，默认 BF16；`analyze --dtype` 传递的是
上游分析请求，不会改写存储权重精度假设。同一名称下的不同导出配置需要显式替换缓存。

`native-mor` 和 `native-circle` 返回 `unsupported_architecture`：其压缩 token 轴依赖运行时
存活 token 和 batch 内最大存活数量，当前静态图无法准确表示。catalog 记录了具体原因，不使用
dense 图替代。已有 spec 覆盖完整序列 prefill，支持 2–2048 token；超出范围返回 `unsupported_settings`。

## 可选的 spec 生成

对于缺少 spec 的模型，先注册以获得上游生成的逐模型 prompt。编写 agent 必须读取这个 prompt、
上游 schema 和审阅要求、模型配置及实际源码；审阅和检查 YAML 后，再与适用配置及限制一同提交。
也可以调用上游自带的 agent 生成和发布流程：

```bash
uv run --project tools/huggingarch --frozen \
  python -m nexttransformer.integrations.huggingarch generate \
  --bundle .agent-work/huggingarch --methods dense --agent-cli codex
```

已有 spec 会直接复用。否则调用 `backend.custom_model.run_spec_generation`，使用已安装和认证的
`codex`、`claude` 或 `kimi` CLI，默认超时 600 秒，记录 `generate.json` 和 `generate.log`。
校验失败或没有发布 spec 时返回 `generation_failed`，不会将草稿升级为正式 spec。
上游发布校验可能需要 checkpoint headers，而本接口的架构导出不提供这些数据。
`export --smoke --methods mha_no_ve` 会导出小型 debug 配置，使用独立的 `-smoke` 名称，
不附带正式训练成本证据，并且需要另行编写适用的 spec。

## 正确性范围

已有 spec 对照实际配置的 PyTorch meta 模型，通过了参数量、张量名称/形状/重复次数检查，
并在 `(batch, sequence)` 为 `(1,16)`、`(2,16)`、`(1,2048)` 时通过图几何检查。
正常分析不需要权重或训练数据；HuggingArch 计算的是 spec，不是自动从任意 `torch.nn.Module`
恢复完全准确的计算图。

这些检查不代表数值输出等价、checkpoint 校验通过，也不代表测得了延迟或峰值显存。
未提供 checkpoint，上游发布校验返回 `CheckpointUnavailable`。各模型记录中说明了动态执行假设、
自定义算子开销、精度和缓存估算限制。训练协议、模型实现及校准证明保持不变。
