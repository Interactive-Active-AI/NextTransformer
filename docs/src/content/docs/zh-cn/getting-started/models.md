---
title: 模型 API
description: 在 Python 中构建、加载、评测任意已注册的方法并采样。
---

所有内置方法和插件都通过同一个接口构建。

```python
from nexttransformer.models import build_model, load_model, METHODS, FAMILY
from nexttransformer.models.config import candidate_config, debug_config

model = build_model(debug_config("dsa"))                       # 调试尺寸的已注册方法（CPU 也能跑）
model, saved = load_model("runs/dsa/checkpoint_00046875", "cuda")   # 训练好的检查点
print(len(METHODS), sorted(set(FAMILY.values())))
```

正式使用前要先准入配置；准入会对照代码和 PyTorch 构建检查成本证明：

```python
from nexttransformer.protocol.admission import admit_config
from nexttransformer.models import build_model
from nexttransformer.training.optimizer import build_optimizer

config, proof = admit_config("configs/linear/kda.json")
model = build_model(config, "cuda")
model.set_training_stage(0)   # 在构建优化器和 DDP 之前确定可训练参数
optimizer = build_optimizer(model)
```

## 调用模型

- `model(x, y)` 返回标量训练目标（交叉熵加上已登记的辅助损失）。
- `model(x)` 返回 float32 logits。
- `model(x, y, loss_reduction="none")` 返回不含辅助项的逐 token 交叉熵；BPB 使用它。
- `model.eval()` 选择已登记的评测深度或退出策略。
- `last_lm_loss`、`last_aux_loss`、`last_stats` 和 `last_trace` 提供最近一次调用的诊断信息。

用 `set_step(step)` 设置优化器步数，用 `set_row_offset(row_id)` 设置每个 microbatch 的首个全局行号；带有键控随机性的方法（例如
Huginn 的采样深度）依赖它们。

## 评测与采样

```bash
python -m nexttransformer.evaluate --checkpoint runs/dsa/checkpoint_00046875 --output eval.json --eval bpb,core
python -m nexttransformer.generate --help
```

检查点按参数名保存张量，配置保存为普通字典，因此加载时不需要 pickle 代码。参数名和配置字段是检查点兼容性的一部分。

## 模型族与模块

| 模块 | 内容 |
|---|---|
| `registry.py` | 已注册的方法及其所属模型族 |
| `config.py` | `ModelConfig`、hybrid 布局、候选配置和调试配置 |
| `backbone.py` | nanochat d24 GPT 主干（按字节锁定的上游文件） |
| `lm.py` | `ReferenceLM`：由主干和各族的 mixer 组装出每个已注册的方法 |
| `looped.py`、`sparse.py`、`linear.py`、`residual.py`、`others.py` | 各模型族的实现 |
| `native/` | 历史 native 轨道：SSSL、MoR、LoopFormer、Circle Loop |
| `plugins.py` | `nexttransformer/methods/` 中贡献方法的插件接口 |
