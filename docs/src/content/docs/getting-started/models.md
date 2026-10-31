---
title: Models API
description: Build, load, evaluate and sample any registered method from Python.
---

Every built-in method and every plugin is built through one interface.

```python
from nexttransformer.models import build_model, load_model, METHODS, FAMILY
from nexttransformer.models.config import candidate_config, debug_config

model = build_model(debug_config("dsa"))                       # a registered method at debug size (CPU-friendly)
model, saved = load_model("runs/dsa/checkpoint_00046875", "cuda")   # a trained checkpoint
print(len(METHODS), sorted(set(FAMILY.values())))
```

For formal use, admit the configuration first; admission checks the cost proof against the code and PyTorch build:

```python
from nexttransformer.protocol.admission import admit_config
from nexttransformer.models import build_model
from nexttransformer.training.optimizer import build_optimizer

config, proof = admit_config("configs/linear/kda.json")
model = build_model(config, "cuda")
model.set_training_stage(0)   # decide trainable parameters before building the optimizer and DDP
optimizer = build_optimizer(model)
```

## Calling a model

- `model(x, y)` returns the scalar training objective (cross-entropy plus any registered auxiliary loss).
- `model(x)` returns float32 logits.
- `model(x, y, loss_reduction="none")` returns plain per-token cross-entropy without auxiliary terms; BPB uses this.
- `model.eval()` selects the registered evaluation depth or exit policy.
- `last_lm_loss`, `last_aux_loss`, `last_stats` and `last_trace` expose diagnostics of the last call.

Set the optimizer step with `set_step(step)` and each microbatch's first global row with `set_row_offset(row_id)`;
methods with keyed randomness, such as Huginn's sampled depth, depend on them.

## Evaluate and sample

```bash
python -m nexttransformer.evaluate --checkpoint runs/dsa/checkpoint_00046875 --output eval.json --eval bpb,core
python -m nexttransformer.generate --help
```

Checkpoints store tensors by parameter name and a plain-dictionary configuration, so they load without pickled code.
Parameter names and configuration fields are part of checkpoint compatibility.

## Families and modules

| Module | Contents |
|---|---|
| `registry.py` | Registered methods and their families |
| `config.py` | `ModelConfig`, hybrid layouts, candidate and debug configurations |
| `backbone.py` | The nanochat d24 GPT backbone (byte-pinned upstream file) |
| `lm.py` | `ReferenceLM`: assembles every registered method from the backbone and the family mixers |
| `looped.py`, `sparse.py`, `linear.py`, `residual.py`, `others.py` | Family implementations |
| `native/` | The historical native track: SSSL, MoR, LoopFormer, Circle Loop |
| `plugins.py` | The plugin interface for contributed methods in `nexttransformer/methods/` |
