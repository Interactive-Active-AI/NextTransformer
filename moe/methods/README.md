# MoE method plugins

Each contributed MoE method is one directory here, `moe/methods/<name>/`, with a `method.json` declaration
and a Python package that implements its hooks (`router`, `ffn`, `mixer`, `block`, `model`, `init`, `check`).
The registry discovers it; a plugin never edits `moe/model.py`, the trainer, the data stream, the ledger or the
evaluation, so it cannot change another method.

The hook contract is in [`moe/plugins.py`](../plugins.py) and the end-to-end submission flow (calibrate,
train `--formal`, evaluate, audit, export, pull request) is in [`moe/README.md`](../README.md#submitting-a-method).
Test plugins in [`tests/moe_plugins/`](../../tests/moe_plugins) show a router and a whole-FFN replacement.

## d768 residual comparison

The [240 TPP configurations](../configs/d768-240tpp/) share the same attention, experts and routing. The three
plugins each carry four residual streams. Each plugin's declaration links to the paper and fixes its options.

| Method | Residual computation | Measured FLOPs/token vs baseline |
|---|---|---:|
| [baseline](../model.py) | The reference model adds every attention and FFN output to one residual stream. | 0.0000% |
| [HC](hc/) | Dynamic read, residual mixing and write weights for each attention and FFN sublayer. | +0.3463% |
| [mHC](mhc/) | Constrained mixing with two Sinkhorn iterations per sublayer. | +0.9129% |
| [SimpleHC](simplehc/) | One group's read and write gates are computed at block entry for its two sublayers. | +0.3148% |

The [cost records](../evidence/costs/d768-240tpp/) contain each measured configuration and its source hash.
