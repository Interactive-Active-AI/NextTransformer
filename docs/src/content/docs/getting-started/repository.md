---
title: Repository layout
description: Where code, configurations, results and evidence live.
---

| Path | Contents |
|---|---|
| `nexttransformer/models/` | Built-in architectures, the registry and the shared shell |
| `nexttransformer/methods/` | Contributed methods, one plugin package each |
| `nexttransformer/protocol/` | The protocol specification, cost ledger, calibration, admission and fingerprints |
| `nexttransformer/train.py`, `evaluate.py`, `generate.py`, `prepare.py` | Command-line entry points |
| `nexttransformer/audit_run.py`, `export_run.py` | Audit a finished run; export it as a results record |
| `nexttransformer/checks/` | Bounded GPU checks, change classification, bitwise comparison, public-content scan |
| `nexttransformer/arena.py` | Arena disclosure cards and the scope of arena pull requests |
| `nanochat/` | The upstream runtime (data loading, evaluation, optimizer), pinned |
| `configs/<family>/<method>.json` | Admitted scientific configurations |
| `evidence/costs/` | Cost proofs; `evidence/audits/` holds cumulative audits of published runs |
| `results/` | Run records, model cards and arena cards; the website is built from them |
| `launch/` | Launchers: `train.sh` (torchrun, one node or several), `slurm.sbatch`, environment setup |
| `tests/` | The test suite; `tests/plugins/` holds example plugins |
| `website/` | The observatory (Vite + React) |
| `docs/` | This documentation (Starlight), in English and Chinese |

Scientific configurations never contain machine paths: data, caches and outputs come from command-line options, a
local site file (`*.local.json`, ignored by git) or environment variables.
