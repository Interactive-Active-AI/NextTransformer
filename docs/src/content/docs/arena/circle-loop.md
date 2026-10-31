---
title: "Example: Circle Loop"
description: How an LLM-assisted architecture was researched, and why it is a legacy entry.
---

Circle Loop is the first arena example: the `circle` method of the historical native track. Its disclosure card,
`results/arena/circle-loop.json`, classifies it as **L3**: the idea and several turning points came from a person; LLMs
assessed novelty, formalized the mechanism, designed the halter, budget controller and loss, and implemented, tested and
launched every version. It is a `legacy` entry: it predates Arena Protocol v1 and does not meet it.

## Timeline

| Date (2026) | Who | What happened |
|---|---|---|
| Oct 9 | Person | Asks whether a looped Transformer can be a *circle*: prefix and suffix blocks around a ring of blocks with no absolute depth, where each token decides where to leave the ring |
| Oct 9 | ChatGPT | Compares the idea with Universal Transformer, Zero Token Transformer and Mixture-of-Recursions; argues that the ring is not new and that the open question is online, per-block exit from the computed state; formalizes a shared halter that receives no step count |
| Oct 10 | Codex | Writes an implementation plan on a GPT-2 backbone: 2 prefix + 8 ring + 2 suffix blocks, per-block exit, one shared halter, exited tokens frozen but kept as context |
| Oct 10–12 | Codex | Surveys recent looped-model work and proposes a different research question; the person keeps Circle as the main line |
| Oct 11 | Codex | Straight-through halting with a controller on mean depth, replacing a fixed cost penalty |
| Oct 12 | Person | Moves all models to the nanochat d24 shell with matched training compute; asks whether Circle's algorithm can be improved |
| Oct 12 | Codex | Proposes halter features from the state and its change, plus depth biases; approved and launched |
| Oct 13 | Person | Sees Circle trailing at 8,000 updates; questions why exited tokens are still computed, proposes computing only surviving tokens, a higher depth cap with a moderate mean, and three prefix and suffix blocks |
| Oct 13 | Codex | Proposes the final design: 3 + 9 + 3 blocks, 9–27 ring steps with a mean of 17.5, a halter with state-change features and 18 depth biases, a maximum-entropy depth target with a dual controller, and a random-route curriculum |
| Oct 13–14 | Person, Codex | Four formal launches: two exit before training (a mount path, a reused output directory), one stops at update 430 on a storage-quota error without a checkpoint, and the fourth runs to completion |

## Result

At update 46,875: validation BPB **0.6912** and CORE **28.97**, against 0.6671 and 33.78 for SSSL and 0.6672 and 33.10 for
the Dense Baseline. Its historical training estimate is 4.977 GFLOPs per input token, 4.2% above SSSL's estimate under
the same accounting. Circle Loop did not beat the baselines; it is the example because its research process is fully
known.

## Why it is legacy

- It was researched and trained before the arena rules existed.
- Its run uses the historical matrix-FLOPs estimate, and its compute is 4.2% above SSSL under that accounting; the arena
  requires an admitted configuration and a cumulative audit within 2%.
- Candidates were judged on periodic validation BPB and subset CORE, which the arena reserves for the formal run.
- No checkpoint or training logs are published.

An Arena v1 version would recalibrate Circle Loop under the operator ledger, select with the dev split, and train once
more under the reference protocol.

## What the example shows

- **Where the person mattered:** the original idea; keeping Circle when the agent suggested another question; noticing
  that exited tokens were still computed; and the 3 + 9 + 3 layout.
- **Where the agents mattered:** the novelty assessment, the halter and budget controller, the loss and curriculum, cost
  estimates, all code and tests, launch commands, and the diagnosis of every failed launch.
- **What an L0 entry would have to do alone:** notice a semantic inefficiency in its own implementation and change the
  design after a disappointing intermediate result, without using reserved evaluation signals to decide.
