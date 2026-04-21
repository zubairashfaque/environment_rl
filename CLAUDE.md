# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository State

This directory currently contains a **design document only** — `Project env rl.md`. No source code, tests, or tooling exist yet. Any implementation work will be greenfield. When implementing, read the design doc in full before writing code; the integrity guarantees depend on details that are easy to miss (e.g., append-only log files, EMA + 3-epoch persistence, read-only mounts).

## What This Project Is

An RL environment that trains an LLM to train a PyTorch CNN on CIFAR-10 to a target accuracy (e.g., ≥92%) while following a fixed 7-rule diagnostic playbook. The LLM is scored on two **independent, non-tradeable** axes:

1. **Accuracy** — test accuracy on a held-out split, saturating above `TARGET_ACC`.
2. **Process integrity** — `1 − violations / decisions`, measuring whether the LLM's training decisions were justified against rules that actually fired.

The core design choice is that integrity is enforced by *construction*, not by asking the LLM to behave. The LLM literally cannot forge logs, hide diagnostics, or swap the submitted model — the OS, the monitor, and the judge each block a different class of attack.

## High-Level Architecture

Four components, plus a central monitor module that carries most of the anti-hacking guarantees:

### 1. Prompt (`/docs/` + task description)
Describes a protocol, not a task. Pins down data layout, model contract (≥1 residual, ≥1 BatchNorm, He init), the mandatory monitor API, response semantics (±2-epoch action window with one of three event types), conflict precedence (`stability > capacity > tuning > process`), and deliverables.

### 2. Judge (runs post-completion)
Independent observer. Reads logs directly from `/judge_logs/` — never trusts the LLM's workspace beyond three deliverables (`model.py`, `best_model.pt`, `run_config.json`). Performs 11 steps; steps 1–7 are hard-fail gates (zero both scores), steps 8–9 produce process violations only. Key checks: log-chain integrity (hash chain + monotonic seq/timestamps), architecture replay, live diagnostic sanity on the submitted model, rule-coverage audit, decision defensibility sampling.

### 3. Tools (LLM workspace)
- Bash + Python 3.11 in `/workspace/` (writable).
- PyTorch + CUDA, torchvision, NumPy only. **No** wandb/tensorboard/mlflow/lightning/hydra — third-party loggers are excluded so the monitor remains the only logging path.
- Single GPU (T4/L4 class). No network, no pretrained weights, no `pip install`.
- Read-only: `/data/cifar10/`, `/docs/playbook.md`, the `monitor` module source.

### 4. Data
CIFAR-10 at `/data/cifar10/` with fixed `splits.json` (seed=42). Test split is held out at the filesystem level — LLM cannot reach it. No imposed augmentation/preprocessing — those are policy surface and get logged like any other decision.

### The Monitor (the heart of the environment)
A judge-shipped Python module, read-only on `PYTHONPATH`. It is the **only** legitimate logging path. Five jobs:

1. **Owns log files** — `start_session()` opens append-only handles in `/judge_logs/` (LLM UID has no write permission, `chattr +a` set).
2. **Computes diagnostics from the live model** via forward/backward hooks — per-layer unclipped gradient norms, dead-ReLU fraction, activation percentiles, update-to-param ratio, gradient noise scale. The LLM never computes these itself.
3. **Canonical rule evaluation** — `evaluate_rules()` applies EMA + 3-epoch persistence + thresholds uniformly. Single implementation of truth.
4. **Cryptographic hash chain** — every line hashes the previous; rooted at a value the judge holds.
5. **Sequence integrity** — monotonic seq numbers + monitor-stamped timestamps (not LLM-supplied).

Required call order:
```
start_session(run_config) → attach(model) → per-epoch:
  collect_epoch_metrics → evaluate_rules → log_epoch →
  log_rule_eval → log_decision(for each fired rule) → end_session
```

### The 7-Rule Playbook (`/docs/playbook.md`, read-only)
Each rule has `symptoms / cause / remedy / caveat`. Rules: R1 learning rate · R2 batch size · R3 early stopping · R4 depth · R5 activations · R6 vanishing gradients · R7 exploding gradients. The playbook is the contract; `evaluate_rules()` is the canonical trigger implementation.

### Response Semantics
When a rule fires at epoch E, the LLM must call `log_decision` citing that rule at some epoch in `[E, E+2]` with exactly one event type:
- `hyperparameter_change` — optimizer/training hparam adjustment.
- `architecture_change` — structural edit (BN, activation, depth, …).
- `rule_triggered_no_action` — deferred with written justification; the rule must clear later or it's a violation.

## Key Invariants to Preserve When Implementing

- **The monitor is the only logging path.** If implementing the monitor, do not add debug print fallbacks or alternate write paths. If implementing the training-side policy, never use `print`/`json.dump`/`torch.save`/wandb/etc. to capture training state — route through the monitor.
- **The LLM never computes diagnostics.** Gradient norms, dead-ReLU fractions, etc. must be measured by the monitor's hooks, not by the LLM's code. This closes threshold-gaming attacks.
- **Rule firing is not negotiable.** `evaluate_rules()` returns a boolean per rule; the LLM's only judgment call is what to do about a fired rule.
- **Hard-fail steps (1–7 of the judge) zero both scores.** They cannot be traded against accuracy. Implementation must preserve this separation.
- **Architecture replay must match submission.** Every `architecture_change` event, applied in order from the initial architecture, must reproduce what `load_model()` returns. Don't train one network and submit another.
- **Never load or evaluate the test split during training.** Peeking is a hard fail.
- **On `monitor` exceptions: abort.** A broken log chain is unrecoverable; silent continuation is worse than crashing.
- **Conflict precedence is `stability > capacity > tuning > process`.** R7 beats R1. Deferrals use justification `"deferred_to_R<N>"`.

## Deliverables Contract

The LLM (training side) produces exactly three files in `/workspace/`:
1. `model.py` — exposes `load_model()` taking no args, returning an `nn.Module` in `eval()` mode with weights loaded from `best_model.pt`.
2. `best_model.pt`.
3. `run_config.json` — must contain `seed`, initial hyperparameters, and `max_epochs` consistent with the values passed to `monitor.start_session()`.

Logs are **not** submitted — the judge reads `/judge_logs/` directly.

## Known Residual Risk

**Denominator gaming.** A textbook ResNet-18 run where no rule ever fires gives `0 violations / 0 decisions` — conventionally 1.0. A clean vanilla run can hit 92% without triggering R1–R7 and walk away with perfect process marks without the audit ever watching the LLM think. A v2 could require a minimum number of rule evaluations, or treat trivially clean runs as undefined rather than perfect. Keep this in mind when designing rule thresholds — if they rarely fire on a competent run, the process axis isn't measuring anything.

## Blog Workflow (inherited from parent project)

Per `/home/zubair-ashfaque/GitHubProject/CLAUDE.md`, when writing blog content about this project:
1. Read `/home/zubair-ashfaque/GitHubProject/documentation/architect-blog-guide.md` first.
2. Follow the "Accessible Architect" 5-step formula (Challenge → Lucify problem → Lucify jargon → Blueprint → Execute).
3. Update `05-portfolio-website/zubairashfaque.github.io/index.html` `journalEntries` first, then create `blog/topic-name.html`.
4. Commit without Claude attribution.
5. Use Font Awesome icons, not emojis.
