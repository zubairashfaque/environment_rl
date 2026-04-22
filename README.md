# env_rl — RL Evaluation Environment for Disciplined Model Training

> An RL-style environment that trains an LLM to train a PyTorch CNN on CIFAR-10 under a fixed **7-rule diagnostic playbook**, with a judge-monitored integrity system that scores the LLM on two independent, non-tradeable axes: **final-model accuracy** and **process discipline**.

<p align="center">
  <a href="https://github.com/zubairashfaque/environment_rl"><img alt="repo" src="https://img.shields.io/badge/github-zubairashfaque%2Fenvironment__rl-f59e0b?logo=github"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-162%20passing-10b981">
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
  <img alt="poetry" src="https://img.shields.io/badge/poetry-managed-60a5fa">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-8b5cf6">
</p>

| | |
|---|---|
| 📖 **Design doc** | [`Project env rl.md`](./Project%20env%20rl.md) |
| 📋 **7-rule playbook** | [`docs/playbook.md`](./docs/playbook.md) |
| 🤖 **LLM setup guide** | [`docs/setup-llm.md`](./docs/setup-llm.md) |
| 🏗️ **Implementation plan** | `~/.claude/plans/plan-the-project-make-optimized-squirrel.md` |
| ✍️ **Blog post** | https://zubairashfaque.github.io/blog/env-rl-disciplined-training.html |

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Architecture](#architecture)
3. [The 7 rules at a glance](#the-7-rules-at-a-glance)
4. [Prerequisites](#prerequisites)
5. [Step-by-step setup](#step-by-step-setup)
6. [Running the project](#running-the-project)
   - [6.1 Synthetic smoke test (no CIFAR download)](#61-synthetic-smoke-test-no-cifar-download)
   - [6.2 Real CIFAR-10 with an OpenAI model](#62-real-cifar-10-with-an-openai-model)
   - [6.3 The scripted reference agent (no API key)](#63-the-scripted-reference-agent-no-api-key)
7. [Per-attempt output files — what everything means](#per-attempt-output-files--what-everything-means)
8. [Inspecting a run with `show_full_run.py`](#inspecting-a-run-with-show_full_runpy)
9. [Understanding the scores](#understanding-the-scores)
10. [Cheat-attempt defenses](#cheat-attempt-defenses)
11. [Tuning & knobs](#tuning--knobs)
12. [Troubleshooting](#troubleshooting)
13. [Running the test suite](#running-the-test-suite)
14. [Project layout](#project-layout)
15. [Why this is "Iterative Self-Refine," not RL](#why-this-is-iterative-self-refine-not-rl)
16. [Contributing / extending](#contributing--extending)

---

## What this is

Training a deep learning model well isn't the training loop — it's the **hundreds of judgment calls** you make during it. Those calls are the hardest to audit and easiest to fake after the fact. `env_rl` closes that gap by making the LLM a *decision-maker* with an independent observer auditing every decision.

**Key numbers:**

| Metric | Value |
|---|---|
| Tests passing | **162** |
| Diagnostic rules | **7** (R1–R7) |
| Judge audit steps | **11** (7 hard-fail gates + 2 process + 2 scoring) |
| Cheat-attempt defenses | **5** (shadow log, model swap, trajectory forge, skipped decision, forged log line) |
| Typical decisions per run | 20–60 |
| Cost per attempt (`gpt-4o-mini`) | ~**\$0.01** |

---

## Architecture

```mermaid
flowchart LR
    subgraph "Agent Workspace (RW)"
        A[Training Loop]
        M1[model.py]
        W1[best_model.pt]
        C1[run_config.json]
    end
    subgraph "Read-Only Mounts"
        D[CIFAR-10 +<br/>splits.json]
        P[playbook.md]
        MS[monitor source]
    end
    subgraph "Judge-Owned (Append-Only)"
        L1[metrics_log.jsonl]
        L2[decision_log.jsonl]
        L3[rule_evaluations.jsonl]
    end
    subgraph "LLM Harness"
        H1[OpenAI API]
        H2[Iterative Self-Refine Loop]
    end
    subgraph "Judge (Post-Run)"
        J[11-Step Audit]
        S1[Accuracy Score]
        S2[Process Integrity Score]
    end

    A <-->|per fired rule| H1
    H1 --> H2
    H2 --> A
    A -->|reads| D
    A -->|reads| P
    A -->|imports| MS
    A -.->|API calls only| L1
    A -.->|API calls only| L2
    A -.->|API calls only| L3
    A -->|writes| M1
    A -->|writes| W1
    A -->|writes| C1
    L1 --> J
    L2 --> J
    L3 --> J
    M1 --> J
    W1 --> J
    C1 --> J
    D --> J
    J --> S1
    J --> S2
```

**Trust boundary:** the dashed arrows go *through* the `monitor` API — the agent never writes to `/judge_logs/` directly.

### Four components

```
src/env_rl/
├── monitor/     Owns /judge_logs/. PyTorch hooks. Canonical rule evaluator. Hash-chained writer.
├── judge/       Post-run 11-step audit. Two-axis scoring. Architecture replay. Live-diag sanity.
├── data/        CIFAR-10 deterministic splits (seed=42). Held-out test split.
├── agent/       ResidualCNN meeting the model contract. Pluggable training loop.
└── harness/     OpenAI-backed DecisionPolicy + iterative feedback loop.
```

---

## The 7 rules at a glance

The LLM's job is to respond correctly when each rule fires. Rules are evaluated by a single canonical `evaluate_rules()` — the LLM does not get a vote on whether a rule fired, only what to do about it.

| ID | Name | Signal | Precedence |
|---|---|---|---|
| **R7** | Exploding gradients | max per-layer grad-norm EMA > 10 for 3 epochs, or NaN/Inf loss | **Stability** (highest) |
| **R6** | Vanishing gradients | min per-layer grad-norm EMA < 1e-5 for 3 epochs | **Stability** |
| **R5** | Dead activations | dead-ReLU fraction EMA > 0.40 for 3 epochs | Capacity |
| **R4** | Depth / capacity | train-acc delta < saturation_gap for 3 epochs | Capacity |
| **R1** | Learning rate | update-to-param ratio EMA out of [1e-4, 1e-2], or val-loss plateau | Tuning |
| **R2** | Batch size | gradient noise scale EMA outside [50, 5000] for 3 epochs | Tuning |
| **R3** | Early stopping | val loss no improvement over patience window | **Process** (lowest) |

**Conflict precedence when multiple fire:** `stability > capacity > tuning > process`. Action the highest-precedence rule; defer the rest with `"deferred_to_R<N>"`.

Full rule definitions with symptoms / cause / remedy / caveat live in [`docs/playbook.md`](./docs/playbook.md).

---

## Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11+ | Type syntax, dataclasses |
| Poetry | 2.0+ | Dependency management |
| `git` | any | Cloning / version control |
| Disk | ~2 GB | PyTorch wheels + CIFAR-10 |
| RAM | 4 GB | CPU-only training |
| GPU | optional | Faster real-data runs |
| OpenAI API key | optional | Only for LLM harness (not for scripted agent) |

Check what you have:

```bash
python3 --version        # 3.11 or newer
poetry --version         # 2.0+
git --version            # any recent
```

If Poetry isn't installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

---

## Step-by-step setup

### Step 1 — Clone the repository

```bash
cd ~/GitHubProject          # or anywhere you like
git clone https://github.com/zubairashfaque/environment_rl.git
cd environment_rl
```

### Step 2 — Install dependencies

```bash
export PATH="$HOME/.local/bin:$PATH"     # if poetry not on PATH
poetry install
```

First install takes ~1–3 minutes (torch is the heavy one). Verify:

```bash
poetry run python -c "import torch; import openai; print('ok')"
# ok
```

### Step 3 — Run the test suite

```bash
poetry run pytest -q
# ... (takes ~3 seconds)
# 162 passed
```

If this isn't green, stop — something is off with your install.

### Step 4 — Set your OpenAI API key (optional, only for the LLM harness)

Skip this step if you only want to run the scripted reference agent.

**Easiest — per-shell export:**
```bash
export OPENAI_API_KEY=sk-proj-your-actual-key-here
echo $OPENAI_API_KEY | cut -c1-10   # verify (shows first 10 chars)
```

**Per-project `.env` file (survives across shells):**
```bash
echo 'OPENAI_API_KEY=sk-proj-your-actual-key-here' > .env
export $(grep -v '^#' .env | xargs)
```

`.gitignore` already excludes `.env` from git. Full guide in [`docs/setup-llm.md`](./docs/setup-llm.md).

### Step 5 — Download CIFAR-10 (optional — only if running without `--synthetic`)

```bash
# ~170 MB, ~1-3 minute download
poetry run python -c "from torchvision import datasets; datasets.CIFAR10(root='./data/cifar10', train=True, download=True); datasets.CIFAR10(root='./data/cifar10', train=False, download=True)"
```

### Step 6 — Build the deterministic split manifest

```bash
poetry run python -m env_rl.data.splits --seed 42 --out data/cifar10/splits.json
# wrote data/cifar10/splits.json (train=45000, val=5000)
```

You're ready to run.

---

## Running the project

Three ways to run, in increasing realism:

### 6.1 Synthetic smoke test (no CIFAR download)

Fastest path. Uses random tensors. No real training; only verifies the plumbing works end-to-end.

```bash
poetry run python examples/run_llm_agent.py --synthetic --attempts 3 --epochs 8 --model gpt-4o-mini --base-dir ./llm_runs
```

**Expected output:**
```json
{
  "mode": "iterative_self_refine",
  "best_attempt": 2,
  "best_scores": {
    "accuracy_score": 0.625,
    "process_score": 0.65,
    "hard_fail": false,
    "test_accuracy": 0.125,
    "violations": 7,
    "total_decisions": 20
  },
  "all_attempts": [...]
}
```

Time: ~30 seconds. Cost: ~\$0.01.

Test accuracy stays at ~0.125 (chance-level on 10-class random data) — that's expected, synthetic data has no pattern to learn.

### 6.2 Real CIFAR-10 with an OpenAI model

Real training, real LLM-driven decisions. Accuracy will actually climb.

```bash
poetry run python examples/run_llm_agent.py --attempts 3 --epochs 20 --batch-size 128 --lr 0.05 --model gpt-4o-mini --target-acc 0.70 --base-dir ./llm_runs_real
```

**Expected output:**
```json
{
  "best_scores": {
    "accuracy_score": 0.86,
    "process_score": 0.65,
    "hard_fail": false,
    "test_accuracy": 0.60,
    "violations": 18,
    "total_decisions": 55
  }
}
```

Time: ~10–20 min on CPU, ~3–5 min on GPU. Cost: ~\$0.05 per run with `gpt-4o-mini`.

> **Note on the rules:** R2, R3, R4, R6 are *waived* in the default reference harness (it can't rebuild the DataLoader mid-run, add blocks to the optimizer, or retrofit BatchNorm). R1, R5, R7 are *active*. The judge treats waived firings as advisory — no process penalty for deferring them. See [`docs/setup-llm.md`](./docs/setup-llm.md#internals) for why.

### 6.3 The scripted reference agent (no API key)

Runs the hard-coded baseline policy — good for CI, integration testing, and a sanity check without spending tokens.

```bash
poetry run python examples/run_reference_agent.py --synthetic --epochs 3 --batch-size 16 --workspace /tmp/ref/workspace --judge-logs /tmp/ref/judge_logs
# {"best_val_acc": 0.094, "final_epoch": 2}
```

No API key needed. This is what the plan calls "a scripted demonstrator that proves the monitor and judge function correctly end-to-end."

---

## Per-attempt output files — what everything means

Every attempt creates a folder `llm_runs/attempt_NN/` with nine artifacts:

```
llm_runs/attempt_01/
├── workspace/
│   ├── model.py                  ← what load_model() reconstructs
│   ├── best_model.pt             ← best validation weights
│   └── run_config.json           ← seed + hyperparams + max_epochs
├── judge_logs/
│   ├── metrics_log.jsonl         ← hash-chained per-epoch metrics
│   ├── decision_log.jsonl        ← hash-chained decisions (what the LLM chose)
│   └── rule_evaluations.jsonl    ← hash-chained {R1..R7: bool} per epoch
├── summary.json                  ← scores + violation summary (the top-level answer)
├── feedback_in.json              ← what was fed INTO this attempt from priors
├── training_trace.jsonl          ← per-epoch human-readable trace
├── judge_trace.json              ← 11-step audit pass/fail/duration/detail
└── llm_transcript.jsonl          ← every OpenAI call: request, response, tokens, cost
```

| File | Purpose | Key fields |
|---|---|---|
| `summary.json` | The attempt's score card | `scores`, `violation_summary`, `mode` |
| `feedback_in.json` | Prior-attempts summary fed into this attempt's system prompt | `prior_attempts[].violation_summary` |
| `training_trace.jsonl` | Per-epoch training events | `kind ∈ {session_start, epoch, decision, remedy_applied, session_end}` |
| `judge_trace.json` | 11-step audit trail | `steps[].{step, name, kind, status, duration_ms, detail, extra}` |
| `llm_transcript.jsonl` | Every OpenAI call | `kind ∈ {system_prompt, call}`, `usage`, `user_message`, `response` |
| `judge_logs/metrics_log.jsonl` | Hash-chained metrics (authoritative) | `seq, ts, prev_hash, payload.kind, payload.epoch, payload.max_layer_grad_norm, ...` |
| `judge_logs/decision_log.jsonl` | Hash-chained decisions (authoritative) | `seq, ts, prev_hash, payload.event_type, payload.cites, payload.justification` |
| `judge_logs/rule_evaluations.jsonl` | Hash-chained rule evaluations | `seq, ts, prev_hash, payload.evals.R1..R7` |
| `workspace/*` | The LLM's submission (judge reads these post-run) | `load_model()`, weights, initial hparams |

---

## Inspecting a run with `show_full_run.py`

One unified viewer prints every log for one attempt:

```bash
# Everything
poetry run python examples/show_full_run.py llm_runs/attempt_01

# Just one section
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section summary
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section judge
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section training
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section llm
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section feedback

# Zoom in on one epoch's LLM calls
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section llm --epoch 5
```

Example sections you'll see:

<details>
<summary>▸ Summary</summary>

```
========================================================================
SUMMARY
========================================================================
  mode: iterative_self_refine
  attempt_index: 2
  hard_fail: false
  accuracy_score: 0.6250
  process_score:  0.6500
  test_accuracy:  0.1250
  violations:     7 / 20
  7 violations:
      6  unresolved_deferral       rule=R1
      1  precedence_violation      rule=R5
```
</details>

<details>
<summary>▸ Judge trace (11 steps)</summary>

```
========================================================================
JUDGE TRACE (11-step audit)
========================================================================
  [✓] step  1  deliverables_exist             hard_fail_gate        0.42 ms
  [✓] step  2  signature_and_load             hard_fail_gate        8.13 ms
  [✓] step  5  chain_integrity                hard_fail_gate       12.40 ms
  [✓] step  4  run_config_consistency         hard_fail_gate        0.51 ms
  [✓] step  6  architecture_replay            hard_fail_gate        1.22 ms
  [✓] step  7  live_diagnostic_sanity         hard_fail_gate      120.45 ms
        live_max: 0.4210
  [✓] step  8  rule_coverage_audit            process_only          1.85 ms
        detail: 7 coverage violations
  [✓] step  9  decision_defensibility         process_only          0.67 ms
  [✓] step 10  test_set_evaluation            scoring             450.12 ms
        test_accuracy: 0.1250
  [✓] step 11  emit_two_scores                scoring               0.08 ms
        accuracy_score: 0.625  process_score: 0.650
```
</details>

<details>
<summary>▸ Training trace (per-epoch)</summary>

```
========================================================================
TRAINING TRACE (per-epoch)
========================================================================
  START  seed=42 lr=0.3 bs=32 max_epochs=8 num_blocks=2 activation=relu
  ep  0  loss=2.30/2.30  acc=0.10/0.10  lr=0.3  grad_max=4.21  dead_relu=0.08 fired={}
  ep  1  loss=2.25/2.26  acc=0.12/0.11  lr=0.3  grad_max=3.98  dead_relu=0.22 fired={}
  ep  2  loss=2.20/2.24  acc=0.13/0.12  lr=0.3  grad_max=4.12  dead_relu=0.73 fired={R1,R2,R5}
       DECISION  cite=R5  event=architecture_change  direction=swap_activation  just="Dead-ReLU fraction 0.73 over 3 epochs..."
       REMEDY    lr -> 0.3
  ep  3  loss=2.15/2.20  acc=0.15/0.14  lr=0.3  grad_max=3.85  dead_relu=0.51 fired={R1,R2,R5}
  ...
  END    best_val_acc=0.1250
```
</details>

<details>
<summary>▸ LLM transcript (calls with tokens)</summary>

```
========================================================================
LLM TRANSCRIPT
========================================================================
  [system] model=gpt-4o-mini  T=0.2
  system_prompt length: 4280 chars

  [call] epoch=2 top_rule=R5 fired=['R1','R2','R5'] tokens=872
  --- user message ---
  Epoch 2. Rule(s) fired: ['R1', 'R2', 'R5']...
  --- response ---
  {
    "event_type": "architecture_change",
    "cites": ["R5"],
    "justification": "Dead-ReLU fraction 0.73 ...",
    "remedy_direction": "swap_activation",
    "remedy_params": {"lr_new": 0.3, "edit_op": "swap_activation", "edit_to": "leaky_relu"}
  }
  ...
  [total]  5 LLM calls, 4360 tokens
```
</details>

<details>
<summary>▸ Iterative feedback (what flowed in from prior attempts)</summary>

```
========================================================================
ITERATIVE FEEDBACK (fed in to this attempt)
========================================================================
  1 prior attempts fed into this attempt's system prompt:
    • attempt 1: accuracy=0.625  process=0.550  violations=9/20
      - precedence_violation rule=R5 epoch=4
      - unresolved_deferral rule=R1 epoch=2
      - unresolved_deferral rule=R1 epoch=3
```
</details>

---

## Understanding the scores

Two decoupled scalars, reported independently.

```
accuracy_score = test_accuracy / target_acc   if test_accuracy < target_acc
               = 1.0                           otherwise (saturates)

process_score  = 1 − violations / total_decisions
               = 1.0                           if total_decisions == 0 (denominator-gaming caveat)
```

**Hard fails** (judge steps 1–7) zero both scores. **Process violations** (steps 8–9) reduce only the process score. **Low test accuracy** reduces only the accuracy score.

### Decoupling matters

Accuracy and process are **not tradeable**. Skipping a rule to save an epoch of compute drops process score by exactly `1/total_decisions`, while accuracy keeps its own measurement. Training one model for accuracy and submitting a different one for the architecture check hard-fails everything — step 6 catches architecture mismatches.

### Denominator-gaming caveat

If no rule ever fires (a textbook ResNet-18 run), `total_decisions = 0` → process_score = 1.0 by convention. The rule book never got exercised. Future work: require a minimum number of rule firings before the process axis is considered valid.

---

## Cheat-attempt defenses

Every class of cheat has a catch-point:

```mermaid
flowchart LR
    C1[Shadow log in /workspace/] -->|ignored| J5[Step 5: read only /judge_logs/]
    C2[Train model A, submit model B] -->|hard fail| J6[Step 6: architecture replay]
    C3[Forge hash on past line] -->|hard fail| J5b[Step 5: chain verification]
    C4[Submit zero-weight model] -->|hard fail| J7[Step 7: live diag sanity]
    C5[Skip decision for fired rule] -->|process -1/N| J8[Step 8: coverage audit]
```

All five are covered by integration tests in `tests/integration/test_cheat_attempts.py`.

---

## Tuning & knobs

```bash
poetry run python examples/run_llm_agent.py --help
```

| Flag | Default | Effect |
|---|---|---|
| `--attempts` | 3 | Outer loop iterations. More attempts = more iterative refinement. |
| `--epochs` | 3 | Training epochs per attempt. Rules need 3 consecutive epochs to fire — use ≥8 for real runs. |
| `--batch-size` | 32 | Training batch size. 128 is common for CIFAR-10. |
| `--lr` | 0.3 | Initial learning rate. 0.05–0.1 typical for real CIFAR. |
| `--model` | `gpt-4o-mini` | OpenAI model. `gpt-4o` is stronger but 15× more expensive. |
| `--temperature` | 0.2 | LLM sampling temperature. 0.0 deterministic, 0.8 exploratory. |
| `--target-acc` | 0.20 | Accuracy score saturation point. Use 0.70–0.90 for real runs. |
| `--base-dir` | `./llm_runs` | Where attempt folders go. |
| `--synthetic` | (flag) | Use random data. No CIFAR download needed. Chance-level accuracy expected. |
| `--data-dir` | `./data/cifar10` | Path to torchvision CIFAR-10. |
| `--manifest` | `./data/cifar10/splits.json` | Path to splits manifest (from step 6). |

---

## Troubleshooting

### "OPENAI_API_KEY environment variable is not set"

Step 4 didn't survive into the current shell. Re-export, or add to `~/.bashrc`:

```bash
echo 'export OPENAI_API_KEY=sk-proj-...' >> ~/.bashrc
source ~/.bashrc
```

### `httpx.ConnectError: Temporary failure in name resolution`

Your machine can't reach `api.openai.com`. Check in order:
```bash
ping -c 2 8.8.8.8                              # basic internet?
nslookup api.openai.com                        # DNS working?
curl -I https://api.openai.com/v1/models       # OpenAI reachable?
```
If DNS is broken: `sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'`.

### `hard_fail: true` and scores are zero

Check the exact step that failed:
```bash
poetry run python examples/show_full_run.py llm_runs/attempt_01 --section judge
```
Common causes and fixes are below.

| Failing step | Symptom in `judge_trace.json` | Fix |
|---|---|---|
| Step 5 — chain | `ChainVerificationError` or "missing bookend" | Delete `llm_runs/attempt_NN/` and re-run (stale logs from a prior run) |
| Step 6 — architecture replay | "architecture mismatch on activation" | Pull latest — activation threading bug was fixed |
| Step 7 — live diag | "live/logged gradient norm disagreement" | Pull latest — tolerance + multi-batch averaging was increased for real runs |

### Paste keeps wrapping at `--manifest`

Your terminal is breaking long lines. Use the short form (now available since defaults added):

```bash
poetry run python examples/run_llm_agent.py --attempts 1 --epochs 20 --batch-size 128 --lr 0.05 --model gpt-4o-mini --target-acc 0.70 --base-dir llm_runs_real
```

Or write a script file:
```bash
echo 'poetry run python examples/run_llm_agent.py --attempts 1 --epochs 20 --batch-size 128 --lr 0.05 --model gpt-4o-mini --target-acc 0.70 --base-dir llm_runs_real' > run.sh
bash run.sh
```

### Process score identical across attempts

Iterative self-refine needs something to refine. If scores are identical, try:
- Raise temperature: `--temperature 0.8`
- Use a stronger model: `--model gpt-4o`
- More epochs (more decisions to get right): `--epochs 15`

### Tests fail on a fresh install

```bash
poetry install --no-cache   # force fresh wheel downloads
poetry run pytest -v        # see which tests fail
```
If only integration tests fail, you may be offline or hitting CPU limits; unit tests should always pass.

---

## Running the test suite

```bash
poetry run pytest -q                                         # all 162 tests
poetry run pytest tests/unit/ -v                             # just unit tests (154)
poetry run pytest tests/integration/ -v                      # just integration (8)
poetry run pytest tests/integration/test_cheat_attempts.py   # just cheat tests (5)
poetry run pytest -k "hash_chain" -v                         # any test with "hash_chain"
poetry run pytest --cov=src --cov-report=term-missing        # with coverage
```

Coverage is 95%+ across monitor and judge.

---

## Project layout

```
environment_rl/
├── README.md                                    ← you are here
├── Project env rl.md                            ← original design doc
├── pyproject.toml                               ← Poetry dependencies
├── CLAUDE.md                                    ← project-local Claude Code instructions
├── conf/                                        ← Hydra configs
│   ├── config.yaml
│   ├── monitor/default.yaml
│   ├── judge/default.yaml
│   └── training/reference_agent.yaml
├── docs/
│   ├── playbook.md                              ← the 7-rule contract (read-only in prod)
│   └── setup-llm.md                             ← OpenAI configuration walkthrough
├── src/env_rl/
│   ├── monitor/                                 ← logging + rule evaluation + hooks
│   │   ├── __init__.py                          ← public API (start_session, attach, ...)
│   │   ├── session.py                           ← session lifecycle
│   │   ├── hooks.py                             ← fwd/bwd PyTorch hooks
│   │   ├── rules.py                             ← 7-rule evaluator (canonical)
│   │   ├── logging.py                           ← append-only hash-chained writer
│   │   └── ema.py                               ← EMA + consecutive persistence
│   ├── judge/                                   ← 11-step post-run audit
│   │   ├── __init__.py                          ← run_judge() orchestrator
│   │   ├── deliverables.py                      ← steps 1-4
│   │   ├── chain.py                             ← step 5
│   │   ├── replay.py                            ← step 6
│   │   ├── live_diag.py                         ← step 7
│   │   ├── coverage.py                          ← step 8 (precedence + waived rules)
│   │   ├── defensibility.py                     ← step 9
│   │   ├── evaluation.py                        ← step 10
│   │   └── scoring.py                           ← step 11
│   ├── data/
│   │   ├── splits.py                            ← deterministic CIFAR-10 splits
│   │   └── loaders.py                           ← train/val loaders + test-split guard
│   ├── agent/
│   │   ├── model.py                             ← ResidualCNN meeting the contract
│   │   └── reference_run.py                     ← training loop, policy-pluggable
│   └── harness/
│       ├── __init__.py                          ← HARNESS_MODE = "iterative_self_refine"
│       ├── prompt.py                            ← system + user prompt builders
│       ├── policy.py                            ← ScriptedDecisionPolicy, OpenAIDecisionPolicy
│       ├── edits.py                             ← swap_activation applier
│       └── iterative.py                         ← N-attempt driver with feedback
├── examples/
│   ├── run_reference_agent.py                   ← scripted agent (no API key)
│   ├── run_llm_agent.py                         ← OpenAI LLM harness
│   ├── show_full_run.py                         ← unified per-attempt viewer
│   ├── show_llm_transcript.py                   ← LLM-only viewer
│   └── show_iterative_feedback.py               ← feedback loop viewer
└── tests/                                       ← 162 tests
    ├── unit/                                    ← 154 unit tests
    │   ├── monitor/, judge/, data/, agent/, harness/
    └── integration/                             ← 8 integration tests
        ├── test_full_reference_run.py
        ├── test_llm_harness_end_to_end.py
        └── test_cheat_attempts.py
```

---

## Why this is "Iterative Self-Refine," not RL

The harness is labeled `"mode": "iterative_self_refine"` throughout the code and documentation. This matters:

| | **Real reinforcement learning** | **This harness (self-refine)** |
|---|---|---|
| Policy | Parametric model with trainable weights | OpenAI model with frozen weights |
| Update signal | Gradient from reward | Text appended to the prompt |
| Per-decision reward | Required for PPO/DPO | End-of-run scores only |
| Learning persists | In weights, across episodes | In prompt, discarded at conversation end |
| Access needed | Open-weights (Llama, Qwen, Mistral) | API-only |
| Time / cost | Weeks + GPU hours | Hours + pennies |

The *environment* half of RL (observations, actions, rewards) is what this project builds — a building block for real RL, not RL itself. A future `src/env_rl/rl/` module could decompose per-decision rewards and fine-tune with DPO on collected trajectories.

More detail in [`docs/setup-llm.md#this-is-not-reinforcement-learning`](./docs/setup-llm.md#this-is-not-reinforcement-learning) and in the blog post at https://zubairashfaque.github.io/blog/env-rl-disciplined-training.html.

---

## Contributing / extending

### Un-waive rules

To expand what the harness can action:

| Rule | What it needs | Rough shape |
|---|---|---|
| R2 (batch size) | `DataLoader` rebuild | Add `rebuild_loader(new_batch_size)` in `reference_run.py` |
| R3 (early stop) | Loop exit signal | Add `early_stop` flag; check after each epoch |
| R4 (add block) | Optimizer param-group add | Extend `edits.py` with `add_block`; call `optimizer.add_param_group` |
| R6 (vanishing) | Model rebuild + weight transfer | Nontrivial; may require checkpoint + reload |

Then remove from `HARNESS_WAIVED_RULES` in `src/env_rl/harness/__init__.py`.

### Real reinforcement learning on top

```
src/env_rl/rl/                      ← new module
├── rewards.py                      ← decompose process score into per-decision rewards
├── trajectories.py                 ← collect (state, decision, reward) tuples
└── dpo.py                          ← fine-tune Llama/Qwen with DPO on (better, worse) pairs
```

This is the natural next milestone if you want the LLM to *learn* (in weights) rather than just *refine* (in prompt).

### Dockerize the hermetic environment

The current env relies on Linux filesystem semantics documented in this README. Containerizing makes the integrity guarantees portable and reviewable.

---

## Links

- 📦 **This repository:** https://github.com/zubairashfaque/environment_rl
- 🌐 **Blog post:** https://zubairashfaque.github.io/blog/env-rl-disciplined-training.html
- 📓 **Portfolio:** https://zubairashfaque.github.io
- 👤 **Author:** Zubair Ashfaque (mianashfaque@gmail.com)
