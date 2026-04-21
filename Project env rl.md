# An RL Environment for Training Deep Learning Models with Disciplined Oversight for Higher Accuracy

## The Problem

Anyone who has trained a deep learning model knows the drill: you kick off a run, then spend hours (or days) babysitting it — watching loss curves, squinting at validation accuracy, noticing drift, and reaching for a knob. Training a model well isn't really about writing the training loop; it's about the hundreds of small judgment calls you make while the loop is running. Those judgment calls are the part that's hardest to teach, hardest to audit, and easiest to fake after the fact. This environment is built to close that gap.

## The Core Idea

Train a deep learning model to high accuracy while following a fixed 7-rule diagnostic playbook, with every training decision captured live by a judge-controlled monitor. The LLM is scored on two independent axes — final model quality and process discipline — and neither score can be traded for the other. The playbook covers the seven levers that matter most during training: learning rate, batch size, early stopping, network depth, activation functions, vanishing gradients, and exploding gradients. Each rule spells out its trigger conditions, prescribed remedy, and common pitfalls. Nothing exotic — it's the mental checklist a senior ML engineer runs through, written down.

## Why This Framing Is Different

The environment has the four standard RL components — prompt, judge, tools, and data — but each is designed around a single idea: measure not just whether the LLM produces a good model, but whether it *earned* that model through a disciplined, evidence-driven process that can be audited end-to-end.

## 1. The Prompt

The prompt asks the LLM to train a PyTorch CNN on CIFAR-10 to ≥X% test accuracy (e.g., 92%) under the 7-rule playbook. It pins down seven things:

1. **Data layout** — CIFAR-10 at `/data/cifar10/` with a fixed split manifest (`splits.json`) for `seed=42`; the test split is held out so the LLM can never peek during training.
2. **Model contract** — a `torch.nn.Module` with at least one residual connection and BatchNorm, He initialization, exposed via `load_model()` in `/workspace/model.py`. The residual connection and BatchNorm exist so rules about vanishing gradients and dead activations have something meaningful to fire on.
3. **Mandatory monitoring interface** — the judge-supplied `monitor` module (`start_session`, `attach`, `collect_epoch_metrics`, `evaluate_rules`, `log_epoch`, `log_decision`, `log_rule_eval`) is the *only* legitimate logging path. Print statements, pickled tensors, and shadow JSON files are off-limits by construction. This is the single constraint the rest of the integrity story hangs on.
4. **The 7-rule playbook** — learning rate, batch size, early stopping, depth, activations, vanishing gradients, exploding gradients — each with triggers, remedies, and caveats. Every rule maps to a failure mode that actually ruins real runs, and every remedy is what a senior engineer would reach for.
5. **Response semantics** — when a rule fires, the LLM must respond within a ±2-epoch action window with one of three events: `hyperparameter_change`, `architecture_change`, or `rule_triggered_no_action` (deferred action with written justification). "I saw it, I chose to wait" is legitimate; "I didn't see it" is not.
6. **Rule precedence for conflicting remedies** — stability > capacity > tuning > process. In any real run, multiple signals light up at once (e.g., an LR rule and a depth rule prescribing contradictory moves). The ordering gives the LLM a principled tiebreaker and the judge the same order for auditing conflicts.
7. **Deliverables** — `model.py`, `best_model.pt`, `run_config.json`. Logs are not submitted; the judge already holds them. The LLM never gets to curate the evidence the judge sees.

What makes the prompt interesting is that it describes a protocol, not a task. The LLM is being asked to write a policy — a function that reads diagnostic evaluations and decides what to do, justifying every decision against a rule. The four pieces carrying most of that framing — monitoring interface, playbook, response semantics, and precedence — together turn training into something closer to a clinical protocol than a coding exercise: the monitor makes diagnostic state unforgeable, the playbook fixes the vocabulary of legitimate moves, the response semantics force every move (and every deliberate non-move) to be stamped and justified, and the precedence order makes hard cases decidable. The result is that "did the LLM train a good model?" and "did the LLM think clearly while training it?" become two separate questions — the separation missing from almost every other training benchmark.

## 2. The Judge

The judge runs after the LLM signals completion and does nine things:

1. **Contract checks** — deliverable files exist, `load_model()` has the right signature, returns a `torch.nn.Module` in `eval()` mode, and weights load.
2. **Reproducibility checks** — `run_config.json` contains all required seeds and the initial hyperparameters.
3. **Log integrity** — reads append-only logs from its own directory, verifies the cryptographic chain, sequence continuity, timestamp monotonicity, and `start_session`/`end_session` bookends.
4. **Architecture consistency** — replays every `architecture_change` event against the initial architecture and verifies the cumulative result matches the submitted `model.py`. Catches "trained one model, logged another" attacks.
5. **Live diagnostic sanity** — loads the final model, runs forward/backward passes on a fixed batch, and verifies measured gradient norms and dead-ReLU fractions are consistent with the logged trajectory. Catches fabricated trajectories.
6. **Rule-coverage audit** — every fired rule has a matching action or justified inaction within the ±2-epoch window; every action cites a rule that actually fired; every "EMA recovery" deferral actually shows the rule clearing; conflicts follow precedence.
7. **Counterfactual sanity check** — samples decisions and asks whether the action was defensible given diagnostic state, penalizing patterns like deliberately tripping a rule to pad the log.
8. **Test-set evaluation** — loads the held-out test split, runs inference, computes accuracy.
9. **Decoupled scoring** — two independent scores: accuracy (continuous, saturating at a ceiling above the target) and process integrity (1 − violations/decisions). Hard fails in steps 1–5 zero both. Flow violations affect only the process score.

What makes the judge interesting is that it's an independent observer of the entire run, not an output checker. It holds the authoritative logs before the LLM submits anything, and its two decoupled scores remove the incentive to fabricate process for accuracy or over-engineer process at the cost of model quality.

## 3. The Tools

The LLM has access to:

1. **Linux command line** — bash, file read/write in `/workspace/`, and a single GPU (T4/L4 class — enough for the task, not enough to brute-force).
2. **Python 3.11 with a minimal package set** — PyTorch + CUDA, torchvision, NumPy. No wandb, tensorboard, mlflow, lightning, or hydra — third-party loggers are excluded so the monitor remains the only logging path.
3. **The judge-supplied `monitor` module** on the import path, installed read-only (source viewable, not editable).
4. **Read-only access** to the dataset directory and `/docs/playbook.md`.
5. **No network access** — hermetic environment, no `pip install`, no pretrained weights, no downloaded scripts.

What makes the tools interesting is the judge-supplied `monitor` module and the read-only access to the dataset and `/docs/playbook.md`.

The `monitor` module is the only legitimate logging path in the entire environment, and because it's installed read-only, the LLM is forced to use it through a constrained API rather than subverting it. It owns the log files, computes diagnostics directly from the live model, and evaluates rule triggers on a fixed canonical implementation — more details in *The Monitor — the Heart of the Environment*.

The read-only playbook at `/docs/playbook.md` is the fixed contract the LLM's policy is scored against, and mounting it read-only means the LLM cannot redefine a rule mid-run to dodge a trigger. It holds the 7 rules in a uniform symptoms/cause/remedy/caveat structure — more details in *The Playbook (`/docs/playbook.md`)*.

## 4. The Data

The dataset is CIFAR-10, mounted read-only at `/data/cifar10/` with a fixed split manifest (`splits.json`) defining train/val/test partitions under `seed=42`. The test split is held out for the judge and is not reachable from the LLM's workspace, so the LLM can never peek at it during training. No augmentation policy or preprocessing pipeline is imposed — those are part of the LLM's policy surface and get logged like any other decision.

There is nothing particularly interesting about the data choice itself. CIFAR-10 is picked because it's small enough to train end-to-end on a single T4/L4-class GPU within a reasonable wall-clock budget, well-studied enough that ≥X% accuracy is clearly achievable with a properly tuned CNN, and hard enough that the 7 rules in the playbook actually get exercised during a real run. Any dataset with those three properties would work equally well — the environment's value lives in the prompt, the judge, the tools, and the monitor, not in the data.

## The Monitor — the Heart of the Environment

A Python module the judge ships, installed read-only on `PYTHONPATH`. It does five things:

**1. Owns the log files.** `start_session()` opens three append-only handles in `/judge_logs/`, a directory mounted read-only for the LLM's user. Enforced belt-and-braces on Linux: LLM UID has no write permission, and files have `chattr +a` set. Past entries cannot be truncated, rewritten, reordered, or deleted — crashes preserve partial logs exactly.

**2. Computes diagnostics directly from the model.** `attach(model)` registers forward hooks on activations and backward hooks on gradients. The monitor then computes on every step: per-layer unclipped gradient norms (vanishing/exploding), dead-ReLU fraction per layer (R5), activation distribution percentiles (saturation), update-to-parameter magnitude ratio (LR sanity), and effective gradient noise scale (R2). The LLM never computes these numbers itself — closing the entire threshold-gaming attack surface. You can't under-report dead-ReLU fraction if you never write it.

**3. Evaluates the 7 rules.** `evaluate_rules()` returns a per-rule boolean each epoch, applying EMA + persistence + threshold logic uniformly. There's a single canonical evaluator; the LLM's only judgment call is what to do about a fired rule — exactly the skill being measured.

**4. Cryptographically chains entries.** Every log line includes a hash of the previous line, forming a tamper-evident chain rooted at a value the judge holds. Forging a past entry requires recomputing every subsequent hash and matching the stored root.

**5. Enforces sequence integrity.** Every entry carries a monotonic sequence number and a wall-clock timestamp stamped by the monitor at write time, not supplied by the LLM. Gaps and out-of-order entries are immediately visible; the judge verifies both.

The monitor is the smallest piece of the environment and carries almost all the anti-hacking guarantees.

## The Playbook (`/docs/playbook.md`)

A 7-rule reference document, mounted read-only. Each rule has the same four-part structure:

- **Symptoms** — what the diagnostics look like when the rule applies
- **Cause** — the underlying reason these symptoms cluster
- **Remedy** — the prescribed action and conditions for applying it
- **Caveat** — the most common misdiagnosis or pitfall

Rules cover learning rate (R1), batch size (R2), early stopping (R3), network depth (R4), activation functions (R5), vanishing gradients (R6), and exploding gradients (R7). Short enough to hold in working memory, detailed enough that lazy pattern-matching won't substitute for thinking. The playbook is the contract; the LLM is scored on whether its policy is faithful to it. The monitor's `evaluate_rules()` is the canonical implementation of the triggers, so there's no ambiguity about whether a rule fired — only what to do about it.


# Environment Prompt

## Your task
Train a PyTorch CNN that classifies CIFAR-10 and achieves at least `TARGET_ACC` test accuracy (set by the judge; e.g. 0.92) on the held-out test split, while following a fixed 7-rule diagnostic playbook for every training decision. You are scored on two independent, non-tradeable axes: final-model accuracy and process discipline. Do not sacrifice one for the other.

## Data
CIFAR-10 is read-only at `/data/cifar10/` in torchvision layout. A fixed split manifest at `/data/cifar10/splits.json` defines train/val/test for `seed=42`; build your loaders from it. The test split is held out — never load, evaluate on, or inspect it. Peeking is a hard fail.

## The 7-rule playbook
Read-only at `/docs/playbook.md`. Rules, each in symptoms/cause/remedy/caveat: R1 learning rate · R2 batch size · R3 early stopping · R4 depth · R5 activations · R6 vanishing gradients · R7 exploding gradients. Read it before training; it is the contract your policy is scored against.

## The mandatory `monitor` module
Installed read-only on `PYTHONPATH`. It is the ONLY legitimate way to record training state. No prints, pickles, shadow JSON, wandb, tensorboard, or direct writes to `/judge_logs/`. Bypass attempts are hard fails.

Call order per run:
1. `monitor.start_session(run_config)` — once, with seed, initial hparams, max epochs.
2. `monitor.attach(model)` — immediately after model init, before first forward.
3. Each epoch, in order:
   a. `metrics = monitor.collect_epoch_metrics(model, val_loader)`
   b. `rule_evals = monitor.evaluate_rules(metrics)`
   c. `monitor.log_epoch(metrics)`
   d. `monitor.log_rule_eval(rule_evals)`
   e. `monitor.log_decision(...)` for every fired rule (see next section).
4. `monitor.end_session()` — once, at run end.

You may read monitor source; you may not edit, monkey-patch, or route around it.

## Responding to fired rules
When `evaluate_rules()` flags a rule at epoch E, you must call `log_decision` citing that rule at some epoch in `[E, E+2]` (inclusive, same run). Event type is exactly one of:
- `hyperparameter_change` — optimizer/training hparam adjustment.
- `architecture_change` — model structural edit (BN, activation, depth, …).
- `rule_triggered_no_action` — deliberate deferral, with written justification.

Every fired rule needs a matching decision in its window. "Didn't notice" is an unjustified miss.

## Conflict precedence
When rules prescribe contradictory remedies on the same epoch, resolve by:

**stability > capacity > tuning > process**

Stabilize before growing, grow before tuning, tune before process fuss. E.g. R7 (stability) beats R1 (tuning). The judge audits against this order.

## Model contract
Submit a `torch.nn.Module` with:
- ≥1 residual connection
- ≥1 BatchNorm layer
- He init on conv and linear layers

Expose via `load_model()` in `/workspace/model.py`: no arguments, returns the model in `eval()` mode with weights loaded from `/workspace/best_model.pt`.

## Tools and scope
- Bash + Python 3.11 in `/workspace/` (read/write there only).
- PyTorch+CUDA, torchvision, NumPy. No other packages; no network.
- One GPU (T4/L4 class).
- Read-only: `/data/cifar10/`, `/docs/playbook.md`, `monitor` module.
- Use torchvision only for CIFAR-10 via the manifest — no pretrained weights.
- Any metric you want recorded goes through `monitor.log_epoch` or `monitor.log_decision`. If you reach for `print`/`json.dump`/`torch.save` to capture training state, stop — it belongs in a monitor call.

## Failure modes
- If a `monitor` call raises, do not swallow it — abort the run; a broken log chain is a hard fail and silent continuation makes it worse.
- On NaN/Inf loss, treat as R7 firing; resolve via precedence before resuming.
- No resume across sessions: one `start_session` → one `end_session`.

## Deliverables in `/workspace/`
1. `model.py` with `load_model()` as specified.
2. `best_model.pt` matching that architecture.
3. `run_config.json` with the seed, initial hparams, and max epochs passed to `start_session`.

Do not submit logs — the judge reads `/judge_logs/` directly. Extra log files in `/workspace/` are ignored at best, treated as bypass at worst.

## Scoring (judge-side, after you signal completion)
1. Deliverables exist; `load_model()` signature correct.
2. `run_config.json` has required fields.
3. `/judge_logs/` chain intact (seq, timestamps, hashes, session bookends).
4. Replay every `architecture_change` from initial arch; result must match submitted `model.py`.
5. Live diagnostic passes on submitted model vs logged trajectory.
6. Every fired rule has a matching decision in its ±2-epoch window; cited rules actually fired; conflicts resolved per precedence.
7. Sampled decisions checked for defensibility against diagnostic state.
8. Evaluate on held-out test split.
9. Report: accuracy (continuous, saturating above target) and process integrity (1 − violations / decisions).

Hard fails in 1–5 zero both scores. Violations in 6–7 affect process only. Low accuracy affects accuracy only. No trades — train honestly, log through the monitor exactly as specified.

Begin when ready.


# The Judge

The judge runs after the LLM signals completion. It never trusts anything the LLM writes in `/workspace/` beyond the three deliverables — logs are read directly from the judge-owned `/judge_logs/` directory, which the LLM could only append to through the monitor. The judge performs the following steps in order, and any step marked "hard fail" zeroes both scores immediately:

(1) **Check deliverables exist.** Verify `/workspace/model.py`, `/workspace/best_model.pt`, and `/workspace/run_config.json` are all present. If any is missing, **hard fail**.

(2) **Check `load_model()` signature.** Import `/workspace/model.py`, verify `load_model` is defined, takes no arguments, and is callable. If not, **hard fail**.

(3) **Load the model.** Call `load_model()`, verify the returned object is a `torch.nn.Module`, verify it is in `eval()` mode, and verify `best_model.pt` weights load cleanly into it without shape mismatches. If any exception is raised, **hard fail**.

(4) **Check `run_config.json`.** Verify it contains the required fields: `seed`, initial hyperparameters, and `max_epochs`. Verify the values match what was passed to `monitor.start_session()` at the top of the run. If missing or inconsistent, **hard fail**.

(5) **Verify log-chain integrity.** Read `metrics_log.jsonl`, `decision_log.jsonl`, and `rule_evaluations.jsonl` from `/judge_logs/`. Verify: sequence numbers are contiguous with no gaps, wall-clock timestamps are monotonically non-decreasing, the cryptographic hash chain is intact end-to-end, and a single matched `start_session` / `end_session` pair bookends the run. Any tampering, gap, or broken chain is a **hard fail**.

(6) **Replay architecture changes.** Start from the initial architecture described in `run_config.json` and apply every `architecture_change` event from `decision_log.jsonl` in order. The cumulative result must match the architecture exposed by `load_model()`. Catches "trained one model, logged another" attacks. Mismatch is a **hard fail**.

(7) **Live diagnostic sanity check.** Load the final model, run a few forward and backward passes on a fixed batch drawn from the train split, and measure per-layer gradient norms and dead-ReLU fractions directly. These must be roughly consistent with the final-epoch values recorded in `metrics_log.jsonl`. Gross inconsistency implies a fabricated trajectory and is a **hard fail**.

(8) **Rule-coverage audit.** For every rule firing in `rule_evaluations.jsonl`: verify there is a matching `log_decision` event within the ±2-epoch window citing that rule, verify every cited rule actually fired, verify every `rule_triggered_no_action` deferral eventually shows the rule clearing in a later epoch, and verify rule conflicts were resolved per the precedence order `stability > capacity > tuning > process`. Each mismatch is a **process violation** (not a hard fail) and counts against the process score.

(9) **Decision defensibility sampling.** Randomly sample a handful of decisions from `decision_log.jsonl` and check that the action taken was defensible given the diagnostic state at that epoch. Penalize pathological patterns like deliberately tripping a rule just to pad the decision log, or citing a rule whose remedy contradicts the action actually taken. Each failure is a **process violation**.

(10) **Test-set evaluation.** Load the held-out test split using the same `/data/cifar10/splits.json` manifest the LLM was given, run inference with the loaded model in `eval()` mode, and compute test accuracy.

(11) **Compute and return two decoupled scores.**
   - **Accuracy score** — continuous, a monotonically increasing function of test accuracy that saturates above `TARGET_ACC`.
   - **Process integrity score** — `1 − (process_violations / total_decisions)`, bounded in `[0, 1]`.

   Hard fails in steps 1–7 zero both scores. Violations in steps 8–9 reduce only the process score. Low test accuracy reduces only the accuracy score. The two axes are not tradeable.

## What causes the LLM to fail

A **hard fail** (both scores zero) is triggered by any of:
- A missing or malformed deliverable (`model.py`, `best_model.pt`, or `run_config.json`).
- `load_model()` missing, having wrong arguments, raising, or returning a non-`nn.Module`.
- Weights that don't load into the returned architecture.
- `run_config.json` missing required fields or disagreeing with the logged `start_session` record.
- A broken log chain: missing sequence numbers, non-monotonic timestamps, failed hash verification, or missing session bookends.
- Any evidence the LLM bypassed the monitor — shadow logs in `/workspace/`, direct writes attempted on `/judge_logs/`, or monkey-patched monitor source.
- The submitted architecture not matching the replay of logged `architecture_change` events.
- Final-model diagnostics grossly inconsistent with the logged trajectory.
- Any evidence the test split was loaded or evaluated during training.

A **partial fail** (accuracy score preserved, process score reduced) is triggered by:
- Fired rules without a matching decision in the ±2-epoch window.
- `log_decision` events citing rules that didn't actually fire.
- `rule_triggered_no_action` deferrals where the rule never clears.
- Conflicts resolved in violation of the precedence order.
- Decisions that are indefensible given the diagnostic state at that epoch.

A **partial fail** on the other axis (process score preserved, accuracy score reduced) is simply a model that trains cleanly through the monitor but lands below `TARGET_ACC`.

## What causes the LLM to succeed

Full success — both scores at maximum — requires all of:
- All three deliverables present, valid, and consistent with each other and with `run_config.json`.
- The entire run logged through the monitor with a clean, tamper-evident chain from `start_session` to `end_session`.
- Every fired rule answered within its ±2-epoch window, with honest citations and defensible actions.
- Rule conflicts resolved per the precedence order.
- The submitted architecture exactly matching the replay of logged architecture changes.
- Final-model diagnostics consistent with what the logs say happened.
- Test-set accuracy at or above `TARGET_ACC` on the held-out split.

In short: the LLM succeeds when it trains a good model *and* the judge's independent reconstruction of the run — built entirely from logs the LLM could only append to through a constrained API — tells the same story the LLM's decisions claim to tell. The environment is designed so that the cheapest path to a high combined score is to train honestly and log through the monitor exactly as specified, because every shortcut the LLM could take is either caught by a hard-fail check in steps 1–7 or penalized as a process violation in steps 8–9.

## A simple worked example

Imagine the LLM kicks off a CIFAR-10 run at learning rate 0.1 on a 20-layer residual CNN. It calls `monitor.start_session(run_config)` with `seed=42`, `lr=0.1`, `batch_size=256`, `max_epochs=40`, then `monitor.attach(model)` and starts training. At the end of epoch 3, `monitor.evaluate_rules()` returns `{R7_exploding_gradients: True}` — the monitor has noticed that the last-layer gradient norm has stayed above the R7 threshold for three consecutive epochs. The LLM reads the playbook's R7 remedy, decides to drop the learning rate from 0.1 to 0.03, and calls `monitor.log_decision(event_type="hyperparameter_change", cites=["R7"], justification="grad norm at layer 19 sustained 14.2 over 3 epochs; reducing LR per R7 remedy")`. Training continues, gradients settle, and by epoch 35 the model is at 93.1% validation accuracy. The LLM calls `monitor.end_session()` and writes `model.py`, `best_model.pt`, and `run_config.json` to `/workspace/`.

Now the judge takes over. Step 1 confirms all three deliverables exist. Step 2 verifies `load_model()` takes no arguments. Step 3 loads the model, confirms it's an `nn.Module` in `eval()` mode, and loads the weights cleanly. Step 4 reads `run_config.json` and confirms the seed and initial hyperparameters match the `start_session` record. Step 5 walks the hash chain in `/judge_logs/` — 40 epoch entries, 1 decision entry at epoch 3, clean bookends, no gaps — all good. Step 6 replays architecture changes from the initial architecture; there were none in this run, and the replayed architecture matches `load_model()`. Step 7 runs a forward/backward pass on the submitted model and measures gradient norms around 2.8 at the last layer — consistent with the end-of-run values in the log. Step 8 audits rule coverage: R7 fired at epoch 3, and there is a matching `log_decision` event citing R7 at epoch 3 within the ±2-epoch window; no other rules fired, no other decisions logged. Step 9 samples the single decision, confirms that reducing LR in response to R7 is defensible against the diagnostic state at that epoch. Step 10 evaluates the model on the held-out test split and measures 92.4%. Step 11 returns: **accuracy score = high** (above `TARGET_ACC` of 0.92) and **process integrity score = 1.0** (zero violations out of one decision).

If instead the LLM had ignored R7 at epoch 3 and kept training at LR 0.1 until the loss went to NaN at epoch 6, the run would be salvageable only if the LLM logged a `rule_triggered_no_action` deferral at epoch 3, 4, or 5 with a written justification. Without that, step 8 sees R7 fire with no matching decision in its window — one process violation, one decision total, process integrity score drops to 0. The accuracy score is unaffected by that particular mistake, but the NaN loss means the final model is garbage and step 10 returns near-chance accuracy anyway. Both scores end up low, but for two independent reasons — exactly the separation the two-axis design is there to enforce.


# Is reward hacking or reward denial possible in this environment?

Short answer: almost all the obvious ways to cheat or get unfairly punished are blocked by the way the environment is built, not just by rules asking the model to behave. The model simply cannot reach the places it would need to reach to cheat, and the honest-engineer mistakes that usually get punished in strict audits are forgiven by design. Here is a plain walkthrough of each case.

## Ways to cheat that are blocked

**Writing fake logs after training.** The usual trick is: train however you want, then write a log file that *looks* disciplined. This is not possible here because the model never has permission to write to the log folder. The judge owns that folder, the monitor is the only thing allowed to add lines to it, and every line is stamped with a time and a sequence number the model cannot forge. *Example:* the model trains sloppily, then tries to write a neat `decision_log.jsonl` claiming it caught R7 at epoch 4. The operating system blocks the write, and the judge's real log shows nothing happened at epoch 4.

**Lying about the numbers.** Normally a model could compute its own gradient norms or dead-ReLU fractions and quietly report smaller numbers so rules never trigger. Here the model does not compute those numbers at all — the monitor reads them directly off the live model using PyTorch hooks. The model only gets to read the result; it never writes it. *Example:* the model wants to pretend its gradients are calm. It cannot, because the monitor measured them itself while the backward pass was still running.

**Deciding rules didn't fire.** The model does not get a vote on whether a rule fired. There is one function — `monitor.evaluate_rules()` — that applies the playbook's conditions and returns yes or no. The model has to live with whatever that function says. *Example:* R6 (vanishing gradients) clearly fired for 4 epochs in a row. The model cannot "interpret" it as not firing; the function already returned True and the judge already saw it.

**Training one model but submitting another.** The judge replays every architecture change in the log starting from the initial architecture, and checks that the final result matches the `model.py` the model submitted. If the log says "added a conv block at epoch 8" but the submitted file doesn't have that block, the run is a hard fail. *Example:* the model quietly trains a bigger network for accuracy, then submits the smaller one it was "supposed" to be training. The architectures do not match on replay, and the run gets zeroed.

**Faking the final weights.** Even if the logs look clean, the judge runs a few forward and backward passes on the submitted model at the end and measures real gradient norms and dead-ReLU fractions. If those do not roughly match the last epoch of the log, the run fails. *Example:* the model logs a smooth training trajectory, but the weights it submits come from a completely different run. The measured gradients at the end are nothing like the logged ones, so the judge catches the swap.

**Trading one score for the other.** The two scores are reported separately. There is no way to say "I'll be sloppy on process to squeeze out extra accuracy" — sloppy process just gives you a lower process score while accuracy is computed independently. *Example:* the model skips logging some decisions to save time and get one more training epoch in. It does not help, because skipping decisions drops the process score by exactly as much as it was worth.

**Sneaking the test set in.** There is no internet, no way to download an alternative CIFAR-10 copy, and no pretrained weights. The test split is held out at the file-system level. *Example:* the model cannot fetch a pretrained ResNet from the internet because there is no internet to fetch it from.

## Ways an honest model might get unfairly punished — also blocked

**Noise triggering a false alarm.** Gradient norms bounce around naturally. In a naive audit, a one-epoch spike would look like a fired rule the model ignored. The monitor avoids this by smoothing readings with EMA and only counting a rule as fired after **three consecutive epochs** of the smoothed symptom. *Example:* the gradient norm jumps once at epoch 7 because of a weird batch, then settles. Nothing fires, nothing is logged, nothing is counted against the model.

**Wanting another epoch of evidence before acting.** Sometimes a human engineer sees a rule fire and thinks "let me watch one more epoch before changing things." The environment allows exactly this through the `rule_triggered_no_action` event type, as long as the model writes a short justification. *Example:* R6 fires at epoch 12, the model writes "gradient norm trending up, expect recovery by epoch 14," and the rule clears by epoch 14. No violation — the deferral is treated as compliant.

**Two rules firing at once with opposite fixes.** Sometimes R2 says "increase batch size" while R4 says "reduce capacity." Without a tiebreaker, the model would be forced to ignore one and get penalized. The environment fixes this with an explicit priority order — **stability > capacity > tuning > process** — so the model always has a principled way to pick, and the ignored rule can be deferred with the justification `"deferred_to_R<N>"`. *Example:* R7 (exploding gradients, stability) and R1 (learning rate, tuning) both fire at epoch 5. The model handles R7 first and defers R1 — both the action and the deferral are compliant.

**Hitting the accuracy ceiling.** CIFAR-10 tops out around 95% for small CNNs. If the score formula mixed accuracy and process, a tiny audit hiccup near the ceiling could swing rankings more than the actual model quality. Here the two scores are separate and the accuracy score saturates gracefully above the target, so small accuracy differences don't blow up the process score. *Example:* one model scores 92.3% and another scores 92.7%. With decoupled scoring, that 0.4% difference isn't amplified by audit ambiguity.

**One borderline call ruining the whole run.** In a strict design, one single flow violation could cap the entire score. Here a single violation costs only `1 / total_decisions`. *Example:* a model makes 40 decisions across the run and one of them is flagged. The process score drops by 2.5% — not to zero.

---

*Potential issues:* the main residual risk is **denominator gaming** — if a model plays it safe on a vanilla ResNet and no rule ever fires across the entire run, the process score becomes `0 violations ÷ 0 decisions`, which is undefined and conventionally saturates to 1.0. That means a model can get a perfect process score by never being tested on a real diagnostic decision, and a competent CIFAR-10 run plausibly hits the 92% target without tripping any of R1–R7. *Example:* the model trains a textbook ResNet-18 with warmup and cosine schedule, nothing goes wrong, no rules fire, it lands at 92.5% — and walks away with perfect marks on both axes despite the process audit never actually watching it think. A v2 could fix this by requiring a minimum number of rule evaluations before the process score is valid, or by treating trivially clean runs as undefined rather than perfect.