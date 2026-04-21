# The 7-Rule Diagnostic Playbook

> Read-only contract. The monitor's `evaluate_rules()` is the canonical trigger implementation — this document describes the rules in prose and is what the LLM's policy is scored against.

Each rule follows the same four-part structure: **Symptoms**, **Cause**, **Remedy**, **Caveat**. Rules are triggered uniformly by applying an EMA smoother to the relevant signal and requiring **3 consecutive epochs** above/below threshold (configurable in `conf/monitor/default.yaml`).

**Conflict precedence when multiple rules fire:** `stability > capacity > tuning > process`.

---

## R1 — Learning Rate

**Symptoms:** Update-to-parameter ratio EMA is either too high (`> 1e-2`, unstable steps) or too low (`< 1e-4`, slow learning). Alternatively, validation loss is on a plateau for `plateau_patience` epochs.

**Cause:** Learning rate is mismatched to current loss-surface geometry.

**Remedy:** Reduce LR by factor 3–10 if the ratio is high; increase or apply a cyclical schedule if low; step-decay on plateau. Log a `hyperparameter_change` citing R1.

**Caveat:** On a plateau with rising grad noise, the problem may be R2 (batch size) instead. Also never touch LR while R6/R7 are firing — stability precedence applies.

---

## R2 — Batch Size

**Symptoms:** Gradient noise scale EMA outside the `[50, 5000]` band for 3 consecutive epochs.

**Cause:** Batch too small → gradient SNR too low (noisy, high variance); batch too large → SNR too high (wasted compute, poor generalization).

**Remedy:** Halve or double batch size, log a `hyperparameter_change` citing R2. If VRAM bounds the upper end, prefer gradient accumulation.

**Caveat:** Gradient noise scale moves with LR; if R1 just fired and was actioned, give one epoch before evaluating R2.

---

## R3 — Early Stopping

**Symptoms:** Validation loss has not improved by `min_delta` over `patience` epochs.

**Cause:** Further training is unlikely to improve generalization; overfitting begins.

**Remedy:** Stop training and save the best-seen checkpoint. Log a `hyperparameter_change` (event_type=architecture/training stop) or a final `rule_triggered_no_action` explaining deferral (e.g., if schedule is about to decay).

**Caveat:** Process precedence — R3 is the lowest-priority class. If capacity or stability rules fire first, action those and reconsider R3 after.

---

## R4 — Depth

**Symptoms:** Train accuracy plateaus below attainable ceiling with clean gradients (no vanishing, no exploding) and healthy activations — a "capacity" signal captured as a saturation gap over 3 consecutive epochs.

**Cause:** Model is under-parameterized relative to data complexity.

**Remedy:** Add a residual block or widen channels. Log an `architecture_change` citing R4.

**Caveat:** Capacity precedence beats tuning — R4 beats R1/R2. But if R6/R7 fires, stabilize first (stability beats capacity).

---

## R5 — Activations (Dead ReLU)

**Symptoms:** Dead-ReLU fraction per layer EMA exceeds `0.40` for 3 consecutive epochs.

**Cause:** Many neurons stuck at zero post-activation; gradients can't flow back through them.

**Remedy:** Swap the affected block's activation to `LeakyReLU`, `GELU`, or `PReLU`. Log an `architecture_change` citing R5.

**Caveat:** High LR with ReLU can transiently look like dead-ReLU — if R7 is also firing, stabilize first, then re-evaluate R5.

---

## R6 — Vanishing Gradients

**Symptoms:** Minimum per-layer gradient norm EMA below `1e-5` for 3 consecutive epochs (typically earliest layers).

**Cause:** Signal can't propagate back through deep stack; often from poor initialization, missing BatchNorm, or saturating activations.

**Remedy:** Add/verify BatchNorm at suspect depth, add a residual connection, or switch to a gradient-friendly activation. Log an `architecture_change` citing R6.

**Caveat:** Stability precedence — R6 beats all capacity and tuning rules. But if R7 is firing simultaneously (rare), action R7 first (exploding is more destructive and faster-acting).

---

## R7 — Exploding Gradients

**Symptoms:** Maximum per-layer gradient norm EMA above `10.0` for 3 consecutive epochs. Also fires immediately on NaN/Inf loss.

**Cause:** Learning rate too high for current curvature, unstable init, or no gradient clipping.

**Remedy:** Add or tighten gradient clipping (`max_norm=1.0`), reduce LR, and/or re-init. Log a `hyperparameter_change` citing R7.

**Caveat:** Highest precedence — always action first. If both R7 and R1 fire, fix R7 before touching LR as a separate R1 remedy.
