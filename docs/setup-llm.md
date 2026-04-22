# Configuring the LLM (Iterative Self-Refine Harness)

This guide walks through plugging a real OpenAI model into `env_rl` as the decision-maker. The result is an **Iterative Self-Refine** loop, not reinforcement learning — model weights never change. See `src/env_rl/harness/__init__.py` for why that distinction matters.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  examples/run_llm_agent.py  (CLI)                          │
│    └─ loops over N attempts                                │
│        ├─ builds system prompt (+ prior attempts' scores) │
│        ├─ runs one full training cycle:                   │
│        │    - OpenAIDecisionPolicy.decide(...)             │
│        │      (called once per rule firing)                │
│        ├─ runs the judge → (accuracy, process) scores     │
│        └─ appends this attempt's summary to "prior"       │
└────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────┐
│  src/env_rl/harness/                                       │
│    prompt.py       system prompt + JSON schema             │
│    policy.py       OpenAIDecisionPolicy + Scripted…Policy  │
│    iterative.py    multi-attempt driver + feedback loop    │
└────────────────────────────────────────────────────────────┘
```

## Step-by-step

### 1. Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys.
2. Create a new secret key.
3. Copy it immediately (you won't see it again).
4. Set a spending limit on your account (https://platform.openai.com/account/limits). A 3-attempt `gpt-4o-mini` run with 3 epochs of synthetic data costs well under \$0.10; full CIFAR-10 at 40 epochs is still only a couple of dollars.

### 2. Install dependencies

Already handled by `poetry install` — the repo pins `openai>=1.50.0`.

```bash
cd /home/zubair-ashfaque/GitHubProject/environment_rl
poetry install
poetry run python -c "import openai; print(openai.__version__)"
# 2.32.x or newer
```

### 3. Export your API key

```bash
export OPENAI_API_KEY=sk-proj-...   # paste your key
```

To make it permanent in your shell, add that line to `~/.bashrc` or `~/.zshrc`. To keep it per-project, use a `.env` file:

```bash
echo "OPENAI_API_KEY=sk-proj-..." > .env
# then in your shell, when you want to use it:
export $(grep -v '^#' .env | xargs)
```

**Never commit `.env` to git.** The repo's `.gitignore` already ignores it.

### 4. Choose a model

Default: **`gpt-4o-mini`** — fast, cheap, supports JSON-schema structured output, plenty capable for this task.

| Model | Approx. cost per attempt (3 epochs, synthetic) | Use when |
|---|---|---|
| `gpt-4o-mini` *(default)* | ~\$0.01 | Always, unless you have a reason not to |
| `gpt-4o` | ~\$0.15 | You suspect decision quality is the bottleneck |
| `gpt-4.1-mini` | ~\$0.02 | Want a newer model, still cheap |
| `o3-mini` | ~\$0.05 | Reasoning-heavy decisions, slower per call |

Pass with `--model`:

```bash
poetry run python examples/run_llm_agent.py --model gpt-4o
```

### 5. Run the harness (synthetic-data smoke test first)

```bash
poetry run python examples/run_llm_agent.py \
    --attempts 3 \
    --epochs 3 \
    --batch-size 32 \
    --synthetic \
    --model gpt-4o-mini \
    --base-dir ./llm_runs
```

The `--synthetic` flag replaces CIFAR-10 with small random tensors — no download, no GPU needed. The run completes in ~30 seconds.

**Expected output** (JSON to stdout):

```json
{
  "mode": "iterative_self_refine",
  "best_attempt": 3,
  "best_scores": {
    "accuracy_score": 0.45,
    "process_score": 1.0,
    "test_accuracy": 0.125,
    "violations": 0,
    "total_decisions": 3
  },
  "all_attempts": [
    { "index": 1, "accuracy_score": 0.40, "process_score": 0.67, "violations": 1, "total_decisions": 3 },
    { "index": 2, "accuracy_score": 0.42, "process_score": 0.67, "violations": 1, "total_decisions": 3 },
    { "index": 3, "accuracy_score": 0.45, "process_score": 1.00, "violations": 0, "total_decisions": 3 }
  ],
  "base_dir": "./llm_runs"
}
```

Each attempt's full workspace + logs + summary are persisted so you can inspect exactly what the LLM decided:

```bash
ls llm_runs/attempt_01/
# workspace/  judge_logs/  summary.json
cat llm_runs/attempt_01/summary.json | jq .violation_summary
cat llm_runs/attempt_03/judge_logs/decision_log.jsonl | jq -c '.payload | select(.kind=="decision")'
```

### 6. Run against real CIFAR-10 (when you want an actual trained model)

Prerequisites:

```bash
# Download CIFAR-10 once (~170MB)
poetry run python -c "
from torchvision import datasets
datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
"

# Build the split manifest
poetry run python -m env_rl.data.splits --seed 42 --out data/cifar10/splits.json
```

Then run the harness without `--synthetic`:

```bash
poetry run python examples/run_llm_agent.py \
    --attempts 5 \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --model gpt-4o-mini \
    --target-acc 0.75 \
    --data-dir ./data/cifar10 \
    --manifest ./data/cifar10/splits.json \
    --base-dir ./llm_runs_real
```

Budget roughly 20–30 min on a single GPU for 20 epochs × 5 attempts.

### 7. Inspect what the LLM saw and did

Every OpenAI call's messages are reconstructable from the prompt builders (no logging of raw messages needed). If you want to see the *actual* prompt sent in attempt 3, run:

```bash
poetry run python - <<'PY'
from env_rl.harness.prompt import build_iterative_system_prompt, AttemptSummary
import json

# Load prior attempts
prior = []
for i in range(1, 3):
    s = json.loads(open(f"llm_runs/attempt_{i:02d}/summary.json").read())
    prior.append(AttemptSummary(
        attempt_index=i,
        accuracy_score=s["summary"]["accuracy_score"],
        process_score=s["summary"]["process_score"],
        test_accuracy=s["scores"]["test_accuracy"],
        total_decisions=s["summary"]["total_decisions"],
        violations=s["summary"]["violations"],
        violation_summary=s["violation_summary"],
    ))

print(build_iterative_system_prompt(prior))
PY
```

Each attempt's `decision_log.jsonl` gives you the exact JSON decision the LLM returned, plus the judge's coverage + defensibility audits.

## Internals — what `OpenAIDecisionPolicy.decide()` actually does

From `src/env_rl/harness/policy.py`:

```python
def decide(self, *, top_rule, all_fired, metrics, epoch,
           current_lr, current_batch_size, recent_history):
    messages = build_decision_messages(
        system_prompt=self._system_prompt,      # playbook + response rules + priors
        epoch=epoch,
        top_rule=top_rule,                      # picked by Python precedence
        all_fired=all_fired,
        metrics=metrics,                        # live diagnostics from the monitor
        current_lr=current_lr,
        current_batch_size=current_batch_size,
        recent_history=recent_history,          # last 3 epochs
    )
    response = self._client.chat.completions.create(
        model=self._model,                      # gpt-4o-mini by default
        messages=messages,
        temperature=self._temperature,          # 0.2 — we want consistency
        response_format={                       # JSON-schema structured output
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "strict": True,
                "schema": DECISION_SCHEMA,      # enforced by OpenAI's API
            },
        },
    )
    return _decision_from_dict(json.loads(response.choices[0].message.content), ...)
```

Three things the LLM does NOT see:

1. The test split. It's held out at the filesystem level.
2. Historical metrics beyond the last 3 epochs. Keeps the prompt small.
3. Its own conversation from prior attempts. Only the scores+violations summary is carried forward.

## Tuning knobs

| Flag | Default | Effect |
|---|---|---|
| `--attempts` | 3 | More attempts → better chance of convergence, linear cost |
| `--epochs` | 3 | More epochs per attempt → more decisions per run, more signal |
| `--model` | gpt-4o-mini | `gpt-4o` for harder judgment calls, `gpt-4.1-mini` for newer |
| `--temperature` | 0.2 | Higher = more exploration across attempts; 0.0 for deterministic |
| `--target-acc` | 0.20 (syn) / 0.92 (real) | Acts as the saturation point of `accuracy_score` |

## Common pitfalls

1. **"`OPENAI_API_KEY is not set`"** — you skipped step 3, or the var didn't survive a new shell. Add it to `~/.bashrc`.
2. **JSON schema validation errors** — OpenAI's API enforces the schema in `prompt.py:DECISION_SCHEMA` with `strict=True`. If you see unexpected validation failures, some rules may have been updated; re-run `poetry install` in case the `openai` package was bumped.
3. **Live diagnostic tolerance too tight** — real runs use `live_diag_tolerance=0.50` by default; synthetic uses `0.99`. If a real run hard-fails at judge step 7, relax this or train for more epochs so the LLM has more signal to work with.
4. **Cheap models can ignore precedence** — if `gpt-4o-mini` starts consistently ignoring the stability > capacity > tuning > process order, upgrade to `gpt-4o`. The iterative feedback will tell you: look for `precedence_violation` in `violation_summary`.

## This is not reinforcement learning

If anyone asks: this harness changes the *prompt* between attempts, never the *model weights*. The OpenAI model is used via a frozen, public API — you have no weights access. Real RL would require:

1. Weights access (open-weights model: Llama, Qwen, Mistral).
2. Per-decision reward (not the sparse end-of-run scores we produce now).
3. A training loop that updates the model — PPO, DPO, GRPO, etc.

None of that exists in `env_rl` today. If you want to add it, the natural home is a new `src/env_rl/rl/` module. This `harness/` module stays self-refine by design.
