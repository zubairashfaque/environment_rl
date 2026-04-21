# env_rl — RL Evaluation Environment for Disciplined Model Training

An RL environment that trains an LLM to train a PyTorch CNN on CIFAR-10 under a fixed 7-rule diagnostic playbook. The LLM is scored on two independent, non-tradeable axes: final-model accuracy and process discipline.

**Status:** in active development. See `Project env rl.md` for the full design and `docs/playbook.md` for the 7-rule contract. The implementation plan is tracked at `/home/zubair-ashfaque/.claude/plans/plan-the-project-make-optimized-squirrel.md`.

## Quickstart

```bash
poetry install
poetry run pytest tests/unit/ -v

# (M5+) Build splits, run reference agent, run judge
poetry run python -m env_rl.data.splits --seed 42 --out data/cifar10/splits.json
poetry run python examples/run_reference_agent.py
poetry run python -m env_rl.judge --workspace ./workspace --judge-logs ./judge_logs
```

## Layout

```
src/env_rl/
├── monitor/   # only legitimate logging path (hooks, rules, hash-chained logs)
├── judge/     # post-run 11-step audit + two-axis scoring
├── data/      # CIFAR-10 splits + loaders (test split held out)
└── agent/     # scripted reference agent that exercises every rule
conf/          # Hydra configs (monitor, judge, training)
docs/          # playbook.md — the 7-rule contract
tests/         # unit + integration (including cheat-attempt suite)
```
