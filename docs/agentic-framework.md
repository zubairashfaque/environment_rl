# Agentic Framework — What We Use and Why

## Short answer

**Env-RL does not use an external agentic framework.** No LangChain. No LangGraph. No CrewAI. No AutoGen. No DSPy. No framework at all.

The whole thing runs on:

1. The **OpenAI Python SDK** — direct `chat.completions.create(...)` calls
2. **OpenAI Structured Outputs** — JSON Schema with `strict=true` at the API level
3. **Hand-rolled Python orchestration** — plain dataclasses, dependency injection, explicit control flow

That's it. Six agents coordinate via plain Python objects. Each one is a small class with a narrow responsibility, unit-testable with a mock client.

## The six agents and how they compose

```
┌───────────────────────────────────────────────────────────────────────┐
│                          env_rl.harness                                │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │ Decision LLM │   │    Tuner     │   │   Adversary  │  ← 6 agents  │
│  │ (policy.py)  │   │  (tuner.py)  │   │(adversary.py)│    total     │
│  └──────────────┘   └──────────────┘   └──────────────┘              │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │    Tester    │   │     Judge    │   │   Distiller  │              │
│  │ (tester.py)  │   │  (judge.py)  │   │(distiller.py)│              │
│  └──────────────┘   └──────────────┘   └──────────────┘              │
│                                                                         │
│                      ┌──────────────┐                                 │
│                      │  MetaLoop    │  ← orchestrator                 │
│                      │(meta_loop.py)│                                 │
│                      └──────────────┘                                 │
└───────────────────────────────────────────────────────────────────────┘
```

Each class has:
- An `__init__` that takes its dependencies (the OpenAI client, sibling agents, the shared tracer) as explicit constructor arguments.
- One or two public methods that do the actual work.
- No globals, no decorators, no implicit registration.

Example from `PromptTuner`:

```python
class PromptTuner:
    def propose_edit(self, *, violations: list[dict], attempt_index: int) -> PromptEdit:
        ...
```

That's the whole API. No `BaseAgent` subclass. No `@agent` decorator. No `AgentExecutor`.

## Why no framework?

The usual reasons you reach for LangChain or similar:

| Framework promise | Our situation |
|---|---|
| "Connect many LLM calls in a chain" | We have 6 agents; the composition is ~100 lines of Python. Wrapping a 100-line orchestration in a 50KB framework is a net loss. |
| "Standard memory abstraction" | We have three purpose-built memories: the hash-chained audit log, the meta-loop prompt version history, and the prompt-tuning scoreboard. None match what a generic memory abstraction provides. |
| "Tool use / function calling" | The agents don't need tools — they consume structured diagnostic state and emit structured decisions. OpenAI Structured Outputs handles the schema enforcement at the API level. |
| "Agent autonomy loops (ReAct, etc.)" | The decision loop is driven by the training loop. The agent decides once per epoch when a rule fires; there is no open-ended agentic planning. |
| "Framework-level observability" | We built our own `AgentTracer` (see `src/env_rl/harness/agent_trace.py`) that records every interaction in a single JSONL. Fewer concepts, exact shape we need. |
| "Multi-agent coordination" | Our coordination is a simple 3-step pipeline: Tuner → Tester → Judge. MetaLoop is a 50-line orchestrator. No message bus needed. |

More concretely, these are the costs we avoided:

### 1. Reproducibility cost of frameworks

Frameworks ship with pinned dependency graphs that change fast. LangChain has had many breaking API revisions. For a project whose main value is *auditability*, depending on framework-level plumbing whose behavior shifts between minor versions is the opposite of what we want.

### 2. Explainability cost

When a decision goes wrong in env_rl, there is exactly one code path to read: `OpenAIDecisionPolicy.decide()`. ~80 lines. Any failure can be debugged from the transcript + the call site. A framework-hosted agent would put several layers of abstraction (agent runtime, tool router, memory store, callback manager) between the decision and the log.

### 3. Schema enforcement

OpenAI's built-in structured outputs (with `response_format={"type": "json_schema", "strict": True}`) guarantee the response shape at the API level. We do not need a framework's `PydanticOutputParser` or `StructuredChatOutputParser` to do the same thing less reliably.

### 4. Testing

Every agent in env_rl is unit-testable with a `StubOpenAIClient` that implements the same surface as `openai.OpenAI`:

```python
client = StubOpenAIClient(responses=[{
    "event_type": "hyperparameter_change",
    "cites": ["R7"],
    ...
}])
policy = OpenAIDecisionPolicy(client=client, model="gpt-4o-mini")
decision = policy.decide(...)
assert decision.cites == ["R7"]
```

With a framework, this test would require mocking the framework's agent runtime, not just the LLM. A simpler stack means simpler tests.

## What we would use a framework for

Hypothetically, the cases where bringing in a framework would *actually* help:

1. **If we needed many orthogonal tools** (web search + code execution + SQL + a vector DB + …) and a planner to route between them. LangGraph's explicit state machine would become worth the weight.
2. **If we wanted open-weights RL** on top of the current environment (the "real RL" extension). Then `trl`, `TRL`, or `unsloth` become directly applicable — but those aren't generic agentic frameworks, they're specialised fine-tuning tooling.
3. **If we wanted to distribute agents across processes**. We don't — the orchestration fits in-process and runs in seconds.

None of those apply today.

## Observability we do have

Instead of framework-level tracing, we built a purpose-specific tracer.

`src/env_rl/harness/agent_trace.py` exposes:

```python
class AgentTracer:
    def record(self, *, agent, action, duration_ms, ...) -> None: ...
    @contextmanager
    def timed(self, *, agent, action, ...) -> Iterator[dict]: ...
```

Every agent takes an optional `tracer=` constructor argument. Events land in `attempt_NN/agent_trace.jsonl` as one JSON object per line. Fields:

```json
{
  "ts": 1776811167.04,
  "attempt_index": 1,
  "epoch": 2,
  "agent": "decision_llm",
  "action": "api_call",
  "duration_ms": 843.2,
  "input_summary": {"top_rule": "R7", "fired_count": 3},
  "output_summary": {"event_type": "hyperparameter_change",
                     "cites": ["R7"], "remedy_direction": "decrease_lr"},
  "token_cost": {"prompt_tokens": 2450, "completion_tokens": 119, "total_tokens": 2569},
  "model": "gpt-4o-mini"
}
```

Every agent records:

| Agent | Actions recorded |
|---|---|
| `decision_llm` | `api_call` · `cache_hit` · `ensemble_call` |
| `tuner` | `propose_edit` |
| `tester` | `run_suite` |
| `judge` | `compare_prompts` |
| `scoreboard` | `record` |
| `distiller` | `distill` (when used) |
| `adversary` | `generate_candidates` (when used) |

The tracer is opt-in via dependency injection — `tracer=None` makes every `record()` a no-op. Zero cost for anyone who doesn't want the trace.

## Comparison with framework-based designs

For the record, here's what the equivalent LangChain / LangGraph design would look like, and where each approach lands:

| Concern | Framework approach | Env-RL approach |
|---|---|---|
| Tool routing | `RouterChain` or LangGraph edges | Explicit `if/else` in the training loop |
| State management | `AgentState` TypedDict in LangGraph | Plain dataclasses (`Decision`, `PromptEdit`, `Scores`) |
| Memory | `ConversationBufferMemory` etc. | Hash-chained JSONL (canonical) + per-attempt summary JSON (readable) |
| Tracing | LangSmith / LangFuse | `AgentTracer` → `agent_trace.jsonl` |
| Schema | Pydantic output parser | OpenAI `strict: true` structured output |
| Testing | Mock the framework | Mock the OpenAI client |
| Lines of orchestration code | ~50–100 lines of chain definitions + ~200 LoC imports | ~150 LoC of hand-rolled Python total |
| Dependency weight | LangChain + langchain-openai + pydantic + tiktoken | openai only |
| Onboarding time for a reader | Read the framework first, then the code | Read the 8 classes |

Neither is objectively better — they are appropriate for different constraints. Env-RL's constraint is **auditability above all else**. Every design choice — including the framework-less stack — flows from that.

## If you want to change it

All agents communicate via plain Python dataclasses. Wrapping them in a framework later is a mechanical change, not an architectural one. If you want to expose the MetaLoop as a LangGraph graph for observability-in-LangSmith purposes:

1. Add a LangGraph node per agent
2. Map the existing `Decision`/`PromptEdit`/`PromptJudgment` dataclasses to `AgentState` TypedDicts
3. Keep the underlying classes unchanged

None of the core logic changes. The framework layer would sit on top. That path is open.
