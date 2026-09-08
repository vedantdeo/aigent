# Entropic

A personal AI agent, built one capability at a time over eight weeks. The plan is in
`~/workspace/MLAI/ML/llm-engineer-roadmap.md`; this repo is the agent itself. Every script prints
what it cost.

## Setup

```bash
cp .env.example .env        # then paste your ANTHROPIC_API_KEY
uv sync
uv run entropic             # menu of the five modes (needs a real terminal)
uv run entropic chat        # or: call, stream, extract, loop "task"
```

Or skip the key file: install the `ant` CLI, run `ant auth login`, and the SDK finds the profile itself.

## Week 1: the primitives Entropic is made of

| Mode | Module | What it shows |
|------|--------|---------------|
| `call` | `week01/first_call.py` | one call, token count, cost |
| `stream` | `week01/streaming.py` | thinking and text arriving separately |
| `extract` | `week01/structured_output.py` | schema in, validated object out |
| `loop` | `week01/tool_loop.py` | the agent loop, by hand; takes a task |
| `chat` | `week01/chat.py` | multi-turn conversation with a cost meter |

Each module also runs on its own: `uv run python -m entropic.week01.first_call`.

## Budget guards

Two ceilings in USD, each overridable in `.env`. A per-eval ceiling arrives with the eval harness in
Week 2. A tripped guard raises `BudgetExceeded` with the
numbers in the message. Trim the input, lower `max_tokens`, or raise the ceiling on purpose.

| Guard | Default | Enforced where |
|-------|---------|----------------|
| per request | $0.25 | before every call: free token count, worst case is input plus the full `max_tokens` |
| per run | $1.00 | after every call in a tool loop or a chat session |

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -q            # unit tests plus a free API smoke test; skips paid tests
uv run pytest -m live       # the paid integration tests (about two cents)
```

## Layout

- `src/entropic/cli.py`     the `entropic` command and its menu
- `src/entropic/config.py`  credentials, model choice, cost accounting
- `src/entropic/tools.py`   framework-free tools reused from Week 1 through the capstone
- `src/entropic/week01/`    the five primitives
- `tests/`                  unit tests plus a free API smoke test
- `LOG.md`                  weekly log: what shipped, what broke, numbers
