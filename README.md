# Entropic

A personal AI agent, built one capability at a time over eight weeks. The plan is in
`~/workspace/MLAI/ML/llm-engineer-roadmap.md`; this repo is the agent itself. Every script prints
what it cost.

## Setup

```bash
cp .env.example .env        # then paste your ANTHROPIC_API_KEY
uv sync
uv run entropic             # talk to the agent
uv run entropic-steps       # list the build steps
```

Or skip the key file: install the `ant` CLI, run `ant auth login`, and the SDK finds the profile itself.

## Week 1: the primitives Entropic is made of

```bash
uv run python -m entropic.week01.first_call         # one call, token count, cost
uv run python -m entropic.week01.streaming          # stream thinking and text
uv run python -m entropic.week01.structured_output  # schema in, validated object out
uv run python -m entropic.week01.tool_loop          # the agent loop, by hand
uv run python -m entropic.week01.chat               # what `uv run entropic` runs
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -q            # API smoke test skips itself when no credentials are set
```

## Layout

- `src/entropic/config.py`  credentials, model choice, cost accounting
- `src/entropic/tools.py`   framework-free tools reused from Week 1 through the capstone
- `src/entropic/week01/`    the five primitives
- `tests/`                  unit tests plus a free API smoke test
- `LOG.md`                  weekly log: what shipped, what broke, numbers
