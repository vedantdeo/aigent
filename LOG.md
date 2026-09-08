# Log

## Week 0 (2026-09-07)
- Repo scaffolded: uv, Python 3.12, anthropic SDK 1.4.0, pydantic, pytest, ruff, pyright.
- Five Week 1 scripts written, unit tests green. API not yet exercised: no key configured.
- Next: set ANTHROPIC_API_KEY, run all five scripts, do the two Week 1 exercises, start minbpe.

## Week 1 (2026-09-07 to 2026-09-13)
- 09-08: named the agent Entropic. Repo, package, and CLI renamed; `uv run entropic` now opens the chat.
- 09-08: budget guards added ($0.25 per request, $1.00 per run). First live calls: first_call 62 in /
  205 out, $0.0054; tool_loop 3 turns with a parallel tool call in turn 1, $0.023; guard verified to
  refuse at a $0.0001 ceiling before spending.
