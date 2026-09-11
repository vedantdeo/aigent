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
- 09-09: `read_file` added as a third tool — sandboxed with `.resolve()` then `is_relative_to`, so
  `..`, an absolute path and a symlink are all refused, each with a test. The parallel-tool exercise
  is done: two tools in one turn, both results back in one user message, asserted in the tests.
- 09-10: `docs/knowledge-graph.md` written and wired to a pre-commit hook plus two CI workflows, so
  the map cannot quietly rot. Project rules moved from private session memory into a tracked
  `CLAUDE.md`.
- 09-10: eval harness v1 (`src/entropic/evals/`): strict JSONL loader, five graders, a runner with a
  third budget ceiling ($2.00 per eval), and a three-table Markdown report. 82 tests, all free —
  tasks in the runner tests are plain functions, so nothing here needed a fake client. Not yet run
  against a real dataset; Project 1a is next.
- 09-11: split `config.py` in two — `config` keeps settings (credentials, model choice, the three
  ceilings), `pricing` takes the `Price`/`Budget` classes and the cost arithmetic. Invariant 8 is now
  two rules instead of one compound one, and `pricing` imports `config` one way. Added
  `ENTROPIC_JUDGE_MODEL` (defaults to sonnet, not opus) so the judge is never the model it grades.
- 09-11: every tunable constant now lives in `config.py` — the six per-call `MAX_TOKENS`, the agent
  turn cap, the report's failure cap. `tools.py` keeps its own, and that exception is written down:
  it imports nothing from the package, which is what makes it liftable into Week 5's SDK tool runner
  and Week 6's LangGraph node. Rule recorded in `CLAUDE.md` and in the global one.
- 09-11: `tools_config.py` added, so the tool pair carries its own constants instead of importing the
  package-wide `config`. `tools` still imports nothing else from the package, which is the property
  that lets Weeks 5 and 6 reuse it — now kept without leaving constants scattered.
- 09-11: the constants rule is now recorded twice on purpose — generic text verbatim in both
  `~/.claude/CLAUDE.md` and this repo's tracked `CLAUDE.md`, with the repo-specific examples appended
  after it. The global file is machine-local and unbacked, so the tracked copy is what survives.
- 09-11: CI actions bumped off the deprecated Node 20 runtime — `actions/checkout` v4 to v7,
  `astral-sh/setup-uv` v5 to v10.1.0. The uv action is pinned to a full version because it stopped
  publishing floating major tags at v8, so `@v10` would 404 the run.
