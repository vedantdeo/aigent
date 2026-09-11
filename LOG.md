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
- 09-11: test suite compacted into input/output tables — 78 test functions became 58 over the same
  ground, plus one shared `conftest` fixture replacing a scripted judge that existed in two modules.
  Rule recorded verbatim in both `CLAUDE.md` files, same mirror arrangement as the constants rule.
- 09-11: **Project 1a shipped and run.** 50 headlines in `evals/datasets/headlines.jsonl`, 30
  labelled; `week01.extraction` extracts `{company, metric, quarter, direction, change_pct}` through
  `messages.parse` behind a `Task`. The model copies the company as the headline writes it and
  `Resolver` maps it to an NSE ticker from `evals/reference/nse-tickers.json` — 11 of the 29 tickers
  are not derivable from any name a headline uses, so that recall left the model's job entirely.
  First live eval: 60 calls, **$0.41943** of a $2.00 ceiling, ~$0.0070 a row.
    - `ticker` **59/59** across both arms. The lookup did what it was supposed to do.
    - zero_shot 28/29 fields with 1 error; few_shot 28/30. The arms agree on 29 of 30 cases — at
      n=30 that is no measurable difference, and few-shot cost 22% more ($0.230 vs $0.189) for it.
    - Two failures, both pointing at the labels rather than the model. `hl-006` ("L&T bags orders
      worth Rs 5,000 crore in Q2") — both arms said `direction=unknown`; winning orders is an
      absolute figure, not a stated move, and they have a case. `hl-018` ("Delhivery narrows Q1
      loss; revenue up 63%") — few_shot read the metric as `profit`, which is the headline's first
      clause; the label picked the clause with the number.
    - One `max_tokens` error in 60 calls, on that same ambiguous row. `MAX_TOKENS_HEADLINE` at 256
      is tight enough to clip a record the model deliberated over.
- 09-11: acted on the first run's findings. `METRICS` is now a total order with the tie-break stated
  in the `metric` description — earliest-ranked metric among those whose percentage change the
  headline states, else earliest overall — because a headline naming two metrics with no stated rule
  produces a label that is a coin flip. The order is a placeholder, asserted rather than researched.
  It needed one clarification to be safe: a forward-looking statement about a metric is `guidance`,
  not that metric, or the mechanical rule reads "cuts revenue guidance" as cueing `revenue` and
  flips `hl-003` and `hl-023` to labels no reader would write. `MAX_TOKENS_HEADLINE` 256 to 384.
  `hl-006` relabelled `direction=unknown`: both arms said so independently, and winning orders is an
  absolute figure rather than a stated move. `hl-018` stays as labelled — the new tie-break is
  exactly what justifies it, since only revenue carries a stated percentage.
- 09-11: re-ran after the fixes. **60 calls, $0.48181, 30/30 on every field in both arms** — the
  `hl-006` relabel and the metric tie-break both hold, and 384 tokens clipped nothing. Digest moved
  to `b78772e05ba0` with the relabel, which is what the digest is for.
  - A perfect score is a warning, not a win: the eval now has no discriminating power. It cannot
    rank two prompts, so it cannot tell whether the next prompt change helped. What it can still do
    is catch a regression, which is worth having but is not what this was built for.
  - few_shot ties zero_shot 30/30 and costs **29% more** ($0.2715 vs $0.2103). On this dataset the
    field descriptions are doing all the work and the examples are dead weight. That is a claim
    about this dataset at this difficulty, not about few-shot prompting.
  - Where the discrimination went: the 20 unlabelled rows are the hard ones (GMV and EBITDA outside
    the metric vocabulary, two companies with equal claim, sequential vs YoY). Labelling them is now
    the highest-value work on this project — an eval that everything passes has stopped measuring.
