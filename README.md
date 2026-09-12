# Entropic

A personal AI agent, built one capability at a time over eight weeks. The plan is in
`~/workspace/MLAI/ML/llm-engineer-roadmap.md`; this repo is the agent itself. Every script prints
what it cost.

## Setup

```bash
cp .env.example .env        # then paste your ANTHROPIC_API_KEY
uv sync
git config core.hooksPath .githooks   # one-time, per clone: enables the pre-commit hook
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

`week01/failures.py` is not a mode — it provokes nine failure modes and prints the table in
`docs/failure-modes.md`. It costs nothing to run: a request rejected with a 4xx is never billed.

```
uv run python -m entropic.week01.failures > docs/failure-modes.md
```

Each module also runs on its own: `uv run python -m entropic.week01.first_call`.

## Budget guards

Three ceilings in USD, each overridable in `.env`. A tripped guard raises `BudgetExceeded` with the
numbers in the message. Trim the input, lower `max_tokens`, or raise the ceiling on purpose.

| Guard | Default | Enforced where |
|-------|---------|----------------|
| per request | $0.25 | before every call: free token count, worst case is input plus the full `max_tokens` |
| per run | $1.00 | after every call in a tool loop or a chat session |
| per eval | $2.00 | across a whole eval run; a trip keeps the rows already paid for and marks the report partial |

## Evals

The harness is in `src/entropic/evals/`, the data and results in `evals/`. It runs a task over a
labelled JSONL dataset, grades each row, and writes a Markdown table.

```python
from entropic.evals import digest, field_match, load_jsonl, run_eval, write_report

cases = load_jsonl(path)
run = run_eval(
    cases,
    {"baseline": my_task, "few_shot": my_other_task},  # a Task is just Callable[[Case], Outcome]
    {"fields": field_match(["company", "quarter"])},
    dataset=path.name,
    digest=digest(path),
)
print(write_report(run))
```

Five graders ship: exact match, contains, regex, Pydantic validity, and `LlmJudge` — the only one
that spends, and it is billed to the same ceiling as the task. A task that raises becomes one error
row rather than a lost run, and a run that hits the ceiling keeps what it already paid for.

The judge runs on `ENTROPIC_JUDGE_MODEL` (`claude-sonnet-5`), not `ENTROPIC_MODEL` — a model asked
to grade its own output favours it, and applying a rubric is an easier job than the one being
graded. Raise it when the rubric is hard; a judge weaker than the task cannot see the failures that
matter. The report header records which model graded, read back from the run rather than from
config.

The task owns its own API call, so the harness never assumes there is one: a retrieval eval that
returns `Outcome(usage=None)` costs nothing and reports through the same table.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -q            # unit tests plus a free API smoke test; skips paid tests
uv run pytest -m live       # the paid integration tests (about two cents)
```

Everything above except the last line runs in CI on every push and pull request
(`.github/workflows/checks.yml`). No API key is configured there, so the smoke test skips itself and
the paid tests stay deselected — CI spends nothing. `uv sync --locked` also fails the run if
`uv.lock` has drifted from `pyproject.toml`.

`docs/knowledge-graph.md` maps this repo, and it moves with the code. A commit that touches
`src/` or `pyproject.toml` without staging the graph is refused by `.githooks/pre-commit` — enable
it with the `core.hooksPath` line above, since git never installs hooks on clone. `--no-verify` is
the deliberate exception. The same rule runs in CI, so a clone that skipped the hook is still
caught before the graph goes stale.

## Layout

- `src/entropic/cli.py`     the `entropic` command and its menu
- `src/entropic/config.py`  every tunable constant bar the tool pair's: models, ceilings, caps
- `src/entropic/pricing.py` token prices, cost arithmetic, the two budget guards
- `src/entropic/tools.py`   framework-free tools reused from Week 1 through the capstone
- `src/entropic/tools_config.py` their constants, so the pair lifts into any framework intact
- `src/entropic/week01/`    the five primitives
- `src/entropic/evals/`     the eval harness: dataset, graders, runner, report
- `evals/`                  eval datasets and the reports they produce
- `tests/`                  unit tests plus a free API smoke test
- `LOG.md`                  weekly log: what shipped, what broke, numbers
- `CLAUDE.md`               the three project rules Claude sessions follow here
- `docs/knowledge-graph.md` map of every module, edge, and invariant in the repo
- `.githooks/pre-commit`    refuses a commit that leaves the graph behind
- `scripts/`                the rule that hook and CI share
- `.github/workflows/`      CI: the checks above, and the graph rule for anyone who skipped the hook
