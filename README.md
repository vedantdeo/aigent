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

## The primitives Entropic is made of

| Mode | Module | What it shows |
|------|--------|---------------|
| `call` | `primitives/first_call.py` | one call, token count, cost |
| `stream` | `primitives/streaming.py` | thinking and text arriving separately |
| `extract` | `primitives/structured_output.py` | schema in, validated object out |
| `loop` | `primitives/tool_loop.py` | the agent loop, by hand; takes a task |
| `chat` | `primitives/chat.py` | multi-turn conversation with a cost meter |

`primitives/failures.py` is not a mode — it provokes nine failure modes and prints the table in
`docs/failure-modes.md`. It costs nothing to run: a request rejected with a 4xx is never billed.

```
uv run python -m entropic.primitives.failures > docs/failure-modes.md
```

Each module also runs on its own: `uv run python -m entropic.primitives.first_call`.

## Retrieval

| Module | What it does |
|--------|--------------|
| `retrieval/chunk.py` | `Document` → `Chunk`, four ways: fixed, fixed with overlap, by sentence, by heading |
| `retrieval/embed.py` | chunks → L2-normalised vectors, locally and free |
| `retrieval/store.py` | vectors → a ranked list of chunk ids, brute-force cosine over NumPy |

Written against the tests rather than the other way round, the same arrangement as the `underhood`
repo. The pipeline runs end to end — chunk, embed, rank, grade, report — but nothing has been
*measured* yet: the corpus and the question set are still to come.

Anthropic ships no embedding model, so this is the one part of Entropic that runs on someone else's
weights — `bge-small-en-v1.5` through `sentence-transformers`, on the laptop's GPU. It costs nothing
to run, which is the point: a retrieval number can be re-measured as often as the question is worth
asking, and the eval harness already treats a task that spends nothing as a first-class one.

Brute force on purpose. A dot product against 100k rows of 384 floats is a few milliseconds, and it
is the *exact* answer — so when retrieval misses, the miss belongs to the chunking or the embedding
rather than to a recall knob buried in an index. Chroma and LanceDB come when the corpus makes that
false.

Two retrieval graders join the harness: `recall_at_k` (what fraction of the relevant chunks came
back in the top k) and `reciprocal_rank` (one over the rank of the first one, averaged into MRR by
the report). Both report a **number** as well as a verdict, which is why the report needs a means
table — a retriever that reliably finds two of three relevant chunks scores 0% on strict pass rate
and 0.667 on recall, and only one of those two numbers is useful on its own.

Retrieval labels are **quotes, not chunk ids**. Chunk ids are positional, so they do not survive a
re-chunk; a labelled sentence does, and `Inventory.containing` resolves it to whichever ids hold it
in the inventory being scored. That is what lets four chunking strategies be four columns of one
table instead of four runs nobody can compare.

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

Seven graders: exact match, contains, regex, Pydantic validity, `recall_at_k` and
`reciprocal_rank` for a retriever (unwritten), and `LlmJudge` — the only one that spends, billed to
the same ceiling as the task. A task that raises becomes one error row rather than a lost run, and a
run that hits the ceiling keeps what it already paid for.

The judge runs on `ENTROPIC_JUDGE_MODEL` (`claude-sonnet-5`), not `ENTROPIC_MODEL` — a model asked
to grade its own output favours it, and applying a rubric is an easier job than the one being
graded. Raise it when the rubric is hard; a judge weaker than the task cannot see the failures that
matter. The report header records which model graded, read back from the run rather than from
config.

The task owns its own API call, so the harness never assumes there is one: a retrieval eval that
returns `Outcome(usage=None)` costs nothing and reports through the same table.

## Editor

`.vscode/` is committed, so the settings arrive with the clone; VS Code offers the extensions on
first open (Ruff, Python, Pylance) and everything below is already wired up.

| On save | Does what |
|---------|-----------|
| `ruff format` | whitespace and layout, to the 100-char limit in `pyproject.toml` |
| `source.fixAll.ruff` | every safe autofix, including removing unused imports |
| `source.organizeImports.ruff` | sorts imports — the formatter does not do this |

`ruff.importStrategy` is `fromEnvironment`, so the editor runs `.venv/bin/ruff` rather than the copy
inside the extension. That is what keeps it reading `pyproject.toml`, and keeps the editor from
disagreeing with `uv run ruff check`. Run `uv sync` before opening the folder or there is no ruff to
find.

The test runner is wired to pytest over `tests/`, and inherits `addopts = "-m 'not live'"` from
`pyproject.toml` — so running the whole suite from the editor cannot spend anything. Pylance
type-checks; its warnings are signal, not noise.

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
- `src/entropic/tools.py`   framework-free tools, reused by everything that calls a tool
- `src/entropic/tools_config.py` their constants, so the pair lifts into any framework intact
- `src/entropic/primitives/` the five modes, plus the failure catalogue
- `src/entropic/retrieval/` chunking, local embeddings, the vector store
- `src/entropic/extraction/` Project 1a: headline extraction, graded by the harness
- `src/entropic/evals/`     the eval harness: dataset, graders, runner, report
- `evals/`                  eval datasets and the reports they produce
- `tests/`                  unit tests plus a free API smoke test
- `LOG.md`                  the log: what shipped, what broke, numbers
- `CLAUDE.md`               the eight project rules Claude sessions follow here
- `docs/knowledge-graph.md` map of every module, edge, and invariant in the repo
- `.githooks/pre-commit`    refuses a commit that leaves the graph behind
- `scripts/`                the rule that hook and CI share
- `.github/workflows/`      CI: the checks above, and the graph rule for anyone who skipped the hook
