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
| `retrieval/corpus.py` | annual report PDFs → `Document`s: extract, normalise, strip running headers |
| `retrieval/questions.py` | the question set, and the gate every label has to pass |
| `retrieval/evaluate.py` | the run: three chunking strategies over one question set, free |
| `retrieval/answer.py` | the generation half: answer from retrieved passages, closed-book against RAG |

Written against the tests rather than the other way round, the same arrangement as the `underhood`
repo. The pipeline runs end to end — chunk, embed, rank, grade, report — and it has been measured.

### Measured

`uv run python -m entropic.retrieval.evaluate` — 54 quote-labelled questions over three Indian
annual reports (1,148 pages), `bge-small-en-v1.5`, k=5, **$0.00**
([full report](evals/reports/retrieval-20260918-1345.md)):

| metric | `fixed` | `fixed+overlap` | `by_sentence` |
|---|---|---|---|
| resolvable | 0.963 | **1.000** | **1.000** |
| hit@5 | 0.635 | 0.630 | **0.685** |
| recall@5 | **0.619** | 0.583 | 0.608 |
| MRR | 0.471 | **0.511** | 0.452 |

**Three metrics, three different winners — read that as "not separable", not as a ranking.**
The largest gap is four cases out of 54, and a four-case gap on 54 paired rows cannot reach
p<0.05 by a sign test even if every discordant row falls the same way. What the table does say
is that `fixed` caps itself before the embedder runs: 2 of 54 labels straddle one of its
boundaries, so no retriever could have found them, which is what the `resolvable` row is for.

Retrieval is half the system. `uv run python -m entropic.retrieval.answer --yes --sample 12`
answers the same questions from those passages, on `claude-opus-5`, judged by `claude-sonnet-5` —
**12 of 54 cases, $0.295**
([full report](evals/reports/retrieval-20260919-0754.md)):

| | `closed_book` | `rag` |
|---|---|---|
| answered | 1/12 | 12/12 |
| cites a passage holding the answer | 0/12 | 8/12 |
| correct | 0/12 | **10/12** |

**0% to 83%** is what retrieval bought. Both arms get the identical questions; only one gets the
passages. An absolute RAG score would not say this — it would conflate what the retriever found
with what a large model already knows about three of India's best-covered companies.

The `answered` row is the one to read second. Closed-book **abstained 11 times and invented a figure
once** (`rq-001`: "over Rs 32,000 crore" where the report says Rs 34,000 crores). A correctness
grader scores an honest refusal and a plausible fabrication identically, and only the one of them
can be caught by a reader. That is why the grader exists.

One more thing the smoke run corrected. `cites_relevant` came out at 0.667 against the retriever's
own hit@5 of 0.685 — the model cites what it was handed and nothing else. But **correctness (0.83)
runs ahead of both**, because a label names one passage and a report often states the fact in
several: on `rq-054` retrieval missed chunk `#0532` and returned `#0531` and `#0530`, its
neighbours, and the answer was right anyway. Read hit@5 as a floor on what RAG can answer, not a
ceiling.

Two caveats travel with these numbers. The questions were written *from* the passages, so they
inherit that vocabulary and absolute recall reads high — the ordering is what survives, since all
three face the identical set. And `recall@5` is not the headline it looks like: where a quote
resolves to several chunks, those are usually one sentence seen through several overlapping
windows, and demanding all of them charges a strategy for the overlap that made the quote
resolve at all. That is why `hit@5` is there, and why it ranks the three the other way round.

Anthropic ships no embedding model, so this is the one part of Entropic that runs on someone else's
weights — `bge-small-en-v1.5` through `sentence-transformers`, on the laptop's GPU. It costs nothing
to run, which is the point: a retrieval number can be re-measured as often as the question is worth
asking, and the eval harness already treats a task that spends nothing as a first-class one.

Brute force on purpose. A dot product against 100k rows of 384 floats is a few milliseconds, and it
is the *exact* answer — so when retrieval misses, the miss belongs to the chunking or the embedding
rather than to a recall knob buried in an index. Chroma and LanceDB come when the corpus makes that
false.

Four retrieval graders join the harness, and the order is the argument: `resolvable` (did the
label survive chunking at all — a property of the splitter, not the retriever), `hit_at_k` (did
*anything* relevant come back, which is what decides whether an answer is possible), `recall_at_k`
(what fraction of them did) and `reciprocal_rank` (how far down the first one sat, averaged into
MRR by the report). All four report a **number** as well as a verdict, which is why the report
needs a means table — a retriever that reliably finds two of three relevant chunks scores 0% on
strict pass rate and 0.667 on recall, and only one of those two numbers is useful on its own.

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

Eleven graders: exact match, contains, regex, Pydantic validity, per-field match, `flag` (a boolean
the task reported about itself), the four that read a ranked list of chunk ids (`resolvable`,
`hit_at_k`, `recall_at_k`, `reciprocal_rank`), and `LlmJudge` — the only one that spends, billed to the same ceiling as the task. A task that raises becomes one error row rather than a lost run, and a
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
