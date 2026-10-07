# aigent

A personal AI agent, built one capability at a time over eight weeks. The plan is in
`~/workspace/MLAI/ML/llm-engineer-roadmap.md`; this repo is the agent itself. Every script prints
what it cost.

## Setup

```bash
cp .env.example .env        # then paste your ANTHROPIC_API_KEY
uv sync
git config core.hooksPath .githooks   # one-time, per clone: enables the pre-commit hook
```

`uv sync` also clones [underhood](https://github.com/vedantdeo/underhood), a private repo, for its
torch-free GPU pricing, so git needs read access to it; CI gets that from the `UNDERHOOD_TOKEN`
secret.

Or skip the key file: install the `ant` CLI, run `ant auth login`, and the SDK finds the profile itself.

## One door to a model

Every model call in the repo goes through `src/aigent/llm/core.py`: counted for free, checked against the
per-request ceiling, admitted against the run's budget, then sent, billed and traced. A test fails
on any other module that calls the SDK itself.

## Retrieval

| Module | What it does |
|--------|--------------|
| `retrieval/chunk.py` | `Document` → `Chunk`, four ways: fixed, fixed with overlap, by sentence, by heading |
| `retrieval/embed.py` | chunks → L2-normalised vectors, locally and free |
| `retrieval/dense.py` | the dense index: brute-force cosine over the chunk vectors, in NumPy |
| `retrieval/corpus.py` | annual report PDFs → `Document`s: extract, normalise, strip running headers |
| `retrieval/questions.py` | the question set, and the gate every label has to pass |
| `retrieval/sparse.py` | the sparse index: BM25 over the same chunks, as an inverted index |
| `retrieval/fuse.py` | reciprocal rank fusion, because two rankers' scores are not comparable |
| `retrieval/rerank.py` | a cross-encoder that reads query and passage together, then reorders |
| `retrieval/hits.py` | `Hit`, what every index and ranker returns |
| `retrieval/rank.py` | `build_ranker(method)`: any of the six methods as one `rank(query, k, doc_id=)` call |
| `retrieval/evaluate.py` | the runs: chunking strategies, or six retrieval arms, both free |
| `retrieval/langchain_rag.py` | the same pipeline in LangChain, scored by the same harness |
| `retrieval/answer.py` | the generation half: answer from retrieved passages, closed-book against RAG |

Written against the tests rather than the other way round, the same arrangement as the `underhood`
repo. The pipeline runs end to end — chunk, embed, rank, grade, report — and it has been measured.

### Measured

`uv run python -m aigent.retrieval.evaluate` — 54 quote-labelled questions over three Indian
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

### The result worth reading first

Six retrieval arms, run over **two question sets that differ only in wording** — same corpus, same
chunks, same 54 quotes, same labels. `uv run python -m aigent.retrieval.evaluate --compare
retrieval`, hit@5, **$0.00**:

| question style | `dense` | `dense+rr` | `bm25` | `bm25+rr` | `hybrid` | `hybrid+rr` |
|---|---|---|---|---|---|---|
| written *from* the passages | 0.685 | 0.778 | 0.889 | **0.907** | 0.833 | 0.870 |
| paraphrased ([why](evals/datasets/retrieval-paraphrased.jsonl)) | 0.389 | **0.500** | 0.315 | 0.426 | 0.426 | 0.481 |

**The two tables recommend opposite systems.** On questions authored from the passages, BM25 wins
outright and fusing it with embeddings makes it *worse*. On paraphrased questions, BM25 falls from
best to worst — below dense — and fusion becomes the best base.

Nothing changed but how the questions were worded. So: **an eval whose questions were written while
looking at the passages will recommend the wrong retriever.** Every retrieval number below is
conditional on its question style, and that is the finding, not a caveat on one.

Three smaller things fell out of it:

- **Naive RRF lost to BM25 alone** on the original set. Dense contributed exactly one unique row in
  54, and fusion gave back four that BM25 had at **rank 1** — because `RRF_K=60` makes agreement
  between rankers beat a single confident first place, and agreement is worthless when one ranker is
  noise. The property that makes fusion good is the one that made it bad here.
- **Reranking cannot retrieve.** It moves MRR hardest (dense 0.452 → 0.650 on the original set) and
  moves recall only as far as the candidate pool reaches.
- **The candidate pool is a real optimum, not a free knob.** Widening 30 → 100 helped `bm25+rerank`
  (+0.037 hit@5) and *hurt* `dense+rerank` (−0.019): seventy more candidates are seventy more
  chances for a small cross-encoder to promote a plausible wrong chunk.

The paraphrased set carries two caveats of its own, stated where it lives: its author knew the
hypothesis, and the absolute scores are confounded with the questions simply being vaguer. The
*relative* inversion is the robust part, since both rankers faced identical questions.

Retrieval is half the system. `uv run python -m aigent.retrieval.answer --yes` answers the same
questions from those passages, on `claude-opus-5`, judged by `claude-sonnet-5` — **all 54 cases,
$1.342** ([full report](evals/reports/retrieval-20260919-0821.md)):

| | `closed_book` | `rag` |
|---|---|---|
| answered | 10/54 (19%) | 43/54 (80%) |
| cites a passage holding the answer | 0/54 | 37/54 (69%) |
| correct | 5/54 (9%) | **42/54 (78%)** |

**9% to 78%** is what retrieval bought. Both arms get the identical questions; only one gets the
passages. An absolute RAG score would not say this — it would conflate what the retriever found
with what a large model already knows about three of India's best-covered companies.

### The number that matters is not in the `correct` row

| | abstained | answered | right when it answered |
|---|---:|---:|---:|
| `closed_book` | 44 | 10 | 5/10 — **50%** |
| `rag` | 11 | 43 | 42/43 — **98%** |

Without documents the model is a **coin flip whenever it chooses to speak**. Its judgement about
what it doesn't know is good — it refuses 44 of 54 — and then it gets half the rest wrong, with the
figures that make an annual report worth reading: `rq-001` answered "over Rs 32,000 crore" where ITC
reports Rs 34,000 crores.

With passages, **one wrong answer in 43**, and it declines the other eleven rather than guessing.
The failure mode is silence, not fabrication. A `correct` column alone cannot tell you that — 5/54
and 42/54 do not say that one of these is reckless and the other is not, which is why `answered` is
graded beside it.

### Two evals, written separately, agreeing row for row

`cites_relevant` is 37/54. The free retrieval eval measured `by_sentence` hit@5 at 37/54 — and not
merely the same count: **the same 37 rows, agreeing on all 54.** No row where retrieval surfaced a
relevant chunk and the model failed to cite it; no row where it claimed one it was never given. The
model cites what it was handed and invents no ids, so a divergence here would mean something is
broken — a stale inventory, a mismatched strategy, a chunk-id collision.

It also means the 5 rows where `correct` beats `cites_relevant` are purely the
neighbouring-chunk effect: on `rq-054` retrieval missed chunk `#0532` and returned `#0531` and
`#0530`, and the answer was right anyway, because a label names one passage and a report states the
fact across a section. Read hit@5 as a floor on what RAG can answer, not a ceiling.

`cites_relevant` is an any-hit metric with no precision term, so a model citing all five passages
every time would score the same as one citing the single right passage. It is a floor on citation
quality rather than a measure of it.

Two caveats travel with these numbers. The questions were written *from* the passages, so they
inherit that vocabulary and absolute recall reads high — the ordering is what survives, since all
three face the identical set. And `recall@5` is not the headline it looks like: where a quote
resolves to several chunks, those are usually one sentence seen through several overlapping
windows, and demanding all of them charges a strategy for the overlap that made the quote
resolve at all. That is why `hit@5` is there, and why it ranks the three the other way round.

**The framework comparison.** The whole pipeline rebuilt in LangChain and run through this repo's own eval: **within one row of 54 on every metric**, for 14 statements against 336. The
framework cost no accuracy — what it cost was 31 transitive packages, a loader that warns it is
being sunset, chunks that cannot cross a page boundary, and a `Document` with no identity whose
absence scored 0.000 on everything without raising. Ten lines on what it abstracted and what it
hid: [docs/framework-comparison.md](docs/framework-comparison.md).

Anthropic ships no embedding model, so this is the one part of aigent that runs on someone else's
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

## Workflows

The five workflow patterns from Anthropic's "Building effective agents", each under 100 lines, each
answering questions over the same three annual reports. A workflow is code that fixes the path and
calls a model at set points; an agent lets the model choose the path.

| Pattern | Module | Over the reports |
|---------|--------|------------------|
| prompt chaining | `workflows/chaining.py` | extract facts with quotes → a code gate drops any quote not in its passage → write a note from the survivors |
| routing | `workflows/routing.py` | Haiku classifies the question; a lookup goes to Haiku, an analysis to Opus, anything else is declined for free |
| parallelization | `workflows/parallelization.py` | one call per report at once (sectioning), then three reviewers each able to veto the answer (voting) |
| orchestrator-workers | `workflows/orchestrator_workers.py` | a model writes the plan, workers run each subtask on its own search, a model combines the findings |
| orchestrator-workers, as a graph | `workflows/orchestrator_workers_graph.py` | the same, rebuilt in LangGraph to compare; needs `--group graph` |
| evaluator-optimizer | `workflows/evaluator_optimizer.py` | Opus drafts, Sonnet grades against four criteria, redraft until it passes or three rounds are up |

Each is a function over an `Llm` and the reports' search, tested against the fake client. Every
call goes through `llm/core.py`, which checks its worst case against the budget before sending it, and a
batch of parallel calls as a whole.

## Agent

`agent.py` is the other side of that line: the model gets the reports as a `search_reports` tool
beside the calculator and the web, and decides for itself what to search, how often, and when it has
enough. It loops on the Anthropic SDK's tool runner, driven through `llm.run_tools`, which counts
and admits every turn against the $1.00 run ceiling before it is sent, so the budget decides how
long it searches; a 25-turn cap is only a backstop against a runaway loop. The conversation is
cached, so each turn reads what the last one sent instead of paying for it again. Each turn is held
to a $0.60 ceiling of its own, since it resends the whole conversation. When it runs out of turns or
budget it is told to stop searching and answer from what it has; only if not even a short answer
fits does it stop empty-handed, printing what it searched and why. Web search is the API's own tool,
called directly and capped at three searches a turn, $0.01 each; the pages it cited or returned are
listed after the answer. `docs/tool-runner.md` is what the runner does and hides, read from its
source.

It runs through Project 2's eval below, which estimates first and spends only with `--yes`.

### Project 2: graded on how it got there

`agent_tasks.py` runs the agent over 25 hand-written multi-step tasks and grades each run four
ways: the right tools, in a sane order, a finished answer (all three free, read off the trajectory),
and a correct one by an LLM judge that is never the agent's own model. A task passes when all four
do.

| Build | Agent | Passed | Cost a task | Input read from cache |
|---|---|---|---|---|
| **LangGraph, final** | **Opus 5** | **19/25** | **$0.116** | **55%** |
| LangGraph | Sonnet 5 | 23/25 | $0.038 | 58% |
| Google ADK | Sonnet 5 | 20/25 | $0.037 | 56% |
| SDK tool runner | Sonnet 5 | 16/25 | $0.039 | 54% |

Each row is one run, and one build scored 15 then 23 on consecutive runs, so read them as a range.
The write-up, with what failed and why, is `docs/v0.2-agent.md`; the framework memo is
`docs/agent-frameworks.md`.

```bash
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5    # worst case only
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5 --cache --yes
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5 --agent adk # or sdk
```

## Budget guards

Five ceilings in USD, each overridable in `.env`. A tripped guard raises `BudgetExceeded` with the
numbers in the message. Trim the input, lower `max_tokens`, or raise the ceiling on purpose.

| Guard | Default | Enforced where |
|-------|---------|----------------|
| per request | $0.25 | before every call: free token count, worst case is input plus the full `max_tokens`, with input at the cache-write price when the call caches, plus each web search it may run with an allowance for its results |
| per agent turn | $0.60 | before every turn of the agent, in place of the per-request ceiling, since a turn resends the whole conversation |
| per workflow | $0.25 | before every call and every parallel batch in a workflow demo, on the worst case |
| per run | $1.00 | before every call in a tool loop, the agent or a chat session, on that call's worst case, and after it on the real cost |
| per eval | $2.00 | before the run starts, on the whole run's worst case; then after each row, where a trip keeps the rows already paid for and marks the report partial |

## Evals

The harness is in `src/aigent/evals/`, the data and results in `evals/`. It runs a task over a
labelled JSONL dataset, grades each row, and writes a Markdown table.

```python
from aigent.evals import digest, field_match, load_jsonl, run_eval, write_report

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

The judge runs on `AIGENT_JUDGE_MODEL` (`claude-sonnet-5`), not `AIGENT_MODEL` — a model asked
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

Everything above except the last line runs in CI, in two pipelines over one set of steps
(`.github/workflows/checks.yml`): the **mains CI** (`main.yml`) on every push to `main` and every
pull request, against underhood's latest `main` rather than the pin, and started again whenever
underhood's own `main` goes green; and the **roll CI** (`roll.yml`) on each `snap-*` branch, against
the underhood snap it pins, which is what a roll requires. No API key is configured there, so the smoke test skips itself and
the paid tests stay deselected — CI spends nothing. `uv sync --locked` also fails the run if
`uv.lock` has drifted from `pyproject.toml`.

`docs/knowledge-graph.md` maps this repo, and it moves with the code. A commit that touches
`src/` or `pyproject.toml` without staging the graph is refused by `.githooks/pre-commit` — enable
it with the `core.hooksPath` line above, since git never installs hooks on clone. `--no-verify` is
the deliberate exception. The same rule runs in CI, so a clone that skipped the hook is still
caught before the graph goes stale.

## Snap and roll

Twice a week both repos branch `main` and promote the branch a day later, all times IST:

| | Snap | Roll |
|---|---|---|
| first half | Mon 00:00 (Sun night) | Tue 00:00 (Mon night) |
| second half | Thu 00:00 (Wed night) | Fri 00:00 (Thu night) |

- **Snap** points `main` at underhood's `snap-YYYY-MM-DD` branch, then branches `main` as aigent's
  own `snap-YYYY-MM-DD`, so the pair is tested together. Each repo keeps its last eight snaps.
- **Roll** tags the snap's head `roll-YYYY-MM-DD` and moves aigent's `stable` tag to it, if CI
  passed there and underhood rolled the commit it pins. Otherwise nothing rolls and the run goes
  red. `roll-…` and `stable` are tags on snap branches, not branches of their own.
- So `main` always runs against underhood's latest snap, re-pointed only by a snap, Sun and Wed
  nights.
- **A fix during the window** lands on `main` first, then is cherry-picked onto the snap branch
  (`git checkout snap-YYYY-MM-DD && git cherry-pick <sha> && git push`). After an underhood fix,
  run the `repin` step so `main` and the snap pick it up.
- `.github/workflows/release.yml` does it, with `.github/release/`'s two scripts; run a step by
  hand from the Actions tab.

## Layout

- `src/aigent/config.py`  every tunable constant bar the tool pair's: models, ceilings, caps
- `src/aigent/llm/`      everything that talks to a model: `core.py`, the one door (count, admit,
  send, bill, trace); `messages.py`, the neutral types; `pricing.py`, token prices and the budget
  guards; `adapters/`, the wire each client speaks
- `src/aigent/errors.py` every error the package defines, to reuse before adding one
- `src/aigent/tools.py`   framework-free tools, reused by everything that calls a tool
- `src/aigent/tools_config.py` their constants, so the pair lifts into any framework intact
- `src/aigent/report_tools.py` tools that need the package, like search over the reports
- `src/aigent/retrieval/` chunking, local embeddings, the dense and sparse indexes, the rankers
- `src/aigent/workflows/` the five workflow patterns, over the same reports
- `src/aigent/agent.py`   the first agent: search as a tool, on the SDK's tool runner
- `src/aigent/agent_graph.py` Project 2's agent as a LangGraph graph; `agent_adk.py`, on Google's ADK
- `src/aigent/agent_tasks.py` Project 2's eval: 25 tasks, four graders, a pass rate and a cost a task
- `src/aigent/extraction/` Project 1a: headline extraction, graded by the harness
- `src/aigent/evals/`     the eval harness: dataset, graders, runner, report
- `evals/`                  eval datasets and the reports they produce
- `tests/`                  unit tests plus a free API smoke test
- `LOG.md`                  the log: what shipped, what broke, numbers
- `CLAUDE.md`               the nine project rules Claude sessions follow here
- `docs/knowledge-graph.md` map of every module, edge, and invariant in the repo
- `docs/tool-runner.md`    the SDK tool runner, read from source against our own loop
- `docs/context-management.md` what goes in the agent's context, what it costs, when to summarize
- `docs/langgraph.md`      the LangGraph executor read from source, and orchestrator-workers rebuilt in it
- `docs/agent-frameworks.md` one agent in three frameworks, and which to use when
- `docs/v0.2-agent.md`     milestone `v0.2-agent`: Project 2's numbers, failures and frameworks
- `.githooks/pre-commit`    refuses a commit that leaves the graph behind
- `scripts/`                the rule that hook and CI share
- `.github/workflows/`      CI: the checks above, and the graph rule for anyone who skipped the hook
