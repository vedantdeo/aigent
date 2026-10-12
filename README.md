# aigent

An AI agent that reads Indian companies' annual reports and answers with citations — built
incrementally over eight weeks, with every capability measured by an eval and every model call
priced before it is sent.

It searches 1,100+ pages of filings (ITC, Reliance, Tata Motors), calls tools for arithmetic, files
and the web, cites the passage behind each figure, resists prompt injection planted in the documents
it reads, and is graded not only on its answers but on how it reached them. Its from-scratch
counterpart, [underhood](https://github.com/vedantdeo/underhood), holds the shared tools and the
fine-tuning of a small open model for one of the agent's jobs.

## Highlights

| | Result | Where |
|---|---|---|
| **Agent, graded on its trajectory** | 25 multi-step tasks, four graders each (right tools, sane order, finished, correct by a judge model). LangGraph + Sonnet 5: **23/25 at $0.038 a task**; the same agent on Google ADK and the bare SDK tool runner scored 20 and 16 | [v0.2-agent](docs/v0.2-agent.md) |
| **One door to every model** | every call counted for free, checked against per-request, per-turn, per-run and per-eval ceilings, then sent, billed and traced; a test fails if any other module touches the SDK | [`llm/core.py`](src/aigent/llm/core.py) |
| **What retrieval buys** | same 54 questions, same model: **9% correct closed-book → 78% with RAG**, and right on **42 of the 43** it chose to answer | [report](evals/reports/retrieval-20260919-0821.md) |
| **An eval can pick the wrong retriever** | reword the questions and BM25 goes from best (0.889 hit@5) to worst (0.315); fusion inverts too | [below](#retrieval) |
| **Prompt-injection guardrails** | 8 attacks planted in files the agent reads (hijack, exfiltration, forged citations, false figures): **8/8 resisted**; untrusted text is fenced, credentials redacted, citations checked against what was retrieved | [`guardrails.py`](src/aigent/guardrails.py) |
| **A fine-tune for one tool** | Qwen3-4B taught to extract figures and call a calculator instead of doing arithmetic: **3/43 → 28/43** on a hand-labelled test set, tool use right on 17 of 22 that needed it and 0 false calls; the labels have since been tightened and the next round is training | [underhood](https://github.com/vedantdeo/underhood#tool-use-fine-tune) |

Everything is reproducible from the repo, 1,348 tests run free in CI, and [`LOG.md`](LOG.md)
records what each run cost and measured.

## Architecture

```
               ┌──────────── aigent ─────────────┐
 task ──▶ guardrails ──▶ agent (LangGraph | ADK | SDK runner)
                          │        │           │
                          ▼        ▼           ▼
                     search_reports  calculate/files  web search
                     (hybrid RAG)    (underhood.tools) (server tool)
                          │
              every call ─┴─▶ llm/core.py: count → admit against budgets → send → bill → trace (Langfuse)
                          │
                          ▼
                 answer + citations ──▶ guardrails ──▶ evals (graders, judge, reports)
```

- **`llm/`** is the only code that talks to a model. `core.py` counts tokens with the free endpoint,
  prices the worst case (input plus the full `max_tokens`), admits it against five USD ceilings, and
  only then sends. `adapters/` speak each client's wire; `pricing.py` is the only place a price lives.
- **Budget guards**, each overridable in `.env`; a trip raises `BudgetExceeded` with the numbers:

  | Guard | Default | Enforced |
  |---|---|---|
  | per request | $0.25 | before every call, on its worst case |
  | per agent turn | $0.60 | before every turn, since a turn resends the conversation |
  | per workflow | $0.25 | before every call and every parallel batch |
  | per run | $1.00 | before every call in a tool loop, and after it on the real cost |
  | per eval | $2.00 | before the run on its whole worst case; a mid-run trip keeps the rows already paid for |

- **Tracing**: every eval row, model call, tool call and grade nests into one Langfuse trace, with
  credentials masked before anything leaves the machine.

## Agent

[`agent_graph.py`](src/aigent/agent_graph.py) gives the model the reports as a `search_reports` tool
beside a calculator, the clock, a sandboxed file reader and web search, and lets it decide what to
call and when it has enough. The budget, not a turn count, decides how long it searches; when it runs
out it is told to answer from what it has. The conversation is cached turn to turn (55–58% of input
read from cache). The same agent is also built on Google's ADK ([`agent_adk.py`](src/aigent/agent_adk.py))
and on the Anthropic SDK's tool runner ([`agent.py`](src/aigent/agent.py)), to compare frameworks
on one eval: [docs/agent-frameworks.md](docs/agent-frameworks.md).

**Graded on how it got there.** [`agent_tasks.py`](src/aigent/agent_tasks.py) runs 25 hand-written
tasks — search then calculate, compare two reports, convert at today's exchange rate, combine a
sandbox file with a report, and three that must be declined — and grades each run four ways. Three
graders are free and read the trajectory; the fourth is a judge that is never the agent's own model.

| Build | Agent | Passed | Cost a task | Input from cache |
|---|---|---|---|---|
| **LangGraph** | **Sonnet 5** | **23/25** | **$0.038** | 58% |
| Google ADK | Sonnet 5 | 20/25 | $0.037 | 56% |
| SDK tool runner | Sonnet 5 | 16/25 | $0.039 | 54% |
| LangGraph | Opus 5 | 19/25 | $0.116 | 55% |

One build scored 15 then 23 on consecutive runs, so read single runs as a range. What failed and
why: [docs/v0.2-agent.md](docs/v0.2-agent.md).

```bash
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5          # prints the worst case
uv run --group graph python -m aigent.agent_tasks --model claude-sonnet-5 --yes    # spends
```

**Guardrails** ([`guardrails.py`](src/aigent/guardrails.py)): a task is trimmed and refused if it
carries control characters or a credential; anything a third party wrote reaches the model inside an
`<untrusted>` fence it cannot close, with a system note that fenced text is data; an answer must cite
only passages actually retrieved in that run, and is withheld if it contains anything shaped like a
credential. Eight injection cases — planted instructions, exfiltration attempts, forged citation ids,
false figures in a memo — all resisted.

**MCP server** ([`mcp_server.py`](src/aigent/mcp_server.py)): the report search and the sandbox's
files served to any MCP client over stdio.

```bash
claude mcp add aigent-reports -- uv --directory /path/to/aigent run python -m aigent.mcp_server
```

## Evals

The harness ([`src/aigent/evals/`](src/aigent/evals/)) runs any task over a labelled JSONL dataset,
grades each row and writes a Markdown report. A task is just `Callable[[Case], Outcome]`, so a
free retrieval eval and a paid agent eval go through the same table.

```python
from aigent.evals import digest, field_match, load_jsonl, run_eval, write_report

cases = load_jsonl(path)
run = run_eval(
    cases,
    {"baseline": my_task},
    {"fields": field_match(["company", "quarter"])},
    dataset=path.name,
    digest=digest(path),
)
print(write_report(run))
```

Fourteen free graders — exact, contains, includes and excludes, regex, schema validity, per-field
match, a self-reported flag, right tools and tool order read off a trajectory, and four ranking
graders (`resolvable`, `hit_at_k`, `recall_at_k`, `reciprocal_rank`) — plus `LlmJudge`, the only one
that spends, billed to the same ceiling and run on a different model from the one it grades.
A row that raises becomes an error row, not a lost run.

## Retrieval

Chunking four ways, local embeddings (`bge-small-en-v1.5`), a brute-force dense index in NumPy,
BM25, reciprocal rank fusion and a cross-encoder reranker, all behind one `rank(query, k)`
([`src/aigent/retrieval/`](src/aigent/retrieval/)). Labels are **quotes, not chunk ids**, so one
labelled set scores every chunking strategy.

**The finding worth reading first.** Six retrieval arms over two question sets that differ only in
wording — same corpus, chunks, quotes and labels (hit@5, $0.00):

| question style | dense | dense+rr | bm25 | bm25+rr | hybrid | hybrid+rr |
|---|---|---|---|---|---|---|
| written *from* the passages | 0.685 | 0.778 | 0.889 | **0.907** | 0.833 | 0.870 |
| paraphrased | 0.389 | **0.500** | 0.315 | 0.426 | 0.426 | 0.481 |

The two rows recommend opposite systems. **An eval whose questions were written while looking at the
passages will recommend the wrong retriever** — BM25 wins on vocabulary the questions copied. Also:
naive fusion lost to BM25 alone (`RRF_K=60` rewards rankers agreeing, which is worthless when one is
noise); reranking moves MRR, not recall; and widening the rerank pool helped one arm and hurt another.

**What retrieval buys** — the same 54 questions answered closed-book and with retrieved passages
(`claude-opus-5`, judged by `claude-sonnet-5`, $1.34):

| | correct | abstained | right when it answered |
|---|---:|---:|---:|
| closed-book | 5/54 (9%) | 44 | 5/10 — a coin flip |
| RAG | **42/54 (78%)** | 11 | **42/43** |

Without documents the model knows when it doesn't know — and is wrong half the time when it speaks.
With them, the failure mode is silence, not fabrication. The same pipeline rebuilt in LangChain scored
within one row of 54 on every metric: [docs/framework-comparison.md](docs/framework-comparison.md).

## Workflows

The five patterns from Anthropic's *Building effective agents*, each under 100 lines over the same
reports ([`src/aigent/workflows/`](src/aigent/workflows/)): prompt chaining with a code gate that
drops any quote not in its passage; routing that sends lookups to Haiku and analysis to Opus;
parallelization by sectioning and by voting; orchestrator-workers (also rebuilt in LangGraph); and
evaluator-optimizer.

## The figure fine-tune

`extract_figures` turns a passage into `{metric, value, unit, year}` records — the structured input
the capstone analyst reasons over. aigent owns the task (schema, prompt, grader, hand-labelled test
set) and the teacher labels, made by Claude through the Batch API with a warmed prompt cache;
underhood owns the training. The interesting part was the eval loop: a fault census over 484 cases,
adjudicated by hand, showed **half the model's apparent errors were label errors** — scales the
passage never states, currencies read off exchange-rate columns — so the fixes went into the label
rules and checks rather than the model. Details: [underhood](https://github.com/vedantdeo/underhood#tool-use-fine-tune),
[`extraction/figures.py`](src/aigent/extraction/figures.py), and the census in
[`figure-adjudications.jsonl`](evals/datasets/figure-adjudications.jsonl).

## Setup

```bash
cp .env.example .env        # then paste your ANTHROPIC_API_KEY
uv sync
git config core.hooksPath .githooks   # enables the pre-commit hook
uv run pytest -q                      # free: paid tests are opt-in
```

Every paid entry point prints its worst-case cost and stops; `--yes` is what spends.

## Layout

- `src/aigent/llm/` — the one door to a model: counting, budgets, pricing, client adapters
- `src/aigent/agent_graph.py`, `agent_adk.py`, `agent.py` — the agent on three frameworks; `agent_tasks.py` its eval
- `src/aigent/guardrails.py`, `tools.py`, `report_tools.py`, `mcp_server.py` — guards, tools, the MCP server
- `src/aigent/retrieval/` — chunking, embeddings, dense and sparse indexes, fusion, reranking
- `src/aigent/workflows/` — the five workflow patterns
- `src/aigent/extraction/` — structured extraction: headlines, and the figure task and its labels
- `src/aigent/evals/` — the eval harness; `evals/` the datasets and the reports they produced
- `docs/` — write-ups: [the agent milestone](docs/v0.2-agent.md), [frameworks](docs/agent-frameworks.md),
  [context management](docs/context-management.md), [the capstone's design](docs/capstone.md),
  [the code map](docs/knowledge-graph.md), [development and release](docs/development.md)
- `LOG.md` — what shipped each week, what broke, and what it cost
