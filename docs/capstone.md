# The finance agent: a generalist with routed specialists

*Milestone `v1.0-capstone`, Fri 2026-10-30. First written 2026-10-08 as a four-agent filings analyst;
rewritten 2026-10-12 around a generalist with specialists, at Vedant's call. Week 6 builds it, Week 7
evaluates and serves it, Week 8 ships it.*

## The idea

One capable, expensive model with the right tools can answer almost any finance question: search
the filings, look up fundamentals, calculate, read and write files, search the web. But many of the
steps inside an answer are narrow and repetitive — pull the figures out of these passages, classify
this section — and a small fine-tuned model can do those at a fraction of the cost, locally.

So the system is a **generalist that orchestrates, specialists that do narrow jobs, and a router
behind each specialist that decides whether the cheap path is good enough** — falling back to the
generalist when it is not. The headline measurement is the trade the router makes: cost against
quality, against a generalist doing everything.

An analyst asks *"Did ITC's dividend payout ratio rise in FY25, and how does it compare with
Reliance's?"* and gets a short memo in which every claim cites the page it came from, every computed
figure shows its inputs, and a critic has checked each claim against its source before a human
approves it.

## Architecture

```mermaid
flowchart TD
    Q([question]) --> G[generalist: plan, call tools, draft]
    G -->|tool calls| T[[MCP tools: search_filings · fundamentals · calculate · files · web]]
    G -->|specialist calls| R{router}
    R -->|cheap path| S[specialist: local fine-tune]
    S -->|validated| G
    S -->|fails validation| F[generalist does the job itself]
    F --> G
    G -->|claims + draft memo| C[critic]
    C -->|unsupported claim, revisions left| G
    C -->|all supported, or revisions used| H{{human approval}}
    H -->|approve| M([memo])
    H -->|reject, with a note| G
```

| Part | Does | Model |
|---|---|---|
| **generalist** | Plans, calls tools and specialists, writes claims and the memo, each claim citing its sources | Sonnet 5 while building, Opus 5 for reported numbers |
| **specialists** | Narrow jobs, each behind a tool: the **headline reader** (`read_headline`: a market headline → `{ticker, metric, quarter, direction, change_pct}`) and the **report analyst** (`extract_figures`: filing passages → `{metric, value, unit, year}` records) | LoRA fine-tunes of Qwen3-4B, both served from one GPU endpoint |
| **router** | Per specialist call: try the specialist, validate its output, fall back to the generalist on failure; log both | Code, not a model (a learned router is a Week 7 experiment) |
| **critic** | Checks each claim against exactly the sources it cites: supported, unsupported, or wrong (figure, year, company). Never sees the generalist's reasoning | A different model from the generalist |
| **human gate** | Approves the memo, or rejects it with a note that becomes a constraint | — |

## Tools, over MCP

The generalist gets its tools from **MCP servers through one client adapter** — which is the Week 6
MCP client: a tool it did not write joins the loop like its own, counted, fenced as untrusted when it
returns third-party text, and traced.

Built 2026-10-13 as `mcp_client.connect`: a server's listing becomes the schemas the model is shown,
and its calls go through the same dispatcher type as our own tools. Our own server fences what it
returns, so it is connected unfenced; any other server's text, errors included, is fenced.

| Tool | Server | Status |
|---|---|---|
| `search_filings(query, company, year)` | aigent's MCP server | built (step 1) and served (step 3, 2026-10-13): six filings, scoped by company and financial year |
| `fundamentals(ticker, metric, year)` | aigent's MCP server | built (step 1) and served (step 3): 24 figures, each cited to its page |
| `calculate(expression)` | aigent's MCP server | built (`underhood.tools`) and served (step 3) |
| `read_file`, `write_file` | aigent's MCP server | built, sandboxed |
| `web_search` | Anthropic's server tool | built; **off in evals**, since a web figure has no frozen ground truth |

## Specialists, and how the generalist calls them

**A specialist is prompted by its adapter, never by the generalist.** A fine-tune works only on the
prompt it was trained on — a paraphrase is out of distribution. So the generalist calls a specialist
as a tool with **structured arguments**, and the adapter renders the exact training prompt.

What the generalist sees is a **specialist card** — the tool's description and schema:

| Field | Headline reader: `read_headline` | Report analyst: `extract_figures` |
|---|---|---|
| good at | an Indian market headline's company, metric, quarter, direction and percentage change | the figures a filing passage states that answer a question: metric, value with any named scale applied, unit, financial year |
| arguments | `headline` (one line of text) | `chunk_ids` (passages from `search_filings`), `wanted` (what to extract) |
| returns | one record; the company resolved to an NSE ticker in code, by `Resolver` | `[Figure]` with each figure's `chunk_id` |
| limits | one headline at a time; a company the directory lacks resolves to none | one passage's worth at a time; no arithmetic across passages; per-₹-crore intensities are a known weakness |

Behind the card, each specialist is a small contract, in code:

| Piece | Headline reader | Report analyst |
|---|---|---|
| render | the zero-shot headline prompt it was trained on (`extraction.headlines`) | `figure_messages(passage, source, wanted)` + the `calculate` tool |
| model | a PEFT LoRA on Qwen3-4B, retrained from the Week 4 headline data (the Week 4 adapter is MLX on a 4-bit 1.7B base: 41/50, and Apple-only) | the v3 PEFT LoRA on Qwen3-4B |
| validate | parses as the headline record, in its three vocabularies; the percentage printed in the headline; the mention, if there is one, resolves | parses as `Figures`; every value's number printed in its passage (`check_call`'s rules), all or nothing: one figure that fails sends the whole call to the generalist |
| fallback | the same job on the generalist, with the same schema and the same check; an answer that fails it too is an error | the same |
| log | which path answered, the validation failure if any, tokens and latency | the same |

## Serving

The specialists run on a GPU, not the laptop: local serving loads the machine and does not scale.

```
aigent's router ──HTTP, OpenAI format──▶ vLLM endpoint: Qwen3-4B (loaded once)
                                          ├─ LoRA: headline-flavour
                                          └─ LoRA: report-analyst
```

- **One endpoint per base model, many adapters per endpoint.** vLLM serves one base per process and
  applies whichever LoRA a request names (multi-LoRA), batching across adapters. Both specialists are
  Qwen3-4B fine-tunes, so they share one endpoint. A future specialist on another base is a second
  endpoint; aigent's client table already maps a specialist to a URL and a model name.
- **Scale to zero.** The endpoint is serverless: it starts on the first call and stops when idle, so
  cost follows use. The price is a cold start, which the latency numbers report.
- **Why not MLX:** the Week 4 headline adapter runs only on Apple silicon (MLX targets Metal; cloud
  GPUs are CUDA), and it was trained on a 4-bit base, so the headline reader is retrained with PEFT on
  the shared base instead of converted.

## State

<!-- fmt:off -->
```python
class State(TypedDict):
    question: str
    messages: list[Message]                           # the generalist's loop
    evidence: Annotated[dict[str, Passage], merge]    # by chunk id: a re-search adds, never repeats
    routed: list[Routing]                             # each specialist call: path taken, why, cost
    claims: list[Claim]
    verdicts: dict[str, Verdict]                      # claim id -> supported | unsupported | wrong
    memo: Memo | None
    revisions: int                                    # critic loops used, capped at two
    review: Review | None                             # the human's decision and note
```
<!-- fmt:on -->

The graph runs with a checkpointer, which is what lets it stop at the human gate and resume later.
The output is one validated Pydantic `Memo`: a two-sentence summary, claims each with at least one
citation (a chunk id or a fundamentals key) and the figures behind it, and caveats.

## Failure modes, and what catches each

| Failure | Guard |
|---|---|
| The specialist returns malformed or unfaithful figures | validation in the router; fallback to the generalist, counted |
| The generalist prompts a specialist in its own words | it cannot: the adapter renders the prompt from structured arguments |
| The right figure from the wrong year | `search_filings` scoped by year; every figure carries its year; the critic names year mismatches |
| A claim whose citation does not say what it claims | the critic, per claim |
| A claim with no citation | `Claim.citations` has a minimum of one |
| Arithmetic done in the head | computed values come from `calculate`, with their inputs |
| A critic–generalist loop | revisions capped at two, then the human sees what is still unsupported |
| Instructions planted in a filing or a web page | tool output fenced as untrusted (invariant 28) |
| A run that costs too much | `llm.core`'s free count and per-task ceiling on every call |
| The specialists' endpoint is cold | the call waits for it to start and takes the specialist's answer; a session warms it with one request first |
| The specialists' endpoint cannot be reached, even after that wait and one retry | the task fails and says so: nothing answers in the specialist's place (Vedant, 2026-10-13, while we are learning) |

## Eval plan

**50 tasks with ground truth**, written by hand — single figures, year over year, across companies,
computed from filings and fundamentals, multi-hop, must-decline, and **headline-to-filing** tasks
that start from a pasted headline ("*Tata Motors' Q4 profit falls 51%* — what did its FY25 report
say about the year?"), so one task exercises both specialists — each naming the facts the answer
must hold and the pages they are on. Each specialist is also scored on its own set: the 50
hand-labelled headlines and the 43 hand-labelled figure cases.

**The comparison that is the headline:** the same 50 tasks, three times each to measure noise
first, under

| Arm | Who reads headlines and extracts figures |
|---|---|
| generalist only | the generalist itself |
| routed | the specialists, each falling back on failed validation |

| Metric | How |
|---|---|
| task success, faithfulness | free graders plus the judge, as in Project 2 |
| cost per task, latency | from `llm.core`'s billing and the trace |
| **cost saved by routing** | generalist-only cost per task minus routed cost per task, p50 and total over the 50 |
| **saved per specialist call** | what each specialist call would have cost on the generalist (priced from its tokens at the generalist's rates) minus what it cost — a counterfactual, logged per call |
| **specialist hit rate** | share of specialist calls that passed validation |
| **routing regret** | tasks the routed arm failed that the generalist-only arm passed |
| critic agreement | the critic's verdicts against the judge's |

**A self-hosted specialist is not free per call.** On a GPU billed by the second, a specialist call
costs its share of the endpoint's running time, and an idle endpoint costs nothing only because it
scales to zero — cold starts are the price. So `s` below is measured, not assumed, and the saving is
reported at the request rates the runs actually had.

**Cost saved is not free money.** A specialist call that fails validation costs its own latency
and local compute and then the generalist's call on top, so routing pays only above a break-even hit
rate: with the generalist's cost for the job `g`, the specialist's `s` (near zero locally) and hit rate
`h`, the routed cost is `s + (1 - h)·g` against `g`, so it saves when `h > s / g`, and every point of
hit rate above that is saved money. Latency has the same shape, and a slow local model can lose on
time while winning on money. Both are reported, so the claim is "routing saved X% at Y regret", never
a saving alone.

An iteration counts only if it clears the run-to-run spread. Every paid run is estimated first and
needs its own go-ahead.

## Build order

**Week 6:**

1. ~~The corpus and tools: FY24 reports, `search_filings` scoped by company and year, the
   fundamentals table.~~ Done 2026-10-12 (`analyst.filings`, `analyst.fundamentals`).
2. **Serve the specialists:** retrain the headline reader as a PEFT LoRA on Qwen3-4B and score it
   on the 50 headlines against the Week 4 adapter's 41/50; serve it and the report analyst from one
   vLLM endpoint that scales to zero; check the served outputs match the training-time evals.
3. ~~**The MCP side:** add `search_filings` and `fundamentals` to aigent's MCP server; the MCP client
   adapter that loads a server's tools into the generalist's loop.~~ Done 2026-10-13
   (`mcp_server`, `mcp_client`); wired into the generalist in step 5.
4. **The router and the specialist contract**, for both specialists, with validation and fallback,
   tested against fakes before any model runs. Built 2026-10-13, all against fakes: the contract and both checks
   (`analyst.specialists`), the router (`analyst.router`), the `hostllm` clients priced at the
   served rate, and both specialists wired to models with the warm-up (`analyst.served`). Not yet
   run against the live endpoint.
5. **The graph:** generalist loop, critic, human gate with a checkpointer — stubs first, prompts
   after. Built 2026-10-13 against fakes (`analyst.toolbox`, `analyst.memo`, `analyst.graph`): the
   critic is Opus 5 while building (Vedant's call), and an eval's approver always approves. The
   prompts are first drafts; no live run yet.
6. The 50 tasks, then the first baseline runs.

**Week 7:** the routed-vs-generalist comparison, the critic iteration, the judge bake-off, a learned
router from the eval logs, HTTP and a container, the CI gate.

**Week 8:** README, diagram, walkthrough, write-up, tag.

## Open questions

- **Is the specialist good enough to be worth routing to?** v2 passed 28/43 on the old labels; v3's
  numbers are pending. A low hit rate is a result, not a failure: it says when a fine-tune earns its
  place.
- **The critic's model.** A different model from the generalist; which way round is a Week 7
  measurement.
- **More specialists.** Once the contract serves two, a third (section classification, note
  summarisation) is a data and training job, not new plumbing.
