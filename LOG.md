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
- 09-11: labelled 8 of the 20 backlog headlines (the ones the current schema already settles) and
  added 7 companies to the ticker directory. De-duped the directory: matching is case-, punctuation-
  and Ltd-insensitive, so `PVR INOX`/`PVR Inox` was one spelling stored twice and an alias restating
  the ticker or the registered name was a third copy — 61 aliases to 44, nothing resolves
  differently. Two guards added for it, one of which has teeth beyond tidiness: `Resolver` indexes
  with `setdefault`, so two companies claiming one spelling would silently route every mention to
  whichever sorted first (`SBI` vs `SBI Cards` is the near miss).
- 09-11: three rules recorded, each verbatim in `~/.claude/CLAUDE.md` and this repo's `CLAUDE.md` —
  hand-edited reference data stays in one canonical order (with the exception for sequences whose
  order *is* the data, which is why `METRICS` is not in the registry), and comments stay to one
  line. `tests/test_reference_data.py` is the opt-in registry; generalising it exposed that its
  duplicate-key check was vacuous, since `json.loads` collapses repeated keys before the assertion
  ever sees them. It reads raw pairs now.
- 09-11: dropped the `near-miss-ticker` tag — it described a relationship between two tickers, which
  the model never sees now that code resolves them. Replaced with `name-contains-name`, which names
  a difficulty the model does have: `Tech Mahindra` contains `Mahindra` (M&M) and `SBI Cards`
  contains `SBI` (SBIN), so a truncated mention resolves to a real but wrong company and the
  resolver reports success. That is the only silent failure this design has. A test derives the
  tagged set from the directory in both directions, so it cannot drift as the directory grows.
- 09-11: labelled the last 12 backlog headlines, so all 50 rows now carry an `expected`. Each of the
  12 was deferred because it needed a schema decision rather than a judgment call, and the decisions
  are recorded where the model can act on them — in the field descriptions. `metric` gained `other`
  (a company metric outside the six: GMV, provisions, subscribers, volumes) and `none` (a headline
  reporting no metric at all), which closes the vocabulary; without them the tie-break rule forced
  `revenue` onto every off-vocabulary row. `quarter` gained `H1`/`H2`, and a bare month is now
  explicitly not a quarter. `company` became nullable, because a headline about a sector has no
  company to extract and `null` is an answer rather than a miss. Two more rules settle what used to
  be coin flips: two metrics that still tie go to the one mentioned first, and a share price move is
  the market's number rather than the company's, so it is never the metric and its percentage is
  never `change_pct` — that one alone decides `hl-034`, where the only stated percentage belongs to
  the stock.
- 09-11: unlisted subsidiaries resolve to the listed parent, stated in the `company` description.
  This is the one place the model is asked for a fact rather than a copy, and it is a deliberate
  exception to invariant 13: the set of subsidiaries is not enumerable, so a directory cannot hold
  it, where the set of listed companies is exactly what the directory *is*. Code still does the
  join. Eight companies added to the directory (45 now), including `ETERNAL` with `Zomato` as an
  alias — a rename the model's training data may predate, which is its own tag now.
- 09-11: found three places where the prompt quoted its own test set — the `change_pct` description
  used `hl-008`'s exact numbers as its example of a level, and two illustrations I had just written
  reused `hl-038`'s and `hl-043`'s wording. An example that is also a test case hands the model that
  row and the row still counts as a pass, which is the quiet way an eval stops measuring. Recorded
  as invariant 15 and guarded: no four-word run of any headline may appear in the prompt. Three
  words fires on financial boilerplate like "Q1 revenue up"; four caught every real leak and nothing
  else. Writing the spec is exactly when this happens, because the clearest example of a rule is
  usually the case that forced you to write it.
- 09-11: cut the extraction prompts by a third, 4,378 characters to 3,155. The field descriptions
  are prompt rather than documentation, and they had drifted into explaining themselves — "A
  headline often touches more than one", two worked illustrations where one carries the rule.
  `metric` alone came down from 1,016 characters to 661 with no rule lost. The bigger cut was
  `FEW_SHOT`'s trailing 680-character paragraph, which restated each field description in prose:
  that made the few-shot arm a *third* thing — the same instructions said twice, plus examples — so
  the ablation could not say which half was paying. Two lines survive, for the two examples that
  read backwards. Cut one thing too far on the first pass: the quarter description was left
  anchoring only Q1 and Q4, and four labelled rows ride on the middle two, so the fiscal calendar
  went back in as "the quarters ending June, September, December and March are Q1 to Q4" — still
  shorter than the longhand it replaced. Prompt caching will now save less, because there is less
  to cache.
- 09-12: first paid run over all 50 rows, $1.00809. 49/50 whole-record on both arms after two label
  fixes, and — the result that matters — the two arms fail *identically*. `few_shot` costs 26% more
  per row and bought nothing measurable. That is a cleaner answer than the previous run could give,
  because the arms now differ only by the six worked examples: the prose that restated the field
  descriptions is gone, so "the examples paid nothing" is a claim about examples rather than about
  saying the same rules twice. The rules were already in the descriptions.
- 09-12: one of the two failures was mine. `hl-032` was labelled `ticker: null` when the rule I had
  just written says two companies with equal claim go to the one named first — the model answered
  INFY in both arms and was right. Caught only because the model disagreed with it, which is worth
  remembering: a label nobody argues with is not the same as a label that is correct.
- 09-12: relabelled `hl-043` (DMart same-store sales) from `other` to `revenue`, agreeing with both
  arms — SSS is revenue narrowed to comparable stores, not a quantity of its own. That leaves a
  known inconsistency: `hl-036` (LIC new business premium) is still `other`, and NBP is a narrower
  cut of premium income by the same argument. Both rows now carry `narrowed-metric` so the tension
  is sliceable rather than invisible. I did **not** write the rule that would resolve it — "a
  narrower cut of one of the six is that metric" would flip `hl-036`, which currently passes, and
  the honest boundary (SSS is reported as a cut of revenue, NBP as its own line) is domain
  convention, not something the schema can state. Debt, recorded as debt.
- 09-12: kept `hl-032`'s `metric: none` against both arms, which read "IT spending outlook dims" as
  `guidance`/`down`. A defensible reading of the text; the label's case is that the dimming outlook
  is the customers' spending rather than either company's guidance, a line the schema does not draw.
  Left as the one real disagreement on the set rather than tuned away.
- 09-12: prompt caching, $0.00991/row down to $0.00238 — 76% off. The result that matters is not the
  saving but that both arms failed the *same two rows with the same two answers*: the cache is a
  serving detail, so exact agreement is what proves the wiring rather than the discount. Two things
  the seam note had wrong. The cacheable bulk is not the system prompt at all — `FEW_SHOT` is 567
  tokens and the `output_format` schema is 1,148, and since the minimum cacheable prefix is 1,024,
  neither prompt could have been cached on its own. And the pre-flight estimate had been ~37% low
  for all of Project 1a, because `_rough_input_tokens` divided the prompt by four and never saw the
  schema. `measure()` counts the real request now; counting is free, so there was never a reason to
  guess. Checked before spending that a request without `cache_control` does not read a warm cache,
  or the baseline arm would have been silently subsidised and the comparison void.
- 09-12: asked for `temperature=0` and it does not exist. **Claude 5 deprecated `temperature` and
  `top_p`** — both 400 on Opus 5 and Sonnet 5; only Haiku 4.5 still takes them. Reverted rather than
  ship code that would fail every call. Chasing why a five-field record ever needed 384 output
  tokens found the real cause: every call emitted a thinking block of 200-280 tokens, billed as
  output and charged against `max_tokens`. That one fact explained all three symptoms — the row
  that flipped between runs, the two truncations, and a cost outlier at 3x the median. Not sampling
  noise; thinking spending the budget before the answer began. `MAX_TOKENS_HEADLINE` 384 to 128,
  and its old comment claiming to size "a five-field record" was never true.
- 09-12: thinking off is **not** a free win, which is worth stating plainly because it looked like
  one for an hour. 50/50 with it on, 48/50 with it off, for 56% less per cached row. Both losses are
  legible: `hl-031` stops applying the subsidiary rule and returns `Reliance Jio` verbatim, so the
  rule in `Extraction.company` is thinking-dependent — the one place we ask for a fact rather than a
  copy is the one place that needed the reasoning step. Kept off regardless: the 50/50 it replaced
  was unrepeatable, and an instrument that cannot be repeated cannot rank two prompts.
- 09-12: a labelling lesson with a cost attached. `hl-043` was moved to `revenue` yesterday because
  both arms said `revenue` — decided on an instrument we did not yet know was nondeterministic, and
  with thinking off the model now says `other`, the label I had originally argued for on the merits.
  The label stands by decision, but "both arms agreed" has stopped being a reason for it. Agreeing
  with the model is only as good as the run doing the agreeing, and a second opinion from the same
  model on the same day is not two opinions.
- 09-12: `week01.failures` — break things on purpose. Nine failure modes provoked for real and
  tabulated into `docs/failure-modes.md`, rather than described from the docs. The design constraint
  is that the whole thing is **free**: a request rejected with a 4xx never reaches the model, so
  there is nothing to bill, and the last two rows never touch the network at all. Two rows started
  out paid — a truncated record and a below-minimum cache breakpoint — and were cut, which made the
  module better: a catalogue nobody has to think twice about running is one people will actually
  run. A test asserts `messages.parse` does not appear in the source, since a *successful* call is
  the only way this could start costing money.
- 09-12: two findings from building it. `max_tokens=10_000_000` never reaches the API — the SDK
  raises `ValueError` client-side about streaming being required past ten minutes. And
  `context-too-long` never becomes a 400: `check_request` counts it on the free endpoint and trips
  `BudgetExceeded` first, with the number and the knob in the message. The guard we wrote is
  strictly more useful than the rejection it pre-empts, which is the argument for pre-flighting.
- 09-12: the failure deliberately left out is the one I most wanted in: a `cache_control` breakpoint
  on a prefix under the model's minimum is silently ignored — no error, correct answer, full price
  on every call. Provoking it needs a successful call, so it is recorded in the module docstring and
  the graph as a known hazard rather than bought. Worth restating because it bit us in reverse this
  week: the value of the caching work was 1,148 tokens of `output_format` schema, and had the prompt
  been the only thing behind the breakpoint it would have been under the minimum and cached nothing.

## Week 2 (2026-09-14 to 2026-09-20)
- 09-14: Week 2 laid out as a **skeleton to write by hand**, the same arrangement as `underhood` on
  Track B: signatures and docstrings in `src/`, the tests written in full as the specification, and
  `uv run pytest -q` as the to-do list. 436 green, 58 red on purpose. What I have to write:
  `week02/chunk.py` (four splitters), `week02/embed.py` (`LocalEmbedder`), `week02/store.py`
  (brute-force cosine), `grade.recall_at_k` and `grade.reciprocal_rank`, and
  `report._metrics_table`. `config.py` holds every constant they read; `Score.value` is in place,
  because a dataclass field is shape rather than logic.
- 09-14: three design decisions settled before any of it is written, each recorded where the code
  will have to honour it.
  - **Labels are quotes, not chunk ids.** Chunk ids are positional, so they deliberately do not
    survive a re-chunk — the 43rd chunk of a differently-split document is different text.
    Labelling ids would mean re-labelling for every chunking strategy, and then the four strategies
    could not appear as four columns of one table. `Inventory.containing` resolves a labelled
    sentence against whichever inventory is being scored, which is the ticker directory's rule one
    week on: code does the join, the model is never asked where the answer lives. A quote that
    resolves to nothing is a finding rather than a bad label — the chunker split the answer across
    a boundary, so that strategy has capped its own recall before the embedder runs. Invariant 17.
  - **`Score` needed a number.** The seam note predicted the retrieval metrics would need no
    harness change; half right. The free-task path (`Outcome(usage=None)`) needs nothing, but a
    reciprocal rank of 0.5 is not a failure, and a boolean column averages rank 2 and rank 40 into
    the same nothing. Hence `Score.value` and a means table beside the pass-rate table. The test
    that pins why both belong: a retriever finding the right chunk *every* time and never at rank 1
    reads 0/4 in the summary and 0.500 in the metrics.
  - **`fixed` must not snap to word boundaries.** It is the baseline the other three are measured
    against, and a baseline quietly improved flatters everything compared against it. The test
    asserts the mid-word cut rather than apologising for it.
- 09-14: two traps written into the specs because they are silent when missed. BGE v1.5 wants its
  instruction prefix on the **query side only** — embed both sides alike and retrieval still works,
  just measurably worse, with nothing in the output to say so. And the model reads 512 tokens and
  drops the rest without a word, so `count_truncated` has to count what a character-budgeted chunker
  actually fed it; a strategy whose recall looks bad gets checked there before it gets blamed.
- 09-14: `Inventory.build` has to refuse two documents sharing a `doc_id`. Their chunk ids would
  collide, `by_id` would keep whichever came last, and every label naming one would resolve to the
  wrong text with nothing raised — the same silent shape as the `setdefault` near-miss in the ticker
  directory three days ago (`SBI` vs `SBI Cards`).
- 09-14: deps added for the week — `sentence-transformers` (torch, transformers, numpy) and `pypdf`.
  The first dependencies that are not about talking to Anthropic, because Anthropic ships no
  embedding model. MPS confirmed available; `bge-small-en-v1.5` is 384 dimensions, 512 tokens.
- Next: fill the skeleton top-down (`sentences` → `headings` → splitters → `Inventory` → store →
  graders → metrics table), then the corpus: annual report PDFs into `Document` with `page_starts`
  so citations survive, then the question set labelled with quotes.
- 09-16: `week02/` written by hand against the tests — the four splitters, `LocalEmbedder`, and the
  brute-force store. 38 tests green. Findings worth keeping, in the order they bit: the overlap
  step was off by one (`curr_index - overlap + 1` gives no overlap at all at `overlap=1`, and at
  `overlap=0` it skips a sentence and then runs off the end of the span list); chunk text has to be
  stripped or no chunk ends on a full stop, since a sentence span carries its trailing whitespace by
  design; and `by_sentence` had to **lose** the `CHUNK_MIN_CHARS` floor entirely. That last one is
  the real lesson — a tiny sentence in front of an over-long one was flushed alone, dropped for
  being short, and never revisited, so `containing` reported the quote inside it as split across a
  boundary when the text had simply been deleted. The floor guards artefacts of the *splitting
  mechanism*; packing whole sentences produces none, so bounding both ends only cost text. `fixed`
  and `by_heading` keep it, and the test that proves a drop moved to `fixed` where it is real.
- 09-16: `EMBED_DIMENSIONS`, `EMBED_MAX_TOKENS` and the query prefix now come off the loaded model
  rather than from constants. The prefix is the interesting one: bge-small-en-v1.5 **cannot**
  declare it — its `config_sentence_transformers.json` was written by sentence-transformers 2.2.2
  and the `prompts` mechanism postdates it, so `model.prompts` is `{'query': '', 'document': ''}`.
  The string exists only as prose in the model card's comparison table. So `LocalEmbedder` prefers
  what the model declares and falls back to `EMBED_QUERY_PREFIX`, and the constant is documented as
  transcribed-by-hand rather than authoritative. A two-document probe had the prefix *narrowing*
  the margin slightly, which is far too small a sample to mean anything — it is a variant to
  measure once the question set exists, not a setting to argue about.
- 09-16: comment and docstring rule rewritten in both `CLAUDE.md` files. The old one sent paragraphs
  *into* docstrings, which is exactly how they got long; the new one says a comment is one line, a
  docstring is a sentence or two, and rationale belongs in this file or the knowledge graph. Applied
  across the repo: 5,588 lines to 5,015, prose from 22% to 13%, no behaviour changed and the test
  count identical either side. All 98 graph anchors recomputed, since every line number moved.
- Next: `recall_at_k`, `reciprocal_rank`, `_metrics_table`, then the corpus.
- 09-16: package reorganised by capability rather than by week. `week01/` was two different things
  wedged together — five CLI demo modes and Project 1a, which has a dataset and an eval behind it —
  so it split into `primitives/` and `extraction/headlines.py`; `week02/` became `retrieval/`. The
  test tree mirrors the package (`tests/retrieval/`, `tests/evals/`, and so on), with `conftest.py`
  staying at the root so its fixtures still reach everything. Moved with `git mv`, so history
  follows the files. Nothing behavioural changed: 474 passing and 21 red either side. Every
  `parents[N]` path computation survived because each module kept its depth. The graph's module ids,
  diagram, edge list, test paths and all 98 anchors were rewritten with it — `extraction.py`
  anchors became `headlines.py`, which is the rename a line-number check alone would have missed.
- 09-16: the corpus exists as documents. Three Indian annual reports — ITC (412pp), Reliance
  (146pp), Tata Motors (590pp) — 1,148 pages and 4.2M characters, about 3,500 chunks at
  `CHUNK_CHARS`. Not committed (85MB, and not ours to redistribute); `corpus-manifest.json` holds
  the URL, SHA-256, size and page count, and `scripts/fetch-corpus.sh` rebuilds the directory from
  it. Acquisition is its own small finding: Infosys, TCS, HUL and Tata Motors all serve a bare
  `curl` a 403, so the script sends a browser `User-Agent` and a `Referer`; guessed URLs missed six
  times out of six and the working Tata Motors link came from reading the IR page.
- 09-16: `retrieval.corpus` written as a skeleton with 18 red tests, and the specification changed
  twice before a line of it was implemented, both times because the real PDFs were consulted first.
  **Furniture cannot be found verbatim.** The plan was "a line repeating on half the pages"; ITC's
  running header sits on 163 of 412 pages and on *none* of them twice, because the page number is
  glued to it (`REPORT AND ACCOUNTS 2025 47ITC Limited`). Masking digits finds it; so the rule
  counts a digits-masked skeleton. And it counts only lines at the top or bottom of a page, because
  `(Rs. in crore)` repeats on 86 ITC pages mid-table and is content, not furniture. Tuned on the
  corpus rather than guessed: edge lines, digits masked, 20% of pages removes 1.1% of ITC's lines
  and 1.7% of Reliance's, and nothing that reads as content.
- 09-16: **the typography fold belongs in `_squeeze`, not in `normalise`.** The corpus holds
  `ITC’s` and an em dash; a label is typed by hand and holds `ITC's` and a hyphen, so `containing`
  would have resolved a correct label to nothing — the same silent shape as a quote straddling a
  chunk boundary, but with no chunking problem behind it. The fix is one `str.maketrans`, and the
  question worth recording is *where*: folding in `normalise` would have put ASCII quotes in the
  chunk text and therefore in every citation. A curly apostrophe is not an extraction artefact, it
  is what the document says. So `normalise` repairs what extraction broke (ligatures, `CO₂`,
  hyphenated line breaks, tabs) and `_squeeze` handles what two spellings of the same text means.
- 09-16: what PDF text actually looks like, for whoever is surprised later. Reliance renders `₹` as
  `C` (`(C in crore)`) because the font's encoding does not survive; kerning becomes spaces inside
  words, so `STANDALONE FINANCIAL STATEMENTS` extracts as `ST ANDALONE FINANCIAL ST A TEMENTS`;
  ITC uses tabs between words mid-sentence. None of that is fixable in general, and a heading is
  where it shows up most — which is a reason to expect `by_heading` to underperform its promise and
  a reason to measure it rather than assume.
- 09-18: `corpus.py` written, all 18 tests green, and the corpus loads: 4,201,217 raw characters
  become 4,040,587 across three `Document`s, about 3,400 chunks. The 3.8% removed is furniture —
  326 lines from ITC, 434 from Reliance, 1,778 from Tata Motors. Page offsets verified monotone and
  round-tripping on all three. Two ordering constraints in the implementation are worth knowing
  because both fail silently: `rstrip` per line has to run *before* the hyphen rejoin, since a PDF
  line ends `manage- ` with a trailing space and the regex cannot match across it; and pages have to
  be normalised *before* furniture is counted, since the count compares lines and un-normalised ones
  differ by trailing spaces and ligatures, so a header would not match its own other printings.
  Known cost of the furniture rule, accepted: Reliance's `CONSOLIDATED FINANCIAL STATEMENTS` is a
  running header on 52 pages and is removed as one, so `by_heading` loses that section label. Check
  there before blaming the splitter if its column disappoints.
- 09-18: the eval design changed before any of it was paid for, because the question "what are we
  really testing" did not have a good answer. A single recall@k column bundles three different
  things — did ingestion keep the sentence, did the chunker keep it whole, did the embedder rank it
  — and only the third is retrieval. The second is now measured **separately and for free**, over
  300 real sentences with no embedder involved at all:
  `by_sentence` 100%, `fixed+overlap` 98.3%, `fixed` 87.7%, `by_heading` 85.3%.
  That is the ceiling on recall@k for each strategy before retrieval happens. Three things fall out.
  `by_sentence` is 100% by construction, so any shortfall in its recall is genuinely the retriever's.
  Overlap is now justified by a number rather than a hand-wave: it buys 87.7% → 98.3%, because a
  sentence severed by one window boundary survives whole in the neighbouring one. And `by_heading`
  is the *worst* of the four while producing the *most* chunks (7,506, averaging 538 characters
  against `by_sentence`'s 880) — heading detection firing on kerning damage, exactly as the 09-16
  entry predicted. Decision: keep the implementation, drop it as an eval variant, revisit later.
- 09-18: `grade.resolvable` added, and it turned out the harness was already half right by
  accident. `_metrics_table` skips `value is None`, and the retrieval graders already returned no
  value for an unresolvable row — so recall@k was *already* meaned over resolvable rows only. What
  was missing was the denominator, which made the column quietly describe a subset while looking
  like a dataset average. The demonstration, on a three-case fixture: `fixed` reads recall@5 1.000
  and resolvable 0.667. Without the second row it looks perfect; with it, "finds everything that
  survives chunking, and only two thirds survives". Same numbers, opposite conclusion.
- 09-18: `retrieval.questions` written — a generator that samples passages, asks for one question
  plus the sentence that answers it, and **verifies every quote back against its own document**
  before writing a row. The verification is the point: a paraphrased quote resolves to no chunk
  under any strategy, so it reads as four retrieval failures when retrieval was never asked a fair
  question. Dry run (free): 60 passages, 1,350 input tokens for the largest, worst case $1.17 on
  Opus and $0.47 on Sonnet. Nothing sent yet.
- 09-18: the question set exists, and it cost nothing. The dry run had priced 60 generation calls
  at $1.17 worst case on Opus; the obvious question — why call the API at all when the session
  model can read the passages and write the questions directly — had no good answer. So the 54
  rows in `evals/datasets/retrieval.jsonl` were authored by hand against passages pulled from the
  corpus, and put through the **same** `verify` gate the scripted path uses. 53 of 54 passed on the
  first attempt; the single rejection was a real quote sitting under the 25-character floor, which
  is the floor doing its job rather than a bad label. `questions.generate` stays as the
  reproducible regeneration route.
- 09-18: two biases in that dataset, stated here so a number quoted from it carries them. The
  questions are written *from* the passages, so they inherit that passage's vocabulary and absolute
  recall will read high — what survives is the **ordering** of strategies, since all three face the
  identical set. And the quotes are clause-length rather than whole sentences: `fixed` resolves
  96.3% of these labels but only 87.7% of 300 random sentences. Both are real and answer different
  questions ("do these labels survive" vs "does a sentence survive"), so quote them together.
- 09-18: passage selection needed its own filter. Even spacing over the inventory put 6 of 60
  candidates in auditor boilerplate — "audit evidence obtained by us", "Responsibilities of
  Management" — which is near-identical across all three reports, so a question drawn from one is
  legitimately answerable from the other two while only one is labelled relevant. Excluding that
  language and requiring two specific facts per passage left 1,234 candidates to choose from.
- 09-18: the plan was to cache the corpus so `load_corpus` would stop costing ~50s a run. Profiling
  first moved the answer. The 75s from PDFs to chunks split as **28s PDF extraction, 0.4s
  normalise and furniture, and 46.9s chunking** — chunking was slower than parsing 1,148 pages of
  PDF, which is not a caching problem. `sentences` read the word before each candidate break with
  `text[: match.start()].split()`, slicing and splitting the entire prefix on every match: O(n²),
  billions of character copies on a 1.3M-character document. Only the last word is ever used, so a
  64-character lookback is enough. **46.9s → 0.05s**, same 4,593 chunks, no test changed. The
  lesson is the ordinary one and worth the entry anyway: the fix for a slow pipeline was in a
  function nobody suspected, and the cache would have hidden it by making the total look fine.
- 09-18: with that gone, the remaining cost is genuinely PDF extraction, so `load_pdf_cached` keeps
  each document's text in `corpus/.cache/` — **cold 27.8s, warm 0.04s**; corpus to chunks is now
  0.09s against ~75s. Cached at the `Document` layer rather than the chunk layer: chunking is free
  now, and a chunk cache would tie itself to one strategy and one set of `CHUNK_*` constants. The
  cache key is the interesting part — `{doc_id}-{pdf digest}-{ingest fingerprint}`, where the
  fingerprint hashes the source of the five ingestion functions plus the constants they read. Bytes
  alone would survive an edit to `normalise` and serve back text the current code never produced,
  silently. Hashing the code means there is no version number to remember to bump, and being wrong
  costs one spare re-parse rather than a wrong answer. Written to a `.partial` and moved into
  place, so a Ctrl-C leaves nothing readable behind.
- 09-18: the cache is **opt-in**, `cache=False` by default, on the argument that it is the wrong
  default however good the keying is. Reading the PDFs is always correct; a cache is correct until
  it is not, and its failure is silent — it hands back plausible text and every number downstream
  moves without anything being raised. The keying makes that unlikely, not impossible, and 28
  seconds is cheap next to a run whose numbers are quietly from a previous version of `normalise`.
  So callers ask for it at the call site, where it is visible in review, and a test pins the
  default so an opt-in convenience cannot drift into being the norm.
- Next: the first measured run — three variants over one dataset, `resolvable` beside `recall@5`
  and `mrr`. Then hybrid and rerank.
