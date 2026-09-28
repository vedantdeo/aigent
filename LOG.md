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
- 09-18: `by_sentence` was emitting mega-chunks and the eval could not see it. 172 chunks came out
  over 2,000 characters, holding **11.5% of the corpus**, the largest 14,947 characters with seven
  sentence breaks inside it — financial tables, which carry almost no sentence-ending punctuation.
  The old rule was "a sentence longer than the budget becomes its own chunk rather than being cut",
  argued from keeping the comparison with `fixed` honest. That argument was wrong about this
  corpus: the embedder reads 512 tokens, so most of a 15k-character chunk was never indexed while
  the chunk went on claiming to contain it. An uncut mega-chunk is not a fairer comparison, it is a
  chunk that lies. `emit_capped` cuts a span with no break in it on the character budget:
  **truncated chunks 185 → 47, recall@5 0.581 → 0.609**, chunk count 4,593 → 5,193.
- 09-18: **the first measured run.** `retrieval.evaluate`, three strategies over the 54 questions,
  206s and **$0.00**. `resolvable` 0.963 / 1.000 / 1.000, hit@5 0.635 / 0.630 / 0.685, recall@5
  0.619 / 0.583 / 0.608, MRR 0.471 / 0.511 / 0.452 for `fixed` / `fixed+overlap` / `by_sentence`.
  **Three metrics, three different winners.** The largest gap is four cases out of 54, and a
  four-case gap on 54 paired rows cannot reach p<0.05 by a sign test even when every discordant row
  falls the same way — the minimum attainable two-sided p is 0.125. So the honest reading is that
  these three are not separable here, which is a finding about the dataset's size before it is one
  about the chunkers. The one thing the table does say cleanly: `fixed` caps itself before the
  embedder runs, losing 2 of 54 labels to its own boundaries.
- 09-18: the run needed `runner.combine`, which the seam note had not predicted. These variants
  disagree about the **label**, not only the task — a chunk id names different text in each
  inventory — so each strategy is scored against its own resolved cases and the three runs merged
  into one table afterwards. It refuses runs differing in dataset, digest or graders, and a
  repeated variant name: the three ways the merged columns would stop being comparable while the
  table still rendered.
- 09-18: `hit@k` added after the fact, and it changes the answer. `recall@k` gives partial credit
  across a label that resolved to several chunks, but those chunks are usually one sentence seen
  through several overlapping windows rather than several facts — any one of them answers the
  question. So recall charges a strategy for the overlap that made its quote resolve at all, and it
  falls hardest on the strategy with the most of it (a label resolves to a mean of 1.10 chunks under
  `fixed`, 1.41 under `by_sentence`). Recall@5 ranks `fixed` first; hit@5 ranks `by_sentence` first,
  **over the identical rankings**. Neither is wrong; they answer different questions, and the one
  that decides whether a generator can answer at all is hit@k.
- 09-18: working a `hit@k` example on a real row found a **bad label**, which is the usual way round
  — the eval's first real output was a bug in its own dataset. `rq-054` asked which firm assured
  Tata Motors' BRSR Core attributes and was labelled `KPMG Assurance and Consulting Services LLP`.
  True. Also true of ITC, whom KPMG also signs. The quote resolved to **21 chunks across two
  documents**, so the retriever was scored wrong for returning a chunk that genuinely held it.
  `verify` had three checks and all three asked whether the quote was *in* the document; none asked
  whether it was *of* it. Fixed to a quote naming one passage (one chunk under each strategy), the
  gate tightened to reject a quote found anywhere else in the corpus, and the whole set swept: no
  second instance. `rq-009` resolves to five chunks but all in ITC — a repeated ESOP sentence, a
  correct label, and the row to remember when reading recall@5, since it cannot score above 1/5 at
  k=5 unless all five come back.
- 09-18: the question set joined `tests.test_reference_data`'s registry, the third file to opt into
  the canonical-order rule. It is hand-edited and code reads it back, which is the whole trigger; a
  repeated `rq-` id would otherwise be a row that quietly scores twice.
- 09-19: `retrieval.answer` written — the generation half, and the first part of Week 2 that
  spends. Two arms over the identical questions, `closed_book` and `rag`, because the measurement
  is the **gap**: an absolute RAG score conflates what the retriever found with what a large model
  already knows about ITC, Reliance and Tata Motors. Graded by `cites_relevant` (free — `hit_at_k`
  pointed at the *cited* ids, so a citation is checked against the label rather than trusted) and
  `correct` (`LlmJudge` against the labelled quote). Nothing sent yet.
- 09-19: the dry run priced it at **$3.83 worst case against a $2.00 ceiling**, and the first thing
  that bought was a re-reading of the output caps. `MAX_TOKENS_ANSWER` was 512 for a record that is
  two sentences plus five chunk ids (~140 tokens) and `MAX_TOKENS_JUDGE` 1024 for one sentence and
  a bool. Both came down to 256, which is the `MAX_TOKENS_HEADLINE` lesson a second time:
  `estimate_eval_usd` prices the full cap, so an oversized cap is a guard that refuses runs it has
  no reason to refuse. **$3.83 → $2.31**, same real spend.
- 09-19: two things the dry run also bought, both free. `LlmJudge` gained `reference`, naming the
  one `expected` field the judge may see — a retrieval label carries the chunk ids it resolved to
  beside the quote, and those are noise in a correctness judgement, a hint at worst, and resent on
  every one of 108 verdicts. And `_prompt` became `prompt_for`, public, so the dry run counts the
  judge's real prompt instead of the 1,200-token guess it started with: the judge is half this
  eval's calls, and pricing half a run from a number typed from memory is the thing the dry run
  exists to replace.
- 09-19: **smoke run, 12 of 54 cases, $0.295 against a $0.49 worst case.** Closed-book 0/12 correct,
  RAG 10/12. The gap is the whole point and it is not subtle: same questions, same model, and the
  only difference is five retrieved passages. `evals/reports/retrieval-20260919-0754.md`.
- 09-19: the smoke run earned its $0.295 by finding two things the full set would have charged
  $2.22 to find. **Closed-book abstained 11 times and confabulated once** — `rq-001` answered "over
  Rs 32,000 crore" where ITC reports Rs 34,000 crores, a wrong figure of exactly the shape nobody
  catches by eye. And the table could not see the difference: the rubric fails a decline, correctly,
  which put the honest 11 and the invented 1 in the same bucket. `grade.flag` closes that — a
  boolean the task reports about itself, counted as a rate, for the thing a dataset cannot label
  because it is a property of the answerer rather than the answer.
- 09-19: `cites_relevant` came out at 0.667 against the retriever's own hit@5 of 0.685, which says
  the model cites what it was handed and does not invent ids. All four citation failures were
  checked by hand and **all four were retrieval misses, not citation failures**. The interesting
  half is that correctness (0.83) runs ahead of both: on `rq-054` retrieval missed chunk `#0532` and
  returned `#0531` and `#0530` — its neighbours — and the answer was right anyway, because a label
  names one passage and an annual report states the fact across a section. **hit@5 is a floor on
  what RAG can answer, not a ceiling**, which corrects how the 09-18 table was written up.
- 09-19: **the full 54, $1.342** against a $2.22 worst case and a ceiling raised to $2.50 for the
  run. Closed-book `answered` 10/54, `correct` 5/54; RAG `answered` 43/54, `cites_relevant` 37/54,
  `correct` 42/54. `evals/reports/retrieval-20260919-0821.md`. **9% to 78%** is the headline, and
  the smoke run had been wrong about the floor — it read closed-book as 0/12 because the 12-case
  sample happened to miss all five rows the model knows from memory. A sample can be wrong about a
  rate in either direction; it was right about the shape.
- 09-19: the real finding is in the `answered` column, and it is a safety number rather than a
  quality one. **Closed-book is a coin flip when it chooses to speak: 10 answers, 5 right.** Its
  judgement about what it does not know is good — 44 abstentions out of 54 — and its confidence
  when it does answer is worth nothing. With passages it is 42 right out of 43 answered, declining
  the other 11. The failure mode moves from fabrication to silence, which is the whole argument for
  RAG stated as a measurement rather than a belief. `grade.flag` is what makes it visible; the
  `correct` column alone reads 5/54 and 42/54 and cannot distinguish a careful model from a
  reckless one.
- 09-19: `cites_relevant` came out at 37/54 and so did the free retrieval eval's hit@5 — and a
  row-level check said **the same 37 rows, agreeing on all 54**, with no row retrieved-but-uncited
  and none cited-but-unretrieved. Worth recording because the equal *counts* were checked before
  the equal *rows* were, and I had claimed the stronger thing off the weaker evidence. Two evals
  written separately, one free over NumPy and one paid over a model, agreeing row for row: that is
  a tripwire on the harness, and a divergence later means a stale inventory or a chunk-id collision
  rather than a worse model.
- 09-19: known limits of that metric, stated so a number quoted from it carries them.
  `cites_relevant` is **any-hit with no precision term** — a model citing all five passages every
  time scores identically to one citing the single right passage — and this run did not persist the
  `cited` lists, so whether it cited one passage or five is not answerable from what is on disk.
  Deliberately not fixed before the tag.
- 09-20: hybrid and rerank land — `sparse.py` (BM25 as an inverted index), `fuse.py` (RRF),
  `rerank.py` (a local cross-encoder). Six arms over one `by_sentence` inventory: three bases with
  and without reranking. One run rather than merged runs, because unlike chunking strategies these
  all rank the same chunks and so share their labels.
- 09-20: first table, original questions. `bm25` 0.889 hit@5 against `dense`'s 0.685 — lexical
  retrieval does not merely help here, it dominates. And **hybrid came out worse than bm25 alone**
  (0.833), which is not what the textbook says. The diagnostic: of dense's 37 hits, 36 were also
  bm25 hits, so **dense contributed exactly one unique row in 54** and there was no complementarity
  to fuse. Worse, hybrid lost four rows bm25 had at **rank 1** and dense had nowhere in its top 100
  — because `RRF_K=60` makes a chunk both rankers liked beat a chunk one ranker put first. That is
  the property fusion is bought for, and it is worthless when one ranker is noise. `bm25+rerank`,
  an arm not in the original plan, then swept everything at 0.907 hit@5: deleting the fusion stage
  beat the textbook pipeline.
- 09-20: the question that changed the conclusion came from the user — long documents eventually
  use pronouns and paraphrase, where dense should earn its keep. Measured on the existing set
  first, free: split the 54 rows at the median by idf-weighted overlap between question and answer
  chunk. **bm25 is 27/27 on the high-overlap half and 21/27 on the low half; dense is flat at 19/27
  and 18/27** (Fisher exact p=0.023). bm25's score is a function of how much the question reuses
  the passage's words. dense does not care.
- 09-20: so the decisive experiment — `retrieval-paraphrased.jsonl`, the same 54 quotes and labels
  with the questions reworded, isolating question style as the only variable. Mean overlap 0.569 →
  0.285. **The ranking inverts.** bm25 falls from best to worst (0.889 → 0.315, below dense's
  0.389); `hybrid` becomes the best base; `dense+rerank` is the best arm at 0.500. The two tables
  recommend opposite systems and nothing changed but the wording. **An eval whose questions were
  written while looking at the passages will recommend the wrong retriever** — which is a finding
  about eval construction, not about BM25.
- 09-20: two things kept honest about that set. Ten first-draft questions were rewritten for being
  contrived rather than natural (`bank-club lending` for a syndicated loan); that cost 0.025 of
  overlap and was right, since a question nobody would ask is not a harder question but a different
  bug. And the absolute collapse is **confounded** — every arm fell, so lower overlap is mixed with
  vaguer phrasing, and only the relative inversion is robust. The author also knew the hypothesis
  while writing the questions; the unbiased version needs someone who never sees the passages.
- 09-20: last free experiment, and the prediction failed. Answers on the paraphrased set sit at
  dense ranks 7, 9, 14, 18, 27, 36, 76 — found but below the cut — so widening `RERANK_CANDIDATES`
  30 → 100 should have recovered them. It helped `bm25+rerank` (+0.037 hit@5) and **hurt**
  `dense+rerank` (−0.019). Seventy more candidates are seventy more chances for a small
  cross-encoder to promote a plausible wrong chunk: the pool size is a real optimum, and "retrieve
  wide and let the expensive model sort it out" holds only while the expensive model is good enough
  to sort. The three plain arms were byte-identical across both runs, which is the control that
  says the flag leaked nowhere.
- 09-20: the LangChain rebuild, $0.00, and the useful move was running it through **our** harness
  rather than reading its code — same three PDFs, same 54 questions, same graders, same report, so
  the comparison is a number. Result: **within one row of 54 on every metric** (hit@5 0.611 against
  our 0.630 at the same 1200/200 and the same model). The framework cost no accuracy. It cost 14
  statements where ours takes 336.
- 09-20: which means the uncomfortable half is ours. Furniture stripping, NFKC normalisation,
  hyphen rejoining and page-offset tracking — measured against LangChain's raw extraction — bought
  **~0.019 hit@5, one row**. Worth knowing before writing the next careful ingestion step, and only
  sayable because both pipelines went through one harness.
- 09-20: a surprise in the other direction. `resolvable` came out 1.000 for LangChain **despite no
  normalisation at all**, because `squeeze` folds typography at *match* time rather than at ingest.
  A design choice from a week ago turned out to make our labels portable to a pipeline that never
  normalised anything, which was not why it was made.
- 09-20: what the framework hid, all found by needing it rather than by reading docs.
  `PyPDFLoader` returns one Document per page, so chunks **cannot cross a page boundary** (510
  extra chunks at the same nominal size); `add_start_index` is off by default, and without it a
  chunk has no offset and no citation is computable; `metadata["page"]` is 0-indexed where every
  citation here is 1-indexed; three direct dependencies pull **31 transitive packages** (61 → 92),
  SQLAlchemy and a telemetry client among them; and `langchain-community` warns on import that it
  is being sunset, at current versions, installed today.
- 09-20: **the first run scored 0.000 on every metric and raised nothing**, and that is the entry
  worth keeping. A LangChain `Document` has no id, so retrieved chunks were matched back by
  `id(chunk)` — and `InMemoryVectorStore` hands back reconstructed objects, so every lookup missed,
  every id list came back empty, and empty id lists grade as clean misses. A broken adapter is
  indistinguishable from a bad retriever unless something refuses to be quiet. Fixed by stamping
  ids into metadata; the task now **errors** on an unmappable chunk instead of skipping it, and a
  test pins that. The rule earned: a metric of exactly 0.000 is plumbing until proven otherwise.
- Next: public plus branch protection on Wed 2026-09-23, and Week 3 (agent patterns) starts
  tomorrow. Worth doing when there is time: a question set written by someone who never sees the
  passages, which is the only clean measurement of what dense is really worth.

## Week 3 (2026-09-21 to 2026-09-27)

- 09-21: the ranking moved out of the eval. The six methods lived inside `retrieval/evaluate.py`,
  so anything else that wanted to search the reports could only copy them or import the eval.
  They now live in `retrieval/search.py` — `Retrievers.rank` for the eval, `Searcher` for callers
  that want passages — and `evaluate` calls the same code. Checked the only way that counts: reran
  the free six-arm eval after the move and it matched the committed report on all 324 rows,
  failures included (105s, $0.00). The duplicate report was not kept.
- 09-21: new with it, a search scoped to one report. It ranks every chunk and drops the other
  reports rather than filtering the global top k, which comes back empty for a report that ranks
  low overall and reads as "the report does not say". Default method `hybrid+rerank`, the winner
  on questions not written from the passages.
- 09-21: **every structured call in the repo was pre-flighted short.** `check_request` never took
  the `output_format` schema, and a schema is billed as input, so the per-request guard priced a
  cheaper request than the one sent. Measured with free counts:

  | call | schema | model | tokens missed | per call |
  |---|---|---|---|---|
  | `extraction.headlines` | `Extraction` | claude-opus-5 | 1,148 | $0.00574 |
  | `primitives.structured_output` | `PaperSummary` | claude-opus-5 | 571 | $0.00285 |
  | `retrieval.questions` | `Question` | claude-opus-5 | 419 | $0.00210 |
  | `retrieval.answer` | `Answer` | claude-opus-5 | 378 | $0.00189 |
  | `evals.judge` | `Verdict` | claude-sonnet-5 | 337 | $0.00067 |

  Small against a $0.25 ceiling, and it never tripped anything; the point is that the guard was
  checking a different request from the one sent. The bills were always right (they read real
  usage) and so were the dry-run estimates (`headlines` and `answer` count the schema separately).
  `check_request` now forwards `output_format` and all five pass it. The guard against it coming
  back is a test that finds every `messages.parse` in the package by parsing the source, so a new
  structured call is covered without anyone remembering to list it; breaking one site, or the
  forwarding, each failed it.
- 09-21: the ranking became classes. `Retrievers.rank(method, ...)` branched on the method name
  inside one function; now `build_ranker(method, indexes)` returns a `Ranker` whose one method is
  `rank(query, k, doc_id=)`. `DenseRanker` and `SparseRanker` each wrap an index, `HybridRanker`
  holds the two and fuses them, and `Reranked` wraps any of them — so `+rerank` is literally one
  ranker around another rather than a branch. The indexes are built once in `Indexes` and shared.
  Names moved to match: `store.py`/`VectorStore` became `dense.py`/`DenseIndex`, `Bm25Index`
  became `SparseIndex`, `Hit` got its own module, `search.py` became `rank.py`, and `context_block`
  went to `chunk.py`, beside `Chunk.citation`.
- 09-21: one behaviour did change: each dense ranker now embeds its own query, where the eval used
  to embed each question once and share the vector. Reran the free six-arm eval to check it
  (110s, $0.00): identical to the committed report on all 324 rows again. The duplicate report was
  not kept.
- 09-21: every budget now refuses **before** it spends. Until today a run's ceiling was checked
  after each call: `Budget.add` billed the call, then tripped, so a run could finish one call over
  its limit. `Budget.admit` checks the other way round: what a call could cost at most — its
  counted input plus its full output cap — is knowable before sending, so a call that could carry
  the run past its ceiling is never sent. Wired into every paid path: the tool loop and chat admit
  each call through `check_request(budget=...)`; `run_eval` admits a whole eval before its first
  row from the worst case the dry run already prints; the question generator does the same. The
  after-the-fact checks stay, as backstops for an estimate that runs low.
- 09-21: the price of the guarantee, stated rather than discovered later. A worst case assumes
  the whole output cap, so runs stop with budget left: a $1.00 tool-loop run now refuses its
  next call at about $0.90 spent, because a 4,096-token cap is ~$0.10 of Opus output. And an eval
  whose worst case exceeds `MAX_USD_PER_EVAL` is refused outright — the 54-case answer run
  ($2.22 worst case) would have been, which is what happened by hand on 09-19 when the ceiling
  was raised to $2.50 for it. Now the raise is required rather than remembered.
- 09-21: the failure catalogue gained `run-admission`; regenerated for free (the provoked requests
  are rejected before the model runs, and the budget rows never leave the machine), and every
  API-provoked row came back word for word.
- 09-21: `llm.py`, the module every model call is meant to go through, landed on its own and
  tested (15 tests, $0.00): count (free, schema included) → per-request ceiling → admit against
  the budget → send → bill and trace, in that order for every call shape the repo uses — plain,
  structured, streamed, tool-using and concurrent. A concurrent batch is admitted as a whole
  before any of it is sent. Nothing calls it yet; the primitives move onto it next, then the eval
  paths, and then a test makes any other call to the SDK fail.
- 09-21: the five demos in `primitives/` now send through `llm`: `first_call` counts and creates,
  `streaming` and `chat` stream, `structured_output` parses, and the tool loop creates one turn at
  a time against its run budget. None of them builds a client or pre-flights by hand any more.
  They are demos without tests, so each was run end to end against the fake client instead — all
  four ran, chat kept its session total across turns and honoured `/effort`. Only `failures`
  still calls the SDK directly in `primitives/`, and it retires with the eval paths' move.
- 09-21: every model call in the repo now goes through `llm`. The last four callers moved — the
  eval judge, the headline extractor (its task and its `measure` dry run), the answer eval and
  the question generator (tasks and dry-run counts). Eval tasks use `Llm.for_eval`: counted and
  checked per call, with no run ceiling of their own, so the runner stays the one budget an eval
  is billed to. The question generator, which had no running budget at all, gets a real one.
  `pricing.check_request` went with its last caller, and a test now scans the package and fails
  on any file but `llm.py` that calls `client.messages`.
- 09-21: `primitives/failures.py` retired, and its catalogue became tests: each failure mode is a
  row in `test_llm` asserting `llm` stops it before anything is spent. Probing the counting
  endpoint first (free) confirmed where the API-side ones land — a missing model is a 404, a wrong
  key a 401 and an empty request a 400, all on the free count, so none gets as far as a billable
  call. The probe also found a real hole: **`claude-opus-4-8`, a real model missing from
  `PRICES`, counted without error and would have been admitted and billed at $0.00**, so no
  ceiling could ever trip on it. The per-request check now refuses a model with no price.
- 09-22: the five workflow patterns are back, rebuilt on `llm` and the rankers rather than on the
  first draft's own call layer — chaining, routing, parallelization, orchestrator-workers and
  evaluator-optimizer, each under 100 lines, $0.00 so far. Routing keeps Haiku on the lookup
  route, to see the difference, and a workflow gets its own $0.25 ceiling. The patterns barely
  changed: `record` and `gather_text` / `gather_records` replaced the first draft's calls, and each
  request now states its own thinking setting. The tests came back with them, and the same three
  mutations as yesterday (skip the gate, swap the routing models, drop the plan cap) each failed.
- 09-22: fresh dry runs on the new code, token counting only (free). The first call of each:

  | pattern | first call | model | input tokens | worst case |
  |---|---|---|---|---|
  | chaining | extract | claude-opus-5 | 2,606 | $0.0322 |
  | routing | route (×3 questions) | claude-haiku-4-5 | 456-461 | $0.0011 |
  | parallelization | section (×3, concurrent) | claude-opus-5 | 2,046 | $0.0198 |
  | orchestrator-workers | plan | claude-opus-5 | 716 | $0.0164 |
  | evaluator-optimizer | draft | claude-opus-5 | 2,648 | $0.0228 |

  Identical to the first draft's, token for token, bar the plan's two: its schema description was
  shortened to keep the module under 100 lines. Later calls depend on earlier answers and cannot
  be counted in advance; the $0.25 ceiling, admitted before every call, is the bound on each run.
- 09-22: **the five patterns, live**, in two passes. The first ran chaining ($0.04244) and routing
  ($0.02449), then crashed inside parallelization: a reviewer's one-sentence verdict was cut off
  at its 128-token cap, and **the SDK validates structured output as it reads it, so the cut-off
  record raised before its usage came back — a paid call that never reached the budget or the
  trace.** Parallelization's trace went with it; its six calls had been admitted against the $0.25
  ceiling before sending, so that is its bound, and a second pass of the identical shape cost
  $0.175. Fixed in `llm`: an unreadable structured reply is now billed at its worst case, exact
  for a cut-off one, and returns as nothing parsed. Three caps went up on the run's evidence —
  facts 768 → 1024 (740 used), routed answer 384 → 768 (an analysis hit 384), vote 128 → 256.
- 09-22: the second pass, $0.31510 for four patterns, every call recorded:

  | pattern | calls | cost | what it did |
  |---|---|---|---|
  | chaining (first pass) | 2 | $0.04244 | 5 of 5 facts through the gate; the note says the passages hold no FY25 cigarette figures |
  | routing | 5 | $0.02442 | Haiku routes at ~$0.0007; a Haiku lookup $0.0028 against an Opus analysis $0.0195; the RBI question declined for nothing |
  | parallelization | 6 | $0.17513 | three sections, three reviewers all passing — the reviewers are 68% of the cost, each resending all 15 passages |
  | orchestrator-workers | 5 | $0.08923 | a plan of three, one per company; "Tata Motors is most exposed", on SCV volumes down 12.7%, with the reports' contradiction flagged |
  | evaluator-optimizer | 2 | $0.02631 | passed in round one, so the loop never revised |

  Total for the day about $0.55: $0.382 recorded, plus the first parallelization pass, bounded at
  $0.25 and about $0.17 by the second's shape.
- 09-22: what the runs say, beyond the costs. **Search, not the models, is the bottleneck**: three
  of five answers were honest "the passages do not cover it" — ITC's FY25 cigarette figures,
  Reliance's dividend (twice), Reliance's capex — each a retrieval miss on a question phrased the
  way a person asks, which is the paraphrased set's hit@5 of 0.48 showing up in use. The models
  behaved: they declined rather than invented, and the evaluator passed an honest decline, which
  its criteria allow. **Thinking off caused no tag leakage** in the 15 Opus replies whose text came back; one section ended
  with a stray "Only one company is covered here.", and the grounding reviewer named it. Two
  cheap next steps: the reviewers could share one system prompt so the 7.4k-token evidence
  becomes a cacheable prefix, and the evaluator needs a question hard enough to fail round one
  before it demonstrates anything.
- 09-22: **the SDK tool runner, read from source** (`anthropic` 1.4.0, `docs/tool-runner.md`). It
  keeps three of our four loop rules — the assistant turn echoed verbatim, every result in one
  user message, a tool error returned as a result — and drops the fourth: `max_iterations`
  defaults to no cap. Worse than missing is quiet: at the cap it simply stops, and `until_done()`
  returns a `tool_use` turn whose tools never ran. Nothing counts or budgets a turn, and its
  per-turn hook fires after the send. `llm.run_tools` drives it anyway, because the hook has one
  opening: `generate_tool_call_response()` runs the turn's tools and caches them, so the next
  request can be built, counted and admitted before the runner sends it. The cap is always
  passed and exhausting it raises. `agent.py` is the first agent on it: search as a tool beside
  the calculator, thinking on. Written and tested against the SDK's real `BetaToolRunner` over
  the fake client, $0.00.
- 09-22: agent dry run, free: `agent:1` on claude-opus-5 is 1,118 input tokens, up to $0.1080
  with the full 4,096-token output cap. Later turns carry every earlier result and cannot be
  counted in advance; each is admitted against the $1.00 run ceiling before it is sent.
- 09-22: **the agent, live: $0.62616 and no answer.** The rural-demand question ran seven Opus
  turns, each a search, and never answered. The eighth turn was refused before it was sent: 29,654
  input tokens plus the 4,096-token cap is $0.2507 at worst, over the $0.25 per-request ceiling.
  The demo stops on `BudgetExceeded`, so the Reliance dividend question never ran.

  | turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | input tokens | 1,118 | 5,510 | 11,247 | 16,376 | 21,723 | 26,875 | 29,464 |
  | cost | $0.0105 | $0.0352 | $0.0664 | $0.0926 | $0.1227 | $0.1485 | $0.1503 |

  **An agent's cost grows with the square of its turns**: each search adds about 5,000 tokens of
  passages, and every turn resends all of them, so turn 7 cost 14× turn 1 and the seven came to
  112k input tokens. At Opus prices the per-request ceiling also caps the context: with a
  4,096-token output cap, $0.25 leaves room for about 29,500 input tokens, so this agent has about
  six searches in it, whatever the turn cap says. The guard did its job — nothing unaffordable was
  sent — but the run lost its evidence: the searches the model chose, and any text it wrote, went
  with the exception, so the trace shows what the turns cost and not what they looked for.
- 09-22: **the agent's two fixes, free.** A stopped run now returns what it searched — each query,
  its scope and the passage ids it found — instead of losing them with the exception, and the demo
  goes on to the next question. The conversation is cached with top-level `cache_control`, so
  each turn reads the last one's prompt at a tenth of the input price instead of paying for it
  again. Priced from the first run's token counts, that would have cut its input bill by about
  60%. **It does not buy reach, and I said it would.** A call that may write the cache is admitted
  at the cache-write price (1.25×), because a miss writes the whole prefix, so the $0.25 per-request
  ceiling now binds at about 23,600 input tokens instead of 29,500: fewer turns than before, each
  cheaper. The first run ended at 29,464. Reach needs a decision about that ceiling, or smaller
  searches, not caching. Along the way: the SDK's `messages.parse` has no `cache_control`
  parameter though the API takes it, so `llm` sends it in the request body there; and the trace's
  `in` column now counts cached tokens as input, since the API's `input_tokens` is only the
  uncached remainder. One gap closed: `Price.cache_write` is the 5-minute cache's 1.25×, and the
  1-hour cache writes at 2×, so a request asking for it would have been under-priced by every
  guard and billed low. `llm` now refuses one outright; nothing uses it.
- 09-22: **the agent, live with caching: $0.25026, one answer and one stop.** Both questions ran
  this time, since a stop now returns instead of raising.

  | question | turns | searches | cost | outcome |
  |---|---|---|---|---|
  | rural-demand exposure | 5 | 10 | $0.19910 | stopped: turn 6 was 25,891 tokens, up to $0.2642 at the write price |
  | Reliance FY25 dividend | 2 | 2 | $0.05117 | ₹5.50 per share, ₹7,443 crore in all, citing `RELIANCE-FY25#0335` |

  **The dividend is the agent's point.** Every workflow run declined this question as "not in the
  passages". The agent's first search missed too. It then rephrased into the report's own words
  ("Directors recommend dividend equity share ₹ per share subject to approval members AGM") and
  found the answer on its second search. A fixed path cannot do that, so its first retrieval miss
  is its last.

  **Caching worked as priced.** Each turn read the previous turn's whole prompt: 1,116 of 5,596,
  then 5,594, 10,326 and 16,210. The rural question cost $0.199 for five turns, against $0.322 for
  the same tokens uncached, 38% less with output included. Its turns ran two searches in parallel,
  one or two reports at a time: rural demand per company, then segment shares of revenue to size the
  exposure, then monsoon risk, rural reach and cigarette volumes. That is a sensible plan, not a
  flailing one. The per-request ceiling cut it off as predicted: it sits at about 23,600 input
  tokens, and turn 6 would have been 25,891. Day's total now about $1.43.
- 09-22: **more room for the agent, and a last answer instead of a stop, free.** Turns of
  `run_tools` are held to a new per-turn ceiling, `MAX_USD_PER_TURN` = $0.40, in place of the
  per-request $0.25. At the cache-write price that admits about 47,600 input tokens a turn, twice
  the reach. And a run about to run out, of turns or of budget, now gets one last turn told to
  answer from what it has, with as much output as still fits. Below 1,024 tokens it stops as
  before. Two things found building it. **Taking the runner's history over runs the tools
  twice**: `append_messages` clears its cached tool results, and the runner calls for them again
  before checking whether it was taken over. A test counting tool calls caught it, so the answer
  turn is sent directly, with the runner's own tool dicts. **`tool_choice: none` would have been
  the obvious way to force an answer**, but changing it invalidates the messages cache, so that turn
  would re-write the whole conversation at 1.25×. The request stays identical and the text asks
  instead. Also closed: `test_pricing` still imported `BudgetExceeded` from `pricing`, missed by the
  errors move's single-line grep, and now imports it from `errors`.
- 09-22: **the agent, live with room and a last answer: $0.40019, both questions answered.**

  | question | turns | searches | cost | outcome |
  |---|---|---|---|---|
  | rural-demand exposure | 7 + the answer | 11 | $0.35115 | ITC by size of exposure, Tata Motors by sharpness, and a section on what the reports do not disclose |
  | Reliance FY25 dividend | 2 | 2 | $0.04903 | ₹5.50 again, citing `RELIANCE-FY25#0335` |

  The turn cap, not the $0.40 ceiling, ended the searching: turn 7 was 28,578 tokens, far under
  the ceiling's ~47,600, and the eighth turn was kept for the answer. **The cache held across
  the switch from the runner to the direct call**: the answer turn read 28,576 of its 28,952 input
  tokens from cache, wrote 2,077 tokens of answer, and cost $0.069. I checked the rural answer's
  figures against the passages it cited, which is free. All were there, in the reports' own
  format (`35893.57` cited as ₹35,894 cr). One citation was a passage off: the SCV unit counts
  are in `TATAMOTORS-FY25#0779`, which the same search returned, but were cited to `#0780`, which
  holds the 12.7% and the "muted rural demand" wording. The answer differs from
  orchestrator-workers' ("Tata Motors is most exposed"). With segment revenue in hand, it separates
  the size of the exposure (ITC, 71.5% of revenue in FMCG, plus an agri arm tied to farm incomes)
  from its sharpness (Tata's SCVs, down 12.7%, in a CV business that is about 17% of revenue).
  Day's total about $1.83.
- 09-22: **the turn cap becomes a backstop.** `MAX_AGENT_TURNS` goes from 8 to 25, for both tool
  loops. Every turn is admitted against the budget before it is sent, so the budget should decide
  how long a run lasts, and at 8 it was the cap that ended the run above, with the per-turn ceiling
  far off. At the ~4–5k tokens a search adds, `MAX_USD_PER_TURN` now ends a search near turn 11. The
  cap binds only on a loop of turns too cheap for the budget to stop soon. Free; no run.
- 09-22: **server-side web search, wired and priced, free so far.** The docs give the terms: $10
  per 1,000 searches on top of tokens, a failed search free, `max_uses` capping searches per
  request. The trap is that "web search results... are counted as input tokens, in search
  iterations executed during a single turn": the API adds them mid-call, so the free count before
  sending cannot see them. So a search is priced at its cap, each with a 10,000-token allowance
  for results, and that allowance is the one estimate in any pre-send worst case. `llm` refuses a
  search with no `max_uses`, and any server tool pricing does not model, as it refuses an unpriced
  model and the 1-hour cache. The agent offers it as a third tool. `tools.ALL_TOOLS` does not, so
  `primitives.tool_loop` stays as it was: its first search would have ended its run under the
  $0.25 per-request ceiling. One test gap was found by mutation: nothing checked that the last
  answer sends the tools in the runner's order, which the cache depends on. **Pending: a paid probe
  to measure what a filtered search really adds**, and set the allowance from it.
- 09-22: **the web search probe: $0.30450, answered, and two problems found.** One question,
  "How has Reliance's share price moved since it announced its FY25 results?", with dynamic
  filtering and at most two searches a turn.

  | turn | in | cached | out | web | cost |
  |---|---|---|---|---|---|
  | 1 | 6,861 | 0 | 305 | 0 | $0.05050 |
  | 2 | 46,343 | 40,578 | 706 | 1 | $0.08396 |
  | 3 | 44,914 | 40,564 | 1,292 | 2 | $0.09976 |
  | 4 | 20,016 | 16,960 | 117 | 0 | $0.03050 |
  | 5 | 20,200 | 20,014 | 1,144 | 0 | $0.03977 |

  It searched the report first for the results date and any price table, then the web three
  times. It answered about −4% from roughly ₹1,300 before the results to about ₹1,248 now, via a
  high of ₹1,611.80. Every report-side claim checked out against its passage. **No web source is
  listed**: `_with_sources` reads citations from the final message only, the answer came two
  turns after the searches, and its text carried none. The unit test passed because it put the
  citations in the final message, which the live run did not. So the web figures cannot be traced.
  **The trace's `in` is not the context on a search turn**: turn 2 counted 46,343 when the
  conversation was about 9,000, because the server-side loop rereads the context at every step
  within the one request, and usage sums them. Turn 4, with no search, shows the real size. The
  rereads are cached within the turn (40,578 read), so turn 2 cost $0.084 against the ~$0.29 it
  was admitted at. The 10,000-token allowance held with about 3× margin. But it prices results,
  when what grows is rereads times context, so it will not scale with a long run. What persisted
  was small: about 13,000 tokens across three turns, report passages and output included, so a
  filtered search leaves perhaps 2,000. Turn 1's $0.050 is mostly caching the 6,861-token prompt,
  5,700 of it the dynamic-filtering tool definition.
- 09-22: **sources fixed, free.** `run_tools` now returns every turn it took, and the agent reads
  every turn for web pages: those cited, then others the searches returned. If searches ran and no
  links came back, it says so. The tests are now shaped like the probe, with the answer a turn
  after the search. Whether links survive `response_inclusion: excluded` with dynamic filtering is
  still unknown; the next live run will show it rather than hide it.
- 09-22: **the web rerun: $0.26387, and the answer to where the links go — nowhere.** Same
  question, sources now read from every turn. **With dynamic filtering and
  `response_inclusion: excluded`, no page link reaches us**: no citations, no result blocks,
  across all five turns. The agent now says so ("2 web searches ran, and no page links came back
  with them") instead of printing nothing. Its figures agree with the first probe: ₹1,248 on
  21 September 2026, a 52-week range of ₹1,232.5–1,611.8. It benchmarked against the report's
  year-end market capitalisation, ₹17,25,378 crore, which gives an implied ₹1,275 a share, so a
  move of about −2%. Every report figure is in a passage it retrieved (`#0023`, `#0586`), though
  this answer did not tag them with ids. It said "my search budget was exhausted after one
  query": at `max_uses` 2, one filtering step can spend both searches. The searching turn read
  81,600 tokens, 78,921 of them cached: the server loop's rereads again, $0.122 for the turn.
  Day's total about $2.39.
- 09-22: **web search, two variants head to head: $0.62532 for both.** Same question, `max_uses`
  5, the per-turn ceiling raised to $0.75 for these two runs only (`ENTROPIC_MAX_USD_PER_TURN`,
  no code change). At 5 searches the 10,000-token allowance reserves $0.36 a turn, which the
  $0.40 ceiling cannot hold. Each variant was run from a scratch script that swapped only the
  web search tool dict.

  | variant | turns | searches | cost | searching turn | pages cited | pages listed |
  |---|---|---|---|---|---|---|
  | direct | 3 | 2 | $0.25025 | 38,744 in, 18,294 cached, $0.171 | 6 | 13 more |
  | dynamic filtering, `response_inclusion: full` | 5 | 7 | $0.37507 | 86,007 in, 73,509 cached, $0.236 | 0 | 51 |

  **Direct wins on every count that matters here.** It cost a third less. Its text carries
  citations, so six claims point at the pages behind them. Dynamic filtering with full inclusion
  returns every page it saw, 51 of them, and cites none. **Dynamic filtering also ran past its
  cap**: usage reported 7 searches in one turn against a `max_uses` of 5. From here I cannot tell
  whether the extra two were errors or were billed; the ledger bills them. A cap the worst case
  depends on has to hold, and with direct search the docs say an extra search is an error, not a
  charge. **The allowance measures right for direct search**: the searching turn wrote about
  20,000 new tokens for two searches, and the next turn's context had grown by about 9,000 a
  search. Both answers agree with each other and with the earlier probes: about −4% from the
  results-day close (₹1,300.05) to about ₹1,245 now, by way of ₹1,611.80. Every report figure
  checked out (`#0250`, `#0625`). Day's total about $3.02.
- 09-22: **web search settled: direct, a cap of 3, a $0.60 per-turn ceiling.** Chosen over a cap
  of 5 at $0.75, because it reserves less per turn for the same room (about 44,800 input tokens
  either way), and the direct run used 2. `response_inclusion` is gone; it only governs filtered
  results. The tests that probed the per-turn ceiling with fixed token counts now derive them
  from the ceiling, so the next change to it will not break them. Free; no run.
- 09-22: **context management written up** (`docs/context-management.md`), free. By the token
  count: the system prompt is 185 tokens, our tool schemas about 1,000 with the API's preamble,
  direct web search 2,726, and a search result of five passages 2,000–2,700. The decision is to
  not summarize yet. The budget ends every run below 80,000 tokens, a summary would drop the ids
  the answer cites, and compaction resets the cache.
- 09-22: **LangGraph, quickstart and source read** (`docs/langgraph.md`), free. `langgraph` 1.2.12
  went into a non-default `graph` group. Five graphs with no model calls measured what the source
  says. The default step limit is **10,007**, not the 25 older material gives, and hitting it
  raises. On resume, **an interrupted node reruns from its first line**, so a paid call before
  `interrupt()` is paid twice. **A step's tasks all run at once**, 6 of 6, which bypasses `llm`'s
  batch admission: each worker is admitted alone while its siblings are in flight. And with a
  checkpointer, a resume after one of three parallel workers failed reran only that one.
- 09-22: **calls in flight now count, free.** `Budget.admit` counted only what was spent, so
  parallel callers could each fit a ceiling alone and cross it together. The batch methods never
  could, since they admit a batch whole, but LangGraph's parallel nodes would. Every call `llm`
  admits is now held until it is billed, or released if it fails. A test makes a second call from
  inside the first's reply, and it is refused where the two would cross the ceiling together.
  Four mutations are each caught.
- 09-22: **orchestrator-workers rebuilt as a LangGraph graph, free.** The same eight tests pass
  against both builds. The orchestration is 67 lines against 24. The graph needed a reducer, an
  input schema, a destination list on its conditional edge and a step limit. Without the
  destination list, the drawn diagram ended the run at `plan`. A failing worker surfaces the same
  error with the same calls billed in both, through 19 frames against 13. The graph admits
  workers one at a time, not as a batch. Verdict: not worth it for a three-step workflow; its
  checkpoints and interrupts are for Week 4's agent. A live comparison run is not done.

- 09-23: **Track B, Week 3: what quantization costs, predicted first and then measured.**
  `underhood`'s `inference/quantization.py` takes Llama-3.2-3B-Instruct in bf16 and converts it
  locally to 8, 6 and 4 bits, so the rows differ in bit width and nothing else. Free; it runs on
  this machine. Predictions were written before the run and are kept beside the results.

  | variant | weights | peak | TTFT | decode | weights read |
  |---|---|---|---|---|---|
  | bf16 | 6.43 GB | 6.56 GB | 375 ms | 14.7 tok/s | 94.2 GB/s |
  | q8 | 3.41 GB | 3.67 GB | 286 ms | 28.2 tok/s | 96.2 GB/s |
  | q6 | 2.61 GB | 2.91 GB | 280 ms | 36.3 tok/s | 94.7 GB/s |
  | q4 | 1.81 GB | 2.17 GB | 276 ms | 49.8 tok/s | 90.1 GB/s |

  **The decode roofline was the right instrument.** Predicted rates were weight bytes ÷ 120 GB/s
  exactly — 18.9, 35.7, 45.5, 66.7 tok/s — and every variant landed at **74–80% of its own
  roofline**, so one efficiency figure explains all four rows. The q4 : bf16 ratio came in at
  **3.39×**, below the 3.5× the weight sizes alone imply, because fixed per-token overhead weighs
  heaviest on the shortest token. That part of the prediction was right; the claim that efficiency
  falls monotonically with bits was half right — q4 is the least efficient at 75.1%, but bf16 does
  not lead, and bf16, q8 and q6 sit in a flat 78.5–80.2% band.

  **Prefill has a crossover between 128 and 512 tokens**, which was not predicted. At a 128-token
  prompt every quantized variant beats bf16 (354–393 ms against 402); at 512 the order has fully
  inverted; at 2048 bf16 wins by 25% over q4 (4156 ms against 5211), with q8 and q6 laddered
  between. Short prefill has too little arithmetic per weight to be compute-bound, so it still pays
  for bytes the way decode does. Long prefill is compute-bound and charges for dequantization.

  **mlx is faster than torch on this silicon.** bf16's 2048-token TTFT of 4156 ms against 14.6
  TFLOP of work implies **3.5 TFLOP/s**, where Week 2's torch matmul on MPS gave 2.75. The estimate
  built on the torch number was 13% high. Week 2's constant is a torch number, not a hardware one.

  **Quality, teacher-forced on bf16's own generations:** q8 KL 0.0007 / 98.5% top-1 / 6 of 10
  answers byte-identical; q6 0.0040 / 97.8% / 3 of 10; q4 0.0477 / 93.9% / 2 of 10. As perplexity
  that is +0.07%, +0.40% and +4.9%. The same harness on a 135M model gave 4–5× worse numbers at
  every bit width, which is redundancy absorbing the error. **The per-prompt spread is 24× and
  tracks how constrained the output is, not how hard the task is**: `extraction` 0.0036 and
  `format` 0.0062 at the robust end, both at 100% top-1 with byte-identical JSON from all four
  variants, against `factual` 0.0851 and `summary` 0.0685 at the other. Open prose has many
  acceptable next tokens, so rounding flips near-ties; `"company": "Northwind` has one, by a margin
  4-bit rounding cannot overturn. That is the good news for pointing a RAG pipeline at a local
  model, since its output is the constrained kind.

  **The caveat outranks the table: this measures fidelity to bf16, not correctness.** On the
  `logic` prompt all four variants answer wrongly, bf16 included — the chain is Asha > Ben >
  Chitra > Dev, so the second shortest is Chitra, and they said Dev, Ben, Ben, Dev, each
  contradicting its own restated premises. A KL of 0.000 there would have meant being perfectly
  wrong. Four of the ten prompts have checkable answers and none of them are graded; the Week 1
  harness is what would.

  **Two upstream findings in mlx-lm 0.31.3, both candidates for Week 7's PR.** `mlx_lm.convert`
  from a hub repo id fails on huggingface-hub 1.32.0, released 09-17: `load` fetches only weights
  and tokenizer, then `save()` calls `snapshot_download(local_files_only=True)` and the new hub
  raises `IncompleteSnapshotError` over a missing README. The stock CLI reproduces it; the
  workaround is to hand `convert` a resolved local path. And the vocab projection runs at **every**
  prefill position and is then discarded — 12% of prefill FLOPs and a 0.53 GB transient at 2048
  tokens. `transformers` has `logits_to_keep` for exactly this; the fix has to be a parameter
  rather than a slice, because speculative decoding needs logits at more than the last position.

  **Still open:** peak memory above the weights grows as the model shrinks (0.13 GB for bf16 to
  0.36 GB for q4) and nothing explains it yet. And part 2 of the roadmap item — `mlx_lm.server`
  behind Project 1 — is not started; it needs `llm.py` to speak to a local model, and the answer
  eval's judge is paid.
- 09-23: **where the memory above the weights goes, chased and half-settled.** Free, local. Two
  throwaway probes against the same four variants. First: peak memory is set **entirely during
  prefill** — generating 32 more tokens moves it by nothing, in all twelve rows — and it scales
  with prompt length at roughly 500–750 KB per prompt token. For bf16 that is close to the sum of
  the two known terms: the vocab-projection logits at 128,256 × 2 bytes = **256 KB per token**,
  which are computed at every prefill position and then discarded, plus the KV cache at 115 KB per
  token. So `logits_to_keep` is not only 12% of prefill FLOPs, it is the largest single term in
  prefill memory too. Second: quantized variants pay a **flat ~65 MB surcharge** visible already at
  a 16-token prompt (+36 MB for bf16 against +101 to +105 MB for q8, q6 and q4). The hypothesis
  that mlx materializes a dequantized weight matrix for prefill-shaped matmuls is **refuted**: one
  3072×8192 layer at batch 512 allocates identically in bf16 and 4-bit, to the byte. What remains
  is a whole-model or allocator effect — a quantized layer holds three arrays where a dense one
  holds one — and it is being left there: 65 MB flat against savings of 2.4 to 4.4 GB changes no
  decision. It also corrects yesterday's reading that the surcharge grows as the bits fall; the
  q8/q6/q4 spread at longer prompts is not consistent enough to be a trend. One instrument note:
  `mx.get_peak_memory()` returns absolute peak active memory, not a delta since
  `reset_peak_memory()`, and intermediates released into the cache mid-call mean active can rise by
  less than the output array's size.
- 09-23: **Track B part 2: Project 1 answered by a model on this laptop, and what it cost to find
  out.** The `llm` layer is now provider-neutral: `messages.py` holds the types a request and a
  reply are made of, `adapters/` holds one class per wire behind `Adapter`, `Streamed` and
  `ToolSession` protocols, and a **client** — a named endpoint in its own `cfg_<name>.py` — maps to
  exactly one of them. `llm` imports no SDK; invariant 8 now reads that only an adapter talks to a
  model, and a test asserts the set of files that do is exactly `adapters/anthropic.py`. Six
  clients: `anthropic`, and five served by one local `mlx_lm.server`.

  **Five served models, one at a time** (the server keeps one resident), four questions each
  through the real call path:

  | client | median s | tok/s end to end | parsed | declined the unanswerable |
  |---|---|---|---|---|
  | Qwen3-4B 4-bit | 1.59 | 29.6 | 4/4 | yes |
  | Qwen3-4B 6-bit | 2.04 | 23.8 | 4/4 | yes |
  | Qwen3-4B 8-bit | 2.43 | 20.4 | 4/4 | yes |
  | Qwen3-4B bf16 | 4.48 | 10.9 | 4/4 | yes |
  | Qwen2.5-7B 4-bit | 1.87 | 20.3 | 4/4 | yes |

  4-bit is **2.8x faster than bf16 end to end** for outputs indistinguishable on this probe, which
  is the underhood quantization result holding on the serving path. The 7B at 4-bit is *faster*
  than the 4B at 8-bit for about the same memory, so the bigger-but-quantized trade costs no
  latency here. The 7B's one miss is the interesting one: on a question its own cited passage
  answers, it set `answered=false` and returned nothing — **a false refusal, not a citation
  failure**, and the opposite of the error a small model is supposed to make.

  **The 54 questions, answered by Qwen3-4B 4-bit and judged by `claude-sonnet-5`** against the same
  rubric and labels as the 09-19 Opus run. $0.2378 of a $0.4368 worst case, 11.6 minutes:

  | | Opus 5 (09-19) | Qwen3-4B 4-bit |
  |---|---|---|
  | RAG `correct` | 42/54 | **37/54** |
  | RAG `answered` | 43/54 | 40/54 |
  | RAG `cites_relevant` | 37/54 | 36/54 |
  | precision when it answers | 42/43 | 37/40 |
  | closed-book `correct` | 5/54 | **0/54** |
  | cost of answering | $1.342 | **$0.0053** of machine time |

  **What got worse is five right answers out of 54**, and what did not is citation fidelity: 36 of
  a 37 ceiling, since `cites_relevant` is bounded by the retriever's hit@5 and both models were
  handed identical passages. The failures are ordinary rather than structural — `rq-005` answers
  "10 countries" where the reference says 41, and contradicts itself by listing more than ten.
  Closed-book inverts: Opus answers 10 of 54 from memory and gets 5 right, the 4B answers **none**,
  which is not caution but ignorance, and is only useful as the floor that makes the RAG number
  mean something. Answering is **250x cheaper**; the judging is what cost real money, and it costs
  the same whoever answers.

  **Two bugs a live call found that 823 green tests did not.** `Request.model` defaulted to
  `config.MODEL` — the *configured* client's model — so `Llm("local")` asked `mlx_lm.server` for
  `claude-opus-5`; the same leak sat in `answer_task` and `LlmJudge`, which pinned their models at
  import. All three now resolve from the client they are given, which is what lets one run answer
  on a local model and judge on a hosted one. And the OpenAI wire's token count read the length of
  a transformers `BatchEncoding`: **2 for every prompt**, which would have waved anything past the
  per-request ceiling. Both are regression-tested, the count one with a stub that answers in the
  shape transformers actually does.
- 09-24: **the rest of that night's runs, which this log missed until 09-28.** Nine more after
  the one above, each answered locally and judged by `claude-sonnet-5`, for **$2.05475** by the
  reports they wrote. The first four sent the record's JSON Schema; the last five describe its keys:

  | report (UTC) | answered by | record asked for as | RAG `correct` | errors | spent |
  |---|---|---|---|---|---|
  | 09-23 16:49 | Qwen3-4B 8-bit | schema | 38/54 | 0 | $0.23804 |
  | 09-23 17:04 | Qwen3-4B bf16 | schema | 39/54 | 0 | $0.24148 |
  | 09-23 17:23 | Qwen2.5-7B 4-bit | schema | 34/38 | 16 | $0.20401 |
  | 09-23 18:14 | Qwen3-8B 4-bit | schema | 25/27 | 27 | $0.18098 |
  | 09-23 19:03 | Qwen3-4B 4-bit | keys | 37/54 | 0 | $0.23876 |
  | 09-23 19:55 | Qwen3-4B 6-bit | keys | 37/54 | 0 | $0.23686 |
  | 09-23 20:08 | Qwen3-4B 8-bit | keys | 35/53 | 1 | $0.23783 |
  | 09-24 00:57 | Qwen3-4B bf16 | keys | 36/53 | 1 | $0.24017 |
  | 09-24 01:17 | **Qwen3-8B 4-bit** | keys | **40/54** | 0 | $0.23662 |

  **An 8B at 4-bit answers two behind Opus 5's 42/54**, for $0.005 of machine time against $1.342.
  **The quantization ladder did not survive the prompt change**: 37/38/39 for 4-bit/8-bit/bf16
  under the schema, 37/37/35/36 for 4/6/8-bit/bf16 under the keys. The prompt moved the 4B more than
  its precision did, so quantization's effect is below what 54 questions resolve, and the first
  ladder was over-read. **A model handed a JSON Schema answers with the schema**: Qwen3-8B parsed
  0 of 12 replies that way and 12 of 12 with the keys described, and the 18:14 run lost 27 rows to
  it. Qwen2.5-7B lost 16 to prose and was dropped. The two errors under the keys are both `rq-009`,
  a reply that ended on `stop` and did not parse.

  **One run is missing from the table.** A first bf16 attempt under the keys stalled at 21:27 UTC
  on a dead pooled socket to the API and sat there until it was killed at 00:48: the SDK's
  600-second read and minutes of keepalive, now 180s and 30s for the API (`dd27b4f`). Its buffered
  output never reached the log, so whatever it spent on verdicts before the stall is billed and
  unrecorded here; the Console's usage for that window is the only record. Every run on this
  night came from a script in `scratch/`, which is what 09-28 moved into the repo.

## Week 4 (2026-09-28 to 2026-10-04)

- 09-28: **the local comparison moved into the repo.** `retrieval.answer` takes `--client` and
  `--judge-client`, so answering on this laptop and judging on the API is
  `python -m entropic.retrieval.answer --client local-8b-4bit --judge-client anthropic --cache`
  rather than a script in `scratch/`. `--cache`, which already reused PDF text, now reuses chunk
  vectors too, there and in `retrieval.evaluate` and the workflow demos: `DenseIndex.build_cached`
  keeps them in `corpus/.cache/`. $0.00: dry runs only, whose one call to the API is the free
  count. The index is **48.3s cold and 0.01s warm**, bit-identical to a fresh build with the same
  top 5 on all 54 questions, and a dry run takes 23s where it took 76s.

  **Two things the scratch harness got wrong, unnoticed.** It keyed its cache on chunk ids, which
  are positional, so any change that kept them would have served the old vectors; the key now
  hashes every chunk's text, the embedder, and the code that embeds a passage (invariant 25). And
  every local run answered under Anthropic's 256-token cap rather than the local client's 512,
  because `MAX_TOKENS_ANSWER` is resolved at import for the configured client. Harmless on the
  night — all 45 failed rows ended on `stop`, none at the cap — but `cfg_local`'s caps were dead
  for any client picked at run time. A call now takes the cap of the client it goes to (invariant
  24). Its reports also recorded no dataset digest; the repo's path records one.

  **`main` was red for four days.** CI's pyright step failed on `dd27b4f` over one line of a test
  that read a connection pool through a transport typed as its base class. The push went out as
  the session ended, and the failure sat unread. Fixed with a `cast` to the concrete transport.
- 09-28: **the repo is `aigent` now, and lives in `projects/Entropic/` beside `underhood`.** The
  package, the CLI, every import and the `ENTROPIC_*` settings became `aigent` and `AIGENT_*`;
  GitHub redirects the old URL. Entropic stays the name of the plan the two repos carry out. The
  entries above keep the commands as they were run, `entropic.*` included.
- 09-28: **the toy fine-tune: Qwen3-1.7B from 11/50 to 41/50 on the real headlines, trained on
  templates alone.** $0.00, all local. 180 template rows from `aigent.extraction.synthetic`; LoRA at
  rank 8 and scale 2 on the last 16 layers, 200 steps of batch 4, loss on the answer only
  (`underhood-toy-lora`). Graded by Project 1a's own graders over the 50 hand-labelled headlines,
  with the prompt the rows were rendered from (`headlines-20260928-0926.md` and `-0928.md`):

  | Qwen3-1.7B 4-bit | whole record | `change_pct` | `quarter` | `metric` |
  |---|---|---|---|---|
  | base, zero-shot | 11/50 | 19/50 | 40/50 | 37/50 |
  | **tuned, zero-shot** | **41/50** | 48/50 | 49/50 | 44/50 |
  | base, few-shot | 16/50 | 27/50 | 44/50 | 37/50 |
  | tuned, few-shot | 27/50 | 36/50 | 48/50 | 43/50 |

  Opus 5 reads 48 or 49 of 50 on this set (09-12). Every reply parsed in all four arms: the base
  model's failures were content, mostly a signed or zero `change_pct` where the label is null.

  **What the templates taught transferred; what they left out did not.** Per tag, zero-shot, base
  to tuned: no stated percentage 1 → 12 of 16, month-named quarters 0 → 4, aliases 1 → 5, no
  period 0 → 4. Never templated: subsidiaries stayed 0 of 2, narrowed metrics 0 of 2, and
  off-vocabulary metrics went only 0 → 2 of 7. Decoy numbers, not a template of their own, went
  0 → 7 of 10: format discipline carries over. **One regression is overfitting in miniature**:
  every no-metric template had an unknown quarter, and `hl-042` ("… Q2 results next week") lost
  its Q2. **Tuning and prompting did not add up**: the tuned model does worse on the few-shot prompt
  (27) than on the one it was trained with (41), since a second prompt is off its distribution.

  **200 steps were about 170 too many.** Train loss 0.66 at step 10 and 0.03 at step 40; validation
  1.81 at the start, 0.020 at step 50, 0.000 at step 200. The run took about half an hour at
  10.2 GB peak, slowing from 0.135 to 0.086 steps a second as the fanless machine heated: each row
  carries a ~570-token schema instruction, of which the loss sees the ~45-token answer. One caveat
  to state with the number: the templates' author had read the test set. The four-word guard stops
  copying, not knowing — the same caveat the paraphrased retrieval set carries.
- 09-28: **Project 2's first live run: 5 of 25 tasks, all passed, $0.14999.** The LangGraph agent on
  `claude-sonnet-5`, judged by `claude-opus-5` (a model never grades its own answers), `--sample 5`
  against a $2.58 worst case (`tasks-20260928-1205.md`). Every task passed all four graders — right
  tools, sane order, finished, correct — at $0.025 to $0.036 a task, $0.030 on average. The sample
  was pt-001, 007, 013, 019 and 025: search-then-calculate, the clock and a sandbox file. It holds
  no web task, no two-report comparison and no decline, so it says the loop works and nothing yet
  about the hard families.
- 09-28: **Project 2, all 25 tasks on `claude-sonnet-5`, judged by `claude-opus-5`: 16 passed, $0.69082
  recorded** (`tasks-20260928-1218.md`), $0.028 a task. A first attempt was refused before any
  spending: judged by Opus the worst case is $12.91, not the $12.66 the default judge prices, and
  the ceiling had been set at $12.70. **The nine misses have three causes, and two are ours.**
  - **Four errors were a bug in the LangGraph agent** (pt-015 to 018). A turn that ran a web
    search came back with `server_tool_use.caller: null`, and echoing the block verbatim sent the
    null back, which the API refuses with a 400. The SDK's runner never hit it; ours dumped every
    field. `reply_of` now drops unset fields, for every path that echoes a turn. **Those four rows'
    first turns were billed and are missing from the $0.69**: the error discarded the task's trace.
    By the other rows' costs that is likely under $0.15. A failing task now reports what it spent.
  - **Two were the judge, not the agent** (pt-020, 023): Opus reasoned past the 256-token verdict
    cap and returned nothing. `JUDGE` is 512 now.
  - **Three were the agent** (pt-003, 008, 014), right answers by the wrong route: 45 − 12 and a
    date gap done in its head rather than with `calculate`, and today's date assumed rather than
    read from `current_time`. The system prompt asks for the calculator; the labels hold it to that.
