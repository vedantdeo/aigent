# The facts database: every reported number, looked up rather than read

*Design, 2026-10-22, before any code. Proposed after the 10-22 teacher re-run showed that most figure
errors happen after retrieval (a right passage, the wrong row, basis or scale), at Vedant's call to
pivot from training a reading specialist to a database the generalist queries.*

## The idea

Today a figure reaches the generalist by search: `search_filings` returns passages, and the
generalist (or the `extract_figures` specialist) reads the number out of a table. Every step of that
can go wrong on a table that is right: a neighbouring row, standalone for consolidated, a scale in a
heading three lines up. The 10-22 re-run measured it: in 28 of 33 dropped trajectories the passages
read held the labelled figure.

So the numbers are read **once, checked, and stored as facts**. The generalist asks for "Wipro,
FY25, consolidated, revenue from operations" and gets the value in rupees with its source. Search
stays for prose and for what the database does not hold. The database starts from the companies'
XBRL filings and keeps growing as figures are read from the PDFs (see Write-back).

## Sources

1. **NSE's XBRL results filings: the source of truth (Vedant).** Listed companies file their
   quarterly and annual results in XBRL under SEBI's listing regulations; the company prepares the
   file, NSE hosts it, and a results PDF is filed beside it with the same numbers. Measured
   2026-10-22:
   - Every company in both corpora has consolidated and standalone files for FY24, FY25 and FY26:
     quarters to Dec 2024 from `api/corporates-financial-results?index=equities&symbol=X&period=Quarterly`
     (fields `toDate`, `consolidated` as `Consolidated`/`Non-Consolidated`, `xbrl`), Mar 2025 on
     from `api/integrated-filing-results?index=equities&symbol=X&size=100&page=1` (fields
     `qe_Date`, `consolidated` as `Consolidated`/`Standalone`, null on governance rows; it shows 20
     rows unless `size` is given, `page` counts from 1). Tata Motors files as `TMPV` (and `TMCV`
     from FY26) since its demerger; `TATAMOTORS` returns nothing. The JSON API answered with a
     browser user agent and no cookie; the home page itself returned 403.
   - About 330 numeric facts per file over about 200 Ind AS concepts: profit and loss, balance
     sheet, cash flow, other comprehensive income, EPS, segment revenue, results, assets and
     liabilities. Banks file under a banking taxonomy.
   - Checked against `evals/reference/fundamentals.json`: 15 of 15 figures identical (revenue,
     profit attributable to owners, basic EPS; ITC and Reliance FY24 and FY25, Tata Motors FY25).
   - **How much of the annual report it covers** (ITC, Reliance, Tata Motors FY25): the files'
     year-end figures are 80-85% printed in the annual report under a strict match (a few are
     printed but missed by the match, such as EPS); its quarter columns mostly are not. Of the
     2,431 table chunks in those three reports only 13 are fully covered by XBRL facts and 37 at
     least 80%; the 126 main-statement chunks average 42%, because the results format holds the
     statements' main lines, not every line the report prints. 12 of the retrieval eval's 88
     questions are answerable from XBRL.
   - **What it adds over the annual report**: quarterly figures; subtotals and lines the report
     does not print as such (current financial assets, an "other expenses" total, SEBI's coverage
     ratios); and above all **certainty about what each number is**: a standard concept, explicit
     period, unit and basis, in rupees, with nothing to parse. It is not earlier than a PDF of the
     same results, which is filed with it.
   - Not in it: the notes, segment detail beyond the results format, prior-year comparatives (they
     are in the prior year's file), and possibly dividends per share (unchecked).
2. **Our parsed PDF tables: the bulk of the database.** About 95% of the annual reports' table
   chunks are notes, segment detail, ratios and governance tables XBRL never touches. Docling's cell
   grid per table, with `table_label`'s section path, basis and multiplier and `years_in`'s column
   years, already computed for retrieval: one fact per numeric cell, its row label as printed.
3. **Figures read from PDF passages** by the generalist or a specialist, written back once
   approved (see Write-back).

## Data model

One SQLite file, `corpus/.cache/facts.sqlite`, shaped after XBRL's fact model (a value, its concept,
entity, period, unit and dimensions), plus where it came from and how far it is trusted. WAL mode,
since runs write to it while others read.

```text
filings   id, ticker, fy, basis, source, origin (URL or doc_id), sha256, fetched
facts     id, filing_id, ticker, basis, concept, label, period_kind ('duration' | 'instant'),
          period_start, period_end, segment, value, unit, decimals, printed,
          tier, page, chunk_id, row_index, passage, formula,
          written_by, written_at, run_id, verified_by, verified_at
derived_from  fact_id, input_fact_id
lookups   id, run_id, asked_by, asked_at, ticker, basis, period, what, concept, found
concepts  concept, label, statement, kind ('xbrl' | 'row' | 'read'), xbrl_concept,
          mapping_checked, vector_key
```

- **concept**: the XBRL element (`RevenueFromOperations`) for XBRL facts; for PDF facts, the row
  label normalised (case, whitespace, footnote marks), so the same row in two years shares one; for
  a read fact, the model's metric and description.
- **label**: the words a human reads: the taxonomy's label for an XBRL concept, the printed row
  label for a PDF fact. **Built 10-23** (Vedant: do it right first, over measuring split element
  names): NSE's four taxonomy packages for the versions the filings use (SEBI's 2025 Ind AS and
  banking, BSE's 2020 Ind AS and 2019 banking) label all 281 concepts; one label per concept, the
  first package in that order winning, every other label an alias in `concept_labels`. Bank and
  non-bank packages label concepts by the same names, so they make one set. SEBI's 2026-01-31
  taxonomy (16 of the 192 filings) is not published as a package, and the 2025 one labels every
  concept those filings use. Element names that are misspelt are labelled right ("Itmes").
- **period**: the context id says which column (`One…` the quarter, `Four…` the year to date, `…I`
  an instant, `PY_I` the prior year-end), and each filing states each column's period in words
  (`DateOfStart/EndOfReportingPeriod` on `OneD` and `FourD`; a June quarter has only `One`). In the
  2020 BSE taxonomy (to Dec 2024) a `Four…` context carries the quarter's dates, in SEBI's 2025
  taxonomy the right ones, so a column's period is the stated one, and a context with no column
  (the integrated `D_…`) is read from its own dates only when every stated column's dates hold.
  The 2020 taxonomy files cash-flow cash twice in one duration, opening then closing; they are
  stored as balances at the day before the period and at its end. Quarterly and annual facts are
  both stored (Vedant, 10-22). A PDF fact takes its column's year and the statement's kind (balance
  sheet: instant).
- **value** in base units (rupees, rupees per share, a ratio as a fraction), `decimals` as filed,
  `printed` as the report shows it for PDF facts.
- **segment**: XBRL's segment members are numbered (`…Revenue01Member`); their names come from the
  file's text facts. A PDF fact's segment is its row or column header when the table is a segment
  table.
- **XBRL is the source of truth** (Vedant): a concept and period an XBRL file holds is not loaded
  from the PDF, and the two are never compared.
- **tier** (Vedant, 10-22) says where a fact came from and whether a person has checked it, and
  is returned with every fact:

  | tier | meaning |
  |---|---|
  | `xbrl` | from the NSE XBRL filing; the source of truth, no check needed |
  | `pdf-cell-ai` | a parsed PDF table cell, not yet checked by a person |
  | `pdf-cell` | a `pdf-cell-ai` fact a person has checked |
  | `pdf-read-ai` | a figure a model read from a PDF passage in a run, not yet checked |
  | `pdf-read` | a `pdf-read-ai` fact a person has checked |
  | `derived-ai` | computed from other facts by a formula, not yet checked |
  | `derived` | a `derived-ai` fact a person has checked |

  A cell is more reliable than a model's read (the grid and the totals check stand behind it), so
  the two keep separate tiers. Only a person moves a fact out of an `-ai` tier (`verified_by`,
  `verified_at`).
- **written_by / written_at** (Vedant, 10-22): who wrote the row and when. For a model's write,
  `written_by` fingerprints the model and agent, `model|agent@hash(prompt + tool schemas)`, so a
  changed prompt is a new writer; for a build row it names the loader and its git commit.
  `run_id` stays beside it, so a bad run's rows go in one delete.
- **ticker and basis** are columns of the fact, not only of its filing (Vedant, 10-22): every lookup
  filters on them, and a model's read need not come from a tracked filing.
- **Duplicates are the agreement record** (Vedant, 10-22): `-ai` rows are never merged. `run_id`
  identifies a session: the same fact written by the same agent in two runs is two rows, written
  twice in one run is one. A job polls these rows and prefers a value per fact (ticker, basis,
  concept, segment, period); its voting rule is decided later. The vote stays a proxy for a
  person's check, never a tier of its own.
- **lookups** logs every `fundamentals` call (Vedant, 10-22): the words asked (`what`), the concept
  matched, and whether a fact was found. Counted across distinct `run_id`s, it says which concepts
  are asked for most, to verify and productionise first, and which are asked for and missing.
- **derived facts** keep their `formula` and their inputs in `derived_from`. A derived fact is
  never trusted above its weakest input, whoever checked the arithmetic, and it is flagged when an
  input changes or is deleted.
- **xbrl_concept** maps a core set of PDF row labels to XBRL concepts (Vedant, 10-22): proposed by
  embedding the label against the taxonomy's labels, each mapping checked by hand
  (`mapping_checked`) before it is used.

## Build

`python -m aigent.facts build` (free; no model calls):

1. **Fetch**: list each ticker's filings, download every quarterly and annual consolidated and
   standalone XBRL file for FY24 to FY26 not already cached (about 192 files for the eight companies), one request a second, with a browser user agent. Each file is
   kept under `corpus/.cache/xbrl/` with its URL and SHA-256 in a sorted manifest, and never
   refetched. NSE blocks scripts at times; a failed fetch is reported, the rest still build.
2. **Load XBRL**: contexts, units, dimensions, then every numeric fact of the results format. A
   FY25-on standalone file also carries the integrated filing's related-party transactions
   (`…RelatedPartyTransaction…` contexts; Reliance's has 8,166 numeric facts); those are skipped.
3. **Load PDF tables**: every numeric cell of every parsed table not already held by XBRL, scaled
   by its label's multiplier.
4. **Check**:
   - PDF totals: a table's total row equals the sum of the rows above it (where the table says
     total); a failure flags the table, as Docling's merged rows on Bajaj's ratio table would be;
   - our XBRL loader: the filing's own sums hold once loaded (`Income = RevenueFromOperations +
     OtherIncome`, `Assets = CurrentAssets + NoncurrentAssets`, plus assets held for sale and
     regulatory deferral balances where filed), which a quarter read as a year would break; this
     checks our parsing, not the filing;
   - the 24 fundamentals, each found and equal.
   The build prints the counts and writes a report of every failure.

A rebuild keeps the write-back facts (`pdf-read-ai`, `pdf-read`), which no source can regenerate,
and every person's check: a rebuilt cell whose value and place are unchanged keeps its `pdf-cell`
tier, and one whose value changed drops back to `pdf-cell-ai` and is reported. A derived fact
follows its inputs the same way.

## Lookup

The generalist's existing `fundamentals` tool keeps its name (Vedant) and is rebuilt over the
database; its fixed four-metric, three-company table becomes any concept, company and year:

```text
fundamentals(company, year, what, period = "year", basis = "consolidated", segment = null)
→ up to 5 facts: cite_as, label, value, unit, period, basis, segment, tier, source
```

`period` is one of `year`, `Q1`-`Q4`, `H1`, `9M` (Vedant, 10-23, over year-and-quarter only and
explicit dates): code turns it into dates in the April-March year, and a balance-sheet item is
answered as at the period's last day, so the model needs neither the calendar nor which items are
instants. Many concepts are filed only year to date (profit attributable to owners has H1, 9M and
year, no quarter alone, in the 2020 taxonomy's filings).

1. **Match `what` to concepts by meaning**: embed it with the search embedder and rank the
   concepts' labels (281 XBRL concepts so far), hybrid with BM25 as the filings search is. A
   concept's text is its standard label from NSE's taxonomy packages plus every other label they
   give it. "Net profit" finds
   `ProfitOrLossAttributableToOwnersOfParent` and the PDF row "Profit for the year attributable to
   owners".
2. **Fetch exactly**: SQL over the matched concepts for that company, year, basis, segment and
   period. Nothing is returned for a different year or basis, however similar the words.
3. **Cite**: `cite_as` is `FACT:{ticker}:{fy}:{basis}:{period}:{concept}[:{segment}]` (the period
   added 10-23, since a concept holds a fact per period); a memo claim may cite
   it as it cites a passage id today, and the critic is shown the fact and its source (the NSE
   filing's URL, the PDF page and row, or a read fact's passage).

**Which row answers** (Vedant, 10-22), for a fact (ticker, basis, concept, segment, period):

1. **`xbrl` or a checked tier** (`pdf-cell`, `pdf-read`, `derived`): the answer; nothing is read.
2. **An `-ai` row from this run**: the answer, so a run stays consistent with itself.
3. **Otherwise the agent reads the PDF**, blind: it is not shown other runs' `-ai` rows (or a
   `pdf-cell-ai` row) first, since a value shown beforehand gets copied and a match would then be an
   echo, not agreement. Its read is then compared with them:
   - **a match** adds a row, which the vote counts as agreement;
   - **a mismatch** shows the agent the other value and its passage as a critic, and it reads again
     before writing. The fact goes to the review queue whichever value it settles on.

No free-form SQL for the model: a fixed, parameterised lookup is what the reliability evidence
favours (FinAI Data Assistant, arXiv 2510.14162), and wrong SQL returns confident wrong numbers. A
vector database is not the store either: similarity search returns the nearest thing even when the
right one is missing; embeddings are used only to match `what` to concept labels.

## Write-back: the database keeps growing (Vedant)

XBRL initialises the database; every figure later read from a PDF and approved is written to it, so
the next question needing it is a lookup. The danger is that `check_figures` proves a number is
printed in its passage at the right scale, not that it is the right row or basis (the 10-22
failures), so a wrong read would become a permanent wrong answer. The guards:

- **Only approved figures**: written after the critic and the human approval gate, from the memo's
  cited figures, never from a model's mid-run reads.
- **Tier `pdf-read-ai`**, with its passage and run id stored; returned as such, so the critic
  re-checks it against its passage.
- **A person promotes it** to `pdf-read`. Later reads of the same fact add rows rather than
  overwrite; the polling job's vote over them orders the review queue, and automating the check
  from it is for later, a proxy for a person's check (Vedant, 10-22).
- **Removable by run**: every write carries its `run_id`, so a bad batch goes in one delete.
- **Evals run on a frozen copy**: otherwise each eval run writes answers the next run looks up, and
  scores rise with nothing improved.

## How it is measured

A coverage eval, free, in the retrieval eval's style:

- **Held**: of the 24 fundamentals, the 34 table questions' figures and the v3 teacher labels
  (named metrics), how many are a fact in the database.
- **Found**: of those held, how many `fundamentals` returns in its top 5 for the question's own
  words, and at rank 1.
- **Exact**: the value equal to the label's, scale included.

Then the analyst eval as today, with the rebuilt `fundamentals` in place of the frozen table,
against the current run: accuracy, tool calls and cost per question.

## What it changes

- `fundamentals` keeps its name and its place in the toolbox; its arguments widen (`what` for a
  fixed `metric`, plus `basis`, `segment` and `period`), so its prompt card and tests change with it.
- `evals/reference/fundamentals.json` becomes a test of the database instead of the tool's table.
- `search_filings` stays, for prose and anything not in the database.
- `extract_figures` and the v4 specialist are parked until the database is measured: if lookups
  answer most figure questions, the specialist has little left to do.
- Citations to a fact are smaller than any chunk, the goal the 10-22 chunk work started on.

## Considered and dropped

- **Removing or down-weighting chunks XBRL covers, then a smaller embedder and reranker**
  (Vedant, 10-22). Measured the same day: only 13 of 2,431 table chunks are fully covered and 37 at
  least 80% (ITC, Reliance, Tata Motors FY25), so removal frees almost nothing and cannot carry a
  smaller model. The 10-21 screen's numbers stand for that question if it returns (hybrid hit@5:
  `bge-small` 0.636, tables 0.47; `bge-large` 0.773, tables 0.62).
- **Comparing XBRL against the PDF**: dropped once XBRL was made the source of truth.
- **Free-form SQL and a vector database as the store**: see Lookup.

## Steps

1. ~~Fetcher and manifest; the XBRL loader; the fact table; the fundamentals check.~~ **Built
   10-22** (`aigent.facts`): 192 filings, 34,904 facts, 18 of 18 checkable fundamentals equal,
   every filing's sums holding; four HDFC Bank FY24 H1 filings reference a context they never
   define and their facts are reported, not loaded. Numbers in `LOG.md`.
2. `fundamentals` rebuilt over the XBRL facts, its coverage eval, and the analyst run with it.
   **The tool built 10-23** (`aigent.facts.lookup`, labels from the taxonomy); building it found
   that step 1 had filed every quarter-alone fact under the segment "INR" (the quarter column's
   currency text read as a segment name), now fixed. The coverage eval and the analyst run remain.
3. PDF table facts for what XBRL lacks, and their totals check.
4. Write-back, with its guards and the frozen copy for evals.
5. Decide on the specialist from the measurements.

## Decided (Vedant, 10-22)

- **Quarters**: stored, as well as the year.
- **Years**: FY24 to FY26 for every company.
- **PDF facts' concept names**: printed labels, plus a core set mapped to XBRL concepts, each mapping
  checked by hand.
- **Companies**: the eight in both corpora.
- **Tiers**: the seven in Data model.
- **Provenance**: `written_by` and `written_at` on every row; duplicate `-ai` rows kept and voted on
  by a polling job; `ticker` and `basis` as columns.

- **Sessions**: `run_id` identifies a session; one row per fact per writer per run.
- **Demand**: every lookup logged in `lookups`, to rank concepts for verification.

## Decided (10-23)

- **Periods** (Vedant): named, `year`, `Q1`-`Q4`, `H1`, `9M`; balances at the period's end.
- **Labels** (Vedant): NSE's taxonomy labels, fetched and cached, before the lookup existed.
- **Which filing answers a year** (finance, mine): as originally reported, from the period's own
  filing. Prior-year comparatives are not loaded, so a period is filed once except for opening
  balances, which repeat identically; a repeat is answered from the earliest filing. ITC's FY24
  restated for the hotels demerger is only in the FY25 PDF, not in XBRL.

## Open

- **What `found` means**: a lookup logs `found` when any fact comes back, though the nearest label
  may not be the figure asked (a dividend per share, which XBRL does not file, returns "Adjustments
  for dividend income"). The coverage eval measures how often; a relevance floor waits on it.
- **The vote**: the polling job's rule for preferring a value, decided once there are rows to vote on.
  If it needs to tell a blind read from one changed after a mismatch, a column for the row shown
  is added then.
