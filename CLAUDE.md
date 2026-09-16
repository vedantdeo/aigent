# Entropic — notes for Claude

Eight project rules live here rather than in a session's private memory, so they travel with the
repo and reach anyone who clones it.

Five of them — constants, testing, reference data, comments and paid calls — are verbatim
mirrors of global preferences in `~/.claude/CLAUDE.md`, which is machine-local and backed up by
nothing. The copies below are the durable ones. Edit both, or neither.

## Read the knowledge graph before exploring

`docs/knowledge-graph.md` maps every module, symbol, edge and invariant in this codebase, plus the
seams where each upcoming roadmap week attaches. Read it first when picking up work here — it is
faster than re-reading the tree, and it records conventions that are invisible from a quick skim.

The load-bearing ones, as a taste of what is in there: every tool result for a turn goes back in
*one* user message; tool errors return as `tool_result` with `is_error: true` and never as
exceptions; every paid call is pre-flighted through `pricing.check_request`; `config` is the only
module that builds a client, and `pricing` the only one that knows a price.

Section 4 of that file (invariants) matters more than the `file:line` anchors in section 2 — anchors
drift, invariants do not.

## Every commit updates the knowledge graph

A commit that changes what the graph records — a module, symbol or edge; a contract, price, default
or ceiling; an invariant or a consumed seam — updates `docs/knowledge-graph.md` in the *same commit*.
Bump the "verified against" line at the top, and re-check the anchors for any file whose line numbers
moved.

**Why:** a graph that lags the code is worse than none. The next session trusts a stale anchor or a
retired invariant and works from a map of a repo that no longer exists.

**How to apply:** treat the doc edit as part of the change, not a follow-up — stage it with the code,
and say in the commit summary that the graph moved with it. If a change genuinely records nothing in
the graph, say so explicitly rather than skipping quietly.

This is enforced, not just asked for: `.githooks/pre-commit` refuses a commit that stages `src/` or
`pyproject.toml` without the graph, and `.github/workflows/knowledge-graph.yml` runs the same rule in
CI for clones that never ran `git config core.hooksPath .githooks`. Both call
`scripts/check-knowledge-graph.sh`, so the rule cannot drift between them. `git commit --no-verify`
is the deliberate exception — use it only for a change the graph truly does not record, and say why.

## Constants and configuration

**The rule: tunable constants live in one config module per project** — `config.py`, `settings.py`,
or whatever the project already calls it — rather than scattered beside the code that uses them.
Qualify names by call site when they would otherwise collide: `MAX_TOKENS_CHAT`, not a local
`MAX_TOKENS` in every module.

- Move the settings: limits, caps, timeouts, retries, defaults, model names, spending ceilings.
- Leave behind what is not a setting: values derived at import from `__file__` (those are locations,
  not knobs), and private algorithm guards nobody would tune from outside.

**Why:** a number sitting alone in its own module is invisible. Side by side the set reads as a set —
you can see which call is the expensive one, notice a value that drifted from its neighbours, and
change a policy in one edit instead of six.

**The fallback, when one config module stops working: split the config — never scatter the constants
back into the code.** Two seams are legitimate:

- **By concern**, once the file is genuinely too big to read: settings in one module, the arithmetic
  that consumes them in another.
- **By isolation**, when a module is deliberately free of internal imports so it can be lifted into
  another project or framework unchanged. It gets its own config module travelling beside it, not a
  dependency on the package-wide one. Both properties then survive — one place per concern, and no
  import. Say so in that module's docstring.

Don't split pre-emptively, don't split alphabetically, and never split into one config per consumer.
The fallback exists so the rule survives a hard case, not so it can be worked around.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss of
this machine. Edit both, or neither.

**In this repo**, concretely — the rule and both seams already have worked examples:

- The one config module is `src/entropic/config.py`: models, spending ceilings, output caps
  (`MAX_TOKENS_CHAT` and friends), the agent turn cap, the report's failure cap.
- *Split by concern:* `pricing.py` holds the `Price`/`Budget` classes and the cost arithmetic;
  `config.py` kept the settings they read. It happened when `config.py` got too big to read, not
  before.
- *Split by isolation:* `tools.py` imports nothing from this package, which is the whole reason it
  can be lifted into the SDK tool runner in Week 5 and a LangGraph node in Week 6. Its constants live
  in `tools_config.py`, travelling beside it. Not in `config.py`, which would cost `tools` its
  independence; not loose in `tools.py`, which would cost the one-place rule.
- Staying put: `report.REPORTS_DIR`, a location computed from `__file__` rather than a knob.

## Tests

**The rule: when several tests differ only in their inputs and their expected outputs, make them
one parametrized test over a table of rows** — not one function per case. A table is the shape the
knowledge already has: a column for each thing that varies, a row per case.

- Club together what differs only in data: the same call, the same assertions, different arguments.
- Give every row an id that reads as a sentence about the case (`id="a symlink pointing out"`), so
  the output names the broken case without anyone opening the file. Plain tuples are fine where the
  values speak for themselves; reach for `pytest.param(..., id=...)` where they do not.
- Keep apart what differs in *shape*: different setup, a sequence of calls, several objects
  interacting. Needing a column of optional expectations that most rows leave empty is the signal
  that those cases are not one test.
- Share fakes and fixtures through `conftest.py` rather than copying a stub into each module that
  needs it. Two copies of a fake client drift, and the drift stays invisible until one is wrong.
- Assert against the row's own data, and pass the interesting value as the assertion message
  (`assert score.passed is passed, score.detail`), so a failing row explains itself.

**Why:** ten near-identical functions hide the one line that differs between them, and adding the
eleventh case means copying a function instead of adding a row. A table makes the coverage readable
as a set — you can see at a glance which case is missing — and it is the difference between a suite
that grows by editing and one that grows by cloning.

**Don't table for its own sake.** A row that needs its own assertions is a test, not a row, and a
table that grows more columns than rows has stopped being one. Fewer test *functions* is the goal;
fewer lines is a frequent side effect, not the point.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

**In this repo**, the worked examples are in `tests/`:

- `tests/test_evals_graders.py` is a table per grader — `regex` is eight rows of
  (pattern, value, passed) where it was five functions, and the row that proves case survives sits
  one line under the row that proves it matters.
- `tests/test_tools.py` tables the dispatcher (six rows of tool, input, `is_error`, expected text)
  and the three ways a path can escape the sandbox.
- `tests/conftest.py` holds the one scripted `LlmJudge`. The grader tests use it to check what the
  judge says, the runner tests to check that the runner bills it — one fake, one set of token
  counts, so the two cannot drift.
- Staying apart: `tests/test_tool_loop.py`, where each test scripts a different conversation, and
  `tests/test_evals_report.py`, where each test reads a different section of the same report.

## Reference data stays in one canonical order

**The rule: a file that people hand-edit and code reads back — a lookup table, an allowlist, a
mapping, a set of fixtures — is written in one canonical order, sorted by the key it is looked up
by.** Sort on write, in the code that dumps it, not as a tidy-up pass afterwards.

- Sort by the key the data is addressed by: the ticker in a ticker table, the package in a
  dependency list, the id in a fixture file. Where there is no natural key, sort by whatever a
  reader would scan for.
- Keep the shape inside each entry stable too — the same fields in the same sequence — so a changed
  value is a one-line diff and not a rewritten block.
- Guard it with a test rather than a convention. A canonical order that nothing checks survives
  exactly until the first tool writes the file back in its own order.

**Why:** a canonical order gives a new entry exactly one place it can go, so its diff shows that
entry and nothing else. Append-anywhere files drift into a private order that only the last editor
knows, and then the first process to re-serialise one produces a diff touching every line — which
buries the single line that actually changed and makes reviewing a data file worthless. Sorted data
is also scannable: a human looks a key up instead of searching for it.

**The exception is when the order is itself the data.** A migration list, a pipeline's stages, a
priority ranking, a changelog, a fallback chain — these are sequences, and sorting them destroys
meaning rather than revealing it. Say so in the file, so the next reader knows the disorder is
deliberate and the next tool does not helpfully fix it.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

**In this repo:** `evals/reference/nse-tickers.json` is sorted by ticker and dumped with
`dict(sorted(companies.items()))`. `tests/test_reference_data.py` holds the registry of files
the rule covers and fails if any of them drifts. The counterexample
sits in the same feature — `week01.extraction.METRICS` is a *total order* over the metric
vocabulary, so alphabetising it would silently rewrite every tie-break the eval depends on.

## Comments and docstrings both stay short

**The rule: a comment is one line, and a docstring is a brief description of the module, class or
function — not an argument for it.** A sentence or two saying what the thing is, plus anything a
caller cannot read off the signature. That is the whole budget.

- Say only what the code does not. `# increment i` is noise; `# the API 1-indexes pages` is not.
- Put it at the line that surprises, not at the top of the block.
- **Rationale is not documentation.** Why a default was chosen, what a run measured, which bug a
  guard was written for, what was tried and rejected — none of that belongs beside the code. It
  goes wherever the project keeps its decision record, which is read once on arrival.
- When a comment grows past its line, or a docstring past its couple of sentences, that is the
  signal to move it, not to wrap it.

**Why:** ten lines above a one-line constant do not get read, they get scrolled past — and then they
rot unnoticed, because nobody was reading closely enough to catch the drift. A docstring that argues
its own case is the same failure at larger scale: it buries the one sentence the reader needed, and
it goes stale the first time the reasoning changes without the code.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

**In this repo**, the decision record is two files. `docs/knowledge-graph.md` holds contracts,
invariants and why a node exists; `LOG.md` holds what a run cost and what it measured. Anything
phrased "because we found that…" belongs in one of those, never in a docstring.

## Paid calls

**The rule: never send a paid API call without explicit per-action confirmation.** Each run needs
its own — previous approval is single-use, exactly as it is for `git commit` and `git push`. A
one-off diagnostic call is not exempt: a two-cent probe is still someone else's money, spent on
your initiative.

- Say what the call is for and what it will cost, then wait. Never report a paid call as already
  running.
- Where a script has a dry-run flag, run *that* first and surface the number it prints, rather than
  estimating the cost in prose.
- Token counting is free. Use it freely, and say it is free when a plan rests on it, so the
  unpriced half of the work is never mistaken for the priced half.
- "It was only a few cents" is the reasoning that erodes this. The amount is not the point; the
  decision belongs to whoever is paying.

**Why:** cost ceilings, pre-flight counts and dry-run flags all exist to put the spend in front of
a person before it happens. Spending unasked routes around machinery someone deliberately built,
and the habit scales badly — the instinct that fires off an unapproved probe is the one that later
fires off an unapproved hundred-row run.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

**In this repo**, the machinery is already there and the rule is about respecting it:

- Every paid entry point stops by default. `python -m entropic.week01.extraction` (add `--cache`
  for the caching arms) prints the per-arm token counts and a worst case, then exits; `--yes` is
  what sends the calls. Run it without `--yes` and paste what it prints — that is the number to get
  approved, not an estimate written from memory.
- `pricing.check_request` pre-flights every call through the free counting endpoint, and
  `config.MAX_USD_PER_REQUEST` / `_RUN` / `_EVAL` are the three ceilings. They are guards against
  mistakes, not a substitute for permission: a run well under the ceiling still needs asking.
- `--sample N` exists for cheap smoke runs, and is the right thing to propose when unsure. One
  exception worth knowing: caching a single row costs *more* than not caching it, so `--sample 1`
  cannot smoke-test the cache.
- Every run's cost goes in `LOG.md` beside what it bought. Keep that ledger honest — it is the only
  record of what this project has spent.

## Turn on branch protection before anyone else can touch this repo

**Trigger: the moment a collaborator is added, or the repo goes public — planned for the `v0.1-rag`
tag at the end of Week 2 (Fri 2026-09-18). Raise it then; do not wait to be asked.**

CI is a detector, not a gate. GitHub accepts a push first and runs the workflows after, so a bad
commit reaches `main` and goes red a minute later. That is fine while this is a one-person repo and
the loop is push-look-fix. It stops being fine the moment someone else can pull a red commit or push
their own.

What to enable on `main` at that point: require a pull request, require both checks to pass, and
block force pushes and deletion. The two required checks are matched by *job* name, not workflow
name, so they are exactly:

- `lint · types · tests` (from `.github/workflows/checks.yml`)
- `graph moves with the code` (from `.github/workflows/knowledge-graph.yml`)

Settings > Rules > Rulesets on GitHub, or `gh api` against the branch-protection endpoint. Surface
the exact call before running it — it changes who can write to the repo.
