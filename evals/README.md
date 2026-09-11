# evals

Data and results live here; the harness that reads them lives in `src/entropic/evals/`. The split is
deliberate — the code ships inside the package, the datasets do not.

- `datasets/*.jsonl` — one case per line, hand-labelled. These are the asset. Prompts change every
  week; a labelled dataset is the thing that lets you tell whether the change helped.
- `reports/*.md` — one file per run, written by `write_report`. Committed on purpose: a score is
  only meaningful next to another score.

## Case format

```json
{"id": "hl-001", "input": {"headline": "..."}, "expected": {"company": "Infosys"}, "tags": ["q3"]}
```

`id` is how you cite a failure, so keep it stable. `input` is whatever the task needs, `expected` is
the label (leave it out for a case you only check for validity), `tags` are for slicing the table.
Unknown keys are refused — a mistyped `expcted` would otherwise label nothing, silently.

Blank lines and `//` lines are skipped, which is where a case you have not decided on can wait.

A case with no `expected` is an **unlabelled** input, not a free pass: `field_match` counts a
missing label as wrong, so a run that includes one drags its per-field table down. Filter to
`[c for c in cases if c.expected]` before scoring accuracy — `week01.extraction.main` does, and
says how many it skipped.

**Label numbers as floats.** `field_match` compares text, and `str(12) != str(12.0)`, so a label
written `12` can never match a model that returns `12.0`. Write `12.0`.

## Reference data

`reference/nse-tickers.json` maps each NSE ticker to its registered name and the alias forms a
headline actually uses (`HUL`, `Infy`, `L&T`). It does two jobs, and they pull in the same
direction: it is the ground truth the `ticker` labels are checked against — a label outside the
directory is unfalsifiable — and it is what the task looks up at runtime.

**The model is never asked for a ticker.** It copies the company as the headline writes it, and
`Resolver` maps that to a symbol. A lookup is exact and free; a model guessing symbols invents
`HEROMOTOCORP` for `HEROMOTOCO` and looks right doing it. 11 of the 29 tickers in `headlines.jsonl`
cannot be derived from any name the headline uses, so that is 38% of the field taken off the
model's plate rather than prompted around.

When a mention does not resolve, the run says so and names it — a missing line in this file, not a
wrong answer. Add the alias and re-run; that row was never the model's fault.

## Datasets

- `headlines.jsonl` — 50 Indian market results headlines for Project 1a, 30 labelled with
  `ticker`, `metric`, `quarter`, `direction`, `change_pct`. The company field is called `ticker`
  because that is what it holds — the **NSE symbol**, not a name: one canonical key per company, so a headline saying Infosys and one saying Infy both label
  as `INFY` and `field_match` has nothing to argue about. The 20 unlabelled rows are the
  backlog, each needing a call the schema does not settle. `tests/test_extraction.py` validates
  every label against the schema the model is given, so a typo in the asset fails the suite rather
  than quietly costing a point.

## Running one

```python
cases = load_jsonl(path)
run = run_eval(
    cases,
    {"baseline": task},
    {"fields": field_match(["company", "quarter"])},
    dataset=path.name,
    digest=digest(path),
)
print(write_report(run))
```

The per-eval ceiling (`ENTROPIC_MAX_USD_PER_EVAL`, $2.00 by default) stops a run that overruns and
keeps the rows it already paid for. Call `estimate_eval_usd` first if you want the worst case before
spending anything.
