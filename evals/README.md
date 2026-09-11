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
