# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-large-en-v1.5`
- run: 2026-11-24 11:14 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| fixed | 84/88 (95%) | 43/88 (49%) | 41/88 (47%) | 24/88 (27%) | 0 | $0.00000 |
| fixed+overlap | 88/88 (100%) | 55/88 (62%) | 46/88 (52%) | 33/88 (38%) | 0 | $0.00000 |
| by_sentence | 86/88 (98%) | 49/88 (56%) | 37/88 (42%) | 26/88 (30%) | 0 | $0.00000 |
| by_element | 88/88 (100%) | 55/88 (62%) | 54/88 (61%) | 30/88 (34%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | fixed | fixed+overlap | by_sentence | by_element |
|---|---|---|---|---|
| resolvable | 0.955 | 1.000 | 0.977 | 1.000 |
| hit@5 | 0.512 | 0.625 | 0.570 | 0.625 |
| recall@5 | 0.500 | 0.573 | 0.499 | 0.618 |
| mrr | 0.371 | 0.462 | 0.414 | 0.451 |

## Failures

- `rq-001` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0006; **mrr**: no relevant chunk found
- `rq-003` [fixed] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0006)
- `rq-004` [fixed] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0006)
- `rq-005` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0006; **mrr**: no relevant chunk found
- `rq-006` [fixed] — **recall@5**: 1/2 in top 5; missed ITC-FY25#0038
- `rq-008` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0068; **mrr**: no relevant chunk found
- `rq-009` [fixed] — **recall@5**: 2/4 in top 5; missed ITC-FY25#0068, ITC-FY25#0586
- `rq-011` [fixed] — **mrr**: first relevant chunk at rank 4 (ITC-FY25#0261)
- `rq-013` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0302; **mrr**: no relevant chunk found
- `rq-014` [fixed] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0328)
- …and more; 244 not shown
