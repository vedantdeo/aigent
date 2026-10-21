# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-small-en-v1.5`
- run: 2026-10-21 06:30 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| fixed | 84/88 (95%) | 41/88 (47%) | 39/88 (44%) | 14/88 (16%) | 0 | $0.00000 |
| fixed+overlap | 88/88 (100%) | 48/88 (55%) | 40/88 (45%) | 20/88 (23%) | 0 | $0.00000 |
| by_sentence | 86/88 (98%) | 46/88 (52%) | 37/88 (42%) | 23/88 (26%) | 0 | $0.00000 |
| by_element | 88/88 (100%) | 43/88 (49%) | 42/88 (48%) | 25/88 (28%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | fixed | fixed+overlap | by_sentence | by_element |
|---|---|---|---|---|
| resolvable | 0.955 | 1.000 | 0.977 | 1.000 |
| hit@5 | 0.488 | 0.545 | 0.535 | 0.489 |
| recall@5 | 0.476 | 0.501 | 0.483 | 0.483 |
| mrr | 0.281 | 0.346 | 0.370 | 0.360 |

## Failures

- `rq-001` [fixed] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0006)
- `rq-002` [fixed] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0006)
- `rq-003` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0006; **mrr**: no relevant chunk found
- `rq-004` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0006; **mrr**: no relevant chunk found
- `rq-005` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0006; **mrr**: no relevant chunk found
- `rq-006` [fixed] — **recall@5**: 1/2 in top 5; missed ITC-FY25#0038; **mrr**: first relevant chunk at rank 2 (ITC-FY25#0033)
- `rq-008` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0068; **mrr**: no relevant chunk found
- `rq-009` [fixed] — **recall@5**: 2/4 in top 5; missed ITC-FY25#0068, ITC-FY25#0586; **mrr**: first relevant chunk at rank 2 (ITC-FY25#0541)
- `rq-011` [fixed] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0261)
- `rq-012` [fixed] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0302; **mrr**: no relevant chunk found
- …and more; 270 not shown
