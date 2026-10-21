# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-small-en-v1.5`
- run: 2026-10-21 06:35 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| dense | 88/88 (100%) | 43/88 (49%) | 42/88 (48%) | 25/88 (28%) | 0 | $0.00000 |
| dense+rerank | 88/88 (100%) | 58/88 (66%) | 57/88 (65%) | 38/88 (43%) | 0 | $0.00000 |
| bm25 | 88/88 (100%) | 62/88 (70%) | 60/88 (68%) | 41/88 (47%) | 0 | $0.00000 |
| bm25+rerank | 88/88 (100%) | 71/88 (81%) | 70/88 (80%) | 42/88 (48%) | 0 | $0.00000 |
| hybrid | 88/88 (100%) | 56/88 (64%) | 56/88 (64%) | 31/88 (35%) | 0 | $0.00000 |
| hybrid+rerank | 88/88 (100%) | 71/88 (81%) | 70/88 (80%) | 43/88 (49%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | dense | dense+rerank | bm25 | bm25+rerank | hybrid | hybrid+rerank |
|---|---|---|---|---|---|---|
| resolvable | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 0.489 | 0.659 | 0.705 | 0.807 | 0.636 | 0.807 |
| recall@5 | 0.483 | 0.652 | 0.692 | 0.802 | 0.636 | 0.802 |
| mrr | 0.360 | 0.516 | 0.547 | 0.599 | 0.463 | 0.606 |

## Failures

- `rq-001` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-001` [dense+rerank] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-001` [hybrid] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-002` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- `rq-002` [bm25] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- `rq-002` [hybrid] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0011)
- `rq-003` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- `rq-003` [dense+rerank] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- `rq-003` [bm25] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0011)
- `rq-003` [bm25+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- …and more; 303 not shown
