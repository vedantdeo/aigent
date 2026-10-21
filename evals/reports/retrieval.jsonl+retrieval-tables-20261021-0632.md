# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-small-en-v1.5`
- run: 2026-10-21 06:32 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| dense | 86/88 (98%) | 46/88 (52%) | 37/88 (42%) | 23/88 (26%) | 0 | $0.00000 |
| dense+rerank | 86/88 (98%) | 56/88 (64%) | 50/88 (57%) | 46/88 (52%) | 0 | $0.00000 |
| bm25 | 86/88 (98%) | 65/88 (74%) | 62/88 (70%) | 46/88 (52%) | 0 | $0.00000 |
| bm25+rerank | 86/88 (98%) | 62/88 (70%) | 57/88 (65%) | 48/88 (55%) | 0 | $0.00000 |
| hybrid | 86/88 (98%) | 58/88 (66%) | 54/88 (61%) | 29/88 (33%) | 0 | $0.00000 |
| hybrid+rerank | 86/88 (98%) | 63/88 (72%) | 56/88 (64%) | 47/88 (53%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | dense | dense+rerank | bm25 | bm25+rerank | hybrid | hybrid+rerank |
|---|---|---|---|---|---|---|
| resolvable | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 |
| hit@5 | 0.535 | 0.651 | 0.756 | 0.721 | 0.674 | 0.733 |
| recall@5 | 0.483 | 0.615 | 0.739 | 0.691 | 0.652 | 0.691 |
| mrr | 0.370 | 0.584 | 0.619 | 0.624 | 0.482 | 0.622 |

## Failures

- `rq-001` [dense] — **mrr**: first relevant chunk at rank 4 (ITC-FY25#0009)
- `rq-001` [bm25] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0009; **mrr**: no relevant chunk found
- `rq-001` [bm25+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0009)
- `rq-001` [hybrid] — **mrr**: first relevant chunk at rank 4 (ITC-FY25#0009)
- `rq-001` [hybrid+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0009)
- `rq-003` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0009; **mrr**: no relevant chunk found
- `rq-003` [bm25] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0009)
- `rq-003` [hybrid] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0009; **mrr**: no relevant chunk found
- `rq-004` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0009; **mrr**: no relevant chunk found
- `rq-004` [dense+rerank] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0009; **mrr**: no relevant chunk found
- …and more; 304 not shown
