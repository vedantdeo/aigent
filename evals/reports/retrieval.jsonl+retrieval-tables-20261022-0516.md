# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-large-en-v1.5`
- run: 2026-10-22 05:16 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| dense | 88/88 (100%) | 55/88 (62%) | 54/88 (61%) | 30/88 (34%) | 0 | $0.00000 |
| dense+rerank | 88/88 (100%) | 72/88 (82%) | 72/88 (82%) | 55/88 (62%) | 0 | $0.00000 |
| bm25 | 88/88 (100%) | 65/88 (74%) | 63/88 (72%) | 40/88 (45%) | 0 | $0.00000 |
| bm25+rerank | 88/88 (100%) | 78/88 (89%) | 78/88 (89%) | 56/88 (64%) | 0 | $0.00000 |
| hybrid | 88/88 (100%) | 71/88 (81%) | 70/88 (80%) | 38/88 (43%) | 0 | $0.00000 |
| hybrid+rerank | 88/88 (100%) | 77/88 (88%) | 77/88 (88%) | 57/88 (65%) | 0 | $0.00000 |
| keys+hybrid+rerank | 88/88 (100%) | 80/88 (91%) | 79/88 (90%) | 58/88 (66%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | dense | dense+rerank | bm25 | bm25+rerank | hybrid | hybrid+rerank | keys+hybrid+rerank |
|---|---|---|---|---|---|---|---|
| resolvable | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 0.625 | 0.818 | 0.739 | 0.886 | 0.807 | 0.875 | 0.909 |
| recall@5 | 0.618 | 0.818 | 0.728 | 0.886 | 0.802 | 0.875 | 0.903 |
| mrr | 0.451 | 0.711 | 0.561 | 0.743 | 0.581 | 0.743 | 0.760 |

## Failures

- `rq-001` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- `rq-001` [dense+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0011)
- `rq-001` [bm25+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- `rq-001` [hybrid] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- `rq-001` [hybrid+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0011)
- `rq-001` [keys+hybrid+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0011)
- `rq-002` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0012; **mrr**: no relevant chunk found
- `rq-002` [bm25] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0012)
- `rq-002` [hybrid] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0012)
- `rq-003` [dense] — **mrr**: first relevant chunk at rank 4 (ITC-FY25#0012)
- …and more; 277 not shown
