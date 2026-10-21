# retrieval.jsonl+retrieval-tables.jsonl

- dataset: `retrieval.jsonl+retrieval-tables.jsonl` (`63c0e99ab5df`, 88 cases)
- model: `BAAI/bge-large-en-v1.5`
- run: 2026-10-21 12:12 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| dense | 88/88 (100%) | 55/88 (62%) | 54/88 (61%) | 31/88 (35%) | 0 | $0.00000 |
| dense+rerank | 88/88 (100%) | 71/88 (81%) | 70/88 (80%) | 52/88 (59%) | 0 | $0.00000 |
| bm25 | 88/88 (100%) | 62/88 (70%) | 60/88 (68%) | 41/88 (47%) | 0 | $0.00000 |
| bm25+rerank | 88/88 (100%) | 75/88 (85%) | 74/88 (84%) | 55/88 (62%) | 0 | $0.00000 |
| hybrid | 88/88 (100%) | 68/88 (77%) | 67/88 (76%) | 39/88 (44%) | 0 | $0.00000 |
| hybrid+rerank | 88/88 (100%) | 78/88 (89%) | 78/88 (89%) | 56/88 (64%) | 0 | $0.00000 |
| keys+hybrid+rerank | 88/88 (100%) | 81/88 (92%) | 80/88 (91%) | 58/88 (66%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | dense | dense+rerank | bm25 | bm25+rerank | hybrid | hybrid+rerank | keys+hybrid+rerank |
|---|---|---|---|---|---|---|---|
| resolvable | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 0.625 | 0.807 | 0.705 | 0.852 | 0.773 | 0.886 | 0.920 |
| recall@5 | 0.620 | 0.802 | 0.692 | 0.847 | 0.766 | 0.886 | 0.918 |
| mrr | 0.446 | 0.677 | 0.547 | 0.716 | 0.563 | 0.736 | 0.766 |

## Failures

- `rq-001` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-001` [dense+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0010)
- `rq-001` [bm25+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0010)
- `rq-001` [hybrid] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0010)
- `rq-001` [hybrid+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0010)
- `rq-001` [keys+hybrid+rerank] — **mrr**: first relevant chunk at rank 3 (ITC-FY25#0010)
- `rq-002` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- `rq-002` [bm25] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- `rq-002` [hybrid] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0011)
- `rq-003` [dense] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0011; **mrr**: no relevant chunk found
- …and more; 281 not shown
