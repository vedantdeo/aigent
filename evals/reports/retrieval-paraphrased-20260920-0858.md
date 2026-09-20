# retrieval-paraphrased.jsonl

- dataset: `retrieval-paraphrased.jsonl` (`2015ac292547`, 54 cases)
- model: `BAAI/bge-small-en-v1.5`
- run: 2026-09-20 08:58 UTC
- cost: $0.00000 of a $2.00 ceiling

## Results

| variant | resolvable | hit@5 | recall@5 | mrr | errors | cost |
|---|---|---|---|---|---|---|
| dense | 54/54 (100%) | 21/54 (39%) | 13/54 (24%) | 9/54 (17%) | 0 | $0.00000 |
| dense+rerank | 54/54 (100%) | 26/54 (48%) | 22/54 (41%) | 13/54 (24%) | 0 | $0.00000 |
| bm25 | 54/54 (100%) | 18/54 (33%) | 15/54 (28%) | 9/54 (17%) | 0 | $0.00000 |
| bm25+rerank | 54/54 (100%) | 25/54 (46%) | 19/54 (35%) | 11/54 (20%) | 0 | $0.00000 |
| hybrid | 54/54 (100%) | 23/54 (43%) | 18/54 (33%) | 9/54 (17%) | 0 | $0.00000 |
| hybrid+rerank | 54/54 (100%) | 26/54 (48%) | 22/54 (41%) | 13/54 (24%) | 0 | $0.00000 |

## Metrics (mean per case)

Each metric is meaned over the cases it could score, not over the whole dataset.

| metric | dense | dense+rerank | bm25 | bm25+rerank | hybrid | hybrid+rerank |
|---|---|---|---|---|---|---|
| resolvable | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 0.389 | 0.481 | 0.333 | 0.463 | 0.426 | 0.481 |
| recall@5 | 0.315 | 0.444 | 0.306 | 0.407 | 0.380 | 0.444 |
| mrr | 0.254 | 0.338 | 0.219 | 0.309 | 0.256 | 0.338 |

## Failures

- `rq-001` [dense] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0010)
- `rq-001` [dense+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0010)
- `rq-001` [bm25] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-001` [bm25+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0010)
- `rq-001` [hybrid] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-001` [hybrid+rerank] — **mrr**: first relevant chunk at rank 2 (ITC-FY25#0010)
- `rq-002` [dense] — **mrr**: first relevant chunk at rank 5 (ITC-FY25#0010)
- `rq-002` [dense+rerank] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-002` [bm25] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- `rq-002` [bm25+rerank] — **hit@5**: none of 1 relevant chunks in top 5; **recall@5**: 0/1 in top 5; missed ITC-FY25#0010; **mrr**: no relevant chunk found
- …and more; 259 not shown
