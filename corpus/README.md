# corpus

The documents Project 1 retrieves over: Indian annual reports, chosen so that Project 1a's ticker
directory and this week's documents describe the same companies.

The PDFs themselves are **not** in git — they are tens of megabytes each and not ours to
redistribute. `evals/reference/corpus-manifest.json` records where each one came from and its
SHA-256, which is what makes a retrieval number reproducible. To fetch them:

```sh
./scripts/fetch-corpus.sh
```

`evals/reference/training-manifest.json` lists a second set: reports from other companies that
only the figure-extraction fine-tune trains on. They live in `corpus/training/`, which
`load_corpus` never reads, so no retrieval number depends on them. The same script fetches both.

Annual report URLs move, and companies quietly republish a report after a correction. Either shows
up as a failure from that script rather than as a retrieval score that drifted for no visible
reason. If a URL has moved, find the current one and update the manifest; if a digest has changed,
the numbers in the README were measured against a document that no longer exists, and say so.

The filename stem is the `doc_id`, so every chunk id in a report is built from it
(`RELIANCE-FY25#0042`). Renaming a file renames every chunk in it.
