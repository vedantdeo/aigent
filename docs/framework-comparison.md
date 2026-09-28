# LangChain, rebuilt and measured

Week 2 asked for the retrieval pipeline once in LangChain, and ten lines on what it abstracted and
what it hid. Both pipelines ran through the **same** harness — same three PDFs, same 54
quote-labelled questions, same graders, same report — so the comparison is numbers rather than
taste. `uv sync --group compare && uv run python -m aigent.retrieval.langchain_rag`.

| | LangChain | ours (`fixed+overlap`) |
|---|---|---|
| chunks | 4,552 | 4,042 |
| resolvable | 1.000 | 1.000 |
| hit@5 | 0.611 | 0.630 |
| recall@5 | 0.568 | 0.583 |
| MRR | 0.452 | 0.511 |
| statements to build it | **14** | 336 |

Same chunk size (1200/200), same embedding model, same k.

## The ten lines

1. **It abstracted the boring 90%, and did it well.** Fourteen statements — load, split, embed,
   index, retrieve — against 336 for the hand-written pipeline, and the scores land within one row
   of 54 of each other. The framework did not cost accuracy.
2. **Which means our extras bought one row.** Furniture stripping, NFKC normalisation, hyphen
   rejoining, page-offset tracking: measured against LangChain's raw extraction, worth ~0.019
   hit@5. That is the uncomfortable half of this exercise and it is only sayable because both went
   through one harness.
3. **It hid that chunks never cross a page boundary.** `PyPDFLoader` returns one `Document` per
   page, so the splitter splits *within* pages. Our pipeline joins pages first, so a sentence
   broken across a page break survives. Nothing says so; it shows up as 510 extra chunks.
4. **It hid that a chunk has no offset into its source** unless you pass `add_start_index=True`,
   which is off by default. Without it a page citation cannot be computed at all.
5. **It hid that a `Document` has no identity.** No id field. Matching a retrieved chunk back to a
   label needs an id you put in `metadata` yourself, because `InMemoryVectorStore` returns
   reconstructed objects — so keying on `id(chunk)` matches nothing.
6. **And that failure is silent.** The first run scored **0.000 on every metric with no error**:
   the mapping missed every time, empty id lists graded as clean misses. A metric of exactly zero
   is a plumbing bug until proven otherwise.
7. **It hid page numbering.** `metadata["page"]` is 0-indexed, where every citation in this repo is
   1-indexed. Off by one, silently, in the field a reader would use to check a quote.
8. **It costs 31 packages for 3 direct dependencies** — 61 installed to 92 — including SQLAlchemy
   and a telemetry client, for a pipeline that reads PDFs and multiplies matrices.
9. **And the loader is already deprecated.** `langchain-community` warns on import that it is being
   sunset, at current versions, installed today.
10. **The honest conclusion is not "write it yourself".** It is that a framework is worth it until
    you need to know *why* a number moved — and every one of the hidden things above is something
    you would need to know. Keep the framework for the parts nobody will debug; own the parts the
    eval points at.

## What this does not say

It is one corpus, one embedding model, and 54 questions in which a one-row difference is 0.019.
LangChain is not slower or less accurate here in any way that survives that noise. The differences
that matter are in what you can *see* and *fix*, not in the scores.

The comparison also flatters the hand-written side on code volume: 336 statements buys four
chunking strategies, an abbreviation-aware sentence splitter, quote resolution, a disk cache and
two guards on building the dense index — most of which LangChain would also give you, from a different
part of its surface, had the comparison been drawn there instead.
