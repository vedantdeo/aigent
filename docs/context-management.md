# Context management, measured

Week 3 asks what goes in the system prompt, what goes in tool results, and when to summarize. The
answers below come from the agent's live runs on 2026-09-22 (LOG.md has every number) and from the
free token count. The model is Opus 5: input $5 per million tokens, a cache write 1.25×, a cache
read 0.1×.

## What each part costs

Input tokens, counted in isolation. The tool rows include the preamble the API adds whenever tools
are present, so they do not sum; the agent's whole first request is 1,139 tokens without web
search and 3,579 with it.

| part | tokens | changes |
|---|---|---|
| system prompt: the reports, how to search, cite, and when to use the web | 185 | never |
| `search_reports` schema | 496 | never |
| calculator, time and file schemas | 716 | never |
| web search, called directly | 2,726 | never |
| web search with dynamic filtering (tried, not kept) | about 5,700 | never |
| one `search_reports` result: five passages, each tagged with its id | 2,000–2,700 | every turn it is called |
| one direct web search's results | about 9,000 | every turn it is called |
| the last-answer instruction | 39 | once, when a run runs out of room |

## What goes where

**The system prompt holds what never changes within a run**: who the model is answering for, the
three reports by id, and the rules of the job — search as often as needed, rephrase on a miss,
scope a search to a named company, cite ids, use the web only for what the reports cannot hold.
Nothing per-question goes there, not even today's date. The system prompt and the tools are the
front of the cached prefix, and a change to either re-writes everything after it at 1.25×.

**Tool definitions are part of that prefix too.** They are fixed per loop, in a fixed order. An
agent that adds or drops a tool mid-run rebuilds its whole cache. Switching `tool_choice` keeps the
tools and system prompt cached but re-writes the whole conversation, which is why the last-answer
turn asks in text rather than setting `tool_choice: none`.

**Tool results carry evidence, shaped for citing and bounded in size.** A search returns five
passages, each wrapped with the id to cite it by: about 2,000–2,700 tokens, set by `TOP_K` and
`CHUNK_CHARS`, not by the query. A miss says so in one line rather than returning nothing. An
error comes back as an `is_error` result the model can act on, never as an exception. Web results
go back exactly as the API returned them, since their encrypted content must be replayed
unchanged.

**An instruction for one turn goes after that turn's tool results**, as a text block, never into
the system prompt. The last-answer instruction does this, which keeps the conversation cached
while it changes what the model does next.

## How a run's context grows

Everything above is resent on every turn, so the context grows by each turn's results plus the
model's output.

| run | turns | context, first → last | cost |
|---|---|---|---|
| rural demand, no cache | 7 | 1,118 → 29,464 | $0.626; turn 7 cost 14× turn 1 |
| rural demand, cached | 5 | 1,118 → 21,133 | $0.199, 38% less for the same tokens |
| rural demand, cached, 8 turns | 8 | 1,118 → 28,952 | $0.351, answered |
| share price, direct web search | 3 | 3,579 → 24,407 | $0.250, 6 pages cited |

**Without a cache, cost grows with the square of the turns**, because every turn pays full price
for everything before it. With one, each turn reads the last one's prompt at a tenth of the price
and writes only what is new, so cost grows roughly with the turns. Caching lowers the bill, not
the ceiling: a turn that may write the cache is admitted at the write price, since a miss writes
the whole prefix.

**A search turn is bigger than its context.** Within one request, the API's search loop re-reads
the conversation at every step. A direct search turn counted 38,744 input tokens, and the
conversation after it was 24,407. With dynamic filtering, a turn counted 86,007, and the next was
22,267. Those rereads are cached within the turn, so they cost little, but the trace's `in` column
shows them.

**The budget, not the window, ends a run.** `MAX_USD_PER_TURN` ($0.60) admits a turn only if its
worst case fits: the context at the write price, the 4,096-token output cap, and, with web search
offered, three searches' allowance. That leaves about 44,800 tokens of context with web search and
about 79,600 without. That is far below the model's context window, so no run here has been
limited by the window. When the next turn will not fit, the run gets one last turn to answer from
what it has.

## When to summarize

**Not yet.** Four reasons, in order of weight:

1. **Nothing here comes near the window.** The budget ends every run at under 80,000 tokens.
2. **A summary drops the citations.** The answer has to name the passage ids and pages behind its
   figures, and a summary of twenty passages keeps the gist and loses the ids.
3. **Summarizing resets the cache.** A compacted history is a new prefix, re-written at 1.25×, and
   the reads it replaces cost a tenth of that.
4. **Editing history on our side is fragile.** Removing old results from the transcript changes a
   prefix that later thinking blocks were created against. The docs' supported routes are
   server-side compaction or context editing, not client-side edits.

**What would change that:** a task that needs more evidence than a turn can hold, which Week 4's
multi-step tasks may be, or a long chat session. `primitives.chat` grows until `/reset`. Then the
route is server-side compaction, triggered by token count, with instructions to keep every id and
URL the answer may cite, and priced like any other call before it is sent.
