# The SDK tool runner, read against the loop we wrote

Week 3 asked for one true agent loop on the SDK's tool runner, and for its source to be read.
`client.beta.messages.tool_runner(...)` lives in `anthropic/lib/tools/_beta_runner.py` (546
lines, `anthropic` 1.4.0) and its tool wrapper in `_beta_functions.py`. `primitives/tool_loop.py`
is the same loop written by hand in about sixty lines, and the knowledge graph's invariants 3–6
are the rules it keeps. This is what the runner does against each of them, and what it leaves out.

## The loop, as the runner writes it

```
while iterations < max_iterations:              # no limit unless you pass one
    message = client.beta.messages.parse(**params)   # sent immediately — no count, no budget
    yield message                               # your per-turn hook: before any tool has run
    match stop_reason:
        tool_use            → run every tool_use block, in order, one at a time
                              append the assistant turn + ONE user message of results
        pause_turn, compaction → append the turn unchanged, go again (server tools)
        anything else       → return
```

## Against our four rules

| Rule (graph invariant) | The runner | |
|---|---|---|
| 3 — echo the assistant turn verbatim | `{"role": message.role, "content": message.content}`: the blocks, unchanged, thinking included | kept |
| 4 — every tool result in ONE user message | one `{"role": "user", "content": results}` per turn | kept |
| 5 — tool errors are results, never exceptions | any exception becomes `is_error: true` with `repr(exc)`; a `ToolError` carries its own content; an unknown tool name gets an error result and a `UserWarning` | kept |
| 6 — the loop is capped | **`max_iterations` defaults to `None`: no cap at all** | **not by default** |

## What it hides

1. **No turn cap by default.** The one parameter that stops a runaway loop is optional and off.
2. **Running out of turns is not an error.** When `max_iterations` is reached the loop just ends,
   and `until_done()` returns the last message — which can be a `tool_use` turn whose tools never
   ran. `tool_loop.run` raises instead; an agent that stopped mid-task should not look finished.
3. **No pre-flight and no budget.** Each turn goes straight to `beta.messages.parse`; nothing
   counts it or checks what it could cost. Usage is on each yielded message, and only a caller
   that iterates sees it — `until_done()` throws every turn's usage away but the last.
4. **The per-turn hook comes after the send.** The loop yields *after* each request, so there is
   no way to veto a request before it goes. There is one way to check the *next* one: call
   `generate_tool_call_response()` during the yield (it runs the tools and caches the result, so
   the runner will not run them twice), build the next request, and stop there if it cannot be
   afforded.
5. **Parallel tool calls run one at a time.** The model can ask for several tools in one turn;
   the sync runner calls them in sequence.
6. **Argument errors lose their detail.** A tool's input is validated with `pydantic.validate_call`,
   and a failure becomes `ValueError("Invalid arguments for function X")` — the reason dropped, so
   the model cannot correct itself from it. `tools.execute_tool` names the bad argument instead.
7. **Appending during the hook takes over the history.** If you call `append_messages` while a
   turn is yielded, the runner skips its own append for that turn — you now own the assistant
   turn and its results too. Easy to half-do.
8. **Schemas come from docstrings unless you say otherwise.** `@beta_tool` derives the JSON schema
   from the signature and the description from the docstring, and sets no `strict`. It does take
   `name`, `description`, `input_schema` and `strict`, so our hand-written schemas can be kept.
9. **It always uses the beta endpoint** (`beta.messages.parse`), even with no output schema, and
   adds an `x-stainless-helper` header naming itself.
10. **Taking the history over runs the tools twice.** `append_messages` clears the cached tool
    results, and after the yield the runner calls `generate_tool_call_response()` again before it
    checks whether the history was taken over, so a turn whose tools already ran in the hook runs
    them a second time. Harmless for a search; a paid or side-effecting tool would be charged or
    applied twice. Found by a test counting tool calls, not by reading.

## What we do about it

`llm.Llm.run_tools` drives the runner rather than letting it drive us. The first request is
counted and admitted before the runner exists; every turn is billed as it is yielded; the next
request is counted and admitted inside the hook, using the tool results the runner cached, so a
turn that cannot be afforded is never sent. The cap is always passed, and exhausting it raises.
Tools are built from `tools.py`'s own schemas, with `strict: true`, dispatched through
`execute_tool` so a bad argument is named. The one thing kept as the runner does it: tools run
one at a time.

One order worth knowing: a turn's tools run *before* the next turn is admitted, because their
results are part of what that turn sends and so of what is counted. Every tool here is free —
local search, arithmetic, a file read — so nothing is spent early; a paid tool would need its own
admission.

Given `finish`, a conversation about to run out — its next turn refused by a ceiling, or the last
under the cap — gets one last turn to answer from what it has, with `finish` after its tool results
and as much output as still fits. That turn is sent directly with the runner's own tool dicts, not
by taking the runner's history over (item 10), and not with `tool_choice: none`, which would
invalidate the cached conversation and re-write all of it at 1.25×.

`entropic/agent.py` is the first agent built on it: the reports as a `search_reports` tool beside
the calculator, thinking on, a $1.00 run ceiling, a $0.40 ceiling per turn, and a 25-turn backstop.
