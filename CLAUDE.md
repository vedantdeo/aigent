# Entropic — notes for Claude

Two project rules live here rather than in a session's private memory, so they travel with the repo
and reach anyone who clones it.

## Read the knowledge graph before exploring

`docs/knowledge-graph.md` maps every module, symbol, edge and invariant in this codebase, plus the
seams where each upcoming roadmap week attaches. Read it first when picking up work here — it is
faster than re-reading the tree, and it records conventions that are invisible from a quick skim.

The load-bearing ones, as a taste of what is in there: every tool result for a turn goes back in
*one* user message; tool errors return as `tool_result` with `is_error: true` and never as
exceptions; every paid call is pre-flighted through `config.check_request`; `config` is the only
module that builds a client or knows a price.

Section 4 of that file (invariants) matters more than the `file:line` anchors in section 2 — anchors
drift, invariants do not.

## Every commit updates the knowledge graph

A commit that changes what the graph records — a module, symbol or edge; a contract, price, default
or ceiling; an invariant or a consumed seam — updates `docs/knowledge-graph.md` in the *same commit*.
Bump the "verified against" line at the top, and re-check the anchors for any file whose line numbers
moved.

**Why:** a graph that lags the code is worse than none. The next session trusts a stale anchor or a
retired invariant and works from a map of a repo that no longer exists.

**How to apply:** treat the doc edit as part of the change, not a follow-up — stage it with the code,
and say in the commit summary that the graph moved with it. If a change genuinely records nothing in
the graph, say so explicitly rather than skipping quietly.

This is enforced, not just asked for: `.githooks/pre-commit` refuses a commit that stages `src/` or
`pyproject.toml` without the graph, and `.github/workflows/knowledge-graph.yml` runs the same rule in
CI for clones that never ran `git config core.hooksPath .githooks`. Both call
`scripts/check-knowledge-graph.sh`, so the rule cannot drift between them. `git commit --no-verify`
is the deliberate exception — use it only for a change the graph truly does not record, and say why.
