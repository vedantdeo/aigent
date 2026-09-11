# Entropic — notes for Claude

Four project rules live here rather than in a session's private memory, so they travel with the
repo and reach anyone who clones it.

One of them — the constants rule — is a verbatim mirror of a global preference in
`~/.claude/CLAUDE.md`, which is machine-local and backed up by nothing. The copy below is the durable
one. Edit both, or neither.

## Read the knowledge graph before exploring

`docs/knowledge-graph.md` maps every module, symbol, edge and invariant in this codebase, plus the
seams where each upcoming roadmap week attaches. Read it first when picking up work here — it is
faster than re-reading the tree, and it records conventions that are invisible from a quick skim.

The load-bearing ones, as a taste of what is in there: every tool result for a turn goes back in
*one* user message; tool errors return as `tool_result` with `is_error: true` and never as
exceptions; every paid call is pre-flighted through `pricing.check_request`; `config` is the only
module that builds a client, and `pricing` the only one that knows a price.

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

## Constants and configuration

**The rule: tunable constants live in one config module per project** — `config.py`, `settings.py`,
or whatever the project already calls it — rather than scattered beside the code that uses them.
Qualify names by call site when they would otherwise collide: `MAX_TOKENS_CHAT`, not a local
`MAX_TOKENS` in every module.

- Move the settings: limits, caps, timeouts, retries, defaults, model names, spending ceilings.
- Leave behind what is not a setting: values derived at import from `__file__` (those are locations,
  not knobs), and private algorithm guards nobody would tune from outside.

**Why:** a number sitting alone in its own module is invisible. Side by side the set reads as a set —
you can see which call is the expensive one, notice a value that drifted from its neighbours, and
change a policy in one edit instead of six.

**The fallback, when one config module stops working: split the config — never scatter the constants
back into the code.** Two seams are legitimate:

- **By concern**, once the file is genuinely too big to read: settings in one module, the arithmetic
  that consumes them in another.
- **By isolation**, when a module is deliberately free of internal imports so it can be lifted into
  another project or framework unchanged. It gets its own config module travelling beside it, not a
  dependency on the package-wide one. Both properties then survive — one place per concern, and no
  import. Say so in that module's docstring.

Don't split pre-emptively, don't split alphabetically, and never split into one config per consumer.
The fallback exists so the rule survives a hard case, not so it can be worked around.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss of
this machine. Edit both, or neither.

**In this repo**, concretely — the rule and both seams already have worked examples:

- The one config module is `src/entropic/config.py`: models, spending ceilings, output caps
  (`MAX_TOKENS_CHAT` and friends), the agent turn cap, the report's failure cap.
- *Split by concern:* `pricing.py` holds the `Price`/`Budget` classes and the cost arithmetic;
  `config.py` kept the settings they read. It happened when `config.py` got too big to read, not
  before.
- *Split by isolation:* `tools.py` imports nothing from this package, which is the whole reason it
  can be lifted into the SDK tool runner in Week 5 and a LangGraph node in Week 6. Its constants live
  in `tools_config.py`, travelling beside it. Not in `config.py`, which would cost `tools` its
  independence; not loose in `tools.py`, which would cost the one-place rule.
- Staying put: `report.REPORTS_DIR`, a location computed from `__file__` rather than a knob.

## Turn on branch protection before anyone else can touch this repo

**Trigger: the moment a collaborator is added, or the repo goes public — planned for the `v0.1-rag`
tag at the end of Week 2 (Fri 2026-09-18). Raise it then; do not wait to be asked.**

CI is a detector, not a gate. GitHub accepts a push first and runs the workflows after, so a bad
commit reaches `main` and goes red a minute later. That is fine while this is a one-person repo and
the loop is push-look-fix. It stops being fine the moment someone else can pull a red commit or push
their own.

What to enable on `main` at that point: require a pull request, require both checks to pass, and
block force pushes and deletion. The two required checks are matched by *job* name, not workflow
name, so they are exactly:

- `lint · types · tests` (from `.github/workflows/checks.yml`)
- `graph moves with the code` (from `.github/workflows/knowledge-graph.yml`)

Settings > Rules > Rulesets on GitHub, or `gh api` against the branch-protection endpoint. Surface
the exact call before running it — it changes who can write to the repo.
