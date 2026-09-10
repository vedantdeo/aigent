#!/bin/sh
# Single source of truth for one rule: a change to the code the knowledge graph
# describes must carry the graph update with it. The pre-commit hook and CI both
# call this, so the rule cannot drift between them.
#
# Reads changed paths on stdin, one per line.
#   usage:  <paths  scripts/check-knowledge-graph.sh [commit|ci]

GRAPH="docs/knowledge-graph.md"
WATCHED='^(src/|pyproject\.toml$)'
mode="${1:-ci}"

changed=$(cat)
watched=$(printf '%s\n' "$changed" | grep -E "$WATCHED")

# Nothing the graph records changed.
[ -z "$watched" ] && exit 0

# The graph moved with it.
printf '%s\n' "$changed" | grep -qx "$GRAPH" && exit 0

if [ "$mode" = "commit" ]; then
  headline="commit blocked — the knowledge graph is not staged with this change."
  label="Staged files the graph describes:"
  fix="    git add $GRAPH

  If this change genuinely records nothing in the graph:

    git commit --no-verify"
else
  headline="the knowledge graph is out of step with this change."
  label="Files the graph describes, changed here:"
  fix="  Update $GRAPH in this branch — the rows this change moves, and the
  \"verified against\" line at the top — and push again."
fi

cat >&2 <<EOF

  $headline

  $label
$(printf '%s\n' "$watched" | sed 's/^/    /' | head -12)

  $GRAPH is unchanged. Update the rows this change moves — nodes, edges,
  contracts, invariants, seams — and bump the "verified against" line at the top.

$fix

EOF
exit 1
