#!/bin/sh
# Point the cltrplusf source at $1, such as `tag = "stable"`, re-lock, and print the commit pinned.
set -eu
line="cltrplusf = { git = \"$CLTRPLUSF\", $1 }"
sed -i "s|^cltrplusf = { git = \"$CLTRPLUSF\", .* }\$|$line|" pyproject.toml
grep -qxF "$line" pyproject.toml
uv lock --upgrade-package cltrplusf >&2
"$(dirname "$0")/pinned-cltrplusf.sh" < uv.lock
