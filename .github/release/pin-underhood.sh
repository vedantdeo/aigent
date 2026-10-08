#!/bin/sh
# Point the underhood source at $1, such as `tag = "stable"`, re-lock, and print the commit pinned.
set -eu
line="underhood = { git = \"$UNDERHOOD\", $1 }"
sed -i "s|^underhood = { git = \"$UNDERHOOD\", .* }\$|$line|" pyproject.toml
grep -qxF "$line" pyproject.toml
uv lock --upgrade-package underhood >&2
"$(dirname "$0")/pinned-underhood.sh" < uv.lock
