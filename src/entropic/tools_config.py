"""Constants for `tools.py`, kept beside it rather than in the package-wide `config.py`.

`tools` imports nothing else from this package — not `config`, not `pricing` — which is what lets it
be lifted into Week 5's SDK tool runner or Week 6's LangGraph node by copying two files and nothing
else. Folding these three numbers into `config` would trade that property for tidiness, so they get
a config module of their own: one that travels with the code it configures.

Move the pair together, or move neither.
"""

from __future__ import annotations

from pathlib import Path

# The only directory `read_file` can reach. Resolved from this file's location, so it survives being
# copied into another checkout but not being moved to a different depth in the tree.
SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"

# Cap on a single read. Interpolated into the tool description, so the model is told the limit
# rather than discovering it by having its output silently cut.
MAX_FILE_READ_CHARS = 20_000

# Guard on `**` in the calculator. `2 ** 10` is arithmetic; `2 ** 10_000_000` is a way to hang the
# process from a tool argument.
MAX_EXPONENT = 1000.0
