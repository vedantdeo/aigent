"""Constants for `tools`, in a config module that travels beside it.

Not in the package-wide `config`: `tools` imports nothing internal, and that is the property that
makes it liftable into another framework.
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

# Web searches per request: the cap that lets a search be priced at all.
MAX_WEB_SEARCHES = 3
