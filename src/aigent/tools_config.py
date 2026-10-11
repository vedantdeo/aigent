"""Constants for `tools`, in a config module that travels beside it. The caps on files and the
calculator are `underhood.tools.config`'s, beside the implementations they bound.

Not in the package-wide `config`: `tools` imports nothing internal, and that is the property that
makes it liftable into another framework.
"""

from __future__ import annotations

from pathlib import Path

# The only directory `read_file` and `write_file` can reach. Resolved from this file's location, so
# it survives being copied into another checkout but not being moved to a different depth.
SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"

# Web searches per request: the cap that lets a search be priced at all.
MAX_WEB_SEARCHES = 3
