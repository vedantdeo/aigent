"""aigent: a personal AI agent, built up one capability at a time over eight weeks."""

import os
from pathlib import Path

# cltrplusf reads its corpus from here: this checkout's corpus/, wherever aigent is run from.
os.environ.setdefault("CLTRPLUSF_HOME", str(Path(__file__).resolve().parents[2]))
