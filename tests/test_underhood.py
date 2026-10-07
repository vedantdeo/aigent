"""aigent reaches underhood's GPU pricing through the installed package, without loading torch.

The import runs in a fresh interpreter: aigent has torch, and pytest may have loaded it already.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import sys
from underhood import config
from underhood.gpu.pricing import estimated_seconds, hourly_usd, rental_usd
seconds = estimated_seconds(30_000_000, 655_360_000, config.GPU_PEAK_FLOPS, config.GPU_MFU)
print(rental_usd(seconds, hourly_usd(config.GPU_OFFER, config.GPU_OFFERS, config.GPU_TAX)))
print(",".join(m for m in ("torch", "transformers", "mlx") if m in sys.modules))
"""


def test_a_gpu_run_is_priced_through_underhood_without_loading_torch() -> None:
    usd, loaded = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    assert 0 < float(usd) < 5, f"${usd} for a 30M-parameter run on underhood's default A100 offer"
    assert loaded == "", f"pricing a run loaded {loaded}"
