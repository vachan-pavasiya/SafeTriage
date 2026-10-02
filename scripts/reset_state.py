"""Delete every derived artifact so the next build starts from a clean slate.

Removes: model weights + calibrators + feature encoders, the train/val/test
split index files, the SQLite database, generated datasets, and every
evaluation / simulation / calibration JSON + PNG.

Leaves untouched: the raw IEEE-CIS input under ieee-fraud-detection/, all source
code, and config/.

    python scripts/reset_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import reset_state

if __name__ == "__main__":
    reset_state()
    print("Clean. Next: python scripts/build_splits.py   (or run_pipeline.py --force)")
