"""Single source of truth for the global random seed.

Every stochastic step in SafeTriage — synthetic data generation, the
train/val/test split, model training, calibration, and audit sampling — imports
SEED from here. Do not hardcode a seed anywhere else.
"""

from __future__ import annotations

import os
import random

SEED: int = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and hash randomisation. Call at the top of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy always present in practice; guard for safety
        pass
