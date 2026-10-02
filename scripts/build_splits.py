"""Step 1 of training: build the fresh 60/20/20 stratified train/val/test split.

For each dataset it ensures the processed data exists (IEEE preprocessing /
synthetic generation), then writes deterministic, seeded split index files to
data/splits/ and prints the per-split counts and fraud prevalence.

    python scripts/build_splits.py                # both, IEEE first
    python scripts/build_splits.py ieee           # just one

The test indices written here are consumed ONLY by safetriage/evaluate.py and
safetriage/audit.py. Training, feature-fitting, tuning and calibration use
train / val only (enforced in safetriage/data/splits.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.seed import SEED, set_all_seeds
from safetriage import DATA_DIR
from safetriage.data import generate_synthetic_dataset, preprocess_ieee
from safetriage.data.splits import make_splits, split_summary

ORDER = ("ieee", "synthetic")


def _ensure_data(dataset: str) -> pd.DataFrame:
    if dataset == "ieee":
        if not (DATA_DIR / "processed_ieee.parquet").exists():
            preprocess_ieee.main()
        return pd.read_parquet(DATA_DIR / "processed_ieee.parquet")
    if not (DATA_DIR / "synthetic.csv").exists():
        generate_synthetic_dataset.main()
    return pd.read_csv(DATA_DIR / "synthetic.csv")


def main() -> None:
    set_all_seeds(SEED)
    targets = [a for a in sys.argv[1:] if not a.startswith("-")] or list(ORDER)
    for ds in targets:
        df = _ensure_data(ds)
        y = df["isFraud"].to_numpy()
        make_splits(ds, y, overwrite=True)
        summ = split_summary(ds, y)
        print(f"\n[{ds}] seed={SEED}  ratios={summ['ratios']}")
        for w, s in summ["splits"].items():
            print(f"   {w:<5} n={s['n']:>7}  fraud={s['n_positive']:>5}  prevalence={s['prevalence']}")
        (DATA_DIR / f"split_summary_{ds}.json").write_text(json.dumps(summ, indent=2))
    print("\nSplits written to data/splits/. Next: python -m safetriage.models.train --epochs 20")


if __name__ == "__main__":
    main()
