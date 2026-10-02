"""Deterministic, enforced train / validation / test splitting.

Rules (see EVAL_PROTOCOL.md):

* The split is 60 / 20 / 20, stratified by the binary label, seeded with
  ``config.seed.SEED``. It is written once to ``data/splits/`` and reused.
* The TEST split is sacred. Training, calibration fitting and threshold tuning
  must call :func:`load_trainable_indices` (which refuses ``"test"``).
* Only :mod:`safetriage.evaluate` and :mod:`safetriage.audit` may call
  :func:`load_test_indices`, and they call :func:`assert_no_test_leakage` on the
  indices any *trainable* step used, as a defence-in-depth check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.seed import SEED
from safetriage import SPLITS_DIR

RATIOS = (0.60, 0.20, 0.20)  # train, val, test
_WHICH = ("train", "val", "test")


def _path(dataset: str, which: str) -> Path:
    return SPLITS_DIR / f"{dataset}_{which}.idx"


def make_splits(dataset: str, labels: pd.Series | np.ndarray, *, overwrite: bool = False) -> dict[str, np.ndarray]:
    """Create (or load) the stratified split for ``dataset``.

    ``labels`` is the full label vector in the dataset's canonical row order.
    Returns a dict {"train": idx, "val": idx, "test": idx} of positional indices.
    """
    if all(_path(dataset, w).exists() for w in _WHICH) and not overwrite:
        return {w: load_indices(dataset, w) for w in _WHICH}

    y = np.asarray(labels)
    all_idx = np.arange(len(y))

    train_idx, tmp_idx = train_test_split(
        all_idx, train_size=RATIOS[0], random_state=SEED, stratify=y
    )
    val_rel = RATIOS[1] / (RATIOS[1] + RATIOS[2])
    val_idx, test_idx = train_test_split(
        tmp_idx, train_size=val_rel, random_state=SEED, stratify=y[tmp_idx]
    )

    splits = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    for w, idx in splits.items():
        np.savetxt(_path(dataset, w), idx, fmt="%d")
    return splits


def load_indices(dataset: str, which: str) -> np.ndarray:
    if which not in _WHICH:
        raise ValueError(f"which must be one of {_WHICH}, got {which!r}")
    p = _path(dataset, which)
    if not p.exists():
        raise FileNotFoundError(f"Split file missing: {p}. Run the data pipeline first.")
    return np.loadtxt(p, dtype=int, ndmin=1)


def load_trainable_indices(dataset: str, which: str) -> np.ndarray:
    """Load a split for a step that must never see test data."""
    if which == "test":
        raise PermissionError(
            "load_trainable_indices() refuses 'test'. Training, calibration and "
            "threshold tuning may only use 'train' or 'val'."
        )
    return load_indices(dataset, which)


def load_test_indices(dataset: str) -> np.ndarray:
    """Load the held-out test split. Only evaluate.py and audit.py may call this."""
    return load_indices(dataset, "test")


def assert_no_test_leakage(dataset: str, used_indices: np.ndarray | list[int]) -> None:
    """Raise if any index used by a trainable step is in the test split."""
    test = set(load_indices(dataset, "test").tolist())
    overlap = test.intersection(np.asarray(used_indices).tolist())
    if overlap:
        raise AssertionError(
            f"TEST LEAKAGE for dataset {dataset!r}: {len(overlap)} test rows were "
            f"used by a training/calibration step (e.g. index {next(iter(overlap))})."
        )


def split_summary(dataset: str, labels: pd.Series | np.ndarray) -> dict:
    """Human-readable counts + prevalence per split (for EVAL_PROTOCOL / logs)."""
    y = np.asarray(labels)
    out: dict = {"dataset": dataset, "seed": SEED, "ratios": RATIOS, "splits": {}}
    for w in _WHICH:
        idx = load_indices(dataset, w)
        out["splits"][w] = {
            "n": int(len(idx)),
            "n_positive": int(y[idx].sum()),
            "prevalence": round(float(y[idx].mean()), 5),
        }
    return out
