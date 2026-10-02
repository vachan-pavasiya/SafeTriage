"""SafeTriage — False-positive-aware human-AI fraud triage & self-auditing system.

This package is import-safe: importing it does no I/O and trains nothing. Heavy
work lives in the scripts under ``scripts/`` and the ``safetriage.models`` module.
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Canonical paths. Everything is resolved relative to the project root so the
# package behaves identically no matter what the current working directory is.
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
SPLITS_DIR: Path = DATA_DIR / "splits"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "safetriage" / "models" / "artifacts"
IEEE_RAW_DIR: Path = PROJECT_ROOT / "ieee-fraud-detection"
DB_PATH: Path = DATA_DIR / "safetriage.db"
POLICY_CONFIG_PATH: Path = CONFIG_DIR / "policy_config.yaml"

# Make ``import config.seed`` work regardless of entry point.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _d in (DATA_DIR, SPLITS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_policy_config() -> dict[str, Any]:
    """Load and cache config/policy_config.yaml."""
    with POLICY_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def policy_version() -> str:
    """Stable identifier for the active policy config (name + content hash)."""
    cfg = load_policy_config()
    digest = hashlib.sha256(POLICY_CONFIG_PATH.read_bytes()).hexdigest()[:12]
    return f"{cfg.get('version', 'policy')}+{digest}"


def artifact_version(*paths: Path) -> str:
    """Short content hash across one or more artifact files (for model_version)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        if Path(p).exists():
            h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]
