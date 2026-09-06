"""Minimal local .env loading without exposing or logging secret values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_local_env(path: Path, *, override: bool = False) -> Dict[str, bool]:
    """Load simple KEY=VALUE entries and report presence only."""
    loaded: Dict[str, bool] = {}
    if not path.exists():
        return loaded
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed .env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid .env key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = bool(os.environ.get(key))
    return loaded
