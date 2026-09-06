"""Dataset JSON I/O with support for legacy EHQ-750 envelopes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = value.get("questions") or value.get("data") or value.get("items")
    else:
        items = None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Unsupported dataset envelope: {path}")
    return items


def write_dataset(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(items), ensure_ascii=False, indent=2, sort_keys=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
