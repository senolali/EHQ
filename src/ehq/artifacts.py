"""Atomic run-artifact writes and content-hash verification."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .hashing import sha256_file


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def write_artifact_catalog(run_dir: Path) -> Dict[str, Any]:
    catalog_path = run_dir / "artifact_catalog.json"
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path != catalog_path
    )
    catalog = {
        "schema_version": "1.0",
        "artifacts": {
            path.relative_to(run_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    write_json(catalog_path, catalog)
    return catalog


def verify_run_artifacts(run_dir: Path) -> Dict[str, Any]:
    catalog_path = run_dir / "artifact_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected = catalog.get("artifacts")
    if not isinstance(expected, Mapping):
        raise ValueError("Invalid artifact catalog")
    missing = []
    mismatched = []
    for relative, metadata in expected.items():
        path = run_dir / str(relative)
        if not path.is_file():
            missing.append(str(relative))
            continue
        expected_hash = metadata.get("sha256") if isinstance(metadata, Mapping) else None
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            mismatched.append(
                {"path": str(relative), "expected": expected_hash, "actual": actual_hash}
            )
    unexpected = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != catalog_path
        and path.relative_to(run_dir).as_posix() not in expected
    )
    return {
        "valid": not missing and not mismatched and not unexpected,
        "n_expected": len(expected),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
    }
