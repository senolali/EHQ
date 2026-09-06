"""Append-only JSONL checkpoints with experiment-fingerprint protection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, Set, Tuple


HEADER_KEY = "_ehq_checkpoint"


class Checkpoint:
    def __init__(self, path: Path, *, fingerprint: str | None = None):
        self.path = path
        self.fingerprint = fingerprint
        self._lock = Lock()
        self._completed: Set[Tuple[str, str]] = set()
        if path.exists():
            raw = list(self._raw_records())
            if raw and HEADER_KEY in raw[0]:
                existing = str(raw[0][HEADER_KEY].get("fingerprint") or "")
                if fingerprint is not None and existing != fingerprint:
                    raise ValueError(
                        f"Checkpoint fingerprint mismatch for {path}: "
                        f"expected {fingerprint}, found {existing or '<missing>'}"
                    )
            elif fingerprint is not None and raw:
                raise ValueError(
                    f"Legacy checkpoint has no experiment fingerprint: {path}"
                )
            for record in raw:
                if HEADER_KEY in record:
                    continue
                key = (str(record.get("model")), str(record.get("question_id")))
                self._completed.add(key)

    def _raw_records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid checkpoint JSON at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Checkpoint record must be an object at line {line_number}"
                    )
                yield value

    def records(self) -> Iterator[Dict[str, Any]]:
        for value in self._raw_records():
            if HEADER_KEY not in value:
                yield value

    def contains(self, model: str, question_id: str) -> bool:
        return (model, question_id) in self._completed

    def append(self, record: Dict[str, Any]) -> None:
        key = (str(record.get("model")), str(record.get("question_id")))
        with self._lock:
            if key in self._completed:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                if stream.tell() == 0 and self.fingerprint is not None:
                    header = {
                        HEADER_KEY: {
                            "schema_version": "1.0",
                            "fingerprint": self.fingerprint,
                        }
                    }
                    stream.write(
                        json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._completed.add(key)
