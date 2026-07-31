"""utils/checkpoint.py — item-level checkpoint / resume for evaluation runs.

Ported from senolali/RQEval (utils/checkpoint.py) essentially unchanged
-- this logic is dataset-agnostic. Only item_key() was adapted: RQEval
keys on (dataset, question_id), EHQ items don't have a "dataset" field
so it keys on (category, question_id) instead.

Problem this solves:
  A full EHQ-3000 run is 3000 items x 20 models x 2 API calls (answer +
  confidence) = 120,000 calls. If the process crashes at item 2500
  (API outage, quota, power cut), everything collected in RAM is lost
  and the model restarts from item 0.

Solution:
  Every completed per-item result (classified response_type, is_correct,
  confidence, rubric_score, raw text) is appended to a JSONL file on
  disk immediately after it finishes. On the next run, existing entries
  are loaded and those items are skipped -- only the missing ones hit
  the API.

Design notes:
  - One JSONL file per (experiment_name, model). Stored under a STABLE
    path (outputs/checkpoints/<experiment_name>/<model>.jsonl),
    independent of the timestamped experiment id, so reruns can find it.
  - The first line is a fingerprint header (dataset path, categories,
    model params, seed). If the current run's fingerprint does not
    match, the checkpoint is ignored (renamed *.stale) -- prevents
    mixing results from a different experimental setup (e.g. dataset
    regenerated, model temperature changed).
  - Item keys combine category, question_id, and an md5 of the question
    text, so a key never silently maps to a different question.
  - Appends are flushed + fsync'd per line; a crash mid-write loses at
    most one line (malformed trailing lines are skipped on load).
  - To force a completely fresh run, delete the checkpoint directory or
    set experiment.checkpoint: false in the config.
"""

import os
import json
import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HEADER_MARK = "__checkpoint_header__"


def item_key(item: Dict[str, Any], idx: int) -> str:
    """Stable, collision-safe key for a dataset item."""
    qhash = hashlib.md5(str(item.get("question", "")).encode("utf-8")).hexdigest()[:10]
    return f"{item.get('category', 'unknown')}::{item.get('question_id', idx)}::{qhash}"


class ItemCheckpoint:
    """Append-only JSONL checkpoint for per-item evaluation results."""

    def __init__(self, path: str, fingerprint: Dict[str, Any], enabled: bool = True):
        self.path        = path
        self.fingerprint = fingerprint
        self.enabled     = enabled
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._load()

    # ------------------------------------------------------------------
    def _fp_digest(self) -> str:
        return hashlib.md5(
            json.dumps(self.fingerprint, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        stale = False
        loaded = {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                first = f.readline()
                try:
                    header = json.loads(first)
                except json.JSONDecodeError:
                    header = {}
                if (header.get(_HEADER_MARK) is not True
                        or header.get("digest") != self._fp_digest()):
                    stale = True
                else:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            loaded[rec["key"]] = rec["entry"]
                        except (json.JSONDecodeError, KeyError):
                            # crash-truncated trailing line — safe to skip
                            continue
        except OSError as e:
            logger.warning(f"[Checkpoint] Could not read {self.path}: {e}")
            return

        if stale:
            stale_path = self.path + ".stale"
            try:
                os.replace(self.path, stale_path)
            except OSError:
                pass
            logger.warning(
                f"[Checkpoint] Fingerprint mismatch (config/dataset changed) — "
                f"existing checkpoint moved to {stale_path}; starting fresh."
            )
            return

        self._entries = loaded
        if loaded:
            logger.info(
                f"[Checkpoint] Resumed {len(loaded)} completed items "
                f"from {self.path}"
            )

    def _ensure_writer(self) -> None:
        if self._fh is not None:
            return
        new_file = not os.path.exists(self.path)
        self._fh = open(self.path, "a", encoding="utf-8")
        if new_file:
            header = {_HEADER_MARK: True,
                      "digest": self._fp_digest(),
                      "fingerprint": self.fingerprint}
            self._fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            self._fh.flush()

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(key) if self.enabled else None

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, key: str, entry: Dict[str, Any]) -> None:
        """Persist one completed item (flushed immediately)."""
        if not self.enabled or entry is None:
            return
        self._entries[key] = entry
        try:
            self._ensure_writer()
            self._fh.write(json.dumps({"key": key, "entry": entry},
                                      ensure_ascii=False) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except OSError as e:
            logger.warning(f"[Checkpoint] Write failed ({e}) — continuing without persistence.")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
