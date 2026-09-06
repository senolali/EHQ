"""Build a knowable-question capability probe from the released CCQ items.

RQ1 needs a capability score measured independently of EHQ. Public benchmark
numbers are collected under different prompts, sampling, and model builds than
this protocol, so this tool derives the score from the same panel, the same
endpoints, and the same inference settings instead.

CCQ items already carry a synthetic document whose answer-bearing span was
replaced by a single redaction token, plus the withheld span itself. Restoring
that span produces a document that *contains* the answer, so the question
becomes answerable by any model regardless of its knowledge cutoff. Accuracy on
the restored set therefore measures whether a model can deliver a correct answer
when the information is available to it, which is exactly the contrast RQ1 needs
against "does it know what it does not know".

Only items whose withheld span is a short, self-contained value are kept. A span
such as ``the inaugural flight to Greenville, scheduled for August 15, 2026`` is
a clause rather than an answer: grading a response against it by containment
would accept ``the Midwest`` or any other fragment. Those items are reported and
skipped rather than graded unreliably.

    python tools/build_capability_probe.py \
        --dataset data/releases/EHQ-3000.json \
        --output  data/releases/EHQ-capability-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import write_json  # noqa: E402
from ehq.constants import REDACTION_TOKEN  # noqa: E402
from ehq.evaluation.correctness import normalize_text  # noqa: E402
from ehq.hashing import sha256_file  # noqa: E402

PROBE_CATEGORY = "CAP"
SCHEMA_VERSION = "capability-probe-1.0"


def _crisp(value: str, *, max_words: int) -> bool:
    """A withheld span usable as a gold answer on its own."""

    stripped = value.strip()
    if not stripped or stripped == REDACTION_TOKEN:
        return False
    return len(stripped.split()) <= max_words


def build_probe(
    dataset_path: Path, *, max_words: int = 4
) -> tuple[Dict[str, Any], Dict[str, int]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, Mapping) else raw

    counts = {
        "ccq_total": 0,
        "kept": 0,
        "skipped_span_not_crisp": 0,
        "skipped_redaction_not_unique": 0,
        "skipped_answer_leaks_into_question": 0,
        "skipped_restore_failed": 0,
    }
    probe_items: List[Dict[str, Any]] = []

    for item in items:
        if str(item.get("category", "")).upper() != "CCQ":
            continue
        counts["ccq_total"] += 1
        document = str(item.get("document") or "")
        question = str(item.get("question") or "").strip()
        answer = str(item.get("redacted_value") or "").strip()

        if document.count(REDACTION_TOKEN) != 1:
            counts["skipped_redaction_not_unique"] += 1
            continue
        if not _crisp(answer, max_words=max_words):
            counts["skipped_span_not_crisp"] += 1
            continue
        # The question must not already contain the answer, or the probe would
        # measure copying rather than retrieval from the document.
        if normalize_text(answer) and normalize_text(answer) in normalize_text(question):
            counts["skipped_answer_leaks_into_question"] += 1
            continue

        restored = document.replace(REDACTION_TOKEN, answer)
        if REDACTION_TOKEN in restored or normalize_text(answer) not in normalize_text(
            restored
        ):
            counts["skipped_restore_failed"] += 1
            continue

        source_id = str(item.get("question_id"))
        probe_items.append(
            {
                "question_id": f"{PROBE_CATEGORY}-{source_id}",
                "category": PROBE_CATEGORY,
                "subcategory": str(item.get("subcategory") or PROBE_CATEGORY),
                "document": restored,
                "question": question,
                "correct_answer": answer,
                "acceptable_answers": [],
                "expected_knowability": 1,
                "source_question_id": source_id,
                "schema_version": SCHEMA_VERSION,
            }
        )
        counts["kept"] += 1

    probe = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "RQ1 capability probe: same panel, same protocol, answer present in "
            "the supplied document"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_dataset": str(dataset_path),
            "source_dataset_sha256": sha256_file(dataset_path),
            "builder": "build_capability_probe.py",
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "selection_rule": (
                "CCQ items with exactly one redaction token whose withheld span "
                f"is at most {max_words} words, does not appear in the question, "
                "and is recoverable in the restored document"
            ),
            "max_answer_words": max_words,
            "counts": counts,
        },
        "items": probe_items,
    }
    return probe, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-answer-words",
        type=int,
        default=4,
        help="Longest withheld span still treated as a self-contained answer",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=100,
        help="Refuse to write a probe smaller than this",
    )
    args = parser.parse_args()

    probe, counts = build_probe(args.dataset, max_words=args.max_answer_words)
    if counts["kept"] < args.min_items:
        raise SystemExit(
            f"Only {counts['kept']} usable item(s); a capability score from that "
            f"few is too noisy. Relax --max-answer-words or lower --min-items "
            "deliberately."
        )
    write_json(args.output, probe)
    print(json.dumps({"output": str(args.output), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
