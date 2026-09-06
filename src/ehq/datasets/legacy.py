"""Auditable normalization of the legacy EHQ-750 dataset."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..evaluation.correctness import normalize_text
from ..hashing import sha256_json


def canonicalize_legacy_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    category = str(item.get("category", "")).upper()
    correct_answer = str(item.get("correct_answer") or "")
    if category == "FEQ" and not correct_answer:
        correct_answer = "[DOES_NOT_EXIST]"
    source_evidence = []
    answer_source = item.get("answer_source")
    if isinstance(answer_source, str) and answer_source.strip():
        source_evidence.append(
            {
                "source_id": f"legacy::{item.get('question_id')}",
                "url": answer_source if answer_source.startswith("http") else "",
                "publisher": "legacy-source",
                "title": "Legacy answer source",
                "published_at": None,
                "verified_at": "legacy-import",
                "evidence": answer_source,
            }
        )
    return {
        "schema_version": "1.0",
        "question_id": str(item.get("question_id") or item.get("id")),
        "category": category,
        "subcategory": str(item.get("subcategory") or category),
        "question": str(item.get("question") or "").strip(),
        "correct_answer": correct_answer,
        "acceptable_answers": list(item.get("acceptable_answers") or []),
        "expected_knowability": 0,
        "difficulty": item.get("difficulty"),
        "source_evidence": source_evidence,
        "legacy_metadata": {
            key: deepcopy(value)
            for key, value in item.items()
            if key
            not in {
                "question_id",
                "id",
                "category",
                "subcategory",
                "question",
                "correct_answer",
                "acceptable_answers",
                "difficulty",
                "answer_source",
            }
        },
        "provenance": {
            "origin": "EHQ_750_dataset_v2_verified.json",
            "generator": {
                "name": "legacy-import",
                "requested_model": None,
                "resolved_model": None,
                "prompt_sha256": None,
                "response_sha256": None,
            },
        },
        "qc": {
            "passed": True,
            "checks": {"legacy_import": True},
            "notes": [],
        },
    }


def migrate_legacy_seed(
    items: Iterable[Mapping[str, Any]],
    *,
    retained_categories: Sequence[str] = ("FEQ", "HNQ"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    retained = {category.upper() for category in retained_categories}
    output: List[Dict[str, Any]] = []
    seen = {}
    dropped = []
    excluded = []
    for item in items:
        canonical = canonicalize_legacy_item(item)
        if canonical["category"] not in retained:
            excluded.append(
                {
                    "question_id": canonical["question_id"],
                    "category": canonical["category"],
                    "reason": "category_rebuilt_under_new_protocol",
                }
            )
            continue
        identity = sha256_json(
            {
                "category": canonical["category"],
                "subcategory": canonical["subcategory"],
                "question": normalize_text(canonical["question"]),
                "correct_answer": normalize_text(canonical["correct_answer"]),
            }
        )
        if identity in seen:
            dropped.append(
                {
                    "question_id": canonical["question_id"],
                    "duplicate_of": seen[identity],
                    "identity_sha256": identity,
                }
            )
            continue
        seen[identity] = canonical["question_id"]
        output.append(canonical)
    report = {
        "migration_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_count": len(output) + len(dropped) + len(excluded),
        "retained_count": len(output),
        "dropped_exact_duplicates": dropped,
        "excluded_for_rebuild": excluded,
        "retained_category_counts": {
            category: sum(item["category"] == category for item in output)
            for category in sorted(retained)
        },
    }
    return output, report
