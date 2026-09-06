"""Corpus-level audit for source-ledger question datasets."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..evaluation.correctness import normalize_text
from ..hashing import sha256_file, sha256_json
from .ledger import validate_fact_ledger
from .validate import validate_dataset


def audit_ledger_dataset(
    items: Iterable[Mapping[str, Any]],
    *,
    ledger: Mapping[str, Any],
    category: str,
    subcategories: Sequence[str],
    attempts: Sequence[Mapping[str, Any]] = (),
    expected_per_subcategory: int | None = None,
    question_similarity_threshold: float = 0.96,
    require_release_evidence: bool = False,
    snapshot_root: Path | None = None,
) -> Dict[str, Any]:
    rows = list(items)
    expected_total = (
        expected_per_subcategory * len(subcategories)
        if expected_per_subcategory is not None
        else len(rows)
    )
    facts = ledger.get("facts")
    facts = facts if isinstance(facts, list) else []
    selected_ids = {str(item.get("question_id") or "") for item in rows}
    selected_facts = [
        fact
        for fact in facts
        if isinstance(fact, Mapping) and str(fact.get("fact_id") or "") in selected_ids
    ]
    ledger_subset = {**dict(ledger), "facts": selected_facts}
    ledger_validation = validate_fact_ledger(
        ledger_subset,
        expected_total=expected_total,
        expected_per_subcategory=expected_per_subcategory,
        expected_subcategories=subcategories,
        require_release_evidence=require_release_evidence,
    )
    validation = validate_dataset(
        rows,
        expected_total=expected_total,
        require_source_evidence=True,
    ).to_dict()

    fact_by_id = {
        str(fact.get("fact_id") or ""): fact
        for fact in selected_facts
        if isinstance(fact, Mapping)
    }
    subcategory_counts = Counter(str(item.get("subcategory") or "") for item in rows)
    ledger_mismatches: List[Dict[str, Any]] = []
    snapshot_mismatches: List[Dict[str, Any]] = []
    requested_models = Counter()
    resolved_models = Counter()
    unresolved_items = []
    source_ids = Counter()
    publishers = Counter()
    event_dates: List[str] = []

    for item in rows:
        item_id = str(item.get("question_id") or "")
        fact = fact_by_id.get(item_id)
        if fact is None:
            ledger_mismatches.append(
                {"question_id": item_id, "code": "missing_ledger_fact"}
            )
            continue
        provenance = item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        if provenance.get("ledger_fact_sha256") != sha256_json(fact):
            ledger_mismatches.append(
                {"question_id": item_id, "code": "ledger_fact_hash_mismatch"}
            )
        for fact_field, item_field in (
            ("subcategory", "subcategory"),
            ("event_date", "event_date"),
            ("gold_answer", "correct_answer"),
        ):
            if item.get(item_field) != fact.get(fact_field):
                ledger_mismatches.append(
                    {"question_id": item_id, "code": f"{fact_field}_mismatch"}
                )
        generator = provenance.get("generator")
        generator = generator if isinstance(generator, Mapping) else {}
        requested_models[str(generator.get("requested_model") or "")] += 1
        resolved = str(generator.get("resolved_model") or "")
        resolved_models[resolved] += 1
        if not resolved:
            unresolved_items.append(item_id)
        event_date = str(item.get("event_date") or "").split("/")[0]
        if event_date:
            event_dates.append(event_date)
        for source in item.get("source_evidence") or []:
            if not isinstance(source, Mapping):
                continue
            source_ids[str(source.get("source_id") or "")] += 1
            publishers[str(source.get("publisher") or "")] += 1
            if (
                require_release_evidence
                and snapshot_root is not None
                and source.get("snapshot_path")
            ):
                snapshot_path = snapshot_root / str(source["snapshot_path"])
                expected_hash = str(source.get("snapshot_sha256") or "").casefold()
                if not snapshot_path.is_file():
                    snapshot_mismatches.append(
                        {
                            "question_id": item_id,
                            "source_id": source.get("source_id"),
                            "code": "snapshot_missing",
                        }
                    )
                elif sha256_file(snapshot_path) != expected_hash:
                    snapshot_mismatches.append(
                        {
                            "question_id": item_id,
                            "source_id": source.get("source_id"),
                            "code": "snapshot_hash_mismatch",
                        }
                    )

    normalized_questions = [
        normalize_text(str(item.get("question") or "")) for item in rows
    ]
    near_duplicate_questions: List[Dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            similarity = SequenceMatcher(
                None,
                normalized_questions[left_index],
                normalized_questions[right_index],
            ).ratio()
            if similarity >= question_similarity_threshold:
                near_duplicate_questions.append(
                    {
                        "left_id": left.get("question_id"),
                        "right_id": rows[right_index].get("question_id"),
                        "similarity": round(similarity, 6),
                    }
                )
    near_duplicate_questions.sort(
        key=lambda pair: pair["similarity"],
        reverse=True,
    )

    attempts_per_item = Counter()
    technical_errors = Counter()
    total_tokens = 0
    total_cost = Decimal("0")
    priced_attempts = 0
    cache_hits = 0
    for record in attempts:
        question_id = str(record.get("question_id") or "")
        if question_id:
            attempts_per_item[question_id] += 1
        if record.get("error_type"):
            technical_errors[str(record["error_type"])] += 1
        attempt_item = record.get("item")
        attempt_item = attempt_item if isinstance(attempt_item, Mapping) else {}
        provenance = attempt_item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        generator = provenance.get("generator")
        generator = generator if isinstance(generator, Mapping) else {}
        cache_hits += int(bool(generator.get("cache_hit")))
        usage = generator.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        try:
            total_tokens += int(usage.get("total_token_count") or 0)
        except (TypeError, ValueError):
            pass
        if usage.get("total_token_cost") is not None:
            try:
                total_cost += Decimal(str(usage["total_token_cost"]))
                priced_attempts += 1
            except InvalidOperation:
                pass

    required_counts = {
        subcategory: subcategory_counts.get(subcategory, 0)
        for subcategory in subcategories
    }
    balanced = (
        expected_per_subcategory is None
        or all(
            count == expected_per_subcategory for count in required_counts.values()
        )
        and not (set(subcategory_counts) - set(subcategories))
    )
    automatic_gate_passed = bool(
        validation["valid"]
        and ledger_validation["valid"]
        and balanced
        and not ledger_mismatches
        and not snapshot_mismatches
        and not unresolved_items
        and not near_duplicate_questions
    )
    attempt_distribution = Counter(attempts_per_item.values())
    return {
        "automatic_gate_passed": automatic_gate_passed,
        "validation": validation,
        "ledger_validation": ledger_validation,
        "n_items": len(rows),
        "category": category,
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "subcategory_balance_passed": balanced,
        "ledger_mismatches": ledger_mismatches,
        "snapshot_mismatches": snapshot_mismatches,
        "near_duplicate_questions": near_duplicate_questions,
        "requested_models": dict(requested_models),
        "resolved_models": dict(resolved_models),
        "unresolved_items": unresolved_items,
        "source_evidence": {
            "unique_source_ids": len([value for value in source_ids if value]),
            "source_id_counts": dict(source_ids),
            "publisher_counts": dict(publishers),
        },
        "event_date_range": {
            "minimum": min(event_dates) if event_dates else None,
            "maximum": max(event_dates) if event_dates else None,
        },
        "attempts": {
            "records": len(attempts),
            "per_item_distribution": {
                str(count): frequency
                for count, frequency in sorted(attempt_distribution.items())
            },
            "technical_errors": dict(technical_errors),
            "cache_hits": cache_hits,
            "total_tokens": total_tokens,
            "priced_attempts": priced_attempts,
            "reported_total_cost": str(total_cost),
        },
    }
