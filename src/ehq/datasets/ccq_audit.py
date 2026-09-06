"""Corpus-level audit statistics for generated CCQ datasets."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..evaluation.correctness import normalize_text
from .validate import validate_dataset


def _ngrams(text: str, size: int = 5) -> set[Tuple[str, ...]]:
    tokens = normalize_text(text).split()
    return {
        tuple(tokens[index : index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    intersection_size = len(left & right)
    union_size = len(left) + len(right) - intersection_size
    return intersection_size / union_size if union_size else 1.0


def audit_ccq(
    items: Iterable[Mapping[str, Any]],
    *,
    attempts: Sequence[Mapping[str, Any]] = (),
    expected_per_subcategory: int | None = None,
    question_similarity_threshold: float = 0.88,
    document_similarity_threshold: float = 0.82,
    document_ngram_threshold: float = 0.40,
) -> Dict[str, Any]:
    rows = list(items)
    validation = validate_dataset(rows, expected_total=len(rows)).to_dict()
    subcategory_counts = Counter(str(item.get("subcategory") or "") for item in rows)
    word_counts = [len(str(item.get("document") or "").split()) for item in rows]
    redacted_ids: Dict[str, List[str]] = {}
    for item in rows:
        value = normalize_text(str(item.get("redacted_value") or ""))
        redacted_ids.setdefault(value, []).append(str(item.get("question_id") or ""))
    redacted_counts = Counter(
        {value: len(question_ids) for value, question_ids in redacted_ids.items()}
    )
    duplicate_redacted_values = {
        value: {
            "count": count,
            "question_ids": redacted_ids[value],
        }
        for value, count in redacted_counts.items()
        if value and count > 1
    }

    question_pairs: List[Dict[str, Any]] = []
    document_pairs: List[Dict[str, Any]] = []
    normalized_questions = [
        normalize_text(str(item.get("question") or "")) for item in rows
    ]
    normalized_documents = [
        normalize_text(str(item.get("document") or "")) for item in rows
    ]
    document_ngrams = [_ngrams(document) for document in normalized_documents]
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            question_ratio = SequenceMatcher(
                None,
                normalized_questions[left_index],
                normalized_questions[right_index],
            ).ratio()
            if question_ratio >= question_similarity_threshold:
                question_pairs.append(
                    {
                        "left_id": left.get("question_id"),
                        "right_id": right.get("question_id"),
                        "similarity": round(question_ratio, 6),
                    }
                )
            ngram_similarity = _jaccard(
                document_ngrams[left_index],
                document_ngrams[right_index],
            )
            document_ratio = (
                SequenceMatcher(
                    None,
                    normalized_documents[left_index],
                    normalized_documents[right_index],
                ).ratio()
                if ngram_similarity >= min(0.08, document_ngram_threshold)
                else 0.0
            )
            if (
                document_ratio >= document_similarity_threshold
                or ngram_similarity >= document_ngram_threshold
            ):
                document_pairs.append(
                    {
                        "left_id": left.get("question_id"),
                        "right_id": right.get("question_id"),
                        "character_similarity": round(document_ratio, 6),
                        "fivegram_jaccard": round(ngram_similarity, 6),
                    }
                )

    question_pairs.sort(key=lambda pair: pair["similarity"], reverse=True)
    document_pairs.sort(
        key=lambda pair: max(
            pair["character_similarity"],
            pair["fivegram_jaccard"],
        ),
        reverse=True,
    )
    requested_models = Counter()
    resolved_models = Counter()
    accepted_origins = Counter()
    transformation_types = Counter()
    unresolved_items = []
    for item in rows:
        provenance = item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        generator = provenance.get("generator")
        generator = generator if isinstance(generator, Mapping) else {}
        requested_models[str(generator.get("requested_model") or "")] += 1
        resolved = str(generator.get("resolved_model") or "")
        resolved_models[resolved] += 1
        if not resolved:
            unresolved_items.append(item.get("question_id"))
        accepted_origins[str(provenance.get("origin") or "")] += 1
        for transformation in provenance.get("transformations") or []:
            if isinstance(transformation, Mapping):
                transformation_types[str(transformation.get("type") or "")] += 1

    attempt_origins = Counter()
    attempts_per_item = Counter()
    total_tokens = 0
    total_cost = Decimal("0")
    priced_attempts = 0
    technical_errors = Counter()
    cache_hits = 0
    for record in attempts:
        question_id = str(record.get("question_id") or "")
        attempts_per_item[question_id] += 1
        if record.get("error_type"):
            technical_errors[str(record["error_type"])] += 1
        item = record.get("item")
        item = item if isinstance(item, Mapping) else {}
        provenance = item.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        attempt_origins[str(provenance.get("origin") or "")] += 1
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

    attempt_distribution = Counter(attempts_per_item.values())
    balanced = (
        expected_per_subcategory is None
        or bool(subcategory_counts)
        and all(count == expected_per_subcategory for count in subcategory_counts.values())
    )
    near_duplicate_warnings = len(question_pairs) + len(document_pairs)
    automatic_gate_passed = bool(
        validation["valid"]
        and balanced
        and not duplicate_redacted_values
        and not unresolved_items
        and not near_duplicate_warnings
    )
    return {
        "automatic_gate_passed": automatic_gate_passed,
        "requires_manual_near_duplicate_review": bool(near_duplicate_warnings),
        "validation": validation,
        "n_items": len(rows),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "subcategory_balance_passed": balanced,
        "document_word_counts": {
            "minimum": min(word_counts) if word_counts else None,
            "maximum": max(word_counts) if word_counts else None,
            "mean": round(mean(word_counts), 3) if word_counts else None,
            "median": median(word_counts) if word_counts else None,
        },
        "duplicate_redacted_values": duplicate_redacted_values,
        "requested_models": dict(requested_models),
        "resolved_models": dict(resolved_models),
        "unresolved_items": unresolved_items,
        "accepted_origins": dict(accepted_origins),
        "accepted_transformations": dict(transformation_types),
        "near_duplicate_questions": question_pairs,
        "near_duplicate_documents": document_pairs,
        "attempts": {
            "records": len(attempts),
            "per_item_distribution": {
                str(count): frequency
                for count, frequency in sorted(attempt_distribution.items())
            },
            "origins": dict(attempt_origins),
            "technical_errors": dict(technical_errors),
            "cache_hits": cache_hits,
            "total_tokens": total_tokens,
            "priced_attempts": priced_attempts,
            "reported_total_cost": str(total_cost),
        },
    }
