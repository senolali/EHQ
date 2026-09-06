"""Release-gate validation for EHQ datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..constants import DATASET_CATEGORIES, REDACTION_TOKEN
from ..evaluation.correctness import contains_normalized_phrase, normalize_text
from ..hashing import sha256_json
from .question_quality import pcq_question_quality_issues
from .temporal import temporal_novelty_issues
from .answer_quality import pcq_answer_quality_issues


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    question_id: Optional[str] = None


@dataclass
class ValidationReport:
    valid: bool
    n_items: int
    category_counts: Dict[str, int]
    exact_duplicate_questions: int
    exact_duplicate_content: int
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value


def _content_identity(item: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "document": item.get("document"),
            "question": item.get("question"),
            "correct_answer": item.get("correct_answer"),
            "redacted_value": item.get("redacted_value"),
        }
    )


def _candidate_qc_is_declared_pending(item: Mapping[str, Any]) -> bool:
    """Accept only explicitly documented human-review candidate gates.

    This is an engineering/pilot exception, not a scientific release gate.
    Unknown or newly failing automatic checks remain fatal.
    """

    if item.get("release_status") != "candidate_requires_human_review":
        return False
    qc = item.get("qc")
    if not isinstance(qc, Mapping):
        return False
    checks = qc.get("checks")
    if not isinstance(checks, Mapping):
        return False
    allowed_false = {
        "human_source_review": lambda value: (
            value.get("status") == "mandatory_review_pending"
        ),
        "independent_secondary_review": lambda value: (
            value.get("status") == "required_independent_review_pending"
            and value.get("reviewer_must_differ_from_primary") is True
            and value.get("fixed_stratified_sample_size") == 300
        ),
        "obscurity_evidence": lambda value: (
            value.get("status") == "majority_known"
            and value.get("human_adjudication_required") is True
        ),
        "full_page_snapshot": lambda value: (
            value.get("status")
            == "retained_search_result_fallback_requires_replacement"
        ),
    }
    found_pending = False
    for name, value in checks.items():
        if not isinstance(value, Mapping) or value.get("passed") is not False:
            continue
        validator = allowed_false.get(str(name))
        if validator is None or not validator(value):
            return False
        found_pending = True
    return found_pending


def _release_human_review_issues(item: Mapping[str, Any]) -> List[tuple[str, str]]:
    """Validate the factual-item attestations behind a release-ready flag."""

    if item.get("category") not in {"PCQ", "HNQ"}:
        return []
    if item.get("release_status") != "release_ready":
        return []
    checks = ((item.get("qc") or {}).get("checks") or {})
    primary = checks.get("human_source_review") or {}
    secondary = checks.get("independent_secondary_review") or {}
    issues: List[tuple[str, str]] = []
    if not (
        primary.get("passed") is True
        and primary.get("status") == "completed"
        and primary.get("reviewer_name")
    ):
        issues.append(
            (
                "release_primary_review_missing",
                "Release-ready PCQ/HNQ item lacks completed named primary review",
            )
        )
    if not (
        secondary.get("passed") is True
        and secondary.get("status")
        in {"completed", "not_selected_release_gate_satisfied"}
        and secondary.get("review_protocol_version")
    ):
        issues.append(
            (
                "release_secondary_review_missing",
                "Release-ready PCQ/HNQ item lacks a completed secondary-review gate",
            )
        )
    if secondary.get("sample_member") is True and not (
        secondary.get("reviewer_identifier")
        and secondary.get("decision") == "accept"
        and secondary.get("independent_of_primary") is True
        and secondary.get("blinded_to_primary_decision") is True
    ):
        issues.append(
            (
                "release_secondary_sample_attestation_incomplete",
                "Selected secondary-review item lacks an independent accept attestation",
            )
        )
    return issues


def validate_dataset(
    items: Iterable[Mapping[str, Any]],
    *,
    expected_total: Optional[int] = None,
    expected_per_category: Optional[int] = None,
    require_source_evidence: bool = False,
    require_pcq_temporal_novelty: bool = False,
    allow_pending_human_review: bool = False,
) -> ValidationReport:
    rows = list(items)
    issues: List[ValidationIssue] = []
    ids: Set[str] = set()
    normalized_questions: Set[str] = set()
    content_ids: Set[str] = set()
    duplicate_questions = 0
    duplicate_content = 0
    category_counts: Counter[str] = Counter()

    for position, item in enumerate(rows):
        question_id = str(item.get("question_id") or item.get("id") or "").strip()
        category = str(item.get("category") or "").upper().strip()
        question = str(item.get("question") or "").strip()
        if not question_id:
            issues.append(
                ValidationIssue("error", "missing_id", f"Item {position} has no ID")
            )
        elif question_id in ids:
            issues.append(
                ValidationIssue("error", "duplicate_id", "Duplicate ID", question_id)
            )
        else:
            ids.add(question_id)
        if category not in DATASET_CATEGORIES:
            issues.append(
                ValidationIssue(
                    "error", "invalid_category", f"Unknown category {category!r}", question_id
                )
            )
        else:
            category_counts[category] += 1
        if len(question.split()) < 4:
            issues.append(
                ValidationIssue(
                    "error", "invalid_question", "Question is empty or too short", question_id
                )
            )
        normalized = normalize_text(question)
        if normalized in normalized_questions:
            duplicate_questions += 1
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_question",
                    "Exact normalized question duplicate",
                    question_id,
                )
            )
        else:
            normalized_questions.add(normalized)
        content_id = _content_identity(item)
        if content_id in content_ids:
            duplicate_content += 1
            issues.append(
                ValidationIssue(
                    "error", "duplicate_content", "Exact item-content duplicate", question_id
                )
            )
        else:
            content_ids.add(content_id)

        qc_passed = item.get("qc_passed")
        qc = item.get("qc")
        nested_qc_passed = (
            qc.get("passed") if isinstance(qc, Mapping) else None
        )
        if qc_passed is False or nested_qc_passed is False:
            if allow_pending_human_review and _candidate_qc_is_declared_pending(item):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "candidate_human_gate_pending",
                        "Candidate item has a declared human-review/replacement gate; "
                        "results are non-publishable",
                        question_id,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        "error", "qc_failed", "Item is marked QC failed", question_id
                    )
                )

        for code, message in _release_human_review_issues(item):
            issues.append(ValidationIssue("error", code, message, question_id))

        if category == "CCQ":
            document = str(item.get("document") or "")
            redacted = str(item.get("redacted_value") or "")
            if document.count(REDACTION_TOKEN) != 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "ccq_redaction_count",
                        "CCQ document must contain exactly one redaction token",
                        question_id,
                    )
                )
            if not redacted:
                issues.append(
                    ValidationIssue(
                        "error",
                        "ccq_missing_redacted_value",
                        "CCQ redacted value is required for audit",
                        question_id,
                    )
                )
            elif contains_normalized_phrase(
                document.replace(REDACTION_TOKEN, ""),
                redacted,
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "ccq_answer_leak",
                        "CCQ redacted value remains in the document",
                        question_id,
                    )
                )
            if item.get("correct_answer") != REDACTION_TOKEN:
                issues.append(
                    ValidationIssue(
                        "error",
                        "ccq_invalid_gold",
                        "CCQ correct_answer must be [REDACTED]",
                        question_id,
                    )
                )

        if category == "PCQ":
            quality_issues = pcq_question_quality_issues(question)
            if quality_issues:
                issues.append(
                    ValidationIssue(
                        "error",
                        "pcq_malformed_question",
                        "PCQ question-form gate failed: "
                        + ", ".join(quality_issues),
                        question_id,
                    )
                )
            answer_issues = pcq_answer_quality_issues(
                question,
                str(item.get("correct_answer") or ""),
            )
            if answer_issues:
                issues.append(
                    ValidationIssue(
                        "error",
                        "pcq_unusable_gold",
                        "PCQ gold-answer gate failed: "
                        + ", ".join(answer_issues),
                        question_id,
                    )
                )
            if require_pcq_temporal_novelty:
                for code, message in temporal_novelty_issues(
                    item,
                    require_human=True,
                ):
                    issues.append(
                        ValidationIssue(
                            "error",
                            code,
                            message,
                            question_id,
                        )
                    )

        if require_source_evidence and category in {"PCQ", "HNQ"}:
            evidence = item.get("source_evidence")
            if not isinstance(evidence, list) or not evidence:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_source_evidence",
                        f"{category} requires at least one source-evidence record",
                        question_id,
                    )
                )
            else:
                valid_evidence = [
                    source
                    for source in evidence
                    if isinstance(source, Mapping)
                    and str(source.get("url") or "").startswith(("https://", "http://"))
                    and source.get("publisher")
                    and source.get("verified_at")
                    and source.get("evidence")
                ]
                if not valid_evidence:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "invalid_source_evidence",
                            f"{category} evidence requires URL, publisher, "
                            "verification date, and supporting fact",
                            question_id,
                        )
                    )

    if expected_total is not None and len(rows) != expected_total:
        issues.append(
            ValidationIssue(
                "error",
                "unexpected_total",
                f"Expected {expected_total} items, found {len(rows)}",
            )
        )
    if expected_per_category is not None:
        for category in DATASET_CATEGORIES:
            count = category_counts.get(category, 0)
            if count != expected_per_category:
                issues.append(
                    ValidationIssue(
                        "error",
                        "category_imbalance",
                        f"{category}: expected {expected_per_category}, found {count}",
                    )
                )
    factual_release = [
        item
        for item in rows
        if item.get("category") in {"PCQ", "HNQ"}
        and item.get("release_status") == "release_ready"
    ]
    if factual_release:
        selected = [
            item
            for item in factual_release
            if (
                ((item.get("qc") or {}).get("checks") or {})
                .get("independent_secondary_review", {})
                .get("sample_member")
                is True
            )
        ]
        if len(factual_release) != 1500 or len(selected) != 300:
            issues.append(
                ValidationIssue(
                    "error",
                    "secondary_review_sample_size",
                    "Release requires 1,500 factual items and a fixed 300-item secondary sample; "
                    f"found {len(factual_release)} and {len(selected)}",
                )
            )
        stratum_counts = Counter(str(item.get("subcategory")) for item in selected)
        if stratum_counts and (
            len(stratum_counts) != 10
            or any(count != 30 for count in stratum_counts.values())
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "secondary_review_stratum_balance",
                    "Secondary sample must contain 30 items from each PCQ/HNQ subcategory",
                )
            )
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        n_items=len(rows),
        category_counts=dict(category_counts),
        exact_duplicate_questions=duplicate_questions,
        exact_duplicate_content=duplicate_content,
        issues=issues,
    )
