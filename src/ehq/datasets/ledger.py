"""Validation for source-backed atomic-fact ledgers."""

from __future__ import annotations

from collections import Counter
from datetime import date
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import unicodedata

from .temporal import temporal_novelty_issues
from .answer_quality import pcq_answer_quality_issues


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TARGET_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _normalize_question_target(value: str) -> str:
    """Normalize semantic question targets for deterministic collision checks."""

    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(_TARGET_TOKEN_RE.findall(normalized)).strip()


def validate_fact_ledger(
    ledger: Mapping[str, Any],
    *,
    expected_total: Optional[int] = None,
    expected_per_subcategory: Optional[int] = None,
    expected_subcategories: Optional[Sequence[str]] = None,
    require_release_evidence: bool = False,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    facts = ledger.get("facts")
    if not isinstance(facts, list):
        return {
            "valid": False,
            "n_facts": 0,
            "subcategory_counts": {},
            "issues": [{"code": "missing_facts", "message": "facts must be an array"}],
        }
    window = ledger.get("event_window") or {}
    try:
        window_start = _parse_date(window["start"])
        window_end = _parse_date(window["end"])
    except (KeyError, TypeError, ValueError):
        window_start = window_end = None
        issues.append(
            {"code": "invalid_event_window", "message": "ISO event window is required"}
        )

    seen_ids = set()
    seen_atomic_facts = set()
    seen_question_targets: Dict[str, Dict[str, str]] = {}
    subcategories: Counter[str] = Counter()
    for fact in facts:
        if not isinstance(fact, Mapping):
            issues.append({"code": "invalid_fact", "message": "Fact must be an object"})
            continue
        fact_id = str(fact.get("fact_id") or "")
        if not fact_id:
            issues.append({"code": "missing_fact_id", "message": "Fact ID is required"})
        elif fact_id in seen_ids:
            issues.append(
                {"code": "duplicate_fact_id", "fact_id": fact_id, "message": "Duplicate ID"}
            )
        else:
            seen_ids.add(fact_id)
        subcategory = str(fact.get("subcategory") or "")
        if not subcategory:
            issues.append(
                {
                    "code": "missing_subcategory",
                    "fact_id": fact_id,
                    "message": "Subcategory is required",
                }
            )
        subcategories[subcategory] += 1
        gold = str(fact.get("gold_answer") or "").strip()
        target = str(fact.get("question_target") or "").strip()
        if not gold or not target:
            issues.append(
                {
                    "code": "missing_gold_or_target",
                    "fact_id": fact_id,
                    "message": "Gold answer and question target are required",
                }
            )
        for answer_issue in pcq_answer_quality_issues(target, gold):
            issues.append(
                {
                    "code": "pcq_unusable_gold",
                    "fact_id": fact_id,
                    "message": f"PCQ gold-answer gate failed: {answer_issue}",
                }
            )
        normalized_target = _normalize_question_target(target)
        normalized_gold = _normalize_question_target(gold)
        identity = (subcategory.casefold(), normalized_target, normalized_gold)
        if identity in seen_atomic_facts:
            issues.append(
                {
                    "code": "duplicate_atomic_fact",
                    "fact_id": fact_id,
                    "message": "Duplicate subcategory/target/gold combination",
                }
            )
        else:
            seen_atomic_facts.add(identity)

        if normalized_target:
            earlier = seen_question_targets.get(normalized_target)
            if earlier is not None:
                issues.append(
                    {
                        "code": "duplicate_question_target",
                        "fact_id": fact_id,
                        "other_fact_id": earlier["fact_id"],
                        "message": (
                            "Question target duplicates an earlier atomic fact: "
                            f"{earlier['fact_id']}"
                        ),
                    }
                )
                if normalized_gold != earlier["gold"]:
                    issues.append(
                        {
                            "code": "conflicting_question_target_gold",
                            "fact_id": fact_id,
                            "other_fact_id": earlier["fact_id"],
                            "message": (
                                "The same question target has different gold answers: "
                                f"{earlier['fact_id']} vs {fact_id}"
                            ),
                        }
                    )
            else:
                seen_question_targets[normalized_target] = {
                    "fact_id": fact_id,
                    "gold": normalized_gold,
                }
        event_date = str(fact.get("event_date") or "").split("/")[0]
        if window_start and window_end:
            try:
                parsed_event = _parse_date(event_date)
                if not window_start <= parsed_event <= window_end:
                    issues.append(
                        {
                            "code": "event_outside_window",
                            "fact_id": fact_id,
                            "message": f"{event_date} is outside the event window",
                        }
                    )
            except ValueError:
                issues.append(
                    {
                        "code": "invalid_event_date",
                        "fact_id": fact_id,
                        "message": "event_date must start with an ISO date",
                    }
                )
        evidence = fact.get("source_evidence")
        if not isinstance(evidence, list) or not evidence:
            issues.append(
                {
                    "code": "missing_evidence",
                    "fact_id": fact_id,
                    "message": "At least one evidence record is required",
                }
            )
        else:
            for source in evidence:
                required = ("source_id", "url", "publisher", "verified_at", "evidence")
                missing = [
                    field
                    for field in required
                    if not isinstance(source, Mapping) or not source.get(field)
                ]
                if missing:
                    issues.append(
                        {
                            "code": "incomplete_evidence",
                            "fact_id": fact_id,
                            "message": f"Missing evidence fields: {missing}",
                        }
                    )
                elif not str(source["url"]).startswith(("https://", "http://")):
                    issues.append(
                        {
                            "code": "invalid_evidence_url",
                            "fact_id": fact_id,
                            "message": "Evidence URL must be HTTP(S)",
                        }
                    )
                if require_release_evidence and isinstance(source, Mapping):
                    release_required = (
                        "title",
                        "published_at",
                        "retrieved_at",
                        "snapshot_path",
                        "snapshot_sha256",
                    )
                    release_missing = [
                        field for field in release_required if not source.get(field)
                    ]
                    if release_missing:
                        issues.append(
                            {
                                "code": "incomplete_release_evidence",
                                "fact_id": fact_id,
                                "message": (
                                    "Missing release evidence fields: "
                                    f"{release_missing}"
                                ),
                            }
                        )
                    snapshot_sha256 = str(source.get("snapshot_sha256") or "")
                    if snapshot_sha256 and not _SHA256_RE.fullmatch(
                        snapshot_sha256.casefold()
                    ):
                        issues.append(
                            {
                                "code": "invalid_snapshot_sha256",
                                "fact_id": fact_id,
                                "message": "snapshot_sha256 must be 64 hexadecimal characters",
                            }
                        )

        verification = fact.get("verification")
        if require_release_evidence:
            for code, message in temporal_novelty_issues(
                fact,
                window_start=window_start,
                window_end=window_end,
                require_human=True,
            ):
                issues.append(
                    {
                        "code": code,
                        "fact_id": fact_id,
                        "message": message,
                    }
                )
            if not isinstance(verification, Mapping):
                issues.append(
                    {
                        "code": "missing_verification",
                        "fact_id": fact_id,
                        "message": "Release facts require a verification record",
                    }
                )
            else:
                required_verification = ("status", "reviewer_type", "verified_at")
                missing_verification = [
                    field for field in required_verification if not verification.get(field)
                ]
                if missing_verification:
                    issues.append(
                        {
                            "code": "incomplete_verification",
                            "fact_id": fact_id,
                            "message": (
                                "Missing verification fields: "
                                f"{missing_verification}"
                            ),
                        }
                    )
                if verification.get("status") not in {
                    "source-verified",
                    "human-verified",
                }:
                    issues.append(
                        {
                            "code": "invalid_verification_status",
                            "fact_id": fact_id,
                            "message": (
                                "Verification status must be source-verified "
                                "or human-verified"
                            ),
                        }
                    )

    if expected_total is not None and len(facts) != expected_total:
        issues.append(
            {
                "code": "unexpected_total",
                "message": f"Expected {expected_total} facts, found {len(facts)}",
            }
        )
    if expected_per_subcategory is not None:
        required_subcategories = (
            list(expected_subcategories)
            if expected_subcategories is not None
            else list(subcategories)
        )
        for subcategory in required_subcategories:
            count = subcategories.get(subcategory, 0)
            if count != expected_per_subcategory:
                issues.append(
                    {
                        "code": "subcategory_imbalance",
                        "message": f"{subcategory}: expected "
                        f"{expected_per_subcategory}, found {count}",
                    }
                )
        if expected_subcategories is not None:
            unexpected = sorted(set(subcategories) - set(expected_subcategories))
            for subcategory in unexpected:
                issues.append(
                    {
                        "code": "unexpected_subcategory",
                        "message": f"Unexpected subcategory: {subcategory}",
                    }
                )
    return {
        "valid": not issues,
        "n_facts": len(facts),
        "subcategory_counts": dict(subcategories),
        "issues": issues,
    }
