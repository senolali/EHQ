"""Temporal-novelty contract for post-cutoff questions."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Tuple


SOURCE_VERIFIED = "source-verified-post-cutoff"
HUMAN_VERIFIED = "human-verified-post-cutoff"
ALLOWED_STATUSES = frozenset({SOURCE_VERIFIED, HUMAN_VERIFIED})
ALLOWED_BASES = frozenset(
    {
        "event_occurrence",
        "official_result",
        "official_decision",
        "official_appointment",
        "new_measurement",
        "new_forecast",
        "new_schedule",
        "new_release_metadata",
    }
)


def _date_interval(value: str) -> Tuple[date, date]:
    parts = value.split("/", 1)
    start = date.fromisoformat(parts[0])
    end = date.fromisoformat(parts[1]) if len(parts) == 2 else start
    if end < start:
        raise ValueError("date interval ends before it starts")
    return start, end


def temporal_novelty_issues(
    fact: Mapping[str, Any],
    *,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    require_human: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """Validate evidence that the tested claim itself became new post-cutoff.

    A source publication date is not a substitute for claim temporality. The
    attestation must identify when the claim became true/observable and how.
    """

    novelty = fact.get("temporal_novelty")
    if not isinstance(novelty, Mapping):
        return (
            (
                "missing_temporal_novelty",
                "PCQ requires a separate attestation that the tested claim, not "
                "merely its source page, became true or observable post-cutoff",
            ),
        )

    required = (
        "status",
        "basis",
        "claim_became_true_at",
        "source_id",
        "reviewer_type",
        "verified_at",
        "evidence",
    )
    missing = [field for field in required if not novelty.get(field)]
    issues = []
    if missing:
        issues.append(
            (
                "incomplete_temporal_novelty",
                f"Temporal-novelty attestation is missing fields: {missing}",
            )
        )

    status = str(novelty.get("status") or "")
    if status and status not in ALLOWED_STATUSES:
        issues.append(
            (
                "invalid_temporal_novelty_status",
                f"Unsupported temporal-novelty status: {status}",
            )
        )
    elif require_human and status != HUMAN_VERIFIED:
        issues.append(
            (
                "temporal_novelty_human_review_required",
                "Release PCQ facts require human-verified-post-cutoff status",
            )
        )

    basis = str(novelty.get("basis") or "")
    if basis and basis not in ALLOWED_BASES:
        issues.append(
            (
                "invalid_temporal_novelty_basis",
                "Temporal novelty must describe the new claim/event, not source "
                f"publication alone; unsupported basis: {basis}",
            )
        )

    claim_value = str(novelty.get("claim_became_true_at") or "")
    event_value = str(fact.get("event_date") or "")
    try:
        claim_start, claim_end = _date_interval(claim_value)
    except ValueError:
        if claim_value:
            issues.append(
                (
                    "invalid_temporal_novelty_date",
                    "claim_became_true_at must be an ISO date or ISO date interval",
                )
            )
    else:
        if window_start and window_end and not (
            window_start <= claim_start <= window_end
            and window_start <= claim_end <= window_end
        ):
            issues.append(
                (
                    "temporal_novelty_outside_window",
                    f"Claim interval {claim_value} is outside the PCQ event window",
                )
            )
        try:
            event_start, event_end = _date_interval(event_value)
        except ValueError:
            pass
        else:
            if claim_end < event_start or event_end < claim_start:
                issues.append(
                    (
                        "temporal_novelty_event_date_mismatch",
                        "claim_became_true_at does not overlap the fact event_date",
                    )
                )

    source_id = str(novelty.get("source_id") or "")
    evidence_ids = {
        str(source.get("source_id") or "")
        for source in fact.get("source_evidence") or []
        if isinstance(source, Mapping)
    }
    if source_id and source_id not in evidence_ids:
        issues.append(
            (
                "temporal_novelty_source_mismatch",
                "Temporal-novelty source_id is not present in source_evidence",
            )
        )
    return tuple(issues)
