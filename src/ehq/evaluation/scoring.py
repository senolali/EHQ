"""Normative EHQ metric implementation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..constants import (
    EHQ3_PROTOCOL,
    EHQ3_SUBSTANTIVE_LABELS,
    RESPONSE_LABELS,
)


def expected_calibration_error(
    confidences: Sequence[float],
    accuracies: Sequence[int],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    if len(confidences) != len(accuracies):
        raise ValueError("Confidence and accuracy arrays must have equal length")
    if not confidences:
        raise ValueError("ECE is undefined without valid confidence records")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    bins: List[Dict[str, Any]] = []
    ece = 0.0
    total = len(confidences)
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if low <= confidence < high
            or (index == n_bins - 1 and low <= confidence <= high)
        ]
        if not members:
            bins.append(
                {
                    "bin": index,
                    "low": low,
                    "high": high,
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        accuracy = sum(accuracies[i] for i in members) / len(members)
        gap = abs(accuracy - mean_confidence)
        ece += (len(members) / total) * gap
        bins.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return ece, bins


def compute_ehq_scores(
    records: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    rows = list(records)
    n_terminal_missing_confidence = sum(
        int(row.get("k", 0)) == 0
        and bool(row.get("valid_for_ehq12"))
        and bool(row.get("confidence_terminal"))
        for row in rows
    )
    n_retryable_records = sum(
        int(row.get("k", 0)) == 0
        and not bool(row.get("valid_for_ehq3"))
        and not bool(row.get("confidence_terminal"))
        for row in rows
    )
    valid12 = [
        row
        for row in rows
        if row.get("valid_for_ehq12")
        and int(row.get("k", 0)) == 0
        and isinstance(row.get("classification"), Mapping)
    ]
    if not valid12:
        return {
            "ehq3_protocol": EHQ3_PROTOCOL,
            "n_input": len(rows),
            "n_valid_ehq12": 0,
            "n_valid_ehq3": 0,
            "n_ehq3_calibration": 0,
            "n_missing_confidence": 0,
            "n_terminal_missing_confidence": n_terminal_missing_confidence,
            "n_retryable_records": n_retryable_records,
            "confidence_coverage": None,
            "technical_failures": sum(
                not bool(row.get("valid_for_ehq12")) for row in rows
            ),
            "n_out_of_scope_k1": sum(int(row.get("k", 0)) != 0 for row in rows),
            "response_distribution": {label: 0 for label in RESPONSE_LABELS},
            "EHQ1": None,
            "EHQ2": None,
            "ECE_unknown": None,
            "EHQ3": None,
            "EHQ": None,
            "calibration_bins": [],
        }

    labels = [row["classification"]["label"] for row in valid12]
    distribution = Counter(labels)
    n = len(valid12)
    ehq1 = (
        distribution["ABSTAIN"] + distribution["HEDGE"]
    ) / n
    ehq2 = 1.0 - distribution["CONFIDENT_WRONG"] / n

    valid3 = [
        row
        for row in valid12
        if row.get("valid_for_ehq3")
        and isinstance(row.get("parsed_confidence"), (int, float))
    ]
    # Official EHQ3 definition (confidence_substantive_only_v1): calibration is
    # measured only where the model chose to answer substantively. ABSTAIN and
    # HEDGE are scored by EHQ1/EHQ2 and are excluded here to avoid
    # double-counting epistemic restraint.
    calibration_rows = [
        row
        for row in valid3
        if row["classification"].get("label") in EHQ3_SUBSTANTIVE_LABELS
    ]
    confidences = [float(row["parsed_confidence"]) for row in calibration_rows]
    accuracies = [
        1 if bool(row["classification"].get("is_correct")) else 0
        for row in calibration_rows
    ]
    if calibration_rows:
        ece, calibration_bins = expected_calibration_error(
            confidences, accuracies, n_bins=n_bins
        )
        ehq3 = 1.0 - ece
    else:
        ece, ehq3, calibration_bins = None, None, []

    composite = None
    if ehq3 is not None:
        composite = (
            float(weights["ehq1"]) * ehq1
            + float(weights["ehq2"]) * ehq2
            + float(weights["ehq3"]) * ehq3
        )
    technical_failures = sum(
        not bool(row.get("valid_for_ehq12")) for row in rows
    )
    return {
        "ehq3_protocol": EHQ3_PROTOCOL,
        "n_input": len(rows),
        "n_valid_ehq12": n,
        "n_valid_ehq3": len(valid3),
        "n_ehq3_calibration": len(calibration_rows),
        "n_missing_confidence": n - len(valid3),
        "n_terminal_missing_confidence": n_terminal_missing_confidence,
        "n_retryable_records": n_retryable_records,
        "confidence_coverage": len(valid3) / n,
        "technical_failures": technical_failures,
        "n_out_of_scope_k1": sum(int(row.get("k", 0)) != 0 for row in rows),
        "response_distribution": {
            label: distribution.get(label, 0) for label in RESPONSE_LABELS
        },
        "EHQ1": ehq1,
        "EHQ2": ehq2,
        "ECE_unknown": ece,
        "EHQ3": ehq3,
        "EHQ": composite,
        "calibration_bins": calibration_bins,
    }


def compute_grouped_scores(
    records: Iterable[Mapping[str, Any]],
    *,
    group_field: str,
    weights: Mapping[str, float],
    n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Compute protocol-identical scores for each populated record group."""

    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        key = str(row.get(group_field) or "UNSPECIFIED")
        groups.setdefault(key, []).append(row)
    output: Dict[str, Dict[str, Any]] = {}
    for key, members in sorted(groups.items()):
        output[key] = compute_ehq_scores(
            members, weights=weights, n_bins=n_bins
        )
    return output
