"""Agreement and classification-validity summaries for human audits."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Sequence

from .constants import RESPONSE_LABELS


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float | None:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label arrays must have equal length")
    if not labels_a:
        raise ValueError("At least one paired label is required")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    values = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[value] / total) * (counts_b[value] / total)
        for value in values
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def binary_response_label(label: str) -> str:
    if label in {"ABSTAIN", "HEDGE"}:
        return "RESTRAINT"
    if label in {"CONFIDENT_CORRECT", "CONFIDENT_WRONG"}:
        return "SUBSTANTIVE"
    raise ValueError(f"Unknown response label: {label}")


def classification_metrics(
    reference: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str] = RESPONSE_LABELS,
) -> Dict[str, Any]:
    if len(reference) != len(predicted):
        raise ValueError("Reference and predicted arrays must have equal length")
    if not reference:
        raise ValueError("At least one classification is required")
    allowed = set(labels)
    unknown = (set(reference) | set(predicted)) - allowed
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")

    confusion = {
        truth: {guess: 0 for guess in labels}
        for truth in labels
    }
    for truth, guess in zip(reference, predicted):
        confusion[truth][guess] += 1

    per_class = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in labels if truth != label)
        fn = sum(confusion[label][guess] for guess in labels if guess != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    accuracy = sum(a == b for a, b in zip(reference, predicted)) / len(reference)
    return {
        "n": len(reference),
        "accuracy": accuracy,
        "cohen_kappa": cohen_kappa(reference, predicted),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }
