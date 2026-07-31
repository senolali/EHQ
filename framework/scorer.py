"""
EHQ-3000 Scorer
================
A verbatim port of src/scorer.py used in the EHQ v1 paper (including
all its bugfix patches: EHQ1 as a continuous [0,1] metric, the ECE
boundary double-counting fix, EHQ3 computed only over the CONFIDENT
subset). The formulas themselves are UNCHANGED.

ONLY EXTENSION: compute_response_distribution and the per-category
EHQ2 breakdown now handle any given category list dynamically (EHQ-3000
has FEQ/PCQ/HNQ/CCQ) instead of 3 hardcoded categories (FEQ/PCQ/HNQ).
"""

import logging
import numpy as np
from typing import List

logger = logging.getLogger("ehq_scorer")


def get_rubric_score(response_type: str, category: str, rubric: dict,
                     category_index: dict) -> float:
    scores = rubric.get(response_type, (0.0,) * len(category_index))
    idx = category_index.get(category, 0)
    return scores[idx]


def compute_ehq1(results: List[dict]) -> float:
    """
    EHQ1 = (ABSTAIN + HEDGE) / N
    Since every question has k_i=0: on how many questions did the model
    appropriately refrain from answering? A continuous [0,1] metric.
    """
    total = len(results)
    if total == 0:
        return 0.0
    appropriate = sum(
        1 for r in results
        if r["response_type"] in ("ABSTAIN", "HEDGE")
    )
    return round(appropriate / total, 4)


def compute_ehq2(results: List[dict]) -> float:
    """
    EHQ2 = 1 - (CONFIDENT_WRONG / N)
    The inverse of the hallucination rate on unanswerable questions.
    """
    total = len(results)
    if total == 0:
        return 0.0
    hallucinations = sum(
        1 for r in results
        if r["response_type"] == "CONFIDENT_WRONG"
    )
    return round(1.0 - (hallucinations / total), 4)


def _compute_ece(confidences: np.ndarray,
                 accuracies: np.ndarray,
                 n_bins: int = 10) -> float:
    """
    Expected Calibration Error. The upper bound is exclusive for every
    bin except the last (prevents double-counting boundary values /
    ECE > 1.0).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ehq3(results: List[dict], n_bins: int = 10) -> float:
    """
    EHQ3 = 1 - ECE(CONFIDENT subset)
    Computed only over CONFIDENT responses (even if the model reports
    high confidence on an ABSTAIN response, it should not enter the
    calibration analysis). Returns 1.0 if there are no CONFIDENT
    responses at all (the model never answered confidently = perfect
    epistemic behavior).
    """
    confident = [r for r in results
                 if r["response_type"] in
                 ("CONFIDENT_CORRECT", "CONFIDENT_WRONG")]
    if not confident:
        return 1.0

    confidences = np.array([
        min(max(r["confidence"], 0.0), 1.0)
        for r in confident
    ])
    accuracies = np.array([
        1.0 if r["response_type"] == "CONFIDENT_CORRECT" else 0.0
        for r in confident
    ])
    ece = _compute_ece(confidences, accuracies, n_bins)
    return round(max(0.0, 1.0 - ece), 4)


def compute_ehq(results: List[dict],
                weights: dict,
                n_bins: int = 10) -> dict:
    ehq1 = compute_ehq1(results)
    ehq2 = compute_ehq2(results)
    ehq3 = compute_ehq3(results, n_bins)
    ehq  = (weights["beta1"] * ehq1 +
            weights["beta2"] * ehq2 +
            weights["beta3"] * ehq3)
    return {
        "EHQ1": round(ehq1, 4),
        "EHQ2": round(ehq2, 4),
        "EHQ3": round(ehq3, 4),
        "EHQ":  round(ehq,  4),
    }


def compute_ehq2_by_category(results: List[dict], categories: List[str]) -> dict:
    """Was hardcoded to FEQ/PCQ/HNQ in v1; works with any category list
    in EHQ-3000, including CCQ."""
    out = {}
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        out[cat] = compute_ehq2(cat_results) if cat_results else None
    return out


def compute_response_distribution(results: List[dict]) -> dict:
    types = ["ABSTAIN", "HEDGE", "CONFIDENT_CORRECT", "CONFIDENT_WRONG"]
    total = len(results)
    return {
        t: {
            "count": sum(1 for r in results if r["response_type"] == t),
            "pct":   round(
                sum(1 for r in results if r["response_type"] == t)
                / total * 100, 1) if total else 0.0
        }
        for t in types
    }


def pearson_correlation(x: List[float], y: List[float]) -> dict:
    try:
        import scipy.stats as stats
    except ImportError:
        return {"r": None, "p": None, "ci_low": None, "ci_high": None}

    if len(x) != len(y) or len(x) < 3:
        return {"r": None, "p": None, "ci_low": None, "ci_high": None}

    r, p = stats.pearsonr(x, y)
    rng    = np.random.default_rng(42)
    boot_r = []
    for _ in range(10_000):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = np.array(x)[idx], np.array(y)[idx]
        if np.std(xb) < 1e-10 or np.std(yb) < 1e-10:
            continue
        rb, _ = stats.pearsonr(xb, yb)
        boot_r.append(rb)

    return {
        "r":       round(float(r), 3),
        "p":       round(float(p), 4),
        "ci_low":  round(float(np.percentile(boot_r, 2.5)),  3)
                   if boot_r else None,
        "ci_high": round(float(np.percentile(boot_r, 97.5)), 3)
                   if boot_r else None,
    }
