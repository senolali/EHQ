"""Dependency-light, deterministic statistical summaries."""

from __future__ import annotations

import itertools
import math
import random
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _validate_pair(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise ValueError("Arrays must have equal length")
    if len(x) < 3:
        raise ValueError("At least three observations are required")
    if max(x) == min(x) or max(y) == min(y):
        raise ValueError("Correlation is undefined for constant arrays")


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    _validate_pair(x, y)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator


def _ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    _validate_pair(x, y)
    return pearson_correlation(_ranks(x), _ranks(y))


def bootstrap_correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    method: str = "pearson",
) -> Tuple[float, float]:
    _validate_pair(x, y)
    statistic = _correlation_statistic(method)
    rng = random.Random(seed)
    correlations = []
    n = len(x)
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        bx = [x[index] for index in indices]
        by = [y[index] for index in indices]
        if max(bx) == min(bx) or max(by) == min(by):
            continue
        correlations.append(statistic(bx, by))
    if not correlations:
        raise ValueError("No non-degenerate bootstrap samples")
    correlations.sort()
    low_index = max(0, int((alpha / 2) * len(correlations)))
    high_index = min(
        len(correlations) - 1,
        int((1 - alpha / 2) * len(correlations)) - 1,
    )
    return correlations[low_index], correlations[high_index]


def _correlation_statistic(method: str):
    if method == "pearson":
        return pearson_correlation
    if method == "spearman":
        return spearman_correlation
    raise ValueError("method must be 'pearson' or 'spearman'")


def correlation_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: str = "pearson",
    n_resamples: int = 20_000,
    seed: int = 42,
    exact_max_n: int = 8,
) -> Dict[str, Any]:
    """Two-sided model-level permutation test for a correlation statistic."""

    _validate_pair(x, y)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    statistic = _correlation_statistic(method)
    observed = abs(statistic(x, y))
    tolerance = 1e-15
    if len(y) <= exact_max_n:
        extreme = 0
        total = 0
        for permuted in itertools.permutations(y):
            total += 1
            if abs(statistic(x, permuted)) >= observed - tolerance:
                extreme += 1
        p_value = extreme / total
        permutation_method = "exact"
    else:
        rng = random.Random(seed)
        extreme = 0
        permuted = list(y)
        for _ in range(n_resamples):
            rng.shuffle(permuted)
            if abs(statistic(x, permuted)) >= observed - tolerance:
                extreme += 1
        p_value = (extreme + 1) / (n_resamples + 1)
        total = n_resamples
        permutation_method = "monte_carlo_plus_one"
    return {
        "p_value": p_value,
        "method": permutation_method,
        "n_permutations": total,
        "alternative": "two_sided",
        "statistic": method,
    }


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Return Holm step-down family-wise-error adjusted p-values."""

    if any(value < 0 or value > 1 for value in p_values.values()):
        raise ValueError("p-values must be within [0, 1]")
    ordered = sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def _paired_sign_permutation_pvalue(differences: Sequence[float]) -> float:
    nonzero = [difference for difference in differences if difference != 0]
    if not nonzero:
        return 1.0
    observed = abs(mean(nonzero))
    extreme = 0
    total = 2 ** len(nonzero)
    for signs in itertools.product((-1, 1), repeat=len(nonzero)):
        permuted = mean(
            sign * abs(difference) for sign, difference in zip(signs, nonzero)
        )
        if abs(permuted) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def paired_generation_analysis(
    pairs: Iterable[Mapping[str, float]],
    *,
    old_key: str = "old",
    new_key: str = "new",
) -> Dict[str, Any]:
    rows = list(pairs)
    if not rows:
        raise ValueError("At least one generational pair is required")
    differences = [float(row[new_key]) - float(row[old_key]) for row in rows]
    old = [float(row[old_key]) for row in rows]
    new = [float(row[new_key]) for row in rows]
    mean_difference = mean(differences)
    if len(differences) > 1:
        sample_variance = sum(
            (difference - mean_difference) ** 2 for difference in differences
        ) / (len(differences) - 1)
        sd_difference = math.sqrt(sample_variance)
    else:
        sd_difference = None
    if sd_difference is None:
        dz = None
        dz_status = "insufficient_pairs"
    elif sd_difference == 0:
        dz = None
        dz_status = "zero_difference_variance"
    else:
        dz = mean_difference / sd_difference
        dz_status = "ok"
    if len(differences) < 2:
        permutation_p = None
        permutation_status = "insufficient_pairs"
        inferential_status = "insufficient_pairs"
    else:
        permutation_p = _paired_sign_permutation_pvalue(differences)
        permutation_status = "ok"
        inferential_status = "ok"
    return {
        "n_pairs": len(rows),
        "old_mean": mean(old),
        "new_mean": mean(new),
        "mean_difference": mean_difference,
        "median_difference": median(differences),
        "improved_pairs": sum(difference > 0 for difference in differences),
        "declined_pairs": sum(difference < 0 for difference in differences),
        "unchanged_pairs": sum(difference == 0 for difference in differences),
        "inferential_status": inferential_status,
        "cohens_dz": dz,
        "cohens_dz_status": dz_status,
        "exact_sign_permutation_p": permutation_p,
        "exact_sign_permutation_status": permutation_status,
    }


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Draw from Binomial(n, p) in time proportional to n*p, not n.

    The capability probe's error rate is under one percent, so drawing 259
    Bernoulli trials per model per resample would spend almost all of its work
    on outcomes that are certain to be zero. Jumping directly to each success
    with a geometric step keeps a ten-thousand-resample null within a second.
    """

    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    successes = 0
    index = -1
    scale = math.log1p(-p)
    while True:
        index += 1 + int(math.log(rng.random()) / scale)
        if index >= n:
            return successes
        successes += 1


def _dispersion(counts: Sequence[Tuple[int, int]], rate: float) -> float:
    """Pearson chi-square of observed successes against one common rate."""

    statistic = 0.0
    for successes, trials in counts:
        for observed, expected in (
            (successes, trials * rate),
            (trials - successes, trials * (1.0 - rate)),
        ):
            if expected > 0:
                statistic += (observed - expected) ** 2 / expected
    return statistic


def homogeneity_test(
    counts: Mapping[str, Tuple[int, int]],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test whether per-model success counts are consistent with one rate.

    A measure on which every model scores alike carries no information about
    how models differ, and correlating it against anything reports the shape of
    its sampling noise. This decides that question before a correlation is
    computed rather than after it is published: the null is a single shared
    success rate, the statistic is the Pearson dispersion of the observed
    counts, and its distribution is obtained by resampling under the null
    instead of assumed from a chi-square table.
    """

    rows = [(int(s), int(n)) for s, n in counts.values() if int(n) > 0]
    if len(rows) < 2:
        return {"status": "insufficient_models", "n_models": len(rows)}

    total_successes = sum(s for s, _ in rows)
    total_trials = sum(n for _, n in rows)
    rate = total_successes / total_trials
    observed = _dispersion(rows, rate)

    rng = random.Random(seed)
    at_least = 0
    for _ in range(resamples):
        simulated = [(_binomial(rng, n, rate), n) for _, n in rows]
        if _dispersion(simulated, rate) >= observed:
            at_least += 1
    p_value = (at_least + 1) / (resamples + 1)

    proportions = [s / n for s, n in rows]
    observed_sd = (
        (
            sum((x - sum(proportions) / len(proportions)) ** 2 for x in proportions)
            / (len(proportions) - 1)
        )
        ** 0.5
    )
    expected_sd = math.sqrt(rate * (1.0 - rate) / (total_trials / len(rows)))

    return {
        "status": "ok",
        "n_models": len(rows),
        "pooled_rate": rate,
        "dispersion": observed,
        "p_value": p_value,
        "resamples": resamples,
        "seed": seed,
        "observed_sd": observed_sd,
        "sampling_sd": expected_sd,
        "sd_ratio": (observed_sd / expected_sd) if expected_sd > 0 else None,
        # A panel that cannot be distinguished from one shared rate has no
        # between-model signal to correlate with anything else.
        "homogeneous": p_value > 0.05,
    }
