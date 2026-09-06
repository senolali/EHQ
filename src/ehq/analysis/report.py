"""Manuscript-aligned RQ1/RQ2 summaries computed from completed runs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

from ..constants import EHQ3_PROTOCOL
from ..types import ModelSpec
from .statistics import (
    bootstrap_correlation_ci,
    correlation_permutation_test,
    holm_adjust,
    homogeneity_test,
    paired_generation_analysis,
    pearson_correlation,
    spearman_correlation,
)


def load_capability_scores(path: Path) -> Dict[str, float]:
    import csv
    import json

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        output = {}
        for row in rows:
            model = str(row.get("model") or "").strip()
            value = row.get("capability_score", row.get("CQ"))
            if model and value not in (None, ""):
                output[model] = float(value)
        return output
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        return {str(key): float(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {
            str(row["model"]): float(
                row.get("capability_score", row.get("CQ"))
            )
            for row in raw
        }
    raise ValueError("Capability scores must be a JSON object/list or CSV")


def load_capability_counts(path: Path) -> Dict[str, tuple]:
    """Read per-model probe counts when the score file carries them.

    A bare score cannot say whether the differences between models exceed what
    resampling the same items would produce, so a probe near ceiling looks
    identical to a probe that discriminates. The counts make that testable.
    Files written before the columns existed simply yield nothing.
    """

    import csv

    if path.suffix.lower() != ".csv":
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: Dict[str, tuple] = {}
    for row in rows:
        model = str(row.get("model") or "").strip()
        correct, scored = row.get("n_correct"), row.get("n_scored")
        if model and correct not in (None, "") and scored not in (None, ""):
            try:
                output[model] = (int(correct), int(scored))
            except ValueError:
                continue
    return output


def _rank_desc(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))
    ranks: Dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in range(start, end):
            ranks[ordered[index][0]] = rank
        start = end
    return ranks


def _correlation_summary(
    x: Sequence[float], y: Sequence[float], *, seed: int
) -> Dict[str, Any]:
    if len(x) < 3:
        return {"status": "insufficient_models", "n": len(x)}
    if max(x) == min(x) or max(y) == min(y):
        return {"status": "constant_input", "n": len(x)}
    pearson = pearson_correlation(x, y)
    spearman = spearman_correlation(x, y)
    pearson_low, pearson_high = bootstrap_correlation_ci(
        x, y, seed=seed, method="pearson"
    )
    spearman_low, spearman_high = bootstrap_correlation_ci(
        x, y, seed=seed + 1, method="spearman"
    )
    pearson_test = correlation_permutation_test(
        x, y, method="pearson", seed=seed
    )
    spearman_test = correlation_permutation_test(
        x, y, method="spearman", seed=seed + 1
    )
    fisher = [None, None]
    if len(x) > 3 and abs(pearson) < 1:
        z = math.atanh(pearson)
        width = 1.959963984540054 / math.sqrt(len(x) - 3)
        fisher = [math.tanh(z - width), math.tanh(z + width)]
    return {
        "status": "ok",
        "n": len(x),
        "inference_unit": "model",
        "pearson_r": pearson,
        "pearson_p": pearson_test["p_value"],
        "spearman_rho": spearman,
        "spearman_p": spearman_test["p_value"],
        "bootstrap_95_ci": [pearson_low, pearson_high],
        "fisher_95_ci": fisher,
        "pearson": {
            "estimate": pearson,
            "bootstrap_95_ci": [pearson_low, pearson_high],
            "permutation_test": pearson_test,
        },
        "spearman": {
            "estimate": spearman,
            "bootstrap_95_ci": [spearman_low, spearman_high],
            "permutation_test": spearman_test,
        },
    }


def build_analysis_report(
    results: Mapping[str, Mapping[str, Any]],
    models: Sequence[ModelSpec],
    *,
    seed: int,
    capability_scores: Mapping[str, float] | None = None,
    capability_counts: Mapping[str, tuple] | None = None,
    excluded_models: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    # A model can be measured and still not be measurable: a route whose
    # answers were systematically truncated produces scores that are artefacts
    # of the truncation rather than of the model. Such a model is withheld from
    # the confirmatory analyses -- every one of which draws on `usable` -- while
    # its scores and the reason are carried in the output, so the exclusion is
    # reported rather than performed silently.
    withheld = dict(excluded_models or {})
    scored = {
        model: result["scores"]
        for model, result in results.items()
        if result["scores"].get("EHQ") is not None
    }
    unknown = sorted(set(withheld) - set(results))
    if unknown:
        raise ValueError(
            f"Cannot exclude models absent from the run: {', '.join(unknown)}"
        )
    usable = {
        model: scores for model, scores in scored.items() if model not in withheld
    }
    if withheld and not usable:
        raise ValueError("Every scored model was excluded; nothing remains to analyse")

    components: Dict[str, Any] = {}
    for left, right in (("EHQ1", "EHQ2"), ("EHQ1", "EHQ3"), ("EHQ2", "EHQ3")):
        names = [
            name
            for name, scores in usable.items()
            if scores.get(left) is not None and scores.get(right) is not None
        ]
        components[f"{left}_vs_{right}"] = _correlation_summary(
            [float(usable[name][left]) for name in names],
            [float(usable[name][right]) for name in names],
            seed=seed,
        )
    component_ok = {
        name: summary
        for name, summary in components.items()
        if summary.get("status") == "ok"
    }
    if component_ok:
        pearson_adjusted = holm_adjust(
            {
                name: float(summary["pearson"]["permutation_test"]["p_value"])
                for name, summary in component_ok.items()
            }
        )
        spearman_adjusted = holm_adjust(
            {
                name: float(summary["spearman"]["permutation_test"]["p_value"])
                for name, summary in component_ok.items()
            }
        )
        for name, summary in component_ok.items():
            summary["pearson"]["holm_adjusted_p"] = pearson_adjusted[name]
            summary["spearman"]["holm_adjusted_p"] = spearman_adjusted[name]

    category_correlations: Dict[str, Any] = {}
    for index, (left, right) in enumerate(
        combinations(("FEQ", "PCQ", "HNQ", "CCQ"), 2)
    ):
        names = []
        for name in usable:
            categories = results[name].get("category_scores") or {}
            if (
                categories.get(left, {}).get("EHQ") is not None
                and categories.get(right, {}).get("EHQ") is not None
            ):
                names.append(name)
        category_correlations[f"{left}_vs_{right}"] = _correlation_summary(
            [
                float(results[name]["category_scores"][left]["EHQ"])
                for name in names
            ],
            [
                float(results[name]["category_scores"][right]["EHQ"])
                for name in names
            ],
            seed=seed + 100 + index * 2,
        )
    category_ok = {
        name: summary
        for name, summary in category_correlations.items()
        if summary.get("status") == "ok"
    }
    if category_ok:
        pearson_adjusted = holm_adjust(
            {
                name: float(summary["pearson"]["permutation_test"]["p_value"])
                for name, summary in category_ok.items()
            }
        )
        spearman_adjusted = holm_adjust(
            {
                name: float(summary["spearman"]["permutation_test"]["p_value"])
                for name, summary in category_ok.items()
            }
        )
        for name, summary in category_ok.items():
            summary["pearson"]["holm_adjusted_p"] = pearson_adjusted[name]
            summary["spearman"]["holm_adjusted_p"] = spearman_adjusted[name]

    pair_members: Dict[str, Dict[str, str]] = defaultdict(dict)
    for model in models:
        if model.pair and model.generation in {"old", "new"}:
            pair_members[model.pair][str(model.generation)] = model.name
    pair_rows = []
    for pair, members in sorted(pair_members.items()):
        old_name, new_name = members.get("old"), members.get("new")
        if old_name not in usable or new_name not in usable:
            continue
        pair_rows.append(
            {
                "pair": pair,
                "old_model": old_name,
                "new_model": new_name,
                "old": float(usable[old_name]["EHQ"]),
                "new": float(usable[new_name]["EHQ"]),
                "difference": float(usable[new_name]["EHQ"])
                - float(usable[old_name]["EHQ"]),
            }
        )
    rq2 = {
        "status": "ok" if pair_rows else "no_complete_pairs",
        "pairs": pair_rows,
        "summary": paired_generation_analysis(pair_rows) if pair_rows else None,
    }

    category_means: Dict[str, Dict[str, float]] = {}
    category_values: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for model_name in usable:
        result = results[model_name]
        for category, scores in result["category_scores"].items():
            for field in ("EHQ1", "EHQ2", "EHQ3", "EHQ"):
                if scores.get(field) is not None:
                    category_values[category][field].append(float(scores[field]))
    for category, values in sorted(category_values.items()):
        category_means[category] = {
            field: mean(entries) for field, entries in values.items() if entries
        }

    base_scores = {name: float(scores["EHQ"]) for name, scores in usable.items()}
    base_ranks = _rank_desc(base_scores)
    sensitivity_correlations_available = len(usable) >= 3
    sensitivity = []
    for scenario, weights in (
        ("drop_EHQ2", {"EHQ1": 0.75, "EHQ2": 0.0, "EHQ3": 0.25}),
        ("drop_EHQ1", {"EHQ1": 0.0, "EHQ2": 0.75, "EHQ3": 0.25}),
        (
            "equal_restraint_calibration",
            {"EHQ1": 0.25, "EHQ2": 0.25, "EHQ3": 0.50},
        ),
    ):
        alternative = {
            name: sum(float(scores[field]) * weight for field, weight in weights.items())
            for name, scores in usable.items()
        }
        alternative_ranks = _rank_desc(alternative)
        sensitivity.append(
            {
                "scenario": scenario,
                "weights": weights,
                "status": (
                    "ok"
                    if sensitivity_correlations_available
                    else "no_complete_scores"
                    if not usable
                    else "insufficient_models_for_correlation"
                ),
                "pearson_with_official_EHQ": (
                    pearson_correlation(
                        [base_scores[name] for name in sorted(usable)],
                        [alternative[name] for name in sorted(usable)],
                    )
                    if sensitivity_correlations_available
                    else None
                ),
                "spearman_with_official_EHQ": (
                    spearman_correlation(
                        [base_scores[name] for name in sorted(usable)],
                        [alternative[name] for name in sorted(usable)],
                    )
                    if sensitivity_correlations_available
                    else None
                ),
                "n_models_with_rank_change": sum(
                    alternative_ranks[name] != base_ranks[name] for name in usable
                ),
                "maximum_absolute_rank_shift": max(
                    (
                        abs(alternative_ranks[name] - base_ranks[name])
                        for name in usable
                    ),
                    default=None,
                ),
                "model_scores_and_ranks": [
                    {
                        "model": name,
                        "alternative_EHQ": alternative[name],
                        "official_rank": base_ranks[name],
                        "alternative_rank": alternative_ranks[name],
                        "rank_shift": alternative_ranks[name] - base_ranks[name],
                    }
                    for name in sorted(usable)
                ],
            }
        )

    rq1: Dict[str, Any] = {"status": "capability_scores_not_provided"}
    if capability_scores is not None:
        names = sorted(set(usable) & set(capability_scores))
        ehq_values = {name: float(usable[name]["EHQ"]) for name in names}
        cq_values = {name: float(capability_scores[name]) for name in names}
        ehq_ranks = _rank_desc(ehq_values)
        cq_ranks = _rank_desc(cq_values)
        rank_shifts = [
            {
                "model": name,
                "capability_rank": cq_ranks[name],
                "ehq_rank": ehq_ranks[name],
                "delta_capability_minus_ehq": cq_ranks[name] - ehq_ranks[name],
            }
            for name in names
        ]
        # A correlation is only interpretable if the capability measure
        # separates the models at all. Where the per-model counts are available
        # and consistent with a single shared success rate, the spread being
        # correlated is sampling noise, so the association is reported as
        # uninterpretable rather than as a result.
        homogeneity = None
        if capability_counts:
            overlap = {
                name: capability_counts[name]
                for name in names
                if name in capability_counts
            }
            if len(overlap) >= 2:
                homogeneity = homogeneity_test(overlap, seed=seed)

        if len(names) < 3:
            status = "insufficient_overlap"
        elif homogeneity and homogeneity.get("homogeneous"):
            status = "capability_homogeneous_no_variance_to_correlate"
        else:
            status = "ok"

        rq1 = {
            "status": status,
            "models": names,
            "correlation": _correlation_summary(
                [cq_values[name] for name in names],
                [ehq_values[name] for name in names],
                seed=seed,
            ),
            "correlation_interpretable": status == "ok",
            "homogeneity": homogeneity,
            "capability_range": (
                max(cq_values.values()) - min(cq_values.values())
                if cq_values
                else None
            ),
            "ehq_range": (
                max(ehq_values.values()) - min(ehq_values.values())
                if ehq_values
                else None
            ),
            "rank_shifts": rank_shifts,
            "n_absolute_rank_shifts_ge_2": sum(
                abs(row["delta_capability_minus_ehq"]) >= 2
                for row in rank_shifts
            ),
        }

    return {
        "schema_version": "1.3",
        "ehq3_protocol": EHQ3_PROTOCOL,
        "confirmatory_panel": sorted(usable),
        "excluded_models": [
            {
                "model": model,
                "reason": reason,
                "EHQ": scored.get(model, {}).get("EHQ"),
                "EHQ1": scored.get(model, {}).get("EHQ1"),
                "EHQ2": scored.get(model, {}).get("EHQ2"),
                "EHQ3": scored.get(model, {}).get("EHQ3"),
            }
            for model, reason in sorted(withheld.items())
        ],
        "rq1_capability_relationship": rq1,
        "rq2_generational_pairs": rq2,
        "component_correlations": components,
        "component_correlation_multiplicity": {
            "method": "Holm step-down",
            "families": [
                "three Pearson component-pair tests",
                "three Spearman component-pair tests",
            ],
        },
        "category_correlations": category_correlations,
        "category_correlation_multiplicity": {
            "status": "exploratory",
            "method": "Holm step-down",
            "families": [
                "six Pearson category-pair tests",
                "six Spearman category-pair tests",
            ],
        },
        "category_means": category_means,
        "composite_weight_sensitivity": {
            "status": "descriptive_sensitivity_analysis",
            "official_weights": {"EHQ1": 0.30, "EHQ2": 0.45, "EHQ3": 0.25},
            "scenarios": sensitivity,
        },
        "n_models_with_valid_ehq": len(usable),
    }
