"""Sensitivity of EHQ to the benchmark scope used for scoring.

The analysis is derived only from retained records.  It never contacts a
provider and never rewrites a response or label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ..evaluation.scoring import compute_ehq_scores
from .statistics import pearson_correlation, spearman_correlation


SCENARIOS = (
    ("official", "Official EHQ", ("FEQ", "PCQ", "HNQ", "CCQ")),
    (
        "strict_unavailability",
        "Strict unavailability (without HNQ)",
        ("FEQ", "PCQ", "CCQ"),
    ),
    (
        "knowledge_boundary",
        "Knowledge boundary (EHQ-K)",
        ("FEQ", "PCQ", "HNQ"),
    ),
    (
        "strict_knowledge_boundary",
        "Strict knowledge boundary",
        ("FEQ", "PCQ"),
    ),
    ("context_boundary", "Context boundary (EHQ-C)", ("CCQ",)),
)


def _rank_desc(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))
    output: Dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            output[ordered[position][0]] = rank
        start = end
    return output


def build_construct_scope_sensitivity(
    records: Iterable[Mapping[str, Any]],
    *,
    models: Sequence[str],
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    """Re-score the same retained records under transparent category scopes."""

    model_set = set(models)
    by_model: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    confident_correct = Counter()
    for row in records:
        model = str(row.get("model") or "")
        if model not in model_set:
            continue
        by_model[model].append(row)
        classification = row.get("classification")
        if (
            isinstance(classification, Mapping)
            and classification.get("label") == "CONFIDENT_CORRECT"
        ):
            confident_correct[str(row.get("category"))] += 1
    missing = sorted(model_set - set(by_model))
    if missing:
        raise ValueError(f"No retained records for models: {', '.join(missing)}")

    scenario_scores: Dict[str, Dict[str, Optional[float]]] = {}
    scenario_details: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scenario_id, _, categories in SCENARIOS:
        category_set = set(categories)
        details = {}
        scores = {}
        for model in models:
            selected = [
                row
                for row in by_model[model]
                if str(row.get("category")) in category_set
            ]
            result = compute_ehq_scores(selected, weights=weights)
            if scenario_id == "official" and result.get("EHQ") is None:
                raise ValueError(
                    f"{scenario_id} produced undefined EHQ for {model}"
                )
            details[model] = result
            scores[model] = (
                float(result["EHQ"]) if result.get("EHQ") is not None else None
            )
        scenario_details[scenario_id] = details
        scenario_scores[scenario_id] = scores

    official = scenario_scores["official"]
    official_values_defined = {
        model: value for model, value in official.items() if value is not None
    }
    official_ranks = _rank_desc(official_values_defined)
    rows = []
    for scenario_id, label, categories in SCENARIOS:
        values = scenario_scores[scenario_id]
        usable_models = [
            model
            for model in models
            if official.get(model) is not None and values.get(model) is not None
        ]
        usable_values = {
            model: float(values[model])
            for model in usable_models
            if values[model] is not None
        }
        ranks = _rank_desc(usable_values)
        complete_ranking = len(usable_models) == len(models)
        shifts = (
            {
                model: ranks[model] - official_ranks[model]
                for model in usable_models
            }
            if complete_ranking
            else {}
        )
        official_vector = [float(official[model]) for model in usable_models]
        scenario_vector = [float(values[model]) for model in usable_models]
        has_correlation = (
            len(usable_models) >= 3
            and len(set(official_vector)) > 1
            and len(set(scenario_vector)) > 1
        )
        rows.append(
            {
                "scenario": scenario_id,
                "label": label,
                "categories": list(categories),
                "n_models_scored": len(usable_models),
                "pearson_with_official": (
                    1.0
                    if scenario_id == "official"
                    else (
                        pearson_correlation(official_vector, scenario_vector)
                        if has_correlation
                        else None
                    )
                ),
                "spearman_with_official": (
                    1.0
                    if scenario_id == "official"
                    else (
                        spearman_correlation(official_vector, scenario_vector)
                        if has_correlation
                        else None
                    )
                ),
                "n_models_with_rank_change": (
                    sum(shift != 0 for shift in shifts.values())
                    if complete_ranking
                    else None
                ),
                "maximum_absolute_rank_shift": (
                    max((abs(shift) for shift in shifts.values()), default=0.0)
                    if complete_ranking
                    else None
                ),
                "model_scores_and_ranks": [
                    {
                        "model": model,
                        "EHQ": values[model],
                        "rank": ranks.get(model),
                        "official_EHQ": official[model],
                        "official_rank": official_ranks[model],
                        "rank_shift": shifts.get(model),
                    }
                    for model in models
                ],
            }
        )

    total_correct = sum(confident_correct.values())
    return {
        "scenarios": rows,
        "confident_correct_by_category": dict(sorted(confident_correct.items())),
        "n_confident_correct": total_correct,
        "hnq_share_of_confident_correct": (
            confident_correct.get("HNQ", 0) / total_correct
            if total_correct
            else None
        ),
        "scenario_details": scenario_details,
    }
