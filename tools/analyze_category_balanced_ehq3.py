"""Post-hoc category-balanced EHQ3 sensitivity analysis.

Official EHQ3 pools every substantive answer, so the four categories receive
weights proportional to the number of substantive answers a model produced in
each category.  This diagnostic instead computes protocol-identical EHQ3 within
each category and averages the four category scores equally.  It is a
sensitivity analysis, not a replacement for the frozen confirmatory metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text, write_artifact_catalog, write_json  # noqa: E402
from ehq.evaluation.scoring import compute_ehq_scores  # noqa: E402

CATEGORIES = ("FEQ", "PCQ", "HNQ", "CCQ")


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Correlation inputs must have equal length")
    if len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - mean_left) ** 2 for value in left)
        * sum((value - mean_right) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _rankdata(values: list[float], *, descending: bool = False) -> list[float]:
    order = sorted(
        range(len(values)),
        key=lambda index: values[index],
        reverse=descending,
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "--"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _latex_text(value: Any) -> str:
    """Escape the subset needed by model names on Python 3.10+."""

    return str(value).replace("_", r"\_")


def analyse(
    records: Iterable[dict[str, Any]],
    *,
    weights: dict[str, float],
    n_bins: int,
    excluded_models: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        model = str(row.get("model") or "")
        if model and model not in excluded_models:
            grouped[model].append(row)
    if not grouped:
        raise ValueError("No included model records were found")

    rows: list[dict[str, Any]] = []
    for model in sorted(grouped):
        model_records = grouped[model]
        official = compute_ehq_scores(
            model_records, weights=weights, n_bins=n_bins
        )
        category_scores: dict[str, dict[str, Any]] = {}
        for category in CATEGORIES:
            members = [
                row for row in model_records if str(row.get("category")) == category
            ]
            if not members:
                raise ValueError(f"{model}: missing category {category}")
            category_scores[category] = compute_ehq_scores(
                members, weights=weights, n_bins=n_bins
            )
        missing = [
            category
            for category, score in category_scores.items()
            if score.get("EHQ3") is None
        ]
        if missing:
            raise ValueError(
                f"{model}: category-balanced EHQ3 undefined for {', '.join(missing)}"
            )
        balanced_ehq3 = sum(
            float(category_scores[category]["EHQ3"]) for category in CATEGORIES
        ) / len(CATEGORIES)
        balanced_composite = (
            float(weights["ehq1"]) * float(official["EHQ1"])
            + float(weights["ehq2"]) * float(official["EHQ2"])
            + float(weights["ehq3"]) * balanced_ehq3
        )
        row: dict[str, Any] = {
            "model": model,
            "official_EHQ3": official["EHQ3"],
            "category_balanced_EHQ3": balanced_ehq3,
            "delta_EHQ3": balanced_ehq3 - float(official["EHQ3"]),
            "official_EHQ": official["EHQ"],
            "category_balanced_EHQ": balanced_composite,
            "delta_EHQ": balanced_composite - float(official["EHQ"]),
        }
        for category in CATEGORIES:
            row[f"{category}_EHQ3"] = category_scores[category]["EHQ3"]
            row[f"{category}_n_calibration"] = category_scores[category][
                "n_ehq3_calibration"
            ]
        rows.append(row)

    official_ehq = [float(row["official_EHQ"]) for row in rows]
    balanced_ehq = [float(row["category_balanced_EHQ"]) for row in rows]
    official_ehq3 = [float(row["official_EHQ3"]) for row in rows]
    balanced_ehq3 = [float(row["category_balanced_EHQ3"]) for row in rows]
    official_ranks = _rankdata(official_ehq, descending=True)
    balanced_ranks = _rankdata(balanced_ehq, descending=True)
    for row, official_rank, balanced_rank in zip(
        rows, official_ranks, balanced_ranks
    ):
        row["official_rank"] = official_rank
        row["category_balanced_rank"] = balanced_rank
        row["rank_shift_balanced_minus_official"] = balanced_rank - official_rank
    rows.sort(key=lambda row: float(row["official_rank"]))

    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "scope": (
            "post_hoc_sensitivity; equal category weights for EHQ3 only; "
            "does not replace the frozen confirmatory score"
        ),
        "categories": list(CATEGORIES),
        "n_models": len(rows),
        "excluded_models": sorted(excluded_models),
        "weights": weights,
        "n_bins": n_bins,
        "ehq3_pearson": _pearson(official_ehq3, balanced_ehq3),
        "ehq3_spearman": _spearman(official_ehq3, balanced_ehq3),
        "composite_pearson": _pearson(official_ehq, balanced_ehq),
        "composite_spearman": _spearman(official_ehq, balanced_ehq),
        "mean_abs_composite_delta": sum(
            abs(float(row["delta_EHQ"])) for row in rows
        )
        / len(rows),
        "max_abs_composite_delta": max(
            abs(float(row["delta_EHQ"])) for row in rows
        ),
        "max_abs_rank_shift": max(
            abs(float(row["rank_shift_balanced_minus_official"])) for row in rows
        ),
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot = (manifest.get("config") or {}).get("snapshot") or {}
    weights = {key: float(value) for key, value in snapshot["weights"].items()}
    n_bins = int((snapshot.get("confidence") or {}).get("n_bins", 10))
    exclusions = {
        str(entry["model"])
        for entry in ((manifest.get("run") or {}).get("analysis_exclusions") or [])
    }
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary, rows = analyse(
        records,
        weights=weights,
        n_bins=n_bins,
        excluded_models=exclusions,
    )
    summary["run_dir"] = str(run_dir)
    summary["source_artifact_catalog"] = str(run_dir / "artifact_catalog.json")

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "category_balanced_ehq3.json",
        {"summary": summary, "models": rows},
    )
    _write_csv(output_dir / "category_balanced_ehq3_by_model.csv", rows)
    _write_csv(output_dir / "category_balanced_ehq3_summary.csv", [summary])

    table_rows = "\n".join(
        f"{_latex_text(row['model'])} & "
        f"{_fmt(float(row['official_EHQ3']))} & "
        f"{_fmt(float(row['category_balanced_EHQ3']))} & "
        f"{_fmt(float(row['delta_EHQ3']), signed=True)} & "
        f"{_fmt(float(row['official_EHQ']))} & "
        f"{_fmt(float(row['category_balanced_EHQ']))} & "
        f"{_fmt(float(row['rank_shift_balanced_minus_official']), digits=0, signed=True)} \\\\"
        for row in rows
    )
    atomic_write_text(
        output_dir / "category_balanced_ehq3_by_model.tex",
        "\\begin{table}[htbp]\n\\centering\n\\scriptsize\n"
        "\\caption{Post-hoc category-balanced EHQ3 sensitivity. Balanced EHQ3 "
        "is the equal-weight mean of the four category-specific EHQ3 values; "
        "the composite retains the frozen component weights. Rank shift is "
        "balanced minus official rank.}\n"
        "\\label{tab:category-balanced-ehq3}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Model & Official EHQ3 & Balanced EHQ3 & $\\Delta$ EHQ3 & "
        "Official EHQ & Balanced EHQ & Rank shift \\\\\n\\midrule\n"
        f"{table_rows}\n\\bottomrule\n\\end{{tabular}}%\n}}\n\\end{{table}}\n",
    )
    atomic_write_text(
        output_dir / "category_balanced_ehq3_summary.tex",
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\caption{Summary of the category-balanced EHQ3 sensitivity.}\n"
        "\\label{tab:category-balanced-ehq3-summary}\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Comparison & Pearson $r$ & Spearman $\\rho$ \\\\\n\\midrule\n"
        f"EHQ3 & {_fmt(summary['ehq3_pearson'])} & "
        f"{_fmt(summary['ehq3_spearman'])} \\\\\n"
        f"Composite EHQ & {_fmt(summary['composite_pearson'])} & "
        f"{_fmt(summary['composite_spearman'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    catalog = write_artifact_catalog(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_models": summary["n_models"],
                "max_abs_rank_shift": summary["max_abs_rank_shift"],
                "artifact_count": len(catalog["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
