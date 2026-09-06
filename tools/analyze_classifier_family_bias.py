"""Post-hoc audit of family-associated response-classifier score bias.

This diagnostic operates only on the completed human-validation artifacts. It
does not relabel the full run or estimate a causal model-family effect.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ehq.artifacts import atomic_write_text, write_artifact_catalog, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Pearson inputs must have equal length of at least two")
    mean_a, mean_b = _mean(values_a), _mean(values_b)
    numerator = sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in values_a)
        * sum((value - mean_b) ** 2 for value in values_b)
    )
    if not denominator:
        raise ValueError("Pearson correlation is undefined for a constant input")
    return numerator / denominator


def _partial_correlation(r_xy: float, r_xz: float, r_yz: float) -> float:
    """Return r(x, y | z) from the three pairwise correlations."""
    denominator = math.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if not denominator:
        raise ValueError("Partial correlation is undefined for collinear inputs")
    return (r_xy - r_xz * r_yz) / denominator


def _exact_group_permutation(
    values: list[float], group_indices: set[int]
) -> tuple[float, float, int, int]:
    all_indices = tuple(range(len(values)))
    group_size = len(group_indices)
    observed = _mean([values[i] for i in group_indices]) - _mean(
        [values[i] for i in all_indices if i not in group_indices]
    )
    differences = []
    for selection in itertools.combinations(all_indices, group_size):
        selected = set(selection)
        differences.append(
            _mean([values[i] for i in selected])
            - _mean([values[i] for i in all_indices if i not in selected])
        )
    n_as_or_more_extreme = sum(
        abs(value) >= abs(observed) - 1e-15 for value in differences
    )
    p_value = n_as_or_more_extreme / len(differences)
    return observed, p_value, n_as_or_more_extreme, len(differences)


def _average_ranks(values: list[float]) -> list[float]:
    return [
        1
        + sum(other > value for other in values)
        + (sum(other == value for other in values) - 1) / 2
        for value in values
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--family-prefix", default="Claude-")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = _read_csv(args.validation_dir / "score_impact_by_model.csv")
    if len(rows) < 4:
        raise ValueError("Family audit requires at least four model rows")
    family_indices = {
        index
        for index, row in enumerate(rows)
        if str(row["model"]).startswith(args.family_prefix)
    }
    if not family_indices or len(family_indices) == len(rows):
        raise ValueError("Family prefix must define two non-empty groups")

    summary_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema_version": "1.1",
        "status": "complete",
        "scope": (
            "post_hoc family-associated contrast in the 40-item-per-model "
            "human-validation sample; not a causal family effect or a full-run correction"
        ),
        "family_prefix": args.family_prefix,
        "n_models": len(rows),
        "n_family": len(family_indices),
        "n_other": len(rows) - len(family_indices),
        "components": {},
    }
    for component in ("EHQ1", "EHQ2"):
        values = [float(row[f"delta_{component}"]) for row in rows]
        family_indicator = [
            1.0 if index in family_indices else 0.0
            for index in range(len(rows))
        ]
        human_scores = [float(row[f"human_{component}"]) for row in rows]
        family_mean = _mean([values[i] for i in family_indices])
        other_mean = _mean(
            [values[i] for i in range(len(values)) if i not in family_indices]
        )
        difference, p_value, n_extreme, n_permutations = (
            _exact_group_permutation(values, family_indices)
        )
        family_bias_r = _pearson(family_indicator, values)
        human_bias_r = _pearson(human_scores, values)
        family_human_r = _pearson(family_indicator, human_scores)
        result = {
            "component": component,
            "family_mean_bias": family_mean,
            "other_mean_bias": other_mean,
            "family_minus_other": difference,
            "exact_label_permutation_p": p_value,
            "n_as_or_more_extreme": n_extreme,
            "n_label_assignments": n_permutations,
            "family_indicator_bias_pearson": family_bias_r,
            "human_score_bias_pearson": human_bias_r,
            "family_indicator_human_score_pearson": family_human_r,
            "family_bias_partial_human_score": _partial_correlation(
                family_bias_r, family_human_r, human_bias_r
            ),
            "human_score_bias_partial_family": _partial_correlation(
                human_bias_r, family_human_r, family_bias_r
            ),
        }
        summary_rows.append(result)
        summary["components"][component] = result

    ranking_rows: list[dict[str, object]] = []
    rank_vectors: dict[str, list[float]] = {}
    for component in ("EHQ1", "EHQ2"):
        for source in ("automated", "human"):
            rank_vectors[f"{source}_{component}"] = _average_ranks(
                [float(row[f"{source}_{component}"]) for row in rows]
            )
    for index, row in enumerate(rows):
        ranking_rows.append(
            {
                "model": row["model"],
                "group": "family" if index in family_indices else "other",
                "automated_EHQ1": float(row["automated_EHQ1"]),
                "human_EHQ1": float(row["human_EHQ1"]),
                "automated_EHQ1_rank": rank_vectors["automated_EHQ1"][index],
                "human_EHQ1_rank": rank_vectors["human_EHQ1"][index],
                "automated_EHQ2": float(row["automated_EHQ2"]),
                "human_EHQ2": float(row["human_EHQ2"]),
                "automated_EHQ2_rank": rank_vectors["automated_EHQ2"][index],
                "human_EHQ2_rank": rank_vectors["human_EHQ2"][index],
            }
        )
    ranking_rows.sort(key=lambda row: (float(row["human_EHQ1_rank"]), row["model"]))

    validation = json.loads(
        (args.validation_dir / "classifier_validation.json").read_text(encoding="utf-8")
    )
    binary = validation["classifier_vs_human_binary"]["confusion_matrix"]
    human_restraint = sum(binary["RESTRAINT"].values())
    false_substantive = int(binary["RESTRAINT"]["SUBSTANTIVE"])
    automated_substantive = (
        false_substantive + int(binary["SUBSTANTIVE"]["SUBSTANTIVE"])
    )
    pool = {
        "human_restraint_n": human_restraint,
        "human_restraint_mislabeled_substantive_n": false_substantive,
        "human_restraint_mislabeled_substantive_rate": false_substantive
        / human_restraint,
        "automated_substantive_pool_n": automated_substantive,
        "automated_substantive_pool_human_restraint_n": false_substantive,
        "automated_substantive_pool_human_restraint_share": false_substantive
        / automated_substantive,
    }
    summary["ehq3_pool_label_sensitivity"] = pool

    _write_csv(output_dir / "classifier_family_bias.csv", summary_rows)
    _write_csv(output_dir / "classifier_human_reference_ranks.csv", ranking_rows)
    write_json(output_dir / "classifier_family_bias.json", summary)

    family_label = args.family_prefix.rstrip("-")
    atomic_write_text(
        output_dir / "classifier_family_bias.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                (
                    r"\caption{Post-hoc family-associated classifier bias in the "
                    r"40-item-per-model validation sample. Bias is automated minus "
                    r"human-reference score; the two-sided exact $p$ permutes the "
                    r"four family labels across fourteen fixed model rows. This is "
                    r"not a causal family-effect test.}"
                ),
                r"\label{tab:classifier-family-bias}",
                r"\small",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                f"Component & {family_label} mean & Other mean & Difference & Extreme/all & Exact $p$ \\\\",
                r"\midrule",
                *[
                    (
                        f"{row['component']} & {row['family_mean_bias']:.3f} & "
                        f"{row['other_mean_bias']:.3f} & "
                        f"{row['family_minus_other']:+.3f} & "
                        f"{row['n_as_or_more_extreme']}/{row['n_label_assignments']} & "
                        f"{row['exact_label_permutation_p']:.3f} \\\\"
                    )
                    for row in summary_rows
                ],
                r"\bottomrule",
                r"\end{tabular}",
                r"\par\vspace{0.6em}",
                r"\begin{tabular}{lrr}",
                r"\toprule",
                r"Bias association & EHQ1 & EHQ2 \\",
                r"\midrule",
                (
                    r"Family indicator, Pearson $r$ & "
                    f"{summary_rows[0]['family_indicator_bias_pearson']:.3f} & "
                    f"{summary_rows[1]['family_indicator_bias_pearson']:.3f} \\\\"
                ),
                (
                    r"Human-reference score, Pearson $r$ & "
                    f"{summary_rows[0]['human_score_bias_pearson']:.3f} & "
                    f"{summary_rows[1]['human_score_bias_pearson']:.3f} \\\\"
                ),
                (
                    r"Family indicator $\mid$ human score, partial $r$ & "
                    f"{summary_rows[0]['family_bias_partial_human_score']:.3f} & "
                    f"{summary_rows[1]['family_bias_partial_human_score']:.3f} \\\\"
                ),
                (
                    r"Human score $\mid$ family indicator, partial $r$ & "
                    f"{summary_rows[0]['human_score_bias_partial_family']:.3f} & "
                    f"{summary_rows[1]['human_score_bias_partial_family']:.3f} \\\\"
                ),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )
    atomic_write_text(
        output_dir / "classifier_human_reference_ranks.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Automated and human-reference EHQ1 within the stratified validation sample. Average ranks are used for ties. Each estimate is based on only 40 responses and is not a replacement full-run ranking.}",
                r"\label{tab:classifier-human-ranks}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                r"Model & Auto EHQ1 & Auto rank & Human EHQ1 & Human rank \\",
                r"\midrule",
                *[
                    (
                        f"{str(row['model']).replace('-', ' ')} & "
                        f"{row['automated_EHQ1']:.3f} & "
                        f"{row['automated_EHQ1_rank']:.1f} & "
                        f"{row['human_EHQ1']:.3f} & "
                        f"{row['human_EHQ1_rank']:.1f} \\\\"
                    )
                    for row in ranking_rows
                ],
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )
    atomic_write_text(
        output_dir / "ehq3_pool_label_sensitivity.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{How binary response-classification error affects the substantive-only EHQ3 pool in the 560-response validation sample. Human-reference restraint includes ABSTAIN and HEDGE; automated substantive includes both confident labels.}",
                r"\label{tab:ehq3-pool-label-sensitivity}",
                r"\small",
                r"\begin{tabular}{lr}",
                r"\toprule",
                r"Quantity & Value \\",
                r"\midrule",
                f"Human-reference restraint records & {human_restraint} \\\\",
                f"Classified substantive by automation & {false_substantive} ({100 * false_substantive / human_restraint:.1f}\\%) \\\\",
                f"Automated substantive pool & {automated_substantive} \\\\",
                f"Human-reference restraint in that pool & {false_substantive} ({100 * false_substantive / automated_substantive:.1f}\\%) \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )
    catalog = write_artifact_catalog(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": "complete",
                "artifact_count": len(catalog["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
