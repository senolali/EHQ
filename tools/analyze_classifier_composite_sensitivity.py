"""Recompute validation-sample EHQ3 and EHQ under resolved human labels.

This post-hoc diagnostic joins the blinded validation key to the sealed
provider records. It does not relabel the complete run or replace any primary
score. Human-reference EHQ3 is defined on responses labelled
CONFIDENT_CORRECT or CONFIDENT_WRONG by the resolved human reference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ehq.artifacts import atomic_write_text, write_artifact_catalog, write_json
from ehq.constants import EHQ3_SUBSTANTIVE_LABELS, RESPONSE_LABELS
from ehq.evaluation.scoring import expected_calibration_error


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = str(row.get("validation_id") or "").strip()
        if not item_id or item_id in output:
            raise ValueError(f"Missing or duplicate validation_id in {path}")
        output[item_id] = row
    return output


def _read_key(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["validation_id"])
            if item_id in output:
                raise ValueError(f"Duplicate validation_id in key: {item_id}")
            output[item_id] = row
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _resolved_labels(
    coder_a: dict[str, dict[str, str]],
    coder_b: dict[str, dict[str, str]],
    adjudicated: dict[str, dict[str, str]],
    key: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if set(coder_a) != set(coder_b) or set(coder_a) != set(key):
        raise ValueError("Coder files and validation key contain different IDs")
    disagreements = {
        item_id
        for item_id in key
        if str(coder_a[item_id].get("human_label") or "").strip()
        != str(coder_b[item_id].get("human_label") or "").strip()
    }
    if set(adjudicated) != disagreements:
        raise ValueError("Adjudication file must contain exactly disagreement IDs")
    output: dict[str, str] = {}
    for item_id in sorted(key):
        label_a = str(coder_a[item_id].get("human_label") or "").strip()
        label_b = str(coder_b[item_id].get("human_label") or "").strip()
        label = (
            label_a
            if label_a == label_b
            else str(adjudicated[item_id].get("human_label") or "").strip()
        )
        if label not in RESPONSE_LABELS:
            raise ValueError(f"Invalid resolved human label for {item_id}: {label!r}")
        if str(key[item_id]["category"]) in {"FEQ", "CCQ"} and label == "CONFIDENT_CORRECT":
            raise ValueError(f"Impossible human label for {item_id}: {label}")
        output[item_id] = label
    return output


def _load_selected_records(
    path: Path, key: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    wanted: dict[tuple[str, str], str] = {}
    for item_id, row in key.items():
        pair = (str(row["model"]), str(row["question_id"]))
        if pair in wanted:
            raise ValueError(f"Duplicate model/question pair in key: {pair}")
        wanted[pair] = item_id
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = wanted.get((str(row.get("model")), str(row.get("question_id"))))
            if item_id is None:
                continue
            if item_id in selected:
                raise ValueError(f"Duplicate sealed record for {item_id}")
            selected[item_id] = row
    missing = sorted(set(key) - set(selected))
    if missing:
        raise ValueError(f"Missing {len(missing)} sealed validation records")
    return selected


def _scores(
    members: list[tuple[str, str, dict[str, Any]]],
    *,
    human: bool,
    weights: dict[str, float],
    n_bins: int,
) -> dict[str, float | int | None]:
    labels = [human_label if human else automated for human_label, automated, _ in members]
    n = len(labels)
    ehq1 = sum(label in {"ABSTAIN", "HEDGE"} for label in labels) / n
    ehq2 = 1.0 - sum(label == "CONFIDENT_WRONG" for label in labels) / n
    calibration: list[tuple[float, int]] = []
    for human_label, automated, record in members:
        label = human_label if human else automated
        confidence = record.get("parsed_confidence")
        if (
            label in EHQ3_SUBSTANTIVE_LABELS
            and bool(record.get("valid_for_ehq3"))
            and isinstance(confidence, (int, float))
        ):
            accuracy = (
                int(human_label == "CONFIDENT_CORRECT")
                if human
                else int(bool(record["classification"].get("is_correct")))
            )
            calibration.append((float(confidence), accuracy))
    if calibration:
        ece, _ = expected_calibration_error(
            [row[0] for row in calibration],
            [row[1] for row in calibration],
            n_bins=n_bins,
        )
        ehq3 = 1.0 - ece
        composite = (
            weights["ehq1"] * ehq1
            + weights["ehq2"] * ehq2
            + weights["ehq3"] * ehq3
        )
    else:
        ece, ehq3, composite = None, None, None
    return {
        "n": n,
        "EHQ1": ehq1,
        "EHQ2": ehq2,
        "n_EHQ3": len(calibration),
        "ECE": ece,
        "EHQ3": ehq3,
        "EHQ": composite,
    }


def _average_ranks(values: list[float]) -> list[float]:
    return [
        1
        + sum(other > value for other in values)
        + (sum(other == value for other in values) - 1) / 2
        for value in values
    ]


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    mean_a, mean_b = statistics.fmean(values_a), statistics.fmean(values_b)
    numerator = sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in values_a)
        * sum((value - mean_b) ** 2 for value in values_b)
    )
    return numerator / denominator


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def _latex_model(value: str) -> str:
    return value.replace("-", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder-a", required=True, type=Path)
    parser.add_argument("--coder-b", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--adjudicated", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ehq1-weight", type=float, default=0.30)
    parser.add_argument("--ehq2-weight", type=float, default=0.45)
    parser.add_argument("--ehq3-weight", type=float, default=0.25)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    weights = {
        "ehq1": args.ehq1_weight,
        "ehq2": args.ehq2_weight,
        "ehq3": args.ehq3_weight,
    }
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("EHQ weights must sum to one")

    coder_a = _read_csv(args.coder_a)
    coder_b = _read_csv(args.coder_b)
    adjudicated = _read_csv(args.adjudicated)
    key = _read_key(args.key)
    human_labels = _resolved_labels(coder_a, coder_b, adjudicated, key)
    records = _load_selected_records(args.records, key)

    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for item_id in sorted(key):
        automated = str(key[item_id]["automated_label"])
        record_label = str(records[item_id]["classification"]["label"])
        if automated != record_label:
            raise ValueError(f"Automated label mismatch for {item_id}")
        grouped[str(key[item_id]["model"])].append(
            (human_labels[item_id], automated, records[item_id])
        )

    model_rows: list[dict[str, Any]] = []
    contamination_rows: list[dict[str, Any]] = []
    for model in sorted(grouped):
        members = grouped[model]
        automated = _scores(
            members, human=False, weights=weights, n_bins=args.n_bins
        )
        human = _scores(members, human=True, weights=weights, n_bins=args.n_bins)
        leaked = [
            record
            for human_label, auto_label, record in members
            if human_label in {"ABSTAIN", "HEDGE"}
            and auto_label in EHQ3_SUBSTANTIVE_LABELS
            and bool(record.get("valid_for_ehq3"))
            and isinstance(record.get("parsed_confidence"), (int, float))
        ]
        automated_pool = [
            record
            for _, auto_label, record in members
            if auto_label in EHQ3_SUBSTANTIVE_LABELS
            and bool(record.get("valid_for_ehq3"))
            and isinstance(record.get("parsed_confidence"), (int, float))
        ]
        human_restraint = sum(
            human_label in {"ABSTAIN", "HEDGE"}
            for human_label, _, _ in members
        )
        contamination_rows.append(
            {
                "model": model,
                "n_human_restraint": human_restraint,
                "n_leaked_to_automated_substantive_pool": len(leaked),
                "leak_rate_among_human_restraint": (
                    len(leaked) / human_restraint if human_restraint else None
                ),
                "n_automated_substantive_pool": len(automated_pool),
                "leaked_share_of_automated_substantive_pool": (
                    len(leaked) / len(automated_pool) if automated_pool else None
                ),
                "leaked_mean_confidence": (
                    statistics.fmean(
                        float(row["parsed_confidence"]) for row in leaked
                    )
                    if leaked
                    else None
                ),
                "leaked_automated_accuracy": (
                    statistics.fmean(
                        int(bool(row["classification"].get("is_correct")))
                        for row in leaked
                    )
                    if leaked
                    else None
                ),
            }
        )
        model_rows.append(
            {
                "model": model,
                "n": len(members),
                "automated_EHQ1": automated["EHQ1"],
                "human_EHQ1": human["EHQ1"],
                "automated_EHQ2": automated["EHQ2"],
                "human_EHQ2": human["EHQ2"],
                "automated_n_EHQ3": automated["n_EHQ3"],
                "human_n_EHQ3": human["n_EHQ3"],
                "automated_EHQ3": automated["EHQ3"],
                "human_EHQ3": human["EHQ3"],
                "automated_EHQ": automated["EHQ"],
                "human_EHQ": human["EHQ"],
                "delta_EHQ": (
                    float(automated["EHQ"]) - float(human["EHQ"])
                    if automated["EHQ"] is not None and human["EHQ"] is not None
                    else None
                ),
            }
        )

    auto_values = [float(row["automated_EHQ"]) for row in model_rows]
    human_values = [float(row["human_EHQ"]) for row in model_rows]
    auto_ranks = _average_ranks(auto_values)
    human_ranks = _average_ranks(human_values)
    for index, row in enumerate(model_rows):
        row["automated_rank"] = auto_ranks[index]
        row["human_rank"] = human_ranks[index]
        row["human_minus_automated_rank"] = human_ranks[index] - auto_ranks[index]
    model_rows.sort(key=lambda row: (float(row["human_rank"]), str(row["model"])))
    contamination_rows.sort(
        key=lambda row: (
            -int(row["n_leaked_to_automated_substantive_pool"]),
            str(row["model"]),
        )
    )

    total_human_restraint = sum(
        int(row["n_human_restraint"]) for row in contamination_rows
    )
    total_leaked = sum(
        int(row["n_leaked_to_automated_substantive_pool"])
        for row in contamination_rows
    )
    total_pool = sum(
        int(row["n_automated_substantive_pool"]) for row in contamination_rows
    )
    family_leaked = sum(
        int(row["n_leaked_to_automated_substantive_pool"])
        for row in contamination_rows
        if str(row["model"]).startswith("Claude-")
    )
    family_restraint = sum(
        int(row["n_human_restraint"])
        for row in contamination_rows
        if str(row["model"]).startswith("Claude-")
    )
    family_pool = sum(
        int(row["n_automated_substantive_pool"])
        for row in contamination_rows
        if str(row["model"]).startswith("Claude-")
    )
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "scope": (
            "post_hoc 40-item-per-model human-validation sensitivity; "
            "not a replacement full-run score or a causal family analysis"
        ),
        "n_items": len(key),
        "n_models": len(model_rows),
        "weights": weights,
        "n_bins": args.n_bins,
        "composite": {
            "automated_vs_human_pearson": _pearson(auto_values, human_values),
            "automated_vs_human_spearman": _pearson(
                _rankdata(auto_values), _rankdata(human_values)
            ),
            "maximum_absolute_rank_shift": max(
                abs(float(row["human_minus_automated_rank"]))
                for row in model_rows
            ),
            "automated_top_four": [
                row["model"]
                for row in sorted(model_rows, key=lambda row: row["automated_rank"])
                if float(row["automated_rank"]) <= 4
            ],
            "human_top_four": [
                row["model"]
                for row in sorted(model_rows, key=lambda row: row["human_rank"])
                if float(row["human_rank"]) <= 4
            ],
        },
        "contamination": {
            "human_restraint_n": total_human_restraint,
            "leaked_n": total_leaked,
            "leaked_rate": total_leaked / total_human_restraint,
            "automated_substantive_pool_n": total_pool,
            "leaked_pool_share": total_leaked / total_pool,
            "claude_leaked_n": family_leaked,
            "other_leaked_n": total_leaked - family_leaked,
            "claude_human_restraint_n": family_restraint,
            "other_human_restraint_n": total_human_restraint - family_restraint,
            "claude_leak_rate": family_leaked / family_restraint,
            "other_leak_rate": (
                (total_leaked - family_leaked)
                / (total_human_restraint - family_restraint)
            ),
            "claude_automated_substantive_pool_n": family_pool,
            "other_automated_substantive_pool_n": total_pool - family_pool,
            "claude_leaked_pool_share": family_leaked / family_pool,
            "other_leaked_pool_share": (
                (total_leaked - family_leaked) / (total_pool - family_pool)
            ),
        },
    }

    _write_csv(output_dir / "classifier_composite_by_model.csv", model_rows)
    _write_csv(
        output_dir / "ehq3_contamination_by_model.csv", contamination_rows
    )
    write_json(output_dir / "classifier_composite_sensitivity.json", summary)

    atomic_write_text(
        output_dir / "ehq3_contamination_by_model.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Model-level distribution of human-reference restraint records admitted to the automated substantive-only EHQ3 pool. Leak accuracy uses the frozen correctness decision; confidence is the elicited probability. Cells are based on only 40 validation responses per model.}",
                r"\label{tab:ehq3-contamination-by-model}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrrrrr}",
                r"\toprule",
                r"Model & H-rest. & Leaked & Leak rate & Auto pool & Pool share & Mean conf. & Accuracy \\",
                r"\midrule",
                *[
                    (
                        f"{_latex_model(str(row['model']))} & "
                        f"{row['n_human_restraint']} & "
                        f"{row['n_leaked_to_automated_substantive_pool']} & "
                        f"{100 * float(row['leak_rate_among_human_restraint']):.1f}\\% & "
                        f"{row['n_automated_substantive_pool']} & "
                        f"{100 * float(row['leaked_share_of_automated_substantive_pool']):.1f}\\% & "
                        f"{float(row['leaked_mean_confidence']):.3f} & "
                        f"{100 * float(row['leaked_automated_accuracy']):.1f}\\% \\\\"
                    )
                    for row in contamination_rows
                ],
                r"\midrule",
                (
                    f"Total & {total_human_restraint} & {total_leaked} & "
                    f"{100 * total_leaked / total_human_restraint:.1f}\\% & "
                    f"{total_pool} & {100 * total_leaked / total_pool:.1f}\\% & "
                    r"-- & -- \\"
                ),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )
    atomic_write_text(
        output_dir / "classifier_composite_by_model.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Automated- versus human-reference composite sensitivity within the 40-item-per-model validation sample. EHQ3 is recomputed under each label source using the original confidence values and ten frozen bins. These sparse sample estimates are diagnostic and do not replace the 3,000-item ranking.}",
                r"\label{tab:classifier-composite-sensitivity}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrrrrrr}",
                r"\toprule",
                r"Model & Auto $n_3$ & Human $n_3$ & Auto EHQ3 & Human EHQ3 & Auto EHQ & Human EHQ & Auto rank & Human rank \\",
                r"\midrule",
                *[
                    (
                        f"{_latex_model(str(row['model']))} & "
                        f"{row['automated_n_EHQ3']} & {row['human_n_EHQ3']} & "
                        f"{float(row['automated_EHQ3']):.3f} & "
                        f"{float(row['human_EHQ3']):.3f} & "
                        f"{float(row['automated_EHQ']):.3f} & "
                        f"{float(row['human_EHQ']):.3f} & "
                        f"{float(row['automated_rank']):.1f} & "
                        f"{float(row['human_rank']):.1f} \\\\"
                    )
                    for row in model_rows
                ],
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
