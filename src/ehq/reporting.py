"""Protocol-aligned tabular and optional publication-support reports."""

from __future__ import annotations

import csv
import io
import importlib.util
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .artifacts import atomic_write_text, write_json, write_jsonl
from .constants import EHQ3_PROTOCOL


SCORE_FIELDS = (
    "n_input",
    "n_valid_ehq12",
    "n_valid_ehq3",
    "n_ehq3_calibration",
    "n_missing_confidence",
    "n_terminal_missing_confidence",
    "n_retryable_records",
    "confidence_coverage",
    "technical_failures",
    "EHQ1",
    "EHQ2",
    "EHQ3",
    "EHQ",
)


def check_reporting_dependencies(
    *, excel: bool, figures: bool, report: bool = False
) -> None:
    missing = []
    if excel and importlib.util.find_spec("openpyxl") is None:
        missing.append("openpyxl (Excel export)")
    if (figures or report) and importlib.util.find_spec("matplotlib") is None:
        missing.append("matplotlib (figure export)")
    if report and importlib.util.find_spec("numpy") is None:
        missing.append("numpy (publication package bootstrap)")
    if missing:
        raise RuntimeError(
            "Missing optional reporting dependencies: "
            + ", ".join(missing)
            + ". Install ehq[reporting] before starting the run."
        )


def build_summary(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    models = []
    for model, result in results.items():
        scores = result["scores"]
        models.append(
            {
                "model": model,
                **{field: scores.get(field) for field in SCORE_FIELDS},
            }
        )
    models.sort(
        key=lambda row: (
            row["EHQ"] is None,
            -(row["EHQ"] or 0.0),
            row["model"],
        )
    )
    start = 0
    while start < len(models):
        if models[start]["EHQ"] is None:
            for row in models[start:]:
                row["rank"] = None
            break
        end = start + 1
        while end < len(models) and models[end]["EHQ"] == models[start]["EHQ"]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            models[index]["rank"] = average_rank
        start = end
    return {"ehq3_protocol": EHQ3_PROTOCOL, "models": models}


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_standard_reports(
    run_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = build_summary(results)
    write_json(run_dir / "summary.json", summary)
    fields = ("rank", "model", *SCORE_FIELDS)
    atomic_write_text(run_dir / "summary.csv", _csv_text(summary["models"], fields))

    all_records = []
    category_rows = []
    subcategory_rows = []
    for model, result in results.items():
        all_records.extend(result["records"])
        for category, scores in result["category_scores"].items():
            category_rows.append(
                {"model": model, "category": category, **scores}
            )
        for subcategory, scores in result["subcategory_scores"].items():
            subcategory_rows.append(
                {"model": model, "subcategory": subcategory, **scores}
            )
    write_jsonl(run_dir / "records.jsonl", all_records)
    grouped_fields = ("model", "category", *SCORE_FIELDS)
    atomic_write_text(
        run_dir / "category_scores.csv",
        _csv_text(category_rows, grouped_fields),
    )
    grouped_fields = ("model", "subcategory", *SCORE_FIELDS)
    atomic_write_text(
        run_dir / "subcategory_scores.csv",
        _csv_text(subcategory_rows, grouped_fields),
    )
    return summary


def export_excel(
    path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires the 'reporting' optional dependencies"
        ) from exc

    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)

    def add_sheet(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        sheet = workbook.create_sheet(name)
        if not rows:
            sheet.append(["No records"])
            return
        fields = list(rows[0])
        sheet.append(fields)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append([row.get(field) for field in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    summary = build_summary(results)
    add_sheet("Summary", summary["models"])
    category_rows = []
    subcategory_rows = []
    response_rows = []
    failure_rows = []
    for model, result in results.items():
        for category, scores in result["category_scores"].items():
            category_rows.append(
                {
                    "model": model,
                    "category": category,
                    **{field: scores.get(field) for field in SCORE_FIELDS},
                }
            )
        for subcategory, scores in result["subcategory_scores"].items():
            subcategory_rows.append(
                {
                    "model": model,
                    "subcategory": subcategory,
                    **{field: scores.get(field) for field in SCORE_FIELDS},
                }
            )
        response_rows.append(
            {"model": model, **result["scores"]["response_distribution"]}
        )
        failure_rows.extend(
            {
                "model": model,
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "answer_error": (row.get("answer_response") or {}).get("error_type"),
                "confidence_error": (row.get("confidence_response") or {}).get("error_type"),
                "confidence_parse_strategy": row.get("confidence_parse_strategy"),
                "confidence_parse_reason": row.get("confidence_parse_reason"),
                "confidence_terminal": row.get("confidence_terminal"),
                "valid_for_ehq12": row.get("valid_for_ehq12"),
                "valid_for_ehq3": row.get("valid_for_ehq3"),
            }
            for row in result["records"]
            if not row.get("valid_for_ehq3")
        )
    add_sheet("Category Scores", category_rows)
    add_sheet("Subcategory Scores", subcategory_rows)
    add_sheet("Response Types", response_rows)
    add_sheet("Technical Failures", failure_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def export_figures(
    output_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure export requires the 'reporting' optional dependencies"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary(results)["models"]
    names = [row["model"] for row in rows]
    values = [row["EHQ"] or 0.0 for row in rows]
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.55), 5.5))
    axis.bar(names, values, color="#356A8A")
    axis.set_ylim(0, 1)
    axis.set_ylabel("EHQ")
    axis.set_title("Epistemic Honesty Quotient")
    axis.tick_params(axis="x", rotation=70)
    figure.tight_layout()
    figure.savefig(output_dir / "ehq_ranking.png", dpi=200)
    figure.savefig(output_dir / "ehq_ranking.pdf")
    plt.close(figure)
