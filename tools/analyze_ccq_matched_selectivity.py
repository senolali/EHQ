"""Matched CCQ selectivity analysis using missing- and restored-span documents.

The capability probe restores the withheld span in 259 released CCQ documents.
For each model and source item, this analysis pairs restraint on the original
missing-span document with correctness on the restored-span document.  It is a
category-specific construct-validity anchor, not a general answerable control
for FEQ, PCQ, or HNQ.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text, write_artifact_catalog, write_json  # noqa: E402

RESTRAINT = {"ABSTAIN", "HEDGE"}


def analyse(
    full_records: Iterable[dict[str, Any]],
    capability_records: Iterable[dict[str, Any]],
    *,
    excluded_models: set[str],
) -> list[dict[str, Any]]:
    missing: dict[tuple[str, str], bool] = {}
    for row in full_records:
        model = str(row.get("model") or "")
        source_id = str(row.get("question_id") or "")
        if (
            model in excluded_models
            or str(row.get("category")) != "CCQ"
            or not row.get("valid_for_ehq12")
        ):
            continue
        label = str((row.get("classification") or {}).get("label") or "")
        key = (model, source_id)
        if key in missing:
            raise ValueError(f"Duplicate full-run record: {model} {source_id}")
        missing[key] = label in RESTRAINT

    restored: dict[tuple[str, str], bool] = {}
    for row in capability_records:
        model = str(row.get("model") or "")
        source_id = str(row.get("source_question_id") or "")
        if model in excluded_models or not row.get("scored"):
            continue
        key = (model, source_id)
        if key in restored:
            raise ValueError(f"Duplicate capability record: {model} {source_id}")
        restored[key] = bool(row.get("is_correct"))

    by_model: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for key in sorted(set(missing) & set(restored)):
        model, _ = key
        by_model[model].append((missing[key], restored[key]))
    if not by_model:
        raise ValueError("No matched CCQ pairs were found")

    rows: list[dict[str, Any]] = []
    for model in sorted(by_model):
        pairs = by_model[model]
        n = len(pairs)
        both_pass = sum(restraint and correct for restraint, correct in pairs)
        restraint_only = sum(restraint and not correct for restraint, correct in pairs)
        restored_only = sum(not restraint and correct for restraint, correct in pairs)
        both_fail = sum(not restraint and not correct for restraint, correct in pairs)
        rows.append(
            {
                "model": model,
                "n_paired": n,
                "missing_span_restraint_rate": sum(x[0] for x in pairs) / n,
                "restored_span_correct_rate": sum(x[1] for x in pairs) / n,
                "joint_pass_rate": both_pass / n,
                "both_pass_count": both_pass,
                "restraint_only_count": restraint_only,
                "restored_correct_only_count": restored_only,
                "both_fail_count": both_fail,
            }
        )
    rows.sort(key=lambda row: (-float(row["joint_pass_rate"]), row["model"]))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_text(value: Any) -> str:
    """Escape the subset needed by model names on Python 3.10+."""

    return str(value).replace("_", r"\_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("capability_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    capability_dir = args.capability_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    excluded = {
        str(entry["model"])
        for entry in ((manifest.get("run") or {}).get("analysis_exclusions") or [])
    }
    full_records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    capability_records = [
        json.loads(line)
        for line in (capability_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = analyse(
        full_records,
        capability_records,
        excluded_models=excluded,
    )
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "scope": (
            "matched CCQ construct-validity anchor only; missing-span labels "
            "are deterministic-classifier outputs; not a general utility control"
        ),
        "n_models": len(rows),
        "excluded_models": sorted(excluded),
        "full_run_dir": str(run_dir),
        "capability_run_dir": str(capability_dir),
        "n_paired_min": min(int(row["n_paired"]) for row in rows),
        "n_paired_max": max(int(row["n_paired"]) for row in rows),
        "pooled_missing_span_restraint_rate": sum(
            float(row["missing_span_restraint_rate"]) * int(row["n_paired"])
            for row in rows
        )
        / sum(int(row["n_paired"]) for row in rows),
        "pooled_restored_span_correct_rate": sum(
            float(row["restored_span_correct_rate"]) * int(row["n_paired"])
            for row in rows
        )
        / sum(int(row["n_paired"]) for row in rows),
        "pooled_joint_pass_rate": sum(
            int(row["both_pass_count"]) for row in rows
        )
        / sum(int(row["n_paired"]) for row in rows),
    }

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "ccq_matched_selectivity.json",
        {"summary": summary, "models": rows},
    )
    _write_csv(output_dir / "ccq_matched_selectivity.csv", rows)
    _write_csv(output_dir / "ccq_matched_selectivity_summary.csv", [summary])
    latex_rows = "\n".join(
        f"{_latex_text(row['model'])} & {row['n_paired']} & "
        f"{100 * float(row['missing_span_restraint_rate']):.1f}\\% & "
        f"{100 * float(row['restored_span_correct_rate']):.1f}\\% & "
        f"{100 * float(row['joint_pass_rate']):.1f}\\% \\\\"
        for row in rows
    )
    atomic_write_text(
        output_dir / "ccq_matched_selectivity.tex",
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\caption{Matched CCQ selectivity anchor. Each pair uses the original "
        "missing-span document and the capability-probe version with that span "
        "restored. Joint pass requires restraint on the former and a correct "
        "answer on the latter.}\n"
        "\\label{tab:ccq-matched-selectivity}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Model & Paired $n$ & Missing: restraint & Restored: correct & "
        "Joint pass \\\\\n\\midrule\n"
        f"{latex_rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
    )
    catalog = write_artifact_catalog(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_models": len(rows),
                "artifact_count": len(catalog["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
