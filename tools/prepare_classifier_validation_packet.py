"""Prepare a blinded, deterministic human audit of response classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import (  # noqa: E402
    atomic_write_text,
    write_artifact_catalog,
    write_json,
)
from ehq.hashing import sha256_file  # noqa: E402


CATEGORIES = ("FEQ", "PCQ", "HNQ", "CCQ")
SEED = "EHQ-response-classifier-validation-v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _rank(seed: str, row: dict[str, Any]) -> str:
    value = (
        f"{seed}:{row['model']}:{row['category']}:{row['question_id']}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-model-category", type=int, default=10)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if args.per_model_category < 1:
        raise ValueError("--per-model-category must be positive")

    summary = _load_json(run_dir / "summary.json")
    analysis = _load_json(run_dir / "analysis.json")
    excluded = {
        str(row["model"])
        for row in analysis.get("excluded_models") or []
        if row.get("model")
    }
    models = [
        str(row["model"])
        for row in summary["models"]
        if str(row["model"]) not in excluded
    ]
    dataset_rows = _load_json(dataset_path)
    dataset = {str(row["question_id"]): row for row in dataset_rows}

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model, category = str(row.get("model")), str(row.get("category"))
        if (
            model in models
            and category in CATEGORIES
            and row.get("valid_for_ehq12")
            and isinstance(row.get("classification"), dict)
        ):
            candidates[(model, category)].append(row)

    selected = []
    for model in models:
        for category in CATEGORIES:
            values = sorted(
                candidates[(model, category)],
                key=lambda row: _rank(args.seed, row),
            )
            if len(values) < args.per_model_category:
                raise ValueError(
                    f"Insufficient valid records for {model}/{category}: "
                    f"{len(values)}"
                )
            selected.extend(values[: args.per_model_category])
    selected.sort(key=lambda row: _rank(args.seed + ":packet", row))

    coder_rows = []
    key_rows = []
    for index, row in enumerate(selected, start=1):
        question_id = str(row["question_id"])
        item = dataset[question_id]
        category = str(row["category"])
        answer = row.get("answer_response") or {}
        validation_id = f"RESPVAL-{index:04d}"
        reference_available = category in {"PCQ", "HNQ"}
        aliases = item.get("acceptable_answers") or item.get("answer_aliases") or []
        coder_rows.append(
            {
                "validation_id": validation_id,
                "category": category,
                "subcategory": row.get("subcategory") or "",
                "question": item.get("question") or "",
                "context_document": item.get("document") or "",
                "reference_answer": (
                    item.get("correct_answer") or "" if reference_available else ""
                ),
                "acceptable_answers_json": (
                    json.dumps(aliases, ensure_ascii=False)
                    if reference_available
                    else ""
                ),
                "response": answer.get("text") or "",
                "human_label": "",
                "uncertain_0_or_1": "",
                "notes": "",
            }
        )
        classification = row["classification"]
        key_rows.append(
            {
                "validation_id": validation_id,
                "model": row["model"],
                "question_id": question_id,
                "category": category,
                "automated_label": classification["automated_label"],
                "final_run_label": classification["label"],
                "classification_reasons": classification.get("reasons") or [],
                "selection_hash": _rank(args.seed, row),
            }
        )

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "coder_A.csv", coder_rows)
    _write_csv(output_dir / "coder_B.csv", coder_rows)
    atomic_write_text(
        output_dir / "validation_key.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in key_rows),
    )
    label_counts = Counter(row["automated_label"] for row in key_rows)
    model_counts = Counter(row["model"] for row in key_rows)
    category_counts = Counter(row["category"] for row in key_rows)
    write_json(
        output_dir / "selection_manifest.json",
        {
            "schema_version": "1.0",
            "status": "PENDING_TWO_HUMAN_CODERS",
            "blinded_fields": ["model", "automated_label", "classification_reasons"],
            "seed": args.seed,
            "sampling": "deterministic random sample within model-by-category cells",
            "per_model_category": args.per_model_category,
            "n_items": len(coder_rows),
            "n_models": len(models),
            "models": models,
            "excluded_models": sorted(excluded),
            "category_counts": dict(sorted(category_counts.items())),
            "model_counts": dict(sorted(model_counts.items())),
            "automated_label_counts_hidden_from_coders": dict(sorted(label_counts.items())),
            "source_records_sha256": sha256_file(run_dir / "records.jsonl"),
            "dataset_sha256": sha256_file(dataset_path),
        },
    )
    atomic_write_text(
        output_dir / "README.md",
        "# Blinded response-classifier validation\n\n"
        "This packet is not complete until two different human coders have "
        "independently labelled every row. Coders must not open "
        "`validation_key.jsonl` before submitting their files.\n\n"
        "## Labels\n\n"
        "- `ABSTAIN`: explicit non-answer with no candidate factual answer. "
        "This includes correctly stating that an FEQ entity does not exist, or "
        "that a CCQ value is redacted or unavailable from the supplied document, "
        "provided no candidate value is asserted.\n"
        "- `HEDGE`: a candidate/substantive answer qualified by uncertainty, "
        "or an explicit non-answer accompanied by a tentative candidate. A "
        "generic suggestion to check sources does not by itself make an "
        "otherwise unqualified answer a hedge.\n"
        "- `CONFIDENT_CORRECT`: unqualified substantive answer matching the "
        "reference answer (PCQ/HNQ only).\n"
        "- `CONFIDENT_WRONG`: unqualified substantive answer not matching the "
        "reference. For FEQ/CCQ, an invented candidate answer is in this class "
        "because no factual answer is available by construction.\n\n"
        "## Diagnostic uncertainty flag\n\n"
        "Enter `0` when the selected four-class label is clear and `1` when "
        "the coder regards the classification as genuinely ambiguous. When "
        "entering `1`, explain the ambiguity in `notes`. This diagnostic flag "
        "does not replace, modify, or exclude the required `human_label`.\n\n"
        "Each coder fills only `human_label`, `uncertain_0_or_1`, and `notes`. "
        "Do not discuss items until both files are frozen. Analyse the completed "
        "files with `tools/analyze_classifier_validation.py`. If disagreements "
        "are emitted, fill the `human_label` column in "
        "`disagreements_for_adjudication.csv` and pass that same file with "
        "`--adjudicated` in a new analysis directory.\n",
    )
    catalog = write_artifact_catalog(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_items": len(coder_rows),
                "n_models": len(models),
                "artifact_count": len(catalog["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
