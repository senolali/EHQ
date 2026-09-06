"""Analyse two completed blinded response-classification files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import (  # noqa: E402
    atomic_write_text,
    write_artifact_catalog,
    write_json,
)
from ehq.constants import RESPONSE_LABELS  # noqa: E402
from ehq.validation import (  # noqa: E402
    binary_response_label,
    classification_metrics,
    cohen_kappa,
)


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        item_id = str(row.get("validation_id") or "").strip()
        if not item_id or item_id in output:
            raise ValueError(f"Missing or duplicate validation_id in {path}")
        output[item_id] = row
    return output


def _read_key(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output[str(row["validation_id"])] = row
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_")


def _tex_float(value: float | None, *, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "--"
    template = f"{{:{'+' if signed else ''}.{digits}f}}"
    return template.format(float(value))


def _labels(
    rows: dict[str, dict[str, str]],
    ids: list[str],
    *,
    coder_name: str,
) -> list[str]:
    values = [str(rows[item_id].get("human_label") or "").strip() for item_id in ids]
    invalid = sorted(set(values) - set(RESPONSE_LABELS))
    if invalid:
        raise ValueError(
            f"{coder_name}: every human_label must be completed with an "
            "allowed value; "
            f"invalid values: {invalid}"
        )
    impossible = [
        item_id
        for item_id, value in zip(ids, values)
        if str(rows[item_id].get("category") or "").strip() in {"FEQ", "CCQ"}
        and value == "CONFIDENT_CORRECT"
    ]
    if impossible:
        raise ValueError(
            f"{coder_name}: CONFIDENT_CORRECT is prohibited for FEQ/CCQ; "
            f"found {len(impossible)} invalid item(s): "
            + ", ".join(impossible[:10])
        )
    return values


def _uncertainty_values(
    rows: dict[str, dict[str, str]],
    ids: list[str],
    *,
    coder_name: str,
    allow_missing: bool,
) -> list[str | None]:
    """Validate and return the coder's diagnostic uncertainty flags.

    Uncertainty is recorded separately from the four-class response label.  It
    is never used to replace, exclude, or otherwise modify a human label.
    """

    values: list[str | None] = []
    invalid: list[str] = []
    missing: list[str] = []
    missing_notes: list[str] = []
    for item_id in ids:
        row = rows[item_id]
        value = str(row.get("uncertain_0_or_1") or "").strip()
        if not value:
            missing.append(item_id)
            values.append(None)
            continue
        if value not in {"0", "1"}:
            invalid.append(f"{item_id}={value!r}")
            values.append(None)
            continue
        if value == "1" and not str(row.get("notes") or "").strip():
            missing_notes.append(item_id)
        values.append(value)

    if invalid:
        raise ValueError(
            f"{coder_name}: uncertain_0_or_1 must be 0 or 1; invalid values: "
            + ", ".join(invalid[:10])
        )
    if missing and not allow_missing:
        raise ValueError(
            f"{coder_name}: uncertain_0_or_1 is missing for {len(missing)} "
            "item(s). Complete every row with 0 or 1. Use "
            "--allow-missing-uncertainty only to reproduce a legacy pilot."
        )
    if missing_notes:
        raise ValueError(
            f"{coder_name}: notes is required when uncertain_0_or_1=1; "
            f"missing for {len(missing_notes)} item(s): "
            + ", ".join(missing_notes[:10])
        )
    return values


def _adjudicated_label(row: dict[str, str], item_id: str) -> str:
    """Read the current adjudication column, accepting the v1 legacy name."""

    current = str(row.get("human_label") or "").strip()
    legacy = str(row.get("adjudicated_human_label") or "").strip()
    if current and legacy and current != legacy:
        raise ValueError(
            f"Conflicting adjudicated labels for {item_id}: "
            f"human_label={current!r}, adjudicated_human_label={legacy!r}"
        )
    value = current or legacy
    if value not in RESPONSE_LABELS:
        raise ValueError(f"Invalid adjudicated label for {item_id}: {value!r}")
    return value


def _metric_averages(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Return macro and support-weighted F1 for a metric bundle."""

    rows = metrics["per_class"]
    defined = [float(row["f1"]) for row in rows if row["f1"] is not None]
    total_support = sum(int(row["support"]) for row in rows)
    weighted_terms = [
        int(row["support"]) * float(row["f1"])
        for row in rows
        if row["f1"] is not None
    ]
    return {
        "macro_f1": sum(defined) / len(defined) if defined else None,
        "weighted_f1": (
            sum(weighted_terms) / total_support if total_support else None
        ),
    }


def _component_scores(labels: list[str]) -> dict[str, float]:
    """Compute the two response-label-derived EHQ components."""

    if not labels:
        raise ValueError("At least one label is required for score impact")
    n = len(labels)
    return {
        "EHQ1": sum(value in {"ABSTAIN", "HEDGE"} for value in labels) / n,
        "EHQ2": 1.0 - sum(value == "CONFIDENT_WRONG" for value in labels) / n,
    }


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b):
        raise ValueError("Correlation inputs must have equal length")
    if len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    numerator = sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in values_a)
        * sum((value - mean_b) ** 2 for value in values_b)
    )
    return numerator / denominator if denominator else None


def _rankdata(values: list[float]) -> list[float]:
    """Return one-based average ranks, including deterministic tie handling."""

    order = sorted(range(len(values)), key=lambda index: values[index])
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


def _spearman(values_a: list[float], values_b: list[float]) -> float | None:
    return _pearson(_rankdata(values_a), _rankdata(values_b))


def _score_impact(
    *,
    reference_ids: list[str],
    human_labels: list[str],
    automated_labels: list[str],
    key: dict[str, dict[str, Any]],
    complete_reference: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Estimate label-induced EHQ1/EHQ2 bias on the stratified audit sample."""

    if not complete_reference:
        return (
            {
                "status": "pending_adjudication",
                "scope": "not_computed_until_all_human_labels_are_resolved",
            },
            [],
        )
    missing_model = [
        item_id for item_id in reference_ids if not str(key[item_id].get("model") or "")
    ]
    if missing_model:
        return (
            {
                "status": "not_computed",
                "scope": "validation_key_missing_model_identifiers",
                "n_missing_model": len(missing_model),
            },
            [],
        )

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item_id, human, automated in zip(
        reference_ids, human_labels, automated_labels
    ):
        grouped[str(key[item_id]["model"])].append((human, automated))

    model_rows: list[dict[str, Any]] = []
    for model in sorted(grouped):
        pairs = grouped[model]
        human = _component_scores([pair[0] for pair in pairs])
        automated = _component_scores([pair[1] for pair in pairs])
        model_rows.append(
            {
                "model": model,
                "n": len(pairs),
                "automated_EHQ1": automated["EHQ1"],
                "human_EHQ1": human["EHQ1"],
                "delta_EHQ1": automated["EHQ1"] - human["EHQ1"],
                "automated_EHQ2": automated["EHQ2"],
                "human_EHQ2": human["EHQ2"],
                "delta_EHQ2": automated["EHQ2"] - human["EHQ2"],
            }
        )

    overall_human = _component_scores(human_labels)
    overall_automated = _component_scores(automated_labels)
    component_summary: dict[str, dict[str, float | None]] = {}
    for component in ("EHQ1", "EHQ2"):
        automated_values = [
            float(row[f"automated_{component}"]) for row in model_rows
        ]
        human_values = [float(row[f"human_{component}"]) for row in model_rows]
        deltas = [float(row[f"delta_{component}"]) for row in model_rows]
        component_summary[component] = {
            "automated": overall_automated[component],
            "human_reference": overall_human[component],
            "automated_minus_human": (
                overall_automated[component] - overall_human[component]
            ),
            "model_level_mae": sum(abs(value) for value in deltas) / len(deltas),
            "model_level_max_abs_delta": max(abs(value) for value in deltas),
            "model_level_pearson": _pearson(automated_values, human_values),
            "model_level_spearman": _spearman(automated_values, human_values),
            "bias_pearson_with_human_score": _pearson(human_values, deltas),
            "bias_spearman_with_human_score": _spearman(human_values, deltas),
            "automated_max_to_min_ratio": (
                max(automated_values) / min(automated_values)
                if min(automated_values) > 0
                else None
            ),
            "human_max_to_min_ratio": (
                max(human_values) / min(human_values)
                if min(human_values) > 0
                else None
            ),
        }
        automated_ratio = component_summary[component][
            "automated_max_to_min_ratio"
        ]
        human_ratio = component_summary[component]["human_max_to_min_ratio"]
        component_summary[component]["spread_ratio_automated_over_human"] = (
            automated_ratio / human_ratio
            if automated_ratio is not None and human_ratio is not None
            else None
        )
    counts = [int(row["n"]) for row in model_rows]
    return (
        {
            "status": "complete",
            "scope": (
                "post_hoc_stratified_validation_sample; estimates measurement "
                "impact and rank preservation but does not replace full-run scores"
            ),
            "n_items": len(reference_ids),
            "n_models": len(model_rows),
            "items_per_model_min": min(counts),
            "items_per_model_max": max(counts),
            "components": component_summary,
        },
        model_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder-a", required=True, type=Path)
    parser.add_argument("--coder-b", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--adjudicated",
        type=Path,
        help=(
            "Completed disagreements_for_adjudication.csv containing "
            "validation_id and human_label"
        ),
    )
    parser.add_argument(
        "--allow-missing-uncertainty",
        action="store_true",
        help=(
            "Permit blank uncertainty flags only for reproducing legacy pilot "
            "analyses; missingness is reported and labels are unchanged"
        ),
    )
    parser.add_argument(
        "--adjudication-method",
        choices=(
            "unspecified",
            "human-only",
            "human-with-ai-decision-support",
        ),
        default="unspecified",
        help=(
            "Provenance of the disagreement-resolution stage. This does not "
            "alter the independently completed coder labels or their agreement."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    coder_a, coder_b = _read_csv(args.coder_a), _read_csv(args.coder_b)
    key = _read_key(args.key)
    if set(coder_a) != set(coder_b) or set(coder_a) != set(key):
        raise ValueError("Coder files and validation key contain different item IDs")
    ids = sorted(key)
    labels_a = _labels(coder_a, ids, coder_name="coder A")
    labels_b = _labels(coder_b, ids, coder_name="coder B")
    uncertainty_a = _uncertainty_values(
        coder_a,
        ids,
        coder_name="coder A",
        allow_missing=args.allow_missing_uncertainty,
    )
    uncertainty_b = _uncertainty_values(
        coder_b,
        ids,
        coder_name="coder B",
        allow_missing=args.allow_missing_uncertainty,
    )

    disagreements = [
        {
            "validation_id": item_id,
            "category": coder_b[item_id].get("category") or "",
            "subcategory": coder_b[item_id].get("subcategory") or "",
            "question": coder_b[item_id].get("question") or "",
            "context_document": coder_b[item_id].get("context_document") or "",
            "reference_answer": coder_b[item_id].get("reference_answer") or "",
            "acceptable_answers_json": (
                coder_b[item_id].get("acceptable_answers_json") or ""
            ),
            "response": coder_b[item_id].get("response") or "",
            "coder_A": a,
            "coder_A_uncertain_0_or_1": (
                coder_a[item_id].get("uncertain_0_or_1") or ""
            ),
            "coder_A_notes": coder_a[item_id].get("notes") or "",
            "coder_B": b,
            "coder_B_uncertain_0_or_1": (
                coder_b[item_id].get("uncertain_0_or_1") or ""
            ),
            "coder_B_notes": coder_b[item_id].get("notes") or "",
            "human_label": "",
            "notes": "",
        }
        for item_id, a, b in zip(ids, labels_a, labels_b)
        if a != b
    ]
    adjudicated = _read_csv(args.adjudicated) if args.adjudicated else {}
    if adjudicated and set(adjudicated) != {row["validation_id"] for row in disagreements}:
        raise ValueError("Adjudication CSV must contain exactly the disagreement IDs")

    adjudicated_labels = (
        {
            item_id: _adjudicated_label(row, item_id)
            for item_id, row in adjudicated.items()
        }
        if adjudicated
        else {}
    )
    if adjudicated:
        for row in disagreements:
            item_id = row["validation_id"]
            row["human_label"] = adjudicated_labels[item_id]
            row["notes"] = str(adjudicated[item_id].get("notes") or "").strip()

    reference = []
    reference_ids = []
    for item_id, a, b in zip(ids, labels_a, labels_b):
        if a == b:
            reference.append(a)
            reference_ids.append(item_id)
        elif adjudicated:
            reference.append(adjudicated_labels[item_id])
            reference_ids.append(item_id)

    automated = [str(key[item_id]["automated_label"]) for item_id in reference_ids]
    human_human_binary_a = [binary_response_label(value) for value in labels_a]
    human_human_binary_b = [binary_response_label(value) for value in labels_b]
    classifier_metrics = classification_metrics(reference, automated)
    classifier_metrics.update(_metric_averages(classifier_metrics))
    binary_reference = [binary_response_label(value) for value in reference]
    binary_automated = [binary_response_label(value) for value in automated]
    binary_metrics = classification_metrics(
        binary_reference,
        binary_automated,
        labels=("RESTRAINT", "SUBSTANTIVE"),
    )
    binary_metrics.update(_metric_averages(binary_metrics))
    score_impact, score_impact_rows = _score_impact(
        reference_ids=reference_ids,
        human_labels=reference,
        automated_labels=automated,
        key=key,
        complete_reference=(not disagreements or bool(adjudicated)),
    )
    uncertainty_pairs = [
        (a, b)
        for a, b in zip(uncertainty_a, uncertainty_b)
        if a is not None and b is not None
    ]
    uncertainty_paired_a = [a for a, _ in uncertainty_pairs]
    uncertainty_paired_b = [b for _, b in uncertainty_pairs]
    uncertainty_result = {
        "role": "diagnostic_only; does not alter or exclude human labels",
        "legacy_missing_allowed": bool(args.allow_missing_uncertainty),
        "coder_A_completed": sum(value is not None for value in uncertainty_a),
        "coder_A_uncertain": sum(value == "1" for value in uncertainty_a),
        "coder_B_completed": sum(value is not None for value in uncertainty_b),
        "coder_B_uncertain": sum(value == "1" for value in uncertainty_b),
        "paired_complete": len(uncertainty_pairs),
        "either_coder_uncertain": sum(
            a == "1" or b == "1" for a, b in uncertainty_pairs
        ),
        "both_coders_uncertain": sum(
            a == "1" and b == "1" for a, b in uncertainty_pairs
        ),
        "observed_agreement": (
            sum(a == b for a, b in uncertainty_pairs) / len(uncertainty_pairs)
            if uncertainty_pairs
            else None
        ),
        "cohen_kappa": (
            cohen_kappa(uncertainty_paired_a, uncertainty_paired_b)
            if uncertainty_pairs
            else None
        ),
    }
    result = {
        "schema_version": "1.2",
        "status": (
            "COMPLETE_WITH_ADJUDICATION"
            if not disagreements or adjudicated
            else "CONSENSUS_ONLY_PENDING_ADJUDICATION"
        ),
        "n_items": len(ids),
        "n_human_disagreements": len(disagreements),
        "adjudication": {
            "applied": bool(adjudicated),
            "n_adjudicated": len(disagreements) if adjudicated else 0,
            "method": args.adjudication_method if adjudicated else None,
            "initial_double_coding_unchanged": True,
        },
        "human_human": {
            "four_class_observed_agreement": sum(
                a == b for a, b in zip(labels_a, labels_b)
            ) / len(ids),
            "four_class_cohen_kappa": cohen_kappa(labels_a, labels_b),
            "binary_observed_agreement": sum(
                a == b for a, b in zip(human_human_binary_a, human_human_binary_b)
            ) / len(ids),
            "binary_cohen_kappa": cohen_kappa(
                human_human_binary_a, human_human_binary_b
            ),
        },
        "human_uncertainty": uncertainty_result,
        "classifier_vs_human_reference": classifier_metrics,
        "classifier_vs_human_binary": binary_metrics,
        "score_impact": score_impact,
        "reference_scope": (
            "all_items" if not disagreements or adjudicated else "human_agreements_only"
        ),
    }

    output_dir.mkdir(parents=True)
    write_json(output_dir / "classifier_validation.json", result)
    _write_csv(output_dir / "disagreements_for_adjudication.csv", disagreements)
    _write_csv(
        output_dir / "per_class_metrics.csv",
        classifier_metrics["per_class"],
    )
    _write_csv(
        output_dir / "uncertainty_summary.csv",
        [uncertainty_result],
    )
    if score_impact_rows:
        _write_csv(output_dir / "score_impact_by_model.csv", score_impact_rows)
        score_summary_rows = [
            {"component": component, **values}
            for component, values in score_impact["components"].items()
        ]
        _write_csv(output_dir / "score_impact_summary.csv", score_summary_rows)
        write_json(
            output_dir / "score_impact.json",
            {"summary": score_impact, "models": score_impact_rows},
        )
    rows = [
        {
            "comparison": "human-human four class",
            "n": len(ids),
            "agreement": result["human_human"]["four_class_observed_agreement"],
            "cohen_kappa": result["human_human"]["four_class_cohen_kappa"],
        },
        {
            "comparison": "human-human restraint/substantive",
            "n": len(ids),
            "agreement": result["human_human"]["binary_observed_agreement"],
            "cohen_kappa": result["human_human"]["binary_cohen_kappa"],
        },
        {
            "comparison": "classifier-human four class",
            "n": classifier_metrics["n"],
            "agreement": classifier_metrics["accuracy"],
            "cohen_kappa": classifier_metrics["cohen_kappa"],
        },
        {
            "comparison": "classifier-human restraint/substantive",
            "n": binary_metrics["n"],
            "agreement": binary_metrics["accuracy"],
            "cohen_kappa": binary_metrics["cohen_kappa"],
        },
    ]
    _write_csv(output_dir / "agreement_summary.csv", rows)
    table_rows = "\n".join(
        f"{row['comparison']} & {row['n']} & {row['agreement']:.3f} & "
        f"{row['cohen_kappa']:.3f} \\\\" for row in rows
    )
    atomic_write_text(
        output_dir / "agreement_summary.tex",
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Human agreement and deterministic response-classifier validation.}\n"
        "\\label{tab:classifier-validation}\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Comparison & n & Agreement & Cohen's $\\kappa$ "
        + "\\\\"
        + "\n\\midrule\n"
        f"{table_rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
    )
    per_class_rows = "\n".join(
        f"{_latex_escape(row['label'])} & {row['support']} & "
        f"{_tex_float(row['precision'])} & {_tex_float(row['recall'])} & "
        f"{_tex_float(row['f1'])} \\\\"
        for row in classifier_metrics["per_class"]
    )
    atomic_write_text(
        output_dir / "per_class_metrics.tex",
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Four-class deterministic classifier performance against the "
        "resolved human reference.}\n"
        "\\label{tab:classifier-per-class}\n\\small\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Class & Support & Precision & Recall & F1 "
        + "\\\\"
        + "\n\\midrule\n"
        f"{per_class_rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
    )
    if score_impact_rows:
        summary_rows = "\n".join(
            f"{component} & {_tex_float(values['automated'], digits=4)} & "
            f"{_tex_float(values['human_reference'], digits=4)} & "
            f"{_tex_float(values['automated_minus_human'], digits=4, signed=True)} & "
            f"{_tex_float(values['model_level_pearson'])} & "
            f"{_tex_float(values['model_level_spearman'])} \\\\"
            for component, values in score_impact["components"].items()
        )
        atomic_write_text(
            output_dir / "score_impact_summary.tex",
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Post-hoc score-impact analysis on the stratified "
            "human-validation sample. Differences are automated minus "
            "human-reference scores; rank correlations are across models.}\n"
            "\\label{tab:classifier-score-impact}\n\\small\n"
            "\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Component & Automated & Human & Difference & Pearson $r$ & "
            "Spearman $\\rho$ "
            + "\\\\"
            + "\n\\midrule\n"
            f"{summary_rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        )
        bias_rows = "\n".join(
            f"{component} & "
            f"{_tex_float(values['bias_pearson_with_human_score'])} & "
            f"{_tex_float(values['bias_spearman_with_human_score'])} & "
            f"{_tex_float(values['automated_max_to_min_ratio'], digits=2)} & "
            f"{_tex_float(values['human_max_to_min_ratio'], digits=2)} \\\\"
            for component, values in score_impact["components"].items()
        )
        atomic_write_text(
            output_dir / "score_impact_bias.tex",
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Association between model score and classifier-induced "
            "bias in the stratified validation sample. Bias is automated minus "
            "human-reference score. Positive correlations therefore indicate "
            "that lower-scoring models receive a larger downward shift.}\n"
            "\\label{tab:classifier-score-bias}\n\\small\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Component & Pearson $r$ & Spearman $\\rho$ & Auto max/min & "
            "Human max/min "
            + "\\\\"
            + "\n\\midrule\n"
            f"{bias_rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        )
        model_table_rows = "\n".join(
            f"{_latex_escape(row['model'])} & {row['n']} & "
            f"{_tex_float(row['automated_EHQ1'])} & "
            f"{_tex_float(row['human_EHQ1'])} & "
            f"{_tex_float(row['delta_EHQ1'], signed=True)} & "
            f"{_tex_float(row['automated_EHQ2'])} & "
            f"{_tex_float(row['human_EHQ2'])} & "
            f"{_tex_float(row['delta_EHQ2'], signed=True)} \\\\"
            for row in score_impact_rows
        )
        atomic_write_text(
            output_dir / "score_impact_by_model.tex",
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Model-level score impact in the stratified validation "
            "sample. Differences are automated minus human-reference scores.}\n"
            "\\label{tab:classifier-score-impact-models}\n\\scriptsize\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Model & $n$ & Auto EHQ1 & Human EHQ1 & $\\Delta$ EHQ1 & "
            "Auto EHQ2 & Human EHQ2 & $\\Delta$ EHQ2 "
            + "\\\\"
            + "\n\\midrule\n"
            f"{model_table_rows}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n",
        )
    catalog = write_artifact_catalog(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": result["status"],
                "n_items": len(ids),
                "n_disagreements": len(disagreements),
                "artifact_count": len(catalog["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
