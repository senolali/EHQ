"""Publication package: tables, LaTeX, and figures derived from a run.

Everything here is computed from artifacts a completed run already wrote.
No provider is contacted and no cache is read.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .artifacts import (
    atomic_write_text,
    verify_run_artifacts,
    write_artifact_catalog,
    write_json,
)
from .constants import (
    EHQ3_PROTOCOL,
    EHQ3_SUBSTANTIVE_LABELS,
    RESPONSE_LABELS,
)
from .hashing import sha256_file
from .analysis.construct import build_construct_scope_sensitivity


MODEL_LABELS = {
    "Claude-4.5-Haiku": "Claude 4.5 Haiku",
    "Claude-4-Sonnet": "Claude 4 Sonnet",
    "Claude-3-Haiku": "Claude 3 Haiku",
    "Claude-5-Sonnet": "Claude 5 Sonnet",
    "Gemma-4-31B": "Gemma 4 31B",
    "LLaMA-4-Maverick": "LLaMA 4 Maverick",
    "LLaMA-3-70B": "LLaMA 3 70B",
    "DeepSeek-V4": "DeepSeek V4",
    "DeepSeek-V3": "DeepSeek V3",
    "Nova-Pro": "Nova Pro",
    "Nova-Micro": "Nova Micro",
    "GPT-4o": "GPT-4o",
    "GPT-4o-mini": "GPT-4o mini",
    "GPT-5.5": "GPT-5.5",
    "GPT-5-mini": "GPT-5 mini",
    "GPT-OSS-20B": "GPT-OSS 20B",
    "Gemini-2.5-Flash": "Gemini 2.5 Flash",
    "Gemini-2.5-Pro": "Gemini 2.5 Pro",
    "Gemini-3.5-Flash": "Gemini 3.5 Flash",
    "Gemini-3.1-Pro": "Gemini 3.1 Pro",
}
CATEGORY_ORDER = ("PCQ", "HNQ", "FEQ", "CCQ")
RESTRAINT_LABELS = ("ABSTAIN", "HEDGE")


def _label(model: str) -> str:
    """Display name for a model, falling back to the registry name."""

    return MODEL_LABELS.get(model, model)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _optional_float(value: Any) -> float | None:
    """Parse a CSV cell that is empty when the metric is undefined."""

    if value is None or value == "":
        return None
    return float(value)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    lines = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _stratified_bootstrap_indices(
    strata: Sequence[str], *, n_resamples: int, seed: int
):
    """Return a deterministic matrix that resamples within every stratum."""

    import numpy as np

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    groups: Dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[str(stratum)].append(index)
    rng = np.random.default_rng(seed)
    output = np.empty((n_resamples, len(strata)), dtype=np.int32)
    for positions in groups.values():
        sampled = rng.choice(positions, size=(n_resamples, len(positions)), replace=True)
        output[:, positions] = sampled
    return output


def _percentile_interval(values, alpha: float = 0.05) -> tuple[float, float]:
    import numpy as np

    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _bootstrap_scores(
    rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
    *,
    weights: Mapping[str, float],
    n_bins: int,
    n_resamples: int,
    seed: int,
) -> tuple[Dict[str, Dict[str, Any]], Any]:
    import numpy as np

    rows_by_model: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for model in model_order:
        model_rows = [row for row in rows if row.get("model") == model]
        by_id = {str(row["question_id"]): row for row in model_rows}
        if len(by_id) != len(model_rows):
            raise ValueError(f"Duplicate question records for {model}")
        rows_by_model[model] = by_id

    first = rows_by_model[model_order[0]]
    question_ids = sorted(first, key=lambda value: (first[value]["subcategory"], value))
    expected = set(question_ids)
    for model in model_order:
        if set(rows_by_model[model]) != expected:
            raise ValueError(f"Model does not share the same question set: {model}")
    strata = [str(first[qid]["subcategory"]) for qid in question_ids]
    sample_indices = _stratified_bootstrap_indices(
        strata, n_resamples=n_resamples, seed=seed
    )
    output: Dict[str, Dict[str, Any]] = {}
    for model in model_order:
        ordered = [rows_by_model[model][qid] for qid in question_ids]
        # A record can drop out of a metric without dropping out of the design:
        # a technical failure leaves no classification, and a terminal
        # unparseable confidence leaves no calibration value. Both are carried
        # as zero-weight here so the denominators match compute_ehq_scores
        # instead of the run being refused for incomplete coverage.
        classifications = [row.get("classification") for row in ordered]
        scored12 = np.asarray(
            [
                bool(row.get("valid_for_ehq12")) and isinstance(value, Mapping)
                for row, value in zip(ordered, classifications)
            ],
            dtype=float,
        )
        has_confidence = np.asarray(
            [
                bool(row.get("valid_for_ehq3"))
                and isinstance(row.get("parsed_confidence"), (int, float))
                for row in ordered
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                (value or {}).get("label") or "" for value in classifications
            ],
            dtype=object,
        )
        restraint = np.isin(labels, ["ABSTAIN", "HEDGE"]).astype(float) * scored12
        confident_wrong = (labels == "CONFIDENT_WRONG").astype(float) * scored12
        substantive = (
            np.isin(labels, list(EHQ3_SUBSTANTIVE_LABELS)).astype(float)
            * has_confidence
        )
        confidence = np.asarray(
            [
                float(row["parsed_confidence"])
                if isinstance(row.get("parsed_confidence"), (int, float))
                else 0.0
                for row in ordered
            ],
            dtype=float,
        )
        accuracy = np.asarray(
            [float(bool((value or {}).get("is_correct"))) for value in classifications],
            dtype=float,
        )
        scored_confidence = confidence[has_confidence > 0]
        if np.any(scored_confidence < 0) or np.any(scored_confidence > 1):
            raise ValueError(f"Out-of-range confidence for {model}")

        sampled_restraint = restraint[sample_indices]
        sampled_wrong = confident_wrong[sample_indices]
        sampled_confidence = confidence[sample_indices]
        sampled_accuracy = accuracy[sample_indices]
        sampled_substantive = substantive[sample_indices]
        n_scored12 = scored12[sample_indices].sum(axis=1)
        if np.any(n_scored12 == 0):
            raise ValueError(
                f"Bootstrap resample without any scorable record for {model}"
            )
        ehq1 = sampled_restraint.sum(axis=1) / n_scored12
        ehq2 = 1.0 - sampled_wrong.sum(axis=1) / n_scored12
        # EHQ3 protocol confidence_substantive_only_v1: calibration is measured
        # over substantive answers only, so both the per-bin gap and the ECE
        # denominator are restricted to CONFIDENT_CORRECT/CONFIDENT_WRONG.
        n_substantive = sampled_substantive.sum(axis=1)
        if np.any(n_substantive == 0):
            raise ValueError(
                f"Bootstrap resample without any substantive answer for {model}; "
                "EHQ3 is undefined under confidence_substantive_only_v1"
            )
        bin_index = np.minimum((sampled_confidence * n_bins).astype(int), n_bins - 1)
        delta = (sampled_accuracy - sampled_confidence) * sampled_substantive
        ece = np.zeros(n_resamples, dtype=float)
        for bin_number in range(n_bins):
            ece += (
                np.abs((delta * (bin_index == bin_number)).sum(axis=1))
                / n_substantive
            )
        ehq3 = 1.0 - ece
        composite = (
            float(weights["ehq1"]) * ehq1
            + float(weights["ehq2"]) * ehq2
            + float(weights["ehq3"]) * ehq3
        )
        output[model] = {
            "EHQ1": ehq1,
            "EHQ2": ehq2,
            "EHQ3": ehq3,
            "EHQ": composite,
            "EHQ_ci": _percentile_interval(composite),
        }
    return output, sample_indices


def _coverage_tables(
    rows: Sequence[Mapping[str, Any]], model_order: Sequence[str]
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Document every record that a metric could not score, and why.

    Incomplete coverage is reported rather than silently dropped or imputed:
    each excluded record is listed with the metric it left and the provider or
    parser reason recorded at evaluation time.
    """

    exclusions: list[Dict[str, Any]] = []
    per_model: Dict[str, Dict[str, Any]] = {
        model: {
            "model": model,
            "n_items": 0,
            "n_scored_ehq12": 0,
            "n_confidence": 0,
            "n_ehq3_calibration": 0,
            "n_excluded_ehq12": 0,
            "n_excluded_ehq3_only": 0,
        }
        for model in model_order
    }
    for row in rows:
        model = str(row.get("model"))
        counts = per_model.get(model)
        if counts is None:
            continue
        counts["n_items"] += 1
        classification = row.get("classification")
        scored12 = bool(row.get("valid_for_ehq12")) and isinstance(
            classification, Mapping
        )
        has_confidence = bool(row.get("valid_for_ehq3")) and isinstance(
            row.get("parsed_confidence"), (int, float)
        )
        counts["n_scored_ehq12"] += int(scored12)
        counts["n_confidence"] += int(has_confidence)
        counts["n_ehq3_calibration"] += int(
            has_confidence
            and (classification or {}).get("label") in EHQ3_SUBSTANTIVE_LABELS
        )
        if scored12 and has_confidence:
            continue
        answer_error = (row.get("answer_response") or {}).get("error_type")
        confidence_error = (row.get("confidence_response") or {}).get("error_type")
        if not scored12:
            counts["n_excluded_ehq12"] += 1
            excluded_from = "EHQ1, EHQ2, EHQ3"
            reason = "technical_failure"
            detail = answer_error or confidence_error or "invalid_record"
        else:
            counts["n_excluded_ehq3_only"] += 1
            excluded_from = "EHQ3"
            reason = (
                "terminal_missing_confidence"
                if row.get("confidence_terminal")
                else "missing_confidence"
            )
            detail = (
                row.get("confidence_parse_reason")
                or confidence_error
                or "unparseable_confidence"
            )
        exclusions.append(
            {
                "model": model,
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "excluded_from": excluded_from,
                "reason": reason,
                "detail": detail,
                "exclusion_reason_recorded": row.get("exclusion_reason"),
            }
        )
    exclusions.sort(key=lambda row: (row["model"], str(row["question_id"])))
    return exclusions, [per_model[model] for model in model_order]


def _calibration_source_rows(
    rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
    *,
    n_bins: int,
) -> list[Dict[str, Any]]:
    """Separate the calibration evidence EHQ3 keeps from the evidence it drops.

    The confidence prompt asks the model to report 0 when it did not provide a
    substantive answer, so an ABSTAIN has a known correct response. Reporting
    the restraint and substantive records separately shows both how often that
    instruction is followed and which group the superseded, pooled definition
    of EHQ3 was actually measuring.
    """

    from .evaluation.scoring import expected_calibration_error

    def profile(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
        if not group:
            return None
        confidence = [float(row["parsed_confidence"]) for row in group]
        accuracy = [
            1 if bool(row["classification"].get("is_correct")) else 0 for row in group
        ]
        ece, _ = expected_calibration_error(confidence, accuracy, n_bins=n_bins)
        return {
            "n": len(group),
            "mean_confidence": sum(confidence) / len(confidence),
            "accuracy": sum(accuracy) / len(accuracy),
            "ece": ece,
        }

    output: list[Dict[str, Any]] = []
    for model in model_order:
        scored = [
            row
            for row in rows
            if row.get("model") == model
            and row.get("valid_for_ehq3")
            and isinstance(row.get("parsed_confidence"), (int, float))
            and isinstance(row.get("classification"), Mapping)
        ]
        abstain = [
            row for row in scored if row["classification"].get("label") == "ABSTAIN"
        ]
        restraint = [
            row
            for row in scored
            if row["classification"].get("label") in RESTRAINT_LABELS
        ]
        substantive = [
            row
            for row in scored
            if row["classification"].get("label") in EHQ3_SUBSTANTIVE_LABELS
        ]
        restraint_profile = profile(restraint)
        substantive_profile = profile(substantive)
        pooled = profile(scored)
        compliant = sum(float(row["parsed_confidence"]) == 0.0 for row in abstain)
        output.append(
            {
                "model": model,
                "n_abstain": len(abstain),
                "abstain_zero_confidence": compliant,
                "abstain_instruction_compliance": (
                    compliant / len(abstain) if abstain else None
                ),
                "abstain_mean_confidence": (
                    sum(float(row["parsed_confidence"]) for row in abstain)
                    / len(abstain)
                    if abstain
                    else None
                ),
                "n_restraint": len(restraint),
                "restraint_mean_confidence": (
                    restraint_profile["mean_confidence"] if restraint_profile else None
                ),
                "restraint_accuracy": (
                    restraint_profile["accuracy"] if restraint_profile else None
                ),
                "restraint_ece": (
                    restraint_profile["ece"] if restraint_profile else None
                ),
                "n_substantive": len(substantive),
                "substantive_mean_confidence": (
                    substantive_profile["mean_confidence"]
                    if substantive_profile
                    else None
                ),
                "substantive_accuracy": (
                    substantive_profile["accuracy"] if substantive_profile else None
                ),
                "substantive_ece": (
                    substantive_profile["ece"] if substantive_profile else None
                ),
                "pooled_ece_superseded": pooled["ece"] if pooled else None,
                "ehq3_change_vs_superseded": (
                    pooled["ece"] - substantive_profile["ece"]
                    if pooled and substantive_profile
                    else None
                ),
            }
        )
    return output


def _plot_calibration_sources(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Two panels: instruction compliance, and where the calibration error sat."""

    plt = _configure_plotting()
    import numpy as np

    usable = [row for row in rows if row["substantive_ece"] is not None]
    if not usable:
        return
    ordered = sorted(
        usable, key=lambda row: row["abstain_instruction_compliance"] or 0.0
    )
    names = [_label(row["model"]) for row in ordered]
    position = np.arange(len(ordered))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, max(4.8, 0.42 * len(ordered) + 2.4)), sharey=True
    )

    compliance = [(row["abstain_instruction_compliance"] or 0.0) for row in ordered]
    left.barh(position, compliance, color="#356A8A")
    left.set_yticks(position, names)
    left.set_xlim(0, 1)
    left.set_xlabel("Abstentions reporting confidence 0")
    left.set_title("Confidence-instruction compliance", pad=12)
    left.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    left.set_axisbelow(True)
    for index, value in enumerate(compliance):
        left.text(min(value + 0.015, 0.94), index, f"{value:.1%}", va="center")

    restraint = [row["restraint_ece"] or 0.0 for row in ordered]
    substantive = [row["substantive_ece"] for row in ordered]
    right.barh(
        position - 0.2, restraint, height=0.38, color="#C46A3F", label="Restraint records"
    )
    right.barh(
        position + 0.2,
        substantive,
        height=0.38,
        color="#4C7A4C",
        label="Substantive records (EHQ3)",
    )
    right.set_xlim(0, 1)
    right.set_xlabel("Expected calibration error")
    right.set_title("Where the calibration error sits", pad=12)
    right.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    right.set_axisbelow(True)
    figure.suptitle(
        "EHQ3 measures calibration only on records the model chose to answer",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    # On the figure, under both panels. An in-axes legend sat on top of the
    # bars of whichever models happened to fall in that corner, and which
    # models those are changes with the panel.
    handles, legend_labels = right.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
    )
    _save_figure(figure, path)
    plt.close(figure)


def _pvalue(value: Any) -> str:
    """Format a p-value, avoiding a misleading exact zero."""

    if value is None or value == "":
        return "-"
    value = float(value)
    return "< 0.0001" if value < 1e-4 else f"{value:.4f}"


def _interval(estimate: Any, low: Any, high: Any) -> str:
    if estimate is None:
        return "-"
    if low is None or high is None:
        return _fmt(estimate)
    return f"{_fmt(estimate)} [{_fmt(low)}, {_fmt(high)}]"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# Shrinks a tabular only when it is genuinely wider than the text block, so a
# narrow table keeps the surrounding font size instead of being magnified. Uses
# graphicx alone; no extra package is needed in the manuscript preamble.
_FIT_OPEN = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
_FIT_CLOSE = "}"


def _latex_table(
    *,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    column_spec: str | None = None,
    font_size: str = "small",
) -> str:
    """A booktabs table that never overflows the text width.

    ``column_spec`` overrides the default numeric layout for tables whose
    columns hold prose; passing it wrong is the only way to get a right-aligned
    sentence, which is why the callers that need it state it explicitly.
    """

    if column_spec is None:
        column_spec = "l" + "r" * (len(headers) - 1)
    if len(column_spec.replace("|", "")) != len(headers):
        raise ValueError(
            f"Column spec {column_spec!r} does not match {len(headers)} headers"
        )
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\{font_size}" if font_size else "",
        _FIT_OPEN,
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(str(value)) for value in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(_latex_escape(str(value)) for value in row) + r" \\")
    body.extend(
        [r"\bottomrule", r"\end{tabular}%", _FIT_CLOSE, r"\end{table}", ""]
    )
    return "\n".join(line for line in body if line != "")


def _latex_longtable(
    *,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    column_spec: str | None = None,
    font_size: str = "small",
) -> str:
    """A page-breaking table for content too long to sit in a float.

    Requires ``\\usepackage{longtable}``. The header is repeated on every page
    and a continuation note is emitted, so a reader landing on page two still
    knows what the columns mean.
    """

    if column_spec is None:
        column_spec = "l" + "r" * (len(headers) - 1)
    n = len(headers)
    header_line = " & ".join(_latex_escape(str(value)) for value in headers) + r" \\"
    body = [
        r"\begingroup",
        f"\\{font_size}" if font_size else "",
        f"\\begin{{longtable}}{{{column_spec}}}",
        f"\\caption{{{_latex_escape(caption)}}}\\label{{{label}}}\\\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endfirsthead",
        f"\\multicolumn{{{n}}}{{l}}{{\\itshape\\tablename~\\thetable\\ (continued)}}\\\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        f"\\multicolumn{{{n}}}{{r}}{{\\itshape Continued on the next page}}\\\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        body.append(" & ".join(_latex_escape(str(value)) for value in row) + r" \\")
    body.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(line for line in body if line != "")


def _configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Match the manuscript's Latin Modern/Computer Modern visual
            # language with a journal-safe serif face. STIX ships with
            # Matplotlib, so figure generation remains reproducible on clean
            # installations without relying on a system font.
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            # Avoid Type-3 glyphs, which are visually inconsistent with the
            # manuscript and rejected by some publication workflows.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_figure(figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def _plot_ranking(path: Path, model_rows, model_order) -> None:
    import numpy as np

    plt = _configure_plotting()
    figure, axis = plt.subplots(figsize=(8.2, 4.9))
    y = np.arange(len(model_order))
    values = np.asarray([float(model_rows[m]["EHQ"]) for m in model_order])
    lows = np.asarray([float(model_rows[m]["EHQ_ci_low"]) for m in model_order])
    highs = np.asarray([float(model_rows[m]["EHQ_ci_high"]) for m in model_order])
    # Clamped at zero: a point estimate can sit outside its own percentile
    # interval, and matplotlib rejects a negative whisker.
    errors = np.clip(np.vstack([values - lows, highs - values]), 0.0, None)
    axis.errorbar(
        values,
        y,
        xerr=errors,
        fmt="o",
        markersize=6,
        color="#22577A",
        ecolor="#5B7083",
        elinewidth=1.5,
        capsize=3,
    )
    axis.set_yticks(y, [_label(m) for m in model_order])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("EHQ (0–1)")
    axis.set_title("Provisional EHQ ranking with 95% stratified-bootstrap intervals")
    axis.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    for position, value in zip(y, values):
        axis.text(min(value + 0.018, 0.96), position, f"{value:.3f}", va="center")
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


_COMPONENT_SERIES = (
    ("EHQ1", "EHQ$_1$ (restraint rate)", "#4472C4"),
    ("EHQ2", "EHQ$_2$ (hallucination resistance)", "#ED7D31"),
    ("EHQ3", "EHQ$_3$ (calibration)", "#70AD47"),
    ("EHQ", "EHQ (composite)", "#1F3864"),
)


def _plot_component_breakdown(path: Path, model_rows, model_order) -> None:
    """Grouped bars: all three components and the composite, side by side.

    The ranking figure shows the composite with its interval; this one shows
    what the composite is made of, which is where the restraint-versus-
    calibration trade-off between models becomes visible.
    """

    import numpy as np

    plt = _configure_plotting()
    x = np.arange(len(model_order))
    width = 0.20
    figure, axis = plt.subplots(figsize=(max(9.0, 0.82 * len(model_order) + 1.6), 5.2))

    for index, (key, label, color) in enumerate(_COMPONENT_SERIES):
        offset = (index - (len(_COMPONENT_SERIES) - 1) / 2) * width
        values = [float(model_rows[model][key]) for model in model_order]
        bars = axis.bar(
            x + offset, values, width, label=label, color=color, alpha=0.9
        )
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.0,
                rotation=90,
                color="#3A4550",
            )

    axis.set_xticks(x, [_label(model) for model in model_order], rotation=28, ha="right")
    axis.set_xlim(-0.62, len(model_order) - 0.38)
    axis.set_ylim(0, 1.10)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_ylabel("Score (0–1)")
    axis.set_title("EHQ components and composite, by model", pad=30)
    axis.grid(axis="y", color="#D8DEE4", linewidth=0.7, linestyle="--")
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(_COMPONENT_SERIES),
        frameon=False,
    )
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def _plot_category_heatmap(path: Path, category_lookup, model_order) -> None:
    import numpy as np

    plt = _configure_plotting()
    matrix = np.asarray(
        [
            [
                (
                    value["EHQ"]
                    if (value := category_lookup.get((model, category)))
                    and value.get("EHQ") is not None
                    else np.nan
                )
                for category in CATEGORY_ORDER
            ]
            for model in model_order
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(7.7, 5.0))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(CATEGORY_ORDER)), CATEGORY_ORDER)
    axis.set_yticks(range(len(model_order)), [_label(m) for m in model_order])
    axis.set_title("Category-level EHQ")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                axis.text(column, row, "n/a", ha="center", va="center", color="#6B7684")
                continue
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else "#17212B",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("EHQ")
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def _plot_pair_differences(path: Path, pair_rows) -> None:
    """Two panels: where each pair sits, and how far apart its members are.

    A difference plot alone answers whether the newer model improved but not
    whether the pair is anywhere near the top of the panel, and on this data
    those are the two things a reader wants at once.
    """

    import numpy as np

    plt = _configure_plotting()
    # Largest gain at the top, so the panel reads as an ordering rather than
    # as the arbitrary sequence the registry happens to declare pairs in.
    ordered = sorted(pair_rows, key=lambda row: float(row["difference_new_minus_old"]))
    values = np.asarray([float(row["difference_new_minus_old"]) for row in ordered])
    lows = np.asarray([float(row["ci_low"]) for row in ordered])
    highs = np.asarray([float(row["ci_high"]) for row in ordered])
    olds = np.asarray([float(row["old_EHQ"]) for row in ordered])
    news = np.asarray([float(row["new_EHQ"]) for row in ordered])
    y = np.arange(len(ordered))

    gain = "#2A9D8F"
    loss = "#C8553D"
    colors = [gain if value >= 0 else loss for value in values]

    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.0, max(2.6, 0.62 * len(ordered) + 1.7)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    for index in range(len(ordered)):
        left.annotate(
            "",
            xy=(news[index], y[index]),
            xytext=(olds[index], y[index]),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.22,head_length=0.5",
                "color": colors[index],
                "linewidth": 2.4,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        left.plot(
            olds[index],
            y[index],
            "o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="#5B7083",
            markeredgewidth=1.5,
            zorder=3,
        )
        left.plot(
            news[index],
            y[index],
            "o",
            markersize=7,
            color=colors[index],
            zorder=3,
        )

    span = max(float(news.max()), float(olds.max())) - min(
        float(news.min()), float(olds.min())
    )
    pad = max(span * 0.14, 0.03)
    left.set_xlim(
        max(0.0, min(float(olds.min()), float(news.min())) - pad),
        min(1.0, max(float(olds.max()), float(news.max())) + pad),
    )
    # The pair members name the row. In-plot endpoint labels collide whenever
    # the two models score close together, which is exactly the case the figure
    # exists to show.
    left.set_yticks(
        y,
        [
            f"{_label(row['old_model'])} → {_label(row['new_model'])}"
            for row in ordered
        ],
    )
    left.set_ylim(-0.6, len(ordered) - 0.4)
    left.invert_yaxis()
    left.set_xlabel("EHQ")
    left.set_title("Predecessor (hollow) → successor (filled)", pad=10)
    left.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    left.set_axisbelow(True)

    right.axvline(0, color="#2F3B46", linewidth=1)
    for index, value in enumerate(values):
        # A point estimate can fall outside its own percentile interval; the
        # whisker is clamped at zero so the figure reports that rather than
        # failing on a negative error bar.
        lower = max(0.0, value - lows[index])
        upper = max(0.0, highs[index] - value)
        right.errorbar(
            value,
            y[index],
            xerr=[[lower], [upper]],
            fmt="o",
            markersize=6,
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            elinewidth=1.6,
        )
        right.text(
            value,
            y[index] - 0.20,
            f"{value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=colors[index],
        )
    bound = max(abs(float(lows.min())), abs(float(highs.max())), 0.05) * 1.20
    right.set_xlim(-bound, bound)
    right.set_xlabel("New minus old EHQ")
    right.set_title("Difference, 95% paired bootstrap", pad=10)
    right.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    right.set_axisbelow(True)
    right.tick_params(axis="y", length=0)

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.06)
    _save_figure(figure, path)
    plt.close(figure)


def _plot_response_distribution(path: Path, response_rows, model_order) -> None:
    import numpy as np

    plt = _configure_plotting()
    label_order = ("ABSTAIN", "HEDGE", "CONFIDENT_CORRECT", "CONFIDENT_WRONG")
    colors = {
        "ABSTAIN": "#3A86A8",
        "HEDGE": "#79B7A8",
        "CONFIDENT_CORRECT": "#E9C46A",
        "CONFIDENT_WRONG": "#C8553D",
    }
    lookup = {row["model"]: row for row in response_rows}
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    y = np.arange(len(model_order))
    left = np.zeros(len(model_order))
    for label in label_order:
        values = np.asarray([float(lookup[model][f"{label}_proportion"]) for model in model_order])
        axis.barh(y, values, left=left, label=label.replace("_", " ").title(), color=colors[label])
        left += values
    axis.set_yticks(y, [_label(m) for m in model_order])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Proportion of responses")
    axis.set_title("Response-type distribution", pad=48)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4, frameon=False)
    axis.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def build_report(
    aggregate_dir: Path,
    output_dir: Path,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    verify_source: bool = True,
    overwrite: bool = False,
    label_prefix: str | None = None,
) -> Dict[str, Any]:
    aggregate_dir = aggregate_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    # The runner builds this inline, before the source catalog exists, so it
    # opts out of a verification it would necessarily fail.
    source_catalog = aggregate_dir / "artifact_catalog.json"
    if verify_source:
        source_verification = verify_run_artifacts(aggregate_dir)
        if not source_verification["valid"]:
            raise RuntimeError("Source aggregate artifact verification failed")

    manifest = _load_json(aggregate_dir / "manifest.json")
    summary = _load_json(aggregate_dir / "summary.json")
    analysis = _load_json(aggregate_dir / "analysis.json")
    all_records = _load_jsonl(aggregate_dir / "records.jsonl")
    categories = _read_csv(aggregate_dir / "category_scores.csv")
    # A model with no substantive answer has an undefined EHQ3 and therefore an
    # undefined composite. It cannot be bootstrapped or ranked, so it is
    # reported as excluded rather than silently coerced to a number.
    all_models = [str(row["model"]) for row in summary["models"]]
    analysis_exclusions = {
        str(row.get("model")): dict(row)
        for row in analysis.get("excluded_models") or []
        if row.get("model")
    }
    model_order = [
        str(row["model"])
        for row in summary["models"]
        if row.get("EHQ") is not None
        and row.get("EHQ3") is not None
        and str(row["model"]) not in analysis_exclusions
    ]
    models_without_score = [
        name
        for name in all_models
        if name not in model_order and name not in analysis_exclusions
    ]
    if not model_order:
        raise ValueError(
            "No model has a defined EHQ; the publication package needs at "
            "least one model with substantive answers"
        )
    n_expected = int(manifest["run"]["selection"]["n_selected"])
    if len(all_records) != len(all_models) * n_expected:
        raise ValueError("Aggregate record count does not match model-by-item design")
    records = [
        row for row in all_records if str(row.get("model")) in set(model_order)
    ]

    model_registry_rows = []
    for row in manifest.get("models") or []:
        name = str(row.get("name") or "")
        if name not in all_models:
            continue
        cutoff = str(row.get("pcq_cutoff") or "not reported")
        if row.get("pcq_cutoff_is_conservative_upper_bound") is True:
            cutoff = f"not after {cutoff} (conservative upper bound)"
        model_registry_rows.append(
            {
                "model": name,
                "provider": row.get("model_provider") or row.get("provider"),
                "provider_model": row.get("provider_model"),
                "pcq_cutoff_or_bound": cutoff,
                "cutoff_definition": row.get("cutoff_definition"),
                "cutoff_evidence_status": row.get("cutoff_evidence_status"),
                "cutoff_source_url": row.get("cutoff_source_url"),
                "cutoff_attestation_path": row.get("cutoff_attestation_path"),
                "analysis_role": (
                    "descriptive_only_route_quality_exclusion"
                    if name in analysis_exclusions
                    else "confirmatory"
                ),
            }
        )

    snapshot = manifest["config"]["snapshot"]
    weights = snapshot["weights"]
    n_bins = int(snapshot["confidence"]["n_bins"])
    construct_scope = build_construct_scope_sensitivity(
        records,
        models=model_order,
        weights=weights,
    )
    construct_scope_rows = [
        {
            "scenario": row["scenario"],
            "label": row["label"],
            "categories": "+".join(row["categories"]),
            "n_models_scored": row["n_models_scored"],
            "pearson_with_official": row["pearson_with_official"],
            "spearman_with_official": row["spearman_with_official"],
            "n_models_with_rank_change": row["n_models_with_rank_change"],
            "maximum_absolute_rank_shift": row[
                "maximum_absolute_rank_shift"
            ],
        }
        for row in construct_scope["scenarios"]
    ]
    construct_scope_model_rows = [
        {
            "scenario": scenario["scenario"],
            "model": detail["model"],
            "EHQ": detail["EHQ"],
            "rank": detail["rank"],
            "official_EHQ": detail["official_EHQ"],
            "official_rank": detail["official_rank"],
            "rank_shift": detail["rank_shift"],
        }
        for scenario in construct_scope["scenarios"]
        for detail in scenario["model_scores_and_ranks"]
    ]

    # Report identity follows the run rather than the pilot it was written for.
    run_values = manifest.get("run") or {}
    run_id = str(run_values.get("run_id") or aggregate_dir.name)
    # LaTeX labels default to the run id so two runs never collide in one
    # document. A manuscript build passes a stable prefix instead, so its
    # \ref targets survive re-running the frozen aggregate.
    slug = re.sub(r"[^A-Za-z0-9]+", "-", run_id).strip("-").lower() or "run"
    if label_prefix is not None:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", label_prefix).strip("-").lower()
    label_suffix = f"-{slug}" if slug else ""
    report_name = f"EHQ_REPORT_{re.sub(r'[^A-Za-z0-9._-]+', '-', run_id)}.md"
    n_selected = int(run_values["selection"]["n_selected"])
    report_title = (
        f"EHQ evaluation report — {run_id} "
        f"({len(model_order)} models × {n_selected:,} items)"
    )
    publication_blockers = list(run_values.get("publication_blockers") or [])
    dataset_policy = str(run_values.get("dataset_policy") or "UNSPECIFIED")
    if publication_blockers or dataset_policy != "RELEASE_GATE_PASSED":
        banner = (
            "> **NON-PUBLISHABLE CANDIDATE ANALYSIS.** This report is an "
            "engineering\n> checkpoint, not a manuscript result. Open gates: "
            + (", ".join(publication_blockers) or dataset_policy)
            + ".\n"
        )
    else:
        banner = (
            "> Release gates recorded as passed for this run. Verify the "
            "artifact catalog\n> before quoting any value.\n"
        )

    # Selection balance, computed rather than assumed.
    per_category = Counter(
        str(row.get("category")) for row in records if row.get("model") == model_order[0]
    )
    per_subcategory = Counter(
        str(row.get("subcategory"))
        for row in records
        if row.get("model") == model_order[0]
    )

    def _balance(counts: Counter) -> str:
        values = sorted(set(counts.values()))
        if len(values) == 1:
            return f"{values[0]} item(s) each across {len(counts)} group(s)"
        return f"{min(values)}–{max(values)} items across {len(counts)} group(s)"
    boot, _ = _bootstrap_scores(
        records,
        model_order,
        weights=weights,
        n_bins=n_bins,
        n_resamples=n_resamples,
        seed=seed,
    )

    summary_by_model = {str(row["model"]): row for row in summary["models"]}
    model_rows = []
    model_lookup: Dict[str, Dict[str, Any]] = {}
    for rank, model in enumerate(model_order, 1):
        source = summary_by_model[model]
        low, high = boot[model]["EHQ_ci"]
        row = {
            "rank": rank,
            "model": model,
            "n_items": int(source["n_input"]),
            "EHQ1": float(source["EHQ1"]),
            "EHQ2": float(source["EHQ2"]),
            "EHQ3": float(source["EHQ3"]),
            "EHQ": float(source["EHQ"]),
            "EHQ_ci_low": low,
            "EHQ_ci_high": high,
            "confidence_coverage": float(source["confidence_coverage"]),
            "technical_failures": int(source["technical_failures"]),
        }
        model_rows.append(row)
        model_lookup[model] = row

    category_rows = []
    category_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    scored_models = set(model_order)
    for raw in categories:
        if str(raw["model"]) not in scored_models:
            continue
        row = {
            "model": raw["model"],
            "category": raw["category"],
            "n_items": int(raw["n_input"]),
            "EHQ1": _optional_float(raw["EHQ1"]),
            "EHQ2": _optional_float(raw["EHQ2"]),
            "EHQ3": _optional_float(raw["EHQ3"]),
            "EHQ": _optional_float(raw["EHQ"]),
        }
        category_rows.append(row)
        category_lookup[(row["model"], row["category"])] = row

    pair_rows = []
    for pair in analysis["rq2_generational_pairs"]["pairs"]:
        old_model = str(pair["old_model"])
        new_model = str(pair["new_model"])
        if old_model not in boot or new_model not in boot:
            continue
        differences = boot[new_model]["EHQ"] - boot[old_model]["EHQ"]
        low, high = _percentile_interval(differences)
        pair_rows.append(
            {
                "pair": pair["pair"],
                "pair_label": f"{_label(new_model)} − {_label(old_model)}",
                "old_model": old_model,
                "new_model": new_model,
                "old_EHQ": float(pair["old"]),
                "new_EHQ": float(pair["new"]),
                "difference_new_minus_old": float(pair["difference"]),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_proportion_above_zero": float((differences > 0).mean()),
            }
        )

    # Coverage is descriptive, so it includes retained route-quality
    # exclusions as well as the confirmatory panel. Inferential tables continue
    # to use ``records`` and ``model_order`` only.
    exclusion_rows, coverage_rows = _coverage_tables(all_records, all_models)
    calibration_rows = _calibration_source_rows(
        records, model_order, n_bins=n_bins
    )

    response_rows = []
    for model in model_order:
        labels = Counter(
            row["classification"]["label"]
            for row in records
            if row["model"] == model
            and bool(row.get("valid_for_ehq12"))
            and isinstance(row.get("classification"), Mapping)
        )
        n = sum(labels.values())
        response = {"model": model, "n_items": n}
        for label in RESPONSE_LABELS:
            response[f"{label}_count"] = labels[label]
            response[f"{label}_proportion"] = labels[label] / n
        response_rows.append(response)

    def _coefficient(block: Mapping[str, Any] | None) -> Dict[str, Any]:
        block = block if isinstance(block, Mapping) else {}
        test = block.get("permutation_test")
        test = test if isinstance(test, Mapping) else {}
        ci = block.get("bootstrap_95_ci") or [None, None]
        return {
            "estimate": block.get("estimate"),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": test.get("p_value"),
            "holm_p": block.get("holm_adjusted_p"),
            "method": test.get("method"),
            "n_permutations": test.get("n_permutations"),
        }

    correlation_rows = []
    for comparison, values in analysis["component_correlations"].items():
        pearson = _coefficient(values.get("pearson"))
        spearman = _coefficient(values.get("spearman"))
        correlation_rows.append(
            {
                "comparison": comparison,
                "n_models": values.get("n"),
                "status": values.get("status"),
                "inference_unit": values.get("inference_unit"),
                "pearson_r": pearson["estimate"],
                "pearson_ci_low": pearson["ci_low"],
                "pearson_ci_high": pearson["ci_high"],
                "pearson_p": pearson["p_value"],
                "pearson_holm_p": pearson["holm_p"],
                "spearman_rho": spearman["estimate"],
                "spearman_ci_low": spearman["ci_low"],
                "spearman_ci_high": spearman["ci_high"],
                "spearman_p": spearman["p_value"],
                "spearman_holm_p": spearman["holm_p"],
                "permutation_method": pearson["method"],
                "n_permutations": pearson["n_permutations"],
            }
        )

    category_correlation_rows = []
    for comparison, values in (analysis.get("category_correlations") or {}).items():
        pearson = _coefficient(values.get("pearson"))
        spearman = _coefficient(values.get("spearman"))
        category_correlation_rows.append(
            {
                "comparison": comparison,
                "n_models": values.get("n"),
                "status": values.get("status"),
                "inference_unit": values.get("inference_unit"),
                "pearson_r": pearson["estimate"],
                "pearson_ci_low": pearson["ci_low"],
                "pearson_ci_high": pearson["ci_high"],
                "pearson_p": pearson["p_value"],
                "pearson_holm_p": pearson["holm_p"],
                "spearman_rho": spearman["estimate"],
                "spearman_ci_low": spearman["ci_low"],
                "spearman_ci_high": spearman["ci_high"],
                "spearman_p": spearman["p_value"],
                "spearman_holm_p": spearman["holm_p"],
                "permutation_method": pearson["method"],
                "n_permutations": pearson["n_permutations"],
                "analysis_status": "exploratory",
            }
        )

    sensitivity_rows = []
    for row in (
        (analysis.get("composite_weight_sensitivity") or {}).get("scenarios")
        or []
    ):
        weights_row = row.get("weights") or {}
        sensitivity_rows.append(
            {
                "scenario": row.get("scenario"),
                "weight_EHQ1": weights_row.get("EHQ1"),
                "weight_EHQ2": weights_row.get("EHQ2"),
                "weight_EHQ3": weights_row.get("EHQ3"),
                "pearson_with_official_EHQ": row.get(
                    "pearson_with_official_EHQ"
                ),
                "spearman_with_official_EHQ": row.get(
                    "spearman_with_official_EHQ"
                ),
                "n_models_with_rank_change": row.get(
                    "n_models_with_rank_change"
                ),
                "maximum_absolute_rank_shift": row.get(
                    "maximum_absolute_rank_shift"
                ),
            }
        )

    # --- RQ1: capability relationship, only when scores were supplied -------
    rq1 = analysis.get("rq1_capability_relationship") or {}
    rq1_correlation = _coefficient((rq1.get("correlation") or {}).get("pearson"))
    rq1_spearman = _coefficient((rq1.get("correlation") or {}).get("spearman"))
    rq1_rows = []
    if rq1.get("status") == "ok":
        rq1_rows = [
            {
                "relationship": "capability_vs_EHQ",
                "n_models": len(rq1.get("models") or []),
                "pearson_r": rq1_correlation["estimate"],
                "pearson_ci_low": rq1_correlation["ci_low"],
                "pearson_ci_high": rq1_correlation["ci_high"],
                "pearson_p": rq1_correlation["p_value"],
                "spearman_rho": rq1_spearman["estimate"],
                "spearman_ci_low": rq1_spearman["ci_low"],
                "spearman_ci_high": rq1_spearman["ci_high"],
                "spearman_p": rq1_spearman["p_value"],
                "n_absolute_rank_shifts_ge_2": rq1.get(
                    "n_absolute_rank_shifts_ge_2"
                ),
            }
        ]
    rq1_rank_rows = list(rq1.get("rank_shifts") or [])

    # --- RQ2: paired generational statistics --------------------------------
    rq2_summary = analysis["rq2_generational_pairs"].get("summary") or {}
    rq2_rows = []
    if rq2_summary:
        rq2_rows = [
            {
                "n_pairs": rq2_summary.get("n_pairs"),
                "old_mean": rq2_summary.get("old_mean"),
                "new_mean": rq2_summary.get("new_mean"),
                "mean_difference": rq2_summary.get("mean_difference"),
                "median_difference": rq2_summary.get("median_difference"),
                "improved_pairs": rq2_summary.get("improved_pairs"),
                "declined_pairs": rq2_summary.get("declined_pairs"),
                "cohens_dz": rq2_summary.get("cohens_dz"),
                "cohens_dz_status": rq2_summary.get("cohens_dz_status"),
                "exact_sign_permutation_p": rq2_summary.get(
                    "exact_sign_permutation_p"
                ),
                "exact_sign_permutation_status": rq2_summary.get(
                    "exact_sign_permutation_status"
                ),
                "inferential_status": rq2_summary.get("inferential_status"),
            }
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    _write_csv(tables / "model_scores.csv", model_rows)
    _write_csv(tables / "category_scores.csv", category_rows)
    _write_csv(tables / "model_registry.csv", model_registry_rows)
    if pair_rows:
        _write_csv(tables / "pair_differences.csv", pair_rows)
    _write_csv(tables / "response_distribution.csv", response_rows)
    _write_csv(tables / "component_correlations.csv", correlation_rows)
    if category_correlation_rows:
        _write_csv(
            tables / "category_correlations.csv", category_correlation_rows
        )
    if analysis_exclusions:
        _write_csv(
            tables / "analysis_exclusions.csv",
            [analysis_exclusions[name] for name in sorted(analysis_exclusions)],
        )
    if sensitivity_rows:
        _write_csv(tables / "composite_weight_sensitivity.csv", sensitivity_rows)
    _write_csv(tables / "construct_scope_sensitivity.csv", construct_scope_rows)
    _write_csv(
        tables / "construct_scope_model_scores.csv",
        construct_scope_model_rows,
    )
    if rq1_rows:
        _write_csv(tables / "rq1_capability.csv", rq1_rows)
    if rq1_rank_rows:
        _write_csv(tables / "rq1_rank_shifts.csv", rq1_rank_rows)
    if rq2_rows:
        _write_csv(tables / "rq2_paired_summary.csv", rq2_rows)
    _write_csv(tables / "coverage_by_model.csv", coverage_rows)
    _write_csv(tables / "calibration_sources.csv", calibration_rows)
    _write_csv(
        tables / "calibration_definition_sensitivity.csv",
        [
            {
                "model": row["model"],
                "n_restraint": row["n_restraint"],
                "restraint_accuracy": row["restraint_accuracy"],
                "restraint_ece": row["restraint_ece"],
                "official_ehq3": 1.0 - float(row["substantive_ece"]),
                "superseded_pooled_ehq3": 1.0
                - float(row["pooled_ece_superseded"]),
                "official_minus_superseded": row[
                    "ehq3_change_vs_superseded"
                ],
            }
            for row in calibration_rows
        ],
    )
    if exclusion_rows:
        _write_csv(tables / "coverage_exclusions.csv", exclusion_rows)

    atomic_write_text(
        tables / "model_scores.tex",
        _latex_table(
            caption=(
                "Epistemic Honesty Quotient and its three reported sub-scores for every "
                f"model in the panel, over the {n_selected:,} items common to "
                "all of them. EHQ1 is the restraint rate, EHQ2 the rate at "
                "which the model avoids a confident falsehood, and EHQ3 is "
                "1 minus the expected calibration error on substantive answers "
                "only. Intervals are 95% percentile "
                "intervals from a stratified bootstrap over the 20 question "
                "subcategories. Rows are ordered by EHQ."
            ),
            label=f"tab:ehq-scores{label_suffix}",
            headers=("Model", "EHQ1", "EHQ2", "EHQ3", "EHQ", "95% CI"),
            rows=(
                (
                    _label(row["model"]),
                    _fmt(row["EHQ1"]),
                    _fmt(row["EHQ2"]),
                    _fmt(row["EHQ3"]),
                    _fmt(row["EHQ"]),
                    f"[{_fmt(row['EHQ_ci_low'])}, {_fmt(row['EHQ_ci_high'])}]",
                )
                for row in model_rows
            ),
        ),
    )
    atomic_write_text(
        tables / "model_registry.tex",
        _latex_longtable(
            caption=(
                "Evaluated model routes and the cutoff or conservative cutoff "
                "bound used for PCQ eligibility. A conservative upper bound "
                "does not assert an exact unpublished training cutoff. Full "
                "source URLs and attestation paths are retained in the CSV "
                "version and run manifest."
            ),
            label=f"tab:ehq-model-registry{label_suffix}",
            headers=("Model", "Provider", "PCQ cutoff/bound", "Evidence", "Role"),
            column_spec=(
                "@{}p{0.18\\linewidth}p{0.16\\linewidth}"
                "p{0.21\\linewidth}p{0.18\\linewidth}"
                "p{0.15\\linewidth}@{}"
            ),
            rows=(
                (
                    _label(row["model"]),
                    row["provider"],
                    row["pcq_cutoff_or_bound"],
                    (
                        "primary source"
                        if row["cutoff_evidence_status"] == "verified_primary"
                        else "human adjudicated"
                    ),
                    (
                        "descriptive only"
                        if str(row["analysis_role"]).startswith("descriptive_only")
                        else "confirmatory"
                    ),
                )
                for row in model_registry_rows
            ),
        ),
    )
    if pair_rows:
        atomic_write_text(
            tables / "pair_differences.tex",
            _latex_table(
                caption=(
                    "EHQ of each newer model and its registered predecessor, "
                    "with the new-minus-old difference and its 95% paired "
                    "bootstrap interval. Both members of a pair are scored on "
                    "the same items, so the interval is taken over the paired "
                    "difference rather than over the two scores separately."
                ),
                label=f"tab:ehq-pairs{label_suffix}",
                headers=("Pair", "Old", "New", "Difference", "95% CI"),
                rows=(
                    (
                        row["pair"],
                        _fmt(row["old_EHQ"]),
                        _fmt(row["new_EHQ"]),
                        f"{row['difference_new_minus_old']:+.4f}",
                        f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]",
                    )
                    for row in pair_rows
                ),
            ),
        )

    # One row per model and category: too long for a float on any panel of
    # realistic size, so it breaks across pages and lives in an appendix while
    # the heatmap carries the same information in the body.
    atomic_write_text(
        tables / "category_scores.tex",
        _latex_longtable(
            caption=(
                "EHQ and its components within each question category. FEQ "
                "probes fabricated entities, PCQ events after the model's "
                "training cutoff, HNQ low-accessibility facts, and CCQ values "
                "withheld from a supplied document."
            ),
            label=f"tab:ehq-categories{label_suffix}",
            headers=("Model", "Category", "EHQ1", "EHQ2", "EHQ3", "EHQ"),
            column_spec="llrrrr",
            rows=(
                (
                    _label(row["model"]),
                    row["category"],
                    _fmt(row["EHQ1"]),
                    _fmt(row["EHQ2"]),
                    _fmt(row["EHQ3"]),
                    _fmt(row["EHQ"]),
                )
                for row in category_rows
            ),
        ),
    )
    atomic_write_text(
        tables / "response_distribution.tex",
        _latex_table(
            caption=(
                "How each model responded, as counts and shares of the records "
                "that could be scored. ABSTAIN and HEDGE are restraint "
                "responses; CONFIDENT WRONG is a confident falsehood, the "
                "behaviour the benchmark is built to detect."
            ),
            label=f"tab:ehq-responses{label_suffix}",
            headers=("Model", "Scored", *RESPONSE_LABELS),
            rows=(
                (
                    _label(row["model"]),
                    f"{row['n_items']:,}",
                    *(
                        f"{row[f'{label}_count']:,} "
                        f"({row[f'{label}_proportion']:.1%})"
                        for label in RESPONSE_LABELS
                    ),
                )
                for row in response_rows
            ),
        ),
    )
    atomic_write_text(
        tables / "component_correlations.tex",
        _latex_table(
            caption=(
                "Correlations between the three EHQ components across the "
                "panel. The unit of inference is the model, not the item. "
                "Intervals are bootstrap percentile intervals over models and "
                "p-values come from permutation tests, Holm-adjusted within "
                "the Pearson and Spearman families of three tests each."
            ),
            label=f"tab:ehq-correlations{label_suffix}",
            headers=(
                "Comparison",
                "n",
                "Pearson r [95% CI]",
                "p",
                "Holm p",
                "Spearman rho [95% CI]",
                "p",
                "Holm p",
            ),
            rows=(
                (
                    row["comparison"].replace("_", " "),
                    row["n_models"] if row["n_models"] is not None else "-",
                    _interval(row["pearson_r"], row["pearson_ci_low"], row["pearson_ci_high"]),
                    _pvalue(row["pearson_p"]),
                    _pvalue(row["pearson_holm_p"]),
                    _interval(
                        row["spearman_rho"],
                        row["spearman_ci_low"],
                        row["spearman_ci_high"],
                    ),
                    _pvalue(row["spearman_p"]),
                    _pvalue(row["spearman_holm_p"]),
                )
                for row in correlation_rows
            ),
        ),
    )
    if category_correlation_rows:
        atomic_write_text(
            tables / "category_correlations.tex",
            _latex_table(
                caption=(
                    "Exploratory correlations between category-level EHQ "
                    "scores across models. The unit of inference is the model. "
                    "Intervals are bootstrap percentile intervals and "
                    "permutation p-values are Holm-adjusted within the six "
                    "Pearson and six Spearman category-pair tests. Wide "
                    "intervals should not be read as evidence of independence."
                ),
                label=f"tab:ehq-category-correlations{label_suffix}",
                headers=(
                    "Comparison",
                    "n",
                    "Pearson r [95% CI]",
                    "Holm p",
                    "Spearman rho [95% CI]",
                    "Holm p",
                ),
                rows=(
                    (
                        row["comparison"].replace("_", " "),
                        row["n_models"],
                        _interval(
                            row["pearson_r"],
                            row["pearson_ci_low"],
                            row["pearson_ci_high"],
                        ),
                        _pvalue(row["pearson_holm_p"]),
                        _interval(
                            row["spearman_rho"],
                            row["spearman_ci_low"],
                            row["spearman_ci_high"],
                        ),
                        _pvalue(row["spearman_holm_p"]),
                    )
                    for row in category_correlation_rows
                ),
            ),
        )
    if analysis_exclusions:
        atomic_write_text(
            tables / "analysis_exclusions.tex",
            _latex_table(
                caption=(
                    "Models withheld from confirmatory inference after the "
                    "documented route-quality audit. Raw records and descriptive "
                    "scores are retained, but the model does not enter rankings, "
                    "correlations, or paired comparisons."
                ),
                label=f"tab:ehq-analysis-exclusions{label_suffix}",
                column_spec="ll",
                headers=("Model", "Reason"),
                rows=(
                    (_label(name), str(row.get("reason") or "unspecified"))
                    for name, row in sorted(analysis_exclusions.items())
                ),
            ),
        )
    if sensitivity_rows:
        atomic_write_text(
            tables / "composite_weight_sensitivity.tex",
            _latex_table(
                caption=(
                    "Descriptive sensitivity of the composite to three "
                    "alternative weighting schemes. Correlations are with the "
                    "official EHQ and rank shifts are computed within the "
                    "fourteen-model confirmatory panel."
                ),
                label=f"tab:ehq-weight-sensitivity{label_suffix}",
                headers=(
                    "Scenario",
                    "Weights (EHQ1/EHQ2/EHQ3)",
                    "Pearson r",
                    "Spearman rho",
                    "Ranks changed",
                    "Max shift",
                ),
                rows=(
                    (
                        row["scenario"].replace("_", " "),
                        f"{row['weight_EHQ1']:.2f}/{row['weight_EHQ2']:.2f}/{row['weight_EHQ3']:.2f}",
                        _fmt(row["pearson_with_official_EHQ"], 3),
                        _fmt(row["spearman_with_official_EHQ"], 3),
                        row["n_models_with_rank_change"],
                        _fmt(row["maximum_absolute_rank_shift"], 1),
                    )
                    for row in sensitivity_rows
                ),
            ),
        )
    atomic_write_text(
        tables / "construct_scope_sensitivity.tex",
        _latex_table(
            caption=(
                "Sensitivity of EHQ to the scored category scope using the "
                "same retained responses. Strict unavailability removes HNQ; "
                "EHQ-K reports the learned-knowledge boundary and EHQ-C the "
                "withheld-context boundary. Correlations and rank shifts are "
                "relative to the official four-category composite."
            ),
            label=f"tab:ehq-construct-scope{label_suffix}",
            headers=(
                "Scenario",
                "Categories",
                "Pearson r",
                "Spearman rho",
                "Ranks changed",
                "Max shift",
            ),
            rows=(
                (
                    row["label"],
                    row["categories"],
                    _fmt(row["pearson_with_official"], 3),
                    _fmt(row["spearman_with_official"], 3),
                    _fmt(row["n_models_with_rank_change"], 0),
                    _fmt(row["maximum_absolute_rank_shift"], 1),
                )
                for row in construct_scope_rows
            ),
        ),
    )
    if rq1_rows:
        atomic_write_text(
            tables / "rq1_capability.tex",
            _latex_table(
                caption=(
                    "RQ1: relationship between measured task capability and "
                    "EHQ across the panel. The rank-shift column counts models "
                    "that move at least two places between the two orderings."
                ),
                label=f"tab:ehq-rq1{label_suffix}",
                headers=(
                    "Relationship",
                    "n",
                    "Pearson r [95% CI]",
                    "p",
                    "Spearman rho [95% CI]",
                    "p",
                    "Rank shifts >= 2",
                ),
                rows=(
                    (
                        row["relationship"].replace("_", " "),
                        row["n_models"],
                        _interval(
                            row["pearson_r"], row["pearson_ci_low"], row["pearson_ci_high"]
                        ),
                        _pvalue(row["pearson_p"]),
                        _interval(
                            row["spearman_rho"],
                            row["spearman_ci_low"],
                            row["spearman_ci_high"],
                        ),
                        _pvalue(row["spearman_p"]),
                        row["n_absolute_rank_shifts_ge_2"],
                    )
                    for row in rq1_rows
                ),
            ),
        )
    if rq2_rows:
        atomic_write_text(
            tables / "rq2_paired_summary.tex",
            _latex_table(
                caption=(
                    "RQ2: the new-minus-old generational comparison summarised "
                    "over the complete pairs. With this many pairs the exact "
                    "sign-permutation test has a smallest attainable p-value, "
                    "which the reported value should be read against."
                ),
                label=f"tab:ehq-rq2{label_suffix}",
                column_spec="rrrrrrr",
                headers=(
                    "Pairs",
                    "Mean diff.",
                    "Median diff.",
                    "Improved",
                    "Declined",
                    "Cohen's dz",
                    "Sign-permutation p",
                ),
                rows=(
                    (
                        row["n_pairs"],
                        _fmt(row["mean_difference"], 6),
                        _fmt(row["median_difference"], 6),
                        row["improved_pairs"],
                        row["declined_pairs"],
                        _fmt(row["cohens_dz"])
                        if row["cohens_dz"] is not None
                        else row["cohens_dz_status"],
                        _pvalue(row["exact_sign_permutation_p"])
                        if row["exact_sign_permutation_p"] is not None
                        else row["exact_sign_permutation_status"],
                    )
                    for row in rq2_rows
                ),
            ),
        )
    atomic_write_text(
        tables / "coverage_by_model.tex",
        _latex_table(
            caption=(
                "Scoring coverage per model. A record that a component could "
                "not score is dropped from that component's denominator and "
                "never imputed, so the denominators differ between EHQ1/EHQ2 "
                "and EHQ3 and are reported separately."
            ),
            label=f"tab:ehq-coverage{label_suffix}",
            headers=(
                "Model",
                "Items",
                "Scored EHQ1/2",
                "With confidence",
                "EHQ3 calibrated",
                "Excluded (all)",
                "Excluded (EHQ3)",
            ),
            rows=(
                (
                    _label(row["model"]),
                    f"{row['n_items']:,}",
                    f"{row['n_scored_ehq12']:,}",
                    f"{row['n_confidence']:,}",
                    f"{row['n_ehq3_calibration']:,}",
                    row["n_excluded_ehq12"],
                    row["n_excluded_ehq3_only"],
                )
                for row in coverage_rows
            ),
        ),
    )
    atomic_write_text(
        tables / "calibration_sources.tex",
        _latex_table(
            caption=(
                "Where the calibration evidence sits. The confidence prompt "
                "asks for 0 when no substantive answer was given, so "
                "Reported 0 is the share of abstentions that complied. The "
                "two ECE columns contrast the restraint records EHQ3 now "
                "excludes with the substantive records it is computed on."
            ),
            label=f"tab:ehq-calibration-sources{label_suffix}",
            headers=(
                "Model",
                "Abstentions",
                "Reported 0",
                "Restraint ECE",
                "Substantive n",
                "Substantive ECE",
            ),
            rows=(
                (
                    _label(row["model"]),
                    f"{row['n_abstain']:,}",
                    f"{row['abstain_instruction_compliance']:.1%}"
                    if row["abstain_instruction_compliance"] is not None
                    else "-",
                    _fmt(row["restraint_ece"])
                    if row["restraint_ece"] is not None
                    else "-",
                    f"{row['n_substantive']:,}",
                    _fmt(row["substantive_ece"])
                    if row["substantive_ece"] is not None
                    else "-",
                )
                for row in calibration_rows
            ),
        ),
    )
    atomic_write_text(
        tables / "calibration_definition_sensitivity.tex",
        _latex_table(
            caption=(
                "Numerical basis for restricting EHQ3 to substantive answers. "
                "Restraint accuracy and ECE describe records labelled ABSTAIN "
                "or HEDGE. The final column is official substantive-only EHQ3 "
                "minus the superseded score that pooled all response labels."
            ),
            label=f"tab:ehq3-definition-sensitivity{label_suffix}",
            headers=(
                "Model",
                "Restraint n",
                "Restraint acc.",
                "Restraint ECE",
                "Official EHQ3",
                "Change vs pooled",
            ),
            rows=(
                (
                    _label(row["model"]),
                    f"{row['n_restraint']:,}",
                    f"{float(row['restraint_accuracy']):.1%}",
                    _fmt(row["restraint_ece"]),
                    _fmt(1.0 - float(row["substantive_ece"])),
                    f"{float(row['ehq3_change_vs_superseded']):+.3f}",
                )
                for row in calibration_rows
            ),
        ),
    )
    if exclusion_rows:
        atomic_write_text(
            tables / "coverage_exclusions.tex",
            _latex_table(
                caption=(
                    "Every record excluded from at least one EHQ component, "
                    "with the component it was dropped from and why. Excluded "
                    "records are never imputed; they are removed from the "
                    "denominator of the affected component only."
                ),
                label=f"tab:ehq-exclusions{label_suffix}",
                headers=("Model", "Question", "Excluded from", "Reason", "Detail"),
                column_spec="lllll",
                rows=(
                    (
                        _label(row["model"]),
                        row["question_id"],
                        row["excluded_from"],
                        row["reason"].replace("_", " "),
                        str(row["detail"]).replace("_", " "),
                    )
                    for row in exclusion_rows
                ),
            ),
        )

    _plot_ranking(figures / "ehq-ranking", model_lookup, model_order)
    _plot_component_breakdown(figures / "component-breakdown", model_lookup, model_order)
    _plot_category_heatmap(figures / "category-heatmap", category_lookup, model_order)
    if pair_rows:
        _plot_pair_differences(figures / "pair-differences", pair_rows)
    _plot_response_distribution(figures / "response-distribution", response_rows, model_order)
    _plot_calibration_sources(figures / "calibration-sources", calibration_rows)

    pair_summary = analysis["rq2_generational_pairs"]["summary"] or {}
    zero_spanning_intervals = sum(
        row["ci_low"] <= 0 <= row["ci_high"] for row in pair_rows
    )
    ranking_lines = "\n".join(
        f"| {row['rank']} | {MODEL_LABELS.get(row['model'], row['model'])} | "
        f"{row['EHQ1']:.4f} | {row['EHQ2']:.4f} | {row['EHQ3']:.4f} | "
        f"{row['EHQ']:.6f} | [{row['EHQ_ci_low']:.4f}, {row['EHQ_ci_high']:.4f}] |"
        for row in model_rows
    )
    pair_lines = "\n".join(
        f"| {row['pair_label']} | {row['difference_new_minus_old']:+.6f} | "
        f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        for row in pair_rows
    )
    category_leaders = []
    for category in CATEGORY_ORDER:
        candidates = [
            row
            for row in category_rows
            if row["category"] == category and row["EHQ"] is not None
        ]
        if not candidates:
            category_leaders.append(f"- **{category}:** no defined EHQ in this panel")
            continue
        leader = max(candidates, key=lambda row: row["EHQ"])
        category_leaders.append(
            f"- **{category}:** {_label(leader['model'])} "
            f"({leader['EHQ']:.4f})"
        )

    # Cohen's dz and the exact sign-permutation p-value are undefined below two
    # pairs, and the whole section is omitted when no pair is complete.
    pair_effect = (
        f"{pair_summary['cohens_dz']:.4f}"
        if pair_summary.get("cohens_dz") is not None
        else f"undefined ({pair_summary.get('cohens_dz_status')})"
    )
    pair_permutation_p = (
        f"{pair_summary['exact_sign_permutation_p']:.4f}"
        if pair_summary.get("exact_sign_permutation_p") is not None
        else f"undefined ({pair_summary.get('exact_sign_permutation_status')})"
    )

    correlation_lines = "\n".join(
        f"| {row['comparison'].replace('_', ' ')} | {row['n_models']} | "
        f"{_interval(row['pearson_r'], row['pearson_ci_low'], row['pearson_ci_high'])} | "
        f"{_pvalue(row['pearson_holm_p'])} | "
        f"{_interval(row['spearman_rho'], row['spearman_ci_low'], row['spearman_ci_high'])} | "
        f"{_pvalue(row['spearman_holm_p'])} |"
        for row in correlation_rows
    )
    category_correlation_lines = "\n".join(
        f"| {row['comparison'].replace('_', ' ')} | {row['n_models']} | "
        f"{_interval(row['pearson_r'], row['pearson_ci_low'], row['pearson_ci_high'])} | "
        f"{_pvalue(row['pearson_holm_p'])} | "
        f"{_interval(row['spearman_rho'], row['spearman_ci_low'], row['spearman_ci_high'])} | "
        f"{_pvalue(row['spearman_holm_p'])} |"
        for row in category_correlation_rows
    )
    if rq1_rows:
        row = rq1_rows[0]
        rq1_line = (
            f"Capability versus EHQ across {row['n_models']} models: Pearson "
            f"{_interval(row['pearson_r'], row['pearson_ci_low'], row['pearson_ci_high'])}"
            f" (p={_pvalue(row['pearson_p'])}), Spearman "
            f"{_interval(row['spearman_rho'], row['spearman_ci_low'], row['spearman_ci_high'])}"
            f" (p={_pvalue(row['spearman_p'])}). "
            f"{row['n_absolute_rank_shifts_ge_2']} model(s) shift by two or more "
            "ranks between the capability and EHQ orderings. See "
            "`tables/rq1_capability.csv` and `tables/rq1_rank_shifts.csv`."
        )
    else:
        rq1_line = (
            "RQ1 is not reported: no capability scores were supplied to this "
            f"run (status: `{rq1.get('status')}`)."
        )

    statistics_section = f"""## Statistical analysis

Component correlations are computed across models, so the inference unit is the
model and n is the number of models with a defined EHQ. Intervals are bootstrap
percentile intervals; p-values come from permutation tests and are Holm-adjusted
within the Pearson and Spearman families of three tests each.

| Comparison | n | Pearson r [95% CI] | Holm p | Spearman rho [95% CI] | Holm p |
|---|---:|---:|---:|---:|---:|
{correlation_lines}

Category-level correlations are an explicitly exploratory analysis. Their
intervals can include substantively important effects even when a point
estimate is near zero; therefore a non-significant result is not evidence of
independence.

| Category comparison | n | Pearson r [95% CI] | Holm p | Spearman rho [95% CI] | Holm p |
|---|---:|---:|---:|---:|---:|
{category_correlation_lines}

{rq1_line}

Unadjusted p-values, permutation counts, and the RQ2 paired summary are in
`tables/component_correlations.csv`, `tables/category_correlations.csv`, and
`tables/rq2_paired_summary.csv`.

"""

    exclusion_notes = []
    if analysis_exclusions:
        exclusion_notes.append(
            "Withheld from confirmatory inference after route-quality audit: "
            + ", ".join(_label(name) for name in sorted(analysis_exclusions))
            + "."
        )
    if models_without_score:
        exclusion_notes.append(
            "Omitted for an undefined EHQ3: "
            + ", ".join(_label(name) for name in models_without_score)
            + "."
        )
    excluded_model_note = " ".join(exclusion_notes) or (
        "Every selected model produced a defined EHQ3 and passed analysis eligibility."
    )

    if pair_rows:
        pair_section = f"""## Generational-pair comparison

| New minus old pair | Difference | Paired-bootstrap 95% CI |
|---|---:|---:|
{pair_lines}

![Generational-pair differences](figures/pair-differences.png)

Across the {pair_summary['n_pairs']} currently complete pair(s), the mean
new-minus-old difference is {pair_summary['mean_difference']:+.6f} and the
median is {pair_summary['median_difference']:+.6f}.
{pair_summary['improved_pairs']} pair(s) increased and
{pair_summary['declined_pairs']} declined.
The model-level summary gives Cohen's dz={pair_effect} and an
exact sign-permutation p-value of {pair_permutation_p}.
With {pair_summary['n_pairs']} of the planned pairs complete, these are
exploratory calculations: they neither support systematic generational
improvement nor establish equivalence.
{zero_spanning_intervals} of {len(pair_rows)} paired-bootstrap interval(s)
span zero in this sample.

"""
    else:
        pair_section = """## Generational-pair comparison

No generational pair is complete in this panel, so RQ2 is not reported here.

"""

    if exclusion_rows:
        by_reason = Counter(
            (row["reason"], row["detail"]) for row in exclusion_rows
        )
        coverage_lines = "\n".join(
            f"| {row['model']} | {row['n_items']} | {row['n_scored_ehq12']} | "
            f"{row['n_ehq3_calibration']} | {row['n_excluded_ehq12']} | "
            f"{row['n_excluded_ehq3_only']} |"
            for row in coverage_rows
            if row["n_excluded_ehq12"] or row["n_excluded_ehq3_only"]
        )
        reason_lines = "\n".join(
            f"- `{reason}` / `{detail}`: {count} record(s)"
            for (reason, detail), count in sorted(by_reason.items())
        )
        coverage_section = f"""## Coverage and excluded records

{len(exclusion_rows)} of {sum(row['n_items'] for row in coverage_rows)} records
could not be scored by at least one component. They are excluded from the
affected denominators and never imputed. Every excluded record is listed with
its provider or parser reason in `tables/coverage_exclusions.csv`.

| Model | Items | Scored EHQ1/2 | EHQ3 calibrated | Excluded all | Excluded EHQ3 only |
|---|---:|---:|---:|---:|---:|
{coverage_lines}

{reason_lines}

The bootstrap applies the same per-record validity mask as the point
estimates, so a model with incomplete coverage is resampled over the records
it actually has rather than being dropped from the panel. Intervals for those
models rest on a marginally smaller effective sample; the affected counts are
given above so the difference can be judged directly.

"""
    else:
        coverage_section = ""

    report = f"""# {report_title}

{banner}
## Scope and provenance

- Source aggregate fingerprint: `{manifest['run']['fingerprint']}`
- Framework: `{manifest['framework_version']}`
- Protocol: `{manifest['protocol_version']}`
- EHQ3 protocol: `{EHQ3_PROTOCOL}` (calibration over substantive answers only)
- Dataset SHA-256: `{manifest['dataset']['sha256']}`
- Selected-question SHA-256: `{manifest['run']['selection']['question_ids_sha256']}`
- Confirmatory design: {len(model_order)} models x {manifest['run']['selection']['n_selected']} common items = {len(records)} records
- Retained descriptive-only models: {len(analysis_exclusions)}
- Selection balance: {_balance(per_category)} by category; {_balance(per_subcategory)} by subcategory
- Composite weights: EHQ1={weights['ehq1']}, EHQ2={weights['ehq2']}, EHQ3={weights['ehq3']}
- Confidence bins: {n_bins}
- Uncertainty procedure: {n_resamples:,} deterministic paired bootstrap resamples,
  stratified across the {len(per_subcategory)} subcategories; seed={seed}
- Source artifacts verified before report generation: {'yes' if verify_source else 'not applicable (generated inline by the runner)'}

## Model-level results

| Rank | Model | EHQ1 | EHQ2 | EHQ3 | EHQ | Stratified-bootstrap 95% CI |
|---:|---|---:|---:|---:|---:|---:|
{ranking_lines}

![EHQ ranking](figures/ehq-ranking.png)

![EHQ components and composite](figures/component-breakdown.png)

The intervals describe item-sampling variability within this fixed
{n_selected:,}-item evaluation. They do not incorporate model-registry
uncertainty, provider drift, or uncertainty in dataset construction and human
verification; those limitations remain outside the interval estimand.

{pair_section}## Category structure

![Category-level EHQ heatmap](figures/category-heatmap.png)

Category leaders in this evaluation:

{chr(10).join(category_leaders)}

Category patterns are heterogeneous. A high overall score can mask a weak
category, so no model should be characterized by the composite alone. In
particular, calibration behavior in CCQ differs substantially from restraint
and confident-error avoidance.

## Response behavior

![Response-type distribution](figures/response-distribution.png)

The response distribution separates abstention, hedging, confident correctness,
and confident error. EHQ1 and EHQ2 are consequently strongly related in this
panel, whereas EHQ3 contributes distinct calibration information. With only
{len(model_order)} models, all component correlations remain exploratory.

{coverage_section}{statistics_section}## Scientific interpretation boundaries

1. Dataset policy recorded for this run: `{dataset_policy}`.
   {excluded_model_note}
2. Open publication gates: {', '.join(publication_blockers) or 'none recorded'}.
3. {len(pair_rows)} generational pair(s) are complete in this panel; pairs whose
   counterpart is absent from the registry selection are not reported.
4. Records that no component could score are listed in
   `tables/coverage_exclusions.csv` and are excluded from the affected
   denominators rather than imputed.
5. EHQ3 follows `{EHQ3_PROTOCOL}`: calibration is measured only over
   substantive answers, so these values are not comparable with results
   produced before that definition took effect.

## Reproducibility outputs

- `tables/model_scores.csv` and `.tex`
- `tables/model_registry.csv` and `.tex`
- `tables/category_scores.csv`
- `tables/pair_differences.csv` and `.tex`
- `tables/response_distribution.csv`
- `tables/component_correlations.csv`
- `tables/category_correlations.csv` and `.tex` (exploratory)
- `tables/analysis_exclusions.csv` and `.tex` when a model is withheld
- `tables/composite_weight_sensitivity.csv` and `.tex`
- `tables/coverage_by_model.csv` and `.tex` and, when any record was excluded,
  `tables/coverage_exclusions.csv`
- `tables/component_correlations.csv` and `.tex` (Pearson and Spearman with
  bootstrap intervals, permutation p-values, and Holm-adjusted p-values)
- `tables/rq1_capability.csv`/`.tex` and `tables/rq1_rank_shifts.csv` when
  capability scores were supplied
- `tables/rq2_paired_summary.csv` and `.tex`
- `tables/calibration_sources.csv` and `.tex`
- `.tex` versions of the model, category, pair, response, correlation, and
  coverage tables
- PNG and PDF versions of every figure
- `report_data.json`, `report_manifest.json`, and `artifact_catalog.json`
"""
    atomic_write_text(output_dir / report_name, report)

    report_data = {
        "model_scores": model_rows,
        "model_registry": model_registry_rows,
        "category_scores": category_rows,
        "pair_differences": pair_rows,
        "response_distribution": response_rows,
        "component_correlations": correlation_rows,
        "category_correlations": category_correlation_rows,
        "analysis_exclusions": [
            analysis_exclusions[name] for name in sorted(analysis_exclusions)
        ],
        "composite_weight_sensitivity": sensitivity_rows,
        "construct_scope_sensitivity": construct_scope,
        "rq1_capability": rq1_rows,
        "rq1_rank_shifts": rq1_rank_rows,
        "rq2_paired_summary": rq2_rows,
        "coverage_by_model": coverage_rows,
        "coverage_exclusions": exclusion_rows,
        "calibration_sources": calibration_rows,
    }
    write_json(output_dir / "report_data.json", report_data)
    write_json(
        output_dir / "report_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "RELEASE_GATE_PASSED"
                if dataset_policy == "RELEASE_GATE_PASSED" and not publication_blockers
                else "NON_PUBLISHABLE_CANDIDATE"
            ),
            "ehq3_protocol": EHQ3_PROTOCOL,
            "latex_label_suffix": label_suffix,
            "coverage_policy": "report_and_exclude_no_imputation",
            "models_without_defined_ehq": models_without_score,
            "analysis_exclusions": [
                analysis_exclusions[name] for name in sorted(analysis_exclusions)
            ],
            "n_excluded_records": len(exclusion_rows),
            "models_with_incomplete_coverage": sorted(
                row["model"]
                for row in coverage_rows
                if row["n_excluded_ehq12"] or row["n_excluded_ehq3_only"]
            ),
            "source_aggregate_dir": str(aggregate_dir),
            "source_aggregate_fingerprint": manifest["run"]["fingerprint"],
            "source_manifest_sha256": sha256_file(aggregate_dir / "manifest.json"),
            # Built inline by the runner, the source catalog does not exist yet:
            # it is written once this package is in place, and covers it.
            "source_artifact_catalog_sha256": (
                sha256_file(source_catalog) if source_catalog.is_file() else None
            ),
            "source_artifacts_verified": verify_source,
            "report_tool_sha256": sha256_file(Path(__file__).resolve()),
            "bootstrap": {
                "method": "paired_percentile_bootstrap_stratified_by_subcategory",
                "n_resamples": n_resamples,
                "seed": seed,
                "confidence_level": 0.95,
            },
        },
    )
    catalog = write_artifact_catalog(output_dir)
    verification = verify_run_artifacts(output_dir)
    return {
        "output_dir": str(output_dir),
        "report_file": report_name,
        "n_models": len(model_order),
        "n_records": len(records),
        "n_resamples": n_resamples,
        "artifact_count": len(catalog["artifacts"]),
        "artifacts_verified": verification["valid"],
    }
