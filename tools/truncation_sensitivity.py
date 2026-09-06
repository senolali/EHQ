"""Price truncated answers into the scores, instead of only counting them.

``audit_truncated_responses.py`` reports how many answers were cut off. This
tool asks the question that follows: if those answers had not been cut, where
would the model stand? It cannot know, so it computes the interval the answer
must lie in.

A cut answer is scored as if the model had finished speaking. When the visible
fragment reads ``the information requested is not available. The document
mentions the acquisition of``, the classifier sees a substantive continuation
and records CONFIDENT_WRONG, although the sentence that was interrupted was a
refusal. Every such record moves EHQ1 down, EHQ2 down, and enters the EHQ3
calibration set with a confidence value attached to an answer that was never
delivered.

Three scorings of the identical run bracket the truth:

``measured``
    The run as published. Truncated answers keep the labels they were given.
``dropped``
    Suspected records are removed from every denominator. This is the reading
    under which a cut answer is missing data rather than a wrong answer.
``restraint``
    Suspected CONFIDENT_WRONG records are relabelled HEDGE, which is what the
    surviving fragments read like. They leave the EHQ3 calibration set by the
    same rule that governs the official protocol, so EHQ3 is recomputed rather
    than held fixed. Truncated CONFIDENT_CORRECT records are left alone: the
    correct value did reach the output before the cut.

``measured`` is not necessarily the low end and ``restraint`` not necessarily
the high end for every model, so the reported interval is the span across all
three, and the rank interval is the span of that model's position in the panel
under each scoring. A model whose interval covers one or two adjacent places is
imprecise. A model whose interval covers half the panel is undetermined, and
that is a statement about the measurement rather than about the model.

Rank intervals are reported for every model, including untruncated ones. A rank
is a position within the panel, so a model whose own scores are identical under
all three scorings can still be uncertain about its place when a peer's are not.

    python tools/truncation_sensitivity.py outputs/real_<run-id>
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text, write_json  # noqa: E402
from ehq.constants import EHQ3_PROTOCOL  # noqa: E402
from ehq.evaluation.scoring import compute_ehq_scores  # noqa: E402
from ehq.publication import _latex_escape  # noqa: E402

SCENARIOS = ("measured", "dropped", "restraint")


def _audit_module():
    """Load the audit tool so both share one definition of "truncated"."""

    path = Path(__file__).resolve().parent / "audit_truncated_responses.py"
    spec = importlib.util.spec_from_file_location("audit_truncated_responses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = _audit_module()


def _is_suspected(row: Mapping[str, Any], answer_max_tokens: int | None) -> bool:
    text, envelope = AUDIT._answer(row)
    if not isinstance(text, str) or not text.strip():
        return False
    tokens = AUDIT._completion_tokens(envelope)
    return AUDIT.signals(text, tokens, answer_max_tokens)["suspected"]


def _as_restraint(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Relabel a truncated confident-wrong answer as the hedge it was becoming."""

    amended = copy.deepcopy(dict(row))
    classification = dict(amended.get("classification") or {})
    classification["label"] = "HEDGE"
    classification["is_correct"] = False
    classification["hedge_detected"] = True
    classification["substantive_answer_detected"] = False
    classification["reasons"] = list(classification.get("reasons") or []) + [
        "truncation_sensitivity: relabelled from CONFIDENT_WRONG"
    ]
    amended["classification"] = classification
    return amended


def analyse(
    run_dir: Path, *, answer_max_tokens: int | None, weights: Mapping[str, float]
) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    by_model: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)

    models: Dict[str, Dict[str, Any]] = {}
    for model, model_rows in by_model.items():
        suspected = [_is_suspected(row, answer_max_tokens) for row in model_rows]
        n_suspected = sum(suspected)
        n_suspected_cw = sum(
            flag
            and (row.get("classification") or {}).get("label") == "CONFIDENT_WRONG"
            for row, flag in zip(model_rows, suspected)
        )
        variants = {
            "measured": list(model_rows),
            "dropped": [
                row for row, flag in zip(model_rows, suspected) if not flag
            ],
            "restraint": [
                _as_restraint(row)
                if flag
                and (row.get("classification") or {}).get("label")
                == "CONFIDENT_WRONG"
                else row
                for row, flag in zip(model_rows, suspected)
            ],
        }
        models[model] = {
            "n_records": len(model_rows),
            "n_suspected": n_suspected,
            "suspected_rate": n_suspected / len(model_rows) if model_rows else 0.0,
            "n_suspected_confident_wrong": n_suspected_cw,
            "scores": {
                name: compute_ehq_scores(records, weights=weights)
                for name, records in variants.items()
            },
        }

    # Rank each model in each scoring, so the reported interval is the span of
    # positions the run is consistent with rather than a spread of point values.
    for scenario in SCENARIOS:
        ordered = sorted(
            (
                (model, stats["scores"][scenario]["EHQ"])
                for model, stats in models.items()
                if stats["scores"][scenario]["EHQ"] is not None
            ),
            key=lambda pair: -pair[1],
        )
        for position, (model, _) in enumerate(ordered, 1):
            models[model].setdefault("ranks", {})[scenario] = position

    for stats in models.values():
        values = [
            stats["scores"][name]["EHQ"]
            for name in SCENARIOS
            if stats["scores"][name]["EHQ"] is not None
        ]
        ranks = list((stats.get("ranks") or {}).values())
        stats["ehq_interval"] = [min(values), max(values)] if values else None
        stats["ehq_span"] = (max(values) - min(values)) if values else None
        stats["rank_interval"] = [min(ranks), max(ranks)] if ranks else None
        stats["rank_span"] = (max(ranks) - min(ranks)) if ranks else None

    return {
        "run_dir": str(run_dir),
        "ehq3_protocol": EHQ3_PROTOCOL,
        "answer_max_tokens": answer_max_tokens,
        "weights": dict(weights),
        "n_records": len(rows),
        "scenarios": list(SCENARIOS),
        "models": models,
    }


def _csv(report: Mapping[str, Any]) -> str:
    header = (
        "model,n_records,n_suspected,suspected_rate,n_suspected_confident_wrong,"
        + ",".join(
            f"{metric}_{scenario}"
            for scenario in SCENARIOS
            for metric in ("EHQ1", "EHQ2", "EHQ3", "EHQ", "rank")
        )
        + ",ehq_span,rank_low,rank_high\n"
    )
    lines = []
    for model, stats in sorted(
        report["models"].items(),
        key=lambda kv: -(kv[1]["scores"]["measured"]["EHQ"] or 0.0),
    ):
        cells = [
            model,
            str(stats["n_records"]),
            str(stats["n_suspected"]),
            f"{stats['suspected_rate']:.6f}",
            str(stats["n_suspected_confident_wrong"]),
        ]
        for scenario in SCENARIOS:
            score = stats["scores"][scenario]
            for metric in ("EHQ1", "EHQ2", "EHQ3", "EHQ"):
                value = score[metric]
                cells.append("" if value is None else f"{value:.6f}")
            cells.append(str((stats.get("ranks") or {}).get(scenario, "")))
        cells.append("" if stats["ehq_span"] is None else f"{stats['ehq_span']:.6f}")
        interval = stats["rank_interval"] or ["", ""]
        cells.extend(str(value) for value in interval)
        lines.append(",".join(cells))
    return header + "\n".join(lines) + "\n"


def _latex(report: Mapping[str, Any]) -> str:
    rows = []
    for model, stats in sorted(
        report["models"].items(),
        key=lambda kv: -(kv[1]["scores"]["measured"]["EHQ"] or 0.0),
    ):
        ranks = stats.get("ranks") or {}
        interval = stats["rank_interval"] or [None, None]
        rank_cell = (
            "--"
            if interval[0] is None
            else (
                str(interval[0])
                if interval[0] == interval[1]
                else f"{interval[0]}--{interval[1]}"
            )
        )

        def cell(scenario: str) -> str:
            value = stats["scores"][scenario]["EHQ"]
            return "--" if value is None else f"{value:.3f}"

        rows.append(
            f"{_latex_escape(model)} & {stats['n_suspected']} & "
            f"{stats['suspected_rate'] * 100:.1f}\\% & "
            f"{cell('measured')} & {cell('dropped')} & {cell('restraint')} & "
            f"{ranks.get('measured', '--')} & {rank_cell} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Generated by tools/truncation_sensitivity.py -- do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Sensitivity of the composite EHQ to truncated answers. "
        "\\emph{Measured} is the published scoring; \\emph{dropped} removes "
        "suspected records from every denominator; \\emph{restraint} relabels "
        "suspected \\textsc{confident\\_wrong} records as \\textsc{hedge}. The "
        "rank interval is the span of panel positions the run is consistent "
        "with.}\n"
        "\\label{tab:truncation-sensitivity}\n"
        "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Model & Cut & Rate & Measured & Dropped & Restraint & Rank & "
        "Rank interval \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def resolve_output_dir(run_dir: Path, requested: Path | None) -> Path:
    """Derived analyses live beside the run, never inside it.

    A completed run is sealed by a SHA-256 catalogue that lists every file it
    contains, and `verify_run_artifacts` treats an uncatalogued file as a
    failure. That is the correct behaviour: it is what makes the catalogue a
    guarantee rather than a description. Writing a later analysis into the run
    directory therefore invalidates the run, and re-cataloguing to make the
    complaint go away would mean the catalogue certifies whatever happens to be
    present rather than what the run produced.
    """

    run_dir = run_dir.resolve()
    if requested is None:
        return run_dir.parent / f"{run_dir.name}_analysis"
    resolved = requested.resolve()
    if resolved == run_dir or run_dir in resolved.parents:
        raise SystemExit(
            f"Refusing to write inside the run directory: {resolved}\n"
            "A completed run is sealed by its artifact catalogue and an extra "
            "file invalidates it. Choose a location outside "
            f"{run_dir}, or omit --output-dir to use "
            f"{run_dir.parent / (run_dir.name + '_analysis')}."
        )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the CSV, LaTeX and JSON (default: <run_dir>/sensitivity)",
    )
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    snapshot = (manifest.get("config") or {}).get("snapshot") or {}
    limit = args.answer_max_tokens
    if limit is None:
        value = (snapshot.get("inference") or {}).get("max_tokens")
        limit = value if isinstance(value, int) else None
    weights = snapshot.get("weights") or {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}

    report = analyse(args.run_dir, answer_max_tokens=limit, weights=weights)
    output_dir = resolve_output_dir(args.run_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "truncation_sensitivity.json", report)
    atomic_write_text(output_dir / "truncation_sensitivity.csv", _csv(report))
    atomic_write_text(output_dir / "truncation_sensitivity.tex", _latex(report))

    print(f"run: {report['run_dir']}")
    print(f"answer max_tokens: {limit if limit is not None else 'unknown'}")
    print(f"written to: {output_dir}")
    print()
    header = (
        f"{'model':<22}{'cut':>7}{'rate':>8}{'measured':>10}{'dropped':>10}"
        f"{'restraint':>11}{'span':>8}{'rank':>7}{'interval':>11}"
    )
    print(header)
    print("-" * len(header))
    for model, stats in sorted(
        report["models"].items(),
        key=lambda kv: -(kv[1]["scores"]["measured"]["EHQ"] or 0.0),
    ):
        ranks = stats.get("ranks") or {}
        interval = stats["rank_interval"] or [None, None]
        span_cell = (
            "-"
            if interval[0] is None
            else (
                str(interval[0])
                if interval[0] == interval[1]
                else f"{interval[0]}-{interval[1]}"
            )
        )

        def cell(scenario: str) -> str:
            value = stats["scores"][scenario]["EHQ"]
            return "-" if value is None else f"{value:.4f}"

        span = stats["ehq_span"]
        span_value = "-" if span is None else f"{span:.4f}"
        print(
            f"{model:<22}{stats['n_suspected']:>7}"
            f"{stats['suspected_rate']:>7.1%}"
            f"{cell('measured'):>10}{cell('dropped'):>10}{cell('restraint'):>11}"
            f"{span_value:>8}"
            f"{str(ranks.get('measured', '-')):>7}{span_cell:>11}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
