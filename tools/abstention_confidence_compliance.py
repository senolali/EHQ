"""How models answer the confidence question after declining to answer at all.

The confidence prompt instructs: *if you did not provide a substantive factual
answer, report 0*. An `ABSTAIN` record is exactly that case, so the correct
response is known in advance and compliance is directly measurable. It turns out
not to be uniform, and the pattern is the reason EHQ3 is now computed over
substantive answers only: a model that declines to answer and then reports
maximum confidence contributes a confidence value that describes nothing about
the answer it did not give.

Two exclusions keep the measurement honest.

Truncated answers are dropped. A cut answer reaches the classifier as though it
were finished, so a model that truncates heavily has an `ABSTAIN` pool
consisting of the refusals whose abstention wording happened to survive the
cut -- a selected subset, not a sample. The truncation rule is imported from
`audit_truncated_responses`, so a record dropped here is one the run-level audit
also counts.

Models whose truncation makes that pool unrepresentative are excluded entirely
via `--exclude`, and the exclusion is printed with the results rather than
applied silently.

    python tools/abstention_confidence_compliance.py outputs/real_<run-id> \\
        --exclude Gemini-2.5-Pro
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text, write_json  # noqa: E402
from ehq.config import load_models  # noqa: E402
from ehq.publication import _latex_escape  # noqa: E402

RESTRAINT_LABELS = ("ABSTAIN", "HEDGE")

# The registry's `model_provider` is the serving route, not the organisation
# that trained the model: Claude, LLaMA, Nova and GPT-OSS all arrive through
# `aws`. Grouping on it would invent a cluster spanning four developers, so the
# developer is read from the model's own name and the serving route is reported
# beside it rather than instead of it.
DEVELOPERS = (
    ("claude", "Anthropic"),
    ("gpt", "OpenAI"),
    ("gemini", "Google"),
    ("gemma", "Google"),
    ("llama", "Meta"),
    ("nova", "Amazon"),
    ("deepseek", "DeepSeek"),
    ("mistral", "Mistral"),
    ("qwen", "Alibaba"),
)


def developer_of(model_name: str) -> str:
    lowered = model_name.lower()
    for prefix, developer in DEVELOPERS:
        if lowered.startswith(prefix):
            return developer
    return "unknown"


def _audit_module():
    path = Path(__file__).resolve().parent / "audit_truncated_responses.py"
    spec = importlib.util.spec_from_file_location("audit_truncated_responses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = _audit_module()


def _is_truncated(row: Mapping[str, Any], answer_max_tokens: int | None) -> bool:
    text, envelope = AUDIT._answer(row)
    if not isinstance(text, str) or not text.strip():
        return False
    tokens = AUDIT._completion_tokens(envelope)
    return AUDIT.signals(text, tokens, answer_max_tokens)["suspected"]


def profile(
    run_dir: Path,
    *,
    answer_max_tokens: int | None,
    excluded: List[str],
    serving: Mapping[str, str],
) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    models: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model"))
        if model in excluded:
            continue
        label = (row.get("classification") or {}).get("label")
        if label not in RESTRAINT_LABELS:
            continue
        if not row.get("valid_for_ehq3"):
            continue
        value = row.get("parsed_confidence")
        if not isinstance(value, (int, float)):
            continue
        stats = models.setdefault(
            model,
            {
                "developer": developer_of(model),
                "serving": serving.get(model, "unknown"),
                "abstain": [],
                "hedge": [],
                "dropped_truncated": 0,
            },
        )
        if _is_truncated(row, answer_max_tokens):
            stats["dropped_truncated"] += 1
            continue
        key = "abstain" if label == "ABSTAIN" else "hedge"
        stats[key].append(float(value))

    for stats in models.values():
        values = stats["abstain"]
        n = len(values)
        stats["n_abstain"] = n
        stats["compliance"] = (
            sum(1 for v in values if v == 0.0) / n if n else None
        )
        stats["max_confidence_rate"] = (
            sum(1 for v in values if v >= 1.0) / n if n else None
        )
        # The distribution is what makes the finding: values cluster at the two
        # ends rather than spreading, so a mean alone would misdescribe it.
        stats["high_confidence_rate"] = (
            sum(1 for v in values if v >= 0.85) / n if n else None
        )
        # Non-compliance takes two forms and they mean different things. A
        # model that answers exactly 1.0 is treating the scale as a switch. A
        # model that answers 0.95, 0.97, 0.98 is using it as a scale, which
        # suggests it is reporting confidence in something other than the
        # answer it declined to give. Collapsing both into "high" hides the
        # distinction that matters.
        stats["graded_high_rate"] = (
            sum(1 for v in values if 0.85 <= v < 1.0) / n if n else None
        )
        stats["scale_use"] = (
            None
            if not n
            else ("graded" if (stats["graded_high_rate"] or 0) >= 0.10 else "binary")
        )
        stats["mean_confidence"] = (sum(values) / n) if n else None
        stats["most_common"] = Counter(values).most_common(4)
        hedge = stats["hedge"]
        stats["n_hedge"] = len(hedge)
        stats["hedge_mean_confidence"] = (
            sum(hedge) / len(hedge) if hedge else None
        )

    by_family: Dict[str, Dict[str, Any]] = {}
    for model, stats in models.items():
        if stats["compliance"] is None:
            continue
        entry = by_family.setdefault(
            stats["developer"], {"models": [], "compliance": []}
        )
        entry["models"].append(model)
        entry["compliance"].append(stats["compliance"])
    for entry in by_family.values():
        values = entry["compliance"]
        entry["n_models"] = len(values)
        entry["min"] = min(values)
        entry["max"] = max(values)
        entry["mean"] = sum(values) / len(values)

    return {
        "run_dir": str(run_dir),
        "answer_max_tokens": answer_max_tokens,
        "excluded_models": excluded,
        "instruction": (
            "if you did not provide a substantive factual answer, report 0"
        ),
        "n_records": len(rows),
        "models": models,
        "developers": by_family,
    }


def _csv(report: Mapping[str, Any]) -> str:
    header = (
        "model,developer,serving_provider,n_abstain,dropped_truncated,compliance_rate,"
        "high_confidence_rate,graded_high_rate,max_confidence_rate,scale_use,"
        "mean_confidence,"
        "n_hedge,hedge_mean_confidence\n"
    )
    lines = []
    for model, stats in sorted(
        report["models"].items(), key=lambda kv: -(kv[1]["compliance"] or 0.0)
    ):
        def cell(value):
            return "" if value is None else f"{value:.6f}"

        lines.append(
            ",".join(
                [
                    model,
                    stats["developer"],
                    stats["serving"],
                    str(stats["n_abstain"]),
                    str(stats["dropped_truncated"]),
                    cell(stats["compliance"]),
                    cell(stats["high_confidence_rate"]),
                    cell(stats["graded_high_rate"]),
                    cell(stats["max_confidence_rate"]),
                    str(stats["scale_use"] or ""),
                    cell(stats["mean_confidence"]),
                    str(stats["n_hedge"]),
                    cell(stats["hedge_mean_confidence"]),
                ]
            )
        )
    return header + "\n".join(lines) + "\n"


def _latex(report: Mapping[str, Any]) -> str:
    rows = []
    for model, stats in sorted(
        report["models"].items(), key=lambda kv: -(kv[1]["compliance"] or 0.0)
    ):
        if stats["compliance"] is None:
            continue
        rows.append(
            f"{_latex_escape(model)} & {_latex_escape(stats['developer'])} & "
            f"{stats['n_abstain']} & {stats['compliance'] * 100:.1f}\\% & "
            f"{stats['graded_high_rate'] * 100:.1f}\\% & "
            f"{stats['max_confidence_rate'] * 100:.1f}\\% & "
            f"{stats['scale_use']} \\\\"
        )
    excluded = report["excluded_models"]
    note = (
        ""
        if not excluded
        else (
            " Excluded from this analysis: "
            + ", ".join(_latex_escape(name) for name in excluded)
            + ", whose truncation rate makes its abstention set a selected "
            "subset rather than a sample."
        )
    )
    return (
        "% Generated by tools/abstention_confidence_compliance.py.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Compliance with the confidence instruction on abstentions. "
        "The prompt directs a model that gave no substantive answer to report "
        "0, so the correct value is known. \\emph{Complied} is the share of "
        "abstentions reporting exactly 0. The remainder splits in two: "
        "\\emph{graded} reports a value in $[0.85, 1)$, treating the scale as "
        "continuous, while \\emph{exactly 1} treats it as a switch."
        " Counts are complete-case counts after dropping restraint responses "
        "flagged as suspected truncations; they can therefore differ from the "
        "official response-distribution counts."
        f"{note}}}\n"
        "\\label{tab:abstention-confidence-compliance}\n"
        "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
        "\\begin{tabular}{llrrrrl}\n\\toprule\n"
        "Model & Developer & Abstentions & Complied & Graded & Exactly 1 & "
        "Scale \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}%\n}\n\\end{table}\n"
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
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--registry", default="config/models.json", type=Path)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    limit = args.answer_max_tokens
    manifest_path = args.run_dir / "manifest.json"
    if limit is None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = (manifest.get("config") or {}).get("snapshot") or {}
        value = (snapshot.get("inference") or {}).get("max_tokens")
        limit = value if isinstance(value, int) else None

    serving = {}
    if args.registry.is_file():
        serving = {
            model.name: (model.model_provider or model.provider)
            for model in load_models(args.registry)
        }

    report = profile(
        args.run_dir,
        answer_max_tokens=limit,
        excluded=list(args.exclude),
        serving=serving,
    )

    output_dir = resolve_output_dir(args.run_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "abstention_confidence_compliance.json", report)
    atomic_write_text(
        output_dir / "abstention_confidence_compliance.csv", _csv(report)
    )
    atomic_write_text(
        output_dir / "abstention_confidence_compliance.tex", _latex(report)
    )

    print(f"run       : {report['run_dir']}")
    print(f"talimat   : \"{report['instruction']}\"")
    if report["excluded_models"]:
        print(f"dislanan  : {', '.join(report['excluded_models'])}")
    print(f"yazildi   : {output_dir}")
    print()
    header = (
        f"{'model':<20}{'gelistirici':<12}{'abst_n':>8}{'kirpik':>7}"
        f"{'uyum':>8}{'kademeli':>10}{'=1.0':>8}{'olcek':>9}   en sik"
    )
    print(header)
    print("-" * (len(header) + 24))
    for model, stats in sorted(
        report["models"].items(), key=lambda kv: -(kv[1]["compliance"] or 0.0)
    ):
        if stats["compliance"] is None:
            continue
        common = ", ".join(
            f"{value:g}:{count}" for value, count in stats["most_common"]
        )
        print(
            f"{model:<20}{stats['developer']:<12}"
            f"{stats['n_abstain']:>8}{stats['dropped_truncated']:>7}"
            f"{stats['compliance']:>7.1%}{stats['graded_high_rate']:>10.1%}"
            f"{stats['max_confidence_rate']:>8.1%}{stats['scale_use']:>9}"
            f"   {common}"
        )

    print("\ngelistiriciye gore uyum:")
    for developer, entry in sorted(
        report["developers"].items(), key=lambda kv: -kv[1]["mean"]
    ):
        print(
            f"  {developer:<16} n={entry['n_models']}  "
            f"{entry['min']:.1%} - {entry['max']:.1%}  "
            f"(ortalama {entry['mean']:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
