"""Detect answers that were cut off rather than completed.

A truncated answer is not a technical failure: the provider returns success and
the text is scored as if the model had finished speaking. It therefore reaches
the classifier and the metrics silently. This audit reads the retained records
of any completed run and reports, per model, how many answers show evidence of
truncation and what they were labelled.

Evidence is reported at two levels, because a single record is often
ambiguous while a model-level rate is not. "The institute was founded in 1967"
with no full stop is probably complete; "the identifier assigned to" is
certainly not; and "...flight to Greenville, South Carolina" is cut off in a
way no syntactic rule can see. Per-record rules therefore carry the strong
signals only, and the weak signal is judged by comparing each model against the
panel.

Strong, counted as suspected:

``token_limit``
    Reported completion tokens reached the configured answer budget.
``incomplete_clause``
    The text stops on a function word ("assigned to", "opened on"), on a
    dangling separator ("March 22,"), or inside an unclosed quote or bracket
    ("is 'US202612").

Weak, reported but not counted:

``unpunctuated``
    Prose ending without sentence-final punctuation. Harmless in isolation, so
    it is compared against the panel median instead: a model whose rate is far
    above its peers is truncating even where no single record proves it.

    python tools/audit_truncated_responses.py outputs/real_<run-id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TERMINAL = tuple(".!?\"'”’)]}")
# A clause that stops on one of these has not finished saying anything.
TAIL_FUNCTION_WORDS = (
    "of to in on at by for with from into over under about as than per via "
    "the a an this that these those its their his her our your "
    "and or but nor so yet because while if when where although though "
    "is are was were be been being am has have had having "
    "will would shall should can could may might must do does did "
    "which who whom whose what including such between during against"
).split()
INCOMPLETE_TAIL = re.compile(
    r"\b(?:" + "|".join(TAIL_FUNCTION_WORDS) + r")$", re.IGNORECASE
)
DANGLING_SEPARATOR = re.compile(r"[,;:\-\u2013\u2014/]$")
OPENING_SINGLE_QUOTE = re.compile(r"(?:^|\s)['\u2018]")
CLOSING_SINGLE_QUOTE = re.compile(r"['\u2019](?=\s|$|[.,;:!?])")
# Completion-token counts appear under different names across providers.
TOKEN_KEYS = (
    "completion_tokens",
    "output_tokens",
    "completion_token_count",
    "output_token_count",
    "generated_tokens",
)
PROSE_WORDS = 5


def signals(text: str, tokens: int | None, answer_max_tokens: int | None) -> Dict[str, bool]:
    """Per-record truncation evidence, shared with the sensitivity analysis.

    Kept as one function so the audit that reports truncation and the analysis
    that prices it in cannot drift apart: a record counted as suspected in the
    table is the same record reclassified in the bounds.
    """

    stripped = text.rstrip()
    at_limit = bool(
        tokens is not None
        and answer_max_tokens is not None
        and tokens >= answer_max_tokens
    )
    prose = len(stripped.split()) >= PROSE_WORDS
    # Unclosed quote or bracket: the model was still inside a citation.
    # Single-quote parity is useless in English because possessives and
    # contractions consume apostrophes, so opening and closing quotation
    # marks are matched by position instead: an apostrophe inside a word
    # ("Graham's") is neither, while a quote opened after whitespace and
    # never closed ("is 'US202612") is evidence of a cut.
    unbalanced = (
        stripped.count('"') % 2 == 1
        or len(OPENING_SINGLE_QUOTE.findall(stripped))
        > len(CLOSING_SINGLE_QUOTE.findall(stripped))
        or stripped.count("(") > stripped.count(")")
        or stripped.count("[") > stripped.count("]")
    )
    incomplete = bool(
        prose
        and not stripped.endswith(TERMINAL)
        and (
            INCOMPLETE_TAIL.search(stripped)
            or DANGLING_SEPARATOR.search(stripped)
            or unbalanced
        )
    )
    return {
        "token_limit": at_limit,
        "incomplete_clause": incomplete,
        "unpunctuated": bool(prose and not stripped.endswith(TERMINAL)),
        "suspected": bool(at_limit or incomplete),
    }


def _answer(row: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any]]:
    """Return the answer text and its response envelope for either run shape."""

    envelope = row.get("answer_response")
    if isinstance(envelope, Mapping):
        return envelope.get("text"), envelope
    # Capability-probe records store the text directly.
    return row.get("response_text"), {}


def _completion_tokens(envelope: Mapping[str, Any]) -> int | None:
    usage = envelope.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for key in TOKEN_KEYS:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def audit(run_dir: Path, *, answer_max_tokens: int | None) -> Dict[str, Any]:
    records_path = run_dir / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    per_model: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model"))
        stats = per_model.setdefault(
            model,
            {
                "n_answers": 0,
                "token_limit": 0,
                "incomplete_clause": 0,
                "unpunctuated": 0,
                "suspected": 0,
                "labels_of_suspected": Counter(),
                "examples": [],
                "max_completion_tokens": 0,
            },
        )
        text, envelope = _answer(row)
        if not isinstance(text, str) or not text.strip():
            continue
        stats["n_answers"] += 1
        stripped = text.rstrip()

        tokens = _completion_tokens(envelope)
        if tokens is not None:
            stats["max_completion_tokens"] = max(
                stats["max_completion_tokens"], tokens
            )
        flags = signals(text, tokens, answer_max_tokens)
        stats["token_limit"] += int(flags["token_limit"])
        stats["incomplete_clause"] += int(flags["incomplete_clause"])
        stats["unpunctuated"] += int(flags["unpunctuated"])

        if flags["suspected"]:
            stats["suspected"] += 1
            label = (row.get("classification") or {}).get("label")
            stats["labels_of_suspected"][str(label)] += 1
            if len(stats["examples"]) < 5:
                stats["examples"].append(
                    {
                        "question_id": row.get("question_id"),
                        "label": label,
                        "completion_tokens": tokens,
                        "text_tail": " ".join(stripped.split())[-110:],
                    }
                )
    # A single unpunctuated answer proves nothing; a model whose rate is far
    # above every peer on the identical item set is another matter. The panel
    # supplies the baseline, so no absolute threshold has to be invented.
    scored = [s for s in per_model.values() if s["n_answers"]]
    for stats in scored:
        n = stats["n_answers"]
        stats["suspected_rate"] = stats["suspected"] / n
        stats["unpunctuated_rate"] = stats["unpunctuated"] / n
    rates = sorted(s["unpunctuated_rate"] for s in scored)
    median = (
        0.0
        if not rates
        else (
            rates[len(rates) // 2]
            if len(rates) % 2
            else (rates[len(rates) // 2 - 1] + rates[len(rates) // 2]) / 2
        )
    )
    for stats in scored:
        rate = stats["unpunctuated_rate"]
        stats["panel_median_unpunctuated"] = median
        stats["rate_vs_panel_median"] = (
            None if median <= 0 else round(rate / median, 1)
        )
        # Outlier when the rate is both several times the panel norm and
        # materially large, so a panel that is uniformly tidy cannot manufacture
        # an alarm out of a rounding difference.
        stats["panel_outlier"] = bool(
            rate >= max(3 * median, 0.10) and rate > 0.02
        )
    return {
        "run_dir": str(run_dir),
        "n_records": len(rows),
        "panel_median_unpunctuated_rate": median,
        "models": per_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=None,
        help="Configured answer budget; read from the run manifest when omitted",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    limit = args.answer_max_tokens
    if limit is None:
        manifest_path = args.run_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot = (manifest.get("config") or {}).get("snapshot") or {}
            inference = snapshot.get("inference") or {}
            value = inference.get("max_tokens")
            if isinstance(value, int):
                limit = value

    report = audit(args.run_dir, answer_max_tokens=limit)
    if args.json:
        serialisable = {
            **report,
            "models": {
                name: {**stats, "labels_of_suspected": dict(stats["labels_of_suspected"])}
                for name, stats in report["models"].items()
            },
        }
        print(json.dumps(serialisable, ensure_ascii=False, indent=2))
        return 0

    print(f"run: {report['run_dir']}")
    print(f"answer max_tokens: {limit if limit is not None else 'unknown'}")
    print()
    print(
        "panel median unpunctuated rate: "
        f"{report['panel_median_unpunctuated_rate']:.1%}"
    )
    print()
    header = (
        f"{'model':<22}{'answers':>8}{'suspected':>11}{'rate':>8}"
        f"{'tok_limit':>10}{'incompl':>9}{'unpunct':>9}{'vs_panel':>10}"
        f"{'max_tok':>9}  flag"
    )
    print(header)
    print("-" * len(header))
    for name, stats in sorted(
        report["models"].items(),
        key=lambda kv: -kv[1].get("suspected_rate", 0),
    ):
        ratio = stats.get("rate_vs_panel_median")
        print(
            f"{name:<22}{stats['n_answers']:>8}{stats['suspected']:>11}"
            f"{stats.get('suspected_rate', 0):>7.1%}"
            f"{stats['token_limit']:>10}{stats['incomplete_clause']:>9}"
            f"{stats['unpunctuated']:>9}"
            f"{(f'{ratio}x' if ratio is not None else '-'):>10}"
            f"{stats['max_completion_tokens'] or '-':>9}"
            f"  {'OUTLIER' if stats.get('panel_outlier') else ''}"
        )

    for name, stats in sorted(
        report["models"].items(), key=lambda kv: -kv[1]["suspected"]
    ):
        if not stats["suspected"]:
            continue
        print(f"\n--- {name}: {stats['suspected']} supheli ---")
        if stats["labels_of_suspected"]:
            print("   etiketler:", dict(stats["labels_of_suspected"]))
        for example in stats["examples"]:
            print(
                f"   [{example['question_id']}] label={example['label']} "
                f"tokens={example['completion_tokens']}"
            )
            print(f"     ...{example['text_tail']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
