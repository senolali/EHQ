"""Add the per-model counts to a capability score file written without them.

`capability_scores.csv` originally carried only a rate. A rate cannot say
whether the models differ by more than resampling the same items would produce,
so RQ1 could correlate a measure that was at ceiling without noticing. The
counts are already in `capability_summary.json` next to it; this copies them
across, so no probe request is repeated.

Deriving the counts by multiplying the rate back out would be wrong. Denominators
differ: a model with one technical failure was scored over 258 items while its
peers were scored over 259, and 257/258 and 258/259 round to different rates for
that reason alone.

    python tools/backfill_capability_counts.py outputs/capability_probe-001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text  # noqa: E402

HEADER = "model,capability_score,n_correct,n_scored\n"


def backfill(run_dir: Path) -> dict:
    summary_path = run_dir / "capability_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"No capability_summary.json in {run_dir}")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["models"]

    lines = []
    for row in sorted(rows, key=lambda r: (-r["capability_score"], r["model"])):
        scored, correct = int(row["n_scored"]), int(row["n_correct"])
        if scored <= 0 or not 0 <= correct <= scored:
            raise SystemExit(
                f"{row['model']}: implausible counts {correct}/{scored}"
            )
        lines.append(
            f"{row['model']},{row['capability_score']},{correct},{scored}"
        )

    target = run_dir / "capability_scores.csv"
    atomic_write_text(target, HEADER + "\n".join(lines) + "\n")
    return {
        "capability_scores": str(target),
        "n_models": len(rows),
        "denominators": sorted({int(r["n_scored"]) for r in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(backfill(args.run_dir), indent=2))
    print(
        "\nNote: a run that recorded this file's SHA-256 will now refuse to "
        "regenerate against it. That refusal is correct -- rerun the analysis "
        "with the new file rather than restoring the old one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
