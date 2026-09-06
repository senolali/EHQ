"""Standalone entry point for the EHQ publication package builder.

The implementation lives in :mod:`ehq.publication` so the runner can generate
the same package inline when a study finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.publication import (  # noqa: E402
    _bootstrap_scores,
    _coverage_tables,
    _percentile_interval,
    _stratified_bootstrap_indices,
    build_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-prefix",
        help=(
            "Stable LaTeX label prefix for a manuscript build; defaults to the "
            "run id so ad-hoc reports never collide. Pass an empty string for "
            "bare labels such as tab:ehq-scores."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate into an existing output directory",
    )
    args = parser.parse_args()
    result = build_report(
        args.aggregate_dir,
        args.output_dir,
        n_resamples=args.n_resamples,
        seed=args.seed,
        overwrite=args.overwrite,
        label_prefix=args.label_prefix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
