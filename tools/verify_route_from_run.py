"""Promote a registry route to `verified` from a completed run's evidence.

The verification fields in `config/models.json` -- `operational_status`,
`endpoint_verified_at`, `endpoint_verified_resolved_model`, and the
non-reasoning attestation -- assert things that a run either demonstrates or
does not. Typing them by hand puts a claim in the registry that no artifact
backs, which is how GPT-5-mini came to be attested at zero thinking tokens on
the strength of one short probe while a later run contradicted it.

This reads a completed pilot and writes those fields only when the run supports
them. It checks four things and refuses on any failure:

* every call resolved to the exact requested route, so no alias substituted a
  different model;
* coverage clears the thresholds, counted over EHQ1/EHQ2 validity and over
  confidence parsing separately, because those failed independently for
  GPT-5-mini;
* no provider-reported reasoning or thinking token appeared under
  `reasoning_mode=disabled`;
* the run's artifacts still match their SHA-256 catalog.

The thresholds are arguments so the rule can be fixed before the run is looked
at, and they are recorded in the printed decision alongside what was measured.

    python tools/verify_route_from_run.py \\
        --run pilot_outputs/real_gpt5-4-mini-verify-001 \\
        --model GPT-5.4-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.artifacts import atomic_write_text, verify_run_artifacts  # noqa: E402

REASONING_KEYS = ("thinking", "reasoning", "thinking_tokens", "reasoning_tokens")


def _reasoning_tokens(envelope: Mapping[str, Any]) -> int:
    """Largest reasoning/thinking count the provider reported for one call."""

    usage = envelope.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    seen = [0]
    details = usage.get("output_token_details")
    if isinstance(details, Mapping):
        seen += [
            int(details[key])
            for key in REASONING_KEYS
            if isinstance(details.get(key), (int, float))
        ]
    seen += [
        int(usage[key])
        for key in REASONING_KEYS
        if isinstance(usage.get(key), (int, float))
    ]
    # The adapter also stores its own reading, which is what the runner acts on.
    metadata = envelope.get("metadata")
    if isinstance(metadata, Mapping):
        observed = metadata.get("ehq_observed_reasoning_tokens")
        if isinstance(observed, (int, float)):
            seen.append(int(observed))
    return max(seen)


def assess(
    run_dir: Path,
    model_name: str,
    *,
    min_valid: float,
    min_confidence: float,
) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("model") == model_name]
    if not rows:
        raise SystemExit(f"{run_dir} holds no records for {model_name}")

    valid12 = [row for row in rows if row.get("valid_for_ehq12")]
    valid3 = [row for row in valid12 if row.get("valid_for_ehq3")]
    valid_rate = len(valid12) / len(rows)
    confidence_rate = (len(valid3) / len(valid12)) if valid12 else 0.0

    resolved: Dict[str, int] = {}
    requested: Dict[str, int] = {}
    max_reasoning = 0
    n_calls = 0
    for row in rows:
        for key in ("answer_response", "confidence_response"):
            envelope = row.get(key)
            if not isinstance(envelope, Mapping) or not envelope.get("ok"):
                continue
            n_calls += 1
            name = envelope.get("resolved_model")
            if name:
                resolved[str(name)] = resolved.get(str(name), 0) + 1
            asked = envelope.get("requested_model")
            if asked:
                requested[str(asked)] = requested.get(str(asked), 0) + 1
            max_reasoning = max(max_reasoning, _reasoning_tokens(envelope))

    failures: List[str] = []
    if len(requested) != 1:
        failures.append(f"more than one route was requested: {requested}")
    if len(resolved) != 1 or set(resolved) != set(requested):
        failures.append(f"route identity is not exact: requested {requested}, resolved {resolved}")
    if valid_rate < min_valid:
        failures.append(f"valid EHQ1/EHQ2 coverage {valid_rate:.1%} < {min_valid:.1%}")
    if confidence_rate < min_confidence:
        failures.append(
            f"confidence coverage {confidence_rate:.1%} < {min_confidence:.1%}"
        )
    if max_reasoning > 0:
        failures.append(
            f"provider reported {max_reasoning} reasoning/thinking token(s) "
            "under reasoning_mode=disabled"
        )

    # A run with no catalog is unverifiable rather than valid, so the absence is
    # a reason to refuse and not an exception to escape through.
    try:
        artifacts_valid = bool(verify_run_artifacts(run_dir)["valid"])
        catalog_note = "valid" if artifacts_valid else "mismatch"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        artifacts_valid, catalog_note = False, f"unreadable ({exc.__class__.__name__})"
    if not artifacts_valid:
        failures.append(f"run artifacts do not verify against their catalog: {catalog_note}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "n_records": len(rows),
        "n_calls": n_calls,
        "n_valid_ehq12": len(valid12),
        "n_valid_ehq3": len(valid3),
        "valid_rate": valid_rate,
        "confidence_rate": confidence_rate,
        "requested_routes": requested,
        "resolved_routes": resolved,
        "max_reasoning_tokens": max_reasoning,
        "artifacts_valid": artifacts_valid,
        "artifact_catalog": catalog_note,
        "thresholds": {"min_valid": min_valid, "min_confidence": min_confidence},
        "run_created_at_utc": manifest.get("created_at_utc"),
        "run_fingerprint": (manifest.get("run") or {}).get("fingerprint"),
        "failures": failures,
        "verified": not failures,
    }


def apply_to_registry(registry_path: Path, report: Mapping[str, Any]) -> None:
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in raw["models"]:
        if entry.get("name") != report["model"]:
            continue
        entry["operational_status"] = "verified"
        # A successful exact-route verification is the evidence needed to make
        # identity mismatch a hard error in subsequent publication runs.
        entry["model_identity_policy"] = "strict"
        entry["endpoint_verified_at"] = report["run_created_at_utc"]
        entry["endpoint_verified_resolved_model"] = next(
            iter(report["resolved_routes"])
        )
        entry["non_reasoning_verified_at"] = report["run_created_at_utc"]
        entry["non_reasoning_observed_thinking_tokens"] = report[
            "max_reasoning_tokens"
        ]
        break
    else:
        raise SystemExit(f"{report['model']} is not in {registry_path}")
    atomic_write_text(
        registry_path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", default="config/models.json", type=Path)
    parser.add_argument("--min-valid", type=float, default=0.99)
    parser.add_argument("--min-confidence", type=float, default=0.99)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the verification fields; without it, only report",
    )
    args = parser.parse_args()

    report = assess(
        args.run,
        args.model,
        min_valid=args.min_valid,
        min_confidence=args.min_confidence,
    )

    print(f"run    : {report['run_dir']}")
    print(f"model  : {report['model']}")
    print(f"records: {report['n_records']}  calls: {report['n_calls']}")
    print(
        f"coverage: EHQ1/EHQ2 {report['valid_rate']:.1%} "
        f"(threshold {report['thresholds']['min_valid']:.0%}), "
        f"confidence {report['confidence_rate']:.1%} "
        f"(threshold {report['thresholds']['min_confidence']:.0%})"
    )
    print(
        f"route  : requested {report['requested_routes']} "
        f"-> resolved {report['resolved_routes']}"
    )
    print(f"reasoning tokens (maximum): {report['max_reasoning_tokens']}")
    print(
        "artifact catalog          : "
        f"{'valid' if report['artifacts_valid'] else 'INVALID'}"
    )
    print()
    if report["verified"]:
        print("DECISION: verified")
    else:
        print("DECISION: not verified")
        for reason in report["failures"]:
            print(f"  - {reason}")

    if args.apply:
        if not report["verified"]:
            raise SystemExit("Refusing to write verification fields for a failed run")
        apply_to_registry(args.registry, report)
        print(f"\nUpdated {args.registry}.")
    elif report["verified"]:
        print("\nUse --apply to write this evidence to the model registry.")
    return 0 if report["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
