"""Aggregate scientifically compatible, artifact-verified EHQ runs locally."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.analysis import build_analysis_report  # noqa: E402
from ehq.artifacts import (  # noqa: E402
    verify_run_artifacts,
    write_artifact_catalog,
    write_json,
)
from ehq.constants import FRAMEWORK_VERSION  # noqa: E402
from ehq.evaluation.scoring import (  # noqa: E402
    compute_ehq_scores,
    compute_grouped_scores,
)
from ehq.hashing import sha256_file, sha256_json, sha256_tree  # noqa: E402
from ehq.provenance import experiment_fingerprint  # noqa: E402
from ehq.reporting import write_standard_reports  # noqa: E402
from ehq.types import ModelSpec  # noqa: E402


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise ValueError("Model name contains no filesystem-safe characters")
    return cleaned


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def compatibility_signature(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return every field that must match before record-level aggregation."""

    run = manifest.get("run") or {}
    config = manifest.get("config") or {}
    selection = run.get("selection") or {}
    snapshot = config.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Source manifest is missing the config snapshot")
    required = {
        "framework_version": manifest.get("framework_version"),
        "protocol_version": manifest.get("protocol_version"),
        "framework_source_sha256": run.get("framework_source_sha256"),
        "dataset_sha256": (manifest.get("dataset") or {}).get("sha256"),
        "config_sha256": config.get("sha256"),
        "config_snapshot_sha256": sha256_json(dict(snapshot)),
        "selection_question_ids_sha256": selection.get("question_ids_sha256"),
        "n_selected": selection.get("n_selected"),
        "answer_prompt_version": run.get("answer_prompt_version"),
        "confidence_prompt_version": run.get("confidence_prompt_version"),
        "dataset_policy": run.get("dataset_policy"),
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(
            "Source manifest lacks compatibility field(s): " + ", ".join(missing)
        )
    return required


def _assert_compatible(
    expected: Mapping[str, Any], actual: Mapping[str, Any], source: Path
) -> None:
    differences = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if expected.get(key) != actual.get(key)
    }
    if differences:
        raise ValueError(
            f"Scientifically incompatible source run {source}: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )


def aggregate_runs(
    source_runs: Sequence[Path], output_dir: Path
) -> Dict[str, Any]:
    if len(source_runs) < 2:
        raise ValueError("At least two source runs are required")
    sources = [path.resolve() for path in source_runs]
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    manifests = []
    source_metadata = []
    expected_signature: Dict[str, Any] | None = None
    for source in sources:
        verification = verify_run_artifacts(source)
        if not verification["valid"]:
            raise RuntimeError(f"Source artifact verification failed: {source}")
        manifest = _load_json(source / "manifest.json")
        signature = compatibility_signature(manifest)
        if expected_signature is None:
            expected_signature = signature
        else:
            _assert_compatible(expected_signature, signature, source)
        if manifest.get("framework_version") != FRAMEWORK_VERSION:
            raise ValueError(
                f"Source {source} uses framework {manifest.get('framework_version')}; "
                f"re-derive it under {FRAMEWORK_VERSION} before aggregation"
            )
        manifests.append(manifest)
        source_metadata.append(
            {
                "run_dir": str(source),
                "manifest_sha256": sha256_file(source / "manifest.json"),
                "artifact_catalog_sha256": sha256_file(
                    source / "artifact_catalog.json"
                ),
                "artifacts_verified": True,
            }
        )
    assert expected_signature is not None

    first_manifest = manifests[0]
    config_snapshot = copy.deepcopy(first_manifest["config"]["snapshot"])
    selection = copy.deepcopy(first_manifest["run"]["selection"])
    expected_question_ids = [str(value) for value in selection["question_ids"]]
    expected_question_set = set(expected_question_ids)
    if len(expected_question_set) != len(expected_question_ids):
        raise ValueError("Selection manifest contains duplicate question IDs")

    model_specs: list[ModelSpec] = []
    rows_by_model: Dict[str, list[Dict[str, Any]]] = {}
    provider_endpoints: Dict[str, Any] = {}
    for source, manifest in zip(sources, manifests):
        manifest_models = {
            str(value["name"]): ModelSpec.from_dict(dict(value))
            for value in manifest.get("models") or []
        }
        source_rows = _load_jsonl(source / "records.jsonl")
        source_names = {str(row.get("model")) for row in source_rows}
        if source_names != set(manifest_models):
            raise ValueError(
                f"Model set differs between manifest and records: {source}"
            )
        routes = (manifest.get("run") or {}).get("provider_endpoints") or {}
        for name in sorted(source_names):
            if name in rows_by_model:
                raise ValueError(f"Model appears in multiple source runs: {name}")
            model_rows = [row for row in source_rows if str(row.get("model")) == name]
            counts = Counter(str(row.get("question_id")) for row in model_rows)
            if set(counts) != expected_question_set or any(
                count != 1 for count in counts.values()
            ):
                raise ValueError(
                    f"Model {name} does not contain exactly one record per selected item"
                )
            if any(
                not row.get("valid_for_ehq12") or not row.get("valid_for_ehq3")
                for row in model_rows
            ):
                raise ValueError(
                    f"Model {name} has incomplete EHQ1–3 coverage; aggregation refused"
                )
            copied = []
            for row in model_rows:
                value = copy.deepcopy(row)
                value["aggregate_source_run"] = str(source)
                copied.append(value)
            rows_by_model[name] = copied
            model_specs.append(manifest_models[name])
            provider_endpoints[name] = routes.get(name)

    weights = {
        "ehq1": float(config_snapshot["weights"]["ehq1"]),
        "ehq2": float(config_snapshot["weights"]["ehq2"]),
        "ehq3": float(config_snapshot["weights"]["ehq3"]),
    }
    n_bins = int(config_snapshot["confidence"]["n_bins"])
    now = datetime.now(timezone.utc).isoformat()
    results: Dict[str, Dict[str, Any]] = {}
    for name, rows in rows_by_model.items():
        scores = compute_ehq_scores(rows, weights=weights, n_bins=n_bins)
        results[name] = {
            "model": name,
            "provider_endpoint": provider_endpoints.get(name),
            "completed_at_utc": now,
            "scores": scores,
            "category_scores": compute_grouped_scores(
                rows, group_field="category", weights=weights, n_bins=n_bins
            ),
            "subcategory_scores": compute_grouped_scores(
                rows, group_field="subcategory", weights=weights, n_bins=n_bins
            ),
            "n_retryable_records": 0,
            "n_terminal_missing_confidence": 0,
            "n_excluded_k1_before_request": 0,
            "records": rows,
        }

    tool_hash = sha256_file(Path(__file__))
    fingerprint = experiment_fingerprint(
        {
            "mode": "derived-aggregate",
            "framework_version": FRAMEWORK_VERSION,
            "compatibility_signature": expected_signature,
            "source_manifests": [
                value["manifest_sha256"] for value in source_metadata
            ],
            "source_catalogs": [
                value["artifact_catalog_sha256"] for value in source_metadata
            ],
            "aggregation_tool_sha256": tool_hash,
        }
    )
    blockers = sorted(
        {
            str(blocker)
            for manifest in manifests
            for blocker in ((manifest.get("run") or {}).get("publication_blockers") or [])
        }
    )
    run_manifest = {
        "experiment_name": "EHQ compatible-run aggregate",
        "created_at_utc": now,
        "framework_version": FRAMEWORK_VERSION,
        "protocol_version": expected_signature["protocol_version"],
        "dataset": {
            "sha256": expected_signature["dataset_sha256"],
        },
        "config": {
            "sha256": expected_signature["config_sha256"],
            "snapshot": config_snapshot,
        },
        "models": [model.__dict__ for model in model_specs],
        "sources": source_metadata,
        "run": {
            "mode": "derived-aggregate",
            "run_id": output_dir.name,
            "fingerprint": fingerprint,
            "framework_source_sha256": sha256_tree(ROOT / "src" / "ehq"),
            "aggregation_tool_sha256": tool_hash,
            "provider_calls_made": 0,
            "source_artifacts_verified": True,
            "dataset_policy": expected_signature["dataset_policy"],
            "publication_blockers": blockers,
            "answer_prompt_version": expected_signature["answer_prompt_version"],
            "confidence_prompt_version": expected_signature[
                "confidence_prompt_version"
            ],
            "selection": selection,
            "provider_endpoints": provider_endpoints,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", run_manifest)
    for name, result in results.items():
        write_json(output_dir / "models" / f"{_safe_name(name)}.json", result)
    summary = write_standard_reports(output_dir, results)
    analysis = build_analysis_report(
        results,
        model_specs,
        seed=int(config_snapshot["seed"]),
    )
    write_json(output_dir / "analysis.json", analysis)
    write_json(
        output_dir / "aggregation.json",
        {
            "mode": "derived-aggregate",
            "provider_calls_made": 0,
            "compatibility_signature": expected_signature,
            "sources": source_metadata,
            "n_sources": len(sources),
            "n_models": len(results),
            "n_records": sum(len(rows) for rows in rows_by_model.values()),
            "aggregation_tool_sha256": tool_hash,
        },
    )
    catalog = write_artifact_catalog(output_dir)
    verification = verify_run_artifacts(output_dir)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "analysis": analysis,
        "n_sources": len(sources),
        "n_models": len(results),
        "n_records": sum(len(rows) for rows in rows_by_model.values()),
        "provider_calls_made": 0,
        "artifact_count": len(catalog["artifacts"]),
        "artifacts_verified": verification["valid"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate_runs(args.run_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
