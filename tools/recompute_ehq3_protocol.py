"""Regenerate a completed run in place under the current EHQ3 protocol.

The retained evaluation records are the input. No provider is contacted, no
network request is issued, and no cache entry is read or written: every
artifact is re-derived from ``records.jsonl`` / ``models/*.json`` that the
original run already wrote.

The run directory, run id, experiment name, and parent fingerprint are left
unchanged, because the underlying model responses are unchanged. Only the EHQ3
post-processing protocol has been updated, so this is a regeneration of the
official outputs rather than a new experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.analysis import (  # noqa: E402
    build_analysis_report,
    load_capability_counts,
    load_capability_scores,
)
from ehq.artifacts import (  # noqa: E402
    verify_run_artifacts,
    write_artifact_catalog,
    write_json,
)
from ehq.config import load_models, model_registry_release_issues  # noqa: E402
from ehq.constants import (  # noqa: E402
    EHQ3_PROTOCOL,
    FRAMEWORK_VERSION,
    PROTOCOL_VERSION,
)
from ehq.datasets import load_dataset, validate_dataset  # noqa: E402
from ehq.evaluation.scoring import (  # noqa: E402
    compute_ehq_scores,
    compute_grouped_scores,
)
from ehq.hashing import sha256_file, sha256_json, sha256_tree  # noqa: E402
from ehq.reporting import (  # noqa: E402
    export_excel,
    export_figures,
    write_standard_reports,
)
from ehq.types import ModelSpec  # noqa: E402

COMPARED_FIELDS = ("EHQ1", "EHQ2", "EHQ3", "EHQ")


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


def _changed(before: Any, after: Any) -> bool:
    if before is None or after is None:
        return before is not after
    return float(before) != float(after)


def recompute(
    run_dir: Path,
    *,
    excel: bool,
    figures: bool,
    excluded_models: Dict[str, str] | None = None,
    release_dataset: Path | None = None,
    models_path: Path | None = None,
    capability_scores_override: Path | None = None,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Run manifest not found: {manifest_path}")
    if not (run_dir / "summary.json").is_file():
        raise SystemExit(f"Run is not complete (summary.json missing): {run_dir}")

    verification = verify_run_artifacts(run_dir)
    if not verification["valid"]:
        raise SystemExit(
            "Refusing to regenerate: the existing run artifacts do not match "
            f"their catalog: {run_dir}"
        )

    manifest = _load_json(manifest_path)
    snapshot = (manifest.get("config") or {}).get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise SystemExit(
            "Run manifest has no embedded configuration snapshot; the scoring "
            "weights cannot be recovered without contacting the config file"
        )
    weights = {
        "ehq1": float(snapshot["weights"]["ehq1"]),
        "ehq2": float(snapshot["weights"]["ehq2"]),
        "ehq3": float(snapshot["weights"]["ehq3"]),
    }
    n_bins = int(snapshot["confidence"]["n_bins"])
    seed = int(snapshot["seed"])

    manifest_models = [
        ModelSpec.from_dict(dict(row))
        for row in manifest.get("models") or []
        if isinstance(row, Mapping) and row.get("name")
    ]
    release_dataset = release_dataset.resolve() if release_dataset else None
    models_path = models_path.resolve() if models_path else None
    if (release_dataset is None) != (models_path is None):
        raise SystemExit(
            "--release-dataset and --models-config must be supplied together"
        )
    if release_dataset is not None and models_path is not None:
        dataset_rows = load_dataset(release_dataset)
        requirements = (snapshot.get("dataset_requirements") or {})
        dataset_validation = validate_dataset(
            dataset_rows,
            expected_total=requirements.get("expected_total"),
            expected_per_category=requirements.get("expected_per_category"),
            require_source_evidence=bool(requirements.get("require_source_evidence")),
            require_pcq_temporal_novelty=bool(
                requirements.get("require_pcq_temporal_novelty")
            ),
            allow_pending_human_review=False,
        )
        if not dataset_validation.valid:
            raise SystemExit(
                "Release dataset does not pass strict validation: "
                + ", ".join(issue.code for issue in dataset_validation.issues[:10])
            )
        current_registry = load_models(models_path)
        selected_names = {model.name for model in manifest_models}
        release_models = [
            model for model in current_registry if model.name in selected_names
        ]
        if len(release_models) != len(selected_names):
            raise SystemExit("Release registry does not contain every run model")
        registry_issues = model_registry_release_issues(models_path, release_models)
        if registry_issues:
            raise SystemExit(
                "Release model registry has open issues: " + "; ".join(registry_issues)
            )
        manifest_models = release_models
    # records.jsonl is rewritten by iterating the results mapping, so the
    # regenerated file only stays byte-identical if models are processed in the
    # order the original run wrote them.
    records_path = run_dir / "records.jsonl"
    if not records_path.is_file():
        raise SystemExit(f"Retained records not found: {records_path}")
    written_order: List[str] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = str(json.loads(line).get("model"))
        if name not in written_order:
            written_order.append(name)
    by_name = {model.name: model for model in manifest_models}
    ordered_models = [by_name[name] for name in written_order if name in by_name]
    ordered_models += [
        model for model in manifest_models if model.name not in written_order
    ]

    results: Dict[str, Dict[str, Any]] = {}
    models: List[ModelSpec] = []
    comparisons: List[Dict[str, Any]] = []
    for model in ordered_models:
        model_path = run_dir / "models" / f"{_safe_name(model.name)}.json"
        if not model_path.is_file():
            continue
        result = _load_json(model_path)
        rows = result.get("records")
        if not isinstance(rows, list):
            raise SystemExit(f"Retained records missing for model: {model.name}")

        previous_scores = copy.deepcopy(result.get("scores") or {})
        # The records themselves are inputs and are never rewritten here.
        result["scores"] = compute_ehq_scores(rows, weights=weights, n_bins=n_bins)
        result["category_scores"] = compute_grouped_scores(
            rows, group_field="category", weights=weights, n_bins=n_bins
        )
        result["subcategory_scores"] = compute_grouped_scores(
            rows, group_field="subcategory", weights=weights, n_bins=n_bins
        )
        results[model.name] = result
        models.append(model)
        comparisons.append(
            {
                "model": model.name,
                **{
                    field: {
                        "before": previous_scores.get(field),
                        "after": result["scores"].get(field),
                        "changed": _changed(
                            previous_scores.get(field), result["scores"].get(field)
                        ),
                    }
                    for field in COMPARED_FIELDS
                },
            }
        )

    if not results:
        raise SystemExit(f"No per-model result files found under: {run_dir}")

    records_before = records_path.read_bytes()

    for model_name, result in results.items():
        write_json(run_dir / "models" / f"{_safe_name(model_name)}.json", result)
    summary = write_standard_reports(run_dir, results)
    # RQ1 exists only when the parent run was given capability scores. Reload
    # them from the path the manifest recorded, or the regenerated analysis
    # would silently drop a research question the run actually answered.
    capability_scores = None
    capability_counts = None
    parent_run = manifest.get("run") or {}
    capability_path = capability_scores_override or parent_run.get(
        "capability_scores_path"
    )
    if capability_path:
        path = Path(str(capability_path)).resolve()
        if not path.is_file():
            raise SystemExit(
                "The parent run used capability scores but the file is gone; "
                f"restore it before regenerating: {path}"
            )
        recorded = parent_run.get("capability_scores_sha256")
        if recorded and sha256_file(path) != recorded:
            raise SystemExit(
                "Capability scores changed since the parent run: "
                f"{path}. Regenerating would silently redefine RQ1."
            )
        capability_scores = load_capability_scores(path)
        capability_counts = load_capability_counts(path)
    analysis = build_analysis_report(
        results,
        models,
        seed=seed,
        capability_scores=capability_scores,
        capability_counts=capability_counts,
        excluded_models=excluded_models,
    )
    write_json(run_dir / "analysis.json", analysis)

    records_after = records_path.read_bytes()
    if records_before != records_after:
        raise SystemExit(
            "Regeneration changed records.jsonl; only derived artifacts may "
            "change under an EHQ3 post-processing update"
        )

    if excel:
        export_excel(run_dir / "EHQ_results.xlsx", results)
    if figures:
        export_figures(run_dir / "figures", results)

    run_values = dict(manifest.get("run") or {})
    # Repeated regeneration must not erase what the original provider run was
    # produced by, so the first recorded values win.
    earlier = run_values.get("ehq3_protocol_recomputation")
    earlier = earlier if isinstance(earlier, Mapping) else {}
    provider_run_framework_version = earlier.get(
        "provider_run_framework_version", manifest.get("framework_version")
    )
    provider_run_source_sha256 = earlier.get(
        "provider_run_framework_source_sha256",
        run_values.get("framework_source_sha256"),
    )

    manifest["framework_version"] = FRAMEWORK_VERSION
    manifest["ehq3_protocol"] = EHQ3_PROTOCOL
    # run_id, experiment_name, and fingerprint are intentionally preserved: the
    # provider responses behind this run did not change.
    run_values["framework_source_sha256"] = sha256_tree(ROOT / "src" / "ehq")
    run_values["ehq3_protocol"] = EHQ3_PROTOCOL
    run_values["ehq3_protocol_recomputation"] = {
        "recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "recompute_ehq3_protocol.py",
        "provider_run_framework_version": provider_run_framework_version,
        "provider_run_framework_source_sha256": provider_run_source_sha256,
        "framework_version": FRAMEWORK_VERSION,
        "provider_calls_made": 0,
        "cache_reads": 0,
        "records_rewritten": False,
        "source": "retained_evaluation_records",
        "capability_scores_reloaded": bool(capability_scores),
    }
    run_values["analysis_exclusions"] = [
        {"model": name, "reason": reason}
        for name, reason in sorted((excluded_models or {}).items())
    ]
    if release_dataset is not None and models_path is not None:
        parent_fingerprint = run_values.get("fingerprint")
        parent_dataset = dict(manifest.get("dataset") or {})
        release_dataset_sha = sha256_file(release_dataset)
        release_registry_sha = sha256_file(models_path)
        release_fingerprint = sha256_json(
            {
                "parent_run_fingerprint": parent_fingerprint,
                "release_dataset_sha256": release_dataset_sha,
                "release_model_registry_sha256": release_registry_sha,
                "analysis_exclusions": run_values["analysis_exclusions"],
                "framework_version": FRAMEWORK_VERSION,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        run_values.update(
            {
                "dataset_policy": "RELEASE_GATE_PASSED",
                "allow_pending_human_review": False,
                "publication_blockers": [],
                "model_registry_release_issues": [],
                "release_fingerprint": release_fingerprint,
                "release_revalidation": {
                    "revalidated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider_calls_made": 0,
                    "records_rewritten": False,
                    "parent_run_fingerprint": parent_fingerprint,
                    "parent_dataset": parent_dataset,
                    "release_dataset_sha256": release_dataset_sha,
                    "release_model_registry_sha256": release_registry_sha,
                    "strict_dataset_validation_passed": True,
                    "model_registry_release_gate_passed": True,
                },
            }
        )
        manifest["dataset"] = {
            "path": str(release_dataset),
            "sha256": release_dataset_sha,
        }
        manifest["models"] = [model.__dict__ for model in manifest_models]
        manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["run"] = run_values
    write_json(manifest_path, manifest)

    catalog = write_artifact_catalog(run_dir)
    output_verification = verify_run_artifacts(run_dir)

    ehq1_changed = any(row["EHQ1"]["changed"] for row in comparisons)
    ehq2_changed = any(row["EHQ2"]["changed"] for row in comparisons)
    ehq3_changed = any(row["EHQ3"]["changed"] for row in comparisons)
    return {
        "run_dir": str(run_dir),
        "run_id": (manifest.get("run") or {}).get("run_id"),
        "experiment_name": manifest.get("experiment_name"),
        "fingerprint": (manifest.get("run") or {}).get("fingerprint"),
        "ehq3_protocol": EHQ3_PROTOCOL,
        "models_processed": len(results),
        "api_calls": 0,
        "cache_hits": 0,
        "ehq1_changed": ehq1_changed,
        "ehq2_changed": ehq2_changed,
        "ehq3_changed": ehq3_changed,
        "summary_regenerated": True,
        "excel_regenerated": bool(excel),
        "figures_regenerated": bool(figures),
        "comparisons": comparisons,
        "summary": summary,
        "artifact_count": len(catalog["artifacts"]),
        "artifacts_verified": output_verification["valid"],
    }


def _print_validation(outcomes: List[Dict[str, Any]], *, publication: bool) -> None:
    models_processed = sum(row["models_processed"] for row in outcomes)
    ehq1_changed = any(row["ehq1_changed"] for row in outcomes)
    ehq2_changed = any(row["ehq2_changed"] for row in outcomes)
    ehq3_changed = any(row["ehq3_changed"] for row in outcomes)
    flag = {True: "YES", False: "NO"}
    print("")
    print(f"Models processed:                 {models_processed}")
    print("API calls:                        0")
    print("Cache hits:                       0")
    print(f"EHQ1 changed:                     {flag[ehq1_changed]}")
    print(f"EHQ2 changed:                     {flag[ehq2_changed]}")
    print(f"EHQ3 changed:                     {flag[ehq3_changed]}")
    print(
        "Summary regenerated:              "
        f"{flag[all(row['summary_regenerated'] for row in outcomes)]}"
    )
    print(f"Publication package regenerated:  {flag[publication]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        action="append",
        help="Completed run directory to regenerate in place (repeatable)",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Also regenerate EHQ_results.xlsx",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Also regenerate the run figures",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help=(
            "Withhold a model from the confirmatory analyses while keeping its "
            "records and reporting its scores and the reason. Repeatable. "
            "Applies to every --run-dir in this pass."
        ),
    )
    parser.add_argument(
        "--release-dataset",
        type=Path,
        help="Strictly validated release dataset used to close the run gate",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        help="Release-verified model registry used to close the run gate",
    )
    parser.add_argument(
        "--capability-scores-override",
        type=Path,
        help="Local hash-identical copy of the parent capability-score file",
    )
    parser.add_argument(
        "--publication-package-regenerated",
        action="store_true",
        help=(
            "Record that the downstream publication package was rebuilt from "
            "these runs in the same regeneration pass"
        ),
    )
    args = parser.parse_args()

    exclusions = {}
    for value in args.exclude_model:
        name, separator, reason = str(value).partition("=")
        name, reason = name.strip(), reason.strip()
        if not name or not separator or not reason:
            raise SystemExit(
                "--exclude-model wants NAME=reason; got " f"{value!r}"
            )
        exclusions[name] = reason

    outcomes = [
        recompute(
            run_dir,
            excel=args.excel,
            figures=args.figures,
            excluded_models=exclusions or None,
            release_dataset=args.release_dataset,
            models_path=args.models_config,
            capability_scores_override=args.capability_scores_override,
        )
        for run_dir in args.run_dir
    ]
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))
    _print_validation(outcomes, publication=args.publication_package_regenerated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
