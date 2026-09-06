"""Reclassify a verified run from retained raw answers without provider calls."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehq.analysis import build_analysis_report  # noqa: E402
from ehq.artifacts import (  # noqa: E402
    verify_run_artifacts,
    write_artifact_catalog,
    write_json,
)
from ehq.config import (  # noqa: E402
    load_experiment_config,
)
from ehq.constants import FRAMEWORK_VERSION  # noqa: E402
from ehq.datasets import load_dataset  # noqa: E402
from ehq.evaluation.classifier import classify_response  # noqa: E402
from ehq.evaluation.scoring import (  # noqa: E402
    compute_ehq_scores,
    compute_grouped_scores,
)
from ehq.hashing import sha256_file, sha256_tree  # noqa: E402
from ehq.provenance import (  # noqa: E402
    build_manifest,
    experiment_fingerprint,
)
from ehq.reporting import write_standard_reports  # noqa: E402
from ehq.types import ModelSpec  # noqa: E402


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise ValueError("Model name contains no filesystem-safe characters")
    return cleaned


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selected_models_from_manifest(
    source_manifest: Dict[str, Any], selected_names: list[str]
) -> list[ModelSpec]:
    """Preserve the exact model-registry snapshot used by the parent run."""

    registry = {
        str(value["name"]): ModelSpec.from_dict(dict(value))
        for value in source_manifest.get("models") or []
    }
    missing = [name for name in selected_names if name not in registry]
    if missing:
        raise RuntimeError(
            "Selected model(s) missing from source manifest: " + ", ".join(missing)
        )
    return [registry[name] for name in selected_names]


def reclassify(source_run: Path, dataset_path: Path, output_dir: Path) -> Dict[str, Any]:
    source_run = source_run.resolve()
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()
    verification = verify_run_artifacts(source_run)
    if not verification["valid"]:
        raise RuntimeError("Source run artifact verification failed")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_manifest = json.loads(
        (source_run / "manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(dataset_path) != source_manifest["dataset"]["sha256"]:
        raise RuntimeError("Dataset hash does not match the source run manifest")
    config_path = Path(source_manifest["config"]["path"])
    if sha256_file(config_path) != source_manifest["config"]["sha256"]:
        raise RuntimeError("Experiment config changed since the source run")

    config = load_experiment_config(config_path)
    items = load_dataset(dataset_path)
    item_by_id = {
        str(item.get("question_id") or item.get("id")): item for item in items
    }
    rows = _load_jsonl(source_run / "records.jsonl")
    reclassified = []
    changed = 0
    for original in rows:
        row = copy.deepcopy(original)
        answer = (row.get("answer_response") or {}).get("text")
        previous = row.get("classification")
        if row.get("valid_for_ehq12") and answer and isinstance(previous, dict):
            question_id = str(row["question_id"])
            item = item_by_id.get(question_id)
            if item is None:
                raise RuntimeError(f"Question missing from dataset: {question_id}")
            adjudicated = previous.get("adjudicated_label")
            current = classify_response(
                answer,
                item,
                adjudicated_label=adjudicated,
            ).to_dict()
            row["classification_previous"] = previous
            row["classification"] = current
            changed += int(previous.get("label") != current.get("label"))
        reclassified.append(row)

    selected_names = list(dict.fromkeys(str(row["model"]) for row in rows))
    selected_models = _selected_models_from_manifest(source_manifest, selected_names)
    weights = {
        "ehq1": config.weights.ehq1,
        "ehq2": config.weights.ehq2,
        "ehq3": config.weights.ehq3,
    }
    source_routes = (source_manifest.get("run") or {}).get("provider_endpoints") or {}
    results: Dict[str, Dict[str, Any]] = {}
    for model in selected_names:
        model_rows = [row for row in reclassified if row["model"] == model]
        scores = compute_ehq_scores(
            model_rows,
            weights=weights,
            n_bins=config.confidence.n_bins,
        )
        result = {
            "model": model,
            "provider_endpoint": source_routes.get(model),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
            "category_scores": compute_grouped_scores(
                model_rows,
                group_field="category",
                weights=weights,
                n_bins=config.confidence.n_bins,
            ),
            "subcategory_scores": compute_grouped_scores(
                model_rows,
                group_field="subcategory",
                weights=weights,
                n_bins=config.confidence.n_bins,
            ),
            "n_retryable_records": scores["n_retryable_records"],
            "n_terminal_missing_confidence": scores[
                "n_terminal_missing_confidence"
            ],
            "n_excluded_k1_before_request": 0,
            "records": model_rows,
        }
        results[model] = result

    classifier_hash = sha256_file(ROOT / "src" / "ehq" / "evaluation" / "classifier.py")
    parent_fingerprint = (source_manifest.get("run") or {}).get("fingerprint")
    derived_fingerprint = experiment_fingerprint(
        {
            "mode": "derived-reclassification",
            "parent_fingerprint": parent_fingerprint,
            "framework_version": FRAMEWORK_VERSION,
            "classifier_sha256": classifier_hash,
            "dataset_sha256": sha256_file(dataset_path),
        }
    )
    parent_run = source_manifest.get("run") or {}
    run_values = {
        **parent_run,
        "fingerprint": derived_fingerprint,
        "framework_source_sha256": sha256_tree(ROOT / "src" / "ehq"),
        "run_id": output_dir.name,
        "mode": "derived-reclassification",
        "parent_run_dir": str(source_run),
        "parent_fingerprint": parent_fingerprint,
        "parent_manifest_sha256": sha256_file(source_run / "manifest.json"),
        "parent_artifact_catalog_sha256": sha256_file(
            source_run / "artifact_catalog.json"
        ),
        "source_artifacts_verified": True,
        "classifier_sha256": classifier_hash,
        "provider_calls_made": 0,
        "classification_labels_changed": changed,
    }
    manifest = build_manifest(
        project_root=ROOT,
        experiment_name=str(source_manifest.get("experiment_name")),
        config_path=config.source_path,
        dataset_path=dataset_path,
        models_path=config.models_path,
        models=selected_models,
        configuration=copy.deepcopy(source_manifest["config"].get("snapshot")),
        run=run_values,
    )
    # The current registry file may have changed operational status since the
    # provider calls. A derived run must retain the parent registry hash rather
    # than silently stamping today's registry over historical model metadata.
    manifest["models_config"] = copy.deepcopy(source_manifest["models_config"])

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", manifest)
    for model, result in results.items():
        write_json(output_dir / "models" / f"{_safe_name(model)}.json", result)
    summary = write_standard_reports(output_dir, results)
    analysis = build_analysis_report(results, selected_models, seed=config.seed)
    write_json(output_dir / "analysis.json", analysis)
    write_json(
        output_dir / "derivation.json",
        {
            "source_run": str(source_run),
            "source_artifacts_verified": True,
            "provider_calls_made": 0,
            "records_reclassified": len(reclassified),
            "classification_labels_changed": changed,
            "classifier_sha256": classifier_hash,
            "model_registry_snapshot_source": "parent_manifest",
        },
    )
    catalog = write_artifact_catalog(output_dir)
    output_verification = verify_run_artifacts(output_dir)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "records_reclassified": len(reclassified),
        "classification_labels_changed": changed,
        "artifact_count": len(catalog["artifacts"]),
        "artifacts_verified": output_verification["valid"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = reclassify(args.run_dir, args.dataset, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
