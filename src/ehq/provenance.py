"""Experiment manifest generation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .constants import EHQ3_PROTOCOL, FRAMEWORK_VERSION, PROTOCOL_VERSION
from .hashing import sha256_file, sha256_json
from .types import ModelSpec


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(project_root: Path) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> Dict[str, Optional[str]]:
    packages = ("requests", "truststore", "numpy", "scipy", "openpyxl", "matplotlib")
    # The running source tree is authoritative for the framework itself. An
    # older dist-info directory can otherwise shadow a newer editable install
    # and stamp a manifest with a false EHQ version even though the imported
    # code and framework_version field are current.
    values: Dict[str, Optional[str]] = {"ehq": FRAMEWORK_VERSION}
    for package in packages:
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = None
    return values


def experiment_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash all scientifically relevant inputs used by checkpoint resume."""

    return sha256_json(dict(payload))


def build_manifest(
    *,
    project_root: Path,
    experiment_name: str,
    config_path: Path,
    dataset_path: Path,
    models_path: Path,
    models: Iterable[ModelSpec],
    configuration: Optional[Mapping[str, Any]] = None,
    run: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework_version": FRAMEWORK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ehq3_protocol": EHQ3_PROTOCOL,
        "git_commit": _git_commit(project_root),
        "git_dirty": _git_dirty(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "snapshot": dict(configuration) if configuration is not None else None,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "models_config": {
            "path": str(models_path),
            "sha256": sha256_file(models_path),
        },
        "models": [
            {
                "name": model.name,
                "provider": model.provider,
                "provider_model": model.provider_model,
                "base_url": model.base_url,
                "api_key_env": model.api_key_env,
                "requires_api_key": model.requires_api_key,
                "model_identity_policy": model.model_identity_policy,
                "model_provider": model.model_provider,
                "pair": model.pair,
                "generation": model.generation,
                "reported_model_name": model.reported_model_name,
                "cutoff": model.cutoff,
                "pcq_cutoff": model.pcq_cutoff,
                "training_cutoff": model.training_cutoff,
                "reliable_knowledge_cutoff": model.reliable_knowledge_cutoff,
                "reported_knowledge_cutoff": model.reported_knowledge_cutoff,
                "cutoff_precision": model.cutoff_precision,
                "cutoff_definition": model.cutoff_definition,
                "cutoff_source_url": model.cutoff_source_url,
                "cutoff_source_type": model.cutoff_source_type,
                "cutoff_evidence_status": model.cutoff_evidence_status,
                "cutoff_verified_at": model.cutoff_verified_at,
                "pcq_eligible": model.pcq_eligible,
                "endpoint_verified_at": model.endpoint_verified_at,
                "endpoint_verified_resolved_model": model.endpoint_verified_resolved_model,
                "operational_status": model.operational_status,
                "reasoning_mode": model.reasoning_mode,
                "thinking_mode": model.thinking_mode,
                "non_reasoning_verified_at": model.non_reasoning_verified_at,
                "non_reasoning_observed_thinking_tokens": (
                    model.non_reasoning_observed_thinking_tokens
                ),
            }
            for model in models
        ],
        "secrets_present": {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "HF_TOKEN": bool(os.getenv("HF_TOKEN")),
            "OPENAI_COMPATIBLE_API_KEY": bool(
                os.getenv("OPENAI_COMPATIBLE_API_KEY")
            ),
        },
    }
    if run is not None:
        manifest["run"] = dict(run)
    return manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
