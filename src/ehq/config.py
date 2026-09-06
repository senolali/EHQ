"""Strict loading and validation for JSON experiment configuration."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlparse

from .constants import PROTOCOL_VERSION, REFERENCE_PANEL_SIZE
from .errors import ConfigurationError
from .types import ModelSpec


@dataclass(frozen=True)
class InferenceConfig:
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_retries: int
    request_delay_seconds: float
    system_prompt: str
    enable_search: bool
    enable_history: bool


@dataclass(frozen=True)
class ConfidenceConfig:
    scale_min: int
    scale_max: int
    n_bins: int
    max_tokens: int = 256


@dataclass(frozen=True)
class WeightConfig:
    ehq1: float
    ehq2: float
    ehq3: float


@dataclass(frozen=True)
class DatasetRequirements:
    expected_total: int | None
    expected_per_category: int | None
    require_source_evidence: bool
    require_pcq_temporal_novelty: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    protocol_version: str
    experiment_name: str
    dataset_path: Path
    output_dir: Path
    cache_dir: Path
    checkpoint_dir: Path
    seed: int
    max_workers: int
    inference: InferenceConfig
    confidence: ConfidenceConfig
    weights: WeightConfig
    dataset_requirements: DatasetRequirements
    models_path: Path
    source_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root must be an object: {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = path.resolve()
    raw = _read_json(path)
    project_root = path.parent.parent

    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ConfigurationError(
            f"Protocol mismatch: config={raw.get('protocol_version')!r}, "
            f"framework={PROTOCOL_VERSION!r}"
        )

    try:
        inference = InferenceConfig(**raw["inference"])
        confidence = ConfidenceConfig(**raw["confidence"])
        weights = WeightConfig(**raw["weights"])
        dataset_requirements = DatasetRequirements(
            **raw.get(
                "dataset_requirements",
                {
                    "expected_total": None,
                    "expected_per_category": None,
                    "require_source_evidence": False,
                    "require_pcq_temporal_novelty": False,
                },
            )
        )
        config = ExperimentConfig(
            schema_version=str(raw["schema_version"]),
            protocol_version=str(raw["protocol_version"]),
            experiment_name=str(raw["experiment_name"]),
            dataset_path=_resolve(project_root, raw["dataset_path"]),
            output_dir=_resolve(project_root, raw["output_dir"]),
            cache_dir=_resolve(project_root, raw["cache_dir"]),
            checkpoint_dir=_resolve(project_root, raw["checkpoint_dir"]),
            seed=int(raw["seed"]),
            max_workers=int(raw["max_workers"]),
            inference=inference,
            confidence=confidence,
            weights=weights,
            dataset_requirements=dataset_requirements,
            models_path=_resolve(project_root, raw["models_path"]),
            source_path=path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid experiment configuration: {exc}") from exc

    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    weights = (config.weights.ehq1, config.weights.ehq2, config.weights.ehq3)
    if any(weight < 0 or weight > 1 for weight in weights):
        raise ConfigurationError("EHQ weights must be within [0, 1]")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ConfigurationError(f"EHQ weights must sum to 1.0, got {sum(weights)}")
    if config.inference.temperature != 0:
        raise ConfigurationError("The normative EHQ protocol requires temperature=0")
    if config.inference.max_tokens <= 0:
        raise ConfigurationError("max_tokens must be positive")
    if config.confidence.scale_min != 0 or config.confidence.scale_max != 100:
        raise ConfigurationError("The EHQ protocol requires a 0-100 confidence scale")
    if config.confidence.n_bins < 2:
        raise ConfigurationError("Confidence calibration requires at least two bins")
    if config.confidence.max_tokens < 1:
        raise ConfigurationError("confidence.max_tokens must be positive")
    if config.max_workers < 1:
        raise ConfigurationError("max_workers must be positive")
    if not isinstance(
        config.dataset_requirements.require_source_evidence, bool
    ) or not isinstance(
        config.dataset_requirements.require_pcq_temporal_novelty, bool
    ):
        raise ConfigurationError("Dataset requirement flags must be boolean")
    if (
        config.dataset_requirements.require_pcq_temporal_novelty
        and not config.dataset_requirements.require_source_evidence
    ):
        raise ConfigurationError(
            "PCQ temporal novelty requires source evidence to be enabled"
        )
    if config.inference.enable_search or config.inference.enable_history:
        raise ConfigurationError(
            "The normative EHQ protocol requires search and history to be disabled"
        )
    for name, value in (
        ("expected_total", config.dataset_requirements.expected_total),
        (
            "expected_per_category",
            config.dataset_requirements.expected_per_category,
        ),
    ):
        if value is not None and value < 1:
            raise ConfigurationError(f"dataset_requirements.{name} must be positive")


def config_snapshot(config: ExperimentConfig) -> Dict[str, Any]:
    """Return the scientifically relevant, path-independent configuration."""

    return {
        "schema_version": config.schema_version,
        "protocol_version": config.protocol_version,
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "max_workers": config.max_workers,
        "inference": asdict(config.inference),
        "confidence": asdict(config.confidence),
        "weights": asdict(config.weights),
        "dataset_requirements": asdict(config.dataset_requirements),
    }


def load_models(path: Path) -> List[ModelSpec]:
    raw = _read_json(path)
    registry_mode = str(raw.get("registry_mode") or "reference")
    if registry_mode not in {"custom", "reference"}:
        raise ConfigurationError("registry_mode must be 'custom' or 'reference'")
    constraints = raw.get("panel_constraints")
    upper_bound_date = None
    if registry_mode == "reference":
        if not isinstance(constraints, dict):
            raise ConfigurationError(
                "A reference models.json must declare panel_constraints"
            )
        cutoff_upper_bound = constraints.get("pcq_cutoff_not_after")
        if not isinstance(cutoff_upper_bound, str) or not cutoff_upper_bound:
            raise ConfigurationError("Panel PCQ-cutoff upper bound is required")
        try:
            upper_bound_date = date.fromisoformat(cutoff_upper_bound)
        except ValueError as exc:
            raise ConfigurationError(
                "Panel PCQ-cutoff upper bound must be an ISO date"
            ) from exc
    models_raw = raw.get("models")
    if not isinstance(models_raw, list):
        raise ConfigurationError("models.json must contain a 'models' array")
    try:
        models = [ModelSpec.from_dict(item) for item in models_raw]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid model entry: {exc}") from exc
    if not models:
        raise ConfigurationError("models.json must contain at least one model")

    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ConfigurationError("Model display names must be unique")
    identities = [(model.provider, model.provider_model) for model in models]
    if len(identities) != len(set(identities)):
        raise ConfigurationError("Provider/model identities must be unique")
    if registry_mode == "reference" and len(models) != REFERENCE_PANEL_SIZE:
        raise ConfigurationError(
            f"EHQ reference panel requires {REFERENCE_PANEL_SIZE} models, "
            f"got {len(models)}"
        )
    # Two entries claiming the same side of the same generational pair would
    # overwrite one another when the pair is assembled, and the survivor would
    # be decided by file order. Refuse the file instead.
    pair_slots: Dict[tuple, str] = {}
    for model in models:
        if not model.pair or model.generation not in {"old", "new"}:
            continue
        slot = (model.pair, model.generation)
        if slot in pair_slots:
            raise ConfigurationError(
                f"Generational pair {model.pair!r} has two {model.generation!r} "
                f"members: {pair_slots[slot]} and {model.name}"
            )
        pair_slots[slot] = model.name

    for model in models:
        if model.provider not in {"openai", "huggingface", "openai_compatible"}:
            raise ConfigurationError(
                f"Unsupported provider {model.provider!r}: {model.name}"
            )
        if model.model_identity_policy not in {"record", "strict"}:
            raise ConfigurationError(
                f"model_identity_policy must be 'record' or 'strict': {model.name}"
            )
        if not isinstance(model.requires_api_key, bool):
            raise ConfigurationError(
                f"requires_api_key must be boolean: {model.name}"
            )
        if model.api_key_env and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", model.api_key_env
        ):
            raise ConfigurationError(
                f"api_key_env is not a valid environment-variable name: {model.name}"
            )
        if model.provider == "openai_compatible" and not model.base_url:
            raise ConfigurationError(
                f"openai_compatible requires base_url: {model.name}"
            )
        if model.base_url:
            parsed = urlparse(model.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ConfigurationError(
                    f"base_url must be an absolute HTTP(S) URL: {model.name}"
                )
        if model.reasoning_mode != "disabled":
            raise ConfigurationError(
                f"The normative EHQ protocol requires reasoning_mode='disabled': {model.name}"
            )
        if model.thinking_mode not in (None, "disabled"):
            raise ConfigurationError(
                f"Unsupported thinking_mode for the normative EHQ protocol: {model.name}"
            )
        if model.cutoff_evidence_status not in {
            "verified_primary",
            "verified_primary_qualified",
            "verified_human_adjudicated",
            "corroborated_secondary",
            "estimated",
            "unknown",
        }:
            raise ConfigurationError(
                f"Unsupported cutoff_evidence_status: {model.name}"
            )
        if model.cutoff_precision not in (None, "day", "month", "year"):
            raise ConfigurationError(f"Unsupported cutoff_precision: {model.name}")
        if model.cutoff_definition not in (
            None,
            "training_data_cutoff",
            "official_knowledge_cutoff",
            "reliable_knowledge_cutoff",
            "conservative_common_upper_bound",
        ):
            raise ConfigurationError(f"Unsupported cutoff_definition: {model.name}")
        if model.pcq_cutoff is not None:
            try:
                cutoff_date = date.fromisoformat(model.pcq_cutoff)
            except ValueError as exc:
                raise ConfigurationError(
                    f"pcq_cutoff must be an ISO date: {model.name}"
                ) from exc
            if (
                model.pcq_eligible
                and upper_bound_date is not None
                and cutoff_date > upper_bound_date
            ):
                raise ConfigurationError(
                    f"PCQ-eligible cutoff exceeds the panel bound: {model.name}"
                )
        elif model.pcq_eligible:
            raise ConfigurationError(
                f"PCQ-eligible model has no pcq_cutoff: {model.name}"
            )

    pair_counts: Dict[str, int] = {}
    for model in models:
        if model.pair:
            pair_counts[model.pair] = pair_counts.get(model.pair, 0) + 1
    invalid_pairs = {name: count for name, count in pair_counts.items() if count != 2}
    if invalid_pairs:
        raise ConfigurationError(f"Generational pairs must contain two models: {invalid_pairs}")
    if registry_mode == "reference" and len(pair_counts) != 8:
        raise ConfigurationError(f"Expected eight generational pairs, got {len(pair_counts)}")
    return models


def model_registry_release_issues(
    path: Path, models: Sequence[ModelSpec]
) -> List[str]:
    """Return evidence gaps that make a real-model panel non-publishable."""

    raw = _read_json(path)
    if str(raw.get("registry_mode") or "reference") == "custom":
        issues: List[str] = []
        for model in models:
            if model.operational_status != "verified":
                issues.append(
                    f"{model.name}: operational_status is not verified"
                )
            if not model.endpoint_verified_at:
                issues.append(
                    f"{model.name}: endpoint_verified_at is missing"
                )
            resolved = model.endpoint_verified_resolved_model
            if not resolved:
                issues.append(
                    f"{model.name}: endpoint_verified_resolved_model is missing"
                )
            elif model.model_identity_policy != "strict":
                issues.append(
                    f"{model.name}: model_identity_policy is not strict"
                )
            elif _normalized_model_identity(resolved) != _normalized_model_identity(
                model.provider_model
            ):
                issues.append(
                    f"{model.name}: endpoint verification resolved a different model"
                )
            if model.pcq_eligible:
                if not model.pcq_cutoff:
                    issues.append(
                        f"{model.name}: PCQ-eligible route has no cutoff"
                    )
                if model.cutoff_evidence_status not in {
                    "verified_primary",
                    "verified_human_adjudicated",
                }:
                    issues.append(
                        f"{model.name}: PCQ cutoff evidence is not release verified"
                    )
                if model.cutoff_source_type not in {
                    "official_model_card",
                    "official_api_documentation",
                    "official_provider_documentation",
                    "official_system_card",
                    "named_human_attestation",
                }:
                    issues.append(
                        f"{model.name}: PCQ cutoff source is not authoritative"
                    )
        return issues
    constraints = raw.get("panel_constraints")
    issues: List[str] = []
    if not isinstance(constraints, Mapping):
        return ["panel_constraints is missing"]
    if constraints.get("status") != "release_verified":
        issues.append("panel_constraints.status is not 'release_verified'")
    release_scope = constraints.get("release_verified_models")
    if not isinstance(release_scope, list):
        issues.append("panel_constraints.release_verified_models is missing")
        release_scope = []
    rows = raw.get("models")
    by_name = (
        {
            str(row.get("name")): row
            for row in rows
            if isinstance(row, Mapping)
        }
        if isinstance(rows, list)
        else {}
    )
    for model in models:
        row = by_name.get(model.name, {})
        if model.name not in release_scope:
            issues.append(f"{model.name}: model is outside the release-verified scope")
        if not row.get("pcq_cutoff"):
            issues.append(
                f"{model.name}: verified per-model pcq_cutoff is missing"
            )
        if row.get("cutoff_definition") not in {
            "training_data_cutoff",
            "official_knowledge_cutoff",
            "reliable_knowledge_cutoff",
            "conservative_common_upper_bound",
        }:
            issues.append(f"{model.name}: cutoff_definition is missing")
        if row.get("cutoff_evidence_status") not in {
            "verified_primary",
            "verified_human_adjudicated",
        }:
            issues.append(
                f"{model.name}: cutoff evidence is not release verified"
            )
        source_type = row.get("cutoff_source_type")
        if source_type not in {
            "official_model_card",
            "official_api_documentation",
            "official_provider_documentation",
            "official_system_card",
            "named_human_attestation",
        }:
            issues.append(f"{model.name}: cutoff_source_type is not authoritative")
        source = str(row.get("cutoff_source_url") or "")
        if source_type == "named_human_attestation":
            if row.get("cutoff_attestation_status") != "completed":
                issues.append(f"{model.name}: cutoff attestation is incomplete")
            if not row.get("cutoff_attestation_path"):
                issues.append(f"{model.name}: cutoff attestation path is missing")
            if not row.get("cutoff_attestation_reviewer"):
                issues.append(f"{model.name}: cutoff attestation reviewer is missing")
            if row.get("pcq_cutoff_is_conservative_upper_bound") is not True:
                issues.append(
                    f"{model.name}: adjudicated cutoff is not marked as an upper bound"
                )
        elif not source.startswith(("https://", "http://")):
            issues.append(f"{model.name}: cutoff_source_url is missing")
        if row.get("cutoff_precision") not in {"day", "month", "year"}:
            issues.append(f"{model.name}: cutoff_precision is missing")
        if not row.get("cutoff_verified_at"):
            issues.append(f"{model.name}: cutoff_verified_at is missing")
        if row.get("pcq_eligible") is not True:
            issues.append(f"{model.name}: model is not PCQ-eligible")
        if not row.get("endpoint_verified_at"):
            issues.append(f"{model.name}: endpoint_verified_at is missing")
        resolved = str(row.get("endpoint_verified_resolved_model") or "")
        if not resolved:
            issues.append(
                f"{model.name}: endpoint_verified_resolved_model is missing"
            )
        elif _normalized_model_identity(resolved) != _normalized_model_identity(
            model.provider_model
        ):
            issues.append(
                f"{model.name}: endpoint verification resolved a different model"
            )
        if row.get("reasoning_mode") != "disabled":
            issues.append(f"{model.name}: reasoning_mode is not explicitly disabled")
        if not row.get("non_reasoning_verified_at"):
            issues.append(f"{model.name}: non_reasoning_verified_at is missing")
        if row.get("non_reasoning_observed_thinking_tokens") != 0:
            issues.append(
                f"{model.name}: zero reasoning/thinking tokens were not verified"
            )
        if row.get("operational_status") != "verified":
            issues.append(f"{model.name}: operational_status is not verified")
    return issues


def _normalized_model_identity(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
