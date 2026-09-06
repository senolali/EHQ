"""End-to-end evaluation runner with resumable checkpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..cache import ResponseCache
from ..checkpoint import Checkpoint
from ..clients import (
    BaseClient,
    MockClient,
    OpenAICompatibleClient,
    OpenAIResponsesClient,
)
from ..config import ExperimentConfig
from ..hashing import sha256_text
from ..prompts import (
    CONFIDENCE_SYSTEM_PROMPT,
    build_answer_prompt,
    build_confidence_prompt,
)
from ..selection import select_items
from ..types import EvaluationRecord, InferenceRequest, ModelSpec
from .classifier import classify_response
from .confidence import parse_confidence_detailed
from .scoring import compute_ehq_scores, compute_grouped_scores


def _request(
    config: ExperimentConfig,
    model: ModelSpec,
    prompt: str,
    purpose: str,
) -> InferenceRequest:
    return InferenceRequest(
        model=model,
        prompt=prompt,
        system_prompt=(
            CONFIDENCE_SYSTEM_PROMPT
            if purpose == "confidence"
            else config.inference.system_prompt
        ),
        temperature=config.inference.temperature,
        max_tokens=(
            config.confidence.max_tokens
            if purpose == "confidence"
            else config.inference.max_tokens
        ),
        timeout_seconds=config.inference.timeout_seconds,
        enable_search=config.inference.enable_search,
        enable_history=config.inference.enable_history,
        purpose=purpose,
    )


def knowability_for_model(
    item: Mapping[str, Any], model: ModelSpec
) -> int:
    """Resolve model-conditional k, falling back to the common panel value."""

    if str(item.get("category") or "").upper() == "PCQ" and not model.pcq_eligible:
        return 1

    for field in ("knowability_by_model", "model_knowability"):
        values = item.get(field)
        if isinstance(values, Mapping):
            for key in (model.name, model.provider_model):
                if key in values:
                    value = int(values[key])
                    if value not in (0, 1):
                        raise ValueError(
                            f"Invalid knowability {value} for {model.name}"
                        )
                    return value
    value = int(item.get("expected_knowability", 0))
    if value not in (0, 1):
        raise ValueError(f"Invalid expected_knowability {value}")
    return value


def client_for_model(
    config: ExperimentConfig,
    model: ModelSpec,
    *,
    offline: bool = False,
    request_event_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> BaseClient:
    if offline:
        return MockClient()
    cache = ResponseCache(config.cache_dir)
    common = {
        "cache": cache,
        "max_retries": config.inference.max_retries,
        "request_delay_seconds": config.inference.request_delay_seconds,
        "event_callback": request_event_callback,
    }
    if model.provider == "openai":
        return OpenAIResponsesClient(
            base_url=model.base_url,
            api_key_env=model.api_key_env or "OPENAI_API_KEY",
            requires_api_key=model.requires_api_key,
            **common,
        )
    if model.provider == "huggingface":
        return OpenAICompatibleClient(
            provider_name="huggingface",
            base_url=model.base_url or "https://router.huggingface.co/v1",
            api_key_env=model.api_key_env or "HF_TOKEN",
            requires_api_key=model.requires_api_key,
            **common,
        )
    if model.provider == "openai_compatible":
        if not model.base_url:
            raise ValueError(
                f"base_url is required for openai_compatible model {model.name!r}"
            )
        return OpenAICompatibleClient(
            provider_name="openai_compatible",
            base_url=model.base_url,
            api_key_env=(
                model.api_key_env or "OPENAI_COMPATIBLE_API_KEY"
                if model.requires_api_key
                else None
            ),
            requires_api_key=model.requires_api_key,
            **common,
        )
    raise ValueError(f"Unsupported provider: {model.provider}")


def credential_env_for_model(model: ModelSpec) -> Optional[str]:
    """Return the configured credential variable, or ``None`` for local APIs."""

    if not model.requires_api_key:
        return None
    if model.api_key_env:
        return model.api_key_env
    return {
        "openai": "OPENAI_API_KEY",
        "huggingface": "HF_TOKEN",
        "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    }.get(model.provider)


def evaluate_item(
    item: Mapping[str, Any],
    model: ModelSpec,
    config: ExperimentConfig,
    client: BaseClient,
    *,
    adjudication: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    question_id = str(item.get("question_id") or item.get("id"))
    category = str(item.get("category")).upper()
    subcategory = str(item.get("subcategory") or "UNSPECIFIED")
    answer_prompt = build_answer_prompt(item)
    answer = client.query(_request(config, model, answer_prompt, "answer"))
    if not answer.ok or not answer.text:
        return EvaluationRecord(
            question_id=question_id,
            model=model.name,
            category=category,
            subcategory=subcategory,
            k=knowability_for_model(item, model),
            started_at_utc=started_at,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            answer_response=answer,
            confidence_response=None,
            parsed_confidence=None,
            confidence_parse_strategy=None,
            confidence_parse_reason=None,
            confidence_terminal=False,
            classification=None,
            valid_for_ehq12=False,
            valid_for_ehq3=False,
            exclusion_reason=answer.error_type or "answer_failure",
            prompt_sha256=sha256_text(answer_prompt),
        ).to_dict()

    confidence_prompt = build_confidence_prompt(answer_prompt, answer.text)
    confidence_response = client.query(
        _request(config, model, confidence_prompt, "confidence")
    )
    confidence_parse = (
        parse_confidence_detailed(confidence_response.text)
        if confidence_response.ok
        else None
    )
    confidence = confidence_parse.value if confidence_parse else None
    confidence_terminal = bool(
        (confidence_response.ok and confidence is None)
        or confidence_response.error_type == "response_format"
    )
    override = None
    if adjudication:
        override = adjudication.get(f"{model.name}::{question_id}")
    classification = classify_response(
        answer.text,
        item,
        adjudicated_label=override,
    )
    return EvaluationRecord(
        question_id=question_id,
        model=model.name,
        category=category,
        subcategory=subcategory,
        k=knowability_for_model(item, model),
        started_at_utc=started_at,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        answer_response=answer,
        confidence_response=confidence_response,
        parsed_confidence=confidence,
        confidence_parse_strategy=(
            confidence_parse.strategy if confidence_parse else None
        ),
        confidence_parse_reason=(
            confidence_parse.reason
            if confidence_parse
            else confidence_response.error_type
        ),
        confidence_terminal=confidence_terminal,
        classification=classification,
        valid_for_ehq12=True,
        valid_for_ehq3=confidence is not None,
        exclusion_reason=None,
        prompt_sha256=sha256_text(answer_prompt),
    ).to_dict()


def run_model(
    items: Sequence[Mapping[str, Any]],
    model: ModelSpec,
    config: ExperimentConfig,
    *,
    checkpoint: Checkpoint,
    offline: bool = False,
    limit: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    adjudication: Optional[Mapping[str, str]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    request_event_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    client = client_for_model(
        config,
        model,
        offline=offline,
        request_event_callback=request_event_callback,
    )
    sampled = select_items(
        items, categories=categories, limit=limit, seed=config.seed
    )
    selected = [
        item for item in sampled if knowability_for_model(item, model) == 0
    ]
    excluded_k1 = len(sampled) - len(selected)
    selected_ids = {
        str(item.get("question_id") or item.get("id")) for item in selected
    }
    existing = {
        str(row["question_id"]): row
        for row in checkpoint.records()
        if row.get("model") == model.name
        and str(row.get("question_id")) in selected_ids
    }
    pending = [
        item
        for item in selected
        if str(item.get("question_id") or item.get("id")) not in existing
    ]

    session_started = monotonic()
    resumed_records = len(existing)
    completed_items = resumed_records
    valid_ehq12 = sum(
        bool(row.get("valid_for_ehq12")) for row in existing.values()
    )
    valid_ehq3 = sum(
        bool(row.get("valid_for_ehq3")) for row in existing.values()
    )
    terminal_missing_confidence = sum(
        bool(row.get("confidence_terminal")) for row in existing.values()
    )
    retryable_records = sum(
        not bool(row.get("valid_for_ehq3"))
        and not bool(row.get("confidence_terminal"))
        for row in existing.values()
    )
    technical_failures = sum(
        not bool(row.get("valid_for_ehq12")) for row in existing.values()
    )

    def notify_progress(phase: str) -> None:
        if progress_callback is None:
            return
        elapsed_seconds = max(0.0, monotonic() - session_started)
        completed_this_session = completed_items - resumed_records
        rate_per_second = (
            completed_this_session / elapsed_seconds
            if completed_this_session and elapsed_seconds > 0
            else None
        )
        remaining_items = max(0, len(selected) - completed_items)
        progress_callback(
            {
                "phase": phase,
                "completed_items": completed_items,
                "total_items": len(selected),
                "completed_this_session": completed_this_session,
                "resumed_records": resumed_records,
                "valid_ehq12": valid_ehq12,
                "valid_ehq3": valid_ehq3,
                "terminal_missing_confidence": terminal_missing_confidence,
                "retryable_records": retryable_records,
                "technical_failures": technical_failures,
                "elapsed_seconds": elapsed_seconds,
                "items_per_minute": (
                    rate_per_second * 60.0
                    if rate_per_second is not None
                    else None
                ),
                "eta_seconds": (
                    remaining_items / rate_per_second
                    if rate_per_second
                    else None
                ),
            }
        )

    def accept_row(row: Dict[str, Any]) -> None:
        nonlocal completed_items
        nonlocal valid_ehq12
        nonlocal valid_ehq3
        nonlocal terminal_missing_confidence
        nonlocal retryable_records
        nonlocal technical_failures
        if row.get("valid_for_ehq3") or (
            row.get("valid_for_ehq12") and row.get("confidence_terminal")
        ):
            checkpoint.append(row)
        existing[row["question_id"]] = row
        completed_items += 1
        valid_ehq12 += int(bool(row.get("valid_for_ehq12")))
        valid_ehq3 += int(bool(row.get("valid_for_ehq3")))
        terminal_missing_confidence += int(
            bool(row.get("confidence_terminal"))
        )
        retryable_records += int(
            not bool(row.get("valid_for_ehq3"))
            and not bool(row.get("confidence_terminal"))
        )
        technical_failures += int(not bool(row.get("valid_for_ehq12")))
        notify_progress("evaluating")

    if resumed_records:
        notify_progress("resume_loaded")

    if config.max_workers == 1:
        for item in pending:
            row = evaluate_item(
                item, model, config, client, adjudication=adjudication
            )
            accept_row(row)
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_item,
                    item,
                    model,
                    config,
                    client,
                    adjudication=adjudication,
                ): item
                for item in pending
            }
            for future in as_completed(futures):
                row = future.result()
                accept_row(row)

    ordered = [
        existing[str(item.get("question_id") or item.get("id"))] for item in selected
    ]
    weights = {
        "ehq1": config.weights.ehq1,
        "ehq2": config.weights.ehq2,
        "ehq3": config.weights.ehq3,
    }
    scores = compute_ehq_scores(
        ordered,
        weights=weights,
        n_bins=config.confidence.n_bins,
    )
    n_retryable_records = sum(
        not bool(row.get("valid_for_ehq3"))
        and not bool(row.get("confidence_terminal"))
        for row in ordered
    )
    n_terminal_missing_confidence = sum(
        bool(row.get("confidence_terminal")) for row in ordered
    )
    scores["n_retryable_records"] = n_retryable_records
    scores["n_terminal_missing_confidence"] = n_terminal_missing_confidence
    return {
        "model": model.name,
        "provider_endpoint": client.endpoint,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "category_scores": compute_grouped_scores(
            ordered,
            group_field="category",
            weights=weights,
            n_bins=config.confidence.n_bins,
        ),
        "subcategory_scores": compute_grouped_scores(
            ordered,
            group_field="subcategory",
            weights=weights,
            n_bins=config.confidence.n_bins,
        ),
        "n_retryable_records": n_retryable_records,
        "n_terminal_missing_confidence": n_terminal_missing_confidence,
        "n_excluded_k1_before_request": excluded_k1,
        "records": ordered,
    }
