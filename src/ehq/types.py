"""Serializable data structures shared across the framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    provider_model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    requires_api_key: bool = True
    model_identity_policy: str = "record"
    model_provider: Optional[str] = None
    pair: Optional[str] = None
    generation: Optional[str] = None
    reported_model_name: Optional[str] = None
    pcq_cutoff: Optional[str] = None
    training_cutoff: Optional[str] = None
    reliable_knowledge_cutoff: Optional[str] = None
    reported_knowledge_cutoff: Optional[str] = None
    cutoff_precision: Optional[str] = None
    cutoff_definition: Optional[str] = None
    cutoff_source_url: Optional[str] = None
    cutoff_source_type: Optional[str] = None
    cutoff_evidence_status: str = "unknown"
    cutoff_verified_at: Optional[str] = None
    cutoff_notes: Optional[str] = None
    pcq_cutoff_is_conservative_upper_bound: bool = False
    cutoff_attestation_path: Optional[str] = None
    cutoff_attestation_reviewer: Optional[str] = None
    cutoff_attestation_status: Optional[str] = None
    model_reference_url: Optional[str] = None
    pcq_eligible: bool = False
    endpoint_verified_at: Optional[str] = None
    endpoint_verified_resolved_model: Optional[str] = None
    operational_status: str = "unverified"
    reasoning_mode: str = "disabled"
    thinking_mode: Optional[str] = None
    non_reasoning_verified_at: Optional[str] = None
    non_reasoning_observed_thinking_tokens: Optional[int] = None

    @property
    def cutoff(self) -> Optional[str]:
        """Backward-compatible alias for the canonical PCQ cutoff."""

        return self.pcq_cutoff

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ModelSpec":
        payload = dict(value)
        legacy_cutoff = payload.pop("cutoff", None)
        if "pcq_cutoff" not in payload and legacy_cutoff is not None:
            payload["pcq_cutoff"] = legacy_cutoff
        return cls(**payload)


@dataclass(frozen=True)
class InferenceRequest:
    model: ModelSpec
    prompt: str
    system_prompt: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    enable_search: bool = False
    enable_history: bool = False
    purpose: str = "answer"


@dataclass
class InferenceResponse:
    ok: bool
    text: Optional[str]
    requested_model: str
    resolved_model: Optional[str]
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    cache_hit: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "InferenceResponse":
        return cls(**value)


@dataclass
class Classification:
    label: str
    is_correct: bool
    abstention_detected: bool
    hedge_detected: bool
    substantive_answer_detected: bool
    reasons: List[str] = field(default_factory=list)
    automated_label: Optional[str] = None
    adjudicated_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRecord:
    question_id: str
    model: str
    category: str
    subcategory: str
    k: int
    started_at_utc: str
    completed_at_utc: str
    answer_response: InferenceResponse
    confidence_response: Optional[InferenceResponse]
    parsed_confidence: Optional[float]
    confidence_parse_strategy: Optional[str]
    confidence_parse_reason: Optional[str]
    confidence_terminal: bool
    classification: Optional[Classification]
    valid_for_ehq12: bool
    valid_for_ehq3: bool
    exclusion_reason: Optional[str] = None
    prompt_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value
