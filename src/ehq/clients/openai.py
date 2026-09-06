"""Adapter for the official OpenAI Responses API."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..cache import ResponseCache
from ..types import InferenceRequest, InferenceResponse
from .http import JSONHTTPClient


def _response_text(data: Dict[str, Any]) -> Optional[str]:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts = []
    output: Iterable[Any] = data.get("output") or []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(
                content.get("text"), str
            ):
                parts.append(content["text"])
    combined = "".join(parts)
    return combined if combined.strip() else None


class OpenAIResponsesClient(JSONHTTPClient):
    """Call ``POST /responses`` without introducing an SDK dependency."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        requires_api_key: bool = True,
        cache: Optional[ResponseCache] = None,
        **kwargs: Any,
    ):
        super().__init__(
            provider_name="openai",
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key_env=api_key_env,
            requires_api_key=requires_api_key,
            cache=cache,
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/responses"

    def _request_once(self, request: InferenceRequest) -> InferenceResponse:
        payload: Dict[str, Any] = {
            "model": request.model.provider_model,
            "instructions": request.system_prompt,
            "input": request.prompt,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        data, failure = self._post_json(request, payload)
        if failure is not None:
            return failure
        assert data is not None
        text = _response_text(data)
        if text is None:
            return self._failure(
                request, "response_format", "OpenAI response text is missing"
            )
        resolved = data.get("model")
        metadata = {
            "id": data.get("id"),
            "created_at": data.get("created_at"),
            "status": data.get("status"),
        }
        identity_failure = self._identity_failure(
            request, resolved, metadata
        )
        if identity_failure is not None:
            return identity_failure
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return InferenceResponse(
            ok=True,
            text=text,
            requested_model=request.model.provider_model,
            resolved_model=str(resolved) if resolved else None,
            provider="openai",
            metadata=metadata,
            usage=usage,
        )
