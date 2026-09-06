"""Adapter for Hugging Face and other OpenAI-compatible chat APIs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..cache import ResponseCache
from ..types import InferenceRequest, InferenceResponse
from .http import JSONHTTPClient


def _chat_text(data: Dict[str, Any]) -> Optional[str]:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        combined = "".join(parts)
        return combined if combined.strip() else None
    return None


class OpenAICompatibleClient(JSONHTTPClient):
    """Call a standard ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key_env: Optional[str],
        requires_api_key: bool,
        cache: Optional[ResponseCache] = None,
        **kwargs: Any,
    ):
        super().__init__(
            provider_name=provider_name,
            base_url=base_url,
            api_key_env=api_key_env,
            requires_api_key=requires_api_key,
            cache=cache,
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_once(self, request: InferenceRequest) -> InferenceResponse:
        payload = {
            "model": request.model.provider_model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        data, failure = self._post_json(request, payload)
        if failure is not None:
            return failure
        assert data is not None
        text = _chat_text(data)
        if text is None:
            return self._failure(
                request,
                "response_format",
                f"{self.provider_name} response text is missing",
            )
        resolved = data.get("model")
        metadata = {
            "id": data.get("id"),
            "created": data.get("created"),
            "system_fingerprint": data.get("system_fingerprint"),
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
            provider=self.provider_name,
            metadata=metadata,
            usage=usage,
        )
