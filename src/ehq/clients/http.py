"""Shared HTTP behavior for public model-provider adapters."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

from ..cache import ResponseCache
from ..tls import configure_platform_truststore
from ..types import InferenceRequest, InferenceResponse
from .base import BaseClient


def normalized_model_identity(value: str) -> str:
    """Normalize only punctuation and case for model-identity comparisons."""

    return "".join(character for character in value.lower() if character.isalnum())


class JSONHTTPClient(BaseClient):
    """Base class for authenticated JSON-over-HTTP model APIs."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key_env: Optional[str],
        requires_api_key: bool,
        cache: Optional[ResponseCache] = None,
        api_key: Optional[str] = None,
        session: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(cache=cache, **kwargs)
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.requires_api_key = requires_api_key
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env) if api_key_env else None
        )
        self.session = session
        self._thread_local = threading.local()

    def _session(self) -> Any:
        if self.session is not None:
            return self.session
        value = getattr(self._thread_local, "session", None)
        if value is None:
            configure_platform_truststore()
            import requests

            value = requests.Session()
            self._thread_local.session = value
        return value

    def _failure(
        self,
        request: InferenceRequest,
        error_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        return InferenceResponse(
            ok=False,
            text=None,
            requested_model=request.model.provider_model,
            resolved_model=None,
            provider=self.provider_name,
            metadata=metadata or {},
            error_type=error_type,
            error_message=message,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(
        self, request: InferenceRequest, payload: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[InferenceResponse]]:
        if self.requires_api_key and not self.api_key:
            name = self.api_key_env or "configured API key"
            return None, self._failure(
                request, "authentication", f"{name} is not configured"
            )
        try:
            session = self._session()
        except ImportError:
            return None, self._failure(
                request,
                "configuration",
                "The 'requests' package is required for real provider calls",
            )
        try:
            http = session.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            kind = (
                "timeout"
                if exc.__class__.__name__.lower().endswith("timeout")
                else "transport"
            )
            return None, self._failure(request, kind, str(exc))

        body_preview = str(getattr(http, "text", ""))[:500]
        if http.status_code in (401, 403):
            return None, self._failure(
                request,
                "authentication",
                f"{self.provider_name} returned HTTP {http.status_code}: "
                f"{body_preview}",
            )
        if http.status_code == 429:
            return None, self._failure(
                request,
                "rate_limit",
                f"{self.provider_name} returned HTTP 429",
            )
        if not http.ok:
            return None, self._failure(
                request,
                "http",
                f"{self.provider_name} returned HTTP {http.status_code}: "
                f"{body_preview}",
            )
        try:
            data = http.json()
        except ValueError:
            return None, self._failure(
                request,
                "response_format",
                f"{self.provider_name} returned non-JSON content",
            )
        if not isinstance(data, dict):
            return None, self._failure(
                request,
                "response_format",
                f"{self.provider_name} returned a non-object JSON envelope",
            )
        return data, None

    def _identity_failure(
        self,
        request: InferenceRequest,
        resolved: Any,
        metadata: Dict[str, Any],
    ) -> Optional[InferenceResponse]:
        if not resolved:
            return None
        matches = normalized_model_identity(str(resolved)) == normalized_model_identity(
            request.model.provider_model
        )
        metadata["model_identity_match"] = matches
        if not matches and request.model.model_identity_policy == "strict":
            return self._failure(
                request,
                "model_mismatch",
                f"Requested {request.model.provider_model!r}, resolved {resolved!r}",
                metadata,
            )
        return None
