"""Deterministic offline client for pipeline and test validation."""

from __future__ import annotations

from typing import Callable, Optional

from ..types import InferenceRequest, InferenceResponse
from .base import BaseClient


class MockClient(BaseClient):
    def __init__(
        self,
        responder: Optional[Callable[[InferenceRequest], str]] = None,
    ):
        super().__init__(cache=None, max_retries=1, request_delay_seconds=0)
        self.responder = responder or self._default_responder

    @property
    def endpoint(self) -> str:
        return "mock://offline"

    @staticmethod
    def _default_responder(request: InferenceRequest) -> str:
        if request.purpose == "confidence":
            return "0"
        return "I do not know. I cannot answer this question reliably."

    def _request_once(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            ok=True,
            text=self.responder(request),
            requested_model=request.model.provider_model,
            resolved_model=request.model.provider_model,
            provider="mock",
            metadata={"offline": True},
        )
