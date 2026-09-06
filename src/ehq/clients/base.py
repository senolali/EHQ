"""Provider-neutral client contract and retry behavior."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..cache import ResponseCache
from ..types import InferenceRequest, InferenceResponse


class BaseClient(ABC):
    def __init__(
        self,
        cache: Optional[ResponseCache] = None,
        max_retries: int = 5,
        request_delay_seconds: float = 1.0,
        sleep=time.sleep,
        clock=time.monotonic,
        event_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        self.cache = cache
        self.max_retries = max_retries
        self.request_delay_seconds = request_delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._pace_lock = Lock()
        self._last_request_started: Optional[float] = None
        self._event_callback = event_callback

    def _emit_request_event(self, **event: Any) -> None:
        if self._event_callback is not None:
            self._event_callback(event)

    @property
    @abstractmethod
    def endpoint(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _request_once(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    def query(self, request: InferenceRequest, use_cache: bool = True) -> InferenceResponse:
        if self.cache and use_cache:
            cached = self.cache.get(request, self.endpoint)
            if cached is not None:
                self._emit_request_event(
                    phase="cache_hit",
                    purpose=request.purpose,
                    attempt=0,
                    max_retries=self.max_retries,
                )
                return cached

        last: Optional[InferenceResponse] = None
        retry_history: List[Dict[str, Any]] = []
        for attempt in range(1, self.max_retries + 1):
            self._pace_request_start()
            response = self._request_once(request)
            response.attempts = attempt
            observed_reasoning = _observed_reasoning_tokens(response.usage)
            if observed_reasoning is not None:
                response.metadata["ehq_observed_reasoning_tokens"] = observed_reasoning
            if (
                response.ok
                and request.model.reasoning_mode == "disabled"
                and observed_reasoning is not None
                and observed_reasoning > 0
            ):
                response.ok = False
                response.error_type = "reasoning_mode_violation"
                response.error_message = (
                    "Provider reported positive reasoning/thinking-token usage "
                    f"({observed_reasoning}) under the non-reasoning EHQ protocol"
                )
            if response.ok:
                response.metadata.setdefault("ehq_retry_history", retry_history)
                if attempt > 1:
                    self._emit_request_event(
                        phase="request_recovered",
                        purpose=request.purpose,
                        attempt=attempt,
                        max_retries=self.max_retries,
                    )
                if self.cache and use_cache:
                    self.cache.put(request, self.endpoint, response)
                return response
            last = response
            retry_history.append(
                {
                    "attempt": attempt,
                    "error_type": response.error_type,
                    "error_message": response.error_message,
                }
            )
            if response.error_type in {
                "authentication",
                "configuration",
                "model_mismatch",
                "reasoning_mode_violation",
            }:
                self._emit_request_event(
                    phase="request_terminal_failure",
                    purpose=request.purpose,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )
                break
            if attempt < self.max_retries:
                wait_seconds = min(2 ** (attempt - 1), 30)
                self._emit_request_event(
                    phase="retry_scheduled",
                    purpose=request.purpose,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    max_retries=self.max_retries,
                    wait_seconds=wait_seconds,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )
                self._sleep(wait_seconds)
            else:
                self._emit_request_event(
                    phase="request_exhausted",
                    purpose=request.purpose,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_type=response.error_type,
                    error_message=response.error_message,
                )

        assert last is not None
        last.metadata.setdefault("ehq_retry_history", retry_history)
        return last

    def _pace_request_start(self) -> None:
        """Enforce one provider-wide start interval across worker threads."""

        if self.request_delay_seconds <= 0:
            return
        with self._pace_lock:
            now = self._clock()
            if self._last_request_started is not None:
                wait = self.request_delay_seconds - (
                    now - self._last_request_started
                )
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last_request_started = now


_REASONING_TOKEN_KEYS = {
    "reasoning_tokens",
    "thinking_tokens",
    "thinking",
}


def _observed_reasoning_tokens(value: Any) -> Optional[int]:
    """Return the largest explicitly reported reasoning/thinking token count."""

    counts: List[int] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key.lower() in _REASONING_TOKEN_KEYS and isinstance(
                    child, (int, float)
                ) and not isinstance(child, bool):
                    counts.append(int(child))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return max(counts) if counts else None
