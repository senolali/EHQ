"""Atomic, content-addressed response cache."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import PROTOCOL_VERSION, RESPONSE_IDENTITY_VERSION
from .hashing import sha256_json
from .types import InferenceRequest, InferenceResponse


class ResponseCache:
    """Stores successful response envelopes under complete request identities."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def identity(request: InferenceRequest, endpoint: str) -> Dict[str, Any]:
        return {
            # Field name and value are deliberately stable: the cache key
            # identifies the provider request, not the scoring protocol.
            "framework_version": RESPONSE_IDENTITY_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "provider": request.model.provider,
            "model_provider": request.model.model_provider,
            "provider_model": request.model.provider_model,
            "reasoning_mode": request.model.reasoning_mode,
            "thinking_mode": request.model.thinking_mode,
            "endpoint": endpoint,
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout_seconds": request.timeout_seconds,
            "enable_search": request.enable_search,
            "enable_history": request.enable_history,
            "purpose": request.purpose,
        }

    def key(self, request: InferenceRequest, endpoint: str) -> str:
        return sha256_json(self.identity(request, endpoint))

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, request: InferenceRequest, endpoint: str) -> Optional[InferenceResponse]:
        key = self.key(request, endpoint)
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            expected = self.identity(request, endpoint)
            if envelope.get("identity") != expected:
                return None
            response = InferenceResponse.from_dict(envelope["response"])
            if not response.ok:
                return None
            response.cache_hit = True
            return response
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def put(
        self,
        request: InferenceRequest,
        endpoint: str,
        response: InferenceResponse,
    ) -> Path:
        if not response.ok:
            raise ValueError("Technical failures must not be cached as model responses")
        key = self.key(request, endpoint)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_key": key,
            "identity": self.identity(request, endpoint),
            "response": asdict(response),
        }
        payload = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path
