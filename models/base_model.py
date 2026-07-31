"""Abstract base class for all EHQ-3000 test models.

Ported from senolali/RQEval (models/base_model.py) for architectural
consistency across the two evaluation frameworks -- same caching
contract, same generate()/generate_with_confidence() shape, same
"raise on total failure" contract (the evaluator decides how to record
a failed item; a model silently returning "" would corrupt EHQ2/EHQ3).

Difference from RQEval's base_model.py: RQEval also tracks provider
-reported output-token counts (for its Efficiency metric) via a
tiktoken fallback. EHQ has no token-efficiency metric, so that
machinery was dropped here -- everything else (in-memory cache keyed
by md5(prompt+kwargs), the deterministic gate, batch_generate) is
unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import hashlib
import json


class BaseModel(ABC):
    """Abstract base class defining the interface for all EHQ test models."""

    def __init__(self, name: str, config: Dict[str, Any], deterministic: bool = False):
        self.name = name
        self.config = config
        # EHQ default: deterministic=False (temperature from config, usually
        # 0.7) -- unlike RQEval, EHQ deliberately wants natural variance in
        # the confidence-elicitation turn, not greedy decoding. Caching is
        # still available when a caller explicitly sets deterministic=True.
        self.deterministic = deterministic
        self._cache: Dict[str, str] = {}

    def _cache_key(self, prompt: str, **kwargs) -> str:
        payload = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _get_cached(self, key: Optional[str]) -> Optional[str]:
        if key is None or not self.deterministic:
            return None
        return self._cache.get(key)

    def _set_cached(self, key: Optional[str], value: str) -> None:
        if key is not None and self.deterministic:
            self._cache[key] = value

    def _maybe_cache_key(self, prompt: str, **kwargs) -> Optional[str]:
        """Return cache key only if caching is active (deterministic mode)."""
        if not self.deterministic:
            return None
        return self._cache_key(prompt, **kwargs)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt. Raises on total failure
        (all retries exhausted) -- callers must not treat "" as success."""
        raise NotImplementedError

    @abstractmethod
    def generate_with_confidence(self, question: str, **kwargs) -> Dict[str, Any]:
        """EHQ 2-turn protocol: answer the question, then ask for a 0-100
        verbalized confidence score. Returns
        {"answer": str, "conf_raw": str, "model": str}."""
        raise NotImplementedError

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
