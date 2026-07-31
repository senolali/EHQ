"""Mock model for testing the EHQ pipeline without real API calls.

Adapted from senolali/RQEval (models/mock_model.py): deterministic,
hash-seeded per prompt so the same question always gets the same mock
answer within a run (useful for reproducible dry-run smoke tests).
Response templates are EHQ-flavored (ABSTAIN / HEDGE / CONFIDENT-style
phrasing, not RQEval's generic reasoning templates) so a --dry-run
actually exercises classifier.classify_response()'s three branches
instead of always landing on CONFIDENT like the framework's old
placeholder text did.
"""

import random
import time
from typing import Any, Dict

from models.base_model import BaseModel

_RESPONSE_STYLES = [
    "I do not have any information about this.",
    "I'm not aware of this and cannot verify it.",
    "I believe the answer is probably {answer}, though I am not fully certain.",
    "As far as I know, it might be {answer}.",
    "The answer is {answer}.",
    "{answer} is the correct answer.",
]

_MOCK_ANSWERS = ["42", "the 1990s", "approximately 100", "New York",
                 "an unspecified value", "1,250"]


class MockModel(BaseModel):
    """Deterministic mock model for pipeline/wiring tests (--dry-run)."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        seed: int = 42,
        deterministic: bool = True,
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self.seed = seed

    def _rng_for(self, prompt: str) -> random.Random:
        # Hash-seeded per prompt: same question -> same mock answer within
        # a run, but different questions get different (varied) responses.
        return random.Random((hash(prompt) ^ self.seed) % (2**31))

    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._maybe_cache_key(prompt, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        rng = self._rng_for(prompt)
        style = rng.choice(_RESPONSE_STYLES)
        answer = rng.choice(_MOCK_ANSWERS)
        response = style.format(answer=answer) if "{answer}" in style else style

        self._set_cached(cache_key, response)
        return response

    def generate_with_confidence(self, question: str, **kwargs) -> Dict[str, Any]:
        answer = self.generate(question, **kwargs)
        rng = self._rng_for(question + "::conf")
        conf_raw = str(rng.randint(0, 100))
        return {"answer": answer, "conf_raw": conf_raw, "model": self.name}
