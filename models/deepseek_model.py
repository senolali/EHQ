"""DeepSeek API model wrapper (OpenAI-compatible endpoint).

Adapted from senolali/RQEval (models/deepseek_model.py): same pacing/
retry/backoff design and "raise RuntimeError on exhaustion" contract.
EHQ-specific additions: a system_prompt (RQEval's DeepSeek wrapper
sends no system message at all -- EHQ needs the SAME neutral system
prompt across all 20 test models for a fair comparison, see
models/asu_model.py DEFAULT_SYSTEM_PROMPT) and generate_with_confidence()
for the 2-turn answer+confidence protocol.
"""

import time
import logging
from typing import Any, Dict, Optional

from models.base_model import BaseModel
from models.asu_model import DEFAULT_SYSTEM_PROMPT, CONFIDENCE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_TIER_RPM: Dict[str, int] = {
    "free": 60,
    "paid": 600,
}


class DeepSeekModel(BaseModel):
    """DeepSeek Chat API wrapper using the OpenAI-compatible interface."""

    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        name: str,
        api_key: str,
        model_id: str,
        config: Dict[str, Any],
        base_url: Optional[str] = None,
        deterministic: bool = False,
        temperature: Optional[float] = None,
        max_retries: int = 5,
        timeout: int = 60,
        max_tokens: int = 512,
        request_delay: Optional[float] = None,   # None -> auto from tier
        tier: str = "free",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._client = None
        super().__init__(name=name, config=config, deterministic=deterministic)
        self._temperature  = temperature
        self.api_key       = api_key
        self.model_id      = model_id
        self.base_url      = base_url or self.BASE_URL
        self.max_retries   = max_retries
        self.timeout       = timeout
        self.max_tokens    = max_tokens
        self.tier          = tier
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._last_call_ts = 0.0

        rpm = _TIER_RPM.get(tier, _TIER_RPM["free"])
        self.request_delay = request_delay if request_delay is not None \
                             else (60.0 / rpm) * 1.1
        logger.info(
            f"[{name}] tier={tier} | request_delay={self.request_delay:.2f}s | "
            f"~{60/self.request_delay:.0f} RPM effective"
        )

    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            return self._client
        except ImportError:
            raise ImportError("Run: pip install openai")

    def _pace(self):
        remaining = self.request_delay - (time.time() - self._last_call_ts)
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        cache_key = self._maybe_cache_key(prompt, temperature=temperature, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client  = self._build_client()
        backoff = 60
        temp = temperature if temperature is not None else (
            self._temperature if self._temperature is not None
            else (0.0 if self.deterministic else 0.7)
        )

        for attempt in range(self.max_retries):
            self._pace()
            try:
                self._last_call_ts = time.time()
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=temp,
                )
                result = response.choices[0].message.content.strip()
                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    logger.warning(
                        f"[{self.name}] Rate limit (attempt {attempt+1}/{self.max_retries}) "
                        f"— waiting {backoff}s."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                elif attempt < self.max_retries - 1:
                    logger.warning(f"[{self.name}] Transient error (attempt {attempt+1}): {e} — retrying in 5s...")
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"[{self.name}] All {self.max_retries} retries failed. Last error: {e}"
                    )

        raise RuntimeError(f"[{self.name}] generate() exited retry loop unexpectedly.")

    def generate_with_confidence(self, question: str, **kwargs) -> Dict[str, Any]:
        answer = self.generate(question, **kwargs)
        conf_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        conf_raw = self.generate(conf_prompt, **kwargs)
        return {"answer": answer, "conf_raw": conf_raw, "model": self.name}
