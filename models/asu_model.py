"""ASU CreateAI gateway model wrapper (REST /query endpoint).

Adapted from senolali/RQEval (models/asu_model.py) -- same retry/backoff/
pacing design, same response-parsing fallbacks, same "raise RuntimeError
on retry exhaustion" contract. Two EHQ-specific differences:

1. DEFAULT_BASE_URL: RQEval defaults to api-main-poc.aiml.asu.edu.
   EHQ's asu_client.py (used by the pcq/hnq/feq/ccq dataset generators)
   found that host 403s for some models and standardized on
   api-main.aiml.asu.edu (no "-poc") in 2026-07 -- kept as the default
   here too, for consistency with the rest of this project. Override
   via config (base_url:) or ASU_BASE_URL if RQEval's host is what you
   actually need.

2. generate_with_confidence(): EHQ's 2-turn protocol (answer, then a
   0-100 verbalized confidence score) -- RQEval has no equivalent
   since it doesn't measure calibration this way.

Payload follows the CreateAI Query endpoint spec:
    endpoint="query", action="query", request_source="override_params",
    model_provider=..., model_name=..., query=<prompt>,
    model_params={temperature, system_prompt, max_tokens}

Rate-limit strategy: proactive pacing (request_delay) plus exponential
backoff on HTTP 429, per ASU's "Best Practices" doc. Pacing is
thread-safe (the evaluator parallelizes across items).
"""

import time
import json
import logging
import threading
from typing import Any, Dict, Optional

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Same text asu_client.EHQ_SYSTEM_PROMPT uses for dataset generation --
# minimal and neutral so the CreateAI gateway doesn't inject its own
# 800-3400 token default template into every call (empty/missing
# system_prompt triggers that injection).
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

CONFIDENCE_PROMPT_TEMPLATE = (
    "You previously answered a question. On a scale from 0 to 100, "
    "how confident are you that your answer is correct? "
    "Reply with ONLY a single integer between 0 and 100.\n\n"
    "Question: {question}\n"
    "Your answer: {answer}\n\n"
    "Confidence (0-100):"
)


class ASUCreateAIModel(BaseModel):
    """ASU CreateAI /query REST endpoint wrapper."""

    DEFAULT_BASE_URL = "https://api-main.aiml.asu.edu"

    def __init__(
        self,
        name: str,
        api_key: str,
        model_name: str,                 # CreateAI model key, e.g. "gpt4o_mini"
        model_provider: str,             # e.g. "openai", "aws", "gcp-deepmind"
        config: Dict[str, Any],
        base_url: Optional[str] = None,
        deterministic: bool = False,
        temperature: Optional[float] = None,
        max_retries: int = 5,
        timeout: int = 120,
        max_tokens: int = 512,
        request_delay: float = 1.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self._temperature   = temperature
        self.api_key        = api_key
        self.model_name     = model_name
        self.model_provider = model_provider
        self.base_url       = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_retries    = max_retries
        self.timeout        = timeout
        self.max_tokens     = max_tokens
        self.request_delay  = request_delay
        self.system_prompt  = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._last_call_ts  = 0.0
        self._pace_lock     = threading.Lock()

        logger.info(
            f"[{name}] ASU CreateAI | provider={model_provider} "
            f"model={model_name} | base_url={self.base_url} | "
            f"request_delay={request_delay:.2f}s"
        )

    # ------------------------------------------------------------------
    def _pace(self):
        """Thread-safe proactive pacing to stay under the gateway limits."""
        with self._pace_lock:
            now = time.time()
            remaining = self.request_delay - (now - self._last_call_ts)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call_ts = time.time()

    def _build_payload(self, prompt: str, temperature: Optional[float]) -> Dict[str, Any]:
        temp = temperature if temperature is not None else (
            self._temperature if self._temperature is not None
            else (0.0 if self.deterministic else 0.7)
        )
        model_params: Dict[str, Any] = {
            "temperature": float(temp),
            "max_tokens": int(self.max_tokens),
            "system_prompt": self.system_prompt,
        }
        return {
            "endpoint": "query",
            "action": "query",
            "request_source": "override_params",
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "query": prompt,
            "model_params": model_params,
            "enable_search": False,
            "enable_history": False,
        }

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Extract the model text from CreateAI response variants."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            inner = data.get("response", data)
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                txt = inner.get("response")
                if isinstance(txt, str):
                    return txt
                txt = inner.get("content") or inner.get("text")
                if isinstance(txt, str):
                    return txt
        raise ValueError(f"Unexpected CreateAI response format: {str(data)[:300]}")

    # ------------------------------------------------------------------
    def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        cache_key = self._maybe_cache_key(prompt, temperature=temperature, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            import requests
        except ImportError:
            raise ImportError("Run: pip install requests")

        url     = f"{self.base_url}/query"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt, temperature)
        backoff = 30

        for attempt in range(self.max_retries):
            self._pace()
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=self.timeout)

                if resp.status_code == 429:
                    logger.warning(
                        f"[{self.name}] 429 rate limit "
                        f"(attempt {attempt+1}/{self.max_retries}) — "
                        f"waiting {backoff}s."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                    continue

                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"[{self.name}] Auth error {resp.status_code}: "
                        f"check ASU_CREATEAI_TOKEN. Body: {resp.text[:300]}"
                    )

                resp.raise_for_status()
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    raise ValueError(f"Non-JSON body: {resp.text[:300]}")

                result = self._extract_text(data).strip()
                if not result:
                    raise ValueError("Empty response text from gateway.")
                self._set_cached(cache_key, result)
                return result

            except RuntimeError:
                raise                       # auth errors: fail fast
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{self.name}] Transient error "
                        f"(attempt {attempt+1}/{self.max_retries}): {e} — "
                        f"retrying in 5s..."
                    )
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"[{self.name}] All {self.max_retries} retries failed. "
                        f"Last error: {e}"
                    )

        raise RuntimeError(f"[{self.name}] generate() exited retry loop unexpectedly.")

    def generate_with_confidence(self, question: str, **kwargs) -> Dict[str, Any]:
        answer = self.generate(question, **kwargs)
        conf_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        conf_raw = self.generate(conf_prompt, **kwargs)
        return {"answer": answer, "conf_raw": conf_raw, "model": self.name}
