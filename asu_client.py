"""
ASU CreateAI API Client — EHQ Pipeline
=======================================
ASU CreateAI /query REST endpoint icin istemci.

Kullanim: hem ana EHQ deneyinde (20 model) hem de CCQ/PCQ uretiminde
(GPT-4.1 uretici, hakem paneli) ASU uzerinden model cagirmak icin.

Ortam degiskenleri:
  ASU_CREATEAI_TOKEN  -> Bearer token (service token, TIRNAK OLMADAN set et)
                         Dogru: set ASU_CREATEAI_TOKEN=eyJhb...
                         Yanlis: set ASU_CREATEAI_TOKEN="eyJhb..."  (403 verir!)
  ASU_BASE_URL        -> default https://api-main.aiml.asu.edu

DOGRULANMIS PAYLOAD (service token, 2026-07):
  endpoint + action + request_source="override_params" ZORUNLU.
  Bos system_prompt -> gateway 800-3400 token sablon enjekte eder.
  thinking_level GONDERILMEZ -> non-thinking mod (EHQ tezi).

CACHE:
  SHA-256(model_name + system_prompt + query) -> disk cache.
  Ayni cagri tekrar ucretlenmez / yeniden sorulmaz (reproducibility).
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("asu_client")

DEFAULT_BASE_URL = "https://api-main.aiml.asu.edu"       # api-main-poc 403 veriyor
QUERY_ENDPOINT = "/query"

# EHQ system prompt (minimal, notr — gateway sablon enjeksiyonunu bastirir).
# Ayni prompt TUM modellere gider -> tutarli karsilastirma.
EHQ_SYSTEM_PROMPT = "You are a helpful assistant."

CACHE_DIR = Path("cache_asu")
CACHE_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Cache yardimcilari
# ----------------------------------------------------------------------

def _cache_key(model_name: str, system_prompt: str, query: str) -> str:
    raw = f"{model_name}||{system_prompt}||{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))["response"]
        except Exception:
            return None
    return None


def _cache_put(key: str, response: str, meta: dict) -> None:
    f = CACHE_DIR / f"{key}.json"
    f.write_text(json.dumps({"response": response, "meta": meta},
                            ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Tek ASU cagrisi
# ----------------------------------------------------------------------

def asu_query(model_name: str,
              model_provider: str,
              query: str,
              system_prompt: str = EHQ_SYSTEM_PROMPT,
              temperature: float = 0.7,
              max_retries: int = 5,
              timeout: int = 120,
              request_delay: float = 1.0,
              use_cache: bool = True) -> Optional[str]:
    """ASU CreateAI /query'e tek istek. Cevap metnini dondurur (veya None).

    model_name: CreateAI key ( or. 'gpt4o', 'claude4_8_opus')
    model_provider: 'openai' | 'aws' | 'gcp-deepmind'
    """
    key = _cache_key(model_name, system_prompt, query)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    token = os.environ.get("ASU_CREATEAI_TOKEN")
    if not token:
        raise RuntimeError("ASU_CREATEAI_TOKEN ortam degiskeni tanimli degil.")
    base_url = os.environ.get("ASU_BASE_URL", DEFAULT_BASE_URL)
    url = base_url.rstrip("/") + QUERY_ENDPOINT

    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    payload = {
        "endpoint": "query",                    # ← EKLENDİ (kritik)
        "action": "query",
        "request_source": "override_params",
        "model_provider": model_provider,
        "model_name": model_name,
        "query": query,
        "model_params": {
            "temperature": temperature,
            "system_prompt": system_prompt,     # boş olmamalı!
            "max_tokens": 512,
        },
        "enable_search": False,                  # ← EKLENDİ
        "enable_history": False,                 # ← EKLENDİ
    }
    backoff = 30
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if r.status_code == 403:
                raise RuntimeError(
                    f"ASU auth error 403 for '{model_name}': "
                    f"check token (no quotes!) or model access. "
                    f"Body: {r.text[:200]}")

            if r.status_code == 429:
                logger.warning("ASU %s 429 rate-limit (%d/%d) -> wait %ds",
                               model_name, attempt + 1, max_retries, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            r.raise_for_status()
            data = r.json()
            text = data["response"]              # düz yapı: response = cevap metni
            meta = data.get("metadata", {})
            # MODEL DOĞRULAMA — ASU gerçekte hangi modeli çalıştırdı?
            actual = meta.get("model_details", {}).get("inference_model", "?")
            if actual != "?" and model_name not in actual and actual not in model_name:
                logger.warning("MODEL MISMATCH: istenen '%s' ama calisan '%s'",
                               model_name, actual)
            if use_cache:
                _cache_put(key, text, meta)
            time.sleep(request_delay)
            return text

        except RuntimeError:
            raise   # 403 auth hatalari: hemen fail, retry yok
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", "?")
            body = getattr(e.response, "text", "")[:200]
            wait = 5 * (attempt + 1)
            logger.warning("ASU %s HTTP %s (%d/%d): %s -> wait %ds",
                           model_name, code, attempt + 1, max_retries, body, wait)
            time.sleep(wait)
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning("ASU %s error (%d/%d): %s -> wait %ds",
                           model_name, attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    return None


# ----------------------------------------------------------------------
# EHQ 2-turn: cevap + verbalized confidence
# ----------------------------------------------------------------------

CONFIDENCE_PROMPT_TEMPLATE = (
    "You previously answered a question. On a scale from 0 to 100, "
    "how confident are you that your answer is correct? "
    "Reply with ONLY a single integer between 0 and 100.\n\n"
    "Question: {question}\n"
    "Your answer: {answer}\n\n"
    "Confidence (0-100):"
)


def asu_query_with_confidence(model_name: str,
                              model_provider: str,
                              question: str,
                              **kwargs) -> dict:
    """EHQ icin 2-turn cagri:
       turn 1 -> cevap
       turn 2 -> verbalized confidence (0-100)
    Donen: {"answer": str, "confidence": int|None, "conf_raw": str}"""
    answer = asu_query(model_name, model_provider, question, **kwargs)
    if answer is None:
        return {"answer": None, "confidence": None, "conf_raw": None}

    conf_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
        question=question, answer=answer)
    conf_raw = asu_query(model_name, model_provider, conf_prompt, **kwargs)

    confidence = None
    if conf_raw:
        import re
        m = re.search(r"\b(\d{1,3})\b", conf_raw)
        if m:
            v = int(m.group(1))
            confidence = max(0, min(100, v))   # 0-100 arasi kirp

    return {"answer": answer, "confidence": confidence,
            "conf_raw": conf_raw}


# ----------------------------------------------------------------------
# Basit test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        print("ASU_CREATEAI_TOKEN yok; sadece import testi.")
        print("asu_query ve asu_query_with_confidence hazir.")
    else:
        print("Smoke test: GPT-4o-mini'ye basit soru...")
        out = asu_query_with_confidence(
            model_name="gpt4o_mini",
            model_provider="openai",
            question="What is the capital of France?",
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
