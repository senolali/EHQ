"""
EHQ-3000 Model Client Router
==============================
config_ehq_20models.yaml'daki her modelin "type" alanina gore dogru
API katmanina yonlendirir:
  type: "asu"       -> asu_client.py (ASU CreateAI, 18 model, ucretsiz)
  type: "deepseek"  -> DeepSeek dogrudan API (2 model, ASU'da yok)

Her iki yol da AYNI 2-turn EHQ protokolunu izler:
  Turn 1 -> soruyu cevapla
  Turn 2 -> "0-100 arasi guven puanin nedir?" (asu_client.py
            CONFIDENCE_PROMPT_TEMPLATE, TEK KAYNAK -- burada
            tekrar tanimlanmadi)

Ayni sistem prompt'u (asu_client.EHQ_SYSTEM_PROMPT, "You are a helpful
assistant.") TUM 20 modele gider -- tutarli karsilastirma icin
(EHQ_PROJECT_CONTEXT_v2.md #3.5).
"""

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asu_client import (   # noqa: E402
    asu_query_with_confidence,
    EHQ_SYSTEM_PROMPT,
    CONFIDENCE_PROMPT_TEMPLATE,
)

logger = logging.getLogger("ehq_model_client")

DEEPSEEK_CACHE_DIR = Path("cache_deepseek")
DEEPSEEK_CACHE_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# DeepSeek dogrudan API (OpenAI-uyumlu istemci)
# ----------------------------------------------------------------------

def _ds_cache_key(model_id: str, prompt: str) -> str:
    raw = f"{model_id}||{EHQ_SYSTEM_PROMPT}||{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ds_cache_get(key: str) -> Optional[str]:
    f = DEEPSEEK_CACHE_DIR / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))["response"]
        except Exception:
            return None
    return None


def _ds_cache_put(key: str, response: str) -> None:
    f = DEEPSEEK_CACHE_DIR / f"{key}.json"
    f.write_text(json.dumps({"response": response}, ensure_ascii=False, indent=2),
                encoding="utf-8")


def _deepseek_query(params: dict, prompt: str, temperature: float,
                    use_cache: bool = True) -> Optional[str]:
    model_id = params["model_id"]
    key = _ds_cache_key(model_id, prompt)
    if use_cache:
        cached = _ds_cache_get(key)
        if cached is not None:
            return cached

    api_key_env = params.get("api_key_env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} ortam degiskeni tanimli degil.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=params.get("base_url", "https://api.deepseek.com"))

    max_retries = params.get("max_retries", 5)
    max_tokens = params.get("max_tokens", 512)
    request_delay = params.get("request_delay", 1.0)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": EHQ_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            if use_cache:
                _ds_cache_put(key, text)
            time.sleep(request_delay)
            return text
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning("DeepSeek %s error (%d/%d): %s -> wait %ds",
                           model_id, attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    logger.error("DeepSeek %s: tum denemeler basarisiz", model_id)
    return None


def _deepseek_query_with_confidence(params: dict, question: str,
                                    temperature: float) -> dict:
    answer = _deepseek_query(params, question, temperature)
    if answer is None:
        return {"answer": None, "conf_raw": None}
    conf_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    conf_raw = _deepseek_query(params, conf_prompt, temperature)
    return {"answer": answer, "conf_raw": conf_raw}


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

def query_with_confidence(model_cfg: dict, question: str) -> dict:
    """model_cfg: config_ehq_20models.yaml'daki tek bir model girdisi
    ({"name":..., "type":..., "params": {...}}).
    Doner: {"answer": str|None, "conf_raw": str|None}
    (confidence sayisal degerini classifier.extract_confidence_score
    conf_raw metninden AYRICA cikarir -- asu_query_with_confidence'in
    kendi zayif regex-tabanli "confidence" alani KULLANILMAZ, bkz.
    run_pipeline.py)."""
    mtype = model_cfg["type"]
    params = model_cfg["params"]
    temperature = params.get("temperature", 0.7)

    if mtype == "asu":
        out = asu_query_with_confidence(
            model_name=params["model_name"],
            model_provider=params["model_provider"],
            question=question,
            temperature=temperature,
            max_retries=params.get("max_retries", 5),
            timeout=params.get("timeout", 120),
            request_delay=params.get("request_delay", 1.0),
            use_cache=True,
        )
        return {"answer": out["answer"], "conf_raw": out["conf_raw"]}

    if mtype == "deepseek":
        return _deepseek_query_with_confidence(params, question, temperature)

    raise ValueError(f"Bilinmeyen model type: '{mtype}' (config_ehq_20models.yaml)")
