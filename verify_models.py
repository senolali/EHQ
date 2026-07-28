"""
ASU Model Key Dogrulama
========================
20 modelin her birine tek soru sorar ve ASU'nun GERCEKTE hangi modeli
calistirdigini (metadata.model_details.inference_model) raporlar.

Amac: 'gpt4o_mini' isteyince ASU 'gpt4_1-mini' calistiriyor mu gibi
uyusmazliklari YAKALAMAK. Deneyin gecerliligi buna bagli.

Calistir: python verify_models.py
"""
import os
import json
import time
import requests

BASE_URL = os.environ.get("ASU_BASE_URL", "https://api-main.aiml.asu.edu")
TOKEN = os.environ.get("ASU_CREATEAI_TOKEN")

# config_ehq_20models.yaml ile ayni 18 ASU modeli
# (DeepSeek ASU'da degil, ayri test edilir)
ASU_MODELS = [
    ("Claude-3-Haiku",     "claude3_haiku",        "aws"),
    ("Claude-4.5-Haiku",   "claude4_5_haiku",      "aws"),
    ("Claude-4-Sonnet",    "claude4_sonnet",       "aws"),
    ("Claude-4.8-Opus",    "claude4_8_opus",       "aws"),
    ("GPT-4o-mini",        "gpt4o_mini",           "openai"),
    ("GPT-5-mini",         "gpt5_mini",            "openai"),
    ("GPT-4o",             "gpt4o",                "openai"),
    ("GPT-5.5",            "gpt5_5",               "openai"),
    ("Gemini-2.5-Flash",   "geminiflash2_5",       "gcp-deepmind"),
    ("Gemini-3.5-Flash",   "geminiflash3_5",       "gcp-deepmind"),
    ("Gemini-2.5-Pro",     "geminipro2_5",         "gcp-deepmind"),
    ("Gemini-3.1-Pro",     "geminipro3_1",         "gcp-deepmind"),
    ("LLaMA-3-70B",        "llama3-70b",           "aws"),
    ("LLaMA-4-Maverick",   "llama4_maverick-17b",  "aws"),
    ("GPT-OSS-20B",        "gpt-oss-20b",          "aws"),
    ("Gemma-3-4B",         "gemma3_4b_it",         "gcp-deepmind"),
    ("Nova-Micro",         "nova-micro",           "aws"),
    ("Nova-Pro",           "nova-pro",             "aws"),
]


def probe(model_name, provider):
    headers = {"Authorization": f"Bearer {TOKEN}",
               "Content-Type": "application/json"}
    payload = {
        "action": "query",
        "model_provider": provider,
        "model_name": model_name,
        "query": "Reply with the single word: OK",
        "model_params": {"temperature": 0.0,
                         "system_prompt": "You are a helpful assistant."},
        "response_format": {"type": "json"},
    }
    try:
        r = requests.post(BASE_URL.rstrip("/") + "/query",
                          headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        meta = data.get("metadata", {})
        md = meta.get("model_details", {})
        return {
            "ok": True,
            "requested": model_name,
            "actual": md.get("inference_model", "?"),
            "provider": md.get("inference_provider", "?"),
            "answer": str(data.get("response", ""))[:40],
        }
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", "")[:200]
        return {"ok": False, "requested": model_name, "error": str(e),
                "body": body}


def main():
    if not TOKEN:
        print("ASU_CREATEAI_TOKEN yok!")
        return
    print(f"{'Model':<20}{'Requested key':<22}{'Actual':<24}{'Durum'}")
    print("-" * 78)
    results = []
    for friendly, key, provider in ASU_MODELS:
        res = probe(key, provider)
        results.append((friendly, res))
        if res["ok"]:
            actual = res["actual"]
            # eslesme kontrolu (gevsek: key parcalari actual'da mi)
            match = (key.replace("_", "").replace("-", "").lower()
                     in actual.replace("_", "").replace("-", "").replace(".", "").lower()) \
                    or actual == "?"
            flag = "OK" if match else f"!!! MISMATCH"
            print(f"{friendly:<20}{key:<22}{actual:<24}{flag}")
        else:
            print(f"{friendly:<20}{key:<22}{'ERROR':<24}{res.get('error','')[:30]}")
        time.sleep(2)   # rate limit'e saygi

    # Ozet
    print("\n" + "=" * 78)
    mismatches = [f for f, r in results
                  if r["ok"] and r["actual"] != "?"
                  and key not in r["actual"]]
    errors = [f for f, r in results if not r["ok"]]
    print(f"Toplam: {len(results)} | Hatali: {len(errors)}")
    # JSON kaydet (detay inceleme icin)
    with open("model_verification.json", "w", encoding="utf-8") as fp:
        json.dump([{"friendly": f, **r} for f, r in results],
                  fp, indent=2, ensure_ascii=False)
    print("Detay -> model_verification.json")


if __name__ == "__main__":
    main()
