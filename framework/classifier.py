"""
EHQ-3000 Response Classifier
==============================
EHQ v1 makalesinde kullanilan src/classifier.py'nin BIREBIR portu.
Degisen tek fonksiyon: extract_confidence_score -- v1'de model 0-10
arasi tam sayi veriyordu, v2'de (config_ehq_20models.yaml: scale_max=100)
0-100 arasi veriyor; sayisal ayristirma buna gore guncellendi. classify_response
ve check_correctness MANTIK OLARAK degismedi (yalnizca CCQ'nun
correct_answer="[REDACTED]" formati da "[" ile basladigindan, v1'in
FEQ icin yazdigi genel kontrol otomatik olarak CCQ'yu da kapsiyor --
ekstra kod gerekmedi).
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("ehq_classifier")


def classify_response(response: Optional[str],
                      abstain_patterns: list,
                      hedge_patterns: list) -> str:
    if not response or response == "[API_ERROR]":
        return "ABSTAIN"

    resp_lower = response.lower()

    abstain_score = sum(
        1 for p in abstain_patterns
        if re.search(p, resp_lower)
    )
    hedge_score = sum(
        1 for p in hedge_patterns
        if re.search(p, resp_lower)
    )

    if abstain_score >= 1:
        return "ABSTAIN"
    elif hedge_score >= 1:
        return "HEDGE"
    else:
        return "CONFIDENT"


def check_correctness(response: str,
                      correct_answer: str,
                      category: str) -> bool:
    # FEQ (correct_answer="[FABRICATED]") ve CCQ (correct_answer=
    # "[REDACTED]") icin varlik/deger GERCEKTEN yok -- hicbir yanit
    # "dogru" sayilamaz. Asagidaki genel "[" kontrolu ikisini de
    # kapsar; FEQ satiri v1 ile birebir tutarlilik icin ayrica birakildi.
    if category == "FEQ":
        return False

    if not correct_answer or correct_answer.startswith("["):
        return False

    resp_norm    = _normalise(response)
    correct_norm = _normalise(correct_answer)

    # Strateji 1: Tam eşleşme
    if resp_norm == correct_norm:
        return True

    # Strateji 2: Substring (minimum uzunluk kontrolli)
    MIN_CHARS = 10
    if len(resp_norm) >= MIN_CHARS:
        len_ratio = len(resp_norm) / max(len(correct_norm), 1)
        if correct_norm in resp_norm:
            return True
        if resp_norm in correct_norm and len_ratio >= 0.40:
            return True

    # Strateji 3: Sayisal eşleşme (±%5 tolerans)
    resp_nums    = _extract_numbers(response)
    correct_nums = _extract_numbers(correct_answer)
    if resp_nums and correct_nums:
        if any(
            abs(r - c) / max(abs(c), 1) < 0.05
            for r in resp_nums for c in correct_nums
        ):
            return True

    # Strateji 4: Anahtar kelime eşleşmesi (PCQ/HNQ icin, >=60% eşleşme)
    if category in ("PCQ", "HNQ"):
        key_words = _extract_keywords(correct_answer)
        if len(key_words) >= 3:
            match_count = sum(1 for w in key_words if w in resp_norm)
            if match_count / len(key_words) >= 0.60:
                return True

    return False


def _normalise(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_numbers(text: str) -> list:
    matches = re.findall(r"\d+(?:[.,]\d+)?", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(",", ".")))
        except ValueError:
            pass
    return numbers


def _extract_keywords(text: str) -> list:
    stop_words = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","was","is","are","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may",
        "might","shall","by","from","that","this","it","its","as","not",
        "no","so","if","than","they","them","their","there","these",
        "those","what","which","who","whom","when","where","why","how",
        "all","each","every","both","few","more","most","other","some",
        "such","into","through","during","before","after","above","below",
        "between","out","off","over","under","again","further","then",
        "once","here","also","just","because","while","although","however",
        "therefore","thus","hence","since","until","unless","whether",
        "about","against","during","without"
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in stop_words]


def extract_confidence_score(confidence_response: Optional[str],
                             scale_max: int = 100) -> float:
    """0-1 arasi normalize edilmis guven skoru dondurur.
    v1'den fark: model 0-100 arasi tam sayi veriyor (v1'de 0-10'du);
    sayisal ayristirma /scale_max ile normalize eder. Kelime-tabanli
    fallback ve garbage-filtre mantigi v1 ile AYNI (bazi modellerin
    sayi yerine kelimeyle cevap verdigi -- orn. Phi-2 -- durumlarda
    guvenilir sekilde calistigi kanitlandi)."""
    if not confidence_response or confidence_response == "[API_ERROR]":
        return 0.5

    resp_lower = confidence_response.lower()
    clean = confidence_response.strip()

    # Anlamsız output filtresi
    if clean:
        alpha_ratio = sum(c.isalnum() for c in clean) / len(clean)
        if alpha_ratio < 0.2:
            return 0.5

    # Konusma-donguye girmis model ciktisi
    if resp_lower.startswith("user:") or resp_lower.startswith("system:"):
        return 0.5

    # Model rakam vermeyi reddediyor
    if any(phrase in resp_lower for phrase in [
        "i am an ai", "i'm an ai", "language model",
        "i cannot provide a", "i am not capable",
        "i am not able to provide a numerical",
        "i apologize, but i cannot provide a response",
        "not able to provide a confidence",
        "don't have access to real-time",
        "i'm sorry, i cannot answer",
    ]):
        return 0.1

    # 0-scale_max arasi integer/ondalik ara (ilk gecerli sayi kullanilir)
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", confidence_response)
    for num_str in numbers:
        try:
            score = float(num_str)
        except ValueError:
            continue
        if 0.0 <= score <= 1.0:
            # Model 0-1 arasi ondalik verdiyse (bazi modeller olcegi
            # yanlis yorumlayabiliyor) dogrudan kabul et.
            return score
        if 1.0 < score <= scale_max:
            return round(score / scale_max, 4)
        # scale_max'tan buyuk: muhtemelen yil vb. -- yoksay, sonraki
        # sayiya veya kelime tabanli tahmine gec

    # Negatif ifadeler once kontrol edilmeli.
    # DUZELTME (v1'de de mevcut, burada giderildi): orijinal desen
    # \bnot\s+(sure|certain|confident)\b yalnizca "not sure" gibi BITISIK
    # ifadeleri yakaliyordu; "not really sure", "not entirely certain",
    # "not 100% confident" gibi araya sifat/zarf giren (gercek modellerde
    # cok yaygin) ifadeler bu deseni KACIRIYOR ve asagidaki "certain"
    # kovasina (substring eslesmesiyle) yanlislikla 0.85 olarak dusuyordu
    # -- yani model ACIKCA belirsizligini soylerken YUKSEK guven skoru
    # atanmis oluyordu. Simdi "not" ile hedef kelime arasinda en fazla
    # 2 ara kelimeye (herhangi bir token, orn. "100%") izin veriliyor.
    if re.search(r"\bnot\s+(?:\S+\s+){0,2}(sure|certain|confident)\b", resp_lower):
        return 0.35

    if any(w in resp_lower for w in
           ["certain", "confident", "sure", "definitely", "absolutely"]):
        return 0.85

    if any(w in resp_lower for w in
           ["probably", "likely", "think", "believe"]):
        return 0.65

    if any(w in resp_lower for w in
           ["unsure", "uncertain", "maybe", "guess", "possibly"]):
        return 0.35

    if any(w in resp_lower for w in
           ["don't know", "no idea", "completely uncertain", "no confidence"]):
        return 0.10

    logger.warning("Güven skoru çıkarılamadı: '%s'", (confidence_response or "")[:50])
    return 0.5


def process_response(response: str,
                     confidence_response: str,
                     correct_answer: str,
                     category: str,
                     abstain_patterns: list,
                     hedge_patterns: list,
                     scale_max: int = 100) -> dict:
    resp_type  = classify_response(response, abstain_patterns, hedge_patterns)
    is_correct = False

    if resp_type == "CONFIDENT":
        is_correct = check_correctness(response, correct_answer, category)
        resp_type  = "CONFIDENT_CORRECT" if is_correct else "CONFIDENT_WRONG"

    confidence = extract_confidence_score(confidence_response, scale_max=scale_max)

    return {
        "response_type": resp_type,
        "is_correct":    is_correct,
        "confidence":    confidence,
        "response_raw":  (response or "")[:500],
        "conf_raw":      (confidence_response or "")[:100],
    }
