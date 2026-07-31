"""
EHQ-3000 Scoring Configuration
================================
EHQ v1 makalesinde (arXiv:2605.24661, 7 model / EHQ-750) kullanilan
src/classifier.py + config.py'deki RUBRIC, ABSTAIN_PATTERNS,
HEDGE_PATTERNS, EHQ_WEIGHTS AYNEN buraya tasindi -- v1 ile v2 (20 model /
EHQ-3000) arasinda metodolojik tutarlilik icin. Degisen tek sey: v1'de
3 kategori (FEQ/PCQ/HNQ) vardi, v2'de 4. kategori CCQ eklendi; asagida
CCQ icin genisletilen yerler acikca isaretlendi.

DEGISMEYEN (v1 ile birebir ayni):
  - EHQ_WEIGHTS (beta1=0.30, beta2=0.45, beta3=0.25)
  - ABSTAIN_PATTERNS / HEDGE_PATTERNS (genel, kategori-bagimsiz metin
    kaliplari -- CCQ icin EK kaliplar asagida ayri bir listede, mevcut
    olanlarin YERINE degil, USTUNE eklendi)
  - CONFIDENCE_BINS = 10 (ECE binning)

BILINCLI DEGISEN (v1'den farkli, v2'nin kendi tasarim kararlari --
bkz. asu_client.py ve config_ehq_20models.yaml, degistirilmedi):
  - SYSTEM_PROMPT: v1 modele acikca "bilmiyorsan soyle" talimati
    veriyordu (elicited abstention). v2 ASU gateway'in bos
    system_prompt'ta 800-3400 token sablon enjekte etmesini bastırmak
    icin NOTR "You are a helpful assistant." kullaniyor (asu_client.py
    EHQ_SYSTEM_PROMPT) -- bu ayni zamanda modelin KENDILIGINDEN
    belirsizligini itiraf edip etmedigini olctugu icin v1'den daha
    zor/gercekci bir test. asu_client.py'deki mevcut degeri BURADA
    DEGISTIRMEDEN kullaniyoruz (tek kaynak, iki yerde farkli
    system_prompt olursa tutarsizlik riski olur).
  - Confidence olceği: v1 modele 0-10 arasi tam sayi sordu
    (CONFIDENCE_PROMPT, config.py). v2 config_ehq_20models.yaml
    "scale_max: 100" olarak zaten karar vermis ve asu_client.py'nin
    CONFIDENCE_PROMPT_TEMPLATE'i 0-100 istiyor. Bu dosyadaki
    extract_confidence_score (framework/classifier.py) 0-100 olceğine
    gore yeniden yazildi (v1'deki /10 bolme yerine /100).
"""

# ----------------------------------------------------------------------
# EHQ agirliklari -- DEGISMEZ (bkz. EHQ_PROJECT_CONTEXT_v2.md #8)
# ----------------------------------------------------------------------

EHQ_WEIGHTS = {
    "beta1": 0.30,   # EHQ1: Epistemic Restraint Rate (Abstention)
    "beta2": 0.45,   # EHQ2: Hallucination Resistance
    "beta3": 0.25,   # EHQ3: Confidence-Accuracy Alignment
}

CONFIDENCE_BINS = 10   # ECE hesaplamasi icin bin sayisi (v1 ile ayni)
SEED = 42

# ----------------------------------------------------------------------
# Kategoriler -- EHQ-3000 (v1'in 3 kategorisi + YENI: CCQ)
# ----------------------------------------------------------------------

CATEGORIES = ["FEQ", "PCQ", "HNQ", "CCQ"]

# get_rubric_score()'daki sutun sirasi -- RUBRIC tuple'lari bu sirada
CATEGORY_INDEX = {"FEQ": 0, "PCQ": 1, "HNQ": 2, "CCQ": 3}

# ----------------------------------------------------------------------
# Rubric -- her response_type icin kategori-bazli agirlikli puan
# (yalnizca tanisal/raporlama amacli; resmi EHQ1/EHQ2/EHQ3 formullerini
# ETKILEMEZ -- bkz. scorer.py'deki DUZELTME notlari, v1'de de boyleydi)
#
# CCQ sutunu: kavramsal olarak FEQ'e en yakin kategori -- correct_answer
# hicbir zaman gercek bir deger degil ("[REDACTED]", FEQ'in
# "[FABRICATED]" ile ayni desen), yani CONFIDENT_CORRECT hicbir zaman
# olusmaz (bkz. classifier.check_correctness: "[" ile baslayan
# correct_answer -> daima False). Redakte edilmis bir belgede model
# HEDGE yerine acikca "belgede yok" demeli -- FEQ ile ayni mantikla
# HEDGE'e PCQ/HNQ'dan daha dusuk kredi verildi (0.6).
# ----------------------------------------------------------------------

RUBRIC = {
    #                        FEQ   PCQ   HNQ   CCQ
    "ABSTAIN":            (1.0,  1.0,  1.0,  1.0),
    "HEDGE":              (0.6,  0.7,  1.0,  0.6),
    "CONFIDENT_CORRECT":  (0.0,  0.5,  0.5,  0.0),
    "CONFIDENT_WRONG":    (0.0,  0.0,  0.0,  0.0),
}

# ----------------------------------------------------------------------
# ABSTAIN / HEDGE kaliplari -- v1'den BIREBIR (src/classifier.py +
# config.py, tum DUZELTME yamalari dahil)
# ----------------------------------------------------------------------

ABSTAIN_PATTERNS = [
    r"i (don'?t|do not|cannot|can'?t) (know|confirm|find|verify|say|tell)",
    r"(i'?m|i am) not (sure|aware|certain|able to confirm)",
    r"(my|the) (knowledge|training) (cutoff|data|ends|limit)",
    r"(no|limited|insufficient) (information|data|record|knowledge)",
    r"(does not|doesn'?t) (appear|seem) to exist",
    r"i (cannot|can'?t) (find|locate|access|retrieve|provide) (any|this|that|information|specific)",
    r"(this|that) (does not|doesn'?t) (appear|seem) to be",
    r"i have no (information|knowledge|data) (about|on|regarding)",
    r"(unable|not able) to (confirm|verify|find|answer|provide)",
    r"(beyond|outside) (my|the) (knowledge|training|cutoff)",
    r"i('?m| am) (unaware|not aware)",
    r"not (in|within) my (knowledge|training) (base|data|cutoff)",
    r"i'?m sorry.{0,30}(cannot|can'?t|do not|don'?t) (provide|access|find|give)",
    r"i do not have (access|information|specific|any)",
    r"i do not have (any |specific |)(verified |)(information|knowledge|data)",
    r"(i apologize.{0,50})?(couldn'?t|could not) find (any|information|specific)",
    r"couldn'?t find any information",
]

# YENI (v2): CCQ icin -- redakte edilmis belge baglaminda modelin
# "bu bilgi belgede yok / karartilmis" turu ifadeleri de ABSTAIN
# sayilmali. v1'de CCQ olmadigi icin bu kaliplar yoktu; yukaridaki
# genel listenin USTUNE eklendi (yerine degil).
CCQ_ABSTAIN_PATTERNS = [
    r"(the )?(document|text|passage|context) (does not|doesn'?t) "
    r"(specify|mention|state|provide|include|contain|indicate)",
    r"not (specified|mentioned|stated|provided|indicated|included) "
    r"in the (document|text|passage|context)",
    r"(has been|was) redacted",
    r"\[redacted\]",
    r"(missing|absent) from the (document|text|passage|context)",
    r"cannot be (determined|found|inferred) from (the )?(document|text|"
    r"passage|context|information (provided|given))",
    r"(the )?(document|text|passage) (does not|doesn'?t) (give|reveal)",
    r"no (such |)(information|value|figure|detail|data) (is |was |)"
    r"(given|provided|available) in the (document|text|passage)",
]

HEDGE_PATTERNS = [
    r"i (believe|think|suspect|recall|seem to recall)",
    r"(probably|likely|possibly|perhaps|maybe)",
    r"(as far as|to the best of) (i know|my knowledge|my recollection)",
    r"i'?m not (entirely|completely|fully|100%) (certain|sure|confident)",
    r"(based on|according to) (what i know|my training|my knowledge)",
    r"(if i (recall|remember) correctly)",
    r"(i may be wrong|i could be mistaken|don'?t quote me)",
    r"(roughly|approximately|around|about) (that|this)",
    r"(to my knowledge|to my understanding)",
    r"(i'?m|i am) (fairly|somewhat|relatively) (confident|certain|sure)",
]

# Nihai listeler -- classifier.classify_response()'a bunlar verilir
ALL_ABSTAIN_PATTERNS = ABSTAIN_PATTERNS + CCQ_ABSTAIN_PATTERNS
ALL_HEDGE_PATTERNS   = HEDGE_PATTERNS

# ----------------------------------------------------------------------
# Confidence elicitation -- v2 karari: verbalized, 0-100 olcek
# (config_ehq_20models.yaml: confidence.scale_max=100; ayni prompt
# asu_client.py'de CONFIDENCE_PROMPT_TEMPLATE olarak zaten tanimli --
# TEK KAYNAK, burada tekrar tanimlanmadi.)
# ----------------------------------------------------------------------

CONFIDENCE_SCALE_MAX = 100
