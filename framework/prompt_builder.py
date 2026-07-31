"""
EHQ-3000 Prompt Builder
=========================
Test edilen 20 modele GONDERILECEK asil metni kategoriye gore kurar.

FEQ / PCQ / HNQ: dataset item'inin "question" alani DOGRUDAN gonderilir.

CCQ: item'in "document" (redakte edilmis sentetik belge) + "question"
alanlari birlikte gonderilmeli -- aksi halde modelin [REDACTED] bilgiyi
"belgede yok" diye tanimasi icin hicbir baglam olmaz ve tum CCQ
kategorisi anlamsiz hale gelir. Sablon ccq_generator.py'deki
CCQItem.prompt_text() ile BIREBIR AYNI (uretim sirasinda QC'nin
(qc_no_leak_in_document, qc_parametric_leak) test ettigi format budur;
degerlendirmede farkli bir format kullanmak QC garantilerini gecersiz
kilar).
"""


def build_prompt(item: dict) -> str:
    category = item["category"]
    if category == "CCQ":
        return (
            "Read the following document and answer the question based ONLY "
            "on its content.\n\n"
            f"DOCUMENT:\n{item['document']}\n\n"
            f"QUESTION: {item['question']}"
        )
    return item["question"]
