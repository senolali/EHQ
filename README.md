# EHQ — Epistemic Honesty Quotient

**Do LLMs Know What They Don't Know?**
Measuring Epistemic Honesty in Large Language Models.

EHQ-3000 expansion: 20 models x 3000 questions (FEQ + PCQ + HNQ + CCQ),
all via ASU CreateAI (18 models) + DeepSeek direct API (2 models).

Evaluation pipeline architecture (`models/`, `utils/`, `evaluation/`,
model registry, item-level checkpointing) mirrors
[senolali/RQEval](https://github.com/senolali/RQEval) for consistency
across the two frameworks.

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env      # sonra ASU_CREATEAI_TOKEN / DEEPSEEK_API_KEY doldur
```

`.env` otomatik yüklenir (`main.py`, `python-dotenv`); `set` ile shell
ortam değişkeni de kullanılabilir, `.env`'e gerek yok.

## Dataset Üretimi

```
pcq_generator.py --full     # PCQ_dataset.json  (750, gercek Haz-Tem 2026 olaylari)
hnq_generator.py --full     # HNQ_dataset.json  (750, asiri nis gercek bilgiler)
feq_generator.py --full     # FEQ_dataset.json  (750, uydurma varliklar)
ccq_generator.py --full     # CCQ_dataset.json  (750, redakte belgeler)
```

Dördü birlikte `EHQ-3000_dataset.json` dosyasını oluşturur (4 kategori x
750, 20 alt kategori x 150). Bu scriptler hâlâ `asu_client.py`'yi
kullanıyor (üretici model Mistral Large, dataset üretimi değerlendirme
pipeline'ından ayrı bir concern — bkz. aşağıdaki "Neden iki ayrı ASU
istemcisi var?").

## Değerlendirme Pipeline'ı

```bash
# Tam degerlendirme (20 model, 3000 soru)
python main.py --models all --categories all

# Tek model, tek kategori
python main.py --models "GPT-5-mini" --categories CCQ

# Hizli test (50 soru)
python main.py --models "Claude-4.5-Haiku" --limit 50

# Dry run (gercek API cagrisi yok, deterministic MockModel ile tum boru hattini test et)
python main.py --dry-run --limit 10

# Yarida kesilen bir kosum: AYNI komutu tekrar calistir -- item-level
# checkpoint (utils/checkpoint.py) tamamlanan sorulari otomatik atlar,
# sadece eksik kalanlari yeniden sorar.
python main.py --models all --categories all
```

## Proje Yapısı

```
EHQ/
├── main.py                     # Ana calistirma scripti (config-driven, model registry)
├── .env.example                 # ASU_CREATEAI_TOKEN / DEEPSEEK_API_KEY sablonu
├── config/
│   └── config_ehq_20models.yaml # 20 model konfigurasyonu (ASU + DeepSeek)
├── models/                      # Model saglayici katmani (RQEval mimarisiyle tutarli)
│   ├── base_model.py             # Soyut arayuz: generate(), generate_with_confidence(), in-memory cache
│   ├── asu_model.py               # ASU CreateAI wrapper (18 model)
│   ├── deepseek_model.py          # DeepSeek dogrudan API wrapper (2 model)
│   └── mock_model.py              # --dry-run icin deterministic sahte model
├── evaluation/
│   └── evaluator.py              # Orkestrasyon: prompt->2-turn cagri->siniflandirma->skor, checkpoint/resume
├── utils/                       # RQEval'dan portlanan altyapi
│   ├── checkpoint.py              # Item-level JSONL checkpoint/resume (fingerprint'li)
│   ├── experiment_tracker.py      # Per-model kayit + Excel/figur tetikleme
│   ├── logger.py                  # Merkezi loglama
│   └── reproducibility.py         # Seed kontrolu
├── framework/                   # EHQ'ye ozgu skorlama motoru (RQEval'da karsiligi yok)
│   ├── config_scoring.py          # EHQ agirliklari, rubric, ABSTAIN/HEDGE kaliplari
│   ├── classifier.py              # Yanit siniflandirma (ABSTAIN/HEDGE/CONFIDENT) + dogruluk
│   ├── scorer.py                   # EHQ1, EHQ2, EHQ3, birlesik EHQ
│   ├── prompt_builder.py           # Kategoriye gore prompt kurma (CCQ: belge+soru)
│   ├── exporter.py                 # Excel raporu (5 sekme)
│   └── visualizer.py               # Makale gorselleri (PDF+PNG, 300 DPI)
├── asu_client.py                # Dataset URETIMI icin ASU istemcisi (generator scriptleri kullanir)
├── verify_models.py             # 20 model key dogrulama
├── {pcq,hnq,feq,ccq}_generator.py   # Dataset uretim scriptleri
├── EHQ-3000_dataset.json        # 3000 soru (4 kategori x 750)
├── outputs_ehq/                 # Calisma ciktilari (gitignore'da)
│   ├── checkpoints/<exp>/<model>.jsonl   # Kalici, tekrar-kosumlar arasi paylasilan checkpoint
│   └── <experiment_id>/                  # Her kosum icin: *_result.json, summary.json, xlsx, figures/
├── cache_asu/, cache_deepseek/  # Dataset uretimi icin SHA-256 disk cache (gitignore'da)
└── docs/EHQ_PROJECT_CONTEXT_v2.md
```

### Neden iki ayrı ASU istemcisi var?

`asu_client.py` (kök dizinde) ve `models/asu_model.py` kasıtlı olarak
ayrı: `asu_client.py` dataset ÜRETİMİ için (`pcq_generator.py` vb.,
üretici model her zaman Mistral Large, SHA-256 disk cache ile), gerekçesi
"aynı prompt tekrar tekrar üretilmesin". `models/asu_model.py`
DEĞERLENDİRME için (20 test modelinin her biri, in-memory cache +
item-level checkpoint ile) — farklı bir tutarlılık/tekrarlanabilirlik
gereksinimi. İkisi birbirine bağımlı değil, karıştırılmamalı.

## EHQ Metrikleri

| Metrik | Formül | Açıklama |
|--------|--------|----------|
| EHQ₁ | (ABSTAIN + HEDGE) / N | Epistemic Restraint Rate |
| EHQ₂ | 1 - (CONFIDENT_WRONG / N) | Hallucination Resistance |
| EHQ₃ | 1 - ECE(CONFIDENT subset) | Confidence-Accuracy Alignment |
| EHQ | 0.30·EHQ₁ + 0.45·EHQ₂ + 0.25·EHQ₃ | Composite score |

Metodoloji, EHQ v1 makalesinde (arXiv:2605.24661, 7 model / EHQ-750)
kullanılan sınıflandırma/skorlama mantığıyla (`framework/classifier.py`,
`framework/scorer.py`) birebir tutarlıdır; tek fark EHQ-3000'in 4.
kategorisi CCQ'nun eklenmiş olmasıdır (bkz. `framework/config_scoring.py`
üst kısmındaki notlar).

## Tekrarlanabilirlik / Checkpoint

- **Item-level resume:** her tamamlanan soru anında
  `outputs_ehq/checkpoints/<experiment_name>/<model>.jsonl`'a yazılır
  (flush + fsync). Çökme/kesinti sonrası aynı komutu tekrar çalıştırmak
  sadece eksik kalan soruları sorar.
- **Fingerprint doğrulama:** checkpoint dosyası model parametrelerini,
  kategori listesini, seed'i ve TÜM skorlama kurallarını (rubric,
  ABSTAIN/HEDGE kalıpları, ağırlıklar) hash'ler. Bunlardan biri
  değişirse eski checkpoint otomatik `*.stale`'e taşınır ve sıfırdan
  başlanır — elle "protokol versiyonu" takip etmeye gerek yok.
- **Kademeli kayıt:** her model bitince Excel + figürler o ana kadarki
  TÜM modellerle güncellenir (sadece kosum sonunda değil) — yarıda
  kesilse bile `outputs_ehq/<experiment_id>/` her zaman en güncel
  kümülatif sonucu içerir.

## Çıktılar

- `outputs_ehq/<experiment_id>/<Model>_result.json` — model bazlı tam sonuç (ham yanıtlar dahil)
- `outputs_ehq/<experiment_id>/summary.json` — tüm modellerin özeti
- `outputs_ehq/<experiment_id>/EHQ_3000_results.xlsx` — 5 sekmeli Excel raporu
- `outputs_ehq/<experiment_id>/figures/` — 5 figür (PDF + PNG, makale için hazır)
