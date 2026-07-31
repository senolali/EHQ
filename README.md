# EHQ — Epistemic Honesty Quotient

**Do LLMs Know What They Don't Know?**
Measuring Epistemic Honesty in Large Language Models.

EHQ-3000 expansion: 20 models x 3000 questions (FEQ + PCQ + HNQ + CCQ),
all via ASU CreateAI (18 models) + DeepSeek direct API (2 models).

Evaluation pipeline architecture (`models/`, `utils/`, `evaluation/`,
model registry, item-level checkpointing) mirrors
[senolali/RQEval](https://github.com/senolali/RQEval) for consistency
across the two frameworks.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in ASU_CREATEAI_TOKEN / DEEPSEEK_API_KEY
```

`.env` is loaded automatically (`main.py`, via `python-dotenv`); a
shell-level environment variable (`set ...`) also works, no `.env`
required.

## Dataset Generation

```
pcq_generator.py --full     # PCQ_dataset.json  (750, real Jun-Jul 2026 events)
hnq_generator.py --full     # HNQ_dataset.json  (750, hyper-niche true facts)
feq_generator.py --full     # FEQ_dataset.json  (750, fabricated entities)
ccq_generator.py --full     # CCQ_dataset.json  (750, redacted documents)
```

Together the four produce `EHQ-3000_dataset.json` (4 categories x 750,
20 subcategories x 150). These scripts still use `asu_client.py`
(generator model is always Mistral Large; dataset generation is a
separate concern from the evaluation pipeline -- see "Why are there
two separate ASU clients?" below).

## Evaluation Pipeline

```bash
# Full evaluation (20 models, 3000 questions)
python main.py --models all --categories all

# Single model, single category
python main.py --models "GPT-5-mini" --categories CCQ

# Quick test (50 questions)
python main.py --models "Claude-4.5-Haiku" --limit 50

# Dry run (no real API calls, exercises the whole pipeline with a deterministic MockModel)
python main.py --dry-run --limit 10

# Resuming an interrupted run: re-run the SAME command -- the
# item-level checkpoint (utils/checkpoint.py) automatically skips
# completed questions and only re-queries the missing ones.
python main.py --models all --categories all
```

## Project Structure

```
EHQ/
├── main.py                     # Main entry point (config-driven, model registry)
├── .env.example                 # ASU_CREATEAI_TOKEN / DEEPSEEK_API_KEY template
├── config/
│   └── config_ehq_20models.yaml # 20-model configuration (ASU + DeepSeek)
├── models/                      # Model provider layer (consistent with RQEval's architecture)
│   ├── base_model.py             # Abstract interface: generate(), generate_with_confidence(), in-memory cache
│   ├── asu_model.py               # ASU CreateAI wrapper (18 models)
│   ├── deepseek_model.py          # DeepSeek direct API wrapper (2 models)
│   └── mock_model.py              # Deterministic mock model for --dry-run
├── evaluation/
│   └── evaluator.py              # Orchestration: prompt -> 2-turn call -> classify -> score, checkpoint/resume
├── utils/                       # Infrastructure ported from RQEval
│   ├── checkpoint.py              # Item-level JSONL checkpoint/resume (fingerprinted)
│   ├── experiment_tracker.py      # Per-model result logging + triggers Excel/figure export
│   ├── logger.py                  # Centralized logging
│   └── reproducibility.py         # Seed control
├── framework/                   # EHQ-specific scoring engine (no RQEval equivalent)
│   ├── config_scoring.py          # EHQ weights, rubric, ABSTAIN/HEDGE patterns
│   ├── classifier.py              # Response classification (ABSTAIN/HEDGE/CONFIDENT) + correctness
│   ├── scorer.py                   # EHQ1, EHQ2, EHQ3, composite EHQ
│   ├── prompt_builder.py           # Builds the prompt per category (CCQ: document+question)
│   ├── exporter.py                 # Excel report (5 sheets)
│   └── visualizer.py               # Publication-ready figures (PDF+PNG, 300 DPI)
├── asu_client.py                # ASU client used for dataset GENERATION (used by the generator scripts)
├── verify_models.py             # Verifies all 20 model keys
├── {pcq,hnq,feq,ccq}_generator.py   # Dataset generation scripts
├── EHQ-3000_dataset.json        # 3000 questions (4 categories x 750)
├── outputs_ehq/                 # Run outputs (in .gitignore)
│   ├── checkpoints/<exp>/<model>.jsonl   # Persistent checkpoint, shared across reruns
│   └── <experiment_id>/                  # Per run: *_result.json, summary.json, xlsx, figures/
├── cache_asu/, cache_deepseek/  # SHA-256 disk cache for dataset generation (in .gitignore)
└── docs/EHQ_PROJECT_CONTEXT_v2.md
```

### Why are there two separate ASU clients?

`asu_client.py` (repo root) and `models/asu_model.py` are deliberately
separate: `asu_client.py` is for dataset GENERATION (`pcq_generator.py`
etc., generator model is always Mistral Large, with a SHA-256 disk
cache -- the rationale being "don't regenerate the same prompt
repeatedly"). `models/asu_model.py` is for EVALUATION (each of the 20
test models, with an in-memory cache + item-level checkpoint) -- a
different consistency/reproducibility requirement. They are not
dependent on each other and should not be conflated.

## EHQ Metrics

| Metric | Formula | Description |
|--------|---------|--------------|
| EHQ₁ | (ABSTAIN + HEDGE) / N | Epistemic Restraint Rate |
| EHQ₂ | 1 - (CONFIDENT_WRONG / N) | Hallucination Resistance |
| EHQ₃ | 1 - ECE(CONFIDENT subset) | Confidence-Accuracy Alignment |
| EHQ | 0.30·EHQ₁ + 0.45·EHQ₂ + 0.25·EHQ₃ | Composite score |

The methodology is exactly consistent with the classification/scoring
logic used in the EHQ v1 paper (arXiv:2605.24661, 7 models / EHQ-750)
(`framework/classifier.py`, `framework/scorer.py`); the only difference
is EHQ-3000's addition of a 4th category, CCQ (see the notes at the
top of `framework/config_scoring.py`).

## Reproducibility / Checkpointing

- **Item-level resume:** every completed question is written
  immediately to
  `outputs_ehq/checkpoints/<experiment_name>/<model>.jsonl` (flush +
  fsync). After a crash/interruption, re-running the same command only
  re-queries the missing questions.
- **Fingerprint validation:** the checkpoint file hashes the model
  parameters, category list, seed, and ALL scoring rules (rubric,
  ABSTAIN/HEDGE patterns, weights). If any of these change, the old
  checkpoint is automatically moved to `*.stale` and a fresh run
  starts -- no manual "protocol version" tracking needed.
- **Incremental saves:** after each model finishes, the Excel report
  and figures are regenerated with ALL models completed so far (not
  just at the end of the run) -- even if the run is killed mid-way,
  `outputs_ehq/<experiment_id>/` always holds the latest cumulative
  results.

## Outputs

- `outputs_ehq/<experiment_id>/<Model>_result.json` — full per-model result (including raw responses)
- `outputs_ehq/<experiment_id>/summary.json` — summary across all models
- `outputs_ehq/<experiment_id>/EHQ_3000_results.xlsx` — 5-sheet Excel report
- `outputs_ehq/<experiment_id>/figures/` — 5 figures (PDF + PNG, publication-ready)
