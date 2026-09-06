# Running Pilot and Full EHQ Studies

This runbook applies to framework 0.4.0 and protocol 1.1.4.

## 1. Install and initialize

```bash
python -m venv .venv
python -m pip install "ehq[analysis,reporting]"
ehq init EHQ-study
cd EHQ-study
ehq --version
```

Repository developers can instead run `python -m pip install -e ".[all,dev]"`.

## 2. Configure a route

Copy `.env.example` to `.env`, add the credential you use, and edit
`config/models.json`. Supported provider values are:

- `openai` for the OpenAI Responses API;
- `huggingface` for Hugging Face Inference Providers;
- `openai_compatible` for hosted or local `/chat/completions` APIs.

Keep `pcq_eligible` false until an applicable cutoff has been verified. See
`docs/model_registry.md` for final-study evidence fields.

## 3. Pre-spend checks

```bash
ehq validate data/releases/EHQ-3000.json \
  --expected-total 3000 \
  --expected-per-category 750 \
  --require-source-evidence \
  --require-pcq-temporal-novelty

ehq dry-run \
  --config config/smoke.json \
  --model openai-example \
  --allow-candidate \
  --run-id framework-smoke

ehq verify-run outputs/dry-run_framework-smoke
```

The offline score has no scientific interpretation. It checks configuration,
selection, classification, scoring, output, and integrity plumbing.

## 4. Pilot sequence

Run one item before spending on a larger sample:

```bash
ehq pilot --model openai-example --limit 1 --run-id health-openai-001
```

Then run common deterministic samples:

```bash
ehq pilot --model openai-example --limit 20 --run-id pilot-openai-n20-001
ehq pilot --model openai-example --limit 100 --run-id pilot-openai-n100-001
```

For multiple routes:

```bash
ehq pilot \
  --models "openai-example,huggingface-example" \
  --limit 100 \
  --run-id pilot-common-n100-001
```

Accept a pilot only after checking requested/resolved identity, technical
failure counts, confidence coverage, reasoning-token metadata when reported,
and `ehq verify-run`.

## 5. Full study

```bash
ehq full --model openai-example --run-id full-openai-001
```

Or:

```bash
ehq full \
  --models "openai-example,huggingface-example" \
  --run-id full-comparative-001
```

A full run usually makes two requests per eligible item. Estimate cost and
quota first. Unverified routes may run because this is useful for engineering,
but the manifest and preflight mark them non-publishable. Publication-grade
claims require the route and any PCQ cutoff evidence to pass the registry gate.

## 6. Progress and interruption

Readable progress is printed to stderr and includes completed items, valid
denominators, failures, throughput, and ETA. Add `--json-events` for JSON Lines.

Stop with `Ctrl+C`. Resume by repeating the identical command with `--resume`:

```bash
ehq full --model openai-example --run-id full-openai-001 --resume
```

Changing the dataset, route, configuration, prompt, selection, or protocol
changes the fingerprint and prevents unsafe checkpoint reuse.

## 7. Completion checks

```bash
ehq status outputs/real_full-openai-001
ehq verify-run outputs/real_full-openai-001
```

Before analysis, confirm that every intended route completed, exclusions and
missingness are explained, confidence coverage is adequate, the selection hash
is common across compared runs, and the artifact catalog is valid.

Never publish `.env`, raw restricted provider payloads, caches, checkpoints, or
outputs when provider terms or research governance prohibit doing so.
