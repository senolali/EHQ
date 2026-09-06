# EHQ Framework Architecture

**Framework:** 0.4.0

**Protocol:** 1.1.4

**Normative scoring change from 0.3.2:** none

## Design boundary

The framework implements the estimands in `docs/protocol.md`. Transport and
provider behavior cannot redefine those estimands. In particular:

- routing, provider, empty-response, safety-gateway, and transport failures are
  technical exclusions rather than abstentions;
- positive provider-reported reasoning or thinking-token use under the frozen
  non-reasoning condition is a terminal protocol failure;
- EHQ3 uses valid confidence values from substantive answers only;
- ambiguous or out-of-range confidence is not imputed;
- FEQ and CCQ cannot receive `CONFIDENT_CORRECT` under the category rules;
- PCQ/HNQ correctness uses gold answers and aliases with strict numeric
  handling rather than fuzzy keyword similarity.

Framework 0.4.0 replaces institution-specific transports with public OpenAI,
Hugging Face, and configurable OpenAI-compatible adapters. Classification,
correctness, scoring, and analysis rules remain protocol-bound rather than
provider-bound.

## Package map

```text
src/ehq/
  clients/       Provider adapters, pacing, retries, and route checks
  datasets/      Dataset loading, validation, and quality audits
  evaluation/    Classification, correctness, confidence, runner, scoring
  analysis/      Correlations, paired comparisons, intervals, tests
  resources/     Version-pinned study templates and EHQ-3000
  artifacts.py   Atomic writes and artifact-catalog verification
  cache.py       Successful-response-only content-addressed cache
  checkpoint.py  Append-only model checkpoints with fingerprint headers
  config.py      Strict experiment and model-registry loading
  provenance.py  Run manifest and experiment fingerprint
  publication.py Manuscript-ready tables, figures, and LaTeX fragments
  reporting.py   JSONL, JSON, CSV, Excel, and figure export
  selection.py   Order-independent deterministic stratified sampling
  cli.py         User-facing orchestration
```

Provider adapters implement `BaseClient` and return an `InferenceResponse`.
They must expose the requested and resolved route identities, error category,
provider metadata, and reported token usage without turning failures into
answer text.

## Evaluation flow

1. Load and validate the experiment configuration and model registry.
2. Strictly validate the dataset and its release gates.
3. Select items by seed-bound SHA-256 ranking with round-robin stratum balance.
4. Build a fingerprint over all scientifically relevant inputs.
5. Request one answer and, when technically possible, one confidence response.
6. Classify the answer, evaluate correctness, and parse confidence.
7. Append terminal item records to the model checkpoint.
8. Compute EHQ1, EHQ2, EHQ3, the composite, and coverage diagnostics.
9. Write run artifacts atomically and create a SHA-256 catalog.
10. Optionally build Excel, figures, LaTeX tables, and statistical reports.

Progress is written to stderr in readable form by default. `--json-events`
switches progress to JSON Lines; `--json` changes only the final result.

## Fingerprint and resume semantics

The run fingerprint covers framework/protocol versions, configuration,
dataset and registry hashes, selected question-list hash, models, prompt
versions, adjudication/capability inputs, override state, and offline/real mode.
Every model checkpoint carries a compatible model-specific header. Changed
inputs cannot silently reuse old records.

A technically successful but unparseable confidence response is checkpointed
once as terminal missing confidence and excluded from EHQ3. This prevents a
deterministic cached response from being retried forever. Transport and provider
failures remain retryable and are not cached.

The cache key is based on the actual request and provider identity, not the
post-processing version. Successful answers are cached with atomic replacement;
technical failures and protocol violations are not.

## Validation gates

The EHQ-3000 release configuration requires 3,000 unique items, 750 per
category, source evidence for PCQ/HNQ, claim-level PCQ temporal novelty, valid
gold/alias fields, and the CCQ one-redaction/no-leak invariants. Near-duplicate
and human-review gates remain mandatory even after machine validation.

`--allow-candidate` permits only explicitly declared open human gates and
stamps the run non-publishable. It does not suppress structural or semantic
errors. `--allow-unverified-model-registry` is likewise an engineering-only
override whose evidence gaps are written into the manifest.

Custom registries may contain any non-empty set of public or local routes.
Reference mode retains the frozen panel constraints needed for exact study
reproduction. The registry is an audit record, not a promise that every route
is comparable or publishable. See `docs/model_registry.md`.

## Scoring and missingness

EHQ1 and EHQ2 use all valid eligible answer records. EHQ3 uses only valid
confidence-bearing `CONFIDENT_CORRECT` and `CONFIDENT_WRONG` records. The
framework reports answer coverage, confidence coverage, substantive calibration
set size, terminal missing confidence, retryable failures, and category-level
denominators. If the substantive calibration set is empty, EHQ3 and the
composite are `null`; no value is invented.

The machine field `ehq3_protocol=confidence_substantive_only_v1` identifies the
calibration definition. The field `n_ehq3_calibration` records its denominator.
Public prose should describe these concepts in words and reserve the identifiers
for methods, data dictionaries, or reproducibility notes.

## Statistical layer

When complete generational pairs exist, RQ2 reports pairwise differences,
direction counts, mean/median change, Cohen's paired effect size, and an exact
sign-permutation test. RQ1 runs only when an independent capability score file
is supplied; otherwise its status is explicitly `capability_scores_not_provided`.

Component correlations use the model as the unit of analysis. Pearson and
Spearman estimates, deterministic bootstrap intervals, permutation tests, and
Holm-adjusted families are available. Question rows are not treated as
independent model replicates, and composite-to-component correlations are not
used as construct-validity evidence because the composite contains those
components by definition.

## Artifact contract

Core output files are `manifest.json`, `records.jsonl`, model JSON files,
overall/category/subcategory summaries, optional analyses, and
`artifact_catalog.json`. The catalog stores SHA-256 and byte counts for every
declared artifact. `ehq verify-run <directory>` fails when an artifact is
missing, changed, or unexpected.

## Extension checklist

For a new provider adapter:

1. preserve exact requested/resolved route identities;
2. map retryable and terminal failures without fabricating answer text;
3. expose provider token/reasoning metadata when available;
4. implement deterministic request serialization for cache identity;
5. add unit tests for success, routing mismatch, retry, and protocol failure.

For a protocol change, bump the protocol version, document compatibility,
invalidate incompatible checkpoints, add regression tests, and never overwrite
the artifacts from an earlier estimand.
