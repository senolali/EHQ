# Changelog

All notable changes to the EHQ evaluation framework are recorded here.

## [0.4.0] - 2026-09-06

### Provider-neutral public release

- Replaced institution-specific adapters and credentials with adapters for the
  OpenAI Responses API, Hugging Face Inference Providers, and configurable
  OpenAI-compatible chat-completion endpoints.
- Added user-defined model registries while retaining an explicit strict
  reference-registry mode for publication-grade panel records.
- Bundled the version-pinned EHQ-3000 release in the wheel and in `ehq init`
  projects, and added `ehq dataset-path` for direct programmatic discovery.
- Added provider examples for OpenAI, Hugging Face, OpenRouter, Together, Groq,
  vLLM, and Ollama, without adding provider-specific SDK dependencies.
- Removed institution-specific generation/probing utilities from the public
  distribution; the released dataset is treated as an immutable research
  artifact rather than regenerated during evaluation.
- Expanded packaging, security, citation, dataset-discovery, and reproducible
  pilot/full-run documentation.

## [0.3.3] - 2026-08-08

### Public package and repository release

- Added complete PyPI metadata, Python-version classifiers, project URLs,
  optional dependency groups, typed-package marker, and a small installed
  project template.
- Added `ehq --version` and `ehq init [directory]` so a PyPI installation can
  create a safe study scaffold without copying credentials or the full dataset.
- Added GitHub CI and PyPI Trusted Publishing workflows, citation metadata,
  software/data license separation, contribution and security policies, a
  current EHQ-3000 data card, checksums, and release instructions.
- Curated the public tree to omit populated `.env` files, results, provider
  responses, caches, checkpoints, build products, historical broken candidates,
  one-off repair scripts, and editor/OS metadata.
- Renamed the bundled non-publishable smoke fixture to `EHQ-20-smoke.json`.
- Preserved protocol 1.1.4 and all response classification, correctness,
  scoring, and statistical behavior from framework 0.3.2.

## [0.3.2] - 2026-08-06

### Construct and reporting validity

- Reframed EHQ as three observable sub-scores spanning two operational axes:
  epistemic control (EHQ1/EHQ2) and substantive-answer calibration (EHQ3).
- Made the exact EHQ1/EHQ2 dependency explicit, distinguished their intended
  behaviors, and identified the observed near-collinearity as dataset-dependent;
  no claim of independent latent dimensions is made.
- Clarified that the protocol label `CONFIDENT_WRONG` denotes an unqualified
  response scored incorrect under the category rules and is not assigned through a numeric-confidence threshold;
  retained EHQ2 as the frozen unconditional per-query avoidance estimand.
- Added the category support matrix and exploratory category-specific
  EHQ1--EHQ2 correlations (Supplementary Table S15): identity in FEQ/CCQ, weak
  separation in PCQ, and material but still correlated separation in HNQ.
- Bounded the capability conclusion to the near-ceiling document-grounded floor
  probe actually administered.
- Added no-HNQ, knowledge-boundary (EHQ-K), strict-knowledge, and
  context-boundary (EHQ-C) sensitivity analyses from unchanged retained records.
- Clarified that FEQ/CCQ category EHQ3 is one-sided false-answer confidence
  alignment and that the frozen composite weights are normative, not fitted.
- Added a category-balanced EHQ3 sensitivity from unchanged records and a
  matched CCQ missing-span/restored-span selectivity analysis using the existing
  capability probe; neither analysis modifies the frozen confirmatory metric.
- Added small-sample Fisher-z intervals for the three model-level component
  correlations and retained the bootstrap intervals as descriptive checks.

### Classifier validation

- Added a deterministic blinded model-by-category response sample generator for
  two independent human coders.
- Added four-class and restraint/substantive agreement, confusion-matrix,
  precision/recall/F1, and adjudication analysis tooling.
- Defined and enforced the coder uncertainty flag as a diagnostic variable,
  required notes for uncertain decisions, and added uncertainty coverage and
  agreement outputs without changing or excluding response labels.
- Made the generated disagreement CSV directly reusable as the adjudication
  input; retained explicit legacy-column and missing-uncertainty compatibility
  only for reproducing the initial calibration pilot.
- Added a hard category-feasibility gate that rejects
  `CONFIDENT_CORRECT` labels in FEQ or CCQ before analysis.
- Completed the fresh 560-response independent human double-coding study and
  recorded the 25-item corresponding-author adjudication with explicit
  `human-with-ai-decision-support` provenance; the original coder labels and
  their agreement remain unchanged.
- Added macro and support-weighted F1 to the machine-readable validation result.
- Added a reproducible post-hoc score-impact analysis that recomputes EHQ1 and
  EHQ2 from automated and resolved human labels within the balanced validation
  sample, then reports pooled bias, per-model differences, and tie-adjusted
  Pearson/Spearman rank preservation in CSV, JSON, and LaTeX.
- Narrowed the framework and manuscript claim: the deterministic classifier is
  not treated as a validated four-class human replacement, although the binary
  restraint/substantive distinction and sampled model ordering are materially
  stronger.
- Added diagnostics for score-dependent classifier bias and automated-versus-
  human score spread; the manuscript no longer interprets HEDGE counts as
  distinct response strategies.
- Added an exact post-hoc Claude/non-Claude label-permutation diagnostic and
  human-reference component ranks. The documented family association qualifies
  the broad rank-stability result but is not interpreted causally.
- Quantified EHQ3 pool-label sensitivity: 85 of 369 human-reference restraint
  responses enter the automated substantive pool, representing 31.6% of that
  pool in the validation sample.
- Expanded the generated adjudication CSV with the blinded question, context,
  reference fields, response, and both coders' uncertainty flags and notes so
  adjudicators can resolve disagreements without joining files manually.
- Added a UTF-8 byte-order mark to generated human-facing CSV files so Excel
  preserves Unicode text and comma-delimited structure by default.
- Recorded honestly that the completed run has no response-level human
  adjudications; dataset human review is not treated as classifier validation.
- Added a nonconfirmatory FEQ lexical audit and conservative sensitivity bound
  for 156 direct non-existence/fabrication review candidates; primary artifacts
  and labels remain unchanged.

### Manuscript and packaging

- Corrected stale table/figure references, caption claims, Claude confidence
  semantics, standalone supplement citations, and unused bibliography entries.
- Restored the Dr. Liu-reviewed Abstract's outcome-oriented structure: the
  Abstract now explains what the findings mean without enumerating scores,
  intervals, or p-values.
- Completed a manuscript-wide prose pass that removes all 50 em-dash
  constructions, improves logical transitions, and preserves the reviewed
  Introduction's motivation and examples while retaining necessary scientific
  corrections.
- Retained the Dr. Liu-reviewed manuscript title and aligned the main paper,
  supplement, and release documentation with it.
- Removed residual three-independent-component and guaranteed-ignorance
  language; reported the exact HNQ/PCQ composition of confident-correct
  responses.
- Added the PCQ cutoff-to-event temporal-distance limitation and documented why
  heterogeneous, incompletely covered public MMLU-Pro/GPQA scores were not
  combined post hoc.
- Bumped the framework/reporting release to `0.3.2`; protocol `1.1.4` and all
  provider responses and primary EHQ scores remain unchanged.
- Corrected the Gemini confidence-compliance denominator distinction, PCQ/HNQ
  confident-correct percentages, secondary-review denominator, and two
  bibliographic records identified by an independent numerical audit.
- Corrected the CCQ figure/table description, Pearson-versus-Spearman Holm
  statement, mixed SD convention, and supplementary coverage. Added CCQ to the
  framework comparison, a per-model calibration-definition table, and the
  sealed response-acquisition dates.
- Recorded the public repository URL and DOI as unresolved pre-submission gates
  instead of making a premature public-availability claim.
- Corrected the component-correlation Spearman p-value order and clarified that
  family membership is observational rather than randomly assigned.
- Made the exact family permutation denominator auditable: EHQ1 has 21/1001
  assignments as or more extreme and EHQ2 has 28/1001, yielding .021 and .028.
- Added family-indicator versus score-level bias correlations, model-level EHQ3
  pool leakage, and a human-reference composite sensitivity from the frozen
  560-response validation sample. The same four Claude routes remain the top
  four within this sparse sample, with a maximum rank shift of three.
- Widened the supplementary registry provider column and shortened duplicated
  response-classifier limitations prose.

## [0.3.1] - 2026-08-06

### Release gates and provenance

- Recorded completion of the fixed blinded 300-item secondary human review
  (30 items per PCQ/HNQ subcategory) without changing scientific item content.
- Marked the 15 evaluated routes as the model registry's release-verified scope.
  DeepSeek-V4 uses a named-human conservative cutoff upper bound rather than an
  unsupported claim about an exact unpublished training date.
- Added a release fingerprint binding the original provider-run fingerprint to
  the verified dataset, registry, protocol, framework, and analysis exclusion.
- Revalidated all 45,000 retained records with zero provider calls and no score
  changes.

### Analysis and reporting

- Made the Gemini-2.5-Pro truncation exclusion operative in every confirmatory
  table, figure, correlation, and pair analysis while retaining its descriptive
  records and scores.
- Added exploratory model-level correlations for all six category pairs, with
  bootstrap intervals, permutation tests, and Holm adjustment.
- Added a frozen composite-weight sensitivity analysis and model-registry table.
- Corrected bootstrap wording from categories to the 20 subcategories and
  replaced independence/equivalence claims with interval-aware interpretations.

### Packaging

- Promoted the framework to `0.3.1` and the protocol to `1.1.4`.
- Updated the manuscript, README, runbook, release status, tests, workbook, and
  publication package to the same release identity.

## [0.3.0] — 2026-08-04

### Changed

EHQ₃ now computes confidence calibration only over substantive answers
(`CONFIDENT_CORRECT` and `CONFIDENT_WRONG`).

`ABSTAIN` and `HEDGE` behaviours are evaluated only through EHQ₁ and EHQ₂.

Reason: avoid double-counting epistemic restraint.

This is not a new metric. It is the updated official definition of EHQ₃, and
it supersedes the definition used up to and including v0.2.12. Results
produced under the earlier definition are not comparable with results produced
under this one.

### Added

- `ehq3_protocol: "confidence_substantive_only_v1"` is stamped on every
  generated summary and on experiment metadata: `summary.json`,
  `analysis.json`, the run `manifest.json` (top level and under `run`), and
  the publication package's `report_manifest.json`.
- `n_ehq3_calibration` reports how many records entered the calibration
  computation. It appears in the per-model scores, `summary.json`,
  `summary.csv`, `category_scores.csv`, and `subcategory_scores.csv`.
- The publication package is built automatically when `ehq full` finishes,
  into `<run>/report`. The report builder moved from
  `tools/build_provisional_pilot_report.py` into `ehq.publication` so the
  runner and the standalone tool share one implementation; the tool is now a
  thin wrapper and gained `--overwrite`. Control it with `--report` /
  `--no-report` and `--report-resamples`. A package that cannot be built is
  reported as skipped and never fails an otherwise successful run.
- The report is named and titled after the run (`EHQ_REPORT_<run-id>.md`), and
  its LaTeX labels are derived from the run id, so two runs no longer collide.
  The scope section, selection balance, pair count, and interpretation
  boundaries are computed from the run instead of describing the n=100 pilot.
- `tools/audit_truncated_responses.py`: finds answers that were cut off
  instead of completed. A truncated answer is not a technical failure -- the
  provider reports success and the text reaches the classifier and the metrics
  as if the model had finished speaking -- so it is reported from three
  strong per-record signals -- completion tokens at the configured budget, and
  a clause stopping on a function word, a dangling separator, or an unclosed
  quote -- plus a panel comparison for the case no per-record rule can settle.
  A finished statement that merely lacks a full stop is not counted; instead
  each model's unpunctuated rate is compared against the panel median on the
  identical item set, so a model that truncates pervasively is identified
  without inventing an absolute threshold. Quotation marks are matched by
  position rather than parity, because English possessives consume
  apostrophes and parity alone reported finished answers as cut off.
- `--exclude-model` now actually reaches the analysis. `_preset_run` builds a
  fresh `argparse.Namespace` for `_run` from a hand-written list of fields, and
  the new option was not on it; the reader tolerated the absence with
  `getattr(args, "exclude_model", None) or []`, so `ehq full --exclude-model ...`
  accepted the flag, forwarded nothing, and excluded nothing. The run that was
  meant to withhold a truncation-affected model instead published the RQ1
  correlation the exclusion existed to suppress. The field is forwarded, the
  reader now reads it directly so a future omission raises instead of resolving
  to "exclude nothing", the flag was added to the direct `run`/`dry-run`
  parsers so the attribute always exists, and `tests/test_cli_argument_forwarding.py`
  compares what `_run` reads against what `_preset_run` forwards so the two
  cannot drift apart again.
- `--exclude-model NAME=REASON` withholds a model from the confirmatory
  analyses without deleting it. A model can be measured and still not be
  measurable: a route whose answers were systematically truncated produces
  scores that are artefacts of the truncation. Excluded models leave the panel
  that feeds component correlations, RQ1, RQ2 and RQ3, while `analysis.json`
  carries `confirmatory_panel` and an `excluded_models` list with each model's
  scores and the stated reason, so the exclusion is reported rather than
  performed silently. A reason is mandatory, excluding a model absent from the
  run is an error, and excluding every model is refused.
  This also closes a trap the homogeneity gate alone leaves open. With a
  truncation-depressed model in the panel the capability scores are
  heterogeneous (p = 0.0001), so RQ1 judges the measure informative and
  publishes r = +0.45 -- an association carried entirely by that one artefact.
  With it withheld the remaining panel is homogeneous (p = 0.45) and no
  coefficient is emitted.
- RQ1 no longer reports a correlation it cannot interpret. `homogeneity_test`
  (analysis/statistics.py) asks whether per-model probe counts are consistent
  with a single shared success rate, by resampling under that null rather than
  assuming a chi-square table. When they are, RQ1's status becomes
  `capability_homogeneous_no_variance_to_correlate`, `correlation_interpretable`
  is false, and the publication package emits no RQ1 correlation table. On the
  confirmatory panel the capability probe sits at 99.1% pooled with a
  between-model standard deviation 1.02x what sampling one rate produces, so
  the r=+0.66 the old code would have published describes the difference
  between one error and five out of 259.
- `capability_scores.csv` gained `n_correct` and `n_scored`, without which the
  above test cannot run; `load_capability_counts` reads them and files written
  before the columns existed still load. `tools/backfill_capability_counts.py`
  copies the counts from an existing run's `capability_summary.json` rather
  than repeating any probe request, and refuses to derive them from the rate,
  since denominators differ between models.
- The derived-analysis tools write beside a run rather than inside it.
  `truncation_sensitivity.py` and `abstention_confidence_compliance.py`
  defaulted to `<run>/sensitivity` and `<run>/analysis`, which added files a
  completed run's SHA-256 catalogue does not list. `verify_run_artifacts` then
  reports the run as invalid and the publication builder refuses to rebuild
  against it -- correctly, since a catalogue that tolerated later additions
  would certify whatever happened to be present. Both tools now default to
  `<run>_analysis/` and refuse an `--output-dir` inside the run, with the
  reason. A run already polluted this way is repaired by deleting the added
  directories; the catalogued files are untouched.
- `tools/abstention_confidence_compliance.py`: measures how models answer the
  confidence question after declining to answer at all. The prompt directs a
  model that gave no substantive answer to report 0, so the correct value is
  known and compliance is measurable rather than inferred. Truncated records are
  dropped, using the same rule as the run-level audit, because a heavily
  truncated model's abstention set is the refusals whose wording survived the
  cut rather than a sample; `--exclude` removes a model outright and names it in
  the output and in the generated table. Grouping is by the organisation that
  trained the model, read from its name, not by the registry's
  `model_provider`, which is the serving route -- Claude, LLaMA, Nova and
  GPT-OSS all arrive through `aws`, and grouping on it would invent a family
  spanning four developers.
- `tools/verify_route_from_run.py`: writes a route's verification fields from a
  completed run instead of by hand. It refuses unless every call resolved to
  the exact requested route, coverage clears thresholds given as arguments
  (EHQ1/EHQ2 and confidence counted separately, because those failed
  independently for GPT-5-mini), no reasoning token was reported under
  `reasoning_mode=disabled`, and the run's artifacts still match their catalog.
  A run with no catalog is refused rather than treated as valid. This is the
  gap that let GPT-5-mini carry a zero-thinking-token attestation from one
  short probe that a later run contradicted.
- The model registry can grow. `REFERENCE_PANEL_SIZE` (constants.py) replaces a
  hardcoded 20 and is now 21, for the newly registered GPT-5.4-mini route. The
  registry is the record of every route considered, so a permanently unusable
  entry such as DeepSeek-V3 is kept as evidence rather than reclaimed.
- `load_models` refuses a registry in which two entries claim the same side of
  the same generational pair. They previously overwrote one another when pairs
  were assembled, and file order silently decided which model represented the
  pair.
- GPT-5-mini is recorded as `response_format_not_scoreable` and detached from
  the `gpt_mini` pair. Probes showed its empty responses are not a rate limit
  and not a reasoning budget: it returns usable text for 8/20 unknowable items
  against 18/20 document-grounded ones, so the retry budget would convert
  silence into answers and depress EHQ1 by policy. See `docs/model_registry.md`.
- `tools/truncation_sensitivity.py`: prices truncated answers into the scores
  instead of only counting them. The withheld continuations are unrecoverable,
  so the run is rescored three ways -- as published, with suspected records
  removed from every denominator, and with suspected `CONFIDENT_WRONG` records
  relabelled `HEDGE` -- and the interval between them is reported. Relabelled
  records leave the calibration set by the same rule that defines
  `confidence_substantive_only_v1`, so EHQ₃ is recomputed under each reading
  rather than held fixed. Rank intervals are reported for every model, because
  a model whose own scores never move can still be uncertain about its
  position when a peer's do. Writes `sensitivity/truncation_sensitivity.{json,
  csv,tex}` next to the run. The truncation rule itself lives in
  `audit_truncated_responses.signals` and is imported by both tools, so the
  records counted in the audit table are the records rescored here.
- `tools/build_capability_probe.py` and `tools/run_capability_probe.py`: an
  RQ1 capability measure taken on the evaluated panel under the evaluation
  protocol, instead of imported benchmark numbers. The probe restores the
  withheld span in released CCQ documents so the answer is present, keeping
  only spans short enough to grade as answers; it elicits no confidence and
  scores a refusal as incorrect, and writes `capability_scores.csv` for
  `--capability-scores`.
- `--label-prefix` on the publication builder (and `--report-label-prefix` on
  the runner). LaTeX labels default to the run id so ad-hoc reports never
  collide in one document; a manuscript build passes an empty prefix to get
  stable targets such as `tab:ehq-scores` that survive regeneration.
- `tables/coverage_by_model.tex`, so scoring coverage can be typeset directly.
- Statistical analysis tables. `component_correlations` now carries both
  coefficients with bootstrap intervals, permutation p-values, and the
  Holm-adjusted p-values that were computed but never surfaced. RQ1 gains
  `rq1_capability` and `rq1_rank_shifts` (written when capability scores were
  supplied) and RQ2 gains `rq2_paired_summary` with Cohen's dz and the exact
  sign-permutation p-value. The report grew a "Statistical analysis" section.
- LaTeX output for every table: category scores, response distribution,
  component correlations, calibration sources, and coverage exclusions join
  the model-score and pair-difference tables.
- New `calibration-sources` figure and `calibration_sources.csv`/`.tex`:
  per-model confidence-instruction compliance on abstentions, and the
  calibration error of the restraint records EHQ3 now excludes next to the
  substantive records it keeps.
- `MODEL_LABELS` covers the full model registry, so figures no longer mix
  display names with raw registry identifiers.
- Human-readable console output. Progress, provider retries, and the closing
  run summary are formatted for reading instead of emitted as JSON lines.
  `--json-events` restores the JSON progress stream and `--json` restores the
  machine-readable run result, so anything that parsed the old output can opt
  back in.
- The publication report tolerates incomplete coverage. The stratified
  bootstrap now applies the same per-record validity mask as the point
  estimates, so a model with a technical failure or an unparseable confidence
  is resampled over the records it has instead of crashing the report. Every
  unscored record is documented with its reason in
  `tables/coverage_exclusions.csv`, alongside `tables/coverage_by_model.csv`,
  and `report_manifest.json` records
  `coverage_policy: "report_and_exclude_no_imputation"`.
- `tools/recompute_ehq3_protocol.py` regenerates a completed run **in place**
  from its retained evaluation records. It issues no provider request, reads
  no cache entry, creates no new output directory, and preserves the run id,
  experiment name, and parent fingerprint, because the underlying model
  responses are unchanged. It refuses to run if the existing artifacts do not
  match their catalog, and it fails if regeneration would alter
  `records.jsonl`.

### Fixed

- Building the package inline crashed the run: it hashed the source artifact
  catalog unconditionally, but the runner writes that catalog only after the
  package is in place. The hash is now recorded as null when the catalog does
  not exist yet, and the runner's guard catches every exception type rather
  than an enumerated few, so no derived artifact can discard a completed
  evaluation.
- Both bootstrap figures crashed when a point estimate fell outside its own
  percentile interval, which matplotlib rejects as a negative error bar. The
  whiskers are clamped at zero.
- The publication report crashed on a model whose EHQ3 was undefined because
  it produced no substantive answer. Such models are now excluded from the
  package, listed in `report_manifest.json` under
  `models_without_defined_ehq`, and named in the report's interpretation
  boundaries. Category cells with no substantive answer render as `n/a`
  instead of aborting the figure.
- The publication report crashed with a `TypeError` when fewer than two
  generational pairs were complete, because Cohen's dz and the exact
  sign-permutation p-value are undefined there and were formatted as numbers.
  The recorded status is now printed instead.

- `tools/recompute_ehq3_protocol.py` dropped RQ1. It rebuilt `analysis.json`
  without the capability scores the parent run was given, so regenerating a
  study that answered RQ1 silently turned that section into "capability scores
  not provided". The scores are now reloaded from the path the manifest
  records, and regeneration is refused if that file is missing or its hash no
  longer matches, rather than redefining RQ1 in place.

### Behavioural consequences

- A model that never produces a substantive answer now has an undefined
  (`null`) EHQ₃ and therefore an undefined composite EHQ. Under the superseded
  definition such a model scored near-perfect calibration by abstaining with
  low stated confidence — precisely the double-counting this change removes.
  The offline mock route, which abstains on every item, is affected.
- The stratified bootstrap in `tools/build_provisional_pilot_report.py`
  applies the same inclusion rule, so published confidence intervals remain
  consistent with the point estimates. Both its per-bin gaps and its ECE
  denominator are restricted to substantive answers.

### Presentation of tables and figures

- Every generated table caption now describes what the table shows instead of
  restating the run id. The run is identified once, in the setup section, not
  nine times in the list of tables.
- Generated tables are set in `\small` and wrapped in a `\resizebox` that
  shrinks only when the tabular is genuinely wider than the text block, so the
  wide ones (component correlations, coverage, response distribution, RQ2, and
  the two derived-analysis tables from `tools/`) no longer run into the margin.
  The idiom needs `graphicx` alone; no new package.
- Text-valued columns are left-aligned. `coverage_exclusions` was right-
  aligning its reason and detail columns because the layout was derived from
  the column count.
- `category_scores` is emitted as a `longtable` and moved to an appendix. At
  one row per model and category it cannot fit in a float on any panel of
  realistic size; the body keeps the heatmap. This adds
  `\usepackage{longtable}` to the manuscript preamble.
- New figure `component-breakdown`: the three components and the composite as
  grouped bars, by model in EHQ order. The ranking figure shows the composite
  with its interval; this one shows what the composite is made of, which is
  where the restraint-versus-calibration trade-off is visible.
- `calibration-sources` moves its legend below both panels. In-axes it sat on
  top of the bars of whichever models happened to fall in that corner, and
  which models those are changes with the panel.
- `pair-differences` is now two panels: each pair on the EHQ scale from
  predecessor to successor, beside the difference with its interval. Pairs are
  ordered by the size of the gain and named by their members on the axis, so
  no in-plot label can collide with a short segment.

### Manuscript

- `manuscript/main.tex` documents the current definition. Methodology gains the
  calibration set $S_m$, the protocol name, the undefined case, and a new
  subsection deriving the change from confidence-instruction compliance rather
  than from double-counting alone.
- Results is written and bound to the publication package through
  `\ehqtable`/`\graphicspath`, behind an `\ifehqresults` release gate that
  stays false until the dataset-review and registry gates pass. Missing
  conditional tables report themselves instead of breaking the build.
- Limitations covers scoring coverage and its non-determinism, incomplete
  generational pairs, cross-protocol non-comparability, and the need to
  re-establish prompt sensitivity under the current calibration set.

### Unchanged

EHQ₁, EHQ₂, classification, confidence parsing, prompts, datasets, API
clients, retry policy, checkpointing, and the response cache are untouched.

`PROTOCOL_VERSION` remains `1.1.3-release-candidate`: the measurement protocol
that produced the model responses did not change, only EHQ₃ post-processing.
Existing experiment configs therefore remain valid.

The response cache is keyed on provider request identity, not on scoring, so
the framework-version bump does not invalidate retained responses. The cache
identity is pinned to `RESPONSE_IDENTITY_VERSION` (`0.2.12`), leaving every
existing cache entry a hit.

## [0.2.12] and earlier

See `docs/FRAMEWORK_VALIDATION_v0.2.*.md` for the per-version validation
records that predate this changelog.
