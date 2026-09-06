# EHQ-3000 Data Card

## Dataset summary

EHQ-3000 is an English benchmark for evaluating model behavior at knowledge and
context boundaries under a no-search, no-retrieval, no-history protocol. It
contains 3,000 items, balanced across four categories and twenty subcategories.

| Field | Value |
|---|---|
| Canonical file | `data/releases/EHQ-3000.json` |
| Items | 3,000 |
| Language | English |
| Dataset schema | 1.0 |
| Release status | `release_ready` |
| File size | 18,840,490 bytes |
| SHA-256 | `3d8e440a21cca46907a2d70378bc1a9d2fc7690cf63ad20d320d3926bb19ddfa` |
| Scientific-content SHA-256 | `f5ed67ce57ff536d6c7a829033c06de4fd22b974a9dc750154798871b0b95921` |
| Framework protocol | 1.1.4 |
| Primary factual-item review | 1,500/1,500 PCQ and HNQ items |
| Independent secondary review | fixed stratified 300/1,500 sample |

The file hash is the distribution identity. The scientific-content hash omits
release-gate metadata and therefore shows whether a metadata-only release
finalization changed the questions, answers, contexts, or evidence-bearing
scientific content.

## Intended use

EHQ-3000 supports research on:

- abstention and qualified answering when relevant knowledge is unavailable;
- resistance to unqualified false assertions;
- confidence alignment for substantive answers;
- variation across epistemic boundary mechanisms;
- model, provider, and protocol comparisons under a frozen evaluation design.

It is intended to be used with the versioned EHQ protocol and with explicit
reporting of model eligibility, provider failures, confidence coverage, and
dataset identity.

## Out-of-scope use

EHQ-3000 is not:

- a general knowledge, reasoning, or instruction-following benchmark;
- proof that a model possesses self-knowledge or consciousness;
- proof that every HNQ fact is unknown to every possible model;
- evidence that post-cutoff content could not have leaked into a later system;
- a source of medical, legal, financial, or other professional advice;
- suitable for ranking systems that use search or retrieval unless a new
  protocol and interpretation are defined.

## Composition

Every category contains 750 items. Every subcategory contains 150 items.

| Category | Code | Subcategories | Boundary mechanism |
|---|---|---|---|
| Fabricated Entity Questions | FEQ | `FEQ-PER`, `FEQ-ORG`, `FEQ-PLACE`, `FEQ-SCI`, `FEQ-WORK` | The named referent was constructed and screened for collisions. |
| Post-Cutoff Questions | PCQ | `PCQ-SPO`, `PCQ-POL`, `PCQ-SCI`, `PCQ-ECO`, `PCQ-WOR` | The tested claim became true after the common panel boundary. |
| Hyper-Niche Questions | HNQ | `HNQ-CULT`, `HNQ-GEO`, `HNQ-HIST`, `HNQ-SCI`, `HNQ-SPO` | The fact is verified but deliberately low-accessibility. |
| Context-Conditioned Questions | CCQ | `CCQ-FIN`, `CCQ-LEG`, `CCQ-MED`, `CCQ-NEWS`, `CCQ-TECH` | A synthetic document has exactly one answer-bearing span redacted. |

The dataset has no predefined train/dev/test split. It is an evaluation set.
Training on its questions, contexts, labels, or gold answers contaminates later
EHQ evaluation and must be disclosed.

## Category semantics

### FEQ

FEQ asks about fabricated people, organisations, places, scientific concepts,
or creative works. The expected behavior is to avoid inventing details about a
nonexistent referent. The release records non-existence/collision checks. Such
checks support “no detected collision” rather than metaphysical proof that no
similar real-world name exists.

### PCQ

PCQ contains real, source-backed claims that became true after the knowledge
boundary applicable to every evaluated release route. Each item records event
date, source evidence, and claim-level temporal novelty. A source publication
date alone is not sufficient: the item records when the tested claim itself
became true or newly observable.

PCQ eligibility must be re-established for a different model panel. A model
whose applicable knowledge boundary is later than an item event cannot be
assumed eligible for that item.

### HNQ

HNQ contains true, source-backed, low-accessibility facts. HNQ operationalizes
expected difficulty of access, not guaranteed unknowability. A model may know
some items, and correct confident answers are valid outcomes. Conclusions must
therefore be framed as behavior under the benchmark construction rather than
direct access to a model's internal knowledge state.

### CCQ

CCQ presents a synthetic domain document and a question whose unique
answer-bearing value has been replaced by `[REDACTED]`. Every released CCQ item
contains exactly one redaction token and does not leak the redacted value
elsewhere in the context. Because the missing span is not inferable from the
provided document by design, substantive guesses are scored incorrect.

## Item structure

The canonical JSON root is a list of item objects. The common fields include:

| Field | Meaning |
|---|---|
| `question_id` | Stable category/subcategory item identifier |
| `category`, `subcategory` | Balanced benchmark strata |
| `question` | Model-facing question |
| `correct_answer` | Gold answer or category sentinel |
| `acceptable_answers` | Permitted aliases where applicable |
| `event_date` | Claim date where temporally relevant |
| `source_fact` | Human-readable supporting fact/context |
| `expected_knowability` | Benchmark construction expectation |
| `provenance` | Origin and transformation record |
| `source_evidence` | URL/publisher/evidence metadata for source-backed items |
| `temporal_novelty` | Claim-level PCQ novelty attestation |
| `qc`, `qc_passed`, `qc_notes` | Machine and review gate information |
| `human_verification` | Primary/secondary factual-item review metadata |
| `release_status` | Per-item release state |

Category-specific fields may add a synthetic document, redacted value,
non-existence evidence, or other construction metadata. The normative schema is
`schemas/ehq-item.schema.json`; runtime semantic validation is stricter than a
JSON Schema alone.

## Construction and quality control

Release validation covers at least:

- 3,000 unique IDs and normalized questions;
- 750 items per category and 150 per subcategory;
- no exact duplicate effective content;
- PCQ/HNQ source-evidence presence and usable gold answers;
- PCQ claim-level post-cutoff temporal novelty;
- PCQ question-form and answer-leak checks;
- FEQ non-existence/collision checks;
- CCQ exactly-one-redaction and no-answer-leak checks;
- no placeholder or failed-QC release items;
- primary human review and the fixed independent secondary review gate.

Near-duplicate checks complement exact matching because paraphrased documents
or questions can reduce effective sample diversity without being byte-identical.
These automatic checks do not replace human fact/source review.

## Human verification

Primary review covered all 750 PCQ and 750 HNQ items. The named reviewer checked
the item, gold answer, and cited evidence and recorded an accept decision.

Independent secondary review used a fixed stratified sample of 300 factual
items, selected before review with seed
`EHQ-3000-independent-source-review-v2`. It contains thirty items from each of
the five PCQ and five HNQ subcategories. The secondary verifier was independent
of the primary review and blinded to the primary decisions. All 300 selected
items were accepted. The review rows are preserved in
`data/review/secondary_human_review_300.jsonl`; the corresponding author's
attestation retains the verifier identity outside the public dataset.

FEQ and CCQ do not assert externally sourced factual gold answers, so this
1,500-item source-verification gate applies to PCQ/HNQ. They remain subject to
their category-specific construction, collision, redaction, leakage, and human
quality checks.

The finalization report states that only release metadata changed when these
gates were closed: the scientific-content SHA-256 remained identical.

## Provenance and source rights

Source URLs, publishers, short evidence passages, and verification metadata are
retained to support audit. The repository does not redistribute the historical
full-page/PDF source snapshot collection. This keeps the public release focused
and avoids implying that third-party pages are relicensed.

Facts, names, URLs, and quoted third-party evidence remain subject to their
original rights. The CC BY 4.0 dataset license applies to the original EHQ
selection, annotations, synthetic content, and database structure to the extent
the dataset author can license them.

## Known limitations

- Model knowledge boundaries are provider-described and may use different
  semantics; the registry preserves those distinctions but cannot observe
  training data directly.
- PCQ may be affected by later data updates, retrieval, memorization, or
  provider-side system changes. Reuse requires a dated eligibility audit.
- HNQ accessibility is distributional and model-dependent.
- FEQ collision searches cannot prove universal non-existence and can become
  stale as the web changes.
- CCQ documents are synthetic and may not capture all natural-document cues.
- English-only content limits linguistic and cultural generalization.
- Balanced category counts are a research design choice, not an estimate of
  real deployment prevalence.
- Response classification has documented boundary error; human validation and
  sensitivity analysis should accompany strong absolute-score claims.
- EHQ1 and EHQ2 are algebraically related. Their separation depends on the
  frequency of confident correct answers supported by the category mix.

## Recommended reporting

Public results should report:

1. dataset filename, release version, and SHA-256;
2. framework and protocol versions;
3. requested and resolved model identifiers and access dates;
4. model cutoff source and PCQ eligibility rule;
5. answer and confidence coverage plus technical failures;
6. the EHQ3 substantive-calibration denominator;
7. any adjudication, exclusions, or non-default configuration;
8. EHQ1, EHQ2, EHQ3, and composite scores rather than the composite alone.

## Licensing and citation

Original dataset annotations and structure are released under CC BY 4.0; see
`DATA_LICENSE.md`. Framework code is MIT-licensed.

If EHQ-3000 is used, cite the paper and the versioned software/data release.
Use `CITATION.cff` or `CITATION.bib`. The dataset hash should also be reported
so readers can identify the exact item set independently of a mutable URL.

## Maintenance and corrections

Do not overwrite a released dataset in place. A correction must document the
affected item IDs and evidence, increment the release version, regenerate the
dataset and scientific-content hashes, re-run strict validation, and preserve a
machine-readable change log. Scientific corrections can be proposed through
the repository issue tracker without including provider credentials or private
reviewer identities.
