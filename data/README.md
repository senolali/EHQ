# EHQ data release

This directory contains only public scientific inputs, not provider responses
or generated results.

## Files

- `releases/EHQ-3000.json`: canonical 3,000-item evaluation dataset.
- `releases/EHQ-capability-probe.json`: matched document-grounded capability
  probe used by the study protocol.
- `examples/EHQ-20-smoke.json`: non-publishable 20-item framework fixture.
- `review/model_cutoff_attestation.json`: frozen cutoff release attestation.
- `review/release_gate_finalization_report.json`: metadata-only gate closure
  and scientific-content hash record.
- `review/secondary_human_review_300.jsonl`: fixed independent secondary-review
  sample and decisions.
- `CHECKSUMS.sha256`: SHA-256 identities for public data/review files.

See `docs/EHQ-3000_DATA_CARD.md` before reuse. The exact full dataset is
included in the GitHub repository and the PyPI wheel. `ehq init` creates a
local, version-pinned copy, while `ehq dataset-path` prints the installed
resource path. Cite the dataset, framework, and accompanying paper using
`CITATION.bib` or `CITATION.cff`.
