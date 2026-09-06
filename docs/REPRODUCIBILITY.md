# Reproducibility and Artifact Policy

An EHQ result is identified by the combination of framework version, protocol
version, dataset SHA-256, registry SHA-256, configuration hash, prompt versions,
selected-question-list hash, requested/resolved routes, and adjudication inputs.

## Public versus restricted material

Public repository material includes framework source, configuration templates,
the canonical dataset, review attestations, schema, documentation, and tests.

Generated model responses, provider payloads, caches, checkpoints, populated
`.env` files, private reviewer identities, and result directories are excluded
from the public source release. A study team may preserve them in a controlled
archive when provider terms and ethics approvals permit.

## Minimum reproducibility record

- tagged source release or commit;
- environment and dependency versions;
- dataset and model-registry hashes;
- full run manifest and artifact catalog;
- raw normalized records and summaries;
- failure, missingness, retry, and exclusion logs;
- human coding and adjudication files when used;
- analysis parameters, random seeds, and bootstrap/permutation counts;
- final tables/figures and their source catalogs.

Run `ehq verify-run` before analysis handoff and again after archival transfer.
Do not regenerate a catalog after an accidental edit; restore the original
artifact or create a new version with a documented change.
