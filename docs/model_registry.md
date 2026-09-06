# Model Registry

`config/models.json` is the auditable boundary between the EHQ protocol and a
provider route. It contains identifiers and evidence, never credentials.

## Registry modes

- `custom` accepts any non-empty user-defined panel. It is the default for new
  OpenAI, Hugging Face, and OpenAI-compatible studies.
- `reference` additionally enforces the frozen panel size, generational-pair
  structure, cutoff bound, and release scope used by the original study.

Custom does not mean scientifically weak. A custom panel can pass the release
gate when every selected route has verified endpoint identity and, when PCQ is
enabled, adequate cutoff evidence. Reference mode exists to reproduce a
specific frozen panel rather than to privilege one provider.

## Required route fields

| Field | Meaning |
|---|---|
| `name` | Unique display name used by the CLI and output files |
| `provider` | `openai`, `huggingface`, or `openai_compatible` |
| `provider_model` | Exact model identifier sent to the endpoint |
| `reasoning_mode` | Must be `disabled` under protocol 1.1.4 |

## Transport fields

| Field | Meaning |
|---|---|
| `base_url` | Optional for OpenAI/Hugging Face; required for `openai_compatible` |
| `api_key_env` | Name of the environment variable holding the secret |
| `requires_api_key` | Set false only for an unauthenticated local endpoint |
| `model_identity_policy` | `record` for diagnosis or `strict` for frozen runs |

Secrets must never appear in the registry. Only the environment-variable name
is persisted in the manifest.

## Operational verification

Begin with:

```json
{
  "model_identity_policy": "record",
  "operational_status": "unverified"
}
```

Run a pilot and inspect the requested/resolved identities, coverage, reasoning
token metadata, and artifact catalog. For a final study, pin the exact route
when possible and record:

```json
{
  "model_identity_policy": "strict",
  "operational_status": "verified",
  "endpoint_verified_at": "2026-09-07T12:00:00Z",
  "endpoint_verified_resolved_model": "exact-provider-model-id",
  "non_reasoning_verified_at": "2026-09-07T12:00:00Z",
  "non_reasoning_observed_thinking_tokens": 0
}
```

The time and identifier above are illustrative. Copy the evidence from the
actual route-verification run. A provider alias that silently changes over time
is unsuitable for a strict longitudinal claim unless the resolved identity is
recorded and the limitation is disclosed.

## PCQ eligibility

PCQ eligibility is model-specific. Leave `pcq_eligible` false until the
applicable knowledge/training boundary is documented. When it is false, the
framework excludes PCQ items for that model instead of treating unsupported
post-cutoff assumptions as data.

A PCQ-eligible route records `pcq_cutoff`, precision, definition, source URL,
source type, evidence status, and verification date. Official model cards,
provider/API documentation, system cards, or documented named-human
attestation are accepted by the release gate. Secondary estimates should be
reported as estimates and do not automatically become release-grade evidence.

## Generational pairs

Optional `pair` and `generation` (`old`/`new`) fields enable paired analyses.
Each named pair must contain exactly two routes. Models outside a pair remain
valid for cross-sectional analyses.
