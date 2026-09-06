# Security policy

## Supported version

Security fixes are applied to the latest released framework version. Research
artifacts remain immutable; a fix produces a new version rather than rewriting
an old release.

## Reporting a vulnerability

Do not open a public issue for a credential leak, arbitrary code execution,
path traversal, unsafe archive handling, or another vulnerability that could
put users or provider accounts at risk. Email `alisenol@tarsus.edu.tr` with:

- affected version and platform;
- minimal reproduction steps;
- expected and observed behavior;
- potential impact;
- whether public disclosure is already planned.

Do not include live API keys or private provider responses. Revoke an exposed
credential immediately through its provider; deleting it from Git history is
not sufficient.

## Scope notes

EHQ makes outbound requests only when a real-provider command is selected.
`dry-run`, dataset validation, run verification, and local analyses do not need
provider credentials. `.env` is a local convenience file and is excluded from
version control and distributions.
