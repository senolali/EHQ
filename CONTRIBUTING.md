# Contributing to EHQ

Contributions that improve correctness, reproducibility, documentation,
provider support, or accessibility are welcome.

## Before opening a pull request

1. Open an issue for protocol, metric, dataset, or public-API changes.
2. Create a focused branch and keep unrelated formatting out of the change.
3. Install the development environment with `pip install -e ".[all,dev]"`.
4. Add deterministic tests for behavior changes.
5. Run `python -m unittest discover -s tests -v` and the offline smoke test.
6. Confirm that no `.env`, token, provider response, result, cache, checkpoint,
   or private reviewer identity is staged.

## Scientific changes

A metric or protocol change must increment the protocol version, document
compatibility, and preserve older artifacts. It must never silently reinterpret
scores produced under a different estimand.

A dataset correction must identify affected item IDs, cite evidence, increment
the dataset release, re-run strict validation, regenerate both dataset and
scientific-content hashes, and preserve the earlier release. Do not make broad
generated rewrites of the frozen dataset in a correction pull request.

A model-registry update must record the exact route, source-qualified cutoff
semantics, dated source, served-identity evidence, operational status, and
reasoning-token behavior. Registry membership alone does not grant eligibility.

## Code style

- Support Python 3.10+.
- Prefer the standard library unless a dependency materially improves the
  scientific or operational contract.
- Preserve deterministic output and stable serialization.
- Keep credentials and raw provider payload secrets out of logs and artifacts.
- Use atomic writes for scientific outputs.
- Document user-facing changes in `CHANGELOG.md`.

## Reporting dataset issues

Include the release hash, item ID, claimed problem, proposed correction, and a
stable source. Do not post personal data that is not already public or any
confidential reviewer information.

By contributing code, you agree that it may be distributed under the MIT
License. Dataset contributions are accepted under the dataset license described
in `DATA_LICENSE.md` unless explicitly agreed otherwise.
