# Publishing EHQ to PyPI

The distribution name is `ehq`; the import package and console command are also
`ehq`. The canonical source repository is `senolali/EHQ`.

## One-time setup

1. Create and verify a PyPI account with two-factor authentication.
2. Confirm that `https://pypi.org/project/ehq/` is still available.
3. In PyPI, add a pending Trusted Publisher with:
   - owner: `senolali`
   - repository: `EHQ`
   - workflow: `release.yml`
   - environment: `pypi`
4. In GitHub, create the `pypi` environment and require manual approval.
5. Protect the default branch and review changes to `.github/workflows/release.yml`.

Trusted Publishing uses short-lived OIDC credentials. Do not store a long-lived
PyPI token in `.env`, repository secrets, or the workflow.

## Release preparation

1. Update `CHANGELOG.md`.
2. Keep the version identical in `pyproject.toml`, `src/ehq/constants.py`, and
   `CITATION.cff`.
3. Update the release date and citation metadata.
4. Run the complete local verification described below.
5. Commit the clean source tree and create an annotated `vX.Y.Z` tag.
6. Push the tag and create a GitHub Release.

The release workflow triggers on a published GitHub Release, rebuilds from the
tag, runs tests and package checks, then publishes through the protected `pypi`
environment.

## Local package verification

```bash
python -m pip install -e ".[all,dev]"
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

Inspect archive contents, including the version-pinned EHQ-3000 resource:

```bash
python tools/check_distribution.py
```

The wheel/sdist must contain no populated `.env`, results, provider responses,
cache, checkpoint, credential, editor file, or nested build output. It must
contain the installed project template, smoke fixture, and EHQ-3000 release.

Install the wheel into a fresh environment and verify:

```bash
python -m venv wheel-test
wheel-test/Scripts/python -m pip install dist/ehq-0.4.0-py3-none-any.whl
wheel-test/Scripts/ehq --version
wheel-test/Scripts/ehq init scaffold
wheel-test/Scripts/ehq dataset-path
wheel-test/Scripts/ehq dry-run --config scaffold/config/smoke.json --model openai-example --allow-candidate --run-id wheel-smoke
```

Use `wheel-test/bin/...` on macOS/Linux.

## Post-publication checks

```bash
python -m pip install --no-cache-dir "ehq==0.4.0"
ehq --version
```

Check the PyPI project page, README rendering, license, project links,
provenance attestations, wheel/sdist hashes, and GitHub release assets. PyPI
versions are immutable; correct a bad upload with a new version rather than
reusing the same number.
