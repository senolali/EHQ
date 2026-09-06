"""Fail if built EHQ distributions contain private or repository-only files."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

FORBIDDEN_PARTS = {
    "outputs",
    "results",
    "cache",
    "checkpoints",
    "pilot_outputs",
    "pilot_cache",
    "pilot_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".git",
}

REQUIRED_WHEEL_SUFFIXES = {
    "ehq/cli.py",
    "ehq/constants.py",
    "ehq/py.typed",
    "ehq/resources/project/.env.example",
    "ehq/resources/project/config/experiment.json",
    "ehq/resources/project/config/models.json",
    "ehq/resources/project/config/smoke.json",
    "ehq/resources/project/data/examples/EHQ-20-smoke.json",
    "ehq/resources/project/data/releases/EHQ-3000.json",
}


def _archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution: {path}")


def _issues(path: Path, names: list[str]) -> list[str]:
    issues: list[str] = []
    normalized = [name.replace("\\", "/") for name in names]
    for name in normalized:
        parts = PurePosixPath(name).parts
        lowered = {part.lower() for part in parts}
        if lowered & FORBIDDEN_PARTS:
            issues.append(f"forbidden path: {name}")
        if parts and parts[-1] == ".env":
            issues.append(f"populated environment filename: {name}")
        if name.endswith((".pyc", ".pyo", "desktop.ini", "Thumbs.db")):
            issues.append(f"generated/editor file: {name}")
    if path.suffix == ".whl":
        for suffix in sorted(REQUIRED_WHEEL_SUFFIXES):
            if not any(name.endswith(suffix) for name in normalized):
                issues.append(f"missing wheel resource: {suffix}")
    return sorted(set(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        type=Path,
        default=DIST,
        help="Distribution directory (default: repository dist/)",
    )
    args = parser.parse_args()
    distribution_dir = args.dist.resolve()
    archives = sorted(distribution_dir.glob("*.whl")) + sorted(
        distribution_dir.glob("*.tar.gz")
    )
    if not archives:
        raise SystemExit(f"No distributions found in {distribution_dir}")
    report = {}
    failed = False
    for path in archives:
        names = _archive_names(path)
        issues = _issues(path, names)
        report[path.name] = {"entries": len(names), "issues": issues}
        failed = failed or bool(issues)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
