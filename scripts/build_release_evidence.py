# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release-evidence"
INCLUDE_ROOTS = (
    "results",
    "reports",
    "benchmarks",
    "paper/figures",
    "paper/tables",
    "runs",
)
CANONICAL_PATHS = (
    "benchmarks/formal_comparison.csv",
    "benchmarks/formal_comparison.json",
    "benchmarks/formal_comparison.md",
    "benchmarks/remote_transport.json",
    "reports/privacy_audit.json",
    "reports/advanced_privacy_audit.json",
    "reports/privacy_research.json",
    "reports/privacy_research_summary.csv",
    "paper/tables/main_results.csv",
    "results/partial_overlap.csv",
    "results/party_dropout.csv",
    "results/missing_party_methods.csv",
    "results/feature_drift.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_files() -> list[Path]:
    files: list[Path] = []
    for relative_root in INCLUDE_ROOTS:
        root = ROOT / relative_root
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    lock = ROOT / "reproducibility/requirements-py312.lock"
    if lock.exists():
        files.append(lock)
    return sorted(set(files))


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    git_sha = os.environ.get("GITHUB_SHA") or git_value("rev-parse", "HEAD")
    git_tag = os.environ.get("GITHUB_REF_NAME")
    manifest = {
        "schema_version": 2,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "git_sha": git_sha,
        "git_tag": git_tag,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reproduction_lock": "reproducibility/requirements-py312.lock",
        "reproduction_lock_sha256": sha256(ROOT / "reproducibility/requirements-py312.lock"),
        "artifacts": [artifact_record(path) for path in files],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    canonical = []
    missing = []
    for relative in CANONICAL_PATHS:
        path = ROOT / relative
        if path.is_file():
            canonical.append(artifact_record(path))
        else:
            missing.append(relative)
    canonical_index = {
        "schema_version": 1,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "git_sha": git_sha,
        "git_tag": git_tag,
        "status": "complete" if not missing else "partial",
        "canonical_artifacts": canonical,
        "missing_optional_or_unproduced_artifacts": missing,
        "policy": (
            "Canonical numeric claims must be sourced from measured files listed here or "
            "from an external-comparator normalized JSON record. No metric may be manually "
            "typed into a release table as a substitute for a missing artifact."
        ),
    }
    (output / "canonical-results.json").write_text(
        json.dumps(canonical_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "manifest.json")
    print(output / "canonical-results.json")


if __name__ == "__main__":
    main()
