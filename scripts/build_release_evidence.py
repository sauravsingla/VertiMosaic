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


def main() -> None:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    manifest = {
        "schema_version": 1,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "git_sha": os.environ.get("GITHUB_SHA") or git_value("rev-parse", "HEAD"),
        "git_tag": os.environ.get("GITHUB_REF_NAME"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reproduction_lock": "reproducibility/requirements-py312.lock",
        "reproduction_lock_sha256": sha256(ROOT / "reproducibility/requirements-py312.lock"),
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
