# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

REQUIRED_FILES = (
    "attestation.json",
    "environment.json",
    "installed-distributions.txt",
    "formal_comparison.csv",
    "reproduction.json",
    "SHA256SUMS",
)

REQUIRED_ATTESTATION_FIELDS = (
    "reproducer",
    "affiliation",
    "relationship_to_project",
    "independent",
    "release_version",
    "package_source",
    "unpublished_maintainer_changes_used",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256sums(path: pathlib.Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line: {raw!r}")
        digest, filename = parts
        filename = filename.lstrip("* ")
        if len(digest) != 64:
            raise ValueError(f"invalid SHA-256 digest for {filename!r}")
        entries[filename] = digest.lower()
    return entries


def validate(directory: pathlib.Path) -> dict[str, object]:
    directory = directory.resolve()
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"reproduction directory is missing files: {missing}")

    attestation = json.loads((directory / "attestation.json").read_text(encoding="utf-8"))
    absent = [key for key in REQUIRED_ATTESTATION_FIELDS if key not in attestation]
    if absent:
        raise ValueError(f"attestation is missing fields: {absent}")
    if not str(attestation["reproducer"]).strip():
        raise ValueError("reproducer must not be empty")
    if not str(attestation["release_version"]).strip():
        raise ValueError("release_version must not be empty")
    if bool(attestation["independent"]) and bool(
        attestation["unpublished_maintainer_changes_used"]
    ):
        raise ValueError(
            "a run using unpublished maintainer changes cannot be indexed as independent"
        )

    reproduction = json.loads((directory / "reproduction.json").read_text(encoding="utf-8"))
    declared_version = str(attestation["release_version"])
    observed_version = str(
        reproduction.get("vertimosaic_version")
        or reproduction.get("version")
        or reproduction.get("package_version")
        or ""
    )
    if observed_version and observed_version != declared_version:
        raise ValueError(
            f"attested release {declared_version!r} does not match reproduction version "
            f"{observed_version!r}"
        )

    sums = _parse_sha256sums(directory / "SHA256SUMS")
    verified: list[str] = []
    for name, expected in sums.items():
        candidate = directory / name
        if not candidate.is_file() or candidate.name == "SHA256SUMS":
            continue
        actual = _sha256(candidate)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
        verified.append(name)

    return {
        "directory": str(directory),
        "release_version": declared_version,
        "reproducer": attestation["reproducer"],
        "independent": bool(attestation["independent"]),
        "verified_hash_entries": sorted(verified),
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VertiMosaic reproduction record")
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
