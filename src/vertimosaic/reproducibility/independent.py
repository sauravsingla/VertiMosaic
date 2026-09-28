# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import platform
import sys
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from vertimosaic.experiments.formal_benchmark import run_formal_benchmark
from vertimosaic.reproducibility.run import environment_snapshot, file_sha256


def _installed_distributions() -> list[str]:
    items: set[str] = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if isinstance(name, str) and name.strip():
            items.add(f"{name.strip()}=={version}")
    return sorted(items, key=str.casefold)


def _result_digest(records: list[dict[str, Any]]) -> str:
    stable = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(stable.encode("utf-8")).hexdigest()


def write_independent_reproduction_bundle(
    *,
    output: Path = Path("vertimosaic-reproduction-evidence"),
    rows: int = 800,
    seed: int = 42,
    expected_version: str | None = None,
    logistic_max_iter: int = 150,
    gbdt_estimators: int = 8,
) -> dict[str, Any]:
    """Write a self-contained evidence bundle suitable for an unaffiliated reproducer."""
    try:
        installed_version = metadata.version("vertimosaic")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("VertiMosaic must be installed before reproduction") from exc
    if expected_version is not None and installed_version != expected_version:
        raise RuntimeError(
            f"installed VertiMosaic version {installed_version!r} does not match "
            f"expected version {expected_version!r}"
        )

    output.mkdir(parents=True, exist_ok=True)
    benchmark_path = output / "formal_comparison.csv"
    frame = run_formal_benchmark(
        rows=rows,
        seed=seed,
        include_robustness=False,
        logistic_max_iter=logistic_max_iter,
        gbdt_estimators=gbdt_estimators,
        output=benchmark_path,
    )
    records = frame.to_dict(orient="records")
    environment = environment_snapshot(seed)
    environment.update(
        {
            "vertimosaic_version": installed_version,
            "python_executable": sys.executable,
            "python_full_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "reproduction_rows": rows,
            "logistic_max_iter": logistic_max_iter,
            "gbdt_estimators": gbdt_estimators,
        }
    )
    environment_path = output / "environment.json"
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    distributions_path = output / "installed-distributions.txt"
    distributions_path.write_text(
        "\n".join(_installed_distributions()) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "completed",
        "installed_version": installed_version,
        "expected_version": expected_version,
        "rows": rows,
        "seed": seed,
        "result_digest_sha256": _result_digest(records),
        "protocols": [str(value) for value in frame["protocol"].tolist()],
        "claim_boundary": (
            "This bundle records a reproduction run. It becomes independent external evidence "
            "only when produced and published by a person or organization independent of the "
            "VertiMosaic author/maintainer."
        ),
    }
    result_path = output / "reproduction.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest_path = output / "SHA256SUMS"
    manifest_path.write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in manifest_targets),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an exact environment/results bundle for independent VertiMosaic reproduction."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("vertimosaic-reproduction-evidence"))
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--logistic-max-iter", type=int, default=150)
    parser.add_argument("--gbdt-estimators", type=int, default=8)
    args = parser.parse_args()
    result = write_independent_reproduction_bundle(
        output=args.output,
        rows=args.rows,
        seed=args.seed,
        expected_version=args.expected_version,
        logistic_max_iter=args.logistic_max_iter,
        gbdt_estimators=args.gbdt_estimators,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
