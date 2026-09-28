# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_result(payload: dict[str, Any]) -> None:
    required = {"model", "roc_auc", "pr_auc", "training_seconds", "inference_seconds"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"external comparator result is missing fields: {missing}")
    for key in ("roc_auc", "pr_auc", "training_seconds", "inference_seconds"):
        value = float(payload[key])
        if not (value == value):
            raise ValueError(f"external comparator field {key!r} must be finite")
    if not 0.0 <= float(payload["roc_auc"]) <= 1.0:
        raise ValueError("roc_auc must be in [0, 1]")
    if not 0.0 <= float(payload["pr_auc"]) <= 1.0:
        raise ValueError("pr_auc must be in [0, 1]")
    if float(payload["training_seconds"]) < 0 or float(payload["inference_seconds"]) < 0:
        raise ValueError("runtime fields must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an external VFL comparator through a stable JSON contract. "
            "The command is expected to accept --manifest and --output arguments."
        )
    )
    parser.add_argument("--backend", required=True, choices=["fate", "secretflow", "other"])
    parser.add_argument("--command", required=True, help="External runner command")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if "split" not in manifest_payload or "parties" not in manifest_payload:
        raise ValueError("benchmark manifest must contain 'split' and 'parties' sections")

    raw_output = args.output.with_suffix(args.output.suffix + ".raw.json")
    command = [*shlex.split(args.command), "--manifest", str(manifest), "--output", str(raw_output)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        timeout=args.timeout_seconds,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"external comparator exited with status {completed.returncode}: {command!r}"
        )
    if not raw_output.is_file():
        raise RuntimeError("external comparator did not create its declared JSON output")

    result = json.loads(raw_output.read_text(encoding="utf-8"))
    _required_result(result)
    normalized = {
        "schema_version": 1,
        "backend": args.backend,
        "runner_command": command,
        "manifest_sha256": _sha256(manifest),
        "raw_result_sha256": _sha256(raw_output),
        "wrapper_elapsed_seconds": elapsed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "result": result,
        "interpretation_boundary": (
            "This record reports measurements emitted by an external framework runner. "
            "It is not a claim that VertiMosaic and the comparator have identical privacy, "
            "cryptographic, networking, or optimization semantics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
