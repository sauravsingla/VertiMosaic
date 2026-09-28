# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vertimosaic.alignment import entity_ids_for
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.pipeline import slice_parties


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a frozen VertiMosaic benchmark exchange bundle for external VFL frameworks"
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/exchange"))
    args = parser.parse_args()

    active, passive = make_vertical_synthetic(args.rows, args.seed)
    split = entity_level_split(active.labels, seed=args.seed)
    partitions = {
        "train": slice_parties(active, passive, split.train),
        "validation": slice_parties(active, passive, split.validation),
        "test": slice_parties(active, passive, split.test),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    party_names = [active.name, *[party.name for party in passive]]

    for split_name, (split_active, split_passive) in partitions.items():
        split_directory = args.output / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        split_parties = [split_active, *split_passive]
        split_record: dict[str, object] = {"rows": split_active.n_rows, "parties": {}}
        party_record = split_record["parties"]
        assert isinstance(party_record, dict)
        for party in split_parties:
            path = split_directory / f"{party.name}.npy"
            np.save(path, party._x, allow_pickle=False)
            party_record[party.name] = {
                "path": str(path),
                "features": party.n_features,
                "sha256": _sha256(path),
            }
        target_path = split_directory / "target.npy"
        np.save(target_path, split_active.labels, allow_pickle=False)
        ids_path = split_directory / "entity_ids.txt"
        ids = entity_ids_for(split_active)
        if ids is None:
            raise RuntimeError("official exchange export requires bound entity identifiers")
        ids_path.write_text("\n".join(ids.tolist()) + "\n", encoding="utf-8")
        split_record["target"] = {"path": str(target_path), "sha256": _sha256(target_path)}
        split_record["entity_ids"] = {"path": str(ids_path), "sha256": _sha256(ids_path)}
        files[split_name] = split_record

    manifest = {
        "schema_version": 1,
        "benchmark": "vertimosaic_synthetic_external_comparator",
        "rows": args.rows,
        "seed": args.seed,
        "active_party": active.name,
        "party_order": party_names,
        "target_owner": active.name,
        "split": files,
        "parties": {
            party.name: {"features": party.n_features, "role": "active" if party is active else "passive"}
            for party in [active, *passive]
        },
        "fairness_contract": (
            "External frameworks must use the supplied entity partitions and feature files. "
            "Framework-specific preprocessing must be fit on training rows only and documented."
        ),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
