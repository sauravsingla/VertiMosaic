from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import yaml

_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "pydantic",
    "pyarrow",
    "psutil",
    "ucimlrepo",
    "PyYAML",
    "typer",
)


def _git_sha() -> str | None:
    """Return a trusted CI/user-provided Git SHA when available."""
    return os.environ.get("GITHUB_SHA") or os.environ.get("VERTIMOSAIC_GIT_SHA")


def environment_snapshot(seed: int | None = None) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    memory = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_bytes": int(memory.total) if memory.total else None,
        "dependency_versions": versions,
        "git_sha": _git_sha(),
        "seed": seed,
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


@dataclass
class RunArtifacts:
    """Writer for the reproducibility bundle required by each experiment run."""

    root: Path
    run_id: str

    @property
    def directory(self) -> Path:
        return self.root / self.run_id

    @classmethod
    def create(cls, root: Path = Path("runs"), run_id: str | None = None) -> RunArtifacts:
        if run_id is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"run-{timestamp}-{os.getpid()}"
        obj = cls(root=root, run_id=run_id)
        obj.directory.mkdir(parents=True, exist_ok=False)
        return obj

    def write_config(self, config: dict[str, Any]) -> None:
        (self.directory / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )

    def write_json(self, name: str, payload: Any) -> None:
        (self.directory / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def write_frame(self, name: str, frame: pd.DataFrame) -> None:
        if name.endswith(".parquet"):
            frame.to_parquet(self.directory / name, index=False)
        else:
            frame.to_csv(self.directory / name, index=False)

    def finalize(
        self,
        *,
        config: dict[str, Any],
        seed: int,
        dataset_provenance: dict[str, Any],
        linkage_manifest: dict[str, Any],
        metrics: dict[str, Any],
        predictions: pd.DataFrame,
        training_history: pd.DataFrame,
        communication: pd.DataFrame,
        feature_provenance: pd.DataFrame,
    ) -> Path:
        environment = environment_snapshot(seed)
        self.write_config(config)
        self.write_json("dataset_provenance.json", dataset_provenance)
        self.write_json("linkage_manifest.json", linkage_manifest)
        self.write_json("environment.json", environment)
        self.write_json("metrics.json", metrics)
        self.write_frame("predictions.parquet", predictions)
        self.write_frame("training_history.csv", training_history)
        self.write_frame("communication.csv", communication)
        self.write_frame("feature_provenance.csv", feature_provenance)
        config_bytes = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        artifact_hashes = {
            item.name: file_sha256(item)
            for item in sorted(self.directory.iterdir())
            if item.is_file() and item.name != "run_manifest.json"
        }
        dataset_hashes = dataset_provenance.get("dataset_hashes", {})
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "git_sha": _git_sha(),
            "configuration_hash": sha256(config_bytes).hexdigest(),
            "dataset_provenance_hash": _json_hash(dataset_provenance),
            "dataset_hashes": dataset_hashes if isinstance(dataset_hashes, dict) else {},
            "dependency_versions": environment["dependency_versions"],
            "artifact_sha256": artifact_hashes,
            "files": sorted(item.name for item in self.directory.iterdir()),
        }
        self.write_json("run_manifest.json", manifest)
        return self.directory
